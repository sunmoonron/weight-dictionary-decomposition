"""Experiment W -- independent audit re-derivations for the WDD paper.

Each headline number is recomputed with FRESH code (same algorithm where the
number depends on it, e.g. OMP selection, but independently written: different
solver calls, different batching, fresh random seeds for controls).

  A: L6 FVU@32 for the weight dictionary and its rotation control
     (fresh dictionary build, fresh OMP, fresh rotation Q).      expect 0.069 / ~0.55
  B: identified-writes coefficient r + per-decile stratification
     + unconditional r + top-1 recall, from a fresh forward pass. expect 0.998 / 0.648 / 0.647
  C: mlp3#614 dose-response under ch447 scaling at the block-2/3
     boundary, re-hooked from scratch, fresh random controls.     expect +0.275 / -0.221
  D: OLMo-1B published super weight from raw weights on CPU,
     zero forward passes.                                         expect rank 0 / share 0.979

Parts are independent; each is try/excepted and results are flushed to
results/exp_w_audit.json after every part.
"""

import json
import os
import traceback

_HF = "/data/mechinterp/hf"          # the original machine's cache; ignored elsewhere
if os.path.isdir(_HF):
    os.environ.setdefault("HF_HOME", _HF)
RESULTS_DIR = os.environ.get("WDD_RESULTS", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(123)
torch.set_grad_enabled(False)
DEV = "cuda:0"
OUT = os.path.join(RESULTS_DIR, "exp_w_audit.json")

CTX, N_TRAIN, N_EVAL, L, N_MLP, D, NH = 512, 400, 64, 6, 3072, 768, 12
HD = D // NH
RES = {}


def save():
    with open(OUT, "w") as f:
        json.dump(RES, f, indent=1)


def pearson(a, b):
    a, b = a.float(), b.float()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-12)).item()


model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")


def token_chunks(split, n):
    text = "\n\n".join(t for t in load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"]
        if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0][: n * CTX]
    assert len(ids) == n * CTX, f"only {len(ids)} tokens in {split}"
    return ids.view(n, CTX)


def unit(a):
    return a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def build_dictionary():
    """Fresh reimplementation of the L6 weight dictionary (mirrors exp_a's
    construction order: wte, wpe, then per block mlp rows / per-head OV SVD /
    two bias atoms)."""
    atoms = [model.transformer.wte.weight, model.transformer.wpe.weight]
    for b in range(L + 1):
        blk = model.transformer.h[b]
        atoms.append(blk.mlp.c_proj.weight)
        w_o = blk.attn.c_proj.weight
        for h in range(NH):
            atoms.append(torch.linalg.svd(
                w_o[h * HD:(h + 1) * HD, :], full_matrices=False).Vh)
        atoms.append(torch.stack([blk.attn.c_proj.bias, blk.mlp.c_proj.bias]))
    return unit(torch.cat([a.detach().float().cpu() for a in atoms]))


def omp_fresh(X, A, k, batch=256):
    """Fresh batched OMP: greedy |corr| pick + per-step batched Gram solve via
    torch.linalg.solve (the original used masked argmax + Cholesky).
    X [N,D] cpu (already centered), A [n_atoms,D] cpu unit rows.
    Returns sel [N,k], coef [N,k], fvu numerator sum ||x - recon||^2."""
    Ad = A.to(DEV)
    N = X.shape[0]
    sel = torch.zeros(N, k, dtype=torch.long)
    coef = torch.zeros(N, k)
    num = 0.0
    for s in range(0, N, batch):
        x = X[s:s + batch].to(DEV)
        r = x.clone()
        S = torch.zeros(len(x), 0, dtype=torch.long, device=DEV)
        c = None
        for step in range(k):
            corr = r @ Ad.T
            if step:
                corr.scatter_(1, S, torch.zeros_like(S, dtype=corr.dtype))
            pick = corr.abs().argmax(1, keepdim=True)
            S = torch.cat([S, pick], 1)
            As = Ad[S]                                        # [n, step+1, D]
            G = As @ As.transpose(1, 2) + 1e-5 * torch.eye(
                step + 1, device=DEV)
            c = torch.linalg.solve(G, As @ x[:, :, None])     # [n, step+1, 1]
            r = x - (c.transpose(1, 2) @ As)[:, 0]
        sel[s:s + batch] = S.cpu()
        coef[s:s + batch] = c[:, :, 0].cpu()
        num += (r ** 2).sum().item()
        if (s // batch) % 16 == 0:
            print(f"  omp k={k}: {s + len(x)}/{N}", flush=True)
    return sel, coef, num


# ---- shared forward sweeps -------------------------------------------------
print("== forward sweeps ==", flush=True)
train_ids = token_chunks("train", N_TRAIN)
eval_ids = token_chunks("test", N_EVAL)

mu_sum, n_mu = torch.zeros(D), 0
for i in range(0, N_TRAIN, 8):
    out = model(train_ids[i:i + 8].to(DEV), output_hidden_states=True)
    h = out.hidden_states[L + 1].float()
    mu_sum += h.sum((0, 1)).cpu()
    n_mu += h.shape[0] * h.shape[1]
    del out
MU = mu_sum / n_mu
print(f"train mean over {n_mu} tokens", flush=True)

acts = {b: [] for b in range(L + 1)}
hooks = [model.transformer.h[b].mlp.c_proj.register_forward_pre_hook(
    (lambda b_: lambda m, inp: acts[b_].append(inp[0].half().cpu()))(b))
    for b in range(L + 1)]
X_ev = []
for i in range(0, N_EVAL, 8):
    out = model(eval_ids[i:i + 8].to(DEV), output_hidden_states=True)
    X_ev.append(out.hidden_states[L + 1].float().cpu())
    del out
for h in hooks:
    h.remove()
X_ev = torch.cat(X_ev).view(-1, D)
acts = {b: torch.cat(a).view(-1, N_MLP) for b, a in acts.items()}
NT = X_ev.shape[0]
print(f"{NT} eval states captured", flush=True)

A_dict = build_dictionary()
RES["n_atoms"] = int(A_dict.shape[0])
print(f"dictionary: {A_dict.shape[0]} atoms (expect 78175)", flush=True)
save()

# ---- part A: FVU@32, weight dict vs rotation control -----------------------
try:
    print("== part A ==", flush=True)
    Xc = X_ev - MU
    denom = (Xc ** 2).sum().item()
    _, _, num_w = omp_fresh(Xc, A_dict, 32)
    fvu_w = num_w / denom
    Q = torch.linalg.qr(torch.randn(
        D, D, generator=torch.Generator().manual_seed(777)))[0]
    _, _, num_r = omp_fresh(Xc, A_dict @ Q, 32)
    fvu_r = num_r / denom
    RES["A_fvu32"] = {"weight_dict": round(fvu_w, 5),
                      "rotated": round(fvu_r, 5),
                      "expected": {"weight_dict": 0.069, "rotated": 0.549,
                                   "note": "rotated uses a FRESH Q (seed 777); "
                                           "value should be near, not equal"}}
    print(f"A: FVU@32 weight {fvu_w:.4f} (expect ~0.069)  "
          f"rotated {fvu_r:.4f} (expect ~0.55)", flush=True)
except Exception:
    RES["A_error"] = traceback.format_exc()
    print(RES["A_error"], flush=True)
save()

# ---- part B: identified-writes r, deciles, unconditional, recall -----------
try:
    print("== part B ==", flush=True)
    WN = {b: model.transformer.h[b].mlp.c_proj.weight.detach().float().cpu()
          .norm(dim=-1) for b in range(L + 1)}
    TRUE = torch.cat([acts[b].float() * WN[b] for b in range(L + 1)], dim=1)
    top3 = TRUE.abs().topk(3, dim=1).indices                  # [NT, 3]
    # eval-mean centering, mirroring exp_f
    Xf = X_ev - X_ev.mean(0)
    sel64, coef64, _ = omp_fresh(Xf, A_dict, 64)
    # mlp column -> global atom index: 51281 + b*3842 + j
    OFF, STRIDE = 50257 + 1024, N_MLP + NH * HD + 2
    b3, j3 = top3 // N_MLP, top3 % N_MLP
    atom_idx = OFF + b3 * STRIDE + j3                         # [NT, 3]
    match = sel64[:, None, :] == atom_idx[:, :, None]         # [NT, 3, 64]
    hit = match.any(-1)
    pos = match.float().argmax(-1)
    pred = coef64.gather(1, pos) * hit                        # 0 where missed
    true = TRUE.gather(1, top3)
    RES["B_attrib"] = {
        "recall_top1": round(hit[:, 0].float().mean().item(), 4),
        "r_identified": round(pearson(pred[hit], true[hit]), 4),
        "r_unconditional": round(
            pearson(pred.flatten(), true.flatten()), 4),
        "n_identified": int(hit.sum()),
        "expected": {"recall_top1": 0.647, "r_identified": 0.998,
                     "r_unconditional": 0.648}}
    # per-decile stratification of the identified sample (the missing analysis)
    pi, ti = pred[hit], true[hit]
    qs = ti.abs().quantile(torch.linspace(0, 1, 11))
    decs = []
    for d_ in range(10):
        m = (ti.abs() >= qs[d_]) & (ti.abs() <= qs[d_ + 1] + 1e-9)
        decs.append(round(pearson(pi[m], ti[m]), 4) if m.sum() > 10 else None)
    RES["B_attrib"]["r_identified_per_decile"] = decs
    print(f"B: recall {RES['B_attrib']['recall_top1']}  "
          f"r_id {RES['B_attrib']['r_identified']}  "
          f"r_uncond {RES['B_attrib']['r_unconditional']}", flush=True)
    print(f"B deciles: {decs}", flush=True)
    del TRUE, sel64, coef64, Xf
except Exception:
    RES["B_error"] = traceback.format_exc()
    print(RES["B_error"], flush=True)
save()

# ---- part C: #614 dose-response, re-hooked from scratch --------------------
try:
    print("== part C ==", flush=True)
    g = torch.Generator().manual_seed(999)
    ctrl = torch.randint(0, N_MLP, (3,), generator=g).tolist()
    watch = [614] + [c for c in ctrl if c != 614]

    def mean_acts(alpha):
        sums = torch.zeros(len(watch))
        cap = []
        h_cap = model.transformer.h[3].mlp.c_proj.register_forward_pre_hook(
            lambda m, inp: cap.append(inp[0][:, :, watch].float().cpu()))
        h_scale = None
        if alpha != 1.0:
            def scale_hook(m, inp, out):
                hh = out[0].clone()
                hh[:, :, 447] = hh[:, :, 447] * alpha
                return (hh,) + out[1:]
            h_scale = model.transformer.h[2].register_forward_hook(scale_hook)
        for i in range(0, N_EVAL, 8):
            model(eval_ids[i:i + 8].to(DEV))
            sums += torch.cat(cap).view(-1, len(watch)).sum(0)
            cap.clear()
        h_cap.remove()
        if h_scale is not None:
            h_scale.remove()
        return sums / (N_EVAL * CTX)

    base = mean_acts(1.0)
    lo = mean_acts(0.5)
    hi = mean_acts(1.5)
    rel = {"alpha_0.5": [round(((lo[i] - base[i]) / base[i].abs().clamp_min(
               1e-6)).item(), 4) for i in range(len(watch))],
           "alpha_1.5": [round(((hi[i] - base[i]) / base[i].abs().clamp_min(
               1e-6)).item(), 4) for i in range(len(watch))]}
    RES["C_dose_response"] = {
        "watched_neurons": [f"mlp3#{j}" for j in watch],
        "rel_change": rel,
        "note": "index 0 is #614; rest are fresh random b3 controls",
        "expected_614": {"alpha_0.5": 0.275, "alpha_1.5": -0.221}}
    print(f"C: #614 rel change a0.5 {rel['alpha_0.5'][0]} "
          f"(expect ~+0.275), a1.5 {rel['alpha_1.5'][0]} (expect ~-0.221); "
          f"controls {rel['alpha_0.5'][1:]} / {rel['alpha_1.5'][1:]}",
          flush=True)
except Exception:
    RES["C_error"] = traceback.format_exc()
    print(RES["C_error"], flush=True)
save()

# ---- part D: OLMo super weight from raw weights, CPU only ------------------
try:
    print("== part D ==", flush=True)
    del model
    torch.cuda.empty_cache()
    olmo = AutoModelForCausalLM.from_pretrained(
        "allenai/OLMo-1B-0724-hf", torch_dtype=torch.float16,
        low_cpu_mem_usage=True)                     # CPU on purpose: no forwards
    W = None
    for nm, p in olmo.named_parameters():
        if nm.endswith("layers.1.mlp.down_proj.weight"):
            W = p.detach().float()
            break
    assert W is not None, "down_proj not found"
    col = W[:, 1710]                                # neuron 1710's write vector
    norms = W.norm(dim=0)                           # all 8192 column norms
    RES["D_olmo_sw"] = {
        "weight_shape": list(W.shape),
        "atom_norm_rank": int((norms > norms[1710]).sum()),
        "dominant_channel": int(col.abs().argmax()),
        "channel_share": round((col[1764] ** 2 / (col ** 2).sum()).item(), 4),
        "expected": {"atom_norm_rank": 0, "dominant_channel": 1764,
                     "channel_share": 0.979}}
    print(f"D: {RES['D_olmo_sw']}", flush=True)
except Exception:
    RES["D_error"] = traceback.format_exc()
    print(RES["D_error"], flush=True)
save()
print("DONE_W_AUDIT", flush=True)
