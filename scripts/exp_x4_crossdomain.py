"""X4: cross-domain check of the core GPT-2 numbers on a non-Wikipedia corpus.
Corpus fallback chain: NeelNanda/pile-10k -> HuggingFaceFW/fineweb-edu
(streaming). Computes, on the new corpus: FVU@32 (weight vs rotated dict),
top-1 identification recall (OMP and one-shot), r on identified writes,
and block-3 role stats (counter fraction, median birth survival, novel mass)
with the within-block shuffle null. Writes results/exp_x4_crossdomain.json."""

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

torch.manual_seed(0)
torch.set_grad_enabled(False)
DEV = "cuda:0"
OUT = os.path.join(RESULTS_DIR, "exp_x4_crossdomain.json")
CTX, N_TRAIN, N_EVAL, L, N_MLP, D, NH = 512, 100, 32, 6, 3072, 768, 12
HD = D // NH

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")


def get_corpus_tokens(n_tokens):
    try:
        ds = load_dataset("NeelNanda/pile-10k", split="train")
        text = "\n\n".join(t for t in ds["text"] if t.strip())
        ids = tok(text, return_tensors="pt").input_ids[0]
        assert len(ids) >= n_tokens
        return ids[:n_tokens], "NeelNanda/pile-10k"
    except Exception:
        print(traceback.format_exc(), flush=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    chunks, total = [], 0
    for row in ds:
        t = row["text"]
        if not t.strip():
            continue
        e = tok(t, return_tensors="pt").input_ids[0]
        chunks.append(e)
        total += len(e)
        if total >= n_tokens:
            break
    return torch.cat(chunks)[:n_tokens], "fineweb-edu (streaming)"


need = (N_TRAIN + N_EVAL) * CTX
all_ids, corpus = get_corpus_tokens(need)
print(f"corpus: {corpus}, {need} tokens", flush=True)
train_ids = all_ids[:N_TRAIN * CTX].view(N_TRAIN, CTX)
eval_ids = all_ids[N_TRAIN * CTX:].view(N_EVAL, CTX)


def unit(a):
    return a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def build_dictionary():
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
            As = Ad[S]
            G = As @ As.transpose(1, 2) + 1e-5 * torch.eye(step + 1, device=DEV)
            c = torch.linalg.solve(G, As @ x[:, :, None])
            r = x - (c.transpose(1, 2) @ As)[:, 0]
        sel[s:s + batch] = S.cpu()
        coef[s:s + batch] = c[:, :, 0].cpu()
        num += (r ** 2).sum().item()
    return sel, coef, num


def pearson(a, b):
    a, b = a.float() - a.float().mean(), b.float() - b.float().mean()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-12)).item()


# sweeps
mu_sum, n_mu = torch.zeros(D), 0
for i in range(0, N_TRAIN, 8):
    out = model(train_ids[i:i + 8].to(DEV), output_hidden_states=True)
    h = out.hidden_states[L + 1].float()
    mu_sum += h.sum((0, 1)).cpu()
    n_mu += h.shape[0] * h.shape[1]
    del out
MU = mu_sum / n_mu

acts = {b: [] for b in range(L + 1)}
hooks = [model.transformer.h[b].mlp.c_proj.register_forward_pre_hook(
    (lambda b_: lambda m, inp: acts[b_].append(inp[0].half().cpu()))(b))
    for b in range(L + 1)]
X_ev, X_b3 = [], []
for i in range(0, N_EVAL, 8):
    out = model(eval_ids[i:i + 8].to(DEV), output_hidden_states=True)
    X_ev.append(out.hidden_states[L + 1].float().cpu())
    X_b3.append(out.hidden_states[4].float().cpu())     # post-block-3 (raw)
    del out
for h in hooks:
    h.remove()
X_ev = torch.cat(X_ev).view(-1, D)
X_b3 = torch.cat(X_b3).view(-1, D)
acts = {b: torch.cat(a).view(-1, N_MLP) for b, a in acts.items()}
NT = X_ev.shape[0]
print(f"{NT} eval states", flush=True)

RES = {"corpus": corpus, "n_eval_tokens": NT}
A_dict = build_dictionary()

# FVU@32, weight vs rotated
Xc = X_ev - MU
denom = (Xc ** 2).sum().item()
_, _, num_w = omp_fresh(Xc, A_dict, 32)
Q = torch.linalg.qr(torch.randn(D, D,
                                generator=torch.Generator().manual_seed(777)))[0]
_, _, num_r = omp_fresh(Xc, A_dict @ Q, 32)
RES["fvu32"] = {"weight": round(num_w / denom, 5),
                "rotated": round(num_r / denom, 5),
                "wikitext_ref": {"weight": 0.069, "rotated": 0.52}}
print(f"FVU@32 {RES['fvu32']}", flush=True)

# attribution
WN = {b: model.transformer.h[b].mlp.c_proj.weight.detach().float().cpu()
      .norm(dim=-1) for b in range(L + 1)}
TRUE = torch.cat([acts[b].float() * WN[b] for b in range(L + 1)], dim=1)
top3 = TRUE.abs().topk(3, dim=1).indices
Xf = X_ev - X_ev.mean(0)
sel64, coef64, _ = omp_fresh(Xf, A_dict, 64)
OFF, STRIDE = 50257 + 1024, N_MLP + NH * HD + 2
atom_idx = OFF + (top3 // N_MLP) * STRIDE + (top3 % N_MLP)
match = sel64[:, None, :] == atom_idx[:, :, None]
hit = match.any(-1)
pos = match.float().argmax(-1)
pred = coef64.gather(1, pos) * hit
true = TRUE.gather(1, top3)
# one-shot top-1 recall: rank atoms by |corr| with the state
Ad = A_dict.to(DEV)
os_hit = 0
for s in range(0, NT, 512):
    x = Xf[s:s + 512].to(DEV)
    cr = (x @ Ad.T).abs()
    top64 = cr.topk(64, dim=1).indices.cpu()
    os_hit += (top64 == atom_idx[s:s + 512, 0:1]).any(-1).sum().item()
RES["attrib"] = {
    "recall_top1_omp": round(hit[:, 0].float().mean().item(), 4),
    "recall_top1_oneshot": round(os_hit / NT, 4),
    "r_identified": round(pearson(pred[hit], true[hit]), 4),
    "r_unconditional": round(pearson(pred.flatten(), true.flatten()), 4),
    "wikitext_ref": {"recall_top1_omp": 0.647, "recall_top1_oneshot": 0.766,
                     "r_identified": 0.998, "r_unconditional": 0.648}}
print(f"attrib {RES['attrib']}", flush=True)

# block-3 role stats + shuffle null (birth survival on centered post-b3 state)
Xb = X_b3 - X_b3.mean(0)
C3 = acts[3].float() * WN[3]
D3 = model.transformer.h[3].mlp.c_proj.weight.detach().float().cpu()
D3 = D3 / D3.norm(dim=-1, keepdim=True).clamp_min(1e-8)
# per-token global top-3 restricted to block-3 membership (as in exp_j2/k)
b3_cols = (top3 // N_MLP) == 3
j3 = (top3 % N_MLP)


def role_stats(perm=None):
    svs = []
    for s in range(0, NT, 2048):
        t_mask = b3_cols[s:s + 2048]
        if not t_mask.any():
            continue
        rows, ks = t_mask.nonzero(as_tuple=True)
        jj = j3[s:s + 2048][rows, ks]
        cc = C3[s + rows, jj]
        if perm is not None:
            jj = jj[perm[:len(jj)] % len(jj)] if False else jj
        dd = D3[jj]
        xx = Xb[s + rows]
        sv = ((dd * xx).sum(-1) / cc).clamp(-4, 5)
        svs.append(sv)
    sv = torch.cat(svs)
    return sv


sv = role_stats()
mask = None
RES["block3"] = {
    "n_cases": int(sv.numel()),
    "counter_frac": round((sv < 0).float().mean().item(), 4),
    "median_birth_surv": round(sv.median().item(), 4),
    "novel_frac": round(((sv - 1).abs() < 0.5).float().mean().item(), 4),
    "wikitext_ref": {"counter_frac": 0.525, "median": -0.09,
                     "novel_frac": 0.244}}
# shuffle null: permute (direction, coefficient) pairs against states
g = torch.Generator().manual_seed(11)
svs_null = []
rows_all, ks_all = b3_cols.nonzero(as_tuple=True)
jj_all = j3[rows_all, ks_all]
cc_all = C3[rows_all, jj_all]
perm = torch.randperm(len(jj_all), generator=g)
dd = D3[jj_all[perm]]
cc = cc_all[perm]
xx = Xb[rows_all]
sv_null = ((dd * xx).sum(-1) / cc).clamp(-4, 5)
RES["block3"]["null_shuffle_counter_frac"] = round(
    (sv_null < 0).float().mean().item(), 4)
RES["block3"]["null_shuffle_novel_frac"] = round(
    ((sv_null - 1).abs() < 0.5).float().mean().item(), 4)
print(f"block3 {RES['block3']}", flush=True)

with open(OUT, "w") as f:
    json.dump(RES, f, indent=1)
print("DONE_X4", flush=True)
