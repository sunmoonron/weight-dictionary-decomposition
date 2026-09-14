"""X7: oracle baselines separating state-induced from estimator-induced failure.
A: decompose SYNTHETIC states (exact sum of the token's MLP writes, blocks 0-6,
   no attention/embedding/LN interference) with the same OMP k=64 and one-shot
   identification. High recall here = the estimator is fine; the gap to
   real-state recall = loss caused by the state itself.
B: ORACLE-SUPPORT least squares on real centered states: hand the solver the
   true top-3 atoms, refit coefficients, compare to true; split by survival.
C: dictionary coherence diagnostics (MLP atoms exact max; full dict sampled).
Writes results/exp_x7_oracle.json."""

import json
import os

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
OUT = os.path.join(RESULTS_DIR, "exp_x7_oracle.json")
CTX, N_EVAL, L, N_MLP, D, NH = 512, 32, 6, 3072, 768, 12
HD = D // NH

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0][:N_EVAL * CTX].view(
    N_EVAL, CTX)


def unit(a):
    return a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)


atoms = [model.transformer.wte.weight, model.transformer.wpe.weight]
for b in range(L + 1):
    blk = model.transformer.h[b]
    atoms.append(blk.mlp.c_proj.weight)
    w_o = blk.attn.c_proj.weight
    for h in range(NH):
        atoms.append(torch.linalg.svd(
            w_o[h * HD:(h + 1) * HD, :], full_matrices=False).Vh)
    atoms.append(torch.stack([blk.attn.c_proj.bias, blk.mlp.c_proj.bias]))
A = unit(torch.cat([a.detach().float().cpu() for a in atoms]))
W = {b: model.transformer.h[b].mlp.c_proj.weight.detach().float().cpu()
     for b in range(L + 1)}
WN = {b: W[b].norm(dim=-1) for b in W}

acts = {b: [] for b in range(L + 1)}
hooks = [model.transformer.h[b].mlp.c_proj.register_forward_pre_hook(
    (lambda b_: lambda m, inp: acts[b_].append(inp[0].half().cpu()))(b))
    for b in range(L + 1)]
X_ev = []
for i in range(0, N_EVAL, 8):
    out = model(ids[i:i + 8].to(DEV), output_hidden_states=True)
    X_ev.append(out.hidden_states[L + 1].float().cpu())
    del out
for h in hooks:
    h.remove()
X_ev = torch.cat(X_ev).view(-1, D)
acts = {b: torch.cat(a).view(-1, N_MLP) for b, a in acts.items()}
NT = X_ev.shape[0]
TRUE = torch.cat([acts[b].float() * WN[b] for b in range(L + 1)], dim=1)
top3 = TRUE.abs().topk(3, dim=1).indices
OFF, STRIDE = 50257 + 1024, N_MLP + NH * HD + 2
b3, j3 = top3 // N_MLP, top3 % N_MLP
atom_idx = OFF + b3 * STRIDE + j3
true_c = TRUE.gather(1, top3)
print(f"{NT} states, dictionary {A.shape[0]}", flush=True)


def pearson(a, b):
    a, b = a.float(), b.float()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-12)).item()


def omp(X, k, batch=256):
    Ad = A.to(DEV)
    sel = torch.zeros(X.shape[0], k, dtype=torch.long)
    coef = torch.zeros(X.shape[0], k)
    for s in range(0, X.shape[0], batch):
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
            G = As @ As.transpose(1, 2) + 1e-5 * torch.eye(step + 1,
                                                           device=DEV)
            c = torch.linalg.solve(G, As @ x[:, :, None])
            r = x - (c.transpose(1, 2) @ As)[:, 0]
        sel[s:s + batch] = S.cpu()
        coef[s:s + batch] = c[:, :, 0].cpu()
    return sel, coef


RES = {"n_tokens": NT}

# ---- A: synthetic states (pure MLP write sums) -----------------------------
X_syn = sum(acts[b].float() @ W[b] for b in range(L + 1))
sel, coef = omp(X_syn, 64)
match = sel[:, None, :] == atom_idx[:, :, None]
hit = match.any(-1)
pos = match.float().argmax(-1)
pred = coef.gather(1, pos) * hit
one_shot_hit = []
Ad = A.to(DEV)
for s in range(0, NT, 1024):
    corr = (X_syn[s:s + 1024].to(DEV) @ Ad.T).abs()
    top64 = corr.topk(64, dim=1).indices.cpu()
    one_shot_hit.append((top64[:, None, :] ==
                         atom_idx[s:s + 1024, :, None]).any(-1))
os_hit = torch.cat(one_shot_hit)
RES["A_synthetic"] = {
    "recall_top1_omp": round(hit[:, 0].float().mean().item(), 4),
    "recall_top1_oneshot": round(os_hit[:, 0].float().mean().item(), 4),
    "recall_top3_omp": round(hit.float().mean().item(), 4),
    "r_identified": round(pearson(pred[hit], true_c[hit]), 4),
    "real_state_ref": {"recall_top1_omp": 0.647, "oneshot": 0.766}}
print(f"A: synth recall {RES['A_synthetic']}", flush=True)

# ---- B: oracle-support least squares on real states ------------------------
Xc = X_ev - X_ev.mean(0)
dirs = torch.cat([W[b] / WN[b][:, None].clamp_min(1e-8)
                  for b in range(L + 1)])          # [7*3072, 768] mlp dirs
d_sel = dirs[top3.flatten()].view(NT, 3, D)
G = d_sel @ d_sel.transpose(1, 2) + 1e-5 * torch.eye(3)
rhs = d_sel @ Xc[:, :, None]
c_or = torch.linalg.solve(G, rhs)[:, :, 0]         # oracle-support coeffs
surv = (d_sel @ X_ev[:, :, None])[:, :, 0] / true_c.where(
    true_c.abs() > 1e-6, torch.ones_like(true_c))
flat_p, flat_t, flat_s = c_or.flatten(), true_c.flatten(), surv.flatten()
mask_big = flat_t.abs() > 0.5
erased = mask_big & (flat_s < 0.25)
intact = mask_big & (flat_s > 0.75)
RES["B_oracle_support"] = {
    "r_all_bigwrites": round(pearson(flat_p[mask_big], flat_t[mask_big]), 4),
    "r_erased(surv<0.25)": round(pearson(flat_p[erased], flat_t[erased]), 4),
    "r_intact(surv>0.75)": round(pearson(flat_p[intact], flat_t[intact]), 4),
    "n_erased": int(erased.sum()), "n_intact": int(intact.sum())}
print(f"B: oracle-support {RES['B_oracle_support']}", flush=True)

# ---- C: coherence ----------------------------------------------------------
mlp_atoms = dirs.to(DEV)
mx = torch.zeros(mlp_atoms.shape[0])
for s in range(0, mlp_atoms.shape[0], 1024):
    cc = (mlp_atoms[s:s + 1024] @ mlp_atoms.T).abs()
    ar = torch.arange(s, min(s + 1024, mlp_atoms.shape[0]))
    cc[torch.arange(len(ar)), ar] = 0.0
    mx[s:s + 1024] = cc.max(1).values.cpu()
RES["C_coherence_mlp"] = {
    "median_max_abs_cos": round(mx.median().item(), 4),
    "p99_max_abs_cos": round(mx.quantile(0.99).item(), 4),
    "max_abs_cos": round(mx.max().item(), 4)}
print(f"C: {RES['C_coherence_mlp']}", flush=True)

with open(OUT, "w") as f:
    json.dump(RES, f, indent=1)
print("DONE_X7", flush=True)
