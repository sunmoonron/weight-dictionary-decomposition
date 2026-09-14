"""Experiment F: write recovery WITHOUT selection bias, and confound controls.

Exp C's r=0.95 was computed over OMP-selected pairs -- conditioned on OMP's own
choices. The selection-bias-free question runs the other way: define the sample
by GROUND TRUTH (each token's largest actual MLP writes, |a_j|*|w_j| over all
21,504 neurons in blocks 0..6) and ask what WDD says about THOSE, scoring an
unselected atom's coefficient as 0.

Reported:
  identification  recall of the true top-m writes in OMP's support (k=32/64),
                  vs a one-shot correlation-ranking baseline (same support size)
  estimation      Pearson r over (token x true-top-3) pairs, unconditional
                  (misses count as 0), plus r restricted to identified pairs
  energy          fraction of total true-write energy carried by OMP's support
  confounds       Spearman rank-r on selected pairs; Pearson per |true|-decile;
                  Pearson with block 2 (the super-activation) excluded;
                  dictionary coherence (near-duplicate atom stats)
"""

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
OUT = RESULTS_DIR

CTX, N_EVAL, L, K = 512, 64, 6, 64
N_MLP = 3072

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
D = model.config.n_embd

text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
eval_ids = tok(text, return_tensors="pt").input_ids[0][:N_EVAL * CTX].view(N_EVAL, CTX)

acts = {b: [] for b in range(L + 1)}
hooks = [model.transformer.h[b].mlp.c_proj.register_forward_pre_hook(
    (lambda b_: lambda m, inp: acts[b_].append(inp[0].half().cpu()))(b))
    for b in range(L + 1)]
resid = []
for i in range(0, N_EVAL, 4):
    out = model(eval_ids[i:i + 4].to(DEV), output_hidden_states=True)
    resid.append(out.hidden_states[L + 1].float().cpu())
    del out
for h in hooks:
    h.remove()
acts = {b: torch.cat(a).view(-1, N_MLP) for b, a in acts.items()}
X = torch.cat(resid).view(-1, D)
X = X - X.mean(0)
NT = X.shape[0]
print(f"{NT} states captured", flush=True)

# dictionary, same construction/order as exp A/C
atoms, meta = [], []
def add(t, m):
    atoms.append(t.detach().float().cpu())
    meta.extend(m)
add(model.transformer.wte.weight, [None] * 50257)
add(model.transformer.wpe.weight, [None] * 1024)
for b in range(L + 1):
    blk = model.transformer.h[b]
    add(blk.mlp.c_proj.weight, [(b, j) for j in range(N_MLP)])
    w_o = blk.attn.c_proj.weight
    for hd in range(12):
        add(torch.linalg.svd(w_o[hd * 64:(hd + 1) * 64], full_matrices=False).Vh,
            [None] * 64)
    add(torch.stack([blk.attn.c_proj.bias, blk.mlp.c_proj.bias]), [None, None])
A = torch.cat(atoms)
NORMS = A.norm(dim=-1)
A = A / NORMS[:, None].clamp_min(1e-8)
NA = A.shape[0]

# global index of each (block, neuron) mlp atom, and per-neuron write norms
mlp_start = {b: 51281 + b * (N_MLP + 768 + 2) for b in range(L + 1)}
wnorm = {b: NORMS[mlp_start[b]:mlp_start[b] + N_MLP] for b in range(L + 1)}

# true signed write coefficients [NT, 7*3072] fp16 (a_j * |w_j|)
TRUE = torch.cat([acts[b].float() * wnorm[b] for b in range(L + 1)], dim=1).half()
del acts
glob_of_col = torch.cat([torch.arange(mlp_start[b], mlp_start[b] + N_MLP)
                         for b in range(L + 1)])
print("true write matrix built", flush=True)

# dictionary coherence: near-duplicate stats over sampled atoms
g = torch.Generator().manual_seed(3)
samp_idx = torch.randperm(NA, generator=g)[:2000]
samp = A[samp_idx].to(DEV)
sims = samp @ A.to(DEV).T
sims.scatter_(1, samp_idx.to(DEV)[:, None], -1.0)   # mask each atom's own column
coh = sims.max(-1).values.cpu()
coherence = {"median_max_cos": coh.median().item(),
             "p99_max_cos": coh.quantile(0.99).item()}
del samp, sims
torch.cuda.empty_cache()
print(f"coherence: {coherence}", flush=True)

# OMP k=64 with support + final coefficients; also one-shot top-64 support
A_dev = A.to(DEV)
sels, cofs, oneshot = [], [], []
for s in range(0, NT, 768):
    x = X[s:s + 768].to(DEV)
    B = len(x)
    corr0 = (x @ A_dev.T).abs_()
    oneshot.append(corr0.topk(K, -1).indices.cpu())
    del corr0
    sel = torch.zeros(B, K, dtype=torch.long, device=DEV)
    taken = torch.zeros(B, NA, dtype=torch.bool, device=DEV)
    r = x.clone()
    for k in range(K):
        pick = (r @ A_dev.T).abs_().masked_fill_(taken, -1.0).argmax(-1)
        sel[:, k] = pick
        taken.scatter_(1, pick[:, None], True)
        A_S = A_dev[sel[:, :k + 1]]
        G = A_S @ A_S.transpose(1, 2) + 1e-5 * torch.eye(k + 1, device=DEV)
        c = torch.cholesky_solve(A_S @ x[:, :, None], torch.linalg.cholesky(G))
        r = x - (c.transpose(1, 2) @ A_S)[:, 0]
    sels.append(sel.cpu())
    cofs.append(c[:, :, 0].cpu())
sel = torch.cat(sels)
cof = torch.cat(cofs)
oneshot = torch.cat(oneshot)
print("OMP done", flush=True)


def pearson(a, b):
    a, b = a.float(), b.float()
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-9)).item()


res = {"coherence": coherence}
glob2col = torch.full((NA,), -1, dtype=torch.long)
glob2col[glob_of_col] = torch.arange(len(glob_of_col))

# ---- identification: recall of true top-m in supports ----
tops = []
for s0 in range(0, NT, 4096):
    tops.append(TRUE[s0:s0 + 4096].float().to(DEV).abs().topk(8, dim=1).indices.cpu())
topm_idx = torch.cat(tops)                                    # [NT, 8] col ids
topm_glob = glob_of_col[topm_idx]                             # global atom ids
for m in [1, 3, 8]:
    tg = topm_glob[:, :m]
    rec = {"omp64": (tg[:, :, None] == sel[:, None, :]).any(-1).float().mean().item(),
           "omp32": (tg[:, :, None] == sel[:, None, :32]).any(-1).float().mean().item(),
           "oneshot64": (tg[:, :, None] == oneshot[:, None, :]).any(-1).float().mean().item()}
    res[f"recall_top{m}"] = rec
    print(f"recall top-{m}: omp64 {rec['omp64']:.3f}  omp32 {rec['omp32']:.3f}  "
          f"oneshot64 {rec['oneshot64']:.3f}", flush=True)

# ---- estimation, unconditional over (token x true-top-3) ----
tg3 = topm_glob[:, :3]                                        # [NT, 3]
true3 = TRUE.float().gather(1, topm_idx[:, :3])               # signed truth
match = tg3[:, :, None] == sel[:, None, :]                    # [NT, 3, K]
pred3 = torch.where(match.any(-1),
                    (match.float() * cof[:, None, :]).sum(-1),
                    torch.zeros_like(true3))
res["uncond_top3"] = {
    "r": pearson(pred3.flatten(), true3.flatten()),
    "identified_frac": match.any(-1).float().mean().item(),
    "r_identified_only": pearson(pred3[match.any(-1)], true3[match.any(-1)])}
print(f"unconditional top-3: r {res['uncond_top3']['r']:.3f}  "
      f"identified {res['uncond_top3']['identified_frac']:.3f}  "
      f"r|identified {res['uncond_top3']['r_identified_only']:.3f}", flush=True)

# ---- energy recall: true-write energy carried by OMP-selected atoms ----
sel_cols = glob2col[sel]                                      # [NT, K], -1 if not mlp
sel_mlp = sel_cols >= 0
tot_e = sel_e = 0.0
for s0 in range(0, NT, 4096):
    t = TRUE[s0:s0 + 4096].float()
    tot_e += (t ** 2).sum().item()
    gathered = t.gather(1, sel_cols[s0:s0 + 4096].clamp(min=0))
    sel_e += ((gathered * sel_mlp[s0:s0 + 4096]) ** 2).sum().item()
res["energy_recall_k64"] = sel_e / tot_e
print(f"energy recall @64: {res['energy_recall_k64']:.3f}", flush=True)

# ---- confound controls on the selected-pair sample (exp C's sample) ----
pt_all = torch.cat([TRUE[s0:s0 + 4096].float().gather(
    1, sel_cols[s0:s0 + 4096].clamp(min=0)) for s0 in range(0, NT, 4096)])
pp = cof[sel_mlp]
pt = pt_all[sel_mlp]
pb = sel_cols[sel_mlp] // N_MLP
res["selected_pairs"] = {"n": len(pp), "pearson": pearson(pp, pt)}
rank = lambda v: v.argsort().argsort().float()
res["selected_pairs"]["spearman"] = pearson(rank(pp), rank(pt))
res["selected_pairs"]["pearson_wo_block2"] = pearson(pp[pb != 2], pt[pb != 2])
dec = pt.abs().quantile(torch.linspace(0, 1, 11))
per_dec = []
for i in range(10):
    m = (pt.abs() >= dec[i]) & (pt.abs() < dec[i + 1] + (1e9 if i == 9 else 0))
    if m.sum() > 100:
        per_dec.append(round(pearson(pp[m], pt[m]), 3))
res["selected_pairs"]["pearson_per_decile"] = per_dec
print(f"selected pairs: pearson {res['selected_pairs']['pearson']:.3f}  "
      f"spearman {res['selected_pairs']['spearman']:.3f}  "
      f"wo-block2 {res['selected_pairs']['pearson_wo_block2']:.3f}", flush=True)
print(f"per-decile pearson: {per_dec}", flush=True)

with open(f"{OUT}/exp_f_recovery.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE_F", flush=True)
