"""Experiment J: transformation vs erasure -- and who erases block 3.

Exp I's claim "erased writes are unfindable by any state-based method" secretly
equated DIRECTIONAL survival (the write's own direction is gone from the state)
with INFORMATION survival (nothing in the state encodes the source anymore).
The reviewer's test: for block-3 transient neurons, regress the neuron's
post-GELU activation from the layer-6 state two ways --
  scalar readout   from x . d_j alone (the original write direction)
  full readout     closed-form ridge from the whole 768-dim state
If full-state R^2 stays high while the scalar R^2 collapses, the signal was
TRANSFORMED into other directions, not destroyed -- and the exp-I claim must be
weakened. If both collapse, erasure is information-theoretic and the claim
hardens. (Ridge here is analysis instrumentation, not part of WDD.)

Also: survival of each group's writes measured at every depth (post-block 3,
4, 5, 6) to name the block that does the erasing, and a qualitative peek at
what the transient block-3 neurons write.
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

CTX, N_EVAL, L, N_MLP = 512, 64, 6, 3072
N_GROUP = 32
MIN_TOP3_COUNT = 20

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
levels = {lv: [] for lv in [4, 5, 6, 7]}          # hs[lv] = post block lv-1
for i in range(0, N_EVAL, 4):
    out = model(eval_ids[i:i + 4].to(DEV), output_hidden_states=True)
    for lv in levels:
        levels[lv].append(out.hidden_states[lv].float().cpu())
    del out
for h in hooks:
    h.remove()
acts = {b: torch.cat(a).view(-1, N_MLP) for b, a in acts.items()}
levels = {lv: torch.cat(v).view(-1, D) for lv, v in levels.items()}
levels = {lv: x - x.mean(0) for lv, x in levels.items()}
NT = levels[7].shape[0]
X6 = levels[7]                                     # layer-6 readout (post block 6)
print(f"{NT} states captured", flush=True)

# unit write directions and per-neuron norms per block
DIRS, WN = {}, {}
for b in range(L + 1):
    w = model.transformer.h[b].mlp.c_proj.weight.detach().float().cpu()
    WN[b] = w.norm(dim=-1)
    DIRS[b] = w / WN[b][:, None].clamp_min(1e-8)

# per-token global top-3 writes (as in exp I) -> per-neuron stats
TRUE = torch.cat([acts[b].float() * WN[b] for b in range(L + 1)], dim=1)
tops = []
for s0 in range(0, NT, 4096):
    tops.append(TRUE[s0:s0 + 4096].abs().to(DEV).topk(3, dim=1).indices.cpu())
top3 = torch.cat(tops)                             # column ids in [0, 7*3072)
count = torch.bincount(top3.flatten(), minlength=7 * N_MLP)
surv_sum = torch.zeros(7 * N_MLP)
for s0 in range(0, NT, 2048):
    t3 = top3[s0:s0 + 2048]
    b3 = t3 // N_MLP
    j3 = t3 % N_MLP
    d = torch.stack([DIRS[int(b3[i, k])][j3[i, k]]
                     for i in range(len(t3)) for k in range(3)]).view(len(t3), 3, D)
    ct = TRUE[s0:s0 + 2048].gather(1, t3)
    sv = ((d @ X6[s0:s0 + 2048][:, :, None])[:, :, 0] / ct).clamp(-2, 3)
    surv_sum.scatter_add_(0, t3.flatten(), sv.flatten())
mean_surv = surv_sum / count.clamp_min(1)
print("per-neuron survival stats built", flush=True)


def pick_group(blocks, low_survival):
    cand = []
    for b in blocks:
        cols = torch.arange(b * N_MLP, (b + 1) * N_MLP)
        ok = cols[count[cols] >= MIN_TOP3_COUNT]
        for c in ok.tolist():
            cand.append((mean_surv[c].item(), c))
    cand.sort(reverse=not low_survival)
    return [c for _, c in cand[:N_GROUP]], [s for s, _ in cand[:N_GROUP]]


transient, surv_t = pick_group([3], low_survival=True)
control, surv_c = pick_group([4, 5, 6], low_survival=False)
print(f"transient: {len(transient)} block-3 neurons, mean surv "
      f"{torch.tensor(surv_t).mean():.2f}; control: {len(control)} neurons "
      f"(blocks 4-6), mean surv {torch.tensor(surv_c).mean():.2f}", flush=True)

# ---- transformation vs erasure: predict a_j from layer-6 state ----
split = NT // 2
Xtr, Xte = X6[:split].to(DEV), X6[split:].to(DEV)
G = Xtr.T @ Xtr + 1e-2 * Xtr.shape[0] * torch.eye(D, device=DEV)
G_chol = torch.linalg.cholesky(G)


def readouts(col):
    b, j = col // N_MLP, col % N_MLP
    y = acts[b][:, j].float()
    ytr, yte = y[:split].to(DEV), y[split:].to(DEV)
    d = DIRS[b][j].to(DEV)
    var = ((yte - yte.mean()) ** 2).sum().clamp_min(1e-9)
    # full-state ridge (closed form)
    w = torch.cholesky_solve((Xtr.T @ ytr)[:, None], G_chol)[:, 0]
    r2_full = (1 - ((yte - Xte @ w) ** 2).sum() / var).item()
    # scalar readout from the write direction alone
    str_, ste = Xtr @ d, Xte @ d
    a = (str_ @ ytr) / (str_ @ str_).clamp_min(1e-9)
    r2_dir = (1 - ((yte - a * ste) ** 2).sum() / var).item()
    return r2_full, r2_dir


res = {"groups": {}}
for name, cols in [("block3_transient", transient), ("control_intact", control)]:
    rf, rd = zip(*[readouts(c) for c in cols])
    rf, rd = torch.tensor(rf), torch.tensor(rd)
    res["groups"][name] = {
        "n_neurons": len(cols),
        "r2_full_median": rf.median().item(), "r2_dir_median": rd.median().item(),
        "r2_full_mean": rf.mean().item(), "r2_dir_mean": rd.mean().item(),
        "frac_transformed": ((rf > 0.5) & (rd < 0.25)).float().mean().item(),
        "frac_info_lost": (rf < 0.25).float().mean().item()}
    g = res["groups"][name]
    print(f"{name}: R2 full {g['r2_full_median']:.3f} (median)  "
          f"R2 dir-only {g['r2_dir_median']:.3f}  "
          f"transformed {g['frac_transformed']:.2f}  "
          f"info-lost {g['frac_info_lost']:.2f}", flush=True)

# ---- survival by depth: who erases block 3? ----
depth = {}
for name, cols in [("block3_transient", transient), ("control_intact", control)]:
    per_level = {}
    for lv in [4, 5, 6, 7]:
        Xl = levels[lv]
        vals = []
        for c in cols:
            b, j = c // N_MLP, c % N_MLP
            if b > lv - 1:                      # write not yet made at this level
                continue
            m = top3.eq(c).any(-1)
            if m.sum() < 10:
                continue
            d = DIRS[b][j]
            ct = TRUE[m, c]
            sv = ((Xl[m] @ d) / ct).clamp(-2, 3)
            vals.append(sv.mean().item())
        if vals:
            per_level[f"post_block_{lv-1}"] = round(
                torch.tensor(vals).mean().item(), 3)
    depth[name] = per_level
    print(f"{name} survival by depth: {per_level}", flush=True)
res["survival_by_depth"] = depth

# ---- qualitative: what do the transient block-3 neurons write? ----
W_U = model.transformer.wte.weight.detach().float()
flat_ids = eval_ids.flatten()
qual = []
for c in transient[:5]:
    b, j = c // N_MLP, c % N_MLP
    d = DIRS[b][j]
    top_tok = [tok.decode([t]) for t in (d @ W_U.T).topk(8).indices.tolist()]
    best = acts[b][:, j].float().abs().topk(3).indices
    ctxs = [tok.decode(flat_ids[max(0, p - 12):p + 1].tolist()).replace("\n", "|")
            for p in best.tolist()]
    qual.append({"neuron": f"mlp3#{j}", "count_top3": int(count[c]),
                 "mean_surv": round(mean_surv[c].item(), 2),
                 "unembed_top": top_tok, "contexts": ctxs})
    print(f"mlp3#{j} (n={int(count[c])}, surv {mean_surv[c]:.2f}) -> {top_tok}",
          flush=True)
    for cx in ctxs:
        print(f"   ...{cx!r}", flush=True)
res["qual_transient"] = qual

with open(f"{OUT}/exp_j_transform.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE_J", flush=True)
