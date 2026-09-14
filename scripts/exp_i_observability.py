"""Experiment I: observability -- predicting WHICH writes are recoverable.

Reviewer's reframe: don't optimize identification toward 90%; the state may no
longer contain the information. For each true dominant write, measure
properties of the write and test whether they predict recovery:

  surv     (x . d_j) / c_true -- fraction of the write surviving along its own
           direction in the observed state (1 = intact, 0 = cancelled,
           negative = overwritten past zero)
  prom     |x . d_j| / |x| -- absolute prominence of the direction in the state
           (NOTE: one-shot ranking selects BY prominence, so prom->oneshot is
           circular; the non-trivial predictions are for OMP and for surv)
  mag      |c_true|;  rel = mag / |x|
  dist     6 - source block (older writes had more chances to be cancelled)
  compete  max |cos| with the token's other top-8 write directions

If identification is near-deterministic in survival, the ~65% is an
observability measurement, not a method deficiency: WDD finds what survives;
what it misses was erased by subsequent computation.
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

CTX, N_EVAL, L, K, N_MLP = 512, 64, 6, 64, 3072

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
    resid.append(model(eval_ids[i:i + 4].to(DEV), output_hidden_states=True)
                 .hidden_states[L + 1].float().cpu())
for h in hooks:
    h.remove()
acts = {b: torch.cat(a).view(-1, N_MLP) for b, a in acts.items()}
X = torch.cat(resid).view(-1, D)
X = X - X.mean(0)
NT = X.shape[0]

atoms = []
def add(t):
    atoms.append(t.detach().float().cpu())
add(model.transformer.wte.weight)
add(model.transformer.wpe.weight)
for b in range(L + 1):
    blk = model.transformer.h[b]
    add(blk.mlp.c_proj.weight)
    w_o = blk.attn.c_proj.weight
    for hd in range(12):
        add(torch.linalg.svd(w_o[hd * 64:(hd + 1) * 64], full_matrices=False).Vh)
    add(torch.stack([blk.attn.c_proj.bias, blk.mlp.c_proj.bias]))
A = torch.cat(atoms)
NORMS = A.norm(dim=-1)
A = A / NORMS[:, None].clamp_min(1e-8)
NA = A.shape[0]
mlp_start = {b: 51281 + b * (N_MLP + 768 + 2) for b in range(L + 1)}
glob_of_col = torch.cat([torch.arange(mlp_start[b], mlp_start[b] + N_MLP)
                         for b in range(L + 1)])
wnorm = torch.cat([NORMS[mlp_start[b]:mlp_start[b] + N_MLP] for b in range(L + 1)])
TRUE = torch.cat([acts[b].float() for b in range(L + 1)], dim=1) * wnorm
print(f"{NT} states, {NA} atoms", flush=True)

# OMP k=64 support + one-shot support
A_dev = A.to(DEV)
sels, oneshot = [], []
for s in range(0, NT, 768):
    x = X[s:s + 768].to(DEV)
    B = len(x)
    oneshot.append((x @ A_dev.T).abs_().topk(K, -1).indices.cpu())
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
sel = torch.cat(sels)
oneshot = torch.cat(oneshot)
print("OMP done", flush=True)

# per-token true top-8 writes and case features for the top-3
tops = []
for s0 in range(0, NT, 4096):
    tops.append(TRUE[s0:s0 + 4096].abs().to(DEV).topk(8, dim=1).indices.cpu())
top8_idx = torch.cat(tops)
top8_glob = glob_of_col[top8_idx]                              # [NT, 8]
true8 = TRUE.gather(1, top8_idx)                               # signed

feat = {k: [] for k in ["mag", "rel", "surv", "prom", "dist", "compete",
                        "rank", "id_omp", "id_one"]}
for s0 in range(0, NT, 2048):
    xb = X[s0:s0 + 2048].to(DEV)
    xn = xb.norm(dim=-1)
    d8 = A_dev[top8_glob[s0:s0 + 2048]]                        # [B, 8, 768]
    dots = (d8 @ xb[:, :, None])[:, :, 0]                      # x . d_j
    cos88 = (d8 @ d8.transpose(1, 2)).abs_()
    cos88 -= 2 * torch.eye(8, device=DEV)                      # kill diagonal
    ct = true8[s0:s0 + 2048, :3].to(DEV)
    for j in range(3):
        g = top8_glob[s0:s0 + 2048, j]
        feat["mag"].append(ct[:, j].abs().cpu())
        feat["rel"].append((ct[:, j].abs() / xn).cpu())
        feat["surv"].append((dots[:, j] / ct[:, j]).cpu())
        feat["prom"].append((dots[:, j].abs() / xn).cpu())
        feat["dist"].append((6 - (g - 51281) // (N_MLP + 770)).float())
        feat["compete"].append(cos88[:, j].max(-1).values.cpu())
        feat["rank"].append(torch.full((len(g),), j, dtype=torch.float))
        feat["id_omp"].append(
            (g[:, None] == sel[s0:s0 + 2048]).any(-1).float())
        feat["id_one"].append(
            (g[:, None] == oneshot[s0:s0 + 2048]).any(-1).float())
F = {k: torch.cat(v) for k, v in feat.items()}
N = len(F["mag"])
print(f"{N} cases; id_omp rate {F['id_omp'].mean():.3f} "
      f"(top-1 only: {F['id_omp'][F['rank'] == 0].mean():.3f})", flush=True)


def auc(score, label):
    """Mann-Whitney AUC of score for predicting label==1."""
    r = score.argsort().argsort().float() + 1
    pos = label == 1
    n1, n0 = pos.sum().item(), (~pos).sum().item()
    return ((r[pos].sum().item() - n1 * (n1 + 1) / 2) / (n1 * n0))


res = {"n_cases": N,
       "id_rate_omp": F["id_omp"].mean().item(),
       "id_rate_oneshot": F["id_one"].mean().item()}

res["auc_omp"] = {k: round(auc(F[k] if k != "surv" else F[k].clamp(-2, 3),
                               F["id_omp"]), 3)
                  for k in ["surv", "prom", "mag", "rel", "dist", "compete"]}
print("AUC (predicting OMP identification):", res["auc_omp"], flush=True)

# identification rate by survival bin -- the observability law
bins = [(-1e9, 0.0), (0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.25), (1.25, 1e9)]
by_surv = []
for lo, hi in bins:
    m = (F["surv"] >= lo) & (F["surv"] < hi)
    by_surv.append({"bin": f"[{lo},{hi})", "n": int(m.sum()),
                    "frac_of_cases": round(m.float().mean().item(), 3),
                    "id_omp": round(F["id_omp"][m].mean().item(), 3),
                    "id_oneshot": round(F["id_one"][m].mean().item(), 3)})
res["by_survival"] = by_surv
for row in by_surv:
    print(f"  surv {row['bin']:>14}: {row['frac_of_cases']*100:4.1f}% of cases, "
          f"id_omp {row['id_omp']:.3f}  id_oneshot {row['id_oneshot']:.3f}", flush=True)

# punchlines
intact = F["surv"] >= 0.75
erased = F["surv"] < 0.25
res["punchline"] = {
    "id_rate_intact": F["id_omp"][intact].mean().item(),
    "id_rate_erased": F["id_omp"][erased].mean().item(),
    "frac_intact": intact.float().mean().item(),
    "frac_erased": erased.float().mean().item(),
    "frac_of_misses_erased_or_weak": F["surv"][F["id_omp"] == 0].lt(0.5)
                                     .float().mean().item()}
print(f"id rate: intact(surv>=.75) {res['punchline']['id_rate_intact']:.3f}  "
      f"erased(surv<.25) {res['punchline']['id_rate_erased']:.3f}", flush=True)
print(f"misses with surv<0.5: "
      f"{res['punchline']['frac_of_misses_erased_or_weak']:.3f}", flush=True)

# per source block: survival and identification
by_block = {}
for b in range(L + 1):
    m = F["dist"] == (6 - b)
    if m.sum() > 50:
        by_block[b] = {"n": int(m.sum()),
                       "mean_surv": round(F["surv"][m].clamp(-2, 3).mean().item(), 3),
                       "id_omp": round(F["id_omp"][m].mean().item(), 3)}
res["by_block"] = by_block
print("by block:", {b: (v["mean_surv"], v["id_omp"]) for b, v in by_block.items()},
      flush=True)

with open(f"{OUT}/exp_i_observability.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE_I", flush=True)
