"""Write Dynamics Ep.2a: skeptical forensics on gpt2-medium block 9 (98.3%).

Distinguish (A) genuine counter-writing specialization from (B) a geometric
artifact -- e.g. one dominant stream axis around block 9 that makes every
write look counter-directional.

  concentration   how many distinct neurons produce b9's counter cases; top
                  neuron shares (one mega-channel => explanation B flavored)
  pc1 alignment   |cos| of case write-directions vs PC1 of the pre-block state,
                  per block (b9 >> others => writes fight the dominant axis)
  quintiles       b9 counter-fraction within global |c| quintiles
  distribution    survival quantiles for b9 (clean negative mode vs smear)
  pre-state       mean pre-write projection ratio (s_birth = 1 + pre/c)
  shuffle null    marginal-preserving within-block shuffle for b9
  energy weight   per-write vs c^2-weighted counter fraction, all blocks
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
CTX, N_EVAL, NB, N_MLP = 512, 32, 24, 4096
B9 = 9

model = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2-medium")
D = model.config.n_embd

text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0][:N_EVAL * CTX].view(N_EVAL, CTX)

acts = {b: [] for b in range(NB)}
hooks = [model.transformer.h[b].mlp.c_proj.register_forward_pre_hook(
    (lambda b_: lambda m, inp: acts[b_].append(inp[0].half().cpu()))(b))
    for b in range(NB)]
levels = {lv: [] for lv in range(0, NB + 1)}      # hs[lv]; lv = pre-block-lv
for i in range(0, N_EVAL, 2):
    out = model(ids[i:i + 2].to(DEV), output_hidden_states=True)
    for lv in levels:
        levels[lv].append(out.hidden_states[lv].float().cpu())
    del out
for h in hooks:
    h.remove()
acts = {b: torch.cat(a).view(-1, N_MLP) for b, a in acts.items()}
levels = {lv: torch.cat(v).view(-1, D) for lv, v in levels.items()}
MUS = {lv: x.mean(0) for lv, x in levels.items()}
levels = {lv: x - MUS[lv] for lv, x in levels.items()}
NT = levels[0].shape[0]
print(f"{NT} states", flush=True)

WN, DIRS = {}, {}
for b in range(NB):
    w = model.transformer.h[b].mlp.c_proj.weight.detach().float().cpu()
    WN[b] = w.norm(dim=-1)
    DIRS[b] = w / WN[b][:, None].clamp_min(1e-8)

TRUE = torch.cat([acts[b].float() * WN[b] for b in range(NB)], dim=1).half()
tops = []
for s0 in range(0, NT, 2048):
    tops.append(TRUE[s0:s0 + 2048].float().to(DEV).abs().topk(3, 1).indices.cpu())
top3 = torch.cat(tops)

g = torch.Generator().manual_seed(29)
res = {}
all_cases = {}
all_c = []
for b in range(NB):
    ts, js, cs = [], [], []
    for k in range(3):
        col = top3[:, k]
        m = (col // N_MLP) == b
        idx = m.nonzero()[:, 0]
        ts.append(idx); js.append(col[idx] % N_MLP)
        cs.append(TRUE[idx, col[idx]].float())
    ts, js, cs = torch.cat(ts), torch.cat(js), torch.cat(cs)
    if len(ts) < 100:
        continue
    s = ((levels[b + 1][ts] * DIRS[b][js]).sum(-1) / cs).clamp(-4, 5)
    all_cases[b] = (ts, js, cs, s)
    all_c.append(cs.abs())

# ---- energy-weighted vs per-write counter fraction, all blocks ----
ew = {}
for b, (ts, js, cs, s) in all_cases.items():
    w2 = cs.float() ** 2
    ew[b] = {"n": len(ts),
             "counter_perwrite": round((s < 0).float().mean().item(), 3),
             "counter_energywt": round(((s < 0).float() * w2).sum().item()
                                       / w2.sum().item(), 3),
             "median_absc": round(cs.abs().median().item(), 1)}
res["counter_by_weighting"] = ew
print("counter per-write vs energy-weighted:",
      {b: (v["counter_perwrite"], v["counter_energywt"]) for b, v in ew.items()},
      flush=True)

# ---- block 9 forensics ----
ts9, js9, cs9, s9 = all_cases[B9]
uniq, cnts = js9.unique(return_counts=True)
shares = cnts.sort(descending=True).values.float() / len(js9)
res["b9_concentration"] = {
    "n_cases": len(ts9), "distinct_neurons": len(uniq),
    "top1_share": round(shares[0].item(), 3),
    "top5_share": round(shares[:5].sum().item(), 3)}
print("b9 concentration:", res["b9_concentration"], flush=True)

qs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
res["b9_survival_quantiles"] = {f"q{int(q*100)}": round(v, 2)
                                for q, v in zip(qs, s9.quantile(qs).tolist())}
pre9 = ((levels[B9][ts9] * DIRS[B9][js9]).sum(-1) / cs9).clamp(-6, 6)
res["b9_pre_projection"] = {"mean": round(pre9.mean().item(), 2),
                            "median": round(pre9.median().item(), 2)}
print("b9 surv quantiles:", res["b9_survival_quantiles"],
      "| pre-proj:", res["b9_pre_projection"], flush=True)

edges = torch.cat(all_c).quantile(torch.tensor([0.2, 0.4, 0.6, 0.8]))
q = torch.bucketize(cs9.abs(), edges)
res["b9_counter_by_quintile"] = {
    f"q{qi}": round((s9[q == qi] < 0).float().mean().item(), 3)
    for qi in range(5) if (q == qi).sum() >= 50}
print("b9 counter by |c| quintile:", res["b9_counter_by_quintile"], flush=True)

perm = torch.randperm(len(ts9), generator=g)
s_shuf = ((levels[B9 + 1][ts9] * DIRS[B9][js9[perm]]).sum(-1)
          / cs9[perm]).clamp(-4, 5)
res["b9_shuffle_null_counter"] = round((s_shuf < 0).float().mean().item(), 3)

# ---- PC1 alignment: do b9 directions fight the dominant stream axis? ----
pc = {}
for b in list(all_cases.keys()):
    x = levels[b][torch.randperm(NT, generator=g)[:8192]].to(DEV)
    v1 = torch.pca_lowrank(x, q=2)[2][:, 0].cpu()
    ts_b, js_b = all_cases[b][0], all_cases[b][1]
    cosv = (DIRS[b][js_b] @ v1).abs()
    pc[b] = round(cosv.mean().item(), 3)
res["mean_abs_cos_with_pc1"] = pc
print("mean |cos(write dir, PC1 of pre-state)| per block:", pc, flush=True)

# what fraction of b9 counter-ness survives after removing PC1 from both?
v1_9 = torch.pca_lowrank(levels[B9 + 1][torch.randperm(NT, generator=g)[:8192]]
                         .to(DEV), q=2)[2][:, 0].cpu()
d9 = DIRS[B9][js9]
d9_perp = d9 - (d9 @ v1_9)[:, None] * v1_9
d9_perp = d9_perp / d9_perp.norm(dim=-1, keepdim=True).clamp_min(1e-8)
x9 = levels[B9 + 1][ts9]
x9_perp = x9 - (x9 @ v1_9)[:, None] * v1_9
c_perp = cs9 * (d9 * d9_perp).sum(-1)          # signed write component in perp
m = c_perp.abs() > 0.5
s_perp = ((x9_perp[m] * d9_perp[m]).sum(-1) / c_perp[m]).clamp(-4, 5)
res["b9_counter_after_removing_pc1"] = {
    "n_kept": int(m.sum()),
    "counter": round((s_perp < 0).float().mean().item(), 3)}
print("b9 counter-fraction after projecting out PC1:",
      res["b9_counter_after_removing_pc1"], flush=True)

with open(f"{OUT}/exp_s3_skeptic.json", "w") as f:
    json.dump({str(k): v for k, v in res.items()}, f, indent=1)
print("DONE_S3", flush=True)
