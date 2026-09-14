"""Experiment K: null controls for the write-role taxonomy.

The taxonomy (reinforcing / novel / counter, from birth survival) must vanish
under nulls or it is just high-dimensional projection numerology.

  real    s = <h_post-b(t), d_j> / c_j(t)  for each top-3 write case
  null-1  same tokens/coefficients, d replaced by a random unit direction
  null-2  within-block shuffle: state from case i, direction+coefficient from
          a different case of the same block (breaks the write-state pairing,
          preserves both marginals)

A real write contributes exactly +1 to its own birth survival; nulls have no
such anchor. If the reinforcing bump near s>=1, the self-anchored mass
|s-1|<0.5, and the per-block counter-fraction differences all collapse under
the nulls, the taxonomy is measuring computation, not chance.

Plus the block-3 uniqueness check: counter-fraction per block within global
|coefficient| quintiles -- does block 3's excess survive magnitude matching?
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
levels = {lv: [] for lv in range(1, L + 2)}        # hs[lv] = post block lv-1
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
NT = levels[1].shape[0]

W = {b: model.transformer.h[b].mlp.c_proj.weight.detach().float().cpu()
     for b in range(L + 1)}
WN = {b: W[b].norm(dim=-1) for b in W}
DIRS = {b: W[b] / WN[b][:, None].clamp_min(1e-8) for b in W}
TRUE = torch.cat([acts[b].float() * WN[b] for b in range(L + 1)], dim=1)
tops = []
for s0 in range(0, NT, 4096):
    tops.append(TRUE[s0:s0 + 4096].abs().to(DEV).topk(3, dim=1).indices.cpu())
top3 = torch.cat(tops)

g = torch.Generator().manual_seed(17)


def stats(s):
    s = s.clamp(-4, 5)
    return {"frac_counter": round((s < 0).float().mean().item(), 3),
            "frac_novel": round(((s >= 0.5) & (s < 1.5)).float().mean().item(), 3),
            "frac_reinforcing": round((s >= 1.5).float().mean().item(), 3),
            "frac_self_anchored": round(((s - 1).abs() < 0.5).float().mean().item(), 3),
            "median": round(s.median().item(), 2)}


res = {"per_block": {}, "quintiles": {}}
all_c = []
per_block_cases = {}
for b in range(L + 1):
    ts, js, cs = [], [], []
    for k in range(3):
        col = top3[:, k]
        m = (col // N_MLP) == b
        idx = m.nonzero()[:, 0]
        ts.append(idx)
        js.append(col[idx] % N_MLP)
        cs.append(TRUE[idx, col[idx]])
    ts, js, cs = torch.cat(ts), torch.cat(js), torch.cat(cs)
    per_block_cases[b] = (ts, js, cs)
    all_c.append(cs.abs())

    h_post = levels[b + 1][ts]                            # [n, 768]
    d = DIRS[b][js]
    s_real = (h_post * d).sum(-1) / cs
    r = torch.randn(len(ts), D, generator=g)
    r = r / r.norm(dim=-1, keepdim=True)
    s_null_dir = (h_post * r).sum(-1) / cs
    perm = torch.randperm(len(ts), generator=g)
    s_null_shuf = (h_post * d[perm]).sum(-1) / cs[perm]
    res["per_block"][b] = {"n": len(ts),
                           "real": stats(s_real),
                           "null_random_dir": stats(s_null_dir),
                           "null_shuffled": stats(s_null_shuf)}
    per_block_cases[b] = (ts, js, cs, s_real)
    print(f"block {b} (n={len(ts)}): real {res['per_block'][b]['real']}", flush=True)
    print(f"          null-dir {res['per_block'][b]['null_random_dir']}", flush=True)
    print(f"          null-shuf {res['per_block'][b]['null_shuffled']}", flush=True)

# block-3 uniqueness under magnitude matching: counter-fraction per block per
# global |c| quintile
edges = torch.cat(all_c).quantile(torch.tensor([0.2, 0.4, 0.6, 0.8]))
for b in range(L + 1):
    ts, js, cs, s_real = per_block_cases[b]
    q = torch.bucketize(cs.abs(), edges)
    row = {}
    for qi in range(5):
        m = q == qi
        if m.sum() >= 50:
            row[f"q{qi}"] = {"n": int(m.sum()),
                             "frac_counter": round((s_real[m] < 0).float().mean().item(), 3)}
    res["quintiles"][b] = row
    print(f"block {b} counter-frac by |c| quintile: "
          + "  ".join(f"q{qi}:{v['frac_counter']}" for qi, v in
                      ((int(k[1]), v) for k, v in row.items())), flush=True)

with open(f"{OUT}/exp_k_nulls.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE_K", flush=True)
