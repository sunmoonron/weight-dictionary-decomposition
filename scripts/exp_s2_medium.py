"""Write Dynamics, Episode 1b: does GPT-2-medium have a counter-writer block?

Same birth-survival role map, all 24 blocks of gpt2-medium (d=1024, cached on
box), 16k wikitext tokens. If a counter-writing specialization recurs (and
where in the stack it sits), block 3's behavior is architectural/general, not
a quirk of one checkpoint. Includes the random-direction null for the most
counter-heavy block.
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
CTX, N_EVAL = 512, 32
NB, N_MLP = 24, 4096

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
levels = {lv: [] for lv in range(1, NB + 1)}
for i in range(0, N_EVAL, 2):
    out = model(ids[i:i + 2].to(DEV), output_hidden_states=True)
    for lv in levels:
        levels[lv].append(out.hidden_states[lv].float().cpu())
    del out
for h in hooks:
    h.remove()
acts = {b: torch.cat(a).view(-1, N_MLP) for b, a in acts.items()}
levels = {lv: torch.cat(v).view(-1, D) for lv, v in levels.items()}
levels = {lv: x - x.mean(0) for lv, x in levels.items()}
NT = levels[1].shape[0]
print(f"{NT} states captured", flush=True)

WN, DIRS = {}, {}
for b in range(NB):
    w = model.transformer.h[b].mlp.c_proj.weight.detach().float().cpu()
    WN[b] = w.norm(dim=-1)
    DIRS[b] = w / WN[b][:, None].clamp_min(1e-8)

TRUE = torch.cat([acts[b].float() * WN[b] for b in range(NB)], dim=1).half()
del acts
tops = []
for s0 in range(0, NT, 2048):
    tops.append(TRUE[s0:s0 + 2048].float().to(DEV).abs().topk(3, 1).indices.cpu())
top3 = torch.cat(tops)
print("top writes found", flush=True)

g = torch.Generator().manual_seed(17)
res = {}
worst = (None, -1.0)
for b in range(NB):
    ts, js, cs = [], [], []
    for k in range(3):
        col = top3[:, k]
        m = (col // N_MLP) == b
        idx = m.nonzero()[:, 0]
        ts.append(idx); js.append(col[idx] % N_MLP)
        cs.append(TRUE[idx.numpy(), col[idx].numpy()] if False else
                  TRUE[idx, col[idx]].float())
    ts, js, cs = torch.cat(ts), torch.cat(js), torch.cat(cs)
    if len(ts) < 100:
        res[b] = {"n": len(ts)}
        continue
    s = ((levels[b + 1][ts] * DIRS[b][js]).sum(-1) / cs).clamp(-4, 5)
    row = {"n": len(ts),
           "counter": round((s < 0).float().mean().item(), 3),
           "novel": round(((s >= 0.5) & (s < 1.5)).float().mean().item(), 3),
           "reinf": round((s >= 1.5).float().mean().item(), 3),
           "median": round(s.median().item(), 2)}
    res[b] = row
    if row["counter"] > worst[1]:
        worst = (b, row["counter"], ts, js, cs)
    print(f"b{b:>2}: n={row['n']:>6} counter={row['counter']:.3f} "
          f"novel={row['novel']:.3f} reinf={row['reinf']:.3f} "
          f"median={row['median']}", flush=True)

# random-direction null for the most counter-heavy block
b, cfrac, ts, js, cs = worst
r = torch.randn(len(ts), D, generator=g)
r = r / r.norm(dim=-1, keepdim=True)
s_null = ((levels[b + 1][ts] * r).sum(-1) / cs).clamp(-4, 5)
res["null_check"] = {"block": b, "real_counter": cfrac,
                     "null_counter": round((s_null < 0).float().mean().item(), 3)}
print(f"most counter-heavy: block {b} ({cfrac}); null counter "
      f"{res['null_check']['null_counter']}", flush=True)

with open(f"{OUT}/exp_s2_medium.json", "w") as f:
    json.dump({str(k): v for k, v in res.items()}, f, indent=1)
print("DONE_S2", flush=True)
