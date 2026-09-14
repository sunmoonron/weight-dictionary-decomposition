"""Experiment J2: case-level depth tracking of the genuinely erased writes.

Exp J showed block-3 transience is per-WRITE, not per-neuron (the least
surviving frequent neurons still average surv 0.86). So select the actual
erased population -- block-3 top-3 writes with final survival < 0.25 -- and
track each write's survival at post-block 3, 4, 5, 6 to localize the erasure.
Contrast: block-3 writes with final survival >= 0.75. Plus the qualitative
peek exp J crashed on: contexts and unembeddings for the biggest erased writes.
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
B_FOCUS = 3

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
levels = {lv: [] for lv in [4, 5, 6, 7]}
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

W = {b: model.transformer.h[b].mlp.c_proj.weight.detach().float().cpu()
     for b in range(L + 1)}
WN = {b: W[b].norm(dim=-1) for b in W}
DIRS = {b: W[b] / WN[b][:, None].clamp_min(1e-8) for b in W}

TRUE = torch.cat([acts[b].float() * WN[b] for b in range(L + 1)], dim=1)
tops = []
for s0 in range(0, NT, 4096):
    tops.append(TRUE[s0:s0 + 4096].abs().to(DEV).topk(3, dim=1).indices.cpu())
top3 = torch.cat(tops)

# collect block-3 cases: (token, neuron, true coeff, survival at each level)
cases_t, cases_j, cases_c = [], [], []
for k in range(3):
    col = top3[:, k]
    m = (col // N_MLP) == B_FOCUS
    idx = m.nonzero()[:, 0]
    cases_t.append(idx)
    cases_j.append(col[idx] % N_MLP)
    cases_c.append(TRUE[idx, col[idx]])
cases_t = torch.cat(cases_t)
cases_j = torch.cat(cases_j)
cases_c = torch.cat(cases_c)
NC = len(cases_t)
d_cases = DIRS[B_FOCUS][cases_j]                        # [NC, 768]
surv = {}
for lv in [4, 5, 6, 7]:
    surv[lv] = ((levels[lv][cases_t] * d_cases).sum(-1) / cases_c).clamp(-2, 3)
print(f"{NC} block-3 top-3 cases; final-surv mean {surv[7].mean():.2f}", flush=True)

erased = surv[7] < 0.25
intact = surv[7] >= 0.75
res = {"n_cases": NC,
       "frac_erased": erased.float().mean().item(),
       "frac_intact": intact.float().mean().item()}
for name, m in [("erased_final<0.25", erased), ("intact_final>=0.75", intact)]:
    res[name] = {f"post_block_{lv-1}": round(surv[lv][m].mean().item(), 3)
                 for lv in [4, 5, 6, 7]}
    res[name]["n"] = int(m.sum())
    print(f"{name} (n={int(m.sum())}): " + "  ".join(
        f"L{lv-1}:{surv[lv][m].mean():.2f}" for lv in [4, 5, 6, 7]), flush=True)

# which single block does the biggest damage to the erased cases?
drops = {f"block_{lv}": round((surv[lv][erased] - surv[lv + 1][erased])
                              .mean().item(), 3) for lv in [4, 5, 6]}
res["mean_drop_by_eraser"] = drops
print("mean survival drop caused by:", drops, flush=True)

# qualitative: largest erased block-3 writes
W_U = model.transformer.wte.weight.detach().float().cpu()
flat_ids = eval_ids.flatten()
order = (cases_c.abs() * erased).topk(6).indices
qual = []
for i in order.tolist():
    t, j = int(cases_t[i]), int(cases_j[i])
    dvec = DIRS[B_FOCUS][j]
    top_tok = [tok.decode([x]) for x in (dvec @ W_U.T).topk(6).indices.tolist()]
    ctx = tok.decode(flat_ids[max(0, t - 14):t + 1].tolist()).replace("\n", "|")
    qual.append({"neuron": f"mlp3#{j}", "coeff": round(cases_c[i].item(), 1),
                 "surv_curve": [round(surv[lv][i].item(), 2) for lv in [4, 5, 6, 7]],
                 "unembed_top": top_tok, "context": ctx})
    print(f"mlp3#{j} c={cases_c[i]:.1f} surv {qual[-1]['surv_curve']} -> {top_tok}",
          flush=True)
    print(f"   ...{ctx!r}", flush=True)
res["qual_erased"] = qual

with open(f"{OUT}/exp_j2_depth.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE_J2", flush=True)
