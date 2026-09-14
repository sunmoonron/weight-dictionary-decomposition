"""X1: population dose-response. Scale ch447 at the block-2/3 boundary and
measure the relative activation change of EVERY block-3 neuron, so #614's
response gets a percentile against the full same-block distribution instead
of n=3 random controls. Writes results/exp_x1_doseresp_all.json."""

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
OUT = os.path.join(RESULTS_DIR, "exp_x1_doseresp_all.json")
CTX, N_EVAL, N_MLP = 512, 64, 3072

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0][:N_EVAL * CTX].view(
    N_EVAL, CTX)


def mean_acts(alpha):
    sums = torch.zeros(N_MLP)
    cap = []
    h_cap = model.transformer.h[3].mlp.c_proj.register_forward_pre_hook(
        lambda m, inp: cap.append(inp[0].float().sum((0, 1)).cpu()))
    h_scale = None
    if alpha != 1.0:
        def scale_hook(m, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            hh = t.clone()
            hh[..., 447] = hh[..., 447] * alpha
            return (hh,) + out[1:] if isinstance(out, tuple) else hh
        h_scale = model.transformer.h[2].register_forward_hook(scale_hook)
    for i in range(0, N_EVAL, 8):
        model(ids[i:i + 8].to(DEV))
        sums += torch.stack(cap).sum(0)
        cap.clear()
    h_cap.remove()
    if h_scale is not None:
        h_scale.remove()
    return sums / (N_EVAL * CTX)


base = mean_acts(1.0)
lo = mean_acts(0.5)
hi = mean_acts(1.5)

denom = base.abs().clamp_min(1e-3)          # ignore never-firing neurons' blowups
rel_lo = (lo - base) / denom
rel_hi = (hi - base) / denom
active = base.abs() > 0.01                   # restrict stats to neurons that fire

def stats(rel):
    r = rel[active]
    qs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {"n_active": int(active.sum()),
            "quantiles": {f"q{int(q*100)}": round(r.quantile(q).item(), 4)
                          for q in qs},
            "frac_abs_ge_10pct": round((r.abs() >= 0.10).float().mean().item(), 4),
            "frac_abs_ge_20pct": round((r.abs() >= 0.20).float().mean().item(), 4)}

def rank_of(j, rel, sign):
    r = rel[active] * sign                   # sign=+1: rank by rise; -1: by fall
    v = rel[j] * sign
    return {"value": round(rel[j].item(), 4),
            "pct_rank": round((r < v).float().mean().item(), 4)}

RES = {"alpha_0.5": stats(rel_lo), "alpha_1.5": stats(rel_hi),
       "n614": {"alpha_0.5": rank_of(614, rel_lo, +1),
                "alpha_1.5": rank_of(614, rel_hi, -1)},
       "inverse_both": {}}
# how many active neurons show the full inverse signature (up at 0.5x, down at 1.5x)
inv = active & (rel_lo > 0.10) & (rel_hi < -0.10)
RES["inverse_both"] = {
    "n_ge_10pct_both": int(inv.sum()),
    "n614_included": bool(inv[614]),
    "top10_by_rise": [{"neuron": int(j), "rel_lo": round(rel_lo[j].item(), 3),
                       "rel_hi": round(rel_hi[j].item(), 3)}
                      for j in rel_lo.masked_fill(~inv, -9).topk(
                          min(10, int(inv.sum()))).indices.tolist()]}
with open(OUT, "w") as f:
    json.dump(RES, f, indent=1)
print(f"X1: 614 lo {rel_lo[614]:.3f} (pct {RES['n614']['alpha_0.5']['pct_rank']}) "
      f"hi {rel_hi[614]:.3f} (pct {RES['n614']['alpha_1.5']['pct_rank']}); "
      f"inverse-both n={int(inv.sum())}", flush=True)
print("DONE_X1", flush=True)
