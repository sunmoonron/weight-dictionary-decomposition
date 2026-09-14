"""X6: the LayerNorm-frozen dose-response control.
Scale ch447 at the block-2/3 boundary as in X1, but freeze block-3's ln_2
normalization statistics (mean and variance per token) at their CLEAN-run
values, so the perturbation reaches the MLP only through the content change
along coordinate 447, not through renormalization of every coordinate.
If the block-wide inverse response is LayerNorm mechanics, it should collapse
under freezing; a neuron genuinely reading the channel should keep responding.
Measures all 3072 block-3 neurons, frozen and unfrozen, at alpha 0.5/1.0/1.5.
Writes results/exp_x6_lnfrozen.json."""

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
OUT = os.path.join(RESULTS_DIR, "exp_x6_lnfrozen.json")
CTX, N_EVAL, N_MLP = 512, 64, 3072

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0][:N_EVAL * CTX].view(
    N_EVAL, CTX)

LN2 = model.transformer.h[3].ln_2

# ---- pass 0: cache clean per-token ln_2 input statistics -------------------
clean_mu, clean_var = [], []
h_stat = LN2.register_forward_pre_hook(
    lambda m, inp: (clean_mu.append(inp[0].mean(-1).float().cpu()),
                    clean_var.append(inp[0].var(-1, unbiased=False)
                                     .float().cpu()))[0])
for i in range(0, N_EVAL, 8):
    model(ids[i:i + 8].to(DEV))
h_stat.remove()
MU = torch.cat(clean_mu).view(N_EVAL, CTX)      # [chunks, ctx]
VAR = torch.cat(clean_var).view(N_EVAL, CTX)
print("clean ln_2 stats cached", flush=True)


def mean_acts(alpha, frozen):
    sums = torch.zeros(N_MLP)
    cap = []
    h_cap = model.transformer.h[3].mlp.c_proj.register_forward_pre_hook(
        lambda m, inp: cap.append(inp[0].float().sum((0, 1)).cpu()))
    hooks = []
    if alpha != 1.0:
        def scale_hook(m, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            hh = t.clone()
            hh[..., 447] = hh[..., 447] * alpha
            return (hh,) + out[1:] if isinstance(out, tuple) else hh
        hooks.append(model.transformer.h[2].register_forward_hook(scale_hook))
    state = {"i0": 0}
    if frozen:
        def ln_hook(m, inp, out):
            x = inp[0]
            b = x.shape[0]
            mu = MU[state["i0"]:state["i0"] + b].to(x.device)[:, :, None]
            var = VAR[state["i0"]:state["i0"] + b].to(x.device)[:, :, None]
            return ((x - mu) / torch.sqrt(var + m.eps)) * m.weight + m.bias
        hooks.append(LN2.register_forward_hook(ln_hook))
    for i in range(0, N_EVAL, 8):
        state["i0"] = i
        model(ids[i:i + 8].to(DEV))
        sums += torch.stack(cap).sum(0)
        cap.clear()
    h_cap.remove()
    for h in hooks:
        h.remove()
    return sums / (N_EVAL * CTX)


base = mean_acts(1.0, frozen=False)
base_frozen = mean_acts(1.0, frozen=True)
sanity = (base_frozen - base).abs().max().item()
print(f"sanity: frozen@alpha=1 vs clean, max |diff| {sanity:.2e}", flush=True)

RES = {"sanity_max_abs_diff_alpha1": sanity}
denom = base.abs().clamp_min(1e-3)
active = base.abs() > 0.01
for frozen in (False, True):
    key = "frozen" if frozen else "unfrozen"
    lo = mean_acts(0.5, frozen)
    hi = mean_acts(1.5, frozen)
    rel_lo = (lo - base) / denom
    rel_hi = (hi - base) / denom
    r = rel_lo[active]
    inv = active & (rel_lo > 0.10) & (rel_hi < -0.10)
    RES[key] = {
        "median_lo": round(rel_lo[active].median().item(), 4),
        "median_hi": round(rel_hi[active].median().item(), 4),
        "q95_lo": round(rel_lo[active].quantile(0.95).item(), 4),
        "n_inverse_both_ge10pct": int(inv.sum()),
        "n614": {"lo": round(rel_lo[614].item(), 4),
                 "hi": round(rel_hi[614].item(), 4),
                 "pct_rank_lo": round(
                     (r < rel_lo[614]).float().mean().item(), 4),
                 "inverse_both": bool(inv[614])}}
    print(f"{key}: median {RES[key]['median_lo']}/{RES[key]['median_hi']}  "
          f"#614 {RES[key]['n614']['lo']}/{RES[key]['n614']['hi']} "
          f"(pct {RES[key]['n614']['pct_rank_lo']})  "
          f"inverse-both n={int(inv.sum())}", flush=True)

with open(OUT, "w") as f:
    json.dump(RES, f, indent=1)
print("DONE_X6", flush=True)
