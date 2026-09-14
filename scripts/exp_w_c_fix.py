"""Part C of the audit, rerun alone: #614 dose-response under ch447 scaling.
Fix: GPT2Block forward hooks receive a plain tensor in this transformers
version; handle tensor-or-tuple. Appends C_dose_response into exp_w_audit.json.
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

torch.manual_seed(123)
torch.set_grad_enabled(False)
DEV = "cuda:0"
OUT = os.path.join(RESULTS_DIR, "exp_w_audit.json")
CTX, N_EVAL, N_MLP = 512, 64, 3072

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    if t.strip())
eval_ids = tok(text, return_tensors="pt").input_ids[0][:N_EVAL * CTX].view(
    N_EVAL, CTX)

g = torch.Generator().manual_seed(999)
ctrl = torch.randint(0, N_MLP, (3,), generator=g).tolist()
watch = [614] + [c for c in ctrl if c != 614]


def mean_acts(alpha):
    sums = torch.zeros(len(watch))
    cap = []
    h_cap = model.transformer.h[3].mlp.c_proj.register_forward_pre_hook(
        lambda m, inp: cap.append(inp[0][:, :, watch].float().cpu()))
    h_scale = None
    if alpha != 1.0:
        def scale_hook(m, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            hh = t.clone()
            hh[..., 447] = hh[..., 447] * alpha
            return (hh,) + out[1:] if isinstance(out, tuple) else hh
        h_scale = model.transformer.h[2].register_forward_hook(scale_hook)
    for i in range(0, N_EVAL, 8):
        model(eval_ids[i:i + 8].to(DEV))
        sums += torch.cat(cap).view(-1, len(watch)).sum(0)
        cap.clear()
    h_cap.remove()
    if h_scale is not None:
        h_scale.remove()
    return sums / (N_EVAL * CTX)


base = mean_acts(1.0)
lo = mean_acts(0.5)
hi = mean_acts(1.5)
rel = {"alpha_0.5": [round(((lo[i] - base[i]) / base[i].abs().clamp_min(
           1e-6)).item(), 4) for i in range(len(watch))],
       "alpha_1.5": [round(((hi[i] - base[i]) / base[i].abs().clamp_min(
           1e-6)).item(), 4) for i in range(len(watch))]}

RES = json.load(open(OUT))
RES.pop("C_error", None)
RES["C_dose_response"] = {
    "watched_neurons": [f"mlp3#{j}" for j in watch],
    "rel_change": rel,
    "note": "index 0 is #614; rest are fresh random b3 controls",
    "expected_614": {"alpha_0.5": 0.275, "alpha_1.5": -0.221}}
with open(OUT, "w") as f:
    json.dump(RES, f, indent=1)
print(f"C: #614 a0.5 {rel['alpha_0.5'][0]} (expect ~+0.275)  "
      f"a1.5 {rel['alpha_1.5'][0]} (expect ~-0.221)  "
      f"controls {rel['alpha_0.5'][1:]} / {rel['alpha_1.5'][1:]}", flush=True)
print("DONE_C_FIX", flush=True)
