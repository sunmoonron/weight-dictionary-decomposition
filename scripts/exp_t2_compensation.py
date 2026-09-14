"""Ep.7: who restores the 88%? (GPT-2 small, ch447 crew ablation)

S8: ablating the four regulators shifts ch447 by the predicted -12.4 directly,
but only -1.49 remains at L11 -- the network restores 88%. Attribute the
restoration: per (block, component-type) change in the ch447 write between
clean and crew-ablated runs, plus named neuron-level compensators, plus the
passive-vs-active test (is the compensation a uniform relative upscaling --
LayerNorm-mediated -- or concentrated in specific components?).
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
CTX, N_EVAL, N_MLP, CH = 512, 32, 3072, 447
REGS = {3: 614, 4: 1894, 5: 1790, 6: 3039}
NB = 12

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
D = model.config.n_embd

text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0][:N_EVAL * CTX].view(N_EVAL, CTX)

W447 = {b: model.transformer.h[b].mlp.c_proj.weight.detach()
        .float().cpu()[:, CH] for b in range(NB)}          # per-neuron ch447 wt


def run(ablate):
    attn_w = {b: [] for b in range(NB)}    # attn write into ch447 per token
    mlp_w = {b: [] for b in range(NB)}
    acts = {b: [] for b in range(NB)}      # mlp post-GELU acts (for naming)
    hooks = []
    for b in range(NB):
        hooks.append(model.transformer.h[b].attn.c_proj.register_forward_hook(
            (lambda b_: lambda m, i, o: attn_w[b_].append(
                o[:, :, CH].float().cpu()))(b)))
        hooks.append(model.transformer.h[b].mlp.c_proj.register_forward_hook(
            (lambda b_: lambda m, i, o: mlp_w[b_].append(
                o[:, :, CH].float().cpu()))(b)))

        def mk(b_):
            def f(m, inp):
                x = inp[0]
                acts[b_].append(x.half().cpu())
                if ablate and b_ in REGS:
                    x = x.clone()
                    x[:, :, REGS[b_]] = 0.0
                    return (x,)
            return f
        hooks.append(model.transformer.h[b].mlp.c_proj
                     .register_forward_pre_hook(mk(b)))
    for i in range(0, N_EVAL, 4):
        out = model(ids[i:i + 4].to(DEV))
        del out
    for h in hooks:
        h.remove()
    return ({b: torch.cat(v).flatten() for b, v in attn_w.items()},
            {b: torch.cat(v).flatten() for b, v in mlp_w.items()},
            {b: torch.cat(v).view(-1, N_MLP) for b, v in acts.items()})


attn_c, mlp_c, acts_c = run(False)
attn_a, mlp_a, acts_a = run(True)
print("both runs captured", flush=True)

# NOTE: hooked mlp c_proj OUTPUT under ablation still includes the zeroed
# neuron's absence automatically (hook order: pre-hook zeroes before forward).
res = {"per_component_delta": {}, "ratio_check": {}}
tot = 0.0
for b in range(NB):
    da = (attn_a[b] - attn_c[b]).mean().item()
    dm = (mlp_a[b] - mlp_c[b]).mean().item()
    tot += da + dm
    base_a, base_m = attn_c[b].mean().item(), mlp_c[b].mean().item()
    res["per_component_delta"][f"attn{b}"] = round(da, 3)
    res["per_component_delta"][f"mlp{b}"] = round(dm, 3)
    res["ratio_check"][f"attn{b}"] = round(da / base_a, 3) if abs(base_a) > 0.1 else None
    res["ratio_check"][f"mlp{b}"] = round(dm / base_m, 3) if abs(base_m) > 0.1 else None
res["sum_of_deltas"] = round(tot, 2)
print("per-component ch447 delta (ablated - clean):", flush=True)
for k, v in res["per_component_delta"].items():
    if abs(v) > 0.05:
        print(f"  {k}: {v:+.3f} (rel {res['ratio_check'][k]})", flush=True)
print(f"sum of deltas {tot:+.2f} (should ~= meas -1.49 - pred -12.4 = +10.9 "
      f"minus the removed writes)", flush=True)

# ---- name the compensating neurons (blocks 4..10, crew excluded) ----
movers = {}
for b in range(4, 11):
    dvec = (acts_a[b].float().mean(0) - acts_c[b].float().mean(0)) * W447[b]
    if b in REGS:
        dvec[REGS[b]] = 0.0
    for j in dvec.abs().topk(3).indices.tolist():
        movers[f"mlp{b}#{j}"] = round(dvec[j].item(), 3)
res["top_neuron_compensators"] = dict(
    sorted(movers.items(), key=lambda kv: -abs(kv[1]))[:10])
print("top neuron-level compensators:", res["top_neuron_compensators"], flush=True)

with open(f"{OUT}/exp_t2_compensation.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE_T2", flush=True)
