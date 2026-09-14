"""Ep.9 (the boss's pick): is the reserve crew thermostatic?

Ep.7 named the components that restore 88% of ch447 when the regulator crew is
ablated. This asks the mechanism question: do those reserves respond causally
to CHANNEL LEVEL, and does their response GROW when the primaries are gone?

Six conditions: {intact, crew-ablated} x {alpha 1.0, 0.5, 1.5} where alpha
scales ch447 at the block-2/3 boundary. Watched: the five named reserves
(mlp5#1888, mlp6#834, mlp7#2402, mlp9#840, mlp10#900), the primary #614 for
reference, and five random controls. Readout: relative change in mean |act|
and in the neuron's ch447 write, per condition.

Signatures: reserves respond in the intact net -> co-regulators (shared gain).
Respond only under ablation -> true standby (gain scheduling). No causal
response to alpha at all -> Ep.7's "active compensation" needs rethinking.
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
RESERVES = [(5, 1888), (6, 834), (7, 2402), (9, 840), (10, 900)]
REF = (3, 614)

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")

text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0][:N_EVAL * CTX].view(N_EVAL, CTX)

W447 = {b: model.transformer.h[b].mlp.c_proj.weight.detach()
        .float().cpu()[:, CH] for b in range(12)}
g = torch.Generator().manual_seed(47)
CTRLS = [(b, int(torch.randint(0, N_MLP, (1,), generator=g)))
         for b, _ in RESERVES]
WATCH = RESERVES + [REF] + CTRLS


def run(alpha, ablate):
    grabbed = {k: [] for k in WATCH}
    hooks = []
    for b in sorted({b for b, _ in WATCH} | set(REGS)):
        def mk(b_):
            def f(m, inp):
                x = inp[0]
                for (bb, jj) in WATCH:
                    if bb == b_:
                        grabbed[(bb, jj)].append(x[:, :, jj].float().cpu())
                if ablate and b_ in REGS:
                    x = x.clone()
                    x[:, :, REGS[b_]] = 0.0
                    return (x,)
            return f
        hooks.append(model.transformer.h[b].mlp.c_proj
                     .register_forward_pre_hook(mk(b)))
    if alpha != 1.0:
        def bh(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            h[:, :, CH] = h[:, :, CH] * alpha
            return (h,) + out[1:] if isinstance(out, tuple) else h
        hooks.append(model.transformer.h[2].register_forward_hook(bh))
    for i in range(0, N_EVAL, 4):
        out = model(ids[i:i + 4].to(DEV))
        del out
    for h in hooks:
        h.remove()
    return {k: torch.cat(v).flatten() for k, v in grabbed.items()}


conds = {}
for ab in [False, True]:
    for al in [1.0, 0.5, 1.5]:
        conds[(ab, al)] = run(al, ab)
        print(f"ran ablate={ab} alpha={al}", flush=True)

res = {}
for (b, j) in WATCH:
    tag = ("RESERVE" if (b, j) in RESERVES else
           "PRIMARY" if (b, j) == REF else "control")
    row = {"role": tag}
    for ab in [False, True]:
        base = conds[(ab, 1.0)][(b, j)]
        b0 = base.abs().mean().clamp_min(1e-6)
        for al in [0.5, 1.5]:
            d = (conds[(ab, al)][(b, j)].abs().mean() - base.abs().mean()) / b0
            row[f"{'abl' if ab else 'intact'}_a{al}"] = round(d.item(), 3)
        row[f"{'abl' if ab else 'intact'}_write447"] = round(
            (base.mean() * W447[b][j]).item(), 3)
    res[f"mlp{b}#{j}"] = row
    print(f"mlp{b}#{j} [{tag}]: intact {row['intact_a0.5']:+.3f}/"
          f"{row['intact_a1.5']:+.3f}  ablated {row['abl_a0.5']:+.3f}/"
          f"{row['abl_a1.5']:+.3f}  w447(intact/abl) "
          f"{row['intact_write447']:+.2f}/{row['abl_write447']:+.2f}", flush=True)

# summary: mean inverse-response gain per role, per condition
def gain(role, ab):
    vals = [(res[k][f"{'abl' if ab else 'intact'}_a0.5"]
             - res[k][f"{'abl' if ab else 'intact'}_a1.5"]) / 2
            for k in res if res[k]["role"] == role]
    return round(float(torch.tensor(vals).mean()), 3)


res["_summary"] = {r: {"gain_intact": gain(r, False), "gain_ablated": gain(r, True)}
                   for r in ["RESERVE", "PRIMARY", "control"]}
print("summary (inverse-response gain, + = thermostatic):", res["_summary"],
      flush=True)

with open(f"{OUT}/exp_u1_reserves.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE_U1", flush=True)
