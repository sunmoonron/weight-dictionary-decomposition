"""Ep.10 (season finale): which component of the primaries' writes is the leash?

Ep.9: reserves are channel-followers that nonetheless surge when the crew is
ablated -> disinhibition. Dissect the crew's writes surgically:

  A  full crew ablation (Ep.9 replication)
  B  remove ONLY the non-ch447 components of the crew's writes (leash cut,
     channel write preserved) -> pure disinhibition predicted: reserves rise
  C  remove ONLY the ch447 component (channel deficit, leash intact) ->
     followers should dip slightly; NO unmasking predicted
  E  condition B restricted to #614 alone -> does one neuron hold the leash?

Readout: the five reserves' activation changes vs clean, random controls,
and ch447 at L11 per condition (sanity: B ~ clean channel, C ~ ablated).
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

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
D = model.config.n_embd

text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0][:N_EVAL * CTX].view(N_EVAL, CTX)

W_ROW = {b: model.transformer.h[b].mlp.c_proj.weight.detach()
         .float()[REGS[b]].to(DEV) for b in REGS}            # [768] per crew nrn
g = torch.Generator().manual_seed(53)
CTRLS = [(b, int(torch.randint(0, N_MLP, (1,), generator=g)))
         for b, _ in RESERVES]
WATCH = RESERVES + CTRLS


def run(mode, only614=False):
    """mode: None | 'full' | 'non447' | 'only447'."""
    grabbed = {k: [] for k in WATCH}
    state = {}
    hooks = []
    crew_blocks = [3] if only614 else list(REGS)
    for b in sorted({bb for bb, _ in WATCH} | set(crew_blocks)):
        def mk_pre(b_):
            def f(m, inp):
                x = inp[0]
                for (bb, jj) in WATCH:
                    if bb == b_:
                        grabbed[(bb, jj)].append(x[:, :, jj].float().cpu())
                if mode and b_ in crew_blocks:
                    state[b_] = x[:, :, REGS[b_]].clone()     # activation a(t)
            return f
        hooks.append(model.transformer.h[b].mlp.c_proj
                     .register_forward_pre_hook(mk_pre(b)))
        if mode and b in crew_blocks:
            def mk_post(b_):
                w = W_ROW[b_]
                w_rm = w.clone()
                if mode == "non447":
                    w_rm[CH] = 0.0                           # remove all BUT ch447? no:
                    # w_rm currently = w with ch447 zeroed = the NON-447 part
                elif mode == "only447":
                    w_rm = torch.zeros_like(w)
                    w_rm[CH] = w[CH]                         # just the ch447 part
                def f(m, i, o):
                    return o - state[b_][:, :, None] * w_rm
                return f
            hooks.append(model.transformer.h[b].mlp.c_proj
                         .register_forward_hook(mk_post(b)))
    l11 = []
    for i in range(0, N_EVAL, 4):
        out = model(ids[i:i + 4].to(DEV), output_hidden_states=True)
        l11.append(out.hidden_states[11][:, :, CH].float().cpu())
        del out
    for h in hooks:
        h.remove()
    return ({k: torch.cat(v).flatten() for k, v in grabbed.items()},
            torch.cat(l11).mean().item())


base, ch_base = run(None)
conds = {}
for tag, mode, o614 in [("A_full", "full", False), ("B_leashcut", "non447", False),
                        ("C_channelcut", "only447", False),
                        ("E_leashcut_614only", "non447", True)]:
    conds[tag] = run(mode, o614)
    print(f"ran {tag}; ch447@L11 {conds[tag][1]:.1f} (clean {ch_base:.1f})",
          flush=True)

res = {"ch447_L11": {"clean": round(ch_base, 1),
                     **{t: round(v[1], 1) for t, v in conds.items()}}}
for (b, j) in WATCH:
    role = "RESERVE" if (b, j) in RESERVES else "control"
    b0 = base[(b, j)].abs().mean().clamp_min(1e-6)
    row = {"role": role}
    for t, (acts, _) in conds.items():
        row[t] = round(((acts[(b, j)].abs().mean() - base[(b, j)].abs().mean())
                        / b0).item(), 3)
    res[f"mlp{b}#{j}"] = row
    print(f"mlp{b}#{j} [{role}]: " + "  ".join(
        f"{t}={row[t]:+.3f}" for t in conds), flush=True)

for role in ["RESERVE", "control"]:
    res[f"_mean_{role}"] = {t: round(float(torch.tensor(
        [res[k][t] for k in res if isinstance(res[k], dict)
         and res[k].get("role") == role]).mean()), 3) for t in conds}
print("means:", {r: res[f"_mean_{r}"] for r in ["RESERVE", "control"]}, flush=True)

# note: in 'full' mode the write removal is o - a*w == exact neuron ablation
with open(f"{OUT}/exp_u2_leash.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE_U2", flush=True)
