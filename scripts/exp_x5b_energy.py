"""X5b: energy-matched random controls for the block-3 ablation.

The random nine-neuron controls of exp_x5_lin carry far less write energy than the nine strongest
counter-writers, so they do not test whether the behavioral cost of the ablation is special to those
neurons or follows from the energy removed. This control draws random sets of block-3 neurons until
their summed write energy matches that of the top-9 set to within 5%, from two pools: all other
block-3 neurons, and the other counter-writers only. Same states, same hooks, same ablation as
exp_x5_lin. Writes results/exp_x5b_energy.json."""

import json
import os

_HF = "/data/mechinterp/hf"          # the original machine's cache; ignored elsewhere
if os.path.isdir(_HF):
    os.environ.setdefault("HF_HOME", _HF)
RESULTS_DIR = os.environ.get("WDD_RESULTS", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(0)
torch.set_grad_enabled(False)
DEV = "cuda:0"
OUT = os.path.join(RESULTS_DIR, "exp_x5b_energy.json")
CTX, N_EVAL, N_MLP, B3 = 512, 32, 3072, 3

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0][:N_EVAL * CTX].view(
    N_EVAL, CTX)

W = model.transformer.h[B3].mlp.c_proj.weight.detach().float().cpu()
WN = W.norm(dim=-1)
DIRS = W / WN[:, None].clamp_min(1e-8)


def run_ce(zero_set):
    hook = None
    if zero_set:
        zs = torch.tensor(sorted(zero_set))

        def h(m, inp):
            x = inp[0].clone()
            x[:, :, zs] = 0.0
            return (x,)
        hook = model.transformer.h[B3].mlp.c_proj.register_forward_pre_hook(h)
    ces, caps = [], []
    h_cap = model.transformer.h[B3].mlp.c_proj.register_forward_pre_hook(
        lambda m, inp: caps.append(inp[0].half().cpu()))
    for i in range(0, N_EVAL, 4):
        out = model(ids[i:i + 4].to(DEV))
        lg = out.logits[:, :-1].float().cpu()
        ces.append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                   ids[i:i + 4, 1:].reshape(-1),
                                   reduction="none"))
        del out, lg
    h_cap.remove()
    if hook is not None:
        hook.remove()
    return (torch.cat(ces).mean().item(),
            torch.cat(caps).view(-1, N_MLP).float())


ce_clean, acts = run_ce(set())
C = acts * WN                                       # true write coefficients
post = None                                          # birth survival vs post-b3 state
# post-block-3 state for birth survival
levels = []
for i in range(0, N_EVAL, 4):
    out = model(ids[i:i + 4].to(DEV), output_hidden_states=True)
    levels.append(out.hidden_states[B3 + 1].float().cpu())
    del out
X4 = torch.cat(levels).view(-1, model.config.n_embd)
X4 = X4 - X4.mean(0)
S = (X4 @ DIRS.T) / C.where(C.abs() > 1e-6, torch.ones_like(C))
big = C.abs() > 0.5
n_big = big.sum(0)
mean_birth = (S * big).sum(0) / n_big.clamp_min(1)
counter = (mean_birth < 0) & (n_big >= 10)
energy = (C ** 2 * big).sum(0)
order = energy.masked_fill(~counter, -1.0).argsort(descending=True)
top9 = order[:9].tolist()
print(f"top-9 counter-writers: {top9}", flush=True)


E9 = float(energy[torch.tensor(top9)].sum())
ce_top9, _ = run_ce(set(top9))
RES = {"ce_clean": round(ce_clean, 4), "top9": top9, "top9_energy": round(E9, 1), "top9_dce": round(ce_top9 - ce_clean, 5), "matched_controls": []}
print(f"top-9: energy {E9:.0f} dCE {ce_top9 - ce_clean:+.5f}", flush=True)
g = torch.Generator().manual_seed(11)
en = energy.clone()
pools = {"any_block3_neuron": [j for j in range(N_MLP) if j not in top9], "other_counter_writers": [j for j in torch.nonzero(counter)[:, 0].tolist() if j not in top9]}
for pool_name, pool in pools.items():
    pool_t = torch.tensor(pool)
    for t in range(3):
        for attempt in range(200):
            perm = pool_t[torch.randperm(len(pool), generator=g)]
            chosen, total = [], 0.0
            for j in perm.tolist():
                if total + float(en[j]) > 1.05 * E9: continue
                chosen.append(j); total += float(en[j])
                if total >= 0.95 * E9: break
            if 0.95 * E9 <= total <= 1.05 * E9: break
        ce, _ = run_ce(set(chosen))
        RES["matched_controls"].append({"pool": pool_name, "neurons": chosen, "n": len(chosen), "energy": round(total, 1), "dce": round(ce - ce_clean, 5)})
        print(f"{pool_name} set {t}: {len(chosen)} neurons, energy {total:.0f} ({total / E9:.2f} of top-9), dCE {ce - ce_clean:+.5f}", flush=True)
with open(OUT, "w") as f:
    json.dump(RES, f, indent=1)
print("DONE_X5B", flush=True)
