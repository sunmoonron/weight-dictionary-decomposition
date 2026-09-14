"""X5: is dCE roughly linear in ablated write energy? Ablate the top-k block-3
counter-writers (k = 1,3,5,7,9, ranked by counter write energy) and measure
dCE at each k, plus 3 random 9-neuron control sets.
Writes results/exp_x5_lin.json."""

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
OUT = os.path.join(RESULTS_DIR, "exp_x5_lin.json")
CTX, N_EVAL, N_MLP, B3 = 512, 32, 3072, 3

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
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

RES = {"ce_clean": round(ce_clean, 4), "top9": top9, "curve": [],
       "controls_9": []}
cum_e = 0.0
for k in [1, 3, 5, 7, 9]:
    sub = top9[:k]
    e = float(energy[torch.tensor(sub)].sum())
    ce, _ = run_ce(set(sub))
    RES["curve"].append({"k": k, "energy": round(e, 1),
                         "dce": round(ce - ce_clean, 5)})
    print(f"k={k}: energy {e:.0f} dCE {ce - ce_clean:+.5f}", flush=True)
g = torch.Generator().manual_seed(7)
for t in range(3):
    rnd = torch.randint(0, N_MLP, (9,), generator=g).tolist()
    ce, _ = run_ce(set(rnd))
    RES["controls_9"].append({"neurons": rnd,
                              "energy": round(float(
                                  energy[torch.tensor(rnd)].sum()), 1),
                              "dce": round(ce - ce_clean, 5)})
    print(f"ctrl {t}: dCE {ce - ce_clean:+.5f}", flush=True)
with open(OUT, "w") as f:
    json.dump(RES, f, indent=1)
print("DONE_X5", flush=True)
