"""Write Dynamics Ep.2b: mechanism vs representation for mlp3#614, and the
counter-subpopulation ablation, on GPT-2 small.

  mechanism  r( a_614(t), pre-block-3 projection along d_614 ) over all
             tokens, vs the same r for 64 random block-3 neurons. Strongly
             negative and unusual => fires IN RESPONSE to a deficit along its
             own axis (corrective mechanism), not merely "writes function
             words".
  ablation   zero ALL block-3 counter-neurons at once (neurons whose top-3
             cases have mean survival < 0, count >= 10) vs an equal-sized
             random block-3 neuron set: dCE overall and on affected tokens.
             Collective necessity test the single-neuron ablation can't give.
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
import torch.nn.functional as Fn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(0)
torch.set_grad_enabled(False)
DEV = "cuda:0"
OUT = RESULTS_DIR
CTX, N_EVAL, L, N_MLP = 512, 64, 6, 3072
NEURON = 614

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
D = model.config.n_embd

text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0][:N_EVAL * CTX].view(N_EVAL, CTX)

acts3 = []
h = model.transformer.h[3].mlp.c_proj.register_forward_pre_hook(
    lambda m, inp: acts3.append(inp[0].half().cpu()))
pre3, ces = [], []
for i in range(0, N_EVAL, 4):
    out = model(ids[i:i + 4].to(DEV), output_hidden_states=True)
    pre3.append(out.hidden_states[3].float().cpu())     # pre-block-3 state
    lg = out.logits[:, :-1].float()
    ces.append(Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                ids[i:i + 4, 1:].to(DEV).reshape(-1),
                                reduction="none").cpu())
    del out
h.remove()
A3 = torch.cat(acts3).view(-1, N_MLP)
PRE3 = torch.cat(pre3).view(-1, D)
PRE3 = PRE3 - PRE3.mean(0)
clean_ce = torch.cat(ces)
NT = PRE3.shape[0]

w3 = model.transformer.h[3].mlp.c_proj.weight.detach().float().cpu()
wn3 = w3.norm(dim=-1)
d3 = w3 / wn3[:, None].clamp_min(1e-8)


def mech_r(j):
    a = A3[:, j].float()
    p = PRE3 @ d3[j]
    a, p = a - a.mean(), p - p.mean()
    return (a @ p / (a.norm() * p.norm()).clamp_min(1e-9)).item()


g = torch.Generator().manual_seed(31)
rand_js = torch.randint(0, N_MLP, (64,), generator=g).tolist()
rand_rs = torch.tensor([mech_r(j) for j in rand_js])
res = {"mech_614_r": round(mech_r(NEURON), 3),
       "mech_random_mean": round(rand_rs.mean().item(), 3),
       "mech_random_p5_p95": [round(rand_rs.quantile(0.05).item(), 3),
                              round(rand_rs.quantile(0.95).item(), 3)]}
print("mechanism:", res, flush=True)

# ---- counter-subpopulation from block-3 top-3 cases ----
WN = {b: model.transformer.h[b].mlp.c_proj.weight.detach().float().cpu()
      .norm(dim=-1) for b in range(L + 1)}
acts_all = {b: [] for b in range(L + 1)}
hooks = [model.transformer.h[b].mlp.c_proj.register_forward_pre_hook(
    (lambda b_: lambda m, inp: acts_all[b_].append(inp[0].half().cpu()))(b))
    for b in range(L + 1)]
l4 = []
for i in range(0, N_EVAL, 4):
    out = model(ids[i:i + 4].to(DEV), output_hidden_states=True)
    l4.append(out.hidden_states[4].float().cpu())
    del out
for hh in hooks:
    hh.remove()
acts_all = {b: torch.cat(a).view(-1, N_MLP) for b, a in acts_all.items()}
L4 = torch.cat(l4).view(-1, D)
L4 = L4 - L4.mean(0)
TRUE = torch.cat([acts_all[b].float() * WN[b] for b in range(L + 1)], dim=1)
tops = []
for s0 in range(0, NT, 4096):
    tops.append(TRUE[s0:s0 + 4096].abs().to(DEV).topk(3, 1).indices.cpu())
top3 = torch.cat(tops)
col = top3.flatten()
m3 = (col // N_MLP) == 3
t3 = (torch.arange(NT)[:, None].expand(-1, 3).flatten())[m3]
j3 = (col % N_MLP)[m3]
c3 = TRUE[t3, col[m3]]
s3 = ((L4[t3] * d3[j3]).sum(-1) / c3).clamp(-4, 5)   # birth survival (post-b3)
neuron_stats = {}
for jj, ss in zip(j3.tolist(), s3.tolist()):
    st = neuron_stats.setdefault(jj, [0, 0.0])
    st[0] += 1
    st[1] += ss
counter_set = [j for j, (n, tot) in neuron_stats.items()
               if n >= 10 and tot / n < 0]
res["counter_subpop_size"] = len(counter_set)
print(f"{len(counter_set)} block-3 counter-neurons (n>=10, mean birth surv<0)",
      flush=True)


def ablate_ce(neurons):
    idx = torch.tensor(neurons, device=DEV)

    def hook(m, inp):
        x = inp[0].clone()
        x[:, :, idx] = 0.0
        return (x,)

    hh = model.transformer.h[3].mlp.c_proj.register_forward_pre_hook(hook)
    tot = []
    try:
        for i in range(0, N_EVAL, 4):
            lg = model(ids[i:i + 4].to(DEV)).logits[:, :-1].float()
            tot.append(Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                        ids[i:i + 4, 1:].to(DEV).reshape(-1),
                                        reduction="none").cpu())
    finally:
        hh.remove()
    return torch.cat(tot)


ce_counter = ablate_ce(counter_set)
rand_set = torch.randperm(N_MLP, generator=g)[:len(counter_set)].tolist()
ce_rand = ablate_ce(rand_set)
res["ablate_counter_subpop_dce"] = round((ce_counter.mean()
                                          - clean_ce.mean()).item(), 4)
res["ablate_random_matched_dce"] = round((ce_rand.mean()
                                          - clean_ce.mean()).item(), 4)
print(f"dCE counter-subpop {res['ablate_counter_subpop_dce']}  "
      f"random-matched {res['ablate_random_matched_dce']}", flush=True)

with open(f"{OUT}/exp_s4_small614.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE_S4", flush=True)
