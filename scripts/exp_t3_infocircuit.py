"""Ep.8: earn or retire the word "information" -- per-circuit (GPT-2 small).

For named neurons (the ch447 regulator #614, the residue counter-writer
mlp5#2070, the heavily-erased writers mlp3#1848 / mlp3#2614, and 3 random
controls per block), predict the neuron's activation from the FULL residual
state at two depths (post-b6 and post-b10) via closed-form ridge, train on the
first half of tokens, test on the second. Then split test tokens into
survival terciles of that neuron's write and report per-tercile R^2 with the
globally trained readout: does information persist on exactly the tokens
where the direction was erased?
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
CTX, N_EVAL, N_MLP = 512, 32, 3072
TARGETS = [(3, 614), (3, 1848), (3, 2614), (5, 2070)]
LEVELS = [7, 11]                       # post-b6, post-b10 (raw)

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
D = model.config.n_embd

text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0][:N_EVAL * CTX].view(N_EVAL, CTX)

need_b = sorted({b for b, _ in TARGETS})
W = {b: model.transformer.h[b].mlp.c_proj.weight.detach().float().cpu()
     for b in need_b}
WN = {b: W[b].norm(dim=-1) for b in need_b}
g = torch.Generator().manual_seed(19)
CONTROLS = [(b, int(j)) for b in need_b
            for j in torch.randint(0, N_MLP, (3,), generator=g)]

acts = {b: [] for b in need_b}
hooks = [model.transformer.h[b].mlp.c_proj.register_forward_pre_hook(
    (lambda b_: lambda m, inp: acts[b_].append(inp[0].half().cpu()))(b))
    for b in need_b]
lv = {l: [] for l in LEVELS}
for i in range(0, N_EVAL, 4):
    out = model(ids[i:i + 4].to(DEV), output_hidden_states=True)
    for l in LEVELS:
        lv[l].append(out.hidden_states[l].float().cpu())
    del out
for h in hooks:
    h.remove()
acts = {b: torch.cat(a).view(-1, N_MLP) for b, a in acts.items()}
lv = {l: torch.cat(v).view(-1, D) for l, v in lv.items()}
NT = lv[LEVELS[0]].shape[0]
split = NT // 2
print(f"{NT} states captured", flush=True)

ridges = {}
for l in LEVELS:
    X = lv[l] - lv[l][:split].mean(0)
    Xtr = X[:split].to(DEV)
    G = Xtr.T @ Xtr + 1e-2 * split * torch.eye(D, device=DEV)
    ridges[l] = (X, Xtr, torch.linalg.cholesky(G))


def probe(b, j):
    y = acts[b][:, j].float()
    d = (W[b][j] / WN[b][j]).clone()
    out = {}
    for l in LEVELS:
        X, Xtr, chol = ridges[l]
        ytr = y[:split].to(DEV)
        w = torch.cholesky_solve((Xtr.T @ ytr)[:, None], chol)[:, 0].cpu()
        yte = y[split:]
        pred = X[split:] @ w
        var = ((yte - yte.mean()) ** 2).sum().clamp_min(1e-9)
        out[f"L{l-1}_r2"] = round((1 - ((yte - pred) ** 2).sum() / var).item(), 3)
        # per-survival-tercile R^2 on test tokens with a meaningful write
        c = y[split:] * WN[b][j]
        surv = ((X[split:] @ d) / c.where(c.abs() > 1e-6, torch.ones_like(c)))
        m = c.abs() > 0.5
        if m.sum() > 300:
            edges = surv[m].quantile(torch.tensor([1 / 3, 2 / 3]))
            terc = torch.bucketize(surv, edges)
            r2t = []
            for t in range(3):
                mm = m & (terc == t)
                v = ((yte[mm] - yte[mm].mean()) ** 2).sum().clamp_min(1e-9)
                r2t.append(round((1 - ((yte[mm] - pred[mm]) ** 2).sum() / v)
                                 .item(), 3))
            out[f"L{l-1}_r2_by_surv_tercile"] = r2t
            if t == 2:
                out[f"L{l-1}_surv_tercile_edges"] = [round(e.item(), 2)
                                                    for e in edges]
    return out


res = {"targets": {}, "controls": {}}
for b, j in TARGETS:
    res["targets"][f"mlp{b}#{j}"] = probe(b, j)
    print(f"mlp{b}#{j}: {res['targets'][f'mlp{b}#{j}']}", flush=True)
ctrl_r2 = {l: [] for l in LEVELS}
for b, j in CONTROLS:
    p = probe(b, j)
    for l in LEVELS:
        ctrl_r2[l].append(p[f"L{l-1}_r2"])
res["controls"] = {f"L{l-1}_r2_median": round(float(
    torch.tensor(ctrl_r2[l]).median()), 3) for l in LEVELS}
print("controls:", res["controls"], flush=True)

with open(f"{OUT}/exp_t3_infocircuit.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE_T3", flush=True)
