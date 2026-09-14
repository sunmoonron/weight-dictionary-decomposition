"""Experiment C: does WDD recover the TRUE computation?

For MLP atoms selected by OMP at layer 6, compare the OMP coefficient against
the ground-truth write coefficient from the forward pass: neuron j of block b
actually wrote a_j * w_j into the stream (a_j = post-GELU activation, w_j =
c_proj row). Along the unit atom w_j/|w_j| the true coefficient is a_j * |w_j|.
If OMP coefficients match, the decomposition recovers the model's real
generative story from the state vector alone -- no internals needed.
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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(0)
torch.set_grad_enabled(False)
DEV = "cuda:0"          # launched with CUDA_VISIBLE_DEVICES to pick the card
OUT = RESULTS_DIR

CTX, N_EVAL, L, K = 512, 64, 6, 32
model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
D = model.config.n_embd

text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
eval_ids = tok(text, return_tensors="pt").input_ids[0][:N_EVAL * CTX].view(N_EVAL, CTX)

# capture post-GELU MLP activations of blocks 0..L via pre-hook on c_proj
acts = {b: [] for b in range(L + 1)}
hooks = [model.transformer.h[b].mlp.c_proj.register_forward_pre_hook(
    (lambda b_: lambda m, inp: acts[b_].append(inp[0].half().cpu()))(b))
    for b in range(L + 1)]
resid = []
for i in range(0, N_EVAL, 4):
    out = model(eval_ids[i:i + 4].to(DEV), output_hidden_states=True)
    resid.append(out.hidden_states[L + 1].float().cpu())
    del out
for h in hooks:
    h.remove()
acts = {b: torch.cat(a).view(-1, 3072) for b, a in acts.items()}
X = torch.cat(resid).view(-1, D)
MU = X.mean(0)          # centering only; matches exp A's convention closely
X = X - MU
print(f"captured {X.shape[0]} states + MLP acts for blocks 0..{L}", flush=True)

# rebuild the same weight dictionary (same order as exp A)
atoms, fams, meta = [], [], []          # meta: (block, neuron) for mlp atoms
def add(t, fam, m):
    atoms.append(t.detach().float().cpu())
    fams.extend([fam] * t.shape[0])
    meta.extend(m)

add(model.transformer.wte.weight, "wte", [None] * 50257)
add(model.transformer.wpe.weight, "wpe", [None] * 1024)
for b in range(L + 1):
    blk = model.transformer.h[b]
    w = blk.mlp.c_proj.weight
    add(w, f"mlp{b}", [(b, j) for j in range(3072)])
    w_o = blk.attn.c_proj.weight
    for hd in range(12):
        add(torch.linalg.svd(w_o[hd * 64:(hd + 1) * 64], full_matrices=False).Vh,
            f"attn{b}", [None] * 64)
    add(torch.stack([blk.attn.c_proj.bias, blk.mlp.c_proj.bias]),
        f"bias{b}", [None, None])
A = torch.cat(atoms)
NORMS = A.norm(dim=-1)                   # |w_j| before unit-normalization
A = A / NORMS[:, None].clamp_min(1e-8)
print(f"dictionary {A.shape[0]} atoms", flush=True)

# OMP to k=32, returning selections and final joint-refit coefficients
A_dev = A.to(DEV)
sels, cofs = [], []
for s in range(0, len(X), 768):
    x = X[s:s + 768].to(DEV)
    B = len(x)
    sel = torch.zeros(B, K, dtype=torch.long, device=DEV)
    taken = torch.zeros(B, A_dev.shape[0], dtype=torch.bool, device=DEV)
    r = x.clone()
    for k in range(K):
        pick = (r @ A_dev.T).abs_().masked_fill_(taken, -1.0).argmax(-1)
        sel[:, k] = pick
        taken.scatter_(1, pick[:, None], True)
        A_S = A_dev[sel[:, :k + 1]]
        G = A_S @ A_S.transpose(1, 2) + 1e-5 * torch.eye(k + 1, device=DEV)
        c = torch.cholesky_solve(A_S @ x[:, :, None], torch.linalg.cholesky(G))
        r = x - (c.transpose(1, 2) @ A_S)[:, 0]
    sels.append(sel.cpu())
    cofs.append(c[:, :, 0].cpu())
sel = torch.cat(sels)
cof = torch.cat(cofs)
print("OMP done", flush=True)

# gather (omp_coeff, true_coeff) for every selected MLP atom
pred, true, blocks = [], [], []
for t in range(sel.shape[0]):
    for k in range(K):
        m = meta[sel[t, k].item()]
        if m is None:
            continue
        b, j = m
        pred.append(cof[t, k].item())
        true.append(acts[b][t, j].item() * NORMS[sel[t, k]].item())
        blocks.append(b)
pred = torch.tensor(pred)
true = torch.tensor(true)
blocks = torch.tensor(blocks)
print(f"{len(pred)} selected MLP-atom instances", flush=True)

def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-9)).item()

res = {"n": len(pred), "r_all": pearson(pred, true),
       "sign_agree": ((pred.sign() == true.sign()) | (true.abs() < 1e-4))
                     .float().mean().item(),
       "per_block": {}}
for b in range(L + 1):
    m = blocks == b
    if m.sum() > 100:
        res["per_block"][b] = {"n": int(m.sum()), "r": pearson(pred[m], true[m])}

# how activated was the selected neuron, really? percentile within its layer
q = []
for t in range(0, sel.shape[0], 16):        # subsample tokens for speed
    for k in range(K):
        m = meta[sel[t, k].item()]
        if m is None:
            continue
        b, j = m
        row = acts[b][t].abs()
        q.append((row < row[j].abs()).float().mean().item())
res["mean_activation_percentile"] = float(torch.tensor(q).mean())
print(json.dumps(res, indent=1), flush=True)
with open(f"{OUT}/exp_c_validate.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE_C", flush=True)
