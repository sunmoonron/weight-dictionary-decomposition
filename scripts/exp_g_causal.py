"""Experiment G: does the decomposition TRACK CAUSAL CHANGES to the computation?

Two levels, per reviewer request.

G1 (decomposition linearity): inject a known delta directly into a cached
layer-6 state along one neuron's write direction; the atom's coefficient should
move by exactly that delta, and other coefficients should not move.

G2 (end-to-end causal tracking): intervene on the SOURCE computation -- scale
one neuron's post-GELU activation at one position in an earlier block (alpha in
{0, 2}), rerun the model, decompose the new layer-6 state, and test whether the
atom's coefficient change matches the predicted direct-write change
(alpha-1) * a_j * |w_j|. Downstream blocks react to the perturbed stream, so
the prediction covers only the direct component; strong correlation means the
decomposition reads real computational causes, not just curve-fits.
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

CTX, N_EVAL, L, K, N_MLP = 512, 16, 6, 64, 3072
N_CASES = 256
ALPHAS = [0.0, 2.0]

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
D = model.config.n_embd

text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
eval_ids = tok(text, return_tensors="pt").input_ids[0][:N_EVAL * CTX].view(N_EVAL, CTX)

# capture baseline activations and layer-6 states
acts = {b: [] for b in range(L + 1)}
hooks = [model.transformer.h[b].mlp.c_proj.register_forward_pre_hook(
    (lambda b_: lambda m, inp: acts[b_].append(inp[0].half().cpu()))(b))
    for b in range(L + 1)]
resid = []
for i in range(0, N_EVAL, 4):
    resid.append(model(eval_ids[i:i + 4].to(DEV), output_hidden_states=True)
                 .hidden_states[L + 1].float().cpu())
for h in hooks:
    h.remove()
acts = {b: torch.cat(a) for b, a in acts.items()}          # [chunks, CTX, 3072]
H6 = torch.cat(resid)                                       # [chunks, CTX, 768]
MU = H6.view(-1, D).mean(0)

# dictionary (same order as exp A/C/F)
atoms, meta = [], []
def add(t, m):
    atoms.append(t.detach().float().cpu())
    meta.extend(m)
add(model.transformer.wte.weight, [None] * 50257)
add(model.transformer.wpe.weight, [None] * 1024)
for b in range(L + 1):
    blk = model.transformer.h[b]
    add(blk.mlp.c_proj.weight, [(b, j) for j in range(N_MLP)])
    w_o = blk.attn.c_proj.weight
    for hd in range(12):
        add(torch.linalg.svd(w_o[hd * 64:(hd + 1) * 64], full_matrices=False).Vh,
            [None] * 64)
    add(torch.stack([blk.attn.c_proj.bias, blk.mlp.c_proj.bias]), [None, None])
A = torch.cat(atoms)
NORMS = A.norm(dim=-1)
A = A / NORMS[:, None].clamp_min(1e-8)
NA = A.shape[0]
mlp_start = {b: 51281 + b * (N_MLP + 768 + 2) for b in range(L + 1)}

# sample cases: (chunk, pos, block, neuron) with a strong true write
g = torch.Generator().manual_seed(11)
cases = []
while len(cases) < N_CASES:
    c = int(torch.randint(0, N_EVAL, (1,), generator=g))
    p = int(torch.randint(64, CTX, (1,), generator=g))
    b = int(torch.randint(0, L + 1, (1,), generator=g))
    a_vec = acts[b][c, p].float() * NORMS[mlp_start[b]:mlp_start[b] + N_MLP]
    j = int(a_vec.abs().argmax())          # that position's strongest write in b
    if a_vec[j].abs() > 1.0:
        cases.append((c, p, b, j, a_vec[j].item()))
print(f"{len(cases)} cases sampled, |write| median "
      f"{torch.tensor([abs(x[4]) for x in cases]).median():.2f}", flush=True)

# G2 forwards: perturb neuron j's post-GELU activation at position p only
def perturbed_h6(c, p, b, j, alpha):
    def hook(mod, inp):
        x = inp[0].clone()
        x[0, p, j] = x[0, p, j] * alpha
        return (x,)
    h = model.transformer.h[b].mlp.c_proj.register_forward_pre_hook(hook)
    try:
        out = model(eval_ids[c:c + 1].to(DEV), output_hidden_states=True)
        return out.hidden_states[L + 1][0, p].float().cpu()
    finally:
        h.remove()

states, tags = [], []
for ci, (c, p, b, j, w) in enumerate(cases):
    states.append(H6[c, p]); tags.append((ci, "base"))
    gidx = mlp_start[b] + j
    delta = w                                  # G1: inject +-|true write| exactly
    states.append(H6[c, p] + delta * A[gidx]); tags.append((ci, "inj+"))
    states.append(H6[c, p] - delta * A[gidx]); tags.append((ci, "inj-"))
    for al in ALPHAS:
        states.append(perturbed_h6(c, p, b, j, al)); tags.append((ci, f"a{al}"))
    if ci % 64 == 0:
        print(f"  forwards {ci}/{len(cases)}", flush=True)
X = torch.stack(states) - MU
print(f"{len(X)} states to decompose", flush=True)

# OMP with support + coefficients
A_dev = A.to(DEV)
sel_all, cof_all = [], []
for s in range(0, len(X), 640):
    x = X[s:s + 640].to(DEV)
    B = len(x)
    sel = torch.zeros(B, K, dtype=torch.long, device=DEV)
    taken = torch.zeros(B, NA, dtype=torch.bool, device=DEV)
    r = x.clone()
    for k in range(K):
        pick = (r @ A_dev.T).abs_().masked_fill_(taken, -1.0).argmax(-1)
        sel[:, k] = pick
        taken.scatter_(1, pick[:, None], True)
        A_S = A_dev[sel[:, :k + 1]]
        Gm = A_S @ A_S.transpose(1, 2) + 1e-5 * torch.eye(k + 1, device=DEV)
        cc = torch.cholesky_solve(A_S @ x[:, :, None], torch.linalg.cholesky(Gm))
        r = x - (cc.transpose(1, 2) @ A_S)[:, 0]
    sel_all.append(sel.cpu())
    cof_all.append(cc[:, :, 0].cpu())
sel = torch.cat(sel_all)
cof = torch.cat(cof_all)
print("OMP done", flush=True)

def coeff_of(row, gidx):
    hit = (sel[row] == gidx).nonzero()
    return cof[row, hit[0, 0]].item() if len(hit) else 0.0

rows_per_case = 3 + len(ALPHAS)
g1_pred, g1_meas, g1_leak = [], [], []
g2_pred, g2_meas = [], []
rng = torch.Generator().manual_seed(5)
for ci, (c, p, b, j, w) in enumerate(cases):
    base_row = ci * rows_per_case
    gidx = mlp_start[b] + j
    c0 = coeff_of(base_row, gidx)
    for off, dl in [(1, w), (2, -w)]:
        g1_pred.append(dl)
        g1_meas.append(coeff_of(base_row + off, gidx) - c0)
        others = torch.randint(0, NA, (20,), generator=rng)
        g1_leak.append(float(torch.tensor(
            [abs(coeff_of(base_row + off, int(o)) - coeff_of(base_row, int(o)))
             for o in others]).mean()))
    for ai, al in enumerate(ALPHAS):
        g2_pred.append((al - 1.0) * w)
        g2_meas.append(coeff_of(base_row + 3 + ai, gidx) - c0)

def fit(pred, meas):
    P, M = torch.tensor(pred), torch.tensor(meas)
    r = ((P - P.mean()) @ (M - M.mean()) /
         ((P - P.mean()).norm() * (M - M.mean()).norm()).clamp_min(1e-9)).item()
    slope = ((P @ M) / (P @ P)).item()
    return {"n": len(P), "r": r, "slope": slope}

res = {"g1_injection": fit(g1_pred, g1_meas),
       "g1_mean_leakage": float(torch.tensor(g1_leak).mean()),
       "g1_mean_injected": float(torch.tensor(g1_pred).abs().mean()),
       "g2_source_intervention": fit(g2_pred, g2_meas)}
print(json.dumps(res, indent=1), flush=True)
with open(f"{OUT}/exp_g_causal.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE_G", flush=True)
