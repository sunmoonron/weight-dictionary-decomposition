"""Experiment D: does computational PROVENANCE matter, or just overcompleteness?

Reviewer's point: low FVU alone can be dismissed as "a huge overcomplete
dictionary approximates anything". The decisive control is a dictionary with
identical size, identical norms, and *identical internal geometry* (the full
Gram matrix is preserved) but destroyed alignment: apply one fixed random
orthogonal rotation Q to every WDD atom. If reconstruction collapses, the win
was alignment with the model's activation space, not generic overcompleteness.

Second axis: atoms from the WRONG blocks. Keep wte+wpe, swap the block-0..6
component atoms for block-7..11 ones (atoms the stream at layer 6 was never
built from), size-matched by subsampling the correct side to the same count.
If future-block atoms work nearly as well, the dictionary's value is about the
shared residual coordinate system; if they fail, it is about the specific
writes. Either answer is informative and reported as found.
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

CTX, N_TRAIN, N_EVAL, L, K_MAX = 512, 400, 64, 6, 64
K_CHECK = [4, 8, 16, 32, 64]

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
D = model.config.n_embd


def chunks(split, n):
    text = "\n\n".join(t for t in load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"] if t.strip())
    return tok(text, return_tensors="pt").input_ids[0][:n * CTX].view(n, CTX)


def resid6(ids, batch=8):
    outs = []
    for i in range(0, len(ids), batch):
        outs.append(model(ids[i:i + batch].to(DEV), output_hidden_states=True)
                    .hidden_states[L + 1].float().cpu())
    return torch.cat(outs).view(-1, D)


MU = resid6(chunks("train", N_TRAIN)).mean(0)
X = resid6(chunks("test", N_EVAL)) - MU
DENOM = (X ** 2).sum().item()
print(f"eval states {X.shape[0]}", flush=True)


def unit(a):
    return a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def component_atoms(blocks):
    atoms = []
    for b in blocks:
        blk = model.transformer.h[b]
        atoms.append(blk.mlp.c_proj.weight.detach().float().cpu())
        w_o = blk.attn.c_proj.weight
        for h in range(12):
            atoms.append(torch.linalg.svd(
                w_o[h * 64:(h + 1) * 64], full_matrices=False).Vh.detach().float().cpu())
        atoms.append(torch.stack([blk.attn.c_proj.bias, blk.mlp.c_proj.bias])
                     .detach().float().cpu())
    return torch.cat(atoms)


EMB = torch.cat([model.transformer.wte.weight.detach().float().cpu(),
                 model.transformer.wpe.weight.detach().float().cpu()])
CORRECT = unit(torch.cat([EMB, component_atoms(range(L + 1))]))

g = torch.Generator().manual_seed(7)
Q = torch.linalg.qr(torch.randn(D, D, generator=g))[0]        # fixed orthogonal
ROTATED = CORRECT @ Q                                          # Gram-preserving

past = component_atoms(range(L + 1))
fut = component_atoms(range(L + 1, 12))                        # blocks 7..11
n_fut = fut.shape[0]
sub = past[torch.randperm(past.shape[0], generator=g)[:n_fut]]
CORRECT_MATCHED = unit(torch.cat([EMB, sub]))
FUTURE = unit(torch.cat([EMB, fut]))
print(f"dicts: full {CORRECT.shape[0]}, matched {CORRECT_MATCHED.shape[0]}, "
      f"future {FUTURE.shape[0]}", flush=True)


TOKENS = {}   # per-token squared residuals at every checkpoint, for any state-level cut (see exp_z2_recompute)


def omp_fvu(A, name=None):
    A_dev = A.to(DEV)
    err = {k: 0.0 for k in K_CHECK}
    tok = {k: [] for k in K_CHECK}
    eye = torch.eye(K_MAX, device=DEV)
    for s in range(0, len(X), 768):
        x = X[s:s + 768].to(DEV)
        B = len(x)
        sel = torch.zeros(B, K_MAX, dtype=torch.long, device=DEV)
        taken = torch.zeros(B, A_dev.shape[0], dtype=torch.bool, device=DEV)
        r = x.clone()
        for k in range(K_MAX):
            pick = (r @ A_dev.T).abs_().masked_fill_(taken, -1.0).argmax(-1)
            sel[:, k] = pick
            taken.scatter_(1, pick[:, None], True)
            A_S = A_dev[sel[:, :k + 1]]
            G = A_S @ A_S.transpose(1, 2) + 1e-5 * eye[:k + 1, :k + 1]
            c = torch.cholesky_solve(A_S @ x[:, :, None], torch.linalg.cholesky(G))
            r = x - (c.transpose(1, 2) @ A_S)[:, 0]
            if k + 1 in K_CHECK:
                e_tok = (r ** 2).sum(-1)
                err[k + 1] += e_tok.sum().item(); tok[k + 1].append(e_tok.cpu())
    del A_dev
    torch.cuda.empty_cache()
    if name is not None: TOKENS[name] = {k: torch.cat(v).numpy() for k, v in tok.items()}
    return {k: e / DENOM for k, e in err.items()}


results = {}
for name, A in [("wdd_rotated", ROTATED), ("wdd_matched_correct", CORRECT_MATCHED),
                ("wdd_future_blocks", FUTURE)]:
    fvu = omp_fvu(A, name)
    results[name] = {"atoms": A.shape[0], "fvu": fvu}
    print(name + ": " + "  ".join(f"k{k}={fvu[k]:.3f}" for k in K_CHECK), flush=True)

with open(f"{OUT}/exp_d_provenance.json", "w") as f:
    json.dump(results, f, indent=1)
import numpy as np
np.savez_compressed(f"{OUT}/exp_d_tokens.npz", xnorm2=(X ** 2).sum(-1).numpy(), **{f"{n}_k{k}": v for n, d in TOKENS.items() for k, v in d.items()})
print("DONE_D", flush=True)
