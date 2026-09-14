"""D2: the provenance null without shared embeddings.

exp_d_provenance keeps the token and position embedding rows in both the correct and the
future-block dictionary, and those rows can contribute to a layer-6 state. This control drops them:
the correct block atoms of blocks 0 to 6, the block atoms of blocks 7 to 11 (which cannot have
written to the state), the correct atoms subsampled to that size, and a Gram-preserving rotation of
the correct set, all without embeddings. Per-token residuals at k = 4 to 64 are saved so the
typical-state cut of exp_z2_recompute applies. GPT-2 small, the states of exp_d_provenance."""
import json
import os

_HF = "/data/mechinterp/hf"
if os.path.isdir(_HF):
    os.environ.setdefault("HF_HOME", _HF)
RESULTS_DIR = os.environ.get("WDD_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(0)
torch.set_grad_enabled(False)
DEV = "cuda:0"
CTX, N_TRAIN, N_EVAL, L, K_MAX = 512, 400, 64, 6, 64
K_CHECK = [4, 8, 16, 32, 64]
BATCH = 1024

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
D = model.config.n_embd


def chunks(split, n):
    text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"] if t.strip())
    return tok(text, return_tensors="pt").input_ids[0][:n * CTX].view(n, CTX)


def resid6(ids, batch=8):
    outs = []
    for i in range(0, len(ids), batch):
        outs.append(model(ids[i:i + batch].to(DEV), output_hidden_states=True).hidden_states[L + 1].float().cpu())
    return torch.cat(outs).view(-1, D)


MU = resid6(chunks("train", N_TRAIN)).mean(0)
X = resid6(chunks("test", N_EVAL)) - MU
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
            atoms.append(torch.linalg.svd(w_o[h * 64:(h + 1) * 64], full_matrices=False).Vh.detach().float().cpu())
        atoms.append(torch.stack([blk.attn.c_proj.bias, blk.mlp.c_proj.bias]).detach().float().cpu())
    return torch.cat(atoms)


g = torch.Generator().manual_seed(7)
past = unit(component_atoms(range(L + 1)))
fut = unit(component_atoms(range(L + 1, 12)))
sub = past[torch.randperm(past.shape[0], generator=g)[:fut.shape[0]]]
Q = torch.linalg.qr(torch.randn(D, D, generator=g))[0]
DICTS = {"correct_blocks_noemb": past, "correct_subsampled_noemb": sub, "future_blocks_noemb": fut, "rotated_correct_noemb": past @ Q}
print({k: v.shape[0] for k, v in DICTS.items()}, flush=True)


def omp_tokens(A):
    A_dev = A.to(DEV); Xd = X.to(DEV); N, NA = Xd.shape[0], A_dev.shape[0]
    err = {k: torch.zeros(N, device=DEV) for k in K_CHECK}
    eye = torch.eye(K_MAX, device=DEV)
    for s in range(0, N, BATCH):
        x = Xd[s:s + BATCH]; n = x.shape[0]
        r = x.clone(); S = torch.zeros(n, 0, dtype=torch.long, device=DEV)
        taken = torch.zeros(n, NA, dtype=torch.bool, device=DEV)
        for step in range(K_MAX):
            pick = (r @ A_dev.T).abs_().masked_fill_(taken, -1.0).argmax(-1, keepdim=True)
            taken.scatter_(1, pick, True); S = torch.cat([S, pick], 1)
            As = A_dev[S]; G = As @ As.transpose(1, 2) + 1e-5 * eye[:step + 1, :step + 1]
            c = torch.cholesky_solve(As @ x[:, :, None], torch.linalg.cholesky(G))
            r = x - (c.transpose(1, 2) @ As)[:, 0]
            if step + 1 in err: err[step + 1][s:s + n] = (r ** 2).sum(-1)
        del taken
    del A_dev; torch.cuda.empty_cache()
    return {k: v.cpu().numpy() for k, v in err.items()}


x2 = (X ** 2).sum(-1).numpy().astype(np.float64); nrm = np.sqrt(x2); typ = nrm <= 10 * np.median(nrm)
tokens = {"xnorm2": x2}; results = {}
for name, A in DICTS.items():
    e = omp_tokens(A)
    for k, v in e.items(): tokens[f"{name}_k{k}"] = v.astype(np.float32)
    results[name] = {"atoms": int(A.shape[0]), "fvu_all": {k: float(v.sum() / x2.sum()) for k, v in e.items()}, "fvu_typical": {k: float(v[typ].sum() / x2[typ].sum()) for k, v in e.items()}}
    print(name + ":  all " + "  ".join(f"k{k}={results[name]['fvu_all'][k]:.3f}" for k in K_CHECK) + " | typical " + "  ".join(f"k{k}={results[name]['fvu_typical'][k]:.3f}" for k in K_CHECK), flush=True)
json.dump(results, open(os.path.join(RESULTS_DIR, "exp_d2_null_noemb.json"), "w"), indent=1)
np.savez_compressed(os.path.join(RESULTS_DIR, "exp_d2_tokens.npz"), **tokens)
print("DONE_D2", flush=True)
