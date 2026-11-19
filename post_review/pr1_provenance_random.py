"""Post-review 1 (2026-11-19): the size-matched random baseline that Reviewer 7MWE asked for.

Section 4.1 compares a dictionary that could have written GPT-2's layer-6 state (blocks 0 to 6)
with one that could not (blocks 7 to 11) and with a Gram-preserving rotation. 7MWE pointed out
that the random-direction baseline had about four times as many atoms as the embedding-free
control, and asked for a size-matched random dictionary next to each provenance control, at
every k. This script reruns exp_d and exp_d2's dictionaries (as a check that the protocol is the
paper's) and adds: random unit directions matched in size to each control (three seeds for the
embedding-free control), each control's own Gram-preserving rotation (same internal geometry, no
alignment with the model's coordinates), and the embedding rows alone. Same states, centering,
OMP, k values and typical-state cut as exp_d2. Writes post_review/results/pr1_provenance_random.json
after each dictionary. Arguments: [--smoke]."""
import json, os, sys, time
_HF = "/data/mechinterp/hf"
if os.path.isdir(_HF): os.environ.setdefault("HF_HOME", _HF)
OUT_DIR = os.environ.get("PR_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")); os.makedirs(OUT_DIR, exist_ok=True)
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
SMOKE = "--smoke" in sys.argv; t0 = time.time()
torch.manual_seed(0); torch.set_grad_enabled(False); DEV = "cuda:0"
CTX, L = 512, 6
N_TRAIN, N_EVAL, K_MAX, K_CHECK = (8, 2, 8, [4, 8]) if SMOKE else (400, 64, 64, [4, 8, 16, 32, 64])
BATCH = 1024
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(DEV).eval(); tok = AutoTokenizer.from_pretrained("openai-community/gpt2"); D = model.config.n_embd
def chunks(split, n):
    text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"] if t.strip())
    return tok(text, return_tensors="pt").input_ids[0][:n * CTX].view(n, CTX)
def resid(ids, batch=8):
    outs = []
    for i in range(0, len(ids), batch): outs.append(model(ids[i:i + batch].to(DEV), output_hidden_states=True).hidden_states[L + 1].float().cpu())
    return torch.cat(outs).view(-1, D)
MU = resid(chunks("train", N_TRAIN)).mean(0); X = resid(chunks("test", N_EVAL)) - MU
def unit(a): return a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
def component_atoms(blocks):
    atoms = []
    for b in blocks:
        blk = model.transformer.h[b]; atoms.append(blk.mlp.c_proj.weight.detach().float().cpu()); w_o = blk.attn.c_proj.weight
        for h in range(12): atoms.append(torch.linalg.svd(w_o[h * 64:(h + 1) * 64], full_matrices=False).Vh.detach().float().cpu())
        atoms.append(torch.stack([blk.attn.c_proj.bias, blk.mlp.c_proj.bias]).detach().float().cpu())
    return torch.cat(atoms)
EMB = torch.cat([model.transformer.wte.weight.detach().float().cpu(), model.transformer.wpe.weight.detach().float().cpu()])
past_raw = component_atoms(range(L + 1)); fut_raw = component_atoms(range(L + 1, 12)); n_fut = fut_raw.shape[0]
# embedding-free family, generator order of exp_d2 (subsample first, then the rotation)
g2 = torch.Generator().manual_seed(7); past = unit(past_raw); sub2 = past[torch.randperm(past.shape[0], generator=g2)[:n_fut]]; Q2 = torch.linalg.qr(torch.randn(D, D, generator=g2))[0]; fut = unit(fut_raw)
# with-embedding family, generator order of exp_d (rotation first, then the subsample)
g1 = torch.Generator().manual_seed(7); Q1 = torch.linalg.qr(torch.randn(D, D, generator=g1))[0]; sub1 = past_raw[torch.randperm(past_raw.shape[0], generator=g1)[:n_fut]]
CORRECT_MATCHED = unit(torch.cat([EMB, sub1])); FUTURE = unit(torch.cat([EMB, fut_raw]))
def rand_dict(n, seed): return unit(torch.randn(n, D, generator=torch.Generator().manual_seed(seed)))
DICTS = [("noemb_correct_subsampled", lambda: sub2), ("noemb_future_blocks", lambda: fut),
         ("noemb_random_s0", lambda: rand_dict(n_fut, 100)), ("noemb_random_s1", lambda: rand_dict(n_fut, 101)), ("noemb_random_s2", lambda: rand_dict(n_fut, 102)),
         ("noemb_future_blocks_rotated", lambda: fut @ Q2), ("noemb_correct_subsampled_rotated", lambda: sub2 @ Q2),
         ("emb_correct_matched", lambda: CORRECT_MATCHED), ("emb_future_blocks", lambda: FUTURE), ("emb_random", lambda: rand_dict(FUTURE.shape[0], 200)),
         ("emb_only", lambda: unit(EMB)), ("emb_future_blocks_rotated", lambda: FUTURE @ Q1)]
if SMOKE: DICTS = DICTS[:3]
del model; torch.cuda.empty_cache()
print(f"states {X.shape[0]}, n_fut {n_fut}, embedding rows {EMB.shape[0]} ({time.time() - t0:.0f}s)", flush=True)
def omp_tokens(A):
    A_dev = A.to(DEV); Xd = X.to(DEV); N, NA = Xd.shape[0], A_dev.shape[0]
    err = {k: torch.zeros(N, device=DEV) for k in K_CHECK}; eye = torch.eye(K_MAX, device=DEV)
    for s in range(0, N, BATCH):
        x = Xd[s:s + BATCH]; n = x.shape[0]; r = x.clone(); S = torch.zeros(n, 0, dtype=torch.long, device=DEV); taken = torch.zeros(n, NA, dtype=torch.bool, device=DEV)
        for step in range(K_MAX):
            pick = (r @ A_dev.T).abs_().masked_fill_(taken, -1.0).argmax(-1, keepdim=True); taken.scatter_(1, pick, True); S = torch.cat([S, pick], 1)
            As = A_dev[S]; G = As @ As.transpose(1, 2) + 1e-5 * eye[:step + 1, :step + 1]
            c = torch.cholesky_solve(As @ x[:, :, None], torch.linalg.cholesky(G)); r = x - (c.transpose(1, 2) @ As)[:, 0]
            if step + 1 in err: err[step + 1][s:s + n] = (r ** 2).sum(-1)
        del taken
    del A_dev; torch.cuda.empty_cache()
    return {k: v.cpu().numpy().astype(np.float64) for k, v in err.items()}
x2 = (X ** 2).sum(-1).numpy().astype(np.float64); nrm = np.sqrt(x2); typ = nrm <= 10 * np.median(nrm)
R = {"n_states": int(X.shape[0]), "n_typical": int(typ.sum()), "smoke": SMOKE, "dicts": {}}; outp = os.path.join(OUT_DIR, f"pr1_provenance_random{'_smoke' if SMOKE else ''}.json")
for name, make in DICTS:
    A = make(); e = omp_tokens(A)
    R["dicts"][name] = {"atoms": int(A.shape[0]), "fvu_all": {k: float(v.sum() / x2.sum()) for k, v in e.items()}, "fvu_typical": {k: float(v[typ].sum() / x2[typ].sum()) for k, v in e.items()}, "fvu_sink": {k: float(v[~typ].sum() / max(x2[~typ].sum(), 1e-12)) for k, v in e.items()}}
    json.dump(R, open(outp, "w"), indent=1)
    print(f"{name} ({A.shape[0]} atoms): typical " + "  ".join(f"k{k}={R['dicts'][name]['fvu_typical'][k]:.3f}" for k in K_CHECK) + f" | sinks k{K_CHECK[-1]}={R['dicts'][name]['fvu_sink'][K_CHECK[-1]]:.4f} ({time.time() - t0:.0f}s)", flush=True)
print("DONE_PR1", f"{time.time() - t0:.0f}s", flush=True)
