"""Post-review 4 (2026-11-21): SAE feature directions decomposed over the weight dictionary.

Two reviewer points meet here. Reviewer 7MWE asked for a setting where the vector cannot be hooked: a learned SAE
feature direction is one, since no forward pass produced it. Reviewer MgDZ asked how the dictionary relates to SAE
features. Every decoder row of the public residual SAE for GPT-2 small at blocks.7.hook_resid_pre (the state after
block 6, the paper's layer) is decomposed by OMP over the paper's layer-6 dictionary and over its Gram-preserving
rotation, with k = 1, 2, 4, 8, 16 atoms. Controls under the same dictionary: random unit directions, unit directions
drawn from the typical states' covariance, and centred typical states. The SAE was trained in TransformerLens
coordinates, where every vector is centred over the model dimension, so atoms and targets are centred the same way
and renormalised (as in the follow-up experiments e494 and e497 this reproduces). Reported: the median unexplained
fraction by k and the type of the first atom chosen. Writes results/pr4_sae_features.json. Arguments: [--smoke]."""
import json, os, sys, time
_HF = "/data/mechinterp/hf"
if os.path.isdir(_HF): os.environ.setdefault("HF_HOME", _HF)
OUT_DIR = os.environ.get("PR_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")); os.makedirs(OUT_DIR, exist_ok=True)
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
SMOKE = "--smoke" in sys.argv; t0 = time.time(); torch.manual_seed(0); torch.set_grad_enabled(False); DEV = "cuda:0"
CTX, L, BATCH = 512, 6, 1024; KS = [1, 2, 4, 8, 16]; N_EVAL, N_CTRL, N_FEAT = (2, 256, 512) if SMOKE else (16, 4096, None)
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(DEV).eval(); tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
cfg = model.config; layers = model.transformer.h; D, NH = cfg.n_embd, cfg.n_head; HD = D // NH
ids = tok("\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip()), return_tensors="pt").input_ids[0][:N_EVAL * CTX].view(N_EVAL, CTX)
X = torch.cat([model(ids[i:i + 4].to(DEV), output_hidden_states=True).hidden_states[L + 1].float() for i in range(0, N_EVAL, 4)]).view(-1, D)
atoms = [model.transformer.wte.weight.detach().float(), model.transformer.wpe.weight.detach().float()]; DFF = layers[0].mlp.c_proj.weight.shape[0]
for b in range(L + 1):
    atoms.append(layers[b].mlp.c_proj.weight.detach().float()); Wo = layers[b].attn.c_proj.weight.detach().float()
    atoms += [torch.linalg.svd(Wo[h * HD:(h + 1) * HD], full_matrices=False).Vh for h in range(NH)]
    atoms.append(torch.stack([layers[b].attn.c_proj.bias, layers[b].mlp.c_proj.bias]).detach().float())
A = torch.cat(atoms); NA = A.shape[0]; NTOK, NPOS = atoms[0].shape[0], atoms[1].shape[0]; OFF = NTOK + NPOS; STRIDE = DFF + NH * HD + 2
del model; torch.cuda.empty_cache()
dimc = lambda v: v - v.mean(-1, keepdim=True); unit = lambda v: v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
g = torch.Generator(device=DEV).manual_seed(7); Q = torch.linalg.qr(torch.randn(D, D, device=DEV, generator=g))[0]
Au = unit(dimc(A)); Ar = unit(dimc(A @ Q)); del A
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
name = f"blocks.{L + 1}.hook_resid_pre"
def fetch(f):
    try: return hf_hub_download("jbloom/GPT2-Small-SAEs-Reformatted", f"{name}/{f}", local_files_only=True)
    except Exception: return hf_hub_download("jbloom/GPT2-Small-SAEs-Reformatted", f"{name}/{f}")
w = load_file(fetch("sae_weights.safetensors")); Wd = w["W_dec"].float().to(DEV); Wd = Wd if Wd.shape[1] == D else Wd.T
F = unit(dimc(Wd)); F = F if N_FEAT is None else F[:N_FEAT]
Xs = dimc(X); Xs = Xs - Xs.mean(0); xn = Xs.norm(dim=1); typ = xn <= 10 * xn.median(); Xt = Xs[typ]
Lc = torch.linalg.cholesky(torch.cov(Xt.T.double(), correction=0) + 1e-6 * torch.eye(D, device=DEV, dtype=torch.float64)).float()
g2 = torch.Generator(device=DEV).manual_seed(11)
targets = {"sae_features": F, "random_directions": unit(dimc(torch.randn(N_CTRL, D, device=DEV, generator=g2))), "covariance_directions": unit(dimc(torch.randn(N_CTRL, D, device=DEV, generator=g2) @ Lc.T)),
           "typical_states": unit(Xt[torch.randperm(Xt.shape[0], device=DEV, generator=g2)[:N_CTRL]])}
print(f"{F.shape[0]} features, {NA} atoms, {Xt.shape[0]} typical states for the covariance ({time.time() - t0:.0f}s)", flush=True)
def omp(T, Dct):
    N = T.shape[0]; K = max(KS); err = {k: torch.zeros(N, device=DEV) for k in KS}; first = torch.zeros(N, dtype=torch.long, device=DEV); eye = torch.eye(K, device=DEV)
    for s in range(0, N, BATCH):
        x = T[s:s + BATCH]; n = x.shape[0]; r = x.clone(); S = torch.zeros(n, 0, dtype=torch.long, device=DEV); taken = torch.zeros(n, Dct.shape[0], dtype=torch.bool, device=DEV)
        for step in range(K):
            pick = (r @ Dct.T).abs_().masked_fill_(taken, -1.0).argmax(-1, keepdim=True); taken.scatter_(1, pick, True); S = torch.cat([S, pick], 1)
            As = Dct[S]; G = As @ As.transpose(1, 2) + 1e-5 * eye[:step + 1, :step + 1]
            c = torch.cholesky_solve(As @ x[:, :, None], torch.linalg.cholesky(G)); r = x - (c.transpose(1, 2) @ As)[:, 0]
            if step + 1 in err: err[step + 1][s:s + n] = (r ** 2).sum(-1) / (x ** 2).sum(-1)
        first[s:s + n] = S[:, 0]; del taken
    return err, first
def types(first):
    rel = (first - OFF) % STRIDE; blk = first >= OFF; n = first.numel()
    out = dict(token=float((first < NTOK).sum() / n), position=float(((first >= NTOK) & (first < OFF)).sum() / n), mlp=float((blk & (rel < DFF)).sum() / n),
               attention=float((blk & (rel >= DFF) & (rel < DFF + NH * HD)).sum() / n), bias=float((blk & (rel >= DFF + NH * HD)).sum() / n))
    m = blk & (rel < DFF); out["mlp_block_hist"] = torch.bincount(torch.div(first[m] - OFF, STRIDE, rounding_mode="floor"), minlength=L + 1).tolist(); return out
R = dict(sae=name, n_features=int(F.shape[0]), atoms=int(NA), smoke=SMOKE, targets={}); outp = os.path.join(OUT_DIR, f"pr4_sae_features{'_smoke' if SMOKE else ''}.json")
for tname, T in targets.items():
    for dname, Dct in (("weight", Au), ("rotated", Ar)):
        if dname == "rotated" and tname not in ("sae_features", "random_directions"): continue
        err, first = omp(T, Dct)
        R["targets"].setdefault(tname, {})[dname] = dict(n=int(T.shape[0]), unexplained_median={k: float(v.median()) for k, v in err.items()}, unexplained_mean={k: float(v.mean()) for k, v in err.items()}, first_atom=types(first))
        json.dump(R, open(outp, "w"), indent=1)
        e = R["targets"][tname][dname]; print(f"  {tname} / {dname}: unexplained by k " + ", ".join(f"{k}: {e['unexplained_median'][k]:.3f}" for k in KS) + f" | first atom: MLP {e['first_atom']['mlp']:.3f}, token {e['first_atom']['token']:.3f}, attention {e['first_atom']['attention']:.3f}, position {e['first_atom']['position']:.3f} ({time.time() - t0:.0f}s)", flush=True)
print("DONE_PR4", f"{time.time() - t0:.0f}s", flush=True)
