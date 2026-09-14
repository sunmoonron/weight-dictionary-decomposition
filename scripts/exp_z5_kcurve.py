"""Z5: reconstruction against sparsity for every model, at matched budgets.

The k=32 and k=64 rows of Table 1 hold k fixed while the width d grows fivefold
from GPT-2 to the 7B models, so a rising rotated (or random) FVU is what a
fixed budget does in a larger space and says nothing by itself. This script
sweeps k for the weight dictionary and its Gram-preserving rotation on 8,192
WikiText-2 states per model (the first 16 evaluation sequences of the Z2
protocol, same dictionary, same centering, same rotation seed), reading the
FVU at k = 4, 8, ..., 256 and at the matched budgets k = d/16 and k = d/12,
with bootstrap intervals over the 16 sequences.
"""
import json
import os
import time

_HF = "/data/mechinterp/hf"
if os.path.isdir(_HF):
    os.environ.setdefault("HF_HOME", _HF)
RESULTS_DIR = os.environ.get("WDD_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

import numpy as np
import torch

from exp_z2_ci import MODELS, CTX, BATCH, DEV
from wdd_z_common import build

torch.set_grad_enabled(False)
OUT = os.path.join(RESULTS_DIR, "exp_z5_kcurve.json")
TOK_DIR = os.path.join(RESULTS_DIR, "exp_z5_tokens"); os.makedirs(TOK_DIR, exist_ok=True)
ONLY = [m for m in os.environ.get("WDD_MODELS", "").split(",") if m]
N_EVAL = int(os.environ.get("WDD_N_EVAL", 16))
BASE = [4, 8, 16, 32, 64, 128, 256]
B_BOOT = 1000


def omp_curve(X, A, ckpts):
    """OMP to max(ckpts) atoms, returning the per-token squared residual at each checkpoint."""
    N, NA = X.shape[0], A.shape[0]; kmax = max(ckpts)
    err = {k: torch.zeros(N, device=DEV) for k in ckpts}
    eye = torch.eye(kmax, device=DEV)
    for s in range(0, N, BATCH):
        x = X[s:s + BATCH]; n = x.shape[0]
        r = x.clone(); S = torch.zeros(n, 0, dtype=torch.long, device=DEV)
        taken = torch.zeros(n, NA, dtype=torch.bool, device=DEV)
        for step in range(kmax):
            pick = (r @ A.T).abs_().masked_fill_(taken, -1.0).argmax(-1, keepdim=True)
            taken.scatter_(1, pick, True)
            S = torch.cat([S, pick], 1)
            As = A[S]
            G = As @ As.transpose(1, 2) + 1e-5 * eye[:step + 1, :step + 1]
            c = torch.cholesky_solve(As @ x[:, :, None], torch.linalg.cholesky(G))
            r = x - (c.transpose(1, 2) @ As)[:, 0]
            if step + 1 in err: err[step + 1][s:s + n] = (r ** 2).sum(-1)
        del taken, As, G
    return err


def run(name):
    t0 = time.time()
    b = build(name, "wikitext", n_eval=N_EVAL)
    X, mu, A, D, NA = b["X"], b["mu"], b["A"], b["D"], b["NA"]
    del b
    Xc = X - mu; del X
    xnorm2 = (Xc ** 2).sum(-1).cpu().numpy().astype(np.float64)
    ck = sorted(set(BASE + [D // 16, D // 12]))
    curves = {}
    err = omp_curve(Xc, A, ck); curves["weight"] = {k: v.cpu().numpy() for k, v in err.items()}; del err
    print(f"  weight curve done ({time.time() - t0:.0f}s): FVU@32 {curves['weight'][32].sum() / xnorm2.sum():.4f}  @{ck[-1]} {curves['weight'][ck[-1]].sum() / xnorm2.sum():.4f}", flush=True)
    g = torch.Generator(device=DEV).manual_seed(7)
    Q = torch.linalg.qr(torch.randn(D, D, device=DEV, generator=g))[0]
    AQ = torch.cat([A[i:i + 65536] @ Q for i in range(0, NA, 65536)]); del A; torch.cuda.empty_cache()
    err = omp_curve(Xc, AQ, ck); curves["rotated"] = {k: v.cpu().numpy() for k, v in err.items()}; del err, AQ
    print(f"  rotated curve done ({time.time() - t0:.0f}s): FVU@32 {curves['rotated'][32].sum() / xnorm2.sum():.4f}  @{ck[-1]} {curves['rotated'][ck[-1]].sum() / xnorm2.sum():.4f}", flush=True)
    seq = np.repeat(np.arange(N_EVAL), CTX)
    m1 = (np.arange(len(seq)) % CTX) >= 1                      # the first state of a sequence is the attention sink; see exp_z2_recompute
    np.savez_compressed(os.path.join(TOK_DIR, f"{name}_wikitext.npz"), seq=seq, xnorm2=xnorm2, **{f"err{k}_{d}": v[k].astype(np.float32) for d, v in curves.items() for k in ck})
    rng = np.random.default_rng(2026)
    counts = np.stack([np.bincount(rng.integers(0, N_EVAL, N_EVAL), minlength=N_EVAL) for _ in range(B_BOOT)])
    res = {"d": int(D), "atoms": int(NA), "n_states": int(len(seq)), "checkpoints": ck, "matched": {"d/16": D // 16, "d/12": D // 12},
           "variance_share_position_0": float(xnorm2[~m1].sum() / xnorm2.sum()), "fvu": {}, "fvu_pos1": {}}
    for cut, mask in (("fvu", np.ones_like(m1)), ("fvu_pos1", m1)):
        den = np.bincount(seq, weights=xnorm2 * mask, minlength=N_EVAL)
        for dname, cv in curves.items():
            res[cut][dname] = {}
            for k in ck:
                e = cv[k].astype(np.float64) * mask
                num = np.bincount(seq, weights=e, minlength=N_EVAL)
                boot = (counts @ num) / (counts @ den)
                res[cut][dname][str(k)] = [float(e.sum() / (xnorm2 * mask).sum()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    res["seconds"] = round(time.time() - t0)
    del Xc; torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    RES = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in MODELS:
        if ONLY and name not in ONLY: continue
        if name in RES: print(f"== {name} done, skip ==", flush=True); continue
        print(f"== {name} ==", flush=True)
        RES[name] = run(name)
        json.dump(RES, open(OUT, "w"), indent=1)
    print("DONE_Z5", flush=True)
