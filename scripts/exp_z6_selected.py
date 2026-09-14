"""Z6: what the support contains, and how well every selected MLP atom is estimated.

The attribution numbers of the sweep score each token's three dominant ledger writes. This script
scores the other direction: every atom that OMP places in the k = 64 support. For each selected MLP
atom the ledger gives the writer's true coefficient at that token, so we can ask how often a selected
atom is a real write (|c| at least 5% of the token's largest write), whether its sign is right, and
how well its magnitude is recovered; the same is reported for the selected atoms that are among the
token's top-3 writes. A per-token least-squares refit on the three true atoms is also recorded so the
oracle of the main text can be compared with OMP on the same identified cases. Same protocol as
exp_z2_ci (states, ledger, dictionary, centering); bootstrap intervals over sequences.
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

from exp_z2_ci import MODELS, CTX, N_EVAL, K, BATCH, DEV, omp
from wdd_z_common import build, atom_type

torch.set_grad_enabled(False)
OUT = os.path.join(RESULTS_DIR, "exp_z6_selected.json")
ONLY = [m for m in os.environ.get("WDD_MODELS", "").split(",") if m]
CORPORA = [c for c in os.environ.get("WDD_CORPORA", "wikitext,pile").split(",") if c]
REAL = 0.05          # a selected MLP atom is a real write if |c| >= REAL x the token's largest |c|
B_BOOT = 1000


def boot(seq, n_seq, rng):
    counts = np.stack([np.bincount(rng.integers(0, n_seq, n_seq), minlength=n_seq) for _ in range(B_BOOT)])
    def mean(x, s):
        ns = np.bincount(s, weights=x.astype(np.float64), minlength=n_seq); ds = np.bincount(s, minlength=n_seq).astype(np.float64)
        b = (counts @ ns) / np.maximum(counts @ ds, 1e-9)
        return {"point": float(x.mean()), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5))}
    return mean


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 else None


def run(name, corpus):
    t0 = time.time()
    b = build(name, corpus, n_eval=N_EVAL)
    X, mu, acts, WN, A = b["X"], b["mu"], b["acts"], b["WN"], b["A"]
    L, DFF, OFF, STRIDE, NH, HD, fam = b["L"], b["DFF"], b["OFF"], b["STRIDE"], b["NH"], b["HD"], b["fam"]
    NT = X.shape[0]
    nrm = X.norm(dim=-1); typ = (nrm <= 10 * nrm.median()).cpu().numpy()
    Xf = X - X.mean(0) if fam == "gpt2" else X - mu
    del X
    _, _, sel, cof = omp(Xf, A, K, want=True)
    sel_c, cof_c = sel.cpu().numpy(), cof.cpu().numpy()
    types = atom_type(sel.reshape(-1), OFF, STRIDE, DFF, NH, HD).cpu().numpy().reshape(NT, K)
    print(f"  support done ({time.time() - t0:.0f}s)", flush=True)
    # ---- the ledger for every selected MLP atom ----
    lmax = np.zeros(NT); true_sel = np.zeros((NT, K)); is_mlp = types == 1
    blk = (sel_c - OFF) // STRIDE; nrn = (sel_c - OFF) % STRIDE
    for bb in range(L + 1):
        led = (acts[bb].float() * WN[bb]).numpy()                                   # [NT, DFF]
        lmax = np.maximum(lmax, np.abs(led).max(1))
        m = is_mlp & (blk == bb)
        tok, pos = np.nonzero(m)
        true_sel[tok, pos] = led[tok, nrn[tok, pos]]
    # ---- the top-3 ledger writes, and a least-squares refit on their true atoms ----
    top3 = torch.cat([torch.cat([acts[bb][s:s + 4096].float() * WN[bb] for bb in range(L + 1)], dim=1).abs().topk(3, dim=1).indices for s in range(0, NT, 4096)])
    true3 = torch.cat([torch.cat([acts[bb][s:s + 4096].float() * WN[bb] for bb in range(L + 1)], dim=1).gather(1, top3[s:s + 4096]) for s in range(0, NT, 4096)]).numpy()
    atom3 = (OFF + (top3 // DFF) * STRIDE + (top3 % DFF)).to(DEV)
    refit = np.zeros((NT, 3))
    for s in range(0, NT, 4096):
        A3 = A[atom3[s:s + 4096]]                                                    # [n, 3, D]
        G = A3 @ A3.transpose(1, 2) + 1e-5 * torch.eye(3, device=DEV)
        c = torch.linalg.solve(G, A3 @ Xf[s:s + 4096, :, None])[:, :, 0]
        refit[s:s + 4096] = c.cpu().numpy()
    hit3 = (sel[:, None, :] == atom3[:, :, None]).any(-1).cpu().numpy()             # [NT, 3]
    pos3 = (sel[:, None, :] == atom3[:, :, None]).float().argmax(-1).cpu().numpy()
    pred3 = np.take_along_axis(cof_c, pos3, 1) * hit3
    del acts, A, Xf, sel, cof; torch.cuda.empty_cache()
    # ---- statistics ----
    seq = np.repeat(np.arange(N_EVAL), CTX); rng = np.random.default_rng(2026); mean = boot(seq, N_EVAL, rng)
    T = typ[:, None] & np.ones((NT, K), bool)
    comp = {n: float((types[typ] == i).mean()) for i, n in enumerate(["embedding", "mlp", "attention", "bias"])}
    sm = is_mlp & T; tok_s = np.nonzero(sm)[0]
    tr = true_sel[sm]; pr = cof_c[sm]; big = np.abs(tr) >= REAL * lmax[tok_s]
    res = {"n_states": int(NT), "support_composition_typical": comp, "selected_mlp_atoms": int(sm.sum()),
           "real_share": mean(big.astype(np.float64), seq[tok_s]),
           "sign_agreement_real": mean((np.sign(pr[big]) == np.sign(tr[big])).astype(np.float64), seq[tok_s][big]),
           "median_rel_error_real": float(np.median(np.abs(pr[big] - tr[big]) / np.abs(tr[big]))),
           "median_ratio_real": float(np.median(pr[big] / tr[big])),
           "median_abs_pred_spurious": float(np.median(np.abs(pr[~big]))) if (~big).any() else None,
           "median_abs_pred_real": float(np.median(np.abs(pr[big]))),
           "spurious_share_of_mlp_coefficient_mass": float(np.abs(pr[~big]).sum() / np.abs(pr).sum())}
    # dominant writes: OMP against the refit on the same identified cases (typical states)
    h = hit3 & typ[:, None]; hh = h.reshape(-1); allt = typ[:, None] & np.ones((NT, 3), bool); at = allt.reshape(-1)
    t3 = true3.reshape(-1); p3 = pred3.reshape(-1); r3 = refit.reshape(-1)
    res["dominant"] = {"recall_top3": float(h.sum() / allt.sum()),
                       "r_omp_identified": corr(p3[hh], t3[hh]), "r_refit_identified": corr(r3[hh], t3[hh]), "r_refit_all": corr(r3[at], t3[at]),
                       "median_rel_error_omp_identified": float(np.median(np.abs(p3[hh] - t3[hh]) / np.abs(t3[hh]))),
                       "median_rel_error_refit_identified": float(np.median(np.abs(r3[hh] - t3[hh]) / np.abs(t3[hh]))),
                       "median_rel_error_refit_all": float(np.median(np.abs(r3[at] - t3[at]) / np.abs(t3[at])))}
    res["seconds"] = round(time.time() - t0)
    print(f"  {name}/{corpus}: support {comp}  real {res['real_share']['point']:.3f}  sign {res['sign_agreement_real']['point']:.3f}  relerr {res['median_rel_error_real']:.2f}  spurious mass {res['spurious_share_of_mlp_coefficient_mass']:.3f} | dominant: r omp {res['dominant']['r_omp_identified']:.3f} refit {res['dominant']['r_refit_identified']:.3f} relerr omp {res['dominant']['median_rel_error_omp_identified']:.2f} refit {res['dominant']['median_rel_error_refit_identified']:.2f}", flush=True)
    return res


if __name__ == "__main__":
    RES = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in MODELS:
        if ONLY and name not in ONLY: continue
        for corpus in CORPORA:
            key = f"{name}/{corpus}"
            if key in RES: print(f"== {key} done, skip ==", flush=True); continue
            print(f"== {key} ==", flush=True)
            RES[key] = run(name, corpus)
            json.dump(RES, open(OUT, "w"), indent=1)
    print("DONE_Z6", flush=True)
