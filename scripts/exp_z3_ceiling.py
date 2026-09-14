"""Z3: the observability ceiling in every model.

The GPT-2 analysis of exp_i_observability (identification of each token's three
dominant MLP writes binned by the survival of the write in the observed state,
and one-property AUCs) rerun under the Z2 protocol for all eight models on both
corpora: 32,768 states, the same dictionary and centering as exp_z2_ci, OMP at
k = 64 and one-shot ranking scored against the ledger. Two additions decompose
the misses: a twin check (is an atom within |cos| >= 0.95 of the true atom in
the support instead of the atom itself?) and the composition of the support by
atom type. Every rate carries a bootstrap interval over the evaluation
sequences. Per-case arrays go to results/exp_z3_cases/.

Definitions, from exp_i: survival s = (x . d) / c, prominence |x . d| / |x|,
magnitude |c|, relative magnitude |c| / |x|, age = dictionary layer minus the
source block, competition = max |cos| with the token's other top-8 write
directions. Bins on s: (-inf, 0), [0, 0.25), [0.25, 0.5), [0.5, 0.75),
[0.75, 1.25), [1.25, inf). Intact means s >= 0.75, erased means s < 0.25.
"""
import json
import os
import time

_HF = "/data/mechinterp/hf"
if os.path.isdir(_HF):
    os.environ.setdefault("HF_HOME", _HF)
RESULTS_DIR = os.environ.get("WDD_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)
CASE_DIR = os.path.join(RESULTS_DIR, "exp_z3_cases")
os.makedirs(CASE_DIR, exist_ok=True)

import numpy as np
import torch

from exp_z2_ci import MODELS, CTX, N_EVAL, K, BATCH, DEV, omp
from wdd_z_common import build, atom_type

torch.set_grad_enabled(False)
OUT = os.path.join(RESULTS_DIR, "exp_z3_ceiling.json")
ONLY = [m for m in os.environ.get("WDD_MODELS", "").split(",") if m]
CORPORA = [c for c in os.environ.get("WDD_CORPORA", "wikitext,pile").split(",") if c]
Z2 = json.load(open(os.environ["WDD_Z2_JSON"])) if os.environ.get("WDD_Z2_JSON") else {}
BINS = [(-1e9, 0.0), (0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.25), (1.25, 1e9)]
TWIN = 0.95
B_BOOT = 1000


def auc(score, label):
    """Mann-Whitney AUC of score for predicting label == 1."""
    r = score.argsort().argsort().astype(np.float64) + 1
    pos = label == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class Boot:
    """Percentile intervals of ratios of per-sequence sums."""
    def __init__(self, seq, n_seq, rng):
        self.seq, self.n = seq, n_seq
        self.counts = np.stack([np.bincount(rng.integers(0, n_seq, n_seq), minlength=n_seq) for _ in range(B_BOOT)])
    def ratio(self, num, den):
        ns = np.bincount(self.seq, weights=num.astype(np.float64), minlength=self.n)
        ds = np.bincount(self.seq, weights=den.astype(np.float64), minlength=self.n)
        if den.sum() == 0: return [None, None, None]
        boot = (self.counts @ ns) / np.maximum(self.counts @ ds, 1e-9)
        return [float(num.sum() / den.sum()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    def mean(self, x, mask=None):
        m = np.ones_like(x, dtype=np.float64) if mask is None else mask.astype(np.float64)
        return self.ratio(x * m, m)


def run(name, corpus):
    t0 = time.time()
    b = build(name, corpus, n_eval=N_EVAL)
    X, mu, acts, WN, A = b["X"], b["mu"], b["acts"], b["WN"], b["A"]
    L, DFF, OFF, STRIDE, NH, HD, fam = b["L"], b["DFF"], b["OFF"], b["STRIDE"], b["NH"], b["HD"], b["fam"]
    NT = X.shape[0]
    Xf = X - X.mean(0) if fam == "gpt2" else X - mu            # the attribution centering of exp_z2
    del X
    # ---- the ledger: each token's top-8 writes by magnitude, on the CPU in chunks ----
    top8, true8 = [], []
    for s in range(0, NT, 4096):
        T = torch.cat([acts[bb][s:s + 4096].float() * WN[bb] for bb in range(L + 1)], dim=1)
        i = T.abs().topk(8, dim=1).indices
        top8.append(i); true8.append(T.gather(1, i)); del T
    top8 = torch.cat(top8); true8 = torch.cat(true8).to(DEV)
    del acts
    atom8 = (OFF + (top8 // DFF) * STRIDE + (top8 % DFF)).to(DEV)                       # [NT, 8]
    # ---- the two supports ----
    _, _, sel, cof = omp(Xf, A, K, want=True)
    oneshot = torch.zeros(NT, K, dtype=torch.long, device=DEV)
    for s in range(0, NT, BATCH):
        oneshot[s:s + BATCH] = (Xf[s:s + BATCH] @ A.T).abs_().topk(K, dim=1).indices
    types = atom_type(sel.reshape(-1), OFF, STRIDE, DFF, NH, HD)
    composition = {n: float((types == i).float().mean()) for i, n in enumerate(["embedding", "mlp", "attention", "bias"])}
    print(f"  supports done ({time.time() - t0:.0f}s); support composition {composition}", flush=True)
    # ---- case features for the three dominant writes ----
    F = {k: [] for k in ["surv", "prom", "mag", "rel", "dist", "compete", "rank", "id_omp", "id_one", "twin_cos", "seq"]}
    eye8 = 2 * torch.eye(8, device=DEV)
    for s in range(0, NT, 2048):
        xb = Xf[s:s + 2048]; n = xb.shape[0]; xn = xb.norm(dim=-1)
        d8 = A[atom8[s:s + n]]                                                       # [n, 8, D]
        dots = (d8 @ xb[:, :, None])[:, :, 0]
        cos88 = (d8 @ d8.transpose(1, 2)).abs_() - eye8
        S = sel[s:s + n]
        cos_sup = (d8[:, :3] @ A[S].transpose(1, 2)).abs_()                          # [n, 3, K]
        for j in range(3):
            g = atom8[s:s + n, j]; ct = true8[s:s + n, j]
            same = g[:, None] == S
            cs = cos_sup[:, j].masked_fill(same, -1.0)
            F["twin_cos"].append(cs.max(-1).values)
            F["surv"].append(dots[:, j] / ct); F["prom"].append(dots[:, j].abs() / xn)
            F["mag"].append(ct.abs()); F["rel"].append(ct.abs() / xn)
            F["dist"].append((L - (g - OFF) // STRIDE).float())
            F["compete"].append(cos88[:, j].max(-1).values)
            F["rank"].append(torch.full((n,), float(j), device=DEV))
            F["id_omp"].append(same.any(-1).float())
            F["id_one"].append((g[:, None] == oneshot[s:s + n]).any(-1).float())
            F["seq"].append(torch.arange(s, s + n, device=DEV) // CTX)
    F = {k: torch.cat(v).cpu().numpy() for k, v in F.items()}
    seq = F["seq"].astype(np.int64); N = len(seq)
    np.savez_compressed(os.path.join(CASE_DIR, f"{name}_{corpus}.npz"), **{k: v.astype(np.float32) if k != "seq" else v for k, v in F.items()})
    # ---- statistics with intervals over sequences ----
    bt = Boot(seq, N_EVAL, np.random.default_rng(2026))
    surv, ido, idn, twin = F["surv"], F["id_omp"], F["id_one"], F["twin_cos"]
    res = {"n_cases": int(N), "id_rate_omp": bt.mean(ido), "id_rate_oneshot": bt.mean(idn),
           "recall_top1": bt.mean(ido, F["rank"] == 0), "support_composition": composition}
    if f"{name}/{corpus}" in Z2:
        z2 = Z2[f"{name}/{corpus}"]["ci"]["recall_top1_omp"]["point"]
        res["z2_recall_top1"] = z2
        print(f"  recall_top1 here {res['recall_top1'][0]:.4f} vs exp_z2 {z2:.4f} ({'ok' if abs(res['recall_top1'][0] - z2) < 0.01 else 'MISMATCH'})", flush=True)
    rows = []
    for lo, hi in BINS:
        m = (surv >= lo) & (surv < hi)
        rows.append({"bin": f"[{lo},{hi})", "n": int(m.sum()), "share": bt.mean(m.astype(np.float64)), "id_omp": bt.mean(ido, m), "id_oneshot": bt.mean(idn, m)})
    res["by_survival"] = rows
    intact, erased, weak = surv >= 0.75, surv < 0.25, (surv >= 0.25) & (surv < 0.75)
    miss = ido == 0
    res["punchline"] = {"frac_intact": bt.mean(intact.astype(np.float64)), "frac_erased": bt.mean(erased.astype(np.float64)),
                        "id_rate_intact": bt.mean(ido, intact), "id_rate_erased": bt.mean(ido, erased),
                        "frac_of_misses_erased_or_weak": bt.mean((surv < 0.5).astype(np.float64), miss)}
    twin_hit = twin >= TWIN
    res["miss_breakdown"] = {"erased": bt.mean(erased.astype(np.float64), miss), "weak": bt.mean(weak.astype(np.float64), miss),
                             "intact_twin": bt.mean((intact & twin_hit).astype(np.float64), miss),
                             "intact_other": bt.mean((intact & ~twin_hit).astype(np.float64), miss),
                             "miss_rate": bt.mean(miss.astype(np.float64))}
    res["twin"] = {"share_of_cases_with_twin_in_support": bt.mean(twin_hit.astype(np.float64)),
                   "id_rate_given_twin": bt.mean(ido, twin_hit), "id_rate_given_no_twin": bt.mean(ido, ~twin_hit),
                   "median_twin_cos": float(np.median(twin))}
    res["auc_omp"] = {k: round(auc(F[k] if k != "surv" else np.clip(F[k], -2, 3), ido), 3) for k in ["surv", "prom", "mag", "rel", "dist", "compete"]}
    by_block = {}
    for bb in range(L + 1):
        m = F["dist"] == (L - bb)
        if m.sum() > 50: by_block[bb] = {"n": int(m.sum()), "mean_surv": float(np.clip(surv[m], -2, 3).mean()), "id_omp": float(ido[m].mean())}
    res["by_block"] = by_block
    res["seconds"] = round(time.time() - t0)
    print(f"  {name}/{corpus}: id_omp {res['id_rate_omp'][0]:.3f}  intact {res['punchline']['frac_intact'][0]:.3f} id|intact {res['punchline']['id_rate_intact'][0]:.3f}  erased {res['punchline']['frac_erased'][0]:.3f} id|erased {res['punchline']['id_rate_erased'][0]:.3f}  misses: erased {res['miss_breakdown']['erased'][0]:.2f} weak {res['miss_breakdown']['weak'][0]:.2f} twin {res['miss_breakdown']['intact_twin'][0]:.2f} other {res['miss_breakdown']['intact_other'][0]:.2f}  AUC surv {res['auc_omp']['surv']:.3f}", flush=True)
    del A, Xf, sel, cof, oneshot; torch.cuda.empty_cache()
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
    print("DONE_Z3", flush=True)
