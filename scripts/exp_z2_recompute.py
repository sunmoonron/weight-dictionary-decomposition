"""Recompute reconstruction metrics of the Z2 sweep from the saved per-token records, without a GPU,
adding the convention that separates sink states from typical states.

In every model a few states per sequence, always the first position and sometimes one more, have a
norm above ten times the median (nothing lies between five and ten times the median in any of the
sixteen entries, so the threshold is not a tuning choice). These are the attention-sink states of the massive-activation
literature. They carry most of the summed variance (recorded as sink_variance_share) and every
dictionary that contains the massive-channel directions reconstructs them from one or two atoms, so
an FVU over all states mostly measures them. The *_typical metrics restrict numerator and
denominator to the other states; the *_sink metrics are the complement; the intervals are the same
bootstrap over sequences. Existing keys of exp_z2_ci.json are kept."""
import json, os, sys
import numpy as np
RESULTS_DIR = os.environ.get("WDD_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
OUT = os.path.join(RESULTS_DIR, "exp_z2_ci.json"); TOK = os.path.join(RESULTS_DIR, "exp_z2_tokens")
B_BOOT = 1000
RES = json.load(open(OUT))


def calibration(T, typ, seq, n_seq, counts):
    """Sign agreement, relative error and the ratio of recovered to true coefficient for the identified
    dominant writes on typical states, overall and by magnitude decile; intervals over sequences."""
    hit = T["hit"] & typ[:, None]; h = hit.reshape(-1); s3 = np.repeat(seq, 3)[h]
    p = T["pred"].reshape(-1)[h].astype(np.float64); t = T["true"].reshape(-1)[h].astype(np.float64)
    at = np.abs(t); rel = np.abs(p - t) / at; ratio = p / t; sign = (np.sign(p) == np.sign(t)).astype(np.float64)
    def boot_mean(x):
        ns = np.bincount(s3, weights=x, minlength=n_seq); ds = np.bincount(s3, minlength=n_seq).astype(np.float64)
        b = (counts @ ns) / np.maximum(counts @ ds, 1e-9)
        return {"point": float(x.mean()), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5))}
    def boot_median(x):
        b = [float(np.median(x[np.isin(s3, np.nonzero(c)[0])])) for c in counts[:200]]
        return {"point": float(np.median(x)), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5))}
    edges = np.quantile(at, np.linspace(0, 1, 11)); dec = []
    for d in range(10):
        m = (at >= edges[d]) & (at <= edges[d + 1] + 1e-9)
        dec.append({"median_ratio": float(np.median(ratio[m])), "median_rel_error": float(np.median(rel[m])), "within_25": float((rel[m] <= 0.25).mean()), "share_negative": float((t[m] < 0).mean())})
    hall = typ[:, None] & np.ones_like(T["hit"], dtype=bool); pa = T["pred"].reshape(-1)[hall.reshape(-1)].astype(np.float64); ta = T["true"].reshape(-1)[hall.reshape(-1)].astype(np.float64)
    return {"n": int(h.sum()), "share_negative": float((t < 0).mean()), "sign_agreement": boot_mean(sign), "median_rel_error": boot_median(rel), "within_25": boot_mean((rel <= 0.25).astype(np.float64)),
            "median_ratio": boot_median(ratio), "deciles": dec,
            "r_identified_typical": float(np.corrcoef(p, t)[0, 1]), "r_unconditional_typical": float(np.corrcoef(pa, ta)[0, 1])}
for key in sorted(RES):
    f = os.path.join(TOK, key.replace("/", "_") + ".npz")
    if not os.path.exists(f): print("no token record for", key); continue
    T = np.load(f); seq = T["seq"]; n_seq = int(seq.max()) + 1; n = len(seq); ctx = n // n_seq
    nrm = np.sqrt(T["xnorm2"].astype(np.float64)); med = np.median(nrm)
    sink = nrm > 10 * med; m1 = ~sink
    assert (nrm > 5 * med).sum() == (nrm > 10 * med).sum(), key
    rng = np.random.default_rng(2026)
    counts = np.stack([np.bincount(rng.integers(0, n_seq, n_seq), minlength=n_seq) for _ in range(B_BOOT)])
    def ratio(num, den):
        ns, ds = np.bincount(seq, weights=num, minlength=n_seq), np.bincount(seq, weights=den, minlength=n_seq)
        b = (counts @ ns) / (counts @ ds)
        return {"point": float(num.sum() / den.sum()), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5))}
    x2 = T["xnorm2"].astype(np.float64); x2k = T["xnorm2_keep"].astype(np.float64)
    ci = RES[key]["ci"]
    for d in ("weight", "rotated", "random"):
        for k in (32, 64):
            e = T[f"err{k}_{d}"].astype(np.float64)
            ci[f"fvu{k}_{d}_typical"] = ratio(e * m1, x2 * m1); ci[f"fvu{k}_{d}_sink"] = ratio(e * sink, x2 * sink)
    for d in ("weight", "rotated"):
        e = T[f"err32_{d}_keep"].astype(np.float64)
        ci[f"fvu32_{d}_offtop5_typical"] = ratio(e * m1, x2k * m1)
    for k_ in [k_ for k_ in ci if k_.endswith("_pos1")]: del ci[k_]
    RES[key].pop("variance_share_position_0", None)
    RES[key]["sink_variance_share"] = float(x2[sink].sum() / x2.sum()); RES[key]["n_sink"] = int(sink.sum()); RES[key]["sink_norm_over_median"] = float(nrm[sink].min() / med)
    RES[key]["calibration"] = calibration(T, m1, seq, n_seq, counts)
    RES[key]["ctx"] = int(ctx)
    print(f"{key:24} sinks {sink.sum():3d} ({100 * RES[key]['sink_variance_share']:.1f}% of variance)  FVU32 weight all {ci['fvu32_weight']['point']:.3f} -> typical {ci['fvu32_weight_typical']['point']:.3f} [{ci['fvu32_weight_typical']['lo']:.3f}, {ci['fvu32_weight_typical']['hi']:.3f}]  k64 {ci['fvu64_weight_typical']['point']:.3f}  rotated {ci['fvu32_rotated_typical']['point']:.3f}/{ci['fvu64_rotated_typical']['point']:.3f}  random {ci['fvu32_random_typical']['point']:.3f}  sink FVU {ci['fvu32_weight_sink']['point']:.4f} vs rot {ci['fvu32_rotated_sink']['point']:.3f}  off-top5 {ci['fvu32_weight_offtop5_typical']['point']:.2f} vs {ci['fvu32_rotated_offtop5_typical']['point']:.2f}")
json.dump(RES, open(OUT, "w"), indent=1)
print("exp_z2_ci.json updated with *_typical and *_sink metrics for", len(RES), "entries")
