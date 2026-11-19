"""Post-review analysis: tables and the pre-registered verdicts (prereg.json) from pr1 and the pr2 runs.
Run exp_z2_recompute.py on each results/z2_* directory first (it adds the typical-state and calibration keys)."""
import json, os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); RD = os.path.join(H, "results"); REL = os.path.join(H, "..", "results")
K = ["4", "8", "16", "32", "64"]; f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
V = {}
# ---------------- pr1
p = json.load(open(os.path.join(RD, "pr1_provenance_random.json")))["dicts"]; ty = lambda n, k: p[n]["fvu_typical"][k]
rec2 = json.load(open(os.path.join(REL, "exp_d2_null_noemb.json")))
t = np.load(os.path.join(REL, "exp_d_tokens.npz")); x2 = t["xnorm2"].astype(np.float64); tm = np.sqrt(x2) <= 10 * np.median(np.sqrt(x2))
rec = {"noemb_correct_subsampled": {k: rec2["correct_subsampled_noemb"]["fvu_typical"][k] for k in K}, "noemb_future_blocks": {k: rec2["future_blocks_noemb"]["fvu_typical"][k] for k in K},
       "emb_correct_matched": {k: float(t[f"wdd_matched_correct_k{k}"][tm].sum() / x2[tm].sum()) for k in K}, "emb_future_blocks": {k: float(t[f"wdd_future_blocks_k{k}"][tm].sum() / x2[tm].sum()) for k in K}}
rand = {k: float(np.mean([ty(f"noemb_random_s{s}", k) for s in range(3)])) for k in K}; rspread = {k: float(np.ptp([ty(f"noemb_random_s{s}", k) for s in range(3)])) for k in K}
print("typical-state FVU by dictionary (atoms), k = " + ", ".join(K))
for n, v in p.items(): print(f"  {n:34s} {v['atoms']:6d}  " + "  ".join(f3(v["fvu_typical"][k]) for k in K) + f"   sinks k64 {v['fvu_sink']['64']:.4f}")
print(f"  {'noemb_random (mean of 3 seeds)':34s} {p['noemb_random_s0']['atoms']:6d}  " + "  ".join(f3(rand[k]) for k in K) + "   seed spread " + " ".join(f"{rspread[k]:.4f}" for k in K))
mw = {k: rand[k] - ty("noemb_correct_subsampled", k) for k in K}; mn = {k: rand[k] - ty("noemb_future_blocks", k) for k in K}; mrot = {k: ty("noemb_future_blocks_rotated", k) - ty("noemb_future_blocks", k) for k in K}
print("margins over size-matched random (positive = better than random), k = " + ", ".join(K))
print("  writers (blocks 0-6, subsampled)   " + "  ".join(f"{mw[k]:+.3f}" for k in K))
print("  non-writers (blocks 7-11)          " + "  ".join(f"{mn[k]:+.3f}" for k in K))
print("  non-writers over their own rotation" + "  ".join(f"{mrot[k]:+.3f}" for k in K))
print("  writers over their own rotation    " + "  ".join(f"{ty('noemb_correct_subsampled_rotated', k) - ty('noemb_correct_subsampled', k):+.3f}" for k in K))
print("  with embeddings: writers / non-writers over size-matched random " + "  ".join(f"{ty('emb_random', k) - ty('emb_correct_matched', k):+.3f}/{ty('emb_random', k) - ty('emb_future_blocks', k):+.3f}" for k in K))
gate = max(abs(ty(n, k) - rec[n][k]) for n in rec for k in K)
V["P1"] = gate <= 0.002; V["P2"] = 0.52 <= rand["32"] <= 0.56 and rspread["32"] <= 0.005; V["P3"] = all(mn[k] > 0 for k in K) and 0.02 <= mn["32"] <= 0.08
V["P4"] = all(mw[k] > 0 for k in K) and mw["32"] >= 0.12; V["P5"] = mrot["32"] >= 0.05; V["P6"] = (mw["32"] - mn["32"]) >= 2 * mn["32"]
V["P7"] = abs(ty("emb_only", "32") - ty("emb_future_blocks", "32")) <= 0.03; V["P8"] = 0.49 <= ty("emb_random", "32") <= 0.52 and p["emb_random"]["fvu_sink"]["64"] > 0.3 and p["emb_future_blocks"]["fvu_sink"]["64"] < 0.01
print(f"gate: largest difference from the recorded typical FVU {gate:.4f}")
# ---------------- pr2
paper = json.load(open(os.path.join(REL, "exp_z2_ci.json")))["gpt2/wikitext"]
def z(tag):
    """typical-state FVU, recall and calibration computed from the saved per-token record, with the definitions of
    exp_z2_recompute (typical = norm at most ten times the median; calibration over identified dominant writes on
    typical states); computed here because the recompute script needs at least one sink state, which a randomly
    initialised model does not have"""
    d = os.path.join(RD, tag)
    if not os.path.exists(os.path.join(d, "exp_z2_ci.json")): return None
    r = json.load(open(os.path.join(d, "exp_z2_ci.json")))["gpt2/wikitext"]; T = np.load(os.path.join(d, "exp_z2_tokens", "gpt2_wikitext.npz"))
    x2 = T["xnorm2"].astype(np.float64); nrm = np.sqrt(x2); typ = nrm <= 10 * np.median(nrm)
    fv = lambda key: float(T[key].astype(np.float64)[typ].sum() / x2[typ].sum())
    h = T["hit"] & typ[:, None]; pr = T["pred"][h].astype(np.float64); tr = T["true"][h].astype(np.float64)
    return dict(L=r["L"], n=r["n_states"], atoms=r["atoms"], w32=fv("err32_weight"), r32=fv("err32_rotated"), x32=fv("err32_random"), w64=fv("err64_weight"),
                rec=float(T["hit"][:, 0].mean()), rec1=float(T["hit1_oneshot"].mean()), sign=float((np.sign(pr) == np.sign(tr)).mean()) if len(tr) else None,
                rel=float(np.median(np.abs(pr - tr) / np.abs(tr))) if len(tr) else None, nid=int(h.sum()), sinkshare=float(x2[~typ].sum() / x2.sum()), nsink=int((~typ).sum()))
Z = {tag: z(tag) for tag in ("z2_L2", "z2_L6", "z2_L10", "z2_init_L6")}
pc = paper["ci"]; pcal = paper["calibration"]
Z["paper_L6"] = dict(L=6, n=paper["n_states"], atoms=paper["atoms"], w32=pc["fvu32_weight_typical"]["point"], r32=pc["fvu32_rotated_typical"]["point"], x32=pc["fvu32_random_typical"]["point"], w64=pc["fvu64_weight_typical"]["point"],
                     rec=pc["recall_top1_omp"]["point"], rec1=pc["recall_top1_oneshot"]["point"], sign=pcal["sign_agreement"]["point"], rel=pcal["median_rel_error"]["point"], nid=pcal["n"], sinkshare=paper.get("sink_variance_share"), nsink=paper.get("n_sink"))
print("GPT-2 small, paper's protocol (typical-state FVU; recall over all states; sign and median relative error of identified dominant writes on typical states)")
print("  run          layer  states  atoms   FVU32 weight / rotated / random   FVU64   OMP recall  one-shot  sign    rel.err  sink var. share")
for tag, v in Z.items():
    if v: print(f"  {tag:12s} {v['L']:5d} {v['n']:7d} {v['atoms']:6d}   {f3(v['w32'])} / {f3(v['r32'])} / {f3(v['x32'])}            {f3(v['w64'])}   {f3(v['rec'])}       {f3(v['rec1'])}     {f3(v['sign'])}   {f3(v['rel'])}   {f3(v['sinkshare'])}")
L6, L2, L10, I6 = Z["z2_L6"], Z["z2_L2"], Z["z2_L10"], Z["z2_init_L6"]
if L6: V["G1"] = abs(L6["w32"] - 0.338) <= 0.01 and abs(L6["rec"] - 0.647) <= 0.03
if L2: V["G2"] = (L2["r32"] - L2["w32"]) >= 0.15 and L2["rec"] >= 0.75
if L10: V["G3"] = L10["rec"] <= 0.40 and (L10["r32"] - L10["w32"]) >= 0.05
if L2 and L10: V["G4"] = all(v["sign"] is not None and v["sign"] >= 0.99 for v in (L2, L10))
if I6:
    V["I1"] = abs(I6["w32"] - I6["r32"]) <= 0.02 and abs(I6["w32"] - I6["x32"]) <= 0.02; V["I2"] = I6["rec"] < 0.10
    tr = (L6 or Z["paper_L6"]); V["I3"] = (tr["r32"] - tr["w32"]) >= 5 * max(I6["r32"] - I6["w32"], 1e-9)
pre = json.load(open(os.path.join(H, "prereg.json")))
txt = {**pre["pr1"]["predictions"], **pre["pr2_layers"]["predictions"], **pre["pr2_random_init"]["predictions"]}
print("pre-registered verdicts:"); [print(f"  {k} {'confirmed' if v else 'refuted'}: {txt[k]['text']} (guess {txt[k]['guess']})") for k, v in V.items()]
json.dump({"verdicts": V, "pr1_margins": {"writers_over_random": mw, "nonwriters_over_random": mn, "nonwriters_over_own_rotation": mrot, "random_mean": rand, "random_seed_spread": rspread, "gate_max_diff": gate}, "pr2": Z}, open(os.path.join(RD, "summary.json"), "w"), indent=1)
