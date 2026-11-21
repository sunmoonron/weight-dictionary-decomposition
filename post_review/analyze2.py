"""Post-review analysis, second round: the numbers and pre-registered verdicts (prereg2.json) for pr3 and pr4."""
import json, os
H = os.path.dirname(os.path.abspath(__file__)); RD = os.path.join(H, "results"); J = lambda n: json.load(open(os.path.join(RD, n))) if os.path.exists(os.path.join(RD, n)) else None
a = J("pr3_omp_oracles_dual.json"); lt = J("pr3_lasso.json"); lf = J("pr3_lasso_lam9.9105.json"); f = J("pr4_sae_features.json"); V = {}
S = dict(a["solvers"]) if a else {}
if lt: S["lasso (tuned)"] = lt["solvers"]["lasso"]
if lf: S["lasso (penalty 9.9105)"] = lf["solvers"]["lasso"]
print("solvers, GPT-2 small layer 6, 64 atoms each: top-1 recall of the largest MLP write [95% interval], refit FVU on typical states")
for k, v in S.items(): print(f"  {k:26s} {v['recall_top1']['point']:.3f} [{v['recall_top1']['lo']:.3f}, {v['recall_top1']['hi']:.3f}]   FVU {v['fvu_refit_typical']:.3f}" + (f"   penalty {v['lam']:.3f}, median nonzeros {v['nnz_median']:.0f}" if "lam" in v else ""))
r = lambda k: S[k]["recall_top1"]["point"]
if a:
    V["S1"] = r("dual_frame_eps0.01") >= r("oneshot") + 0.02 and r("dual_frame_eps0.01") >= r("omp") + 0.10
    if lt: V["S2"] = r("lasso (tuned)") >= 0.80 and r("lasso (tuned)") >= r("omp") + 0.15
    if lt: V["S3"] = all(S["omp"]["fvu_refit_typical"] < v["fvu_refit_typical"] for k, v in S.items() if k != "omp")
    c = a["support_composition_typical"]; at = a["attention_oracle"]; em = a["embedding_oracle"]; sa = at["selected_attention_atoms"]
    print("support composition on typical states: " + ", ".join(f"{k} {v:.3f}" for k, v in c.items()))
    print(f"heads: dominant head has an atom in the support: raw {at['raw']['dominant_head_selected']:.3f}, centred {at['centred']['dominant_head_selected']:.3f}; a random head {at['random_head_selected']:.3f} ({at['heads_with_an_atom_mean']:.1f} of 84 heads have an atom)")
    print(f"       energy of the dominant head's write captured by its selected atoms, median when selected: raw {at['raw']['energy_captured_median_when_selected']:.3f}, centred {at['centred']['energy_captured_median_when_selected']:.3f}; mean over all typical tokens: raw {at['raw']['energy_captured_mean_all_typical']:.4f}, centred {at['centred']['energy_captured_mean_all_typical']:.4f}")
    print(f"       dominant head's share of all head write energy, median: raw {at['raw']['dominant_head_share_of_head_energy_median']:.3f}, centred {at['centred']['dominant_head_share_of_head_energy_median']:.3f}; span check {at['span_check_max_rel_err']:.1e}")
    print(f"       selected attention atoms (n {sa['n']}): sign agrees with the true centred coefficient {sa['sign_agreement']:.3f}, real by the 5% rule {sa['real_share']:.3f}, r {sa['r_recovered_true']:.3f}, median ratio of real ones {sa['median_ratio_real']:.2f}, median relative error {sa['median_rel_error_real']:.2f}; largest true attention atom in the support {at['largest_true_attention_atom_in_support']:.3f}")
    if "all_attention_atoms" in at: print(f"       base rate over all 5,376 attention atoms of a token: real by the 5% rule {at['all_attention_atoms']['real_share']:.3f}; median |true| / largest: all atoms {at['all_attention_atoms']['true_over_max_median']:.3f}, selected atoms {sa['true_over_max_median']:.3f}")
    print(f"embeddings: a selected token row is the current token {em['selected_token_rows']['current_token']:.4f}, an earlier context token {em['selected_token_rows']['earlier_context_token']:.4f} (unigram-matched random tokens {em['unigram_matched_random_tokens']['earlier_context_token']:.4f}; n {em['selected_token_rows']['n']}); own token row in the support {em['own_token_row_in_support']:.4f}; own position row {em['own_position_row_in_support']:.4f}; selected position rows that are the own position {em['selected_position_rows_that_are_own']:.4f}, within 8 positions {em['selected_position_rows_within_8']:.4f}")
    V["H1"] = 0.10 <= at["raw"]["dominant_head_selected"] <= 0.30
    V["H2"] = at["raw"]["energy_captured_median_when_selected"] < 0.2 and at["centred"]["energy_captured_median_when_selected"] < 0.2
    V["H3"] = at["raw"]["dominant_head_selected"] <= 3 * at["random_head_selected"]
    V["H4"] = 0.60 <= sa["sign_agreement"] <= 0.90; V["H5"] = sa["real_share"] >= 0.5; V["H6"] = at["largest_true_attention_atom_in_support"] < 0.30
    V["E1"] = 0.05 <= c["token"] <= 0.40; V["E2"] = em["selected_token_rows"]["current_token"] < 0.05
    V["E3"] = em["selected_token_rows"]["earlier_context_token"] < em["unigram_matched_random_tokens"]["earlier_context_token"]
    V["E4"] = em["own_token_row_in_support"] < 0.10; V["E5"] = em["own_position_row_in_support"] < 0.10
if f:
    T = f["targets"]; u = lambda t, d="weight", k="16": T[t][d]["unexplained_median"][k]; fa = lambda t, d="weight": T[t][d]["first_atom"]
    print(f"SAE features ({f['n_features']} decoder rows of {f['sae']}): median unexplained fraction by number of atoms (1, 2, 4, 8, 16), and the type of the first atom")
    for t in T:
        for d in T[t]: print(f"  {t:22s} {d:8s} " + "  ".join(f"{T[t][d]['unexplained_median'][k]:.3f}" for k in ("1", "2", "4", "8", "16")) + f"   first atom: MLP {fa(t, d)['mlp']:.3f}, token {fa(t, d)['token']:.3f}, attention {fa(t, d)['attention']:.3f}, position {fa(t, d)['position']:.3f}, bias {fa(t, d)['bias']:.3f}")
    print("  block of the first atom when it is an MLP row, SAE features:", fa("sae_features")["mlp_block_hist"])
    V["F1"] = fa("sae_features")["mlp"] >= 0.85; V["F2"] = fa("random_directions")["mlp"] < 0.70
    V["F3"] = 0.40 <= u("sae_features") <= 0.56 and u("sae_features", "rotated") - u("sae_features") >= 0.15
    V["F4"] = u("random_directions") >= 0.65; V["F5"] = abs(u("sae_features") - u("typical_states")) <= 0.05
pre = json.load(open(os.path.join(H, "prereg2.json"))); txt = {**pre["pr3"]["predictions"], **pre["pr4"]["predictions"]}
print(f"pre-registered verdicts ({sum(V.values())} of {len(V)} confirmed):"); [print(f"  {k} {'confirmed' if v else 'refuted'}: {txt[k]['text']} (guess {txt[k]['guess']})") for k, v in V.items()]
json.dump(dict(verdicts=V), open(os.path.join(RD, "summary2.json"), "w"), indent=1)
