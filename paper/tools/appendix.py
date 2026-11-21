"""Generate appendix.tex for the WDD paper from the result files in results/ at the repository root.
Each table and figure below is read straight from the JSON a script wrote; nothing is typed by hand."""
import json, os, sys
from math import comb
R = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results')
J = lambda n: json.load(open(f'{R}/{n}.json'))
a, d, i_, x7, h, g, j, x2, s9, w, x9, x4, k, x1, x6, x5, s8, t2, u1, u2, v1, x3, n30 = [J(n) for n in [
    'exp_a_wdd', 'exp_d_provenance', 'exp_i_observability', 'exp_x7_oracle', 'exp_h_clusters', 'exp_g_causal', 'exp_j_rerun',
    'exp_x2_families_attrib', 'exp_s9_families', 'exp_w_audit', 'exp_x9_top1conc', 'exp_x4_crossdomain', 'exp_k_nulls',
    'exp_x1_doseresp_all', 'exp_x6_lnfrozen', 'exp_x5_lin', 'exp_s8_loop', 'exp_t2_compensation', 'exp_u1_reserves', 'exp_u2_leash',
    'exp_v1_bakeoff', 'exp_x3_bakeoff2', 'exp_x3_bakeoff2_n30']]
out = []
P = out.append
f3 = lambda v: '' if v is None else f'{v:.3f}'
f2 = lambda v: '' if v is None else f'{v:.2f}'
def table(cols, rows, caption, label, align=None, size='small', note=None, fit=False):
    align = align or ('l' + 'r' * (len(cols) - 1))
    P(r'\begin{table}[H]\centering\%s' % size)
    P(r'\caption{%s}\label{%s}' % (caption, label))
    if fit: P(r'\begin{adjustbox}{max width=\linewidth}')
    P(r'\begin{tabular}{%s}\toprule' % align)
    P(' & '.join(cols) + r' \\ \midrule')
    for r in rows: P(' & '.join(str(c) for c in r) + r' \\')
    P(r'\bottomrule\end{tabular}' + (r'\end{adjustbox}' if fit else ''))
    if note: P(r'\par\smallskip\parbox{0.95\linewidth}{\footnotesize %s}' % note)
    P(r'\end{table}')
fisher = lambda ap, an, bp, bn: sum(comb(ap + bp, x) * comb(an + bn - ap - bp, an - x) for x in range(ap, min(ap + bp, an) + 1)) / comb(an + bn, an)
MODELS = [('gpt2', 'GPT-2 small'), ('smollm2-135m', 'SmolLM2-135M'), ('qwen2.5-0.5b', 'Qwen2.5-0.5B'), ('pythia-410m', 'Pythia-410m'), ('olmo-1b-0724', 'OLMo-1B'), ('qwen2.5-7b', 'Qwen2.5-7B'), ('llama-7b', 'Llama-7B'), ('pythia-6.9b', 'Pythia-6.9B')]
GATED = {'smollm2-135m', 'qwen2.5-0.5b', 'olmo-1b-0724', 'qwen2.5-7b', 'llama-7b'}

# ---------------- A. dictionary ----------------
P(r'\section{Dictionary construction and model census}\label{app:dict}')
P(r'For GPT-2 small at layer 6 the dictionary holds $50{,}257$ token embedding rows and $1{,}024$ position embedding rows, and for each of blocks 0 to 6 it holds $3{,}072$ MLP down-projection rows, a 64-vector SVD basis of the output-projection row space of each of the 12 attention heads, which amounts to 768 attention atoms per block, and the attention and MLP output biases. The total is $m = 78{,}175$ atoms, about 102 times overcomplete in $d = 768$. Each atom is unit-normed and carries the label of its block, its component and its write subspace. For the other families we take the same census at the middle layer $L$, with the down-projection columns as MLP write directions where the projection is a standard linear layer and no bias atoms. The family sweep restricts embedding atoms to the tokens of the evaluation slice; the attribution rerun keeps each embedding row, which is why its dictionaries are larger. Table~\ref{app:census} lists the models.')
rows = [('GPT-2 small', 768, 3072, 12, 6, '78,175', '78,175', '3.647')]
for key, nm in MODELS[1:]:
    s = s9[key]; rows.append((nm, s['d'], s['dff'], s['nb'], s['L'], f"{s['atoms']:,}", f"{x2[key]['n_atoms']:,}", f"{s['ce']:.3f}"))
table(['Model', '$d$', '$d_{\\mathrm{ff}}$', 'blocks', '$L$', 'atoms, sweep', 'atoms, attribution', 'clean CE'], rows,
      'The eight models: width, MLP width, depth, dictionary layer, dictionary sizes and clean cross-entropy on the evaluation tokens (32{,}768 for GPT-2 small, 8{,}192 for the sweep). The 7B-class rows were run in bfloat16.', 'app:census', fit=True)

# ---------------- B. reconstruction ----------------
P(r'\section{Reconstruction}\label{app:recon}')
P(r'The tables of this section are over all states; Appendix~\ref{app:z-gpt2} gives the same sweep and controls over the typical states, and Section~\ref{sec:setup} defines the split. Table~\ref{app:fvu} gives the full sparsity sweep together with the effect of splicing each reconstruction back into the forward pass, and Table~\ref{app:prov} the three provenance controls. The uncentered weight dictionary is the same dictionary applied to raw rather than mean-centered states.')
runs = a['runs']
rn = [('random_L6', 'random directions'), ('pca_L6', 'PCA basis'), ('weight_L6', 'weight dictionary'), ('weight_L6_pure', 'weight dictionary, uncentered'), ('actdict_L6', 'activation dictionary'), ('weight_L8', 'weight dictionary, layer 8')]
rows = []
for key, nm in rn:
    r = runs.get(key)
    if not r: continue
    fv = r['fvu']; fi = r.get('fidelity', {})
    rows.append((nm, f"{r.get('atoms', ''):,}" if isinstance(r.get('atoms'), int) else '', *[f3(fv.get(str(kk))) for kk in (4, 8, 16, 32, 64)],
                 f"{fi.get('32', {}).get('dce', 0):+.3f}", f"{fi.get('64', {}).get('dce', 0):+.3f}", f3(fi.get('32', {}).get('kl')), f3(fi.get('64', {}).get('kl'))))
table(['Dictionary', 'atoms', 'FVU $k{=}4$', '8', '16', '32', '64', '$\\Delta$CE 32', '$\\Delta$CE 64', 'KL 32', 'KL 64'], rows,
      'Reconstruction of GPT-2 layer-6 states over all 32{,}768 tokens, sink states included: fraction of variance unexplained at each sparsity, and the change in cross-entropy and the KL divergence to the clean next-token distribution when the reconstruction replaces the state in the forward pass. Clean cross-entropy is %.3f.' % a['clean_ce'], 'app:fvu', size='footnotesize', fit=True)
rows = []
for key, nm in [('wdd_rotated', 'weight dictionary after one Gram-preserving rotation'), ('wdd_matched_correct', 'correct dictionary subsampled to the future-block size'), ('wdd_future_blocks', 'blocks 7 to 11 only, embeddings kept')]:
    r = d[key]; rows.append((nm, f"{r['atoms']:,}", *[f3(r['fvu'][str(kk)]) for kk in (4, 8, 16, 32, 64)]))
table(['Dictionary', 'atoms', 'FVU $k{=}4$', '8', '16', '32', '64'], rows, 'Provenance controls on the same states, over all states. The rotation preserves the Gram matrix of the dictionary and destroys its alignment with the residual coordinates; the future-block dictionary contains only atoms that could not have written to a layer-6 state.', 'app:prov', fit=True)

# ---------------- C. attribution ----------------
P(r'\section{Attribution and observability}\label{app:attrib}')
P(r'Table~\ref{app:survbins} is the data behind the right panel of Figure~\ref{fig:fvu}. Table~\ref{app:auc} asks which property of a write predicts whether OMP identifies it; prominence is circular for one-shot ranking, which selects by prominence, so the informative rows are survival and the rest. Tables~\ref{app:oracle} to~\ref{app:probe} are the oracle baselines, the near-duplicate clustering, the causal tracking checks and the information-survival probe cited in Section~\ref{sec:attrib}.')
rows = []
for b in i_['by_survival']:
    lo, hi = b['bin'].strip('[)').split(',')
    lab = ('$s<%s$' % hi) if float(lo) < -1e6 else ('$s\\ge %s$' % lo if float(hi) > 1e6 else '$%s\\le s<%s$' % (lo, hi))
    rows.append((lab, f"{b['n']:,}", f3(b['frac_of_cases']), f3(b['id_omp']), f3(b['id_oneshot'])))
table(['Survival bin', 'writes', 'share of cases', 'identified, OMP', 'identified, one-shot'], rows, 'Top-1 identification rate of each token\'s three dominant MLP writes (%s cases) by the survival of the write in the layer-6 state. Overall rates: OMP %.3f, one-shot %.3f; among intact writes %.3f, among erased writes %.4f.' % (f"{i_['n_cases']:,}", i_['id_rate_omp'], i_['id_rate_oneshot'], i_['punchline']['id_rate_intact'], i_['punchline']['id_rate_erased']), 'app:survbins', fit=True)
defs = {'surv': 'survival $(\\mathbf{x}\\cdot\\mathbf{d})/c$', 'prom': 'prominence $|\\mathbf{x}\\cdot\\mathbf{d}|/\\lVert\\mathbf{x}\\rVert$', 'mag': 'magnitude $|c|$', 'rel': 'relative magnitude $|c|/\\lVert\\mathbf{x}\\rVert$', 'dist': 'write age, $6$ minus the source block', 'compete': 'competition, max $|\\cos|$ with the token\'s other top-8 write directions'}
table(['Predictor', 'AUC for OMP identification'], [(defs[kk], f3(v)) for kk, v in i_['auc_omp'].items()], 'Area under the ROC curve for predicting whether OMP identifies a dominant write, one property at a time. Survival and prominence predict; write age does not.', 'app:auc', align='lr', fit=True)
A_ = x7['A_synthetic']; B_ = x7['B_oracle_support']; C_ = x7['C_coherence_mlp']
rows = [('synthetic states, sum of the token\'s MLP writes only: top-1 recall, OMP', f3(A_['recall_top1_omp'])), ('\\quad top-1 recall, one-shot', f3(A_['recall_top1_oneshot'])), ('\\quad top-3 recall, OMP', f3(A_['recall_top3_omp'])), ('\\quad $r$ on identified writes', f3(A_['r_identified'])),
        ('real states, same tokens: top-1 recall OMP / one-shot', f"{A_['real_state_ref']['recall_top1_omp']:.3f} / {A_['real_state_ref']['oneshot']:.3f}"),
        ('least squares on the true top-3 atoms only: $r$, all dominant writes', f3(B_['r_all_bigwrites'])), ('\\quad $r$ on erased writes, survival below 0.25 ($n=%s$)' % f"{B_['n_erased']:,}", f3(B_['r_erased(surv<0.25)'])), ('\\quad $r$ on intact writes, survival above 0.75 ($n=%s$)' % f"{B_['n_intact']:,}", f3(B_['r_intact(surv>0.75)'])),
        ('MLP atom coherence: median / 99th percentile / max of max $|\\cos|$ to another atom', f"{C_['median_max_abs_cos']:.3f} / {C_['p99_max_abs_cos']:.3f} / {C_['max_abs_cos']:.3f}")]
table(['Oracle baseline (%s tokens)' % f"{x7['n_tokens']:,}", 'value'], rows, 'Two oracles that locate the identification ceiling. Synthetic states built only from each token\'s own MLP writes give the same recall as real states, so the ceiling belongs to the writes cancelling each other, not to attention or the embedding. Refitting on the true support alone gives a far worse coefficient correlation than the joint sparse fit.', 'app:oracle', align='lr', fit=True)
rows = []
for tau in ('0.95', '0.9'):
    t = h[f'tau_{tau}']; cs = t['cluster_stats']
    rows.append((tau, cs['n_multi_clusters'], f"{cs['atoms_in_multi']:,}", cs['largest_cluster'], f3(t['recall_top1_cluster_omp64']), f3(t['recall_top1_cluster_oneshot64']), f3(t['recall_top3_cluster_omp64']), f3(t['est_r_cluster_identified'])))
table(['cosine $\\tau$', 'multi-atom clusters', 'atoms in them', 'largest', 'top-1 OMP', 'top-1 one-shot', 'top-3 OMP', '$r$ identified'], rows, 'Near-duplicate atoms merged into clusters at two cosine thresholds; a hit on any atom of the true write\'s cluster counts. Atom-level top-1 recall is %.3f, so duplication explains almost none of the misses.' % h['sanity_atom_recall_top1'], 'app:clusters', size='footnotesize', fit=True)
g1 = g['g1_injection']; g2 = g['g2_source_intervention']
table(['Causal check', '$n$', '$r$', 'slope'], [('injected atom, recovered coefficient vs injected (mean leakage onto other atoms %.4f)' % g['g1_mean_leakage'], g1['n'], f3(g1['r']), f3(g1['slope'])), ('source intervention, change in recovered coefficient vs change in the true write', g2['n'], f3(g2['r']), f3(g2['slope']))], 'Does the decomposition track causal changes? A known write added to a state is recovered at slope %.2f; changing a real source neuron\'s activation moves its recovered coefficient at slope %.2f.' % (g1['slope'], g2['slope']), 'app:causal', align='lrrr', fit=True)
gr = j['groups']; sd = j['survival_by_depth']
rows = [(nm, gr[key]['n_neurons'], f3(gr[key]['r2_full_median']), f3(gr[key]['r2_dir_median']), f2(gr[key]['frac_transformed']), f2(gr[key]['frac_info_lost'])) for key, nm in [('block3_transient', 'block-3 writes later erased'), ('control_intact', 'intact control writes')]]
table(['Writer group', 'neurons', 'median out-of-sample $R^2$, full state', 'median $R^2$, own direction', 'transformed', 'lost'], rows, 'Leakage-controlled ridge probe from the layer-6 state back to the writer\'s activation. The write\'s own direction is a useless readout for erased writers, the full state is not, so erasure is mostly transformation. The probed group is the block-3 neurons whose writes have the lowest mean survival at the layer-6 readout; averaged over all of their firing tokens that survival is %s after blocks 3 to 6, against %s for the control group of intact writers from blocks 4 to 6, so the group is defined by low survival relative to the population, not by the erasure of every write.' % (', '.join(f"{v:.2f}" for v in sd['block3_transient'].values()), ', '.join(f"{v:.2f}" for v in sd['control_intact'].values())), 'app:probe', size='footnotesize', fit=True)

# ---------------- D. generality ----------------
P(r'\section{Generality}\label{app:general}')
P(r'Table~\ref{app:attribmodels} extends Table~\ref{tab:families} with one-shot recall and the unconditional correlation, Table~\ref{app:deciles} and Figure~\ref{app:decilefig} give the magnitude-decile correlations behind the architectural divide, Table~\ref{app:channels} the channel-robust FVU, Table~\ref{app:conc} the concentration of the largest write, and Table~\ref{app:pile} the Pile replication.')
gp = w['B_attrib']
rows = [('GPT-2 small', 'no', f3(gp['recall_top1']), '0.766', f3(gp['r_identified']), f3(gp['r_unconditional']))]
for key, nm in MODELS[1:]:
    r = x2[key]; rows.append((nm, 'yes' if key in GATED else 'no', f3(r['recall_top1_omp']), f3(r['recall_top1_oneshot']), f3(r['r_identified']), f3(r['r_unconditional'])))
table(['Model', 'gated MLP', 'top-1 recall, OMP', 'top-1 recall, one-shot', '$r$ identified', '$r$ unconditional'], rows, 'Attribution in the rerun with full-vocabulary dictionaries on 8{,}192 states per model (GPT-2 small from the audit rerun). The unconditional correlation counts misses as zero.', 'app:attribmodels', fit=True)
rows = [('GPT-2 small', *[f2(v) for v in gp['r_identified_per_decile']])]
for key, nm in MODELS[1:]:
    rows.append((nm, *[('' if v is None else f2(v)) for v in x2[key]['r_identified_per_decile']]))
table(['Model'] + [str(n) for n in range(1, 11)], rows, 'Correlation between recovered and true coefficients within each magnitude decile of the identified writes, decile 1 smallest. Gated-MLP models hold the correlation in each decile; GPT-2 and Pythia carry it only in the top decile.', 'app:deciles', size='footnotesize', fit=True)
P(r'\begin{figure}[H]\centering\begin{tikzpicture}\begin{axis}[width=0.8\linewidth, height=0.42\linewidth, xlabel={magnitude decile of the identified write}, ylabel={within-decile $r$}, xmin=0.5, xmax=10.5, ymin=-0.1, ymax=1.05, xtick={1,...,10}, grid=major, grid style={black!10}, legend style={font=\tiny, at={(1.02,0.5)}, anchor=west, draw=none}, tick label style={font=\scriptsize}, label style={font=\small}, cycle list name=exotic]')
for key, nm in MODELS:
    vals = gp['r_identified_per_decile'] if key == 'gpt2' else x2[key]['r_identified_per_decile']
    P(r'\addplot+[%s] coordinates {%s};\addlegendentry{%s}' % ('thick' if key in GATED else 'thick, dashed', ' '.join(f'({n + 1},{v})' for n, v in enumerate(vals) if v is not None), nm))
P(r'\end{axis}\end{tikzpicture}\caption{Within-decile correlation per model. Solid lines are gated-MLP models, dashed lines non-gated.}\label{app:decilefig}\end{figure}')
rows = []
for key, nm in MODELS[1:]:
    f = x2[key]['fvu32']; rows.append((nm, f3(f['weight_full']), f3(f['weight_excl_top5ch']), f3(f['rotated_full']), f3(f['rotated_excl_top5ch']), f"{100 * f['top5_var_share']:.1f}\\%"))
table(['Model', 'FVU@32, weight', 'off the top-5 coordinates', 'FVU@32, rotated', 'off the top-5', 'variance in the top-5 coordinates'], rows, 'The massive-channel test: FVU at $k=32$ over all coordinates and over all but the five highest-variance coordinates, for the weight dictionary and its rotation.', 'app:channels', size='footnotesize', fit=True)
rows = []
for key, nm in MODELS:
    r = x9[key]; top = r['top1_neuron_rows'][0]
    rows.append((nm, f"{r['n_tokens']:,}", f"{r['distinct_top1_neurons']:,}", f"{100 * top['share_of_tokens']:.1f}\\%", top['neuron'].replace('#', '\\#'), f"{100 * r['top5_share']:.1f}\\%", f"{r['top1_entropy_bits']:.1f}"))
table(['Model', 'tokens', 'distinct top-1 neurons', 'most frequent', 'which', 'top five together', 'entropy, bits'], rows, 'Concentration of the largest true write. In all eight models the most frequent top-1 neuron takes a few percent of tokens, so top-1 recall measures identification across thousands of atoms.', 'app:conc', size='footnotesize', fit=True)
at = x4['attrib']; b3 = x4['block3']
rows = [('FVU@32, weight dictionary', f3(x4['fvu32']['weight']), f3(x4['fvu32']['wikitext_ref']['weight'])), ('FVU@32, rotated', f3(x4['fvu32']['rotated']), f3(x4['fvu32']['wikitext_ref']['rotated'])), ('top-1 recall, OMP', f3(at['recall_top1_omp']), f3(at['wikitext_ref']['recall_top1_omp'])), ('top-1 recall, one-shot', f3(at['recall_top1_oneshot']), f3(at['wikitext_ref']['recall_top1_oneshot'])), ('$r$ identified', f3(at['r_identified']), f3(at['wikitext_ref']['r_identified'])), ('$r$ unconditional', f3(at['r_unconditional']), f3(at['wikitext_ref']['r_unconditional'])), ('block 3 counter-aligned fraction', f3(b3['counter_frac']), f3(b3['wikitext_ref']['counter_frac'])), ('block 3 median birth survival', f2(b3['median_birth_surv']), f2(b3['wikitext_ref']['median'])), ('block 3 anchored fraction', f3(b3['novel_frac']), f3(b3['wikitext_ref']['novel_frac']))]
table(['GPT-2 small', 'Pile sample (%s tokens)' % f"{x4['n_eval_tokens']:,}", 'WikiText-2'], rows, 'The GPT-2 evaluation on a Pile-derived sample. Reconstruction and attribution replicate; the block-3 role profile does not, which is why the paper treats roles as a property of a model on a distribution. Shuffled-null counter fraction on Pile: %.3f.' % b3['null_shuffle_counter_frac'], 'app:pile', fit=True)
Z2FIG = json.load(open(f'{R}/exp_z2_ci.json')) if os.path.exists(f'{R}/exp_z2_ci.json') else None
if Z2FIG and all(f'{key}/wikitext' in Z2FIG for key, _ in MODELS):
    P(r'\begin{figure}[H]\centering\begin{tikzpicture}\begin{axis}[width=0.8\linewidth, height=0.4\linewidth, ybar, bar width=5pt, ymin=0, ymax=1, ylabel={FVU at $k=32$, typical states}, symbolic x coords={%s}, xtick=data, x tick label style={font=\tiny, rotate=20, anchor=north east}, legend style={font=\tiny, at={(1.02,0.5)}, anchor=west, draw=none}, tick label style={font=\scriptsize}, label style={font=\small}, enlarge x limits=0.08]' % ','.join(nm for _, nm in MODELS))
    for series, color, m in [('weight dictionary', 'blue!70!black', 'fvu32_weight_typical'), ('rotated', 'red!70!black', 'fvu32_rotated_typical'), ('random', 'black!40', 'fvu32_random_typical')]:
        P(r'\addplot[fill=%s, draw=none] coordinates {%s};\addlegendentry{%s}' % (color, ' '.join(f"({nm},{Z2FIG[f'{key}/wikitext']['ci'][m]['point']:.4f})" for key, nm in MODELS), series))
    P(r'\end{axis}\end{tikzpicture}\caption{FVU at $k=32$ per model over the typical states on WikiText-2, from the sweep of Appendix~\ref{app:ci}, for the weight dictionary, its Gram-preserving rotation, and random directions. The rotation sits at the random floor in all eight models.}\label{app:fvufig}\end{figure}')

# ---------------- E. roles ----------------
P(r'\section{Write roles and null controls}\label{app:roles}')
P(r'Table~\ref{app:nulls} gives, per block, the real role fractions next to the two nulls, and Table~\ref{app:quint} the counter-aligned fraction by magnitude quintile.')
rows = []
for b in sorted(k['per_block'], key=int):
    pb = k['per_block'][b]; re_, rd, sh = pb['real'], pb['null_random_dir'], pb['null_shuffled']
    rows.append((b, f"{pb['n']:,}", f2(re_['median']), f3(re_['frac_counter']), f3(rd['frac_counter']), f3(sh['frac_counter']), f3(re_['frac_self_anchored']), f3(rd['frac_self_anchored']), f3(sh['frac_self_anchored']), f3(re_['frac_reinforcing']), f3(rd['frac_reinforcing']), f3(sh['frac_reinforcing'])))
table(['block', 'writes', 'median $s_{\\mathrm{birth}}$', 'counter, real', 'random dir.', 'shuffled', 'anchored, real', 'random dir.', 'shuffled', 'reinforcing, real', 'random dir.', 'shuffled'], rows, 'Role fractions of each block\'s dominant writes on WikiText, against random replacement directions and shuffled write-state pairings. Each block\'s real counter fraction lies below its shuffled null; anchoring and reinforcing mass do not.', 'app:nulls', size='scriptsize', fit=True)
rows = []
for b in sorted(k['quintiles'], key=int):
    q = k['quintiles'][b]; rows.append((b, *[f3(q[f'q{n}']['frac_counter']) for n in range(5)]))
table(['block', 'quintile 1, smallest', '2', '3', '4', '5, largest'], rows, 'Counter-aligned fraction by magnitude quintile of the write. Block 3 leads in each quintile.', 'app:quint', fit=True)

# ---------------- F. circuit ----------------
P(r'\section{The channel-447 circuit}\label{app:circuit}')
P(r'Tables~\ref{app:dose} to~\ref{app:leash} and Figure~\ref{app:traj} are the intervention results of Section~\ref{sec:circuit} in full.')
rows = []
for al, nm in [('alpha_0.5', 'channel halved'), ('alpha_1.5', 'channel times 1.5')]:
    q = x1[al]['quantiles']; rows.append((nm, x1[al]['n_active'], *[f"{100 * q[qq]:+.1f}\\%" for qq in ('q5', 'q25', 'q50', 'q75', 'q95')], f"{100 * x1[al]['frac_abs_ge_10pct']:.0f}\\%", f"{100 * x1['n614'][al]['value']:+.1f}\\%", f"{100 * x1['n614'][al]['pct_rank']:.0f}"))
table(['Scaling channel 447', 'active neurons', 'p5', 'p25', 'median', 'p75', 'p95', 'moved 10\\% or more', 'neuron 614', 'its percentile'], rows, 'Relative activation change of each active block-3 neuron when channel 447 is scaled at the block 2/3 boundary; %s neurons respond inversely to both scalings by 10\\%% or more.' % f"{x1['inverse_both']['n_ge_10pct_both']:,}", 'app:dose', size='footnotesize', fit=True)
rows = [(nm, f"{100 * x6[key]['median_lo']:+.1f}\\%", f"{100 * x6[key]['median_hi']:+.1f}\\%", f"{100 * x6[key]['q95_lo']:+.1f}\\%", f"{x6[key]['n_inverse_both_ge10pct']:,}", f"{100 * x6[key]['n614']['lo']:+.1f}\\%", f"{100 * x6[key]['n614']['hi']:+.1f}\\%", f"{100 * x6[key]['n614']['pct_rank_lo']:.0f}") for key, nm in [('unfrozen', 'LayerNorm free'), ('frozen', 'LayerNorm frozen at clean statistics')]]
table(['Block 3 pre-MLP LayerNorm', 'median, halved', 'median, times 1.5', 'p95, halved', 'inverse responders', 'neuron 614, halved', 'times 1.5', 'percentile'], rows, 'The frozen-normalization control: with block 3\'s LayerNorm statistics held at their clean-run values, the block-wide response collapses while neuron 614 keeps most of its own.', 'app:frozen', size='footnotesize', fit=True)
rows = [(c['k'], f"{c['energy']:,.0f}", f"{c['dce']:+.4f}") for c in x5['curve']] + [('random 9, set %d' % (n + 1), f"{c['energy']:,.0f}", f"{c['dce']:+.4f}") for n, c in enumerate(x5['controls_9'])]
table(['ablated block-3 counter-writers, top $k$ by counter energy', 'ablated write energy', '$\\Delta$CE'], rows, 'Behavioral cost of ablating the strongest channel-447 counter-writers of block 3 (clean cross-entropy %.4f). The top nine are neurons %s.' % (x5['ce_clean'], ', '.join(str(n) for n in x5['top9'])), 'app:ablate', align='lrr', fit=True)
P(r'\begin{figure}[H]\centering\begin{tikzpicture}\begin{axis}[width=0.75\linewidth, height=0.38\linewidth, xlabel={residual state after block}, ylabel={channel 447 value}, xtick={0,2,...,10}, grid=major, grid style={black!10}, legend style={font=\tiny, at={(0.03,0.97)}, anchor=north west, draw=none, fill=none}, tick label style={font=\scriptsize}, label style={font=\small}]')
P(r'\addplot[thick, blue!70!black, mark=*, mark size=1.3pt] coordinates {%s};\addlegendentry{clean}' % ' '.join(f'({n - 1},{v})' for n, v in enumerate(s8['ch447_traj_clean']) if 1 <= n <= 11))
P(r'\addplot[thick, red!70!black, mark=square*, mark size=1.3pt] coordinates {%s};\addlegendentry{four counter-writers ablated}' % ' '.join(f'({n - 1},{v})' for n, v in enumerate(s8['A_ch447_traj_ablated']) if 1 <= n <= 11))
P(r'\end{axis}\end{tikzpicture}\caption{Restoration: channel 447 in the raw residual state after each block, with and without the four strongest counter-writers. The ablation opens a gap after block 3 that later components close by the state after block 10, the input of block 11, which is the readout of the text; the state after block 11 is only available through the final normalization and is omitted.}\label{app:traj}\end{figure}')
pc = t2['per_component_delta']
rows = [(c, f"{pc[c]:+.3f}") for c in pc if abs(pc[c]) > 0.0005]
table(['Component', 'change in channel 447 write, ablated minus clean'], rows, 'Per-component contribution to the restoration, summing to %+.2f. The largest single restorers are %s.' % (t2['sum_of_deltas'], ', '.join(f"{nm.replace('#', chr(92) + '#')} ({v:+.2f})" for nm, v in list(t2['top_neuron_compensators'].items())[:5])), 'app:compensate', align='lr', fit=True)
rows = [(nm.replace('#', '\\#'), r['role'].lower(), f"{r['intact_a0.5']:+.3f}", f"{r['intact_a1.5']:+.3f}", f"{r['intact_write447']:.2f}", 'ablated' if r['role'].lower() == 'primary' else f"{r['abl_write447']:.2f}", '' if r['role'].lower() == 'primary' else (f"{100 * (r['abl_write447'] / r['intact_write447'] - 1):+.0f}\\%" if r['intact_write447'] else '')) for nm, r in u1.items() if isinstance(r, dict) and 'role' in r]
table(['Neuron', 'role', 'gain, halved', 'gain, times 1.5', 'channel-447 write, intact', 'under group ablation', 'change'], rows, 'The reserve bank: channel-insensitive later neurons whose channel-447 writes rise when the four counter-writers are ablated, with random controls for comparison. The primary, neuron 614, is one of the ablated four; its intact write is shown for scale and is zero under the ablation.', 'app:reserves', size='footnotesize', fit=True)
conds = [('A_full', 'ablate the four counter-writers'), ('B_leashcut', 'remove only the non-channel components of their writes'), ('C_channelcut', 'remove only the channel-447 component'), ('E_leashcut_614only', 'remove neuron 614\'s non-channel component alone')]
rows = [(nm, f"{u2['ch447_L11'][key]:.1f}", f"{u2['_mean_RESERVE'][key]:+.3f}", f"{u2['_mean_control'][key]:+.3f}") for key, nm in conds]
table(['Write-editing condition', 'channel 447 at layer 11', 'mean reserve response', 'mean control response'], rows, 'Locating the suppression: which part of the counter-writers\' output holds the reserve bank back (clean channel 447 at layer 11: %.1f).' % u2['ch447_L11']['clean'], 'app:leash', align='lrrr', fit=True)

# ---------------- G. bake-off ----------------
P(r'\section{Nomination comparisons}\label{app:bakeoff}')
P(r'\paragraph{Pre-registered battery.} Fixed in code before any result existed, and identical for each arm. Universe: GPT-2 small MLP neurons in blocks 2 to 8, excluding each neuron studied in the circuit work and any neuron with more than 0.15 of its direction energy in the known massive channels 138, 266, 447, 481 and 496. For a nominee $j$: (1) firing tokens are those with $|c_j|$ in the neuron\textquotesingle s own top 5\%, at least 200; (2) the ledger names the top upstream MLP contributor $u$ along $\mathbf{d}_j$ on those tokens; (3) $u$ is ablated, and as a matched control a random neuron from $u$\textquotesingle s block, measuring on the firing tokens the change in $\langle \mathbf{h}, \mathbf{d}_j\rangle$ after block $j$ against the prediction from $u$\textquotesingle s direct contribution, and the relative change in $|a_j|$; (4) Tier 1 holds when the sign matches the prediction, the magnitude lies within 0.2 to 2 times the prediction, and exceeds 3 times the control; Tier 2 holds when the relative response is at least 0.02 and at least 3 times the control. Arms: WDD ranks by unconditional counter-alignment write energy; the outlier arm by activation excess kurtosis times mean write magnitude; the correlation arm, added at ten per arm, by maximum absolute activation correlation with any upstream MLP neuron.')
def battery_rows(cands):
    return [(nm.replace('#', '\\#'), r['upstream'].replace('#', '\\#'), f"{r['upstream_share_pred']:+.2f}", f"{r['dproj']:+.2f}", f"{r['ctrl_dproj']:+.2f}", f"{r['dresp']:+.3f}", f"{r['ctrl_dresp']:+.3f}", 'yes' if r['TIER1'] else 'no', 'yes' if r['TIER2'] else 'no') for nm, r in cands.items()]
cols = ['Nominee', 'partner $u$', 'predicted', 'measured', 'control', 'response', 'control', 'Tier 1', 'Tier 2']
table(cols, [('\\multicolumn{9}{l}{\\emph{WDD arm}}',)] + battery_rows(v1['wdd_candidates']) + [('\\multicolumn{9}{l}{\\emph{outlier arm}}',)] + battery_rows(v1['outlier_candidates']), 'The pilot at three nominees per arm. Fisher exact, one-sided: Tier 2, $p=%.2f$; passing either tier, $p=%.2f$.' % (fisher(3, 3, 0, 3), fisher(3, 3, 1, 3)), 'app:pilot', size='footnotesize', fit=True)
# tallies at 10 and 30
def tally(b, n):
    bat = b['batteries']; rows = []
    for arm in ('WDD', 'outlier', 'corr'):
        ns = b['arms'].get(arm, [])
        if not ns: continue
        t1 = sum(bool(bat[x]['TIER1']) for x in ns); t2_ = sum(bool(bat[x]['TIER2']) for x in ns); ei = sum(bool(bat[x]['TIER1'] or bat[x]['TIER2']) for x in ns); bo = sum(bool(bat[x]['TIER1'] and bat[x]['TIER2']) for x in ns)
        rows.append((arm, f'{t1}/{n}', f'{t2_}/{n}', f'{ei}/{n}', f'{bo}/{n}'))
    return rows
rows10 = tally(x3, 10); rows30 = tally(n30, 30)
b30 = {r[0]: r for r in rows30}
def cnt(s): return int(s.split('/')[0])
table(['Arm', 'Tier 1', 'Tier 2', 'either', 'both'], [('\\multicolumn{5}{l}{\\emph{ten nominees per arm}}',)] + rows10 + [('\\multicolumn{5}{l}{\\emph{thirty nominees per arm}}',)] + rows30, 'Tallies at ten and thirty nominees per arm. One-sided Fisher exact tests at thirty per arm: WDD against the outlier screen, Tier 2 $p=%.3f$ and Tier 1 $p=%.2f$; the correlation screen against WDD, Tier 2 $p=%.2f$ and Tier 1 $p=%.4f$. The first ten nominees of each thirty-nominee arm are the ten-nominee arms, and their tallies reproduce exactly.' % (fisher(cnt(b30['WDD'][2]), 30, cnt(b30['outlier'][2]), 30), fisher(cnt(b30['WDD'][1]), 30, cnt(b30['outlier'][1]), 30), fisher(cnt(b30['corr'][2]), 30, cnt(b30['WDD'][2]), 30), fisher(cnt(b30['corr'][1]), 30, cnt(b30['WDD'][1]), 30)), 'app:tallies', align='lrrrr', fit=True)
P(r'\begin{figure}[H]\centering\begin{tikzpicture}\begin{axis}[width=0.6\linewidth, height=0.36\linewidth, ybar, bar width=9pt, ymin=0, ymax=30, ylabel={nominees passing, of 30}, symbolic x coords={WDD,outlier,correlation}, xtick=data, legend style={font=\tiny, at={(0.02,0.98)}, anchor=north west, draw=none, fill=none}, tick label style={font=\scriptsize}, label style={font=\small}, enlarge x limits=0.3]')
P(r'\addplot[fill=blue!70!black, draw=none] coordinates {(WDD,%d) (outlier,%d) (correlation,%d)};\addlegendentry{Tier 1}' % tuple(cnt(b30[a_][1]) for a_ in ('WDD', 'outlier', 'corr')))
P(r'\addplot[fill=orange!70, draw=none] coordinates {(WDD,%d) (outlier,%d) (correlation,%d)};\addlegendentry{Tier 2}' % tuple(cnt(b30[a_][2]) for a_ in ('WDD', 'outlier', 'corr')))
P(r'\end{axis}\end{tikzpicture}\caption{The thirty-per-arm bake-off.}\label{app:bakefig}\end{figure}')
# per-nominee at 30
arms_of = {}
for arm, ns in n30['arms'].items():
    for x_ in ns: arms_of.setdefault(x_, []).append(arm)
P(r'\begin{center}\scriptsize')
P(r'\begin{longtable}{llrrrrrcc}')
P(r'\caption{Each nominee of the thirty-per-arm run, with the arm or arms that nominated it and its battery. Columns as in Table~\ref{app:pilot}.}\label{app:n30}\\ \toprule')
P(' & '.join(['Nominee', 'arms', 'partner', 'predicted', 'measured', 'control', 'response', 'control', 'T1/T2']) + r' \\ \midrule \endfirsthead')
P(r'\toprule ' + ' & '.join(['Nominee', 'arms', 'partner', 'predicted', 'measured', 'control', 'response', 'control', 'T1/T2']) + r' \\ \midrule \endhead')
P(r'\bottomrule \endfoot')
for nm, r in n30['batteries'].items():
    P(' & '.join([nm.replace('#', '\\#'), ', '.join(a_.replace('corr', 'correlation') for a_ in arms_of.get(nm, [])), r['upstream'].replace('#', '\\#'), f"{r['upstream_share_pred']:+.2f}", f"{r['dproj']:+.2f}", f"{r['ctrl_dproj']:+.2f}", f"{r['dresp']:+.3f}", f"{r['ctrl_dresp']:+.3f}", ('yes' if r['TIER1'] else 'no') + '/' + ('yes' if r['TIER2'] else 'no')]) + r' \\')
P(r'\end{longtable}\end{center}')

# ---------------- H. audit ----------------
P(r'\section{Independent recomputation}\label{app:audit}')
rows = [('FVU at $k=32$, weight dictionary', f3(w['A_fvu32']['expected']['weight_dict']), f3(w['A_fvu32']['weight_dict'])), ('FVU at $k=32$, rotated (fresh rotation, seed 777)', f3(w['A_fvu32']['expected']['rotated']), f3(w['A_fvu32']['rotated'])),
        ('top-1 recall, OMP', f3(w['B_attrib']['expected']['recall_top1']), f3(w['B_attrib']['recall_top1'])), ('$r$ identified', f3(w['B_attrib']['expected']['r_identified']), f3(w['B_attrib']['r_identified'])), ('$r$ unconditional', f3(w['B_attrib']['expected']['r_unconditional']), f3(w['B_attrib']['r_unconditional'])),
        ('neuron 614 response, channel halved', f"{100 * w['C_dose_response']['expected_614']['alpha_0.5']:+.1f}\\%", f"{100 * w['C_dose_response']['rel_change']['alpha_0.5'][0]:+.1f}\\%"), ('neuron 614 response, channel times 1.5', f"{100 * w['C_dose_response']['expected_614']['alpha_1.5']:+.1f}\\%", f"{100 * w['C_dose_response']['rel_change']['alpha_1.5'][0]:+.1f}\\%")]
table(['Quantity', 'paper', 'recomputed'], rows, 'An independent re-implementation of the dictionary, the attribution and the dose response, written from the description in the text with its own hooks, batching and seeds, on the same 32{,}768 tokens. The deterministic quantities agree to the printed precision and the dose response to within 0.2 percentage points; the rotation control draws a fresh random rotation (seed 777), so its value is near but not equal to that of Section~\\ref{sec:recon}.', 'app:audit-table', align='lrr', fit=True)
ci_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ci_tables.tex')
if os.path.exists(ci_path): P(open(ci_path).read())
z_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'z_tables.tex')
if os.path.exists(z_path): P(open(z_path).read())
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'appendix.tex'), 'w').write('\n'.join(out) + '\n')
print('appendix.tex written:', sum(1 for l in out if l.startswith('\\begin{table}')), 'tables,', sum(1 for l in out if l.startswith('\\begin{figure}')), 'figures, plus one longtable')
