"""Turn results/exp_z2_ci.json into: the rows of Table 1 (with 95% intervals), ci_tables.tex for the appendix
(per model and corpus: reconstruction, attribution, deciles; old-run comparison), and the decile figure with error bars."""
import json, os, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results')
Z = json.load(open(f'{R}/exp_z2_ci.json'))
S9 = json.load(open(f'{R}/exp_s9_families.json')); X2 = json.load(open(f'{R}/exp_x2_families_attrib.json')); W = json.load(open(f'{R}/exp_w_audit.json'))
MODELS = [('gpt2', 'GPT-2 small'), ('smollm2-135m', 'SmolLM2-135M'), ('qwen2.5-0.5b', 'Qwen2.5-0.5B'), ('pythia-410m', 'Pythia-410m'), ('olmo-1b-0724', 'OLMo-1B'), ('qwen2.5-7b', 'Qwen2.5-7B'), ('llama-7b', 'Llama-7B'), ('pythia-6.9b', 'Pythia-6.9B')]
GATED = {'smollm2-135m', 'qwen2.5-0.5b', 'olmo-1b-0724', 'qwen2.5-7b', 'llama-7b'}
COLORS = {'gpt2': 'black!60', 'smollm2-135m': 'orange!90!black', 'qwen2.5-0.5b': 'green!60!black', 'pythia-410m': 'gray', 'olmo-1b-0724': 'teal', 'qwen2.5-7b': 'red!70!black', 'llama-7b': 'blue!70!black', 'pythia-6.9b': 'violet'}
def ci(key, corpus, m, nd=3, c=None):
    c = c if c is not None else Z.get(f'{key}/{corpus}', {}).get('ci', {}).get(m)
    if not c or c['point'] is None: return ''
    return f"{c['point']:.{nd}f} [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}]"
def pt(key, corpus, m, nd=3):
    c = Z.get(f'{key}/{corpus}', {}).get('ci', {}).get(m); return '' if not c or c['point'] is None else f"{c['point']:.{nd}f}"
out = []
P = out.append
def table(cols, rows, caption, label, align=None, size='footnotesize', fit=True):
    align = align or ('l' + 'r' * (len(cols) - 1))
    P(r'\begin{table}[H]\centering\%s' % size); P(r'\caption{%s}\label{%s}' % (caption, label))
    if fit: P(r'\begin{adjustbox}{max width=\linewidth}')
    P(r'\begin{tabular}{%s}\toprule' % align); P(' & '.join(cols) + r' \\ \midrule')
    for r in rows: P(' & '.join(str(c) for c in r) + r' \\')
    P(r'\bottomrule\end{tabular}' + (r'\end{adjustbox}' if fit else '')); P(r'\end{table}')
# ---- Table 1 rows (main text): WikiText, with intervals ----
t1 = []
for key, nm in MODELS:
    if f'{key}/wikitext' not in Z: continue
    t1.append(f"{nm} & {ci(key,'wikitext','fvu32_weight')} & {pt(key,'wikitext','fvu64_weight')} & {pt(key,'wikitext','fvu32_rotated')} & {ci(key,'wikitext','r_identified')} & {ci(key,'wikitext','recall_top1_omp')} \\\\")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'table1_rows.tex'), 'w').write('\n'.join(t1) + '\n')
# ---- appendix: per corpus, reconstruction and attribution with intervals ----
P(r'\section{Confidence intervals and the second corpus}\label{app:ci}')
P(r'All numbers here come from \texttt{exp\_z2\_ci}, an implementation of the family sweep written from the description in Sections~\ref{sec:method} and~\ref{sec:setup} independently of the individual experiment scripts, run under one protocol for all eight models: 32{,}768 evaluation states per model and corpus, the same three dictionaries, OMP with $k=64$ read at 32 and 64, one-shot attribution, and a bootstrap over the 64 evaluation sequences with 1{,}000 resamples. Brackets are 95\% percentile intervals. GPT-2 keeps the conventions of Section~\ref{sec:setup} (a disjoint centering slice for reconstruction, the evaluation mean for attribution); the other families center on eight disjoint chunks throughout. Table~\ref{app:ci-old} sets its point estimates beside those of the individual experiment scripts.')
for corpus, cname in [('wikitext', 'WikiText-2'), ('pile', 'the Pile sample')]:
    rows = []
    for key, nm in MODELS:
        if f'{key}/{corpus}' not in Z: continue
        rows.append((nm, ci(key, corpus, 'fvu32_weight_typical'), ci(key, corpus, 'fvu64_weight_typical'), ci(key, corpus, 'fvu32_rotated_typical'), pt(key, corpus, 'fvu64_rotated_typical'), ci(key, corpus, 'fvu32_random_typical'), pt(key, corpus, 'fvu32_weight_offtop5_typical', 2), pt(key, corpus, 'fvu32_rotated_offtop5_typical', 2), f"{100 * Z[f'{key}/{corpus}']['top5_var_share']:.1f}\\%"))
    if rows: table(['Model', 'FVU@32 weight', 'FVU@64 weight', 'FVU@32 rotated', 'FVU@64 rotated', 'FVU@32 random', 'off top-5, weight', 'off top-5, rotated', 'top-5 share'], rows, 'Reconstruction of the typical states on %s with 95\\%% bootstrap intervals over sequences. Off top-5 removes the five highest-variance coordinates from numerator and denominator.' % cname, f'app:ci-recon-{corpus}', size='scriptsize')
    rows = []
    for key, nm in MODELS:
        if f'{key}/{corpus}' not in Z: continue
        e = Z[f'{key}/{corpus}']
        rows.append((nm, f"{e['n_sink']:,}", f"{e['sink_norm_over_median']:.0f}$\\times$", f"{100 * e['sink_variance_share']:.1f}\\%", ci(key, corpus, 'fvu32_weight_sink', 4), pt(key, corpus, 'fvu32_rotated_sink'), pt(key, corpus, 'fvu32_random_sink'), pt(key, corpus, 'fvu32_weight'), pt(key, corpus, 'fvu32_rotated')))
    if rows: table(['Model', 'sink states', 'smallest sink norm over median', 'share of variance', 'sink FVU@32 weight', 'rotated', 'random', 'all-state FVU@32 weight', 'rotated'], rows, 'The sink states on %s: how many of the 32{,}768 states exceed ten times the median norm, the variance they carry, and their reconstruction. The all-state columns are what an FVU over all states reports.' % cname, f'app:ci-sink-{corpus}', size='scriptsize')
    rows = []
    for key, nm in MODELS:
        if f'{key}/{corpus}' not in Z: continue
        cal = Z[f'{key}/{corpus}']['calibration']
        rows.append((nm, 'yes' if key in GATED else 'no', ci(key, corpus, 'recall_top1_omp'), ci(key, corpus, 'recall_top1_oneshot'), ci(key, corpus, 'r_identified'), f"{cal['r_identified_typical']:.3f}", ci(key, corpus, 'r_unconditional'), f"{cal['r_unconditional_typical']:.3f}", f"{Z[f'{key}/{corpus}']['n_identified']:,}"))
    if rows: table(['Model', 'gated', 'top-1 recall, OMP', 'top-1 recall, one-shot', '$r$ identified, all states', 'typical states', '$r$ unconditional, all', 'typical', 'identified writes'], rows, 'Attribution on %s with 95\\%% bootstrap intervals over sequences. The all-state correlations are carried by the few identified writes at the sink states, whose magnitudes dominate the statistic; the typical-state columns exclude them, and Table~\\ref{app:ci-calib-%s} is the accuracy statement.' % (cname, corpus), f'app:ci-attrib-{corpus}', size='scriptsize')
    rows = []
    for key, nm in MODELS:
        if f'{key}/{corpus}' not in Z: continue
        c = Z[f'{key}/{corpus}']['calibration']
        rows.append((nm, f"{c['n']:,}", f"{100 * c['share_negative']:.0f}\\%", f"{100 * c['sign_agreement']['point']:.1f}\\%", ci(key, corpus, None, 2, c['median_rel_error']), f"{100 * c['within_25']['point']:.1f}\\%", *[f"{c['deciles'][d]['median_ratio']:.2f}" for d in (0, 4, 9)], *[f"{c['deciles'][d]['median_rel_error']:.2f}" for d in (0, 4, 9)]))
    if rows: table(['Model', 'identified writes', 'negative', 'sign recovered', 'median relative error', 'within 25\\%', 'ratio, decile 1', '5', '10', 'rel. error, decile 1', '5', '10'], rows, 'Calibration of the recovered coefficients of identified dominant writes on %s, typical states. Ratio is recovered over true; a ratio above one is over-estimation.' % cname, f'app:ci-calib-{corpus}', size='scriptsize')
    rows = []
    for key, nm in MODELS:
        if f'{key}/{corpus}' not in Z: continue
        rows.append((nm, *[pt(key, corpus, f'r_decile_{d}', 2) for d in range(1, 11)]))
    if rows: table(['Model'] + [str(d) for d in range(1, 11)], rows, 'Within-decile correlation of recovered and true coefficients on %s (point estimates; intervals in Figure~\\ref{app:ci-decilefig-%s}). In the gated models about half of the writes are negative, so this signed correlation within a magnitude decile mostly records sign recovery, and Table~\\ref{app:ci-calib-%s} is the accuracy statement.' % (cname, corpus, corpus), f'app:ci-deciles-{corpus}', size='scriptsize')
    # decile figure with error bars
    P(r'\begin{figure}[H]\centering\begin{tikzpicture}\begin{axis}[width=0.8\linewidth, height=0.42\linewidth, xlabel={magnitude decile of the identified write}, ylabel={within-decile $r$}, xmin=0.5, xmax=10.5, ymin=-0.2, ymax=1.05, xtick={1,...,10}, grid=major, grid style={black!10}, legend style={font=\tiny, at={(1.02,0.5)}, anchor=west, draw=none}, tick label style={font=\scriptsize}, label style={font=\small}, error bars/y dir=both, error bars/y explicit]')
    for key, nm in MODELS:
        if f'{key}/{corpus}' not in Z: continue
        c = Z[f'{key}/{corpus}']['ci']; pts = []
        for d in range(1, 11):
            v = c.get(f'r_decile_{d}')
            if v and v['point'] is not None: pts.append(f"({d},{v['point']:.4f}) +- (0,{max(v['point'] - v['lo'], v['hi'] - v['point']):.4f})")
        P(r'\addplot[%s, color=%s, mark=*, mark size=1pt, error bars/error bar style={opacity=0.5}] coordinates {%s};\addlegendentry{%s}' % ('thick' if key in GATED else 'thick, dashed', COLORS[key], ' '.join(pts), nm))
    P(r'\end{axis}\end{tikzpicture}\caption{Within-decile correlation per model on %s, with 95\%% bootstrap intervals as error bars. Solid lines are gated-MLP models, dashed non-gated.}\label{app:ci-decilefig-%s}\end{figure}' % (cname, corpus))
# ---- comparison with the original runs ----
rows = []
for key, nm in MODELS:
    if f'{key}/wikitext' not in Z: continue
    if key == 'gpt2':
        old = ('0.069', '0.647', '0.998', '0.648')
    else:
        old = (f"{S9[key]['fvu']['weight']['32']:.3f}", f"{X2[key]['recall_top1_omp']:.3f}", f"{X2[key]['r_identified']:.3f}", f"{X2[key]['r_unconditional']:.3f}")
    rows.append((nm, old[0], ci(key, 'wikitext', 'fvu32_weight'), old[1], ci(key, 'wikitext', 'recall_top1_omp'), old[2], ci(key, 'wikitext', 'r_identified'), old[3], ci(key, 'wikitext', 'r_unconditional')))
table(['Model', 'FVU@32, scripts', 'sweep', 'recall, scripts', 'sweep', '$r$ id., scripts', 'sweep', '$r$ unc., scripts', 'sweep'], rows, 'The individual experiment scripts (8{,}192 states for the non-GPT-2 models, corpus-restricted embedding atoms) against the single-protocol sweep on WikiText-2 (32{,}768 states, full-vocabulary dictionaries), both over all states. Agreement within the intervals is the replication check; the remaining differences come from the token sample and the dictionary size.', 'app:ci-old', size='scriptsize')
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ci_tables.tex'), 'w').write('\n'.join(out) + '\n')
print('table1_rows.tex and ci_tables.tex written for', len([k for k in Z]), 'model/corpus entries')
