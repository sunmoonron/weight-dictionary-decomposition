"""Turn the Z-series results (exp_z3_ceiling, exp_z4_sae, exp_z5_kcurve, exp_z2_fp32) into z_tables.tex:
the observability ceiling in all eight models, the matched SAE comparison, sparsity at matched budgets, and the precision control."""
import json, os
import numpy as np
R = os.path.expanduser(os.environ.get('WDD_RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results')))
def load(n):
    p = os.path.join(R, n); return json.load(open(p)) if os.path.exists(p) else None
Z3, Z4, Z5, F32 = load('exp_z3_ceiling.json'), load('exp_z4_sae.json'), load('exp_z5_kcurve.json'), load('exp_z2_fp32.json')
MODELS = [('gpt2', 'GPT-2 small'), ('smollm2-135m', 'SmolLM2-135M'), ('qwen2.5-0.5b', 'Qwen2.5-0.5B'), ('pythia-410m', 'Pythia-410m'), ('olmo-1b-0724', 'OLMo-1B'), ('qwen2.5-7b', 'Qwen2.5-7B'), ('llama-7b', 'Llama-7B'), ('pythia-6.9b', 'Pythia-6.9B')]
GATED = {'smollm2-135m', 'qwen2.5-0.5b', 'olmo-1b-0724', 'qwen2.5-7b', 'llama-7b'}
COLORS = {'gpt2': 'black!60', 'smollm2-135m': 'orange!90!black', 'qwen2.5-0.5b': 'green!60!black', 'pythia-410m': 'gray', 'olmo-1b-0724': 'teal', 'qwen2.5-7b': 'red!70!black', 'llama-7b': 'blue!70!black', 'pythia-6.9b': 'violet'}
out = []; P = out.append
def ci(v, nd=3):
    if v is None or v[0] is None: return ''
    return f"{v[0]:.{nd}f} [{v[1]:.{nd}f}, {v[2]:.{nd}f}]"
def pt(v, nd=3): return '' if v is None or v[0] is None else f"{v[0]:.{nd}f}"
def table(cols, rows, caption, label, align=None, size='scriptsize'):
    align = align or ('l' + 'r' * (len(cols) - 1))
    P(r'\begin{table}[H]\centering\%s' % size); P(r'\caption{%s}\label{%s}' % (caption, label)); P(r'\begin{adjustbox}{max width=\linewidth}')
    P(r'\begin{tabular}{%s}\toprule' % align); P(' & '.join(cols) + r' \\ \midrule')
    for r in rows: P(' & '.join(str(c) for c in r) + r' \\')
    P(r'\bottomrule\end{tabular}\end{adjustbox}'); P(r'\end{table}')
P(r'\section{The Ceiling in Each Model, the Matched SAE Comparison and Sparsity at Matched Budgets}\label{app:z}')
P(r'This section uses the states, dictionaries, centering and bootstrap of Appendix~\ref{app:ci}.')
# ------------------------------------------------------------------ Z3: the ceiling
if Z3:
    P(r'\subsection{Identification and survival in all models}\label{app:z3}')
    P(r"""\texttt{exp\_z3\_ceiling} repeats the GPT-2 analysis of Table~\ref{app:survbins} and Table~\ref{app:auc} for the eight models on both corpora: each token's three dominant MLP writes (98,304 cases per model and corpus), their survival $s=(\mathbf{x}\cdot\mathbf{d})/c$ in the observed state, and whether OMP at $k=64$ places the write's atom in the support. Two further quantities decompose the misses. A miss is \emph{erased} if $s<0.25$, \emph{weakened} if $0.25\le s<0.75$, a \emph{twin} miss if the write is intact but an atom within $|\cos|\ge0.95$ of its atom sits in the support instead, and \emph{intact, unselected} otherwise. The GPT-2 row reproduces Table~\ref{app:survbins} exactly.""")
    for corpus, cname in [('wikitext', 'WikiText-2'), ('pile', 'the Pile sample')]:
        rows = []
        for key, nm in MODELS:
            e = Z3.get(f'{key}/{corpus}')
            if not e: continue
            b = e['by_survival']
            rows.append((nm, 'yes' if key in GATED else 'no', ci(e['id_rate_omp']), pt(e['punchline']['frac_intact']), ci(e['punchline']['id_rate_intact']), pt(e['punchline']['frac_erased']), ci(e['punchline']['id_rate_erased']), f"{e['auc_omp']['surv']:.2f}", f"{e['auc_omp']['prom']:.2f}"))
        if rows: table(['Model', 'gated', 'identified, all cases', 'share intact', 'identified, intact', 'share erased', 'identified, erased', 'AUC survival', 'AUC prominence'], rows, 'Identification of the three dominant writes on %s against their survival. Intact means $s\\ge0.75$, erased $s<0.25$. Brackets are 95\\%% bootstrap intervals over sequences.' % cname, f'app:z3-surv-{corpus}')
        rows = []
        for key, nm in MODELS:
            e = Z3.get(f'{key}/{corpus}')
            if not e: continue
            m = e['miss_breakdown']
            rows.append((nm, pt(m['miss_rate']), ci(m['erased'], 2), ci(m['weak'], 2), ci(m['intact_twin'], 2), ci(m['intact_other'], 2), f"{100 * e['twin']['share_of_cases_with_twin_in_support'][0]:.1f}\\%"))
        if rows: table(['Model', 'miss rate', 'erased', 'weakened', 'twin in support', 'intact, unselected', 'cases with a twin in the support'], rows, 'Where the misses come from on %s: the share of missed dominant writes in each class (the four shares sum to one).' % cname, f'app:z3-miss-{corpus}')
        rows = []
        for key, nm in MODELS:
            e = Z3.get(f'{key}/{corpus}')
            if not e: continue
            rows.append((nm, *[pt(r['id_omp'], 2) for r in e['by_survival']]))
        if rows: table(['Model', '$s<0$', '$[0,0.25)$', '$[0.25,0.5)$', '$[0.5,0.75)$', '$[0.75,1.25)$', '$s\\ge1.25$'], rows, 'OMP identification rate by survival bin on %s (the six bins of Figure~\\ref{fig:fvu}).' % cname, f'app:z3-bins-{corpus}')
        if rows:
            P(r'\begin{figure}[H]\centering\begin{tikzpicture}\begin{axis}[width=0.8\linewidth, height=0.42\linewidth, xlabel={survival bin of the dominant write}, ylabel={identified by OMP}, xmin=0.5, xmax=6.5, ymin=0, ymax=1.02, xtick={1,...,6}, xticklabels={{$s<0$},{$[0,.25)$},{$[.25,.5)$},{$[.5,.75)$},{$[.75,1.25)$},{$s\ge1.25$}}, grid=major, grid style={black!10}, legend style={font=\tiny, at={(1.02,0.5)}, anchor=west, draw=none}, tick label style={font=\scriptsize}, label style={font=\small}, error bars/y dir=both, error bars/y explicit]')
            for key, nm in MODELS:
                e = Z3.get(f'{key}/{corpus}')
                if not e: continue
                pts = ' '.join(f"({i + 1},{r['id_omp'][0]:.4f}) +- (0,{max(r['id_omp'][0] - r['id_omp'][1], r['id_omp'][2] - r['id_omp'][0]):.4f})" for i, r in enumerate(e['by_survival']) if r['id_omp'][0] is not None)
                P(r'\addplot[%s, color=%s, mark=*, mark size=1pt, error bars/error bar style={opacity=0.5}] coordinates {%s};\addlegendentry{%s}' % ('thick' if key in GATED else 'thick, dashed', COLORS[key], pts, nm))
            P(r'\end{axis}\end{tikzpicture}\caption{Identification against survival for all eight models on %s. Solid lines are gated-MLP models, dashed non-gated.}\label{app:z3-fig-%s}\end{figure}' % (cname, corpus))
    rows = []
    for key, nm in MODELS:
        e = Z3.get(f'{key}/wikitext')
        if not e: continue
        c = e['support_composition']
        rows.append((nm, *[f"{100 * c[t]:.1f}\\%" for t in ('embedding', 'mlp', 'attention', 'bias')]))
    if rows: table(['Model', 'embedding', 'MLP', 'attention', 'bias'], rows, 'Composition of the $k=64$ support by atom type on WikiText-2.', 'app:z3-support')
# ------------------------------------------------------------------ Z4: the SAE
if Z4:
    P(r'\subsection{The SAE comparison on identical states}\label{app:z4}')
    P(r"""\texttt{exp\_z4\_sae} evaluates the residual-stream SAEs of \citet{bloom2024} from their published weights and the weight dictionary on the same GPT-2 states in four cells: WikiText-2 with 512-token contexts (the paper's states) and OpenWebText with 128-token contexts and a leading beginning-of-text token (the SAEs' training convention), each at the state after block 5 (the input of block 6, where the SAE named \texttt{blocks.6} was trained) and after block 6 (the paper's layer, where \texttt{blocks.7} was trained). The SAEs read the residual stream in the TransformerLens frame, in which each state is centered over the model dimension, and the states are centered the same way before encoding. $L_0$ is the SAE's mean number of active features over the typical states; the weight dictionary is run at $k=32$, $k=64$ and at $k$ equal to that $L_0$. FVU is over the typical states with one denominator per cell for both methods (for GPT-2 the sink states are exactly the first positions), and $\Delta$CE is the change in next-token cross-entropy over all positions when the reconstruction replaces the state in the forward pass. On the 512-token states the SAE fails past position 127, which it never saw in training, so the two 128-token cells are the matched comparison.""")
    rows = []
    for cell, cname in [('wikitext512', 'WikiText-2, 512'), ('owt128', 'OpenWebText, 128')]:
        for L in ('L5', 'L6'):
            e = Z4.get(cell, {}).get(L)
            if not e: continue
            sae = e['sae']; lname = 'after block 5' if L == 'L5' else 'after block 6'
            rows.append((cname, lname, f"{e['clean_ce']:.3f}", 'SAE', f"{sae['l0_pos1']:.1f}", ci(sae['fvu']['pos1']), ci(sae['dce'], 3)))
            for k in sorted(int(k) for k in e['wdd'] if k != 'atoms'):
                w = e['wdd'][str(k)]
                rows.append(('', '', '', 'weight dictionary', str(k), ci(w['fvu']['pos1']), ci(w['dce'], 3)))
    if rows: table(['Corpus, context', 'state', 'clean CE', 'method', '$L_0$ or $k$', 'FVU, typical states', '$\\Delta$CE'], rows, 'The trained SAE and the weight dictionary on identical GPT-2 states. Brackets are 95\\% bootstrap intervals over sequences (64 WikiText-2 sequences, 256 OpenWebText documents).', 'app:z4-table', align='llrlrrr')
# ------------------------------------------------------------------ Z5: sparsity at matched budgets
def z5_curves(key):
    """(all-state, typical-state) FVU curves for a model: from the per-token records if present, else the JSON."""
    e = Z5.get(key)
    if not e: return None, None
    f = os.path.join(R, 'exp_z5_tokens', f'{key}_wikitext.npz')
    if os.path.exists(f):
        T = np.load(f); x2 = T['xnorm2'].astype(np.float64); nrm = np.sqrt(x2); med = np.median(nrm); typ = nrm <= 10 * med
        out = {}
        for cut, m in (('all', np.ones_like(typ)), ('typical', typ)):
            out[cut] = {d: {str(k): [float(T[f'err{k}_{d}'][m].sum() / x2[m].sum())] for k in e['checkpoints']} for d in ('weight', 'rotated')}
        return out['all'], out['typical']
    return e['fvu'], e.get('fvu_pos1')
if Z5:
    P(r'\subsection{Sparsity at matched budgets}\label{app:z5}')
    P(r'\texttt{exp\_z5\_kcurve} sweeps $k$ for the weight dictionary and its rotation on 8,192 WikiText-2 states per model (the first 16 evaluation sequences of the sweep), reading the FVU at $k=4$ to $256$ and at the matched budgets $k=d/16$ and $k=d/12$, so that models of different width are compared at the same fraction of their dimension. The table and the figure are over the typical states.')
    rows = []
    for key, nm in MODELS:
        e = Z5.get(key)
        if not e: continue
        _, f = z5_curves(key)
        if not f: continue
        k16, k12 = str(e['matched']['d/16']), str(e['matched']['d/12'])
        rows.append((nm, e['d'], pt(f['weight']['32']), pt(f['rotated']['32']), k16, pt(f['weight'][k16]), pt(f['rotated'][k16]), k12, pt(f['weight'][k12]), pt(f['rotated'][k12]), pt(f['weight']['256']), pt(f['rotated']['256'])))
    if rows: table(['Model', '$d$', 'weight, $k=32$', 'rotated, $k=32$', '$k=d/16$', 'weight', 'rotated', '$k=d/12$', 'weight', 'rotated', 'weight, $k=256$', 'rotated, $k=256$'], rows, 'FVU of the typical states at a fixed budget of 32 atoms, at budgets matched to the width, and at 256 atoms.', 'app:z5-table')
    P(r'\begin{figure}[H]\centering\begin{tikzpicture}\begin{axis}[width=0.8\linewidth, height=0.45\linewidth, xmode=log, ymode=log, xlabel={$k/d$, atoms per dimension}, ylabel={FVU of the typical states}, grid=major, grid style={black!10}, legend style={font=\tiny, at={(1.02,0.5)}, anchor=west, draw=none}, tick label style={font=\scriptsize}, label style={font=\small}]')
    for key, nm in MODELS:
        e = Z5.get(key)
        if not e: continue
        _, f = z5_curves(key)
        if not f: continue
        for dname, style in (('weight', 'thick'), ('rotated', 'thick, dashed')):
            pts = ' '.join(f"({int(k) / e['d']:.5f},{max(v[0], 1e-4):.5f})" for k, v in sorted(f[dname].items(), key=lambda kv: int(kv[0])))
            P(r'\addplot[%s, color=%s, mark=none] coordinates {%s};' % (style, COLORS[key], pts) + (r'\addlegendentry{%s}' % nm if dname == 'weight' else ''))
    P(r'\end{axis}\end{tikzpicture}\caption{Reconstruction of the typical states against the relative budget $k/d$ for the weight dictionary (solid) and its rotation (dashed), eight models on WikiText-2.}\label{app:z5-fig}\end{figure}')
# ------------------------------------------------------------------ GPT-2 sweep and provenance null on typical states
A_TOK, D_TOK = os.path.join(R, 'exp_a_tokens.npz'), os.path.join(R, 'exp_d_tokens.npz')
if os.path.exists(A_TOK) and os.path.exists(D_TOK):
    A, Dn = np.load(A_TOK), np.load(D_TOK)
    x2 = A['xnorm2_L6'].astype(np.float64); nrm = np.sqrt(x2); med = np.median(nrm); typ = nrm <= 10 * med
    KS = [4, 8, 16, 32, 64]
    P(r'\subsection{The GPT-2 sparsity sweep and the provenance null on typical states}\label{app:z-gpt2}')
    P(r'Tables~\ref{app:z-gpt2-typical} and~\ref{app:z-gpt2-all} give the GPT-2 sparsity sweep and the provenance controls of Section~\ref{sec:recon} over the typical states and over all states, from the per-token residuals of \texttt{exp\_a\_wdd} and \texttt{exp\_d\_provenance}. The sink states are %d of the 32{,}768 layer-6 states and carry %.1f\%% of the variance. The main text quotes the typical-state rows; Appendix~\ref{app:recon} has the all-state sweep with the spliced-in cross-entropies.' % (int((~typ).sum()), 100 * x2[~typ].sum() / x2.sum()))
    names = [('weight_L6', 'weight dictionary', A), ('weight_L6_pure', 'weight dictionary, no centering', A), ('actdict_L6', 'sampled-activation dictionary', A), ('pca_L6', 'PCA, best $k$ per state', A), ('random_L6', 'random directions', A), ('wdd_rotated', 'rotated weight dictionary', Dn), ('wdd_matched_correct', 'correct dictionary, subsampled to the null size', Dn), ('wdd_future_blocks', 'provenance null: atoms from blocks 7 to 11', Dn)]
    for cut, mask, label in (('typical states', typ, 'app:z-gpt2-typical'), ('all states', np.ones_like(typ), 'app:z-gpt2-all')):
        rows = []
        for key, nm, npz in names:
            if f'{key}_k32' not in npz: continue
            rows.append((nm, *[f"{npz[f'{key}_k{k}'][mask].sum() / x2[mask].sum():.3f}" for k in KS]))
        table(['Dictionary'] + [f'$k={k}$' for k in KS], rows, 'GPT-2 layer 6, FVU against sparsity over the %s.' % cut, label, align='lrrrrr')
# ------------------------------------------------------------------ the support: selected atoms, and the oracle on the same cases
Z6 = load('exp_z6_selected.json')
if Z6:
    P(r'\subsection{What the support contains}\label{app:z6}')
    P(r"""\texttt{exp\_z6\_selected} scores the other direction of attribution: every atom that OMP places in the $k=64$ support rather than the three dominant ledger writes. A selected MLP atom is a \emph{real} write when the ledger coefficient of its neuron at that token is at least 5\% of the token's largest write; otherwise the pursuit has spent an atom on a write that is not there. The table gives the composition of the support on the typical states, the share of selected MLP atoms that are real, the sign agreement and the median relative error of their recovered coefficients, and the share of the recovered MLP coefficient mass that sits on spurious atoms. The last columns compare OMP with a least-squares refit on the three true atoms on the same identified cases, which the main text's oracle could not do.""")
    for corpus, cname in [('wikitext', 'WikiText-2'), ('pile', 'the Pile sample')]:
        rows = []
        for key, nm in MODELS:
            e = Z6.get(f'{key}/{corpus}')
            if not e: continue
            c = e['support_composition_typical']; d = e['dominant']
            rows.append((nm, f"{100 * c['mlp']:.0f}\\%", f"{100 * c['attention']:.0f}\\%", f"{100 * c['embedding']:.0f}\\%", ci([e['real_share']['point'], e['real_share']['lo'], e['real_share']['hi']], 2), f"{100 * e['sign_agreement_real']['point']:.1f}\\%", f"{e['median_rel_error_real']:.2f}", f"{e['median_ratio_real']:.2f}", f"{100 * e['spurious_share_of_mlp_coefficient_mass']:.0f}\\%", f"{d['r_omp_identified']:.3f}", f"{d['r_refit_identified']:.3f}", f"{d['median_rel_error_omp_identified']:.2f}", f"{d['median_rel_error_refit_identified']:.2f}"))
        if rows: table(['Model', 'MLP', 'attn.', 'emb.', 'real share of selected MLP atoms', 'sign', 'rel. error', 'ratio', 'spurious mass', '$r$ OMP', '$r$ refit', 'rel. err. OMP', 'rel. err. refit'], rows, 'The $k=64$ support on %s, typical states. The first three columns are the composition of the support by atom type; the next five describe the selected MLP atoms; the last four compare OMP and the true-top-3 refit on the same identified dominant writes.' % cname, f'app:z6-{corpus}')
# ------------------------------------------------------------------ recall by magnitude decile, defined over all cases
RB = load('exp_z2_recall_by_decile.json')
if RB:
    P(r'\subsection{Recall by magnitude}\label{app:recall-decile}')
    P(r'Top-1 recall of the largest write by decile of its magnitude, with the deciles defined over all typical-state cases before any conditioning on identification. Identification rises with the magnitude of the write in the non-gated models and in Llama-7B, and is flat or hump-shaped in the small gated models.')
    for corpus, cname in [('wikitext', 'WikiText-2'), ('pile', 'the Pile sample')]:
        rows = [(nm, *[f"{v:.2f}" for v in RB[f'{key}_{corpus}']]) for key, nm in MODELS if f'{key}_{corpus}' in RB]
        if rows: table(['Model'] + [str(d) for d in range(1, 11)], rows, 'Top-1 recall by magnitude decile of the largest write on %s.' % cname, f'app:recall-decile-{corpus}')
# ------------------------------------------------------------------ the null without shared embeddings, and energy-matched ablation controls
D2 = load('exp_d2_null_noemb.json')
if D2:
    P(r'\subsection{The provenance null without embeddings}\label{app:d2}')
    P(r'\texttt{exp\_d2\_null\_noemb} repeats the provenance control of Section~\ref{sec:recon} with the embedding rows removed from both dictionaries, so that no atom of the future-block dictionary can have written to the state.')
    names = [('correct_blocks_noemb', 'correct block atoms, blocks 0 to 6'), ('correct_subsampled_noemb', 'correct block atoms, subsampled to the null size'), ('future_blocks_noemb', 'provenance null: block atoms of blocks 7 to 11'), ('rotated_correct_noemb', 'rotated correct block atoms')]
    for cut, label in (('fvu_typical', 'typical states'), ('fvu_all', 'all states')):
        rows = [(nm, f"{D2[k]['atoms']:,}", *[f"{D2[k][cut][str(kk)]:.3f}" for kk in (4, 8, 16, 32, 64)]) for k, nm in names if k in D2]
        table(['Dictionary', 'atoms', '$k=4$', '$k=8$', '$k=16$', '$k=32$', '$k=64$'], rows, 'GPT-2 layer 6 without embedding atoms, FVU over the %s.' % label, f'app:d2-{cut}', align='lrrrrrr')
X5B = load('exp_x5b_energy.json')
if X5B:
    P(r'\subsection{Energy-matched ablation controls}\label{app:x5b}')
    P(r'\texttt{exp\_x5b\_energy} draws random sets of block-3 neurons whose summed write energy matches that of the nine strongest counter-writers to within 5\%, from all other block-3 neurons and from the other counter-writers only, and measures the cross-entropy cost of ablating each set on the states of Appendix~\ref{app:circuit}.')
    rows = [('nine strongest counter-writers', '9', f"{X5B['top9_energy']:,.0f}", f"{X5B['top9_dce']:+.4f}")] + [(m['pool'].replace('_', ' '), str(m['n']), f"{m['energy']:,.0f}", f"{m['dce']:+.4f}") for m in X5B['matched_controls']]
    table(['Ablated set', 'neurons', 'write energy', '$\\Delta$CE'], rows, 'Cross-entropy cost of ablating energy-matched random sets against the nine strongest counter-writers (clean cross-entropy %.4f).' % X5B['ce_clean'], 'app:x5b-table', align='lrrr')
# ------------------------------------------------------------------ precision control
def prec_stats(f, nseq=None):
    T = np.load(f); seq = T['seq']; m = np.ones(len(seq), bool) if nseq is None else seq < nseq
    x2 = T['xnorm2'].astype(np.float64); nrm = np.sqrt(x2); med = np.median(nrm[m]); typ = m & (nrm <= 10 * med)
    hit = T['hit'][m]; h = hit.reshape(-1); pr = T['pred'][m].reshape(-1); t = T['true'][m].reshape(-1)
    at = np.abs(t[h]); edges = np.quantile(at, np.linspace(0, 1, 11)); dec = []
    for d in range(10):
        mm = h & (np.abs(t) >= edges[d]) & (np.abs(t) <= edges[d + 1] + 1e-9); dec.append(float(np.corrcoef(pr[mm], t[mm])[0, 1]))
    return dict(fvu=float(T['err32_weight'][typ].sum() / x2[typ].sum()), rot=float(T['err32_rotated'][typ].sum() / x2[typ].sum()), recall=float(hit[:, 0].mean()),
                r_id=float(np.corrcoef(pr[h], t[h])[0, 1]), r_unc=float(np.corrcoef(pr, t)[0, 1]), dlo=min(dec[:9]), dhi=max(dec[:9]), d10=dec[9])
F32_TOK, BF_TOK = os.path.join(R, 'exp_z2_tokens_fp32', 'qwen2.5-7b_wikitext.npz'), os.path.join(R, 'exp_z2_tokens', 'qwen2.5-7b_wikitext.npz')
if os.path.exists(F32_TOK) and os.path.exists(BF_TOK):
    P(r'\subsection{Precision control for the 7B rows}\label{app:fp32}')
    b, f = prec_stats(BF_TOK, 16), prec_stats(F32_TOK)
    P(r'The 7B-class rows of Table~\ref{tab:families} were run in bfloat16. Table~\ref{app:fp32-table} gives Qwen2.5-7B, whose aggregate $r$ on identified writes is the least stable number in the table, in both precisions on the same 8{,}192 states, the first 16 evaluation sequences. Everything agrees except the top magnitude decile, where the massive-channel writes live and where the two precisions identify different members of a handful of extreme writes, which is enough to move the aggregate.')
    rows = [(nm, f"{e['fvu']:.3f}", f"{e['rot']:.3f}", f"{e['recall']:.3f}", f"{e['r_id']:.3f}", f"{e['r_unc']:.3f}", f"{e['dlo']:.3f} to {e['dhi']:.3f}", f"{e['d10']:.3f}") for nm, e in (('bfloat16', b), ('float32', f))]
    table(['Qwen2.5-7B, WikiText-2', 'FVU at $k=32$, typical', 'rotated', 'top-1 recall', '$r$ identified', '$r$ unconditional', 'deciles 1 to 9', 'decile 10'], rows, 'Precision control: the same script in bfloat16 and in float32 on the same 8{,}192 states.', 'app:fp32-table')
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'z_tables.tex'), 'w').write('\n'.join(out) + '\n')
print('z_tables.tex written:', 'z3' if Z3 else '-', 'z4' if Z4 else '-', 'z5' if Z5 else '-', 'fp32' if F32 else '-')
