"""Post-review 3 (2026-11-21): the oracles Reviewer NMa5 asked for and the solver comparison behind Reviewer MgDZ's
question, on the paper's GPT-2 layer-6 protocol (the states, centering, dictionary and ledger of exp_z2_ci).

Parts, chosen with PR3_PARTS (comma-separated):
  omp      OMP at k = 64 on the attribution states; the support is kept for the oracles.
  oracles  (needs omp) Ground truth for the non-MLP atoms of the support.
           Attention: each head's true write is its slice of the attention output-projection input times its rows of
           the projection, and its projection on the head's own basis atoms is a true coefficient for every attention
           atom (NMa5's oracle). Scored: whether the dominant head has any atom in the support, against the rate for
           a random head; the share of that head's write energy the selected atoms capture; for the selected attention
           atoms, the sign agreement with the true coefficient and the share that are real (true coefficient at least
           5% of the token's largest true attention coefficient, the paper's rule for MLP atoms); and whether the atom
           with the largest true attention coefficient is in the support. Head writes are scored raw (the definition
           used in follow-up work) and centred on their evaluation mean (what the centred state can contain).
           Embeddings: a token or position row is a true write only if it is the token's own, so the share of selected
           rows that are the current token or position is their precision; also whether selected token rows match an
           earlier context token more often than unigram-matched random tokens do.
  dual     one-shot ranking, and one-shot ranking through the canonical dual frame (S = A^T A, regularised inverse).
  lasso    basis pursuit denoising by FISTA, lambda tuned on 256 states so the median support is about k
           (PR3_LASSO_LAM fixes the penalty instead; the follow-up experiment's tuning stopped at its ceiling of 9.9105).
Every solver is scored by top-1 identification of the token's largest MLP write and by the FVU of a least-squares refit
on its 64-atom support, typical states. Writes results/pr3_<parts>.json. Arguments: [--smoke]."""
import json, math, os, sys, time
_HF = "/data/mechinterp/hf"
if os.path.isdir(_HF): os.environ.setdefault("HF_HOME", _HF)
OUT_DIR = os.environ.get("PR_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")); os.makedirs(OUT_DIR, exist_ok=True)
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
SMOKE = "--smoke" in sys.argv; PARTS = [p for p in os.environ.get("PR3_PARTS", "omp,oracles,dual,lasso").split(",") if p]; t0 = time.time()
torch.manual_seed(0); torch.set_grad_enabled(False); DEV = "cuda:0"
CTX, L, K, BATCH = 512, 6, 64, 1024
N_TRAIN, N_EVAL = (8, 2) if SMOKE else (400, int(os.environ.get("WDD_N_EVAL", 64)))
FISTA_PILOT, FISTA_ITERS = (60, 60) if SMOKE else (200, 300)
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(DEV).eval(); tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
cfg = model.config; layers = model.transformer.h; D, NH = cfg.n_embd, cfg.n_head; HD = D // NH
def ids_of(split): return tok("\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"] if t.strip()), return_tensors="pt").input_ids[0]
eval_ids = ids_of("test")[:N_EVAL * CTX].view(N_EVAL, CTX); train_flat = ids_of("train"); cen_ids = train_flat[:N_TRAIN * CTX].view(N_TRAIN, CTX)
mu = torch.zeros(D, device=DEV); n_mu = 0
for i in range(0, len(cen_ids), 8):
    h = model(cen_ids[i:i + 8].to(DEV), output_hidden_states=True).hidden_states[L + 1].float(); mu += h.sum((0, 1)); n_mu += h.shape[0] * h.shape[1]
mu /= n_mu
acts = {b: [] for b in range(L + 1)}; hin = {b: [] for b in range(L + 1)}
hooks = [layers[b].mlp.c_proj.register_forward_pre_hook((lambda b_: lambda m, inp: acts[b_].append(inp[0].to(torch.float16).cpu()))(b)) for b in range(L + 1)]
hooks += [layers[b].attn.c_proj.register_forward_pre_hook((lambda b_: lambda m, inp: hin[b_].append(inp[0].to(torch.float16).cpu()))(b)) for b in range(L + 1)]
X = []
for i in range(0, N_EVAL, 4): X.append(model(eval_ids[i:i + 4].to(DEV), output_hidden_states=True).hidden_states[L + 1].float())
for h in hooks: h.remove()
X = torch.cat(X).view(-1, D); NT = X.shape[0]
W = {b: layers[b].mlp.c_proj.weight.detach().float() for b in range(L + 1)}; DFF = W[0].shape[0]      # [dff, d]: rows are the MLP writes
WO = {b: layers[b].attn.c_proj.weight.detach().float() for b in range(L + 1)}                          # [d, d]: head h = rows h*HD:(h+1)*HD
acts = {b: torch.cat(a).view(-1, DFF) for b, a in acts.items()}; hin = {b: torch.cat(a).view(-1, D) for b, a in hin.items()}
VH = {b: [torch.linalg.svd(WO[b][h * HD:(h + 1) * HD], full_matrices=False).Vh for h in range(NH)] for b in range(L + 1)}
atoms = [model.transformer.wte.weight.detach().float(), model.transformer.wpe.weight.detach().float()]
for b in range(L + 1):
    atoms.append(W[b]); atoms += VH[b]; atoms.append(torch.stack([layers[b].attn.c_proj.bias, layers[b].mlp.c_proj.bias]).detach().float())
A = torch.cat(atoms); A = A / A.norm(dim=-1, keepdim=True).clamp_min(1e-8); NA = A.shape[0]
NTOK, NPOS = atoms[0].shape[0], atoms[1].shape[0]; OFF = NTOK + NPOS; STRIDE = DFF + NH * HD + 2
del model; torch.cuda.empty_cache()
Xc = X - mu; xn = Xc.norm(dim=1); typ = xn <= 10 * xn.median()          # typical states, as exp_z2_recompute
Xf = X - X.mean(0)                                                       # GPT-2 attribution convention of exp_z2_ci
WN = {b: W[b].norm(dim=-1).cpu() for b in range(L + 1)}
TRUE = torch.cat([acts[b].float() * WN[b] for b in range(L + 1)], dim=1); top1 = TRUE.abs().argmax(1)
row1 = (OFF + (top1 // DFF) * STRIDE + (top1 % DFF)).to(DEV); del TRUE, acts
seq = torch.arange(NT, device=DEV) // CTX; pos = torch.arange(NT, device=DEV) % CTX; cur = eval_ids.reshape(-1).to(DEV)
print(f"{NT} states, {int(typ.sum())} typical, {NA} atoms, parts {PARTS} ({time.time() - t0:.0f}s)", flush=True)
rng = np.random.default_rng(2026); BOOT = np.stack([np.bincount(rng.integers(0, N_EVAL, N_EVAL), minlength=N_EVAL) for _ in range(1000)])
def boot_mean(x_bool):
    x = x_bool.float().cpu().numpy().astype(np.float64); s = seq.cpu().numpy(); num = np.bincount(s, weights=x, minlength=N_EVAL); den = np.bincount(s, minlength=N_EVAL).astype(np.float64)
    b = (BOOT @ num) / (BOOT @ den); return dict(point=float(x.mean()), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)))
def refit(sel):
    err = torch.zeros(NT, device=DEV); cof = torch.zeros(NT, sel.shape[1], device=DEV); eye = torch.eye(sel.shape[1], device=DEV)
    for s in range(0, NT, BATCH):
        x = Xf[s:s + BATCH]; As = A[sel[s:s + BATCH]]; G = As @ As.transpose(1, 2) + 1e-5 * eye
        c = torch.cholesky_solve(As @ x[:, :, None], torch.linalg.cholesky(G)); err[s:s + BATCH] = ((x - (c.transpose(1, 2) @ As)[:, 0]) ** 2).sum(-1); cof[s:s + BATCH] = c[:, :, 0]
    return err, cof
x2 = (Xf ** 2).sum(-1)
def score(name, sel, seconds):
    hit = (sel == row1[:, None]).any(1); err, cof = refit(sel)
    out = dict(recall_top1=boot_mean(hit), recall_top1_typical=float(hit[typ].float().mean()), fvu_refit_typical=float(err[typ].sum() / x2[typ].sum()), seconds=round(seconds))
    print(f"  {name}: top-1 recall {out['recall_top1']['point']:.3f} [{out['recall_top1']['lo']:.3f}, {out['recall_top1']['hi']:.3f}] (typical {out['recall_top1_typical']:.3f}), refit FVU typical {out['fvu_refit_typical']:.3f} ({seconds:.0f}s)", flush=True)
    return out, cof
def omp(Xs):
    N = Xs.shape[0]; sel = torch.zeros(N, K, dtype=torch.long, device=DEV); eye = torch.eye(K, device=DEV)
    for s in range(0, N, BATCH):
        x = Xs[s:s + BATCH]; n = x.shape[0]; r = x.clone(); S = torch.zeros(n, 0, dtype=torch.long, device=DEV); taken = torch.zeros(n, NA, dtype=torch.bool, device=DEV)
        for step in range(K):
            pick = (r @ A.T).abs_().masked_fill_(taken, -1.0).argmax(-1, keepdim=True); taken.scatter_(1, pick, True); S = torch.cat([S, pick], 1)
            As = A[S]; G = As @ As.transpose(1, 2) + 1e-5 * eye[:step + 1, :step + 1]
            c = torch.cholesky_solve(As @ x[:, :, None], torch.linalg.cholesky(G)); r = x - (c.transpose(1, 2) @ As)[:, 0]
        sel[s:s + n] = S; del taken
    return sel
R = dict(n_states=int(NT), n_typical=int(typ.sum()), atoms=int(NA), smoke=SMOKE, parts=PARTS, solvers={})
FIXED_LAM = os.environ.get("PR3_LASSO_LAM")       # optional: a fixed LASSO penalty instead of the tuned one (the follow-up experiment ran at 9.9105)
outp = os.path.join(OUT_DIR, f"pr3_{'_'.join(PARTS)}{'_lam' + FIXED_LAM if FIXED_LAM else ''}{'_smoke' if SMOKE else ''}.json")
def save(): json.dump(R, open(outp, "w"), indent=1)

if "omp" in PARTS:
    t1 = time.time(); sel = omp(Xf); R["solvers"]["omp"], cof = score("omp", sel, time.time() - t1); save()

if "oracles" in PARTS:
    # ---- support composition on typical states ----
    rel = (sel - OFF) % STRIDE; inblk = sel >= OFF
    is_tok = sel < NTOK; is_pos = (sel >= NTOK) & (sel < OFF); is_mlp = inblk & (rel < DFF); is_att = inblk & (rel >= DFF) & (rel < DFF + NH * HD); is_bias = inblk & (rel >= DFF + NH * HD)
    comp = {k_: float(v[typ].float().mean()) for k_, v in (("token", is_tok), ("position", is_pos), ("mlp", is_mlp), ("attention", is_att), ("bias", is_bias))}
    # ---- attention: true coefficients of every attention atom, raw and centred on the evaluation mean ----
    M = {b: [WO[b][h * HD:(h + 1) * HD] @ VH[b][h].T for h in range(NH)] for b in range(L + 1)}       # [HD, HD]: head slice -> coefficients on its own basis
    hmean = {b: hin[b].float().mean(0).to(DEV) for b in range(L + 1)}
    x0 = hin[0][:256].float().to(DEV); w0 = x0[:, :HD] @ WO[0][:HD]; span_check = float((((x0[:, :HD] @ M[0][0]) ** 2).sum(1) - (w0 ** 2).sum(1)).abs().max() / (w0 ** 2).sum(1).max())   # a head's write lies in the span of its atoms
    blk_a = torch.div(sel - OFF, STRIDE, rounding_mode="floor").clamp(0, L); head_a = torch.div(rel - DFF, HD, rounding_mode="floor").clamp(0, NH - 1); rank_a = (rel - DFF) % HD
    flat_a = (blk_a * NH + head_a) * HD + rank_a                                                     # index into the [L+1, NH, HD] coefficient block, valid where is_att
    st = {v: dict(dom_hit=torch.zeros(NT, dtype=torch.bool, device=DEV), capt=torch.zeros(NT, device=DEV), dom_share=torch.zeros(NT, device=DEV)) for v in ("raw", "centred")}
    nheads = torch.zeros(NT, device=DEV); true_sel = torch.zeros(NT, K, device=DEV); max_true = torch.zeros(NT, device=DEV); best_in = torch.zeros(NT, dtype=torch.bool, device=DEV)
    base_real = torch.zeros(NT, device=DEV); base_rel = torch.zeros(NT, device=DEV)      # the same 5% rule and the median |true| / max over ALL attention atoms of a token: the base rate for the selected ones
    for s in range(0, NT, 2048):
        e = min(NT, s + 2048); n = e - s; sl = sel[s:e]; ia = is_att[s:e]
        for v in ("raw", "centred"):
            CT = torch.zeros(n, L + 1, NH, HD, device=DEV)
            for b in range(L + 1):
                xb = hin[b][s:e].float().to(DEV)
                if v == "centred": xb = xb - hmean[b]
                for h in range(NH): CT[:, b, h] = xb[:, h * HD:(h + 1) * HD] @ M[b][h]
            en = (CT ** 2).sum(-1).view(n, -1); dom = en.argmax(1)                                    # write energy per head; the head's write lies in the span of its 64 orthonormal atoms
            hid = blk_a[s:e] * NH + head_a[s:e]; mine = ia & (hid == dom[:, None])
            st[v]["dom_hit"][s:e] = mine.any(1); st[v]["dom_share"][s:e] = en.gather(1, dom[:, None])[:, 0] / en.sum(1).clamp_min(1e-12)
            tc = CT.view(n, -1).gather(1, flat_a[s:e].clamp(0, (L + 1) * NH * HD - 1))                # true coefficient of every selected atom (meaningful where is_att)
            st[v]["capt"][s:e] = ((tc ** 2) * mine).sum(1) / en.gather(1, dom[:, None])[:, 0].clamp_min(1e-12)
            if v == "centred":
                true_sel[s:e] = tc * ia; mx, am = CT.view(n, -1).abs().max(1); max_true[s:e] = mx
                ca = CT.view(n, -1).abs(); base_real[s:e] = (ca >= 0.05 * mx[:, None]).float().mean(1); base_rel[s:e] = (ca / mx[:, None].clamp_min(1e-12)).median(1).values; del ca
                best_atom = OFF + (am // (NH * HD)) * STRIDE + DFF + (am % (NH * HD)); best_in[s:e] = (sl == best_atom[:, None]).any(1)
                # a head counts as selected if any of its atoms is in the support (non-attention atoms go to a spare column)
                hh = torch.zeros(n, (L + 1) * NH + 1, dtype=torch.bool, device=DEV); hh.scatter_(1, torch.where(ia, hid, torch.full_like(hid, (L + 1) * NH)), True)
                nheads[s:e] = hh[:, :(L + 1) * NH].float().sum(1)
    att = {}
    for v in ("raw", "centred"):
        dh_ = st[v]["dom_hit"]; m_ = typ & dh_
        att[v] = dict(dominant_head_selected=float(dh_[typ].float().mean()), energy_captured_median_when_selected=float(st[v]["capt"][m_].median()) if m_.any() else None,
                      energy_captured_mean_all_typical=float(st[v]["capt"][typ].mean()), dominant_head_share_of_head_energy_median=float(st[v]["dom_share"][typ].median()))
    ia_t = is_att & typ[:, None]; ts, oc = true_sel[ia_t], cof[ia_t]; mt = max_true[:, None].expand(-1, K)[ia_t]
    real = ts.abs() >= 0.05 * mt
    att["span_check_max_rel_err"] = span_check; att["random_head_selected"] = float((nheads / ((L + 1) * NH))[typ].mean()); att["heads_with_an_atom_mean"] = float(nheads[typ].mean())
    att["selected_attention_atoms"] = dict(n=int(ia_t.sum()), sign_agreement=float(((ts * oc) > 0).float().mean()), real_share=float(real.float().mean()),
                                           r_recovered_true=float(torch.corrcoef(torch.stack([oc, ts]))[0, 1]), median_ratio_real=float((oc[real] / ts[real]).median()) if real.any() else None,
                                           median_rel_error_real=float(((oc[real] - ts[real]).abs() / ts[real].abs()).median()) if real.any() else None)
    att["largest_true_attention_atom_in_support"] = float(best_in[typ].float().mean())
    att["all_attention_atoms"] = dict(real_share=float(base_real[typ].mean()), true_over_max_median=float(base_rel[typ].median())); att["selected_attention_atoms"]["true_over_max_median"] = float((ts.abs() / mt.clamp_min(1e-12)).median())
    # ---- embeddings: the token's own row and position row are the only true embedding writes ----
    first = torch.full((N_EVAL, NTOK), CTX, dtype=torch.long, device=DEV); ev = eval_ids.to(DEV); ar = torch.arange(N_EVAL, device=DEV)
    for p in range(CTX - 1, -1, -1): first[ar, ev[:, p]] = p
    def tok_stats(tokens, mask):
        t = tokens.clamp(0, NTOK - 1); is_cur = (t == cur[:, None]) & mask; earlier = (first[seq[:, None].expand_as(t), t] < pos[:, None]) & mask; n_ = mask.sum().clamp_min(1)
        return dict(n=int(mask.sum()), current_token=float(is_cur.sum() / n_), earlier_context_token=float(earlier.sum() / n_))
    mt_ = is_tok & typ[:, None]; g = torch.Generator().manual_seed(0); draw = train_flat[torch.randint(0, len(train_flat), (NT, K), generator=g)].to(DEV)
    emb = dict(selected_token_rows=tok_stats(sel, mt_), unigram_matched_random_tokens=tok_stats(draw, mt_),
               own_token_row_in_support=float((sel == cur[:, None]).any(1)[typ].float().mean()), own_position_row_in_support=float((sel == (NTOK + pos)[:, None]).any(1)[typ].float().mean()),
               selected_position_rows_that_are_own=float(((sel == (NTOK + pos)[:, None]) & is_pos & typ[:, None]).sum() / (is_pos & typ[:, None]).sum().clamp_min(1)),
               selected_position_rows_within_8=float((((sel - NTOK - pos[:, None]).abs() <= 8) & is_pos & typ[:, None]).sum() / (is_pos & typ[:, None]).sum().clamp_min(1)))
    R["support_composition_typical"] = comp; R["attention_oracle"] = att; R["embedding_oracle"] = emb; save()
    print(f"  support: " + ", ".join(f"{k_} {v:.3f}" for k_, v in comp.items()), flush=True)
    print(f"  heads: dominant head selected raw {att['raw']['dominant_head_selected']:.3f} / centred {att['centred']['dominant_head_selected']:.3f}, random head {att['random_head_selected']:.3f}; energy captured when selected raw {att['raw']['energy_captured_median_when_selected']} / centred {att['centred']['energy_captured_median_when_selected']}; selected attention atoms: sign {att['selected_attention_atoms']['sign_agreement']:.3f}, real {att['selected_attention_atoms']['real_share']:.3f}, r {att['selected_attention_atoms']['r_recovered_true']:.3f}; largest true attention atom in support {att['largest_true_attention_atom_in_support']:.3f}", flush=True)
    print(f"  embeddings: selected token rows current {emb['selected_token_rows']['current_token']:.4f}, earlier context {emb['selected_token_rows']['earlier_context_token']:.4f} (random tokens {emb['unigram_matched_random_tokens']['earlier_context_token']:.4f}); own token row in support {emb['own_token_row_in_support']:.4f}, own position row {emb['own_position_row_in_support']:.4f}, selected position rows that are own {emb['selected_position_rows_that_are_own']:.4f} (within 8: {emb['selected_position_rows_within_8']:.4f}) ({time.time() - t0:.0f}s)", flush=True)

if "dual" in PARTS or "lasso" in PARTS:
    S_ = A.T @ A; evs, V = torch.linalg.eigh(S_); LIP = float(evs[-1])
if "dual" in PARTS:
    def oneshot(Wm=None):
        s_ = torch.zeros(NT, K, dtype=torch.long, device=DEV)
        for s in range(0, NT, BATCH):
            x = Xf[s:s + BATCH]; s_[s:s + BATCH] = ((x @ Wm.T if Wm is not None else x) @ A.T).abs_().topk(K, dim=1).indices
        return s_
    t1 = time.time(); R["solvers"]["oneshot"], _ = score("one-shot", oneshot(), time.time() - t1)
    for eps in (1e-2, 1e-1):
        t1 = time.time(); Winv = V @ torch.diag(1 / (evs + eps * evs[-1])) @ V.T; R["solvers"][f"dual_frame_eps{eps:g}"], _ = score(f"dual frame, eps {eps:g}", oneshot(Winv), time.time() - t1)
    save()
if "lasso" in PARTS:
    def fista(x, lam, iters):
        c = torch.zeros(x.shape[0], NA, device=DEV); y = c.clone(); t = 1.0
        for _ in range(iters):
            g_ = (y @ A - x) @ A.T; g_.mul_(-1.0 / LIP).add_(y); z = torch.nn.functional.softshrink(g_, lam / LIP); del g_
            t2 = (1 + math.sqrt(1 + 4 * t * t)) / 2; y = z + ((t - 1) / t2) * (z - c); c, t = z, t2
        return c
    nnz = lambda C: float((C.abs() > 1e-3 * C.abs().max(1, keepdim=True).values).sum(1).float().median())
    pilot = Xf[torch.linspace(0, NT - 1, 256).long().to(DEV)]; hi = float((pilot @ A.T).abs().max()); lo = 1e-4 * hi     # above hi every coefficient is zero
    for _ in range(0 if FIXED_LAM else 12):
        mid = math.sqrt(lo * hi); n_ = nnz(fista(pilot, mid, FISTA_PILOT))
        if n_ > K: lo = mid
        else: hi = mid
    lam = float(FIXED_LAM) if FIXED_LAM else math.sqrt(lo * hi); print(f"  lasso lambda {lam:.4f} ({'fixed' if FIXED_LAM else f'tuned: pilot median support about {K}'}) ({time.time() - t0:.0f}s)", flush=True)
    t1 = time.time(); sl = torch.zeros(NT, K, dtype=torch.long, device=DEV); nz = []; raw = torch.zeros(NT, device=DEV)
    for s in range(0, NT, BATCH):
        C = fista(Xf[s:s + BATCH], lam, FISTA_ITERS); sl[s:s + BATCH] = C.abs().topk(K, dim=1).indices; nz.append(nnz(C)); raw[s:s + BATCH] = ((Xf[s:s + BATCH] - C @ A) ** 2).sum(-1); del C
        if s == 0: print(f"  lasso first batch done ({time.time() - t1:.0f}s)", flush=True)
    R["solvers"]["lasso"], _ = score("lasso", sl, time.time() - t1); R["solvers"]["lasso"].update(dict(lam=lam, nnz_median=float(np.median(nz)), fvu_raw_typical=float(raw[typ].sum() / x2[typ].sum()))); save()
print("DONE_PR3", f"{time.time() - t0:.0f}s", flush=True)
