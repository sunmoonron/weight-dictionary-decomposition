"""Z2: one protocol, eight models, two corpora, bootstrap confidence intervals.

A fresh implementation of the family sweep (written from the paper's description,
not copied from exp_s9 / exp_x2), run under one protocol for every model:
32,768 evaluation states at the dictionary layer, the weight dictionary against
its Gram-preserving rotation and random directions, OMP (k = 64, FVU read at
32 and 64) and one-shot attribution scored against the ledger, and a bootstrap
over the 64 evaluation sequences (1,000 resamples) for every reported number.
Corpora: WikiText-2 (test split) and the Pile sample NeelNanda/pile-10k.

Conventions kept from the original scripts so point estimates stay comparable:
  GPT-2 small: dictionary = wte, wpe, and per block <= 6 the c_proj rows, a
    per-head SVD basis of the attention output projection and the two biases;
    reconstruction centers on a disjoint slice (the WikiText train split, or
    100 disjoint Pile chunks); attribution centers on the evaluation mean.
  Other families: embedding rows and per block <= L the down-projection columns
    and per-head SVD bases, no biases; everything centers on 8 disjoint chunks.
Per-token quantities are saved to results/exp_z2_tokens/ so the intervals can
be recomputed without a GPU. Writes results/exp_z2_ci.json incrementally.
"""
import json
import os
import time

_HF = "/data/mechinterp/hf"          # the original machine's cache; ignored elsewhere
if os.path.isdir(_HF):
    os.environ.setdefault("HF_HOME", _HF)
RESULTS_DIR = os.environ.get("WDD_RESULTS", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)
TOK_DIR = os.path.join(RESULTS_DIR, "exp_z2_tokens")
os.makedirs(TOK_DIR, exist_ok=True)

import numpy as np
import torch

torch.set_grad_enabled(False)
OUT = os.path.join(RESULTS_DIR, "exp_z2_ci.json")
CTX, N_EVAL, K, BATCH, B_BOOT = 512, int(os.environ.get("WDD_N_EVAL", 64)), 64, int(os.environ.get("WDD_OMP_BATCH", 2048)), 1000
MODELS = {
    "gpt2": ("openai-community/gpt2", "gpt2"),
    "smollm2-135m": ("HuggingFaceTB/SmolLM2-135M", "llama"),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", "llama"),
    "pythia-410m": ("EleutherAI/pythia-410m", "neox"),
    "olmo-1b-0724": ("allenai/OLMo-1B-0724-hf", "llama"),
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B", "llama"),
    "llama-7b": ("huggyllama/llama-7b", "llama"),
    "pythia-6.9b": ("EleutherAI/pythia-6.9b", "neox"),
}
BF16 = set() if os.environ.get("WDD_FP32") else {"qwen2.5-7b", "llama-7b", "pythia-6.9b"}   # WDD_FP32=1 runs the 7B rows in float32 (the precision control)
ONLY = [m for m in os.environ.get("WDD_MODELS", "").split(",") if m]
CORPORA = [c for c in os.environ.get("WDD_CORPORA", "wikitext,pile").split(",") if c]
DEV = "cuda:0"


# ---------------------------------------------------------------- statistics
def wpearson(a, b, w):
    """Pearson correlation of a and b under non-negative weights w."""
    w = w / w.sum()
    ma, mb = (w * a).sum(), (w * b).sum()
    cov = (w * (a - ma) * (b - mb)).sum()
    va, vb = (w * (a - ma) ** 2).sum(), (w * (b - mb) ** 2).sum()
    return float(cov / np.sqrt(va * vb + 1e-30))


def bootstrap(seq, per_token, n_seq, rng, B=B_BOOT):
    """Percentile intervals over resampled sequences.
    seq: token -> sequence id. per_token: dict of arrays. Returns dict metric -> (point, lo, hi)."""
    T = per_token
    counts = np.stack([np.bincount(rng.integers(0, n_seq, n_seq), minlength=n_seq) for _ in range(B)])  # [B, n_seq]
    per_seq = lambda x: np.bincount(seq, weights=x, minlength=n_seq)                                    # token array -> per-sequence sums
    out = {}
    def ratio(name, num, den):
        ns, ds = per_seq(num), per_seq(den)
        boot = (counts @ ns) / (counts @ ds)
        out[name] = (float(num.sum() / den.sum()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    def mean(name, x):
        ratio(name, x.astype(np.float64), np.ones_like(x, dtype=np.float64))
    def corr(name, a, b, mask):
        idx = np.nonzero(mask)[0]
        if len(idx) < 10:
            out[name] = (None, None, None); return
        a_, b_, s_ = a[idx], b[idx], seq[idx]
        point = wpearson(a_, b_, np.ones(len(idx)))
        boots = np.array([wpearson(a_, b_, counts[i][s_].astype(np.float64)) for i in range(B)])
        out[name] = (point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    for dname in ("weight", "rotated", "random"):
        for kk in (32, 64):
            ratio(f"fvu{kk}_{dname}", T[f"err{kk}_{dname}"], T["xnorm2"])
    ratio("fvu32_weight_offtop5", T["err32_weight_keep"], T["xnorm2_keep"])
    ratio("fvu32_rotated_offtop5", T["err32_rotated_keep"], T["xnorm2_keep"])
    mean("recall_top1_omp", T["hit"][:, 0])
    mean("recall_top1_oneshot", T["hit1_oneshot"])
    hit, pred, true = T["hit"].reshape(-1), T["pred"].reshape(-1), T["true"].reshape(-1)
    seq3 = np.repeat(seq, 3)
    seq_save = seq
    # correlations use the flattened top-3 arrays; swap seq for the flattened version
    def corr3(name, a, b, mask):
        nonlocal seq
        seq = seq3; corr(name, a, b, mask); seq = seq_save
    corr3("r_identified", pred, true, hit)
    corr3("r_unconditional", pred, true, np.ones_like(hit, dtype=bool))
    at = np.abs(true[hit])
    edges = np.quantile(at, np.linspace(0, 1, 11))
    for d_ in range(10):
        m = hit & (np.abs(true) >= edges[d_]) & (np.abs(true) <= edges[d_ + 1] + 1e-9)
        corr3(f"r_decile_{d_ + 1}", pred, true, m)
    return out


# ---------------------------------------------------------------- the sweep
def omp(X, A, k, want=False):
    """Batched orthogonal matching pursuit: greedy |correlation| pick with a mask
    on chosen atoms, least squares on the support at every step. X [N,D] and A
    [NA,D] on the GPU, unit atoms. Returns per-token residual norms squared at
    k = 32 and k = 64, and (optionally) the support and coefficients."""
    N, NA = X.shape[0], A.shape[0]
    err32 = torch.zeros(N, device=DEV); err64 = torch.zeros(N, device=DEV)
    sel = torch.zeros(N, k, dtype=torch.long, device=DEV) if want else None
    cof = torch.zeros(N, k, device=DEV) if want else None
    eye = torch.eye(k, device=DEV)
    for s in range(0, N, BATCH):
        x = X[s:s + BATCH]; n = x.shape[0]
        r = x.clone(); S = torch.zeros(n, 0, dtype=torch.long, device=DEV)
        taken = torch.zeros(n, NA, dtype=torch.bool, device=DEV)
        for step in range(k):
            pick = (r @ A.T).abs_().masked_fill_(taken, -1.0).argmax(-1, keepdim=True)
            taken.scatter_(1, pick, True)
            S = torch.cat([S, pick], 1)
            As = A[S]                                                       # [n, step+1, D]
            G = As @ As.transpose(1, 2) + 1e-5 * eye[:step + 1, :step + 1]
            c = torch.cholesky_solve(As @ x[:, :, None], torch.linalg.cholesky(G))
            r = x - (c.transpose(1, 2) @ As)[:, 0]
            if step + 1 == 32: err32[s:s + n] = (r ** 2).sum(-1)
        err64[s:s + n] = (r ** 2).sum(-1)
        if want: sel[s:s + n] = S; cof[s:s + n] = c[:, :, 0]
        del taken
    return err32, err64, sel, cof


def corpus_text(name):
    from datasets import load_dataset
    if name == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    else:
        ds = load_dataset("NeelNanda/pile-10k", split="train")
    return "\n\n".join(t for t in ds["text"] if t.strip())


def run(name, corpus):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf, fam = MODELS[name]
    t0 = time.time()
    dt = torch.bfloat16 if name in BF16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(hf, dtype=dt, device_map=DEV).eval()
    tok = AutoTokenizer.from_pretrained(hf)
    cfg = model.config
    if fam == "gpt2":
        layers, NB, D, NH = model.transformer.h, cfg.n_layer, cfg.n_embd, cfg.n_head
        L = 6
        mlp_lin = lambda l: l.mlp.c_proj                       # Conv1D: rows are the writes
        wdir = lambda l: l.mlp.c_proj.weight.detach().float()  # [dff, d]
        att_w = lambda l: l.attn.c_proj.weight.detach().float()  # [d(concat heads), d]: head h = rows h*HD:(h+1)*HD
        head_basis = lambda W, h, HD: torch.linalg.svd(W[h * HD:(h + 1) * HD, :], full_matrices=False).Vh
        emb = [model.transformer.wte.weight.detach().float(), model.transformer.wpe.weight.detach().float()]
        biases = lambda l: torch.stack([l.attn.c_proj.bias, l.mlp.c_proj.bias]).detach().float()
    elif fam == "llama":
        layers, NB, D, NH = model.model.layers, cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads
        L = NB // 2
        mlp_lin = lambda l: l.mlp.down_proj
        wdir = lambda l: l.mlp.down_proj.weight.detach().float().T   # columns are the writes -> [dff, d]
        att_w = lambda l: l.self_attn.o_proj.weight.detach().float()  # [d, d]: head h = columns h*HD:(h+1)*HD
        head_basis = lambda W, h, HD: torch.linalg.svd(W[:, h * HD:(h + 1) * HD].T, full_matrices=False).Vh
        emb = [model.model.embed_tokens.weight.detach().float()]
        biases = None
    else:
        layers, NB, D, NH = model.gpt_neox.layers, cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads
        L = NB // 2
        mlp_lin = lambda l: l.mlp.dense_4h_to_h
        wdir = lambda l: l.mlp.dense_4h_to_h.weight.detach().float().T
        att_w = lambda l: l.attention.dense.weight.detach().float()
        head_basis = lambda W, h, HD: torch.linalg.svd(W[:, h * HD:(h + 1) * HD].T, full_matrices=False).Vh
        emb = [model.gpt_neox.embed_in.weight.detach().float()]
        biases = None
    HD = D // NH
    DFF = wdir(layers[0]).shape[0]

    # ---- tokens: a disjoint centering slice, then the evaluation slice ----
    text = corpus_text(corpus)
    ids = tok(text, return_tensors="pt").input_ids[0]
    if fam == "gpt2" and corpus == "wikitext":
        from datasets import load_dataset
        tr = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"] if t.strip())
        cen_ids = tok(tr, return_tensors="pt").input_ids[0][:400 * CTX].view(400, CTX)
        eval_ids = ids[:N_EVAL * CTX].view(N_EVAL, CTX)
    else:
        n_cen = 100 if fam == "gpt2" else 8
        cen_ids = ids[:n_cen * CTX].view(n_cen, CTX)
        eval_ids = ids[n_cen * CTX:(n_cen + N_EVAL) * CTX].view(N_EVAL, CTX)
    assert eval_ids.shape == (N_EVAL, CTX), f"not enough tokens in {corpus} for {name}"

    # ---- forward passes: centering mean, then evaluation states and the ledger activations ----
    mu = torch.zeros(D, device=DEV); n_mu = 0
    for i in range(0, len(cen_ids), 8):
        out = model(cen_ids[i:i + 8].to(DEV), output_hidden_states=True)
        h = out.hidden_states[L + 1].float(); mu += h.sum((0, 1)); n_mu += h.shape[0] * h.shape[1]; del out
    mu /= n_mu
    acts = {b: [] for b in range(L + 1)}
    hooks = [mlp_lin(layers[b]).register_forward_pre_hook((lambda b_: lambda m, inp: acts[b_].append(inp[0].to(torch.float16).cpu()))(b)) for b in range(L + 1)]  # ledger activations live in CPU memory so two large models fit on one card
    X = []
    for i in range(0, N_EVAL, 4):
        out = model(eval_ids[i:i + 4].to(DEV), output_hidden_states=True)
        X.append(out.hidden_states[L + 1].float()); del out
    for h in hooks: h.remove()
    X = torch.cat(X).view(-1, D)                                   # [NT, D] raw states
    acts = {b: torch.cat(a).view(-1, DFF) for b, a in acts.items()}
    NT = X.shape[0]

    # ---- the dictionary: every direction that can write to this layer ----
    W = {b: wdir(layers[b]) for b in range(L + 1)}                    # [dff, d] rows = writes
    atoms = [e for e in emb]
    for b in range(L + 1):
        atoms.append(W[b])
        Wo = att_w(layers[b])
        for h in range(NH): atoms.append(head_basis(Wo, h, HD))
        if biases is not None: atoms.append(biases(layers[b]))
    A = torch.cat([a.to(DEV) for a in atoms]); A = A / A.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    NA = A.shape[0]
    OFF = sum(e.shape[0] for e in emb); STRIDE = DFF + NH * HD + (2 if biases is not None else 0)
    del model; torch.cuda.empty_cache()
    g = torch.Generator(device=DEV).manual_seed(7)
    Q = torch.linalg.qr(torch.randn(D, D, device=DEV, generator=g))[0]
    R_seed_state = g.get_state()   # the random dictionary is drawn later, when the card has room for it
    print(f"  {name}/{corpus}: {NT} states, d={D} dff={DFF} L={L} atoms={NA} ({time.time() - t0:.0f}s)", flush=True)

    # ---- reconstruction (disjoint-slice centering) ----
    Xc = X - mu
    var = Xc.var(0); top5 = var.topk(5).indices; keep = torch.ones(D, dtype=torch.bool, device=DEV); keep[top5] = False
    T = {"xnorm2": (Xc ** 2).sum(-1), "xnorm2_keep": (Xc[:, keep] ** 2).sum(-1)}
    e32, e64, _, _ = omp(Xc, A, K); T["err32_weight"], T["err64_weight"] = e32, e64
    # the off-channel FVU needs the k=32 residual itself: recompute cheaply from the support is not stored, so run OMP at k=32 once more for the two dictionaries that need it
    def resid_keep(Ad):
        N = Xc.shape[0]; e = torch.zeros(N, device=DEV); eye = torch.eye(32, device=DEV)
        for s in range(0, N, BATCH):
            x = Xc[s:s + BATCH]; n = x.shape[0]; r = x.clone(); S = torch.zeros(n, 0, dtype=torch.long, device=DEV); taken = torch.zeros(n, Ad.shape[0], dtype=torch.bool, device=DEV)
            for step in range(32):
                pick = (r @ Ad.T).abs_().masked_fill_(taken, -1.0).argmax(-1, keepdim=True); taken.scatter_(1, pick, True); S = torch.cat([S, pick], 1)
                As = Ad[S]; G = As @ As.transpose(1, 2) + 1e-5 * eye[:step + 1, :step + 1]
                c = torch.cholesky_solve(As @ x[:, :, None], torch.linalg.cholesky(G)); r = x - (c.transpose(1, 2) @ As)[:, 0]
            e[s:s + n] = (r[:, keep] ** 2).sum(-1); del taken
        return e
    T["err32_weight_keep"] = resid_keep(A)
    # one dictionary on the card at a time: the weight dictionary waits in CPU memory while its controls run
    A_cpu = A.cpu(); del A; torch.cuda.empty_cache()
    AQ = torch.cat([(A_cpu[i:i + 65536].to(DEV) @ Q) for i in range(0, NA, 65536)])
    e32, e64, _, _ = omp(Xc, AQ, K); T["err32_rotated"], T["err64_rotated"] = e32, e64
    T["err32_rotated_keep"] = resid_keep(AQ); del AQ; torch.cuda.empty_cache()
    g.set_state(R_seed_state); R = torch.randn(NA, D, device=DEV, generator=g); R = R / R.norm(dim=-1, keepdim=True)
    e32, e64, _, _ = omp(Xc, R, K); T["err32_random"], T["err64_random"] = e32, e64; del R; torch.cuda.empty_cache()
    A = A_cpu.to(DEV); del A_cpu
    print(f"  reconstruction done ({time.time() - t0:.0f}s): FVU32 weight {T['err32_weight'].sum() / T['xnorm2'].sum():.4f}", flush=True)

    # ---- attribution against the ledger ----
    Xf = X - X.mean(0) if fam == "gpt2" else Xc                       # GPT-2 convention: evaluation-mean centering for attribution
    WN = {b: W[b].norm(dim=-1).cpu() for b in range(L + 1)}
    TRUE = torch.cat([acts[b].float() * WN[b] for b in range(L + 1)], dim=1)   # [NT, (L+1)*DFF] on the CPU
    top3 = TRUE.abs().topk(3, dim=1).indices
    true = TRUE.gather(1, top3).to(DEV); del TRUE, acts
    top3 = top3.to(DEV)
    atom_idx = OFF + (top3 // DFF) * STRIDE + (top3 % DFF)
    _, _, sel, cof = omp(Xf, A, K, want=True)
    match = sel[:, None, :] == atom_idx[:, :, None]
    hit = match.any(-1); pos = match.float().argmax(-1)
    pred = cof.gather(1, pos) * hit
    os_hit = torch.zeros(NT, dtype=torch.bool, device=DEV)
    for s in range(0, NT, BATCH):
        cr = (Xf[s:s + BATCH] @ A.T).abs_()
        os_hit[s:s + BATCH] = (cr.topk(K, dim=1).indices == atom_idx[s:s + BATCH, 0:1]).any(-1)
    T["hit"], T["pred"], T["true"], T["hit1_oneshot"] = hit, pred, true, os_hit
    print(f"  attribution done ({time.time() - t0:.0f}s): recall {hit[:, 0].float().mean():.4f}", flush=True)

    # ---- per-token record and the bootstrap ----
    P = {k_: v.detach().float().cpu().numpy() if v.dtype != torch.bool else v.cpu().numpy() for k_, v in T.items()}
    seq = np.repeat(np.arange(N_EVAL), CTX)
    np.savez_compressed(os.path.join(TOK_DIR, f"{name}_{corpus}.npz"), seq=seq, **P)
    rng = np.random.default_rng(2026)
    ci = bootstrap(seq, P, N_EVAL, rng)
    res = {"n_states": int(NT), "n_seq": N_EVAL, "d": int(D), "dff": int(DFF), "L": int(L), "atoms": int(NA),
           "top5_var_share": float((var[top5].sum() / var.sum()).item()), "top5_var_coords": top5.tolist(),
           "n_identified": int(hit.sum().item()), "seconds": round(time.time() - t0),
           "ci": {k_: {"point": v[0], "lo": v[1], "hi": v[2]} for k_, v in ci.items()}}
    del A, X, Xc, Xf, sel, cof; torch.cuda.empty_cache()
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
            c = RES[key]["ci"]
            print(f"  {key}: FVU32 {c['fvu32_weight']['point']:.3f} [{c['fvu32_weight']['lo']:.3f},{c['fvu32_weight']['hi']:.3f}]  rot {c['fvu32_rotated']['point']:.3f}  recall {c['recall_top1_omp']['point']:.3f} [{c['recall_top1_omp']['lo']:.3f},{c['recall_top1_omp']['hi']:.3f}]  r_id {c['r_identified']['point']:.3f} [{c['r_identified']['lo']:.3f},{c['r_identified']['hi']:.3f}]  r_unc {c['r_unconditional']['point']:.3f}", flush=True)
            json.dump(RES, open(OUT, "w"), indent=1)
    print("DONE_Z2", flush=True)
