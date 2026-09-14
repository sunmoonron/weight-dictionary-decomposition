"""R3: can the model's OWN later layers serve as its Tuned Lens?

The Tuned Lens trains, per layer, an affine map h -> h + A h + b that turns an
intermediate state into the best prediction of the final distribution. The
question here is where that translation already lives. The only per-position
(context-free) computation the model itself performs after level l is the
remaining MLP sublayers plus whatever attention does when it has no context.
So the continuation lenses below run the model's remaining blocks on h_l with
attention removed or reduced to its context-free default, and unembed. Zero
trained parameters; the model is its own translator.

Decoders at level l (h = hidden_states[l], blocks l..NL-1 remain):
  logit    W_U ln_f(h)                                   [nostalgebraist 2020]
  tuned    W_U ln_f(h + A_l h + b_l), pretrained lens     [Belrose et al. 2023,
           AlignmentResearch/tuned-lens space, trained on the Pile]
  lastmlp  only the final block's MLP applied, then unembed
  mlp      MLP-only continuation: x <- x + mlp_m(ln_2^m(x)) for m = l..NL-1
  self     attention reduced to the current position (OV of self + b_O), + MLP
  sink     attention over {position 0, self} using the actual layer-m key and
           value of position 0 (context-free by construction), + MLP
  affine   MEAN-FIELD linearisation of `mlp` around the corpus-mean state
           mu_l: x = G(mu) + J(mu)(h - mu). An affine translator, like the
           tuned lens, but obtained from the weights by differentiation, not
           fitted (uses activations only for mu; zero fitted parameters).
  actual   continuation with the true attention outputs added back: must
           reproduce the model's logits exactly (code-path sanity).
Metrics per level: CE to the true next token, KL(final || lens), top-1
agreement with the final prediction, top-1 accuracy. Positions 1..T-2 of 64
WikiText-2 test chunks; also the 16-chunk / positions 0..T-2 subset used by the
kNN lens (exp_b) for comparability. Side result: the level-0 MLP continuation is
a context-free model, compared with an empirical bigram model from the train
split.

Pre-registered: E3 `mlp` beats `logit` on KL at >= 8 of 12 levels; `affine`
tracks `tuned` within 0.3 nats KL at the middle levels 4..8.
Writes results/exp_r3_continuation.json.
"""
import json, os, sys, time, math
_HF = "/data/mechinterp/hf"          # the original machine's cache; ignored elsewhere
if os.path.isdir(_HF):
    os.environ.setdefault("HF_HOME", _HF)
RESULTS_DIR = os.environ.get("WDD_RESULTS", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(0)
torch.set_grad_enabled(False)
OUT = RESULTS_DIR
os.makedirs(OUT, exist_ok=True)
DEV = os.environ.get("R_DEV", "cuda:0")
MODEL = "gpt2"
CTX = 512
N_EVAL = 64
N_TRAIN = 400
N_MU = 64
BS = 4
t0 = time.time()

model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").to(DEV).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
NL, D, H = cfg.n_layer, cfg.n_embd, cfg.n_head
HD = D // H
V = cfg.vocab_size
blocks = model.transformer.h
ln_f = model.transformer.ln_f
lm_head = model.lm_head


def token_chunks(split, n_chunks):
    text = "\n\n".join(t for t in load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0][: n_chunks * CTX]
    assert len(ids) == n_chunks * CTX
    return ids.view(n_chunks, CTX)


eval_ids = token_chunks("test", N_EVAL)
train_ids = token_chunks("train", N_TRAIN)

# ---- pretrained tuned lens ---------------------------------------------------
tuned = None
try:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("AlignmentResearch/tuned-lens", "lens/gpt2/params.pt", repo_type="space")
    c = hf_hub_download("AlignmentResearch/tuned-lens", "lens/gpt2/config.json", repo_type="space")
    sd = torch.load(p, map_location="cpu", weights_only=True)
    print("tuned-lens config:", open(c).read(), flush=True)
    print("tuned-lens keys sample:", list(sd.keys())[:6], flush=True)
    tuned = {}
    for k, v in sd.items():
        parts = k.split(".")
        idx = [int(x) for x in parts if x.isdigit()]
        if not idx:
            continue
        i = idx[0]
        tuned.setdefault(i, {})[parts[-1]] = v.to(DEV).float()
    assert all("weight" in tuned[i] for i in range(NL)), sorted(tuned)
    print(f"tuned lens loaded: {len(tuned)} translators, W {tuned[0]['weight'].shape}", flush=True)
except Exception as e:
    print("TUNED LENS UNAVAILABLE:", repr(e), flush=True)

# ---- capture attention outputs (for `actual` sanity and sink K/V) --------------
cap = {}


def mk_attn_hook(m):
    def f(mod, inp, out):
        cap[m] = out[0] if isinstance(out, tuple) else out
    return f


for m, blk in enumerate(blocks):
    blk.attn.register_forward_hook(mk_attn_hook(m))


def split_heads(x):
    B, T, _ = x.shape
    return x.view(B, T, H, HD)


def continue_from(h, l, mode, hs=None):
    """Run blocks l..NL-1 on h with attention handled per `mode`. Returns pre-ln_f state."""
    x = h
    for m in range(l, NL):
        blk = blocks[m]
        if mode == "actual":
            x = x + cap[m]
        elif mode in ("self", "sink"):
            qkv = blk.attn.c_attn(blk.ln_1(x))
            q, k, v = qkv.split(D, dim=-1)
            if mode == "self":
                o = v
            else:
                k0v0 = blk.attn.c_attn(blk.ln_1(hs[m][:, :1]))       # actual position-0 state at layer m
                _, k0, v0 = k0v0.split(D, dim=-1)                   # [B,1,D]
                qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)
                k0h, v0h = split_heads(k0), split_heads(v0)
                s_self = (qh * kh).sum(-1) * HD ** -0.5              # [B,T,H]
                s_sink = (qh * k0h).sum(-1) * HD ** -0.5
                w = torch.softmax(torch.stack([s_sink, s_self]), dim=0)
                o = (w[0][..., None] * v0h + w[1][..., None] * vh).reshape(x.shape)
            x = x + blk.attn.c_proj(o)
        elif mode == "mlp":
            pass
        else:
            raise ValueError(mode)
        x = x + blk.mlp(blk.ln_2(x))
    return x


def lastmlp(h):
    blk = blocks[NL - 1]
    return h + blk.mlp(blk.ln_2(h))


# ---- mean-field affine lens: mu_l and Jacobian of the MLP-only continuation ----
print("computing mu_l ...", flush=True)
mu = torch.zeros(NL, D, dtype=torch.float64, device=DEV)
n_mu = 0
for bi in range(0, N_MU, BS):
    out = model(train_ids[bi:bi + BS].to(DEV), output_hidden_states=True)
    for l in range(NL):
        mu[l] += out.hidden_states[l][:, 1:].double().sum((0, 1))
    n_mu += out.hidden_states[0][:, 1:].shape[0] * out.hidden_states[0][:, 1:].shape[1]
    del out
mu = (mu / n_mu).float()
cap.clear()

affine = {}
print("computing Jacobians of the MLP-only continuation at mu_l ...", flush=True)
with torch.enable_grad():
    for l in range(NL):
        def g(v):
            return continue_from(v[None, None], l, "mlp")[0, 0]
        J = torch.func.jacrev(g)(mu[l])          # [D_out, D_in]
        c = g(mu[l]).detach()
        affine[l] = (c, J.detach())
    print(f"  jacobians done {time.time() - t0:.0f}s; ||J_0 - I||_F/||I|| = "
          f"{(affine[0][1] - torch.eye(D, device=DEV)).norm() / math.sqrt(D):.3f}", flush=True)
torch.set_grad_enabled(False)

# ---- bigram baseline from train split -----------------------------------------------
cur = train_ids[:, :-1].reshape(-1)
nxt = train_ids[:, 1:].reshape(-1)
pair_keys, pair_cnt = torch.unique(cur * V + nxt, return_counts=True)
uni_cnt = torch.bincount(train_ids.reshape(-1), minlength=V).double()
p_uni = (uni_cnt + 1) / (uni_cnt.sum() + V)
cur_cnt = torch.bincount(cur, minlength=V).double()


def bigram_logp(c, n, lam=0.5):
    keys = c * V + n
    pos = torch.searchsorted(pair_keys, keys)
    pos = pos.clamp_max(len(pair_keys) - 1)
    hit = pair_keys[pos] == keys
    cnt = torch.where(hit, pair_cnt[pos], torch.zeros_like(pos)).double()
    denom = cur_cnt[c]
    p_bi = torch.where(denom > 0, cnt / denom.clamp_min(1), p_uni[n])
    p = lam * p_bi + (1 - lam) * p_uni[n]
    return torch.log(p)


# ---- evaluation ----------------------------------------------------------------
VARIANTS = ["logit", "lastmlp", "mlp", "self", "sink", "affine", "actual"] + (["tuned"] if tuned else [])


class Acc:
    def __init__(self):
        self.d = {}

    def add(self, key, ce, kl, agree, acc, n):
        s = self.d.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0])
        s[0] += ce; s[1] += kl; s[2] += agree; s[3] += acc; s[4] += n

    def out(self):
        return {k: {"ce": v[0] / v[4], "kl": v[1] / v[4], "top1_agree": v[2] / v[4],
                    "top1_acc": v[3] / v[4], "n": v[4]} for k, v in self.d.items()}


main, sub = Acc(), Acc()
bg_main, bg_sub = [0.0, 0], [0.0, 0]
uni_main = [0.0, 0]


def metrics(logits, final_lp, targets):
    lp = torch.log_softmax(logits.float(), -1)
    ce = -lp.gather(-1, targets[..., None]).squeeze(-1)
    kl = (final_lp.exp() * (final_lp - lp)).sum(-1)
    agree = (lp.argmax(-1) == final_lp.argmax(-1)).float()
    acc = (lp.argmax(-1) == targets).float()
    return float(ce.sum()), float(kl.sum()), float(agree.sum()), float(acc.sum()), ce.numel()


def decode(l, var, hs):
    h = hs[l]
    if var == "logit":
        x = h
    elif var == "tuned":
        x = h + h @ tuned[l]["weight"].T + tuned[l].get("bias", torch.zeros(D, device=DEV))
    elif var == "lastmlp":
        x = lastmlp(h)
    elif var == "affine":
        c, J = affine[l]
        x = c + (h - mu[l]) @ J.T
    else:
        x = continue_from(h, l, var, hs=hs)
    return lm_head(ln_f(x))


for bi in range(0, N_EVAL, BS):
    ids = eval_ids[bi:bi + BS].to(DEV)
    out = model(ids, output_hidden_states=True)
    hs = out.hidden_states
    final_lp = torch.log_softmax(out.logits.float(), -1)
    in_sub = bi < 16
    for l in range(NL):
        for var in VARIANTS:
            logits = decode(l, var, hs)
            main.add((l, var), *metrics(logits[:, 1:-1], final_lp[:, 1:-1], ids[:, 2:]))
            if in_sub:
                sub.add((l, var), *metrics(logits[:, :-1], final_lp[:, :-1], ids[:, 1:]))
    # final model itself
    main.add(("final", "model"), *metrics(out.logits[:, 1:-1], final_lp[:, 1:-1], ids[:, 2:]))
    if in_sub:
        sub.add(("final", "model"), *metrics(out.logits[:, :-1], final_lp[:, :-1], ids[:, 1:]))
    # bigram / unigram
    c_, n_ = ids[:, 1:-1].reshape(-1).cpu(), ids[:, 2:].reshape(-1).cpu()
    bg_main[0] += float(-bigram_logp(c_, n_).sum()); bg_main[1] += len(c_)
    uni_main[0] += float(-torch.log(p_uni[n_]).sum()); uni_main[1] += len(n_)
    if in_sub:
        c_, n_ = ids[:, :-1].reshape(-1).cpu(), ids[:, 1:].reshape(-1).cpu()
        bg_sub[0] += float(-bigram_logp(c_, n_).sum()); bg_sub[1] += len(c_)
    cap.clear()
    del out, hs
    print(f"batch {bi // BS + 1}/{N_EVAL // BS} {time.time() - t0:.0f}s", flush=True)

M, S = main.out(), sub.out()
res = {"model": MODEL, "variants": VARIANTS, "tuned_available": tuned is not None,
       "main": {f"{l}|{v}": M[(l, v)] for (l, v) in M},
       "sub16_knnprotocol": {f"{l}|{v}": S[(l, v)] for (l, v) in S},
       "bigram_ce_main": bg_main[0] / bg_main[1], "bigram_ce_sub16": bg_sub[0] / bg_sub[1],
       "unigram_ce_main": uni_main[0] / uni_main[1],
       "mu_train_chunks": N_MU, "elapsed_s": time.time() - t0}
try:
    knn = json.load(open(f"{OUT}/exp_b_knn.json"))
    res["knn_reference_sub16"] = {"final_ce": knn["final_ce"], "levels": knn["levels"]}
except Exception as e:
    res["knn_reference_sub16"] = repr(e)
with open(f"{OUT}/exp_r3_continuation.json", "w") as f:
    json.dump(res, f, indent=1)

print(f"\nfinal model CE {M[('final','model')]['ce']:.3f}   bigram CE {res['bigram_ce_main']:.3f}   unigram CE {res['unigram_ce_main']:.3f}")
print("\n=== KL(final || lens) by level ===")
print("lvl  " + " ".join(f"{v:>8s}" for v in VARIANTS))
for l in range(NL):
    print(f"{l:3d}  " + " ".join(f"{M[(l, v)]['kl']:8.3f}" for v in VARIANTS))
print("\n=== CE to true next token by level ===")
print("lvl  " + " ".join(f"{v:>8s}" for v in VARIANTS))
for l in range(NL):
    print(f"{l:3d}  " + " ".join(f"{M[(l, v)]['ce']:8.3f}" for v in VARIANTS))
print("\n=== top-1 agreement with final by level ===")
print("lvl  " + " ".join(f"{v:>8s}" for v in VARIANTS))
for l in range(NL):
    print(f"{l:3d}  " + " ".join(f"{M[(l, v)]['top1_agree']:8.3f}" for v in VARIANTS))
wins = sum(M[(l, "mlp")]["kl"] < M[(l, "logit")]["kl"] for l in range(NL))
print(f"\nE3: mlp beats logit on KL at {wins}/12 levels")
if tuned:
    gaps = [M[(l, "affine")]["kl"] - M[(l, "tuned")]["kl"] for l in range(4, 9)]
    print("E3b: affine - tuned KL at levels 4..8:", " ".join(f"{g:+.3f}" for g in gaps))
print(f"done {res['elapsed_s']:.0f}s")
