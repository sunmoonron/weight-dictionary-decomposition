"""R1: the READER LENS -- a readiness map of GPT-2 small.

First-principles framing. The Logit Lens applies ONE fixed reader (the
unembedding) to intermediate residual states and asks how early the output is
decided. But the unembedding is only one of the model's readers: every later
MLP neuron reads the stream through a row of W_in, every later head through
W_Q/W_K/W_V. All of them are fixed, trained, and free. The reader lens applies
ALL downstream readers to an intermediate state and compares the would-be
activation with the activation the reader actually receives at its own layer.
Unlike the Logit Lens the ground truth exists (the actual activation), so the
basis-mismatch problem the Tuned Lens had to LEARN a correction for becomes a
MEASURED quantity per reader, with zero training.

Objects, for source level l (hidden_states[l] = input to block l) and reader
layer m >= l:
  MLP readers   z_wb = c_fc(ln_2^m(h_l))   vs   z_act = c_fc(ln_2^m(resid_mid_m))
                per-neuron Pearson r and relative error, l = 0..m
                (l = m means "skip only attention m").
  attention     Q,K,V units of head h from ln_1^m(h_l) vs from ln_1^m(h_m);
  readers       PATTERN agreement: total variation between the would-be pattern
                and the actual one, for (Q_wb,K_wb), (Q_wb,K_act), (Q_act,K_wb)
                -- so query-side and key-side readiness separate. l = 0..m-1.
  unembedding   logit-lens top-1 agreement with the final prediction (sanity:
                the Logit Lens IS the reader lens for the reader W_U).
  nulls         256 isotropic random directions and 256 random directions in
                the span of the layer's reader rows, read the same way: the
                baseline persistence of an arbitrary direction across depth.
Readiness depth of a reader at threshold tau: the earliest l such that r >= tau
at every level from l to m. Slack = m - depth. "Lexical" = ready at l = 0
(decidable from the token+position embedding alone); "needs attention m" =
not ready even at l = m.

Pre-registered expectations:
  E1 readers are heterogeneous relative to the null: a bimodal mix of early-
     ready (slack >= 2) and attention-gated (needs attn m) neurons.
  E2 induction heads (L5.H1, L5.H5) show a KEY-side readiness jump exactly
     after the previous-token head's layer, with query side ready earlier:
     circuit timing recovered with no patching.
Writes results/exp_r1_readiness.json + .npz.
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
MODEL = os.environ.get("R_MODEL", "gpt2")
TAG = os.environ.get("R_TAG", "gpt2")
CTX = 512
N_EVAL = int(os.environ.get("R_NEVAL", "64"))
BS = 4
TAUS = [0.8, 0.9, 0.95]
N_NULL = 256
t0 = time.time()

model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").to(DEV).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
NL, D, H = cfg.n_layer, cfg.n_embd, cfg.n_head
HD = D // H
blocks = model.transformer.h
DFF = blocks[0].mlp.c_fc.weight.shape[1]
W_U = model.lm_head.weight
ln_f = model.transformer.ln_f
print(f"{MODEL}: NL={NL} D={D} H={H} DFF={DFF} dev={DEV}", flush=True)


def token_chunks(split, n_chunks):
    text = "\n\n".join(t for t in load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0][: n_chunks * CTX]
    assert len(ids) == n_chunks * CTX
    return ids.view(n_chunks, CTX)


eval_ids = token_chunks("test", N_EVAL)
print(f"eval tokens {eval_ids.numel()}", flush=True)

# ---- capture hooks -----------------------------------------------------------
cap = {"attn": {}, "z": {}}


def mk_attn_hook(m):
    def f(mod, inp, out):
        cap["attn"][m] = out[0] if isinstance(out, tuple) else out
    return f


def mk_z_hook(m):
    def f(mod, inp, out):
        cap["z"][m] = out
    return f


for m, blk in enumerate(blocks):
    blk.attn.register_forward_hook(mk_attn_hook(m))
    blk.mlp.c_fc.register_forward_hook(mk_z_hook(m))

# ---- accumulators ------------------------------------------------------------
def new_stats(K):
    return torch.zeros(6, K, dtype=torch.float64, device=DEV)


def acc(S, x, y):
    x = x.double(); y = y.double()
    S[0] += x.shape[0]
    S[1] += x.sum(0); S[2] += y.sum(0)
    S[3] += (x * x).sum(0); S[4] += (y * y).sum(0); S[5] += (x * y).sum(0)


def finish(S, eps=1e-12):
    n, Sx, Sy, Sxx, Syy, Sxy = S
    mx, my = Sx / n, Sy / n
    vx = Sxx / n - mx ** 2
    vy = Syy / n - my ** 2
    cov = Sxy / n - mx * my
    r = cov / torch.sqrt((vx * vy).clamp_min(eps))
    re = ((Sxx - 2 * Sxy + Syy) / n) / vy.clamp_min(eps)
    return r.float().cpu().numpy(), re.float().cpu().numpy(), vy.float().cpu().numpy()


S_mlp = {m: [new_stats(DFF) for _ in range(m + 1)] for m in range(NL)}
S_att = {m: [new_stats(3 * D) for _ in range(m)] for m in range(NL)}
S_niso = {m: [new_stats(N_NULL) for _ in range(m + 1)] for m in range(NL)}
S_nspan = {m: [new_stats(N_NULL) for _ in range(m + 1)] for m in range(NL)}
S_act = {m: new_stats(DFF) for m in range(NL)}          # post-GELU activation stats (importance)
PAT = {k: torch.zeros(NL, NL, H, dtype=torch.float64, device=DEV)
       for k in ["tv_full", "tv_q", "tv_k", "top1_full", "top1_q", "top1_k"]}
pat_n = 0
unembed_agree = torch.zeros(NL + 1, dtype=torch.float64, device=DEV)
unembed_n = 0

g = torch.Generator(device="cpu").manual_seed(1)
U_iso = F.normalize(torch.randn(D, N_NULL, generator=g), dim=0).to(DEV)
U_span = {}
for m in range(NL):
    G = torch.randn(DFF, N_NULL, generator=g).to(DEV)
    U_span[m] = F.normalize(blocks[m].mlp.c_fc.weight @ G, dim=0)  # [D, N_NULL]

causal = torch.full((CTX, CTX), float("-inf"), device=DEV).triu(1)


def split_heads(x):  # [B,T,D] -> [B,H,T,HD]
    B, T, _ = x.shape
    return x.view(B, T, H, HD).transpose(1, 2)


def pattern(q, k):
    s = torch.matmul(q, k.transpose(-1, -2)) * (HD ** -0.5) + causal
    return torch.softmax(s.float(), dim=-1)


def qkv_from(blk, x):
    qkv = blk.attn.c_attn(blk.ln_1(x))
    q, k, v = qkv.split(D, dim=-1)
    return qkv, split_heads(q), split_heads(k), split_heads(v)


def pat_metrics(P, P_act):
    # P, P_act: [B,H,T,T]; exclude query position 0
    tv = 0.5 * (P[:, :, 1:] - P_act[:, :, 1:]).abs().sum(-1)      # [B,H,T-1]
    top1 = (P[:, :, 1:].argmax(-1) == P_act[:, :, 1:].argmax(-1)).double()
    return tv.double().mean((0, 2)), top1.mean((0, 2))


sanity = {}
n_batches = 0
for bi in range(0, N_EVAL, BS):
    ids = eval_ids[bi:bi + BS].to(DEV)
    first = (bi == 0)
    out = model(ids, output_hidden_states=True, output_attentions=first)
    hs = out.hidden_states
    B, T = ids.shape
    if first:
        # sanity: hook placement + resid_mid convention + own pattern code
        for m in [0, 5, NL - 1]:
            blk = blocks[m]
            x_mid = hs[m] + cap["attn"][m]
            z_re = blk.mlp.c_fc(blk.ln_2(x_mid))
            sanity[f"z_hook_vs_recompute_L{m}"] = float((z_re - cap["z"][m]).abs().max())
            _, q, k, _ = qkv_from(blk, hs[m])
            sanity[f"pattern_vs_hf_L{m}"] = float((pattern(q, k) - out.attentions[m]).abs().max())
        # final logits == continuation of level NL-1 with actual attention
        blk = blocks[NL - 1]
        x_mid = hs[NL - 1] + cap["attn"][NL - 1]
        x_post = x_mid + blk.mlp(blk.ln_2(x_mid))
        sanity["final_logits_recon"] = float((model.lm_head(ln_f(x_post)) - out.logits).abs().max())
        print("sanity", sanity, flush=True)
        assert max(sanity.values()) < 5e-3, sanity
    final_top1 = out.logits[:, 1:].argmax(-1)
    for l in range(NL + 1):
        ll = model.lm_head(ln_f(hs[l][:, 1:])) if l < NL else out.logits[:, 1:]
        unembed_agree[l] += (ll.argmax(-1) == final_top1).double().mean()
    unembed_n += 1

    for m in range(NL):
        blk = blocks[m]
        x_mid = hs[m] + cap["attn"][m]
        z_act = cap["z"][m][:, 1:].reshape(-1, DFF)
        a_act = blk.mlp.act(z_act)
        acc(S_act[m], a_act, a_act)
        n_mid = blk.ln_2(x_mid)[:, 1:].reshape(-1, D)
        p_iso_act = n_mid @ U_iso
        p_span_act = n_mid @ U_span[m]
        for l in range(m + 1):
            n_l = blk.ln_2(hs[l])[:, 1:].reshape(-1, D)
            z_wb = blk.mlp.c_fc(n_l)
            acc(S_mlp[m][l], z_wb, z_act)
            acc(S_niso[m][l], n_l @ U_iso, p_iso_act)
            acc(S_nspan[m][l], n_l @ U_span[m], p_span_act)
        # attention readers
        qkv_act, q_act, k_act, v_act = qkv_from(blk, hs[m])
        P_act = pattern(q_act, k_act)
        for l in range(m):
            qkv_wb, q_wb, k_wb, v_wb = qkv_from(blk, hs[l])
            acc(S_att[m][l], qkv_wb[:, 1:].reshape(-1, 3 * D), qkv_act[:, 1:].reshape(-1, 3 * D))
            tv, t1 = pat_metrics(pattern(q_wb, k_wb), P_act)
            PAT["tv_full"][m, l] += tv; PAT["top1_full"][m, l] += t1
            tv, t1 = pat_metrics(pattern(q_wb, k_act), P_act)
            PAT["tv_q"][m, l] += tv; PAT["top1_q"][m, l] += t1
            tv, t1 = pat_metrics(pattern(q_act, k_wb), P_act)
            PAT["tv_k"][m, l] += tv; PAT["top1_k"][m, l] += t1
    pat_n += 1
    n_batches += 1
    del out, hs
    cap["attn"].clear(); cap["z"].clear()
    print(f"batch {n_batches}/{N_EVAL // BS} {time.time() - t0:.0f}s", flush=True)

# ---- finish --------------------------------------------------------------------
res = {"model": MODEL, "n_eval_chunks": N_EVAL, "ctx": CTX, "sanity": sanity,
       "unembed_top1_agree": (unembed_agree / unembed_n).cpu().tolist()}
r_mlp, re_mlp, r_iso, r_span = {}, {}, {}, {}
imp = {}
for m in range(NL):
    _, _, va = finish(S_act[m])
    std_a = np.sqrt(np.clip(va, 0, None))
    wnorm = blocks[m].mlp.c_proj.weight.norm(dim=1).cpu().numpy()   # rows = write dirs
    imp[m] = std_a * wnorm
    r_mlp[m] = np.stack([finish(S_mlp[m][l])[0] for l in range(m + 1)])       # [m+1, DFF]
    re_mlp[m] = np.stack([finish(S_mlp[m][l])[1] for l in range(m + 1)])
    r_iso[m] = np.stack([finish(S_niso[m][l])[0] for l in range(m + 1)])
    r_span[m] = np.stack([finish(S_nspan[m][l])[0] for l in range(m + 1)])

mlp = {"r_median": [], "r_wmean": [], "frac_ready": {str(t): [] for t in TAUS},
       "wfrac_ready": {str(t): [] for t in TAUS}, "re_median": [],
       "null_iso_r_median": [], "null_span_r_median": [],
       "null_iso_frac_ready_0.9": [], "null_span_frac_ready_0.9": []}
depth = {str(t): {} for t in TAUS}
for m in range(NL):
    live = imp[m] > np.quantile(imp[m], 0.05)          # drop the 5% least important (dead-ish)
    w = imp[m] * live
    w = w / w.sum()
    mlp["r_median"].append([float(np.median(r_mlp[m][l][live])) for l in range(m + 1)])
    mlp["r_wmean"].append([float((r_mlp[m][l] * w).sum()) for l in range(m + 1)])
    mlp["re_median"].append([float(np.median(re_mlp[m][l][live])) for l in range(m + 1)])
    for t in TAUS:
        mlp["frac_ready"][str(t)].append([float((r_mlp[m][l][live] >= t).mean()) for l in range(m + 1)])
        mlp["wfrac_ready"][str(t)].append([float(((r_mlp[m][l] >= t) * w).sum()) for l in range(m + 1)])
    mlp["null_iso_r_median"].append([float(np.median(r_iso[m][l])) for l in range(m + 1)])
    mlp["null_span_r_median"].append([float(np.median(r_span[m][l])) for l in range(m + 1)])
    mlp["null_iso_frac_ready_0.9"].append([float((r_iso[m][l] >= 0.9).mean()) for l in range(m + 1)])
    mlp["null_span_frac_ready_0.9"].append([float((r_span[m][l] >= 0.9).mean()) for l in range(m + 1)])
    # readiness depth per neuron: earliest l with r >= tau for all l' in [l, m]
    for t in TAUS:
        ok = r_mlp[m] >= t                                  # [m+1, DFF]
        suffix_ok = np.flip(np.cumprod(np.flip(ok, 0), 0), 0).astype(bool)  # all l' >= l ok
        d = np.where(suffix_ok.any(0), suffix_ok.argmax(0), m + 1)          # m+1 = needs attn m
        d = d.astype(float)
        hist = {str(int(k)): float(((d == k) & live).sum() / live.sum()) for k in range(m + 2)}
        whist = {str(int(k)): float(((d == k) * w).sum()) for k in range(m + 2)}
        depth[str(t)][str(m)] = {"hist": hist, "whist": whist,
                                 "median_slack": float(np.median((m - d)[live])),
                                 "frac_lexical": hist["0"], "wfrac_lexical": whist["0"],
                                 "frac_needs_attn": hist[str(m + 1)], "wfrac_needs_attn": whist[str(m + 1)],
                                 "frac_slack_ge2": float((((m - d) >= 2) & live).sum() / live.sum())}
res["mlp"] = mlp
res["mlp_depth"] = depth

att = {k: (v / pat_n).cpu().numpy() for k, v in PAT.items()}
unit = {"q": np.zeros((NL, NL, H)), "k": np.zeros((NL, NL, H)), "v": np.zeros((NL, NL, H))}
for m in range(NL):
    for l in range(m):
        r_u, _, _ = finish(S_att[m][l])                   # [3D]
        for i, nm in enumerate(["q", "k", "v"]):
            unit[nm][m, l] = r_u[i * D:(i + 1) * D].reshape(H, HD).mean(1)
res["attn"] = {k: v.tolist() for k, v in att.items()}
res["attn_unit_r"] = {k: v.tolist() for k, v in unit.items()}
# per-head "ready level": first l at which tv <= 0.2 and stays <= 0.2 up to m-1
def ready_level(tvrow, m, thr=0.2):
    ok = tvrow[:m] <= thr
    suffix_ok = np.flip(np.cumprod(np.flip(ok), 0)).astype(bool)
    return int(suffix_ok.argmax()) if suffix_ok.any() else m
heads = {}
for m in range(1, NL):
    for h in range(H):
        heads[f"L{m}H{h}"] = {"ready_full": ready_level(att["tv_full"][m, :, h], m),
                             "ready_q": ready_level(att["tv_q"][m, :, h], m),
                             "ready_k": ready_level(att["tv_k"][m, :, h], m),
                             "tv_full": [float(x) for x in att["tv_full"][m, :m, h]],
                             "tv_q": [float(x) for x in att["tv_q"][m, :m, h]],
                             "tv_k": [float(x) for x in att["tv_k"][m, :m, h]]}
res["heads"] = heads
res["elapsed_s"] = time.time() - t0
with open(f"{OUT}/exp_r1_readiness_{TAG}.json", "w") as f:
    json.dump(res, f, indent=1)
np.savez_compressed(f"{OUT}/exp_r1_readiness_{TAG}.npz",
                    **{f"r_mlp_{m}": r_mlp[m] for m in range(NL)},
                    **{f"re_mlp_{m}": re_mlp[m] for m in range(NL)},
                    **{f"imp_{m}": imp[m] for m in range(NL)},
                    **{f"r_iso_{m}": r_iso[m] for m in range(NL)},
                    **{f"r_span_{m}": r_span[m] for m in range(NL)},
                    tv_full=att["tv_full"], tv_q=att["tv_q"], tv_k=att["tv_k"],
                    top1_full=att["top1_full"], unit_q=unit["q"], unit_k=unit["k"], unit_v=unit["v"])

# ---- printout -------------------------------------------------------------------
print("\n=== unembedding reader (logit lens) top-1 agreement with final, by level ===")
print(" ".join(f"{x:.3f}" for x in res["unembed_top1_agree"]))
print("\n=== MLP readers: median r of would-be vs actual pre-activation, rows=reader layer m, cols=source level l ===")
for m in range(NL):
    print(f"m={m:2d} " + " ".join(f"{x:.2f}" for x in mlp["r_median"][m]) +
          f"   | null_iso " + " ".join(f"{x:.2f}" for x in mlp["null_iso_r_median"][m]))
print("\n=== readiness depth (tau=0.9): frac lexical (l=0) / frac slack>=2 / frac needs-attn-m / median slack ===")
for m in range(NL):
    d = depth["0.9"][str(m)]
    print(f"m={m:2d} lexical {d['frac_lexical']:.2f}  slack>=2 {d['frac_slack_ge2']:.2f}  needs_attn {d['frac_needs_attn']:.2f}  med_slack {d['median_slack']:.1f}")
print("\n=== attention readers: mean TV(would-be pattern, actual) full/q-only/k-only, rows m, cols l (head-averaged) ===")
for m in range(1, NL):
    print(f"m={m:2d} full " + " ".join(f"{x:.2f}" for x in att["tv_full"][m, :m].mean(-1)) +
          " | q " + " ".join(f"{x:.2f}" for x in att["tv_q"][m, :m].mean(-1)) +
          " | k " + " ".join(f"{x:.2f}" for x in att["tv_k"][m, :m].mean(-1)))
print("\n=== heads with a late key-side readiness (ready_k >= 2) ===")
for name, hd in heads.items():
    if hd["ready_k"] >= 2 or name in ("L5H1", "L5H5", "L4H11", "L6H9", "L7H10", "L7H2"):
        print(f"{name}: ready_full {hd['ready_full']} ready_q {hd['ready_q']} ready_k {hd['ready_k']}  tv_k " +
              " ".join(f"{x:.2f}" for x in hd["tv_k"]))
print(f"\ndone in {res['elapsed_s']:.0f}s -> {OUT}/exp_r1_readiness_{TAG}.json")
