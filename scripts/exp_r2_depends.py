"""R2: the DEPENDENCY MAP -- causal half of the reader lens.

The readiness map (R1) is correlational: a reader's would-be activation at
level l agrees with its actual activation. This script makes each reader
ACTUALLY read the earlier state and measures what the model loses. The
counterfactual is the model's own earlier state in the same forward pass, so
there is no corruption prompt, no second run and no off-distribution source:
"early-read patching" = zero-ablation of the edge from the intervening layers
into one reader's input, everything else untouched (the residual add still
uses the true state).

  MLP edge  (l, m), l <= m : block m's MLP reads hidden_states[l] instead of
                             resid_mid_m.  l = m removes only attention m.
  attn edge (l, m), l <  m : block m's attention (Q, K, V) reads
                             hidden_states[l] instead of hidden_states[m].
  neuron groups            : at chosen (l, m), only a SUBSET of block m's
                             neurons read early -- the R1 "ready" set
                             (r >= 0.9), the "not ready" set (r < 0.6) and a
                             random set of equal size.  Tests whether
                             correlational readiness is functionally real.
Metrics: dCE (mean over positions 1..T-2), KL(clean || patched), top-1 flip
rate, on 64 WikiText-2 test chunks.

Pre-registered: E4 across the 78 MLP-edge cells, dCE is anti-correlated with
R1's importance-weighted mean readiness (Spearman <= -0.7); ready groups cost
< 20% of what equal-size not-ready groups cost.
Writes results/exp_r2_depends.json.
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(0)
torch.set_grad_enabled(False)
OUT = RESULTS_DIR
DEV = os.environ.get("R_DEV", "cuda:0")
MODEL = "gpt2"
CTX, N_EVAL, BS = 512, 64, 8
t0 = time.time()

model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEV).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
NL, D = cfg.n_layer, cfg.n_embd
blocks = model.transformer.h
DFF = blocks[0].mlp.c_fc.weight.shape[1]


def token_chunks(split, n_chunks):
    text = "\n\n".join(t for t in load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0][: n_chunks * CTX]
    assert len(ids) == n_chunks * CTX
    return ids.view(n_chunks, CTX)


eval_ids = token_chunks("test", N_EVAL)

# ---- hooks -------------------------------------------------------------------
state = {"active": None, "stash": {}, "group": None}


def mk_stash(l):
    def f(mod, args):
        state["stash"][l] = args[0]
    return f


def mk_replace_ln1(m):
    def f(mod, args):
        a = state["active"]
        if a and a[0] == "attn" and a[2] == m:
            return (state["stash"][a[1]],) + tuple(args[1:])
    return f


def mk_replace_ln2(m):
    def f(mod, args):
        a = state["active"]
        if a and a[0] == "mlp" and a[2] == m:
            return (state["stash"][a[1]],) + tuple(args[1:])
    return f


def mk_group_hook(m):
    def f(mod, inp, out):
        a = state["active"]
        if a and a[0] == "group" and a[2] == m:
            blk = blocks[m]
            z_wb = blk.mlp.c_fc(blk.ln_2(state["stash"][a[1]]))
            idx = state["group"]
            out = out.clone()
            out[..., idx] = z_wb[..., idx]
            return out
    return f


for m, blk in enumerate(blocks):
    blk.ln_1.register_forward_pre_hook(mk_stash(m))          # registered first: sees the true input
    blk.ln_1.register_forward_pre_hook(mk_replace_ln1(m))
    blk.ln_2.register_forward_pre_hook(mk_replace_ln2(m))
    blk.mlp.c_fc.register_forward_hook(mk_group_hook(m))

# ---- clean reference -----------------------------------------------------------
clean_lp = []
clean_ce = 0.0
n_tok = 0
for bi in range(0, N_EVAL, BS):
    ids = eval_ids[bi:bi + BS].to(DEV)
    lp = torch.log_softmax(model(ids).logits[:, 1:-1].float(), -1)
    tgt = ids[:, 2:]
    clean_ce += float(-lp.gather(-1, tgt[..., None]).sum())
    n_tok += tgt.numel()
    clean_lp.append(lp.half().cpu())
    state["stash"].clear()
clean_ce /= n_tok
print(f"clean CE {clean_ce:.4f} over {n_tok} tokens  ({time.time() - t0:.0f}s)", flush=True)


def run(active, group=None):
    state["active"] = active
    state["group"] = group
    ce = kl = flip = 0.0
    for i, bi in enumerate(range(0, N_EVAL, BS)):
        ids = eval_ids[bi:bi + BS].to(DEV)
        lp = torch.log_softmax(model(ids).logits[:, 1:-1].float(), -1)
        tgt = ids[:, 2:]
        ce += float(-lp.gather(-1, tgt[..., None]).sum())
        c = clean_lp[i].to(DEV).float()
        kl += float((c.exp() * (c - lp)).sum())
        flip += float((c.argmax(-1) != lp.argmax(-1)).sum())
        state["stash"].clear()
    state["active"] = None
    return {"dce": ce / n_tok - clean_ce, "kl": kl / n_tok, "flip": flip / n_tok}


res = {"model": MODEL, "clean_ce": clean_ce, "n_tokens": n_tok, "mlp_edge": {}, "attn_edge": {}}
# sanity: patching a reader with its own input must be a no-op
s = run(("attn", 3, 3))
print("sanity attn(3,3) (identity):", s, flush=True)
res["sanity_identity_attn"] = s
for m in range(NL):
    for l in range(m + 1):
        r = run(("mlp", l, m))
        res["mlp_edge"][f"{l}|{m}"] = r
        print(f"mlp edge l={l:2d} m={m:2d}  dCE {r['dce']:+.4f}  KL {r['kl']:.4f}  flip {r['flip']:.3f}", flush=True)
    for l in range(m):
        r = run(("attn", l, m))
        res["attn_edge"][f"{l}|{m}"] = r
        print(f"attn edge l={l:2d} m={m:2d}  dCE {r['dce']:+.4f}  KL {r['kl']:.4f}  flip {r['flip']:.3f}", flush=True)
    with open(f"{OUT}/exp_r2_depends.json", "w") as f:
        json.dump(res, f, indent=1)
print(f"layer-level map done {time.time() - t0:.0f}s", flush=True)

# ---- neuron groups (needs R1) ------------------------------------------------------
npz_path = f"{OUT}/exp_r1_readiness_gpt2.npz"
waited = 0
while not os.path.exists(npz_path) and waited < 3600:
    time.sleep(30); waited += 30
if not os.path.exists(npz_path):
    print("R1 npz not found; skipping neuron groups", flush=True)
    sys.exit(0)
time.sleep(5)
R1 = np.load(npz_path)
rng = np.random.default_rng(0)
PAIRS = [(0, 4), (2, 4), (4, 4), (0, 8), (6, 8), (8, 8), (0, 11), (9, 11), (11, 11)]
res["groups"] = {}
for (l, m) in PAIRS:
    r = R1[f"r_mlp_{m}"][l]
    imp = R1[f"imp_{m}"]
    ready = np.where(r >= 0.9)[0]
    notready = np.where(r < 0.6)[0]
    n = int(min(len(ready), len(notready)))
    if n < 20:
        res["groups"][f"{l}|{m}"] = {"skipped": True, "n_ready": int(len(ready)), "n_notready": int(len(notready))}
        continue
    g_ready = rng.choice(ready, n, replace=False)
    g_not = rng.choice(notready, n, replace=False)
    g_rand = rng.choice(DFF, n, replace=False)
    outg = {"n": n, "n_ready_total": int(len(ready)), "n_notready_total": int(len(notready)),
            "imp_ready": float(imp[g_ready].sum()), "imp_notready": float(imp[g_not].sum()),
            "imp_random": float(imp[g_rand].sum())}
    for name, idx in [("ready", g_ready), ("notready", g_not), ("random", g_rand)]:
        outg[name] = run(("group", l, m), group=torch.as_tensor(idx, device=DEV))
    res["groups"][f"{l}|{m}"] = outg
    print(f"groups l={l} m={m} n={n}: ready dCE {outg['ready']['dce']:+.4f} (imp {outg['imp_ready']:.1f})  "
          f"notready {outg['notready']['dce']:+.4f} (imp {outg['imp_notready']:.1f})  random {outg['random']['dce']:+.4f}",
          flush=True)
    with open(f"{OUT}/exp_r2_depends.json", "w") as f:
        json.dump(res, f, indent=1)

# ---- E4: readiness vs dCE across cells -----------------------------------------------
try:
    r1 = json.load(open(f"{OUT}/exp_r1_readiness_gpt2.json"))
    xs, ys = [], []
    for m in range(NL):
        for l in range(m + 1):
            xs.append(r1["mlp"]["r_wmean"][m][l]); ys.append(res["mlp_edge"][f"{l}|{m}"]["dce"])
    from scipy.stats import spearmanr
    rho = spearmanr(xs, ys).correlation
    res["E4_spearman_readiness_vs_dce"] = float(rho)
    print(f"E4: Spearman(readiness, dCE) over {len(xs)} MLP-edge cells = {rho:+.3f}", flush=True)
except Exception as e:
    print("E4 skipped:", repr(e))
res["elapsed_s"] = time.time() - t0
with open(f"{OUT}/exp_r2_depends.json", "w") as f:
    json.dump(res, f, indent=1)
print(f"done {res['elapsed_s']:.0f}s")
