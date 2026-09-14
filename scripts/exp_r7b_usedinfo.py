"""R7: ERASED BUT USED? -- does the model's downstream computation still consume
a write whose direction is gone from the residual stream?

Last week (exp_i/exp_j): for block-3 dominant writes in GPT-2 small, ~13% have
survival (x_7 . d_hat)/c < 0.25 at the post-block-6 state ("erased"), and a
full-state ridge probe still recovers the source activation for about two
thirds of them ("transformation, not erasure"). Linearly recoverable is not
the same as used. This script asks the causal question with the model's own
computation as the reader.

Per case (token t, block-3 neuron j with write c d_hat, c = a_j ||d_j||):
  src-ablate   zero a_j(t) in block 3 and run the model on; measures the write's
               total downstream footprint at position t:
                 E_src_full  = KL(final_clean || final_ablated) at t
                 E_src_mlp   = same through the MLP-only continuation from the
                               post-block-6 state (token-local path only)
                 E_src_sink  = same with the sink+self continuation
  dir-remove   subtract the write's SURVIVING component s7 c d_hat from the
               post-block-6 state and run blocks 7..11 (full attention):
                 E_dir_full, E_dir_mlp -- what a direction-based reading of the
               state would attribute to the write. ~0 for erased writes by
               construction.
  dir-all      subtract the entire projection (x . d_hat) d_hat (robustness).
  rand         subtract a random unit direction scaled to |s7 c| (size control).
Groups, magnitude-matched to the erased-transient group's |c| deciles:
  erased-transient  birth survival >= 0.5 at post-block-3, s7 < 0.25
  erased-counter    birth survival < 0 (a counter-write), s7 < 0.25
  mid               0.25 <= s7 < 0.75, birth >= 0.5
  intact            s7 >= 0.75, birth >= 0.5
Reading: if E_src for erased-transient writes is comparable to intact writes
while E_dir is ~0, the information left its direction but is still consumed
("re-encoded and used"); if E_src ~ rand, it was discarded.
Pre-registered: U1 median E_src_full(erased-transient) >= 0.5 x median
E_src_full(intact) at matched |c|; U2 median E_dir_full(erased-transient) <
0.1 x its E_src_full; U3 intact: E_dir_full / E_src_full median >= 0.5.
Writes results/exp_r7b_usedinfo.json
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
CTX, N_EVAL, BS = 512, 64, 4
SRC_BLOCK, MEAS_LEVEL, BIRTH_LEVEL = 3, 7, 4     # write in block 3; measure at input to block 7 (post block 6); birth = post block 3
TOPK, N_PER_GROUP, CASE_BS = 2, 256, 8
t0 = time.time()

model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").to(DEV).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
NL, D, H = cfg.n_layer, cfg.n_embd, cfg.n_head
HD = D // H
blocks = model.transformer.h
ln_f, head = model.transformer.ln_f, model.lm_head
DFF = blocks[0].mlp.c_fc.weight.shape[1]
W_out = blocks[SRC_BLOCK].mlp.c_proj.weight            # [DFF, D]; rows = write directions
d_norm = W_out.norm(dim=1)                             # [DFF]
d_hat = W_out / d_norm[:, None]


def token_chunks(split, n_chunks):
    text = "\n\n".join(t for t in load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0][: n_chunks * CTX]
    assert len(ids) == n_chunks * CTX
    return ids.view(n_chunks, CTX)


eval_ids = token_chunks("test", N_EVAL)

# ---- hooks -----------------------------------------------------------------------------------
cap = {}
edit = {"kind": None}


def act_hook(mod, inp, out):                          # block-3 post-GELU activations
    cap["a3"] = out
    if edit["kind"] == "ablate":
        out = out.clone()
        b = torch.arange(out.shape[0], device=DEV)
        out[b, edit["pos"], edit["neuron"]] = 0.0
        return out


def mk_level_hook(l):
    def f(mod, args, kwargs):
        x = args[0] if args else kwargs["hidden_states"]
        cap[f"h{l}"] = x
        if edit["kind"] in ("dir", "dirall", "rand") and l == MEAS_LEVEL:
            x = x.clone()
            b = torch.arange(x.shape[0], device=DEV)
            x[b, edit["pos"]] += edit["vec"]
            cap[f"h{l}"] = x
            return (x,) + tuple(args[1:]), kwargs
    return f


assert isinstance(blocks[SRC_BLOCK].mlp.act, torch.nn.Module), type(blocks[SRC_BLOCK].mlp.act)
blocks[SRC_BLOCK].mlp.act.register_forward_hook(act_hook)
for l in sorted({BIRTH_LEVEL, *range(MEAS_LEVEL, NL)}):          # each level hooked ONCE (a duplicate would apply the edit twice)
    blocks[l].register_forward_pre_hook(mk_level_hook(l), with_kwargs=True)

# ---- phase 1: dominant block-3 writes, survival at birth and at the measurement level -------------
cases = []          # (chunk, pos, neuron, c, s_birth, s7)
pos0 = {}           # position-0 states per chunk per level 7..11 (for the sink continuation)
for bi in range(0, N_EVAL, BS):
    ids = eval_ids[bi:bi + BS].to(DEV)
    cap.clear(); edit["kind"] = None
    model(ids)
    a3 = cap["a3"]                                    # [B,T,DFF]
    for l in range(MEAS_LEVEL, NL):
        for b in range(ids.shape[0]):
            pos0[(bi + b, l)] = cap[f"h{l}"][b, 0].clone()
    cmag = a3 * d_norm                                # signed write magnitude c_j(t)
    top = cmag.abs().topk(TOPK, dim=-1).indices       # [B,T,TOPK]
    hb, h7 = cap[f"h{BIRTH_LEVEL}"], cap[f"h{MEAS_LEVEL}"]
    cc = cmag.gather(-1, top)                                        # [B,T,K] signed magnitudes
    dsel = d_hat[top]                                                # [B,T,K,D]
    sbv = torch.einsum("btd,btkd->btk", hb, dsel) / cc
    s7v = torch.einsum("btd,btkd->btk", h7, dsel) / cc
    Bn = ids.shape[0]
    bb = torch.arange(Bn, device=DEV)[:, None, None].expand(Bn, CTX, TOPK)
    tt = torch.arange(CTX, device=DEV)[None, :, None].expand(Bn, CTX, TOPK)
    keep = (tt >= 1) & (tt <= CTX - 2) & (cc.abs() >= 1e-3)
    block = torch.stack([(bb + bi).float(), tt.float(), top.float(), cc, sbv, s7v], -1)[keep].cpu().numpy()
    cases.extend(map(tuple, block))
    print(f"phase1 batch {bi // BS + 1}/{N_EVAL // BS} {time.time() - t0:.0f}s", flush=True)
C = np.array(cases, dtype=np.float64)
chunk, pos, neur, cval, sb, s7 = C[:, 0].astype(int), C[:, 1].astype(int), C[:, 2].astype(int), C[:, 3], C[:, 4], C[:, 5]
absc = np.abs(cval)
groups = {"erased_transient": (sb >= 0.5) & (s7 < 0.25),
          "erased_counter": (sb < 0) & (s7 < 0.25),
          "mid": (sb >= 0.5) & (s7 >= 0.25) & (s7 < 0.75),
          "intact_novel": (sb >= 0.5) & (sb <= 1.5) & (s7 >= 0.75) & (s7 <= 1.5),   # born as new content, persists
          "intact_reinforcing": (sb > 1.5) & (s7 > 1.5)}                               # piles onto existing content
print("population:", {k: int(v.sum()) for k, v in groups.items()}, "of", len(C), flush=True)
rng = np.random.default_rng(0)
ref = np.where(groups["erased_transient"])[0]
edges = np.quantile(absc[ref], np.linspace(0, 1, 9))
per_bin = min(N_PER_GROUP, len(ref)) // 8
sel = {}
for gname, mask in groups.items():
    idx = np.where(mask)[0]
    picks = []
    for bnum in range(8):
        lo, hi = edges[bnum], edges[bnum + 1]
        cand = idx[(absc[idx] >= lo) & (absc[idx] <= hi)]
        if len(cand):
            picks.extend(rng.choice(cand, min(per_bin, len(cand)), replace=False).tolist())
    sel[gname] = np.array(picks)
    print(f"selected {gname}: {len(picks)} cases, median |c| {np.median(absc[sel[gname]]):.2f}, "
          f"median s7 {np.median(s7[sel[gname]]):.2f}, median birth {np.median(sb[sel[gname]]):.2f}", flush=True)

# ---- continuation helpers (single position, batched over cases) ----------------------------------
def cont_mlp(X):                                       # X [n, D] level-7 states -> logits [n, V]
    x = X[:, None, :]
    for m in range(MEAS_LEVEL, NL):
        x = x + blocks[m].mlp(blocks[m].ln_2(x))
    return head(ln_f(x[:, 0]))


def cont_sink(X, chunks):
    x = X[:, None, :]
    for m in range(MEAS_LEVEL, NL):
        blk = blocks[m]
        qkv = blk.attn.c_attn(blk.ln_1(x)); q, k, v = qkv.split(D, dim=-1)
        p0 = torch.stack([pos0[(int(cb), m)] for cb in chunks])[:, None, :]
        _, k0, v0 = blk.attn.c_attn(blk.ln_1(p0)).split(D, dim=-1)
        sh = lambda z: z.view(z.shape[0], 1, H, HD)
        s_self = (sh(q) * sh(k)).sum(-1) * HD ** -0.5; s_sink = (sh(q) * sh(k0)).sum(-1) * HD ** -0.5
        w = torch.softmax(torch.stack([s_sink, s_self]), dim=0)
        o = (w[0][..., None] * sh(v0) + w[1][..., None] * sh(v)).reshape(x.shape)
        x = x + blk.attn.c_proj(o)
        x = x + blk.mlp(blk.ln_2(x))
    return head(ln_f(x[:, 0]))


def kl_rows(lp_a, lp_b):                               # KL(a || b) per row
    return (lp_a.exp() * (lp_a - lp_b)).sum(-1)


# ---- phase 2: interventions -------------------------------------------------------------------------
res = {"model": MODEL, "src_block": SRC_BLOCK, "meas_level": MEAS_LEVEL, "n_population": int(len(C)),
       "population": {k: int(v.sum()) for k, v in groups.items()}, "groups": {}}
per_case = {}
for gname, idx in sel.items():
    rows = []
    for ci in range(0, len(idx), CASE_BS):
        ii = idx[ci:ci + CASE_BS]
        ids = eval_ids[chunk[ii]].to(DEV)
        b = torch.arange(len(ii), device=DEV)
        P = torch.as_tensor(pos[ii], device=DEV); J = torch.as_tensor(neur[ii], device=DEV)
        cc = torch.as_tensor(cval[ii], device=DEV, dtype=torch.float32)
        ss7 = torch.as_tensor(s7[ii], device=DEV, dtype=torch.float32)
        dh = d_hat[J]                                                    # [n, D]
        # clean
        cap.clear(); edit["kind"] = None
        lp_clean = torch.log_softmax(model(ids).logits[b, P].float(), -1)
        x7_clean = cap[f"h{MEAS_LEVEL}"][b, P].clone()
        lp_cm_clean = torch.log_softmax(cont_mlp(x7_clean).float(), -1)
        lp_cs_clean = torch.log_softmax(cont_sink(x7_clean, chunk[ii]).float(), -1)
        # source ablation
        cap.clear(); edit.update(kind="ablate", pos=P, neuron=J)
        lp_src = torch.log_softmax(model(ids).logits[b, P].float(), -1)
        x7_src = cap[f"h{MEAS_LEVEL}"][b, P].clone()
        edit["kind"] = None
        e_src_full = kl_rows(lp_clean, lp_src)
        e_src_mlp = kl_rows(lp_cm_clean, torch.log_softmax(cont_mlp(x7_src).float(), -1))
        e_src_sink = kl_rows(lp_cs_clean, torch.log_softmax(cont_sink(x7_src, chunk[ii]).float(), -1))
        state_shift = (x7_src - x7_clean).norm(dim=-1) / x7_clean.norm(dim=-1)
        # direction removals at the measurement level
        outs = {}
        for kind, vec in [("dir", -(ss7 * cc)[:, None] * dh),
                          ("dirown", -(cc)[:, None] * dh),                                   # undo the write's OWN share along its direction
                          ("dirall", -((x7_clean * dh).sum(-1))[:, None] * dh),
                          ("rand", torch.nn.functional.normalize(torch.randn_like(dh), dim=-1) * (ss7 * cc).abs()[:, None])]:
            cap.clear(); edit.update(kind=kind, pos=P, vec=vec)
            lp = torch.log_softmax(model(ids).logits[b, P].float(), -1)
            edit["kind"] = None
            outs[kind + "_full"] = kl_rows(lp_clean, lp)
            outs[kind + "_mlp"] = kl_rows(lp_cm_clean, torch.log_softmax(cont_mlp(x7_clean + vec).float(), -1))
        for n_ in range(len(ii)):
            rows.append({"chunk": int(chunk[ii[n_]]), "pos": int(pos[ii[n_]]), "neuron": int(neur[ii[n_]]), "c": float(cval[ii[n_]]),
                         "s_birth": float(sb[ii[n_]]), "s7": float(s7[ii[n_]]),
                         "E_src_full": float(e_src_full[n_]), "E_src_mlp": float(e_src_mlp[n_]), "E_src_sink": float(e_src_sink[n_]),
                         "E_dir_full": float(outs["dir_full"][n_]), "E_dir_mlp": float(outs["dir_mlp"][n_]),
                         "E_dirall_full": float(outs["dirall_full"][n_]), "E_rand_full": float(outs["rand_full"][n_]),
                         "E_dirown_full": float(outs["dirown_full"][n_]), "E_dirown_mlp": float(outs["dirown_mlp"][n_]),
                         "state_shift": float(state_shift[n_])})
    per_case[gname] = rows
    keys = ["E_src_full", "E_src_mlp", "E_src_sink", "E_dir_full", "E_dir_mlp", "E_dirall_full", "E_rand_full", "E_dirown_full", "E_dirown_mlp", "state_shift"]
    summ = {k: {"median": float(np.median([r[k] for r in rows])), "mean": float(np.mean([r[k] for r in rows]))} for k in keys}
    ratio = np.array([r["E_dir_full"] / max(r["E_src_full"], 1e-9) for r in rows])
    summ["dir_over_src_median"] = float(np.median(ratio))
    summ["dirown_over_src_median"] = float(np.median([r["E_dirown_full"] / max(r["E_src_full"], 1e-9) for r in rows]))
    summ["frac_src_above_rand"] = float(np.mean([r["E_src_full"] > r["E_rand_full"] for r in rows]))
    summ["n"] = len(rows)
    res["groups"][gname] = summ
    print(f"{gname:17s} n={len(rows)}  E_src_full med {summ['E_src_full']['median']:.4f}  E_src_mlp {summ['E_src_mlp']['median']:.4f}  "
          f"E_src_sink {summ['E_src_sink']['median']:.4f}  E_dir_full {summ['E_dir_full']['median']:.4f}  E_dirall {summ['E_dirall_full']['median']:.4f}  "
          f"E_rand {summ['E_rand_full']['median']:.4f}  E_dirown {summ['E_dirown_full']['median']:.4f}  dir/src {summ['dir_over_src_median']:.2f}  dirown/src {summ['dirown_over_src_median']:.2f}  shift {summ['state_shift']['median']:.3f}  {time.time() - t0:.0f}s", flush=True)

G = res["groups"]
res["pre_registered"] = {
    "U1_erased_src_ge_half_intact_novel": G["erased_transient"]["E_src_full"]["median"] >= 0.5 * G["intact_novel"]["E_src_full"]["median"],
    "U1_ratio_vs_novel": G["erased_transient"]["E_src_full"]["median"] / max(G["intact_novel"]["E_src_full"]["median"], 1e-9),
    "U1_ratio_vs_reinforcing": G["erased_transient"]["E_src_full"]["median"] / max(G["intact_reinforcing"]["E_src_full"]["median"], 1e-9),
    "U2_erased_dir_lt_0.1_src": G["erased_transient"]["E_dir_full"]["median"] < 0.1 * G["erased_transient"]["E_src_full"]["median"],
    "U3_intact_novel_dirown_over_src_ge_0.5": G["intact_novel"]["dirown_over_src_median"] >= 0.5}
res["per_case"] = per_case
res["elapsed_s"] = time.time() - t0
with open(f"{OUT}/exp_r7b_usedinfo.json", "w") as f:
    json.dump(res, f, indent=1)
print("pre-registered:", res["pre_registered"])
print(f"done {res['elapsed_s']:.0f}s")
