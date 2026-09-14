"""R1b: circuit TIMING with the reader lens -- induction on synthetic repeats.

R1's natural-text pattern metric averages over every query, and most queries
attend to the sink, so a head whose circuit fires rarely can look "ready from
level 1" (L5H1) while its actual circuit input is not. Two sharper measures:
 (a) SYNTHETIC. 64 sequences: 256 random tokens repeated twice. In the second
     copy the induction target of query t is t-255 (the token that followed the
     previous occurrence), the duplicate-token target is t-256, the previous
     token is t-1. Score of a pattern = attention mass at the target. For each
     head and source level l, the score of the would-be pattern (full, Q-only,
     K-only) is compared with the actual score. The three-head induction
     circuit predicts: the previous-token head L4H11 writes "previous token was
     X" at block 4, so the induction heads' KEY side should reach its actual
     score only from l = 5, while the QUERY side (current token) is ready early.
 (b) NATURAL TEXT (16 test chunks). TV restricted to ACTIVE queries (actual
     argmax is not position 0 and actual max prob >= 0.3), per head and level.
Pre-registered E2': for L5H1 and L5H5, K-side induction score at l <= 4 is
below 50% of actual, and reaches >= 80% at l = 5; Q-side reaches >= 80% by
l <= 2.
Writes results/exp_r1b_induction.json.
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
CTX, HALF = 512, 256
N_SYN, N_NAT, BS = 64, 16, 4
t0 = time.time()

model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").to(DEV).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
NL, D, H = cfg.n_layer, cfg.n_embd, cfg.n_head
HD = D // H
blocks = model.transformer.h
causal = torch.full((CTX, CTX), float("-inf"), device=DEV).triu(1)


def split_heads(x):
    B, T, _ = x.shape
    return x.view(B, T, H, HD).transpose(1, 2)


def pattern(q, k):
    s = torch.matmul(q, k.transpose(-1, -2)) * (HD ** -0.5) + causal
    return torch.softmax(s.float(), dim=-1)


def qk_from(blk, x):
    qkv = blk.attn.c_attn(blk.ln_1(x))
    q, k, _ = qkv.split(D, dim=-1)
    return split_heads(q), split_heads(k)


# ---------------- (a) synthetic repeats ------------------------------------------------
g = torch.Generator().manual_seed(0)
half = torch.randint(1000, 40000, (N_SYN, HALF), generator=g)
syn = torch.cat([half, half], dim=1)
tq = torch.arange(HALF + 1, CTX, device=DEV)                 # second-copy queries (skip its first position)
targets = {"induction": tq - (HALF - 1), "duplicate": tq - HALF, "previous": tq - 1}
score = {k: torch.zeros(NL, NL + 1, 3, H, dtype=torch.float64, device=DEV) for k in targets}  # [m, l(0..m), variant, head]; l=m slot = actual
ent_actual = torch.zeros(NL, H, dtype=torch.float64, device=DEV)
nb = 0
for bi in range(0, N_SYN, BS):
    ids = syn[bi:bi + BS].to(DEV)
    hs = model(ids, output_hidden_states=True).hidden_states
    for m in range(NL):
        blk = blocks[m]
        q_act, k_act = qk_from(blk, hs[m])
        P_act = pattern(q_act, k_act)
        for name, tg in targets.items():
            score[name][m, m, :] += P_act[:, :, tq, :].gather(-1, tg.view(1, 1, -1, 1).expand(P_act.shape[0], H, -1, 1)).squeeze(-1).double().mean((0, 2))
        for l in range(m):
            q_wb, k_wb = qk_from(blk, hs[l])
            for vi, (qq, kk) in enumerate([(q_wb, k_wb), (q_wb, k_act), (q_act, k_wb)]):
                P = pattern(qq, kk)
                for name, tg in targets.items():
                    score[name][m, l, vi] += P[:, :, tq, :].gather(-1, tg.view(1, 1, -1, 1).expand(P.shape[0], H, -1, 1)).squeeze(-1).double().mean((0, 2))
    nb += 1
    del hs
    print(f"syn batch {nb}/{N_SYN // BS} {time.time() - t0:.0f}s", flush=True)
score = {k: (v / nb).cpu().numpy() for k, v in score.items()}

# ---------------- (b) natural text, active queries ----------------------------------------
def token_chunks(split, n_chunks):
    text = "\n\n".join(t for t in load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0][: n_chunks * CTX]
    assert len(ids) == n_chunks * CTX
    return ids.view(n_chunks, CTX)


nat = token_chunks("test", N_NAT)
tv_active = torch.zeros(NL, NL, 3, H, dtype=torch.float64, device=DEV)
n_active = torch.zeros(NL, H, dtype=torch.float64, device=DEV)
n_queries = 0
for bi in range(0, N_NAT, BS):
    ids = nat[bi:bi + BS].to(DEV)
    hs = model(ids, output_hidden_states=True).hidden_states
    for m in range(NL):
        blk = blocks[m]
        q_act, k_act = qk_from(blk, hs[m])
        P_act = pattern(q_act, k_act)                                  # [B,H,T,T]
        mx, am = P_act.max(-1)
        active = ((am != 0) & (mx >= 0.3))[:, :, 1:]                   # [B,H,T-1]
        n_active[m] += active.double().sum((0, 2))
        for l in range(m):
            q_wb, k_wb = qk_from(blk, hs[l])
            for vi, (qq, kk) in enumerate([(q_wb, k_wb), (q_wb, k_act), (q_act, k_wb)]):
                P = pattern(qq, kk)
                tv = 0.5 * (P[:, :, 1:] - P_act[:, :, 1:]).abs().sum(-1)   # [B,H,T-1]
                tv_active[m, l, vi] += (tv * active).double().sum((0, 2))
    n_queries += ids.shape[0] * (CTX - 1)
    del hs
    print(f"nat batch {bi // BS + 1}/{N_NAT // BS} {time.time() - t0:.0f}s", flush=True)
tv_active = (tv_active / n_active.clamp_min(1)[:, None, None, :]).cpu().numpy()
frac_active = (n_active / n_queries).cpu().numpy()

# ---------------- report ------------------------------------------------------------------
VAR = ["full", "q_only", "k_only"]
res = {"model": MODEL, "n_syn": N_SYN, "n_nat": N_NAT, "heads": {}}
for m in range(NL):
    for h in range(H):
        d = {"frac_active_nat": float(frac_active[m, h])}
        for name in targets:
            d[f"{name}_actual"] = float(score[name][m, m, 0, h])
            for vi, v in enumerate(VAR):
                d[f"{name}_{v}"] = [float(score[name][m, l, vi, h]) for l in range(m)]
        for vi, v in enumerate(VAR):
            d[f"tv_active_{v}"] = [float(tv_active[m, l, vi, h]) for l in range(m)]
        res["heads"][f"L{m}H{h}"] = d


def first_level(vals, actual, frac):
    for l, x in enumerate(vals):
        if x >= frac * actual:
            return l
    return len(vals)


print("\n=== heads with actual induction score >= 0.2 (synthetic): K-side / Q-side level reaching 80% of actual ===")
ind_heads = []
for m in range(NL):
    for h in range(H):
        d = res["heads"][f"L{m}H{h}"]
        if d["induction_actual"] >= 0.2:
            ind_heads.append(f"L{m}H{h}")
            lk = first_level(d["induction_k_only"], d["induction_actual"], 0.8)
            lq = first_level(d["induction_q_only"], d["induction_actual"], 0.8)
            lf = first_level(d["induction_full"], d["induction_actual"], 0.8)
            d.update({"k80": lk, "q80": lq, "full80": lf})
            print(f"L{m}H{h}: actual {d['induction_actual']:.2f}  k80@{lk} q80@{lq} full80@{lf}   "
                  f"k-only " + " ".join(f"{x:.2f}" for x in d["induction_k_only"]) +
                  "   q-only " + " ".join(f"{x:.2f}" for x in d["induction_q_only"]))
print("\n=== previous-token heads (actual prev score >= 0.3) ===")
for m in range(NL):
    for h in range(H):
        d = res["heads"][f"L{m}H{h}"]
        if d["previous_actual"] >= 0.3:
            print(f"L{m}H{h}: actual {d['previous_actual']:.2f}  full " + " ".join(f"{x:.2f}" for x in d["previous_full"]))
print("\n=== duplicate-token heads (actual dup score >= 0.2) ===")
for m in range(NL):
    for h in range(H):
        d = res["heads"][f"L{m}H{h}"]
        if d["duplicate_actual"] >= 0.2:
            print(f"L{m}H{h}: actual {d['duplicate_actual']:.2f}  k-only " + " ".join(f"{x:.2f}" for x in d["duplicate_k_only"]) +
                  "  q-only " + " ".join(f"{x:.2f}" for x in d["duplicate_q_only"]))
print("\n=== natural text, ACTIVE-query TV (full), head-averaged, rows m cols l; and frac active ===")
for m in range(1, NL):
    print(f"m={m:2d} " + " ".join(f"{tv_active[m, l, 0].mean():.2f}" for l in range(m)) + f"   frac_active {frac_active[m].mean():.2f}")
for name in ["L4H11", "L5H1", "L5H5", "L6H9", "L7H2", "L7H10"]:
    d = res["heads"][name]
    print(f"{name} active-TV full " + " ".join(f"{x:.2f}" for x in d["tv_active_full"]) + f"  (active {d['frac_active_nat']:.2f})")
res["induction_heads"] = ind_heads
res["elapsed_s"] = time.time() - t0
with open(f"{OUT}/exp_r1b_induction.json", "w") as f:
    json.dump(res, f, indent=1)
print(f"done {res['elapsed_s']:.0f}s")
