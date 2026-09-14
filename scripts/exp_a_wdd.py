"""Experiment A: Weight-Dictionary Decomposition (WDD).

Hypothesis: the residual stream is approximately sparse in the model's OWN
innate write-dictionary -- the union of token-embedding rows, positional rows,
attention per-head OV write-subspaces, MLP down-projection rows, and bias
vectors of all blocks up to the readout layer. If true, inference-time OMP over
this fixed, data-free dictionary gives SAE-style sparse decomposition with
ZERO training: no learned dictionary (contra SAEs), no activation sampling
(contra ITDA), no factorization fit (contra SNMF/ICA).

GPT-2 small. Conv1D stores weight [in, out], so c_proj rows ARE per-neuron
write directions -- no transpose needed here (the superweight-study transpose
trap applies to [row=out, col=in] coordinate lookups, the other direction).
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

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(0)
torch.set_grad_enabled(False)

OUT = RESULTS_DIR
os.makedirs(OUT, exist_ok=True)

free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
DEV = f"cuda:{max(range(len(free)), key=lambda i: free[i])}"
print(f"device {DEV} ({[f//2**20 for f in free]} MiB free)", flush=True)

CTX = 512
N_TRAIN_CHUNKS = 400          # ~205k tokens: mean, PCA, activation-dict control
N_EVAL_CHUNKS = 64            # ~33k held-out tokens: all reported numbers
LAYERS = [6, 8]               # decompose resid_post of these blocks
K_MAX = 64
K_CHECKPOINTS = [4, 8, 16, 32, 64]
K_PATCH = [32, 64]            # store recons at these sparsities for fidelity
OMP_BATCH = 768

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
D = model.config.n_embd
HEAD_DIM = D // model.config.n_head
W_U = model.transformer.wte.weight            # tied unembedding [50257, 768]


def token_chunks(split, n_chunks):
    text = "\n\n".join(t for t in load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    ids = ids[: n_chunks * CTX]
    assert len(ids) == n_chunks * CTX, f"only {len(ids)} tokens in {split}"
    return ids.view(n_chunks, CTX)


train_ids = token_chunks("train", N_TRAIN_CHUNKS)
eval_ids = token_chunks("test", N_EVAL_CHUNKS)
print(f"tokens: train {train_ids.numel()}, eval {eval_ids.numel()}", flush=True)


def collect(ids, layers, want_logits=False, batch=4):
    """resid_post activations at `layers` (fp32 CPU) and optionally final logits."""
    acts = {L: [] for L in layers}
    logits = []
    for i in range(0, len(ids), batch):
        out = model(ids[i:i + batch].to(DEV), output_hidden_states=True)
        for L in layers:
            acts[L].append(out.hidden_states[L + 1].float().cpu())  # hs[i+1] = post block i
        if want_logits:
            logits.append(out.logits.float().cpu())
        del out
    acts = {L: torch.cat(a).view(-1, D) for L, a in acts.items()}
    return (acts, torch.cat(logits)) if want_logits else acts


# sanity: hs[12] is post-ln_f; unembedding it must reproduce the model's logits
out = model(eval_ids[:1].to(DEV), output_hidden_states=True)
assert torch.allclose(out.hidden_states[12] @ W_U.T, out.logits, atol=1e-3), \
    "hidden_states indexing does not match expectation"
del out

print("collecting activations...", flush=True)
t0 = time.time()
train_acts = collect(train_ids, LAYERS, batch=8)
eval_acts, eval_logits = collect(eval_ids, LAYERS, want_logits=True)
print(f"  done in {time.time()-t0:.0f}s", flush=True)

MU = {L: train_acts[L].mean(0) for L in LAYERS}

clean_ce = F.cross_entropy(
    eval_logits[:, :-1].reshape(-1, eval_logits.shape[-1]),
    eval_ids[:, 1:].reshape(-1), reduction="mean").item()
print(f"clean eval CE {clean_ce:.4f}", flush=True)

# clean log-probs for KL, kept on CPU; raw logits no longer needed after this
clean_lp = torch.empty_like(eval_logits)
for i in range(N_EVAL_CHUNKS):
    clean_lp[i] = F.log_softmax(eval_logits[i], -1)
del eval_logits


# ---------------------------------------------------------------- dictionaries
def unit(a):
    return a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def weight_dictionary(L):
    """Every direction the model could have written into resid_post(L).
    Built from weights alone -- zero forward passes, zero data."""
    atoms, fams = [], []

    def add(t, fam):
        atoms.append(t.detach().float().cpu())
        fams.extend([fam] * t.shape[0])

    add(model.transformer.wte.weight, "wte")
    add(model.transformer.wpe.weight, "wpe")
    for b in range(L + 1):
        blk = model.transformer.h[b]
        add(blk.mlp.c_proj.weight, f"mlp{b}")            # Conv1D: rows = writes
        w_o = blk.attn.c_proj.weight                     # [768(concat heads), 768]
        for h in range(model.config.n_head):
            hb = w_o[h * HEAD_DIM:(h + 1) * HEAD_DIM, :]  # head write map [64, 768]
            add(torch.linalg.svd(hb, full_matrices=False).Vh, f"attn{b}")
        add(torch.stack([blk.attn.c_proj.bias, blk.mlp.c_proj.bias]), f"bias{b}")
    return unit(torch.cat(atoms)), fams


def random_dictionary(n):
    return unit(torch.randn(n, D, generator=torch.Generator().manual_seed(1)))


def activation_dictionary(L, n):
    """ITDA-style control: atoms are activations sampled from held-in data."""
    idx = torch.randperm(len(train_acts[L]),
                         generator=torch.Generator().manual_seed(2))[:n]
    return unit(train_acts[L][idx] - MU[L])


# ------------------------------------------------------------------------- OMP
def omp(X, A, denom):
    """Batched orthogonal matching pursuit of rows of X over dictionary A.
    Returns FVU (vs `denom`) at each checkpoint, recons at K_PATCH, sels@32."""
    A_dev = A.to(DEV)
    err = {k: 0.0 for k in K_CHECKPOINTS}
    tok = {k: [] for k in K_CHECKPOINTS}      # per-token squared residuals, for state-level cuts (see exp_z2_recompute)
    recons = {k: [] for k in K_PATCH}
    sel32 = []
    eye = torch.eye(K_MAX, device=DEV)
    for s in range(0, len(X), OMP_BATCH):
        x = X[s:s + OMP_BATCH].to(DEV)                       # [B, D]
        B = len(x)
        sel = torch.zeros(B, K_MAX, dtype=torch.long, device=DEV)
        taken = torch.zeros(B, A_dev.shape[0], dtype=torch.bool, device=DEV)
        r = x.clone()
        for k in range(K_MAX):
            corr = (r @ A_dev.T).abs_().masked_fill_(taken, -1.0)
            pick = corr.argmax(-1)
            del corr
            sel[:, k] = pick
            taken.scatter_(1, pick[:, None], True)
            A_S = A_dev[sel[:, :k + 1]]                      # [B, k+1, D]
            G = A_S @ A_S.transpose(1, 2) + 1e-5 * eye[:k + 1, :k + 1]
            c = torch.cholesky_solve(A_S @ x[:, :, None], torch.linalg.cholesky(G))
            recon = (c.transpose(1, 2) @ A_S)[:, 0]
            r = x - recon
            kk = k + 1
            if kk in K_CHECKPOINTS:
                e_tok = (r ** 2).sum(-1)
                err[kk] += e_tok.sum().item(); tok[kk].append(e_tok.cpu())
            if kk in K_PATCH:
                recons[kk].append(recon.half().cpu())
        sel32.append(sel[:, :32].cpu())
    fvu = {k: err[k] / denom for k in K_CHECKPOINTS}
    return fvu, {k: torch.cat(v) for k, v in recons.items()}, torch.cat(sel32), {k: torch.cat(v).numpy() for k, v in tok.items()}


TOKENS = {}


# -------------------------------------------------------------------- fidelity
def patched_metrics(L, recon, mu):
    """Replace resid_post(L) with recon+mu everywhere; ΔCE and KL vs clean."""
    full = (recon.float() + mu).view(N_EVAL_CHUNKS, CTX, D)

    state = {}

    def hook(mod, inp, out):
        rep = state["rep"].to(DEV)
        if isinstance(out, tuple):
            return (rep,) + out[1:]
        return rep

    h = model.transformer.h[L].register_forward_hook(hook)
    ce_sum = kl_sum = n_ce = n_kl = 0
    B = 4
    try:
        for i in range(0, N_EVAL_CHUNKS, B):
            state["rep"] = full[i:i + B]
            lg = model(eval_ids[i:i + B].to(DEV)).logits.float()
            tgt = eval_ids[i:i + B, 1:].to(DEV)
            ce_sum += F.cross_entropy(
                lg[:, :-1].reshape(-1, lg.shape[-1]), tgt.reshape(-1),
                reduction="sum").item()
            n_ce += tgt.numel()
            lp = F.log_softmax(lg, -1).cpu()
            del lg
            cl = clean_lp[i:i + B]
            kl_sum += (cl.exp() * (cl - lp)).sum().item()
            n_kl += lp.shape[0] * lp.shape[1]
    finally:
        h.remove()
        torch.cuda.empty_cache()
    return ce_sum / n_ce, kl_sum / n_kl


results = {"clean_ce": clean_ce, "runs": {}}


# ---------------------------------------------------------------------- sweeps
def run(name, L, A, fams=None, center=True):
    print(f"== {name} (L{L}, {A.shape[0]} atoms, center={center})", flush=True)
    mu = MU[L] if center else torch.zeros(D)
    X = eval_acts[L] - mu
    denom = ((eval_acts[L] - MU[L]) ** 2).sum().item()   # shared across runs
    t0 = time.time()
    fvu, recons, sel32, TOKENS[name] = omp(X, A, denom)
    entry = {"layer": L, "atoms": A.shape[0], "center": center, "fvu": fvu,
             "fidelity": {}}
    for k in K_PATCH:
        ce, kl = patched_metrics(L, recons[k], mu)
        entry["fidelity"][k] = {"ce": ce, "dce": ce - clean_ce, "kl": kl}
        print(f"   k={k}: FVU {fvu[k]:.4f}  dCE {ce-clean_ce:+.4f}  KL {kl:.4f}",
              flush=True)
    print("   FVU curve: " + "  ".join(f"k{k}={fvu[k]:.3f}" for k in K_CHECKPOINTS)
          + f"   ({time.time()-t0:.0f}s)", flush=True)
    if fams is not None:
        counts = {}
        for i in sel32.flatten().tolist():
            f_ = fams[i]
            counts[f_] = counts.get(f_, 0) + 1
        entry["composition_k32"] = counts
        by_group = {}
        for f_, c in counts.items():
            g = "".join(ch for ch in f_ if not ch.isdigit())
            by_group[g] = by_group.get(g, 0) + c
        total = sum(by_group.values())
        print("   composition@k32: " + "  ".join(
            f"{g}={100 * c / total:.1f}%" for g, c in sorted(by_group.items())),
            flush=True)
    results["runs"][name] = entry
    torch.cuda.empty_cache()
    return sel32


# main runs: the weight dictionary at both layers, centered (SAE parity)
saved = {}
for L in LAYERS:
    A, fams = weight_dictionary(L)
    sel = run(f"weight_L{L}", L, A, fams)
    if L == 6:
        saved = {"sel": sel, "A": A, "fams": fams}

A6, fams6, sel_wdd6 = saved["A"], saved["fams"], saved["sel"]
n6 = A6.shape[0]

# purity run: zero data touches anything (no mean subtraction either);
# FVU still reported against the shared centered denominator
run("weight_L6_pure", 6, A6, fams6, center=False)

# controls at matched size, layer 6
run("random_L6", 6, random_dictionary(n6))
run("actdict_L6", 6, activation_dictionary(6, n6))

# PCA control: best-k-per-sample from a 768-dim basis fit on train data
print("== pca_L6", flush=True)
sub = train_acts[6][torch.randperm(len(train_acts[6]))[:65536]] - MU[6]
V = torch.linalg.svd(sub.to(DEV), full_matrices=False).Vh        # [768, 768]
del sub
X = eval_acts[6] - MU[6]
denom6 = (X ** 2).sum().item()
pca_entry = {"layer": 6, "fvu": {}, "fidelity": {}}
TOKENS["pca_L6"] = {}
for k in K_CHECKPOINTS:
    err = 0.0
    recs = []
    e_toks = []
    for s in range(0, len(X), 4096):
        x = X[s:s + 4096].to(DEV)
        cf = x @ V.T
        mask = torch.zeros_like(cf).scatter_(
            1, cf.abs().topk(k, -1).indices, 1.0)
        rec = (cf * mask) @ V
        e_tok = ((x - rec) ** 2).sum(-1); err += e_tok.sum().item(); e_toks.append(e_tok.cpu())
        if k in K_PATCH:
            recs.append(rec.half().cpu())
    pca_entry["fvu"][k] = err / denom6
    TOKENS["pca_L6"][k] = torch.cat(e_toks).numpy()
    if k in K_PATCH:
        ce, kl = patched_metrics(6, torch.cat(recs), MU[6])
        pca_entry["fidelity"][k] = {"ce": ce, "dce": ce - clean_ce, "kl": kl}
        print(f"   k={k}: FVU {pca_entry['fvu'][k]:.4f}  dCE {ce-clean_ce:+.4f}"
              f"  KL {kl:.4f}", flush=True)
print("   FVU curve: " + "  ".join(f"k{k}={pca_entry['fvu'][k]:.3f}"
      for k in K_CHECKPOINTS), flush=True)
results["runs"]["pca_L6"] = pca_entry

# -------------------------------------------------- qualitative atom readouts
print("== qualitative: most-used MLP atoms at L6", flush=True)
counts = torch.bincount(sel_wdd6.flatten(), minlength=n6)
mlp_mask = torch.tensor([f.startswith("mlp") for f in fams6])
mlp_top = (counts * mlp_mask).topk(8).indices
qual = []
X6 = eval_acts[6] - MU[6]
flat_eval = eval_ids.flatten()
for a in mlp_top.tolist():
    if counts[a] == 0:
        continue
    atom = A6[a].to(DEV)
    toks = (atom @ W_U.T.float()).topk(8).indices
    top_tok = [tok.decode([t]) for t in toks.tolist()]
    used = (sel_wdd6 == a).any(-1).nonzero()[:, 0]
    coeffs = (X6[used].to(DEV) @ atom).abs()
    best = used[coeffs.topk(min(3, len(used))).indices.cpu()]
    ctxs = []
    for p in best.tolist():
        ctxs.append(tok.decode(
            flat_eval[max(0, p - 10):p + 1].tolist()).replace("\n", "|"))
    qual.append({"atom": a, "family": fams6[a], "count": counts[a].item(),
                 "unembed_top": top_tok, "contexts": ctxs})
    print(f"   [{fams6[a]}#{a}] n={counts[a].item()}  ->{top_tok}", flush=True)
    for c in ctxs:
        print(f"      ...{c!r}", flush=True)
results["qual_mlp_atoms_L6"] = qual

with open(f"{OUT}/exp_a_wdd.json", "w") as f:
    json.dump(results, f, indent=1)
import numpy as np
np.savez_compressed(f"{OUT}/exp_a_tokens.npz", **{f"xnorm2_L{L}": ((eval_acts[L] - MU[L]) ** 2).sum(-1).numpy() for L in LAYERS},
                    **{f"{n}_k{k}": v for n, d in TOKENS.items() for k, v in d.items()})
print("DONE_A", flush=True)
