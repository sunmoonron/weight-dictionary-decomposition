"""Z4: the SAE comparison matched in both directions (GPT-2 small).

The paper's Figure 1 places the trained SAE of Bloom (2024) on the weight
dictionary's sparsity curve at its reported L0 and variance explained, which
were measured on OpenWebText with 128-token contexts, whereas the curve is on
WikiText-2 with 512-token contexts. This script evaluates both methods on
identical states in four cells: {WikiText-2 512-token, OpenWebText 128-token
with the SAE's BOS convention} x {the state after block 5 (the input of block
6, where the "blocks.6" SAE was trained), the state after block 6 (the paper's
layer, where the "blocks.7" SAE was trained)}. In each cell the SAE's L0, FVU
and spliced-in cross-entropy are measured directly from its published weights,
and the weight dictionary is run at k = 32, k = 64 and k equal to the SAE's
measured L0, with FVU and spliced-in cross-entropy, all with bootstrap
intervals over sequences. FVU uses one denominator per cell for both methods.
"""
import json
import os
import time

_HF = "/data/mechinterp/hf"
if os.path.isdir(_HF):
    os.environ.setdefault("HF_HOME", _HF)
RESULTS_DIR = os.environ.get("WDD_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

import numpy as np
import torch
import torch.nn.functional as Fn

torch.set_grad_enabled(False)
DEV = "cuda:0"
OUT = os.path.join(RESULTS_DIR, "exp_z4_sae.json")
SAE_REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
N_WIKI, CTX_WIKI, N_OWT, CTX_OWT, N_CEN_OWT = 64, 512, 256, 128, 100
BATCH, B_BOOT = 2048, 1000
LAYERS = [5, 6]                      # dictionary layer L: states are hidden_states[L + 1]; the matching SAE is blocks.{L+1}.hook_resid_pre

from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", dtype=torch.float32, device_map=DEV).eval()
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
D = model.config.n_embd


# ---------------------------------------------------------------- data
def wikitext_ids():
    from datasets import load_dataset
    te = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
    tr = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"] if t.strip())
    ev = tok(te, return_tensors="pt").input_ids[0][:N_WIKI * CTX_WIKI].view(N_WIKI, CTX_WIKI)
    ce = tok(tr, return_tensors="pt").input_ids[0][:400 * CTX_WIKI].view(400, CTX_WIKI)
    return ev, ce


def owt_ids():
    """The SAE's training convention: BOS followed by the first 127 tokens of a document."""
    from datasets import load_dataset
    docs = []
    try:
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        for ex in ds:
            ids = tok(ex["text"]).input_ids
            if len(ids) >= CTX_OWT - 1: docs.append(ids[:CTX_OWT - 1])
            if len(docs) >= N_OWT + N_CEN_OWT: break
    except Exception as e:  # fall back to a pre-tokenized GPT-2 OpenWebText sample
        print("  openwebtext stream failed:", repr(e)[:200], "-> apollo-research pretokenized fallback", flush=True)
        ds = load_dataset("apollo-research/Skylion007-openwebtext-tokenizer-gpt2", split="train", streaming=True)
        for ex in ds:
            ids = [t for t in ex["input_ids"] if t != tok.bos_token_id][:CTX_OWT - 1]
            if len(ids) >= CTX_OWT - 1: docs.append(ids)
            if len(docs) >= N_OWT + N_CEN_OWT: break
    bos = tok.bos_token_id
    ids = torch.tensor([[bos] + d for d in docs])
    return ids[:N_OWT], ids[N_OWT:N_OWT + N_CEN_OWT]


def states(ids, L, want_ce=False):
    """States after block L; optionally the clean per-sequence cross-entropy sums, computed per batch so no logits accumulate."""
    hs, sums, cnts = [], [], []
    for i in range(0, len(ids), 8):
        out = model(ids[i:i + 8].to(DEV), output_hidden_states=True)
        hs.append(out.hidden_states[L + 1].float())
        if want_ce:
            s, c = ce_of(out.logits.float(), ids[i:i + 8]); sums.append(s); cnts.append(c)
        del out
    X = torch.cat(hs).view(-1, D)
    return (X, np.concatenate(sums), np.concatenate(cnts)) if want_ce else X


def ce_of(logits, ids):
    """Per-sequence next-token cross-entropy sums and counts."""
    tgt = ids[:, 1:].to(DEV)
    ce = Fn.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape)
    return ce.sum(1).cpu().numpy().astype(np.float64), np.full(len(ids), tgt.shape[1], dtype=np.float64)


def spliced_ce(ids, L, recon):
    """Replace the output of block L by recon (raw coordinates) and read the cross-entropy."""
    full = recon.view(len(ids), -1, D)
    state = {}
    def hook(mod, inp, out):
        rep = state["rep"]
        return (rep,) + tuple(out[1:]) if isinstance(out, tuple) else rep
    h = model.transformer.h[L].register_forward_hook(hook)
    sums, cnts = [], []
    try:
        for i in range(0, len(ids), 8):
            state["rep"] = full[i:i + 8]
            lg = model(ids[i:i + 8].to(DEV)).logits.float()
            s, c = ce_of(lg, ids[i:i + 8]); sums.append(s); cnts.append(c); del lg
    finally:
        h.remove()
    return np.concatenate(sums), np.concatenate(cnts)


# ---------------------------------------------------------------- the two methods
def dictionary(L):
    layers = model.transformer.h
    atoms = [model.transformer.wte.weight.detach().float(), model.transformer.wpe.weight.detach().float()]
    for b in range(L + 1):
        blk = layers[b]
        atoms.append(blk.mlp.c_proj.weight.detach().float())
        w_o = blk.attn.c_proj.weight.detach().float()
        for h in range(12): atoms.append(torch.linalg.svd(w_o[h * 64:(h + 1) * 64, :], full_matrices=False).Vh)
        atoms.append(torch.stack([blk.attn.c_proj.bias, blk.mlp.c_proj.bias]).detach().float())
    A = torch.cat([a.to(DEV) for a in atoms]); return A / A.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def omp_recon(X, A, ks):
    """OMP to max(ks); the reconstruction at each k in ks."""
    N, NA = X.shape[0], A.shape[0]; kmax = max(ks)
    rec = {k: torch.zeros_like(X) for k in ks}
    eye = torch.eye(kmax, device=DEV)
    for s in range(0, N, BATCH):
        x = X[s:s + BATCH]; n = x.shape[0]
        r = x.clone(); S = torch.zeros(n, 0, dtype=torch.long, device=DEV)
        taken = torch.zeros(n, NA, dtype=torch.bool, device=DEV)
        for step in range(kmax):
            pick = (r @ A.T).abs_().masked_fill_(taken, -1.0).argmax(-1, keepdim=True)
            taken.scatter_(1, pick, True); S = torch.cat([S, pick], 1)
            As = A[S]; G = As @ As.transpose(1, 2) + 1e-5 * eye[:step + 1, :step + 1]
            c = torch.cholesky_solve(As @ x[:, :, None], torch.linalg.cholesky(G))
            r = x - (c.transpose(1, 2) @ As)[:, 0]
            if step + 1 in rec: rec[step + 1][s:s + n] = x - r
        del taken
    return rec


def load_sae(L):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    name = f"blocks.{L + 1}.hook_resid_pre"
    cfg = json.load(open(hf_hub_download(SAE_REPO, f"{name}/cfg.json")))
    w = load_file(hf_hub_download(SAE_REPO, f"{name}/sae_weights.safetensors"))
    w = {k: v.to(DEV).float() for k, v in w.items()}
    print(f"  SAE {name}: d_sae {w['W_enc'].shape[1]}, act {cfg.get('activation_fn_str', cfg.get('activation_fn'))}, b_dec to input {cfg.get('apply_b_dec_to_input')}, normalize {cfg.get('normalize_activations')}, ctx {cfg.get('context_size')}, data {cfg.get('dataset_path')}", flush=True)
    assert cfg.get("normalize_activations") in (None, "none", False), cfg.get("normalize_activations")
    assert str(cfg.get("activation_fn_str", cfg.get("activation_fn", "relu"))).lower() == "relu"
    return w, bool(cfg.get("apply_b_dec_to_input", True)), name


def sae_recon(X, w, sub_b_dec):
    """The SAEs were trained on TransformerLens activations, whose writing weights are centered so that every residual
    vector has zero mean over the model dimension. Center each state the same way, decode, and add the mean back."""
    rec, l0 = torch.zeros_like(X), torch.zeros(X.shape[0], device=DEV)
    for s in range(0, X.shape[0], BATCH):
        x = X[s:s + BATCH]; c = x.mean(-1, keepdim=True); xc = x - c
        f = torch.relu(((xc - w["b_dec"]) if sub_b_dec else xc) @ w["W_enc"] + w["b_enc"])
        rec[s:s + BATCH] = f @ w["W_dec"] + w["b_dec"] + c; l0[s:s + BATCH] = (f > 0).sum(-1).float()
    return rec, l0


# ---------------------------------------------------------------- cells
def boot(seq, n_seq, rng):
    counts = np.stack([np.bincount(rng.integers(0, n_seq, n_seq), minlength=n_seq) for _ in range(B_BOOT)])
    def ratio(num, den):
        ns, ds = np.bincount(seq, weights=num, minlength=n_seq), np.bincount(seq, weights=den, minlength=n_seq)
        b = (counts @ ns) / (counts @ ds)
        return [float(num.sum() / den.sum()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    def ratio_seq(ns, ds):
        b = (counts @ ns) / (counts @ ds)
        return [float(ns.sum() / ds.sum()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    return ratio, ratio_seq


def cell(cname, ev_ids, cen_ids, L, res):
    t0 = time.time()
    n_seq, ctx = ev_ids.shape
    X, clean_s, clean_c = states(ev_ids, L, want_ce=True)
    mu = states(cen_ids, L).mean(0)
    Xc = X - mu; xnorm2 = (Xc ** 2).sum(-1).cpu().numpy().astype(np.float64)
    seq = np.repeat(np.arange(n_seq), ctx)
    ratio, ratio_seq = boot(seq, n_seq, np.random.default_rng(2026))
    pos = np.arange(X.shape[0]) % ctx; m1 = pos >= 1
    def fvu_both(err): return {"all": ratio(err, xnorm2), "pos1": ratio(err * m1, xnorm2 * m1)}
    out_share0 = float(xnorm2[~m1].sum() / xnorm2.sum())
    out = {"n_states": int(X.shape[0]), "clean_ce": float(clean_s.sum() / clean_c.sum()), "variance_share_position_0": out_share0}
    # the SAE
    w, sub, sae_name = load_sae(L)
    rec, l0 = sae_recon(X, w, sub)
    l0_1 = float(l0[torch.from_numpy(m1).to(DEV)].mean())
    err = ((X - rec) ** 2).sum(-1).cpu().numpy().astype(np.float64)
    s, c = spliced_ce(ev_ids, L, rec)
    out["sae"] = {"name": sae_name, "l0": float(l0.mean()), "l0_pos1": l0_1, "fvu": fvu_both(err),
                  "ce": float(s.sum() / c.sum()), "dce": ratio_seq(s - clean_s, c)}
    del rec, w
    # the weight dictionary at k = 32, 64 and the SAE's L0
    A = dictionary(L)
    ks = sorted(set([32, 64, int(round(out["sae"]["l0_pos1"]))]))
    recs = omp_recon(Xc, A, ks)
    out["wdd"] = {"atoms": int(A.shape[0])}
    for k in ks:
        err = ((Xc - recs[k]) ** 2).sum(-1).cpu().numpy().astype(np.float64)
        s, c = spliced_ce(ev_ids, L, recs[k] + mu)
        out["wdd"][str(k)] = {"fvu": fvu_both(err), "ce": float(s.sum() / c.sum()), "dce": ratio_seq(s - clean_s, c)}
    out["seconds"] = round(time.time() - t0)
    res[cname][f"L{L}"] = out
    print(f"  {cname} L{L}: clean {out['clean_ce']:.3f} var@pos0 {out_share0:.3f} | SAE L0 {out['sae']['l0_pos1']:.1f} FVU(pos>=1) {out['sae']['fvu']['pos1'][0]:.4f} dCE {out['sae']['dce'][0]:+.4f} | " + " ".join(f"WDD k={k}: FVU(pos>=1) {out['wdd'][str(k)]['fvu']['pos1'][0]:.4f} dCE {out['wdd'][str(k)]['dce'][0]:+.4f}" for k in ks), flush=True)
    del A, recs, X, Xc; torch.cuda.empty_cache()


if __name__ == "__main__":
    res = {"wikitext512": {}, "owt128": {}}
    ev, cen = wikitext_ids()
    for L in LAYERS: cell("wikitext512", ev, cen, L, res)
    ev, cen = owt_ids()
    print(f"  openwebtext: {len(ev)} evaluation documents, {len(cen)} centering documents", flush=True)
    for L in LAYERS: cell("owt128", ev, cen, L, res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("DONE_Z4", flush=True)
