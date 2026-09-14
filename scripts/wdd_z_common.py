"""Shared state, ledger and dictionary construction for the Z-series scripts
(exp_z3_ceiling, exp_z4_sae, exp_z5_kcurve). The conventions are those of
exp_z2_ci.py, from which this code is lifted unchanged: the dictionary at the
middle layer, the ledger activations captured at the MLP output projection,
a disjoint centering slice, and the evaluation slice after it."""
import os
import time

import torch

from exp_z2_ci import MODELS, BF16, CTX, DEV, corpus_text

torch.set_grad_enabled(False)


def build(name, corpus, n_eval=64, fp32=False, keep_model=False):
    """Returns a dict with the raw evaluation states X [NT, D] (GPU, float32),
    the centering mean mu (GPU), the ledger activations acts {block: [NT, DFF]}
    (CPU, float16), the write norms WN {block: [DFF]} (CPU), the unit-atom
    dictionary A [NA, D] (GPU), and the layout constants."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf, fam = MODELS[name]
    t0 = time.time()
    dt = torch.float32 if fp32 or name not in BF16 else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(hf, dtype=dt, device_map=DEV).eval()
    tok = AutoTokenizer.from_pretrained(hf)
    cfg = model.config
    if fam == "gpt2":
        layers, NB, D, NH = model.transformer.h, cfg.n_layer, cfg.n_embd, cfg.n_head
        L = 6
        mlp_lin = lambda l: l.mlp.c_proj
        wdir = lambda l: l.mlp.c_proj.weight.detach().float()
        att_w = lambda l: l.attn.c_proj.weight.detach().float()
        head_basis = lambda W, h, HD: torch.linalg.svd(W[h * HD:(h + 1) * HD, :], full_matrices=False).Vh
        emb = [model.transformer.wte.weight.detach().float(), model.transformer.wpe.weight.detach().float()]
        biases = lambda l: torch.stack([l.attn.c_proj.bias, l.mlp.c_proj.bias]).detach().float()
    elif fam == "llama":
        layers, NB, D, NH = model.model.layers, cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads
        L = NB // 2
        mlp_lin = lambda l: l.mlp.down_proj
        wdir = lambda l: l.mlp.down_proj.weight.detach().float().T
        att_w = lambda l: l.self_attn.o_proj.weight.detach().float()
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

    text = corpus_text(corpus)
    ids = tok(text, return_tensors="pt").input_ids[0]
    if fam == "gpt2" and corpus == "wikitext":
        from datasets import load_dataset
        tr = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"] if t.strip())
        cen_ids = tok(tr, return_tensors="pt").input_ids[0][:400 * CTX].view(400, CTX)
        eval_ids = ids[:n_eval * CTX].view(n_eval, CTX)
    else:
        n_cen = 100 if fam == "gpt2" else 8
        cen_ids = ids[:n_cen * CTX].view(n_cen, CTX)
        eval_ids = ids[n_cen * CTX:(n_cen + n_eval) * CTX].view(n_eval, CTX)
    assert eval_ids.shape == (n_eval, CTX), f"not enough tokens in {corpus} for {name}"

    mu = torch.zeros(D, device=DEV); n_mu = 0
    for i in range(0, len(cen_ids), 8):
        out = model(cen_ids[i:i + 8].to(DEV), output_hidden_states=True)
        h = out.hidden_states[L + 1].float(); mu += h.sum((0, 1)); n_mu += h.shape[0] * h.shape[1]; del out
    mu /= n_mu
    acts = {b: [] for b in range(L + 1)}
    hooks = [mlp_lin(layers[b]).register_forward_pre_hook((lambda b_: lambda m, inp: acts[b_].append(inp[0].to(torch.float16).cpu()))(b)) for b in range(L + 1)]
    X = []
    for i in range(0, n_eval, 4):
        out = model(eval_ids[i:i + 4].to(DEV), output_hidden_states=True)
        X.append(out.hidden_states[L + 1].float()); del out
    for h in hooks: h.remove()
    X = torch.cat(X).view(-1, D)
    acts = {b: torch.cat(a).view(-1, DFF) for b, a in acts.items()}

    W = {b: wdir(layers[b]) for b in range(L + 1)}
    WN = {b: W[b].norm(dim=-1).cpu() for b in range(L + 1)}
    atoms = [e for e in emb]
    for b in range(L + 1):
        atoms.append(W[b])
        Wo = att_w(layers[b])
        for h in range(NH): atoms.append(head_basis(Wo, h, HD))
        if biases is not None: atoms.append(biases(layers[b]))
    A = torch.cat([a.to(DEV) for a in atoms]); A = A / A.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    OFF = sum(e.shape[0] for e in emb); STRIDE = DFF + NH * HD + (2 if biases is not None else 0)
    del W
    if not keep_model:
        del model; torch.cuda.empty_cache()
    print(f"  {name}/{corpus}: {X.shape[0]} states, d={D} dff={DFF} L={L} atoms={A.shape[0]} ({time.time() - t0:.0f}s)", flush=True)
    return dict(X=X, mu=mu, acts=acts, WN=WN, A=A, NA=A.shape[0], OFF=OFF, STRIDE=STRIDE, DFF=DFF, L=L, D=D, NH=NH, HD=HD,
                fam=fam, eval_ids=eval_ids, n_eval=n_eval, model=model if keep_model else None, tok=tok if keep_model else None)


def atom_type(idx, OFF, STRIDE, DFF, NH, HD):
    """0 embedding, 1 MLP, 2 attention, 3 bias, for a tensor of atom indices."""
    t = torch.zeros_like(idx)
    inblk = idx >= OFF
    off = (idx - OFF) % STRIDE
    t[inblk & (off < DFF)] = 1
    t[inblk & (off >= DFF) & (off < DFF + NH * HD)] = 2
    t[inblk & (off >= DFF + NH * HD)] = 3
    return t
