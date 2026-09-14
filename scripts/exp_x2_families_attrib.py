"""X2: per-family attribution deep-dive + channel-robust FVU.
For each non-GPT-2 family: (a) magnitude-decile stratification of r on
identified writes, (b) one-shot top-1 recall alongside OMP recall,
(c) unconditional r, (d) FVU@32 for weight vs rotated dictionaries computed
EXCLUDING the top-5 highest-variance residual coordinates from the metric
(tests whether massive channels inflate FVU). Mirrors exp_s9's verified
conventions: mid-layer L = NB//2, raw hidden states only, write directions =
down_proj COLUMNS, post-gating activations via pre-hook, fp16 storage.
Writes results/exp_x2_families_attrib.json incrementally."""

import json
import os
import traceback

_HF = "/data/mechinterp/hf"          # the original machine's cache; ignored elsewhere
if os.path.isdir(_HF):
    os.environ.setdefault("HF_HOME", _HF)
RESULTS_DIR = os.environ.get("WDD_RESULTS", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
torch.set_grad_enabled(False)
OUT = os.path.join(RESULTS_DIR,
                   "exp_x2_families_attrib.json")
ONLY = [m for m in os.environ.get("WDD_MODELS", "").split(",") if m]
CTX, N_MEAN, N_EVAL, K = 512, 8, 16, 64

MODELS = {
    "smollm2-135m": ("HuggingFaceTB/SmolLM2-135M", "llama"),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", "llama"),
    "pythia-410m": ("EleutherAI/pythia-410m", "neox"),
    "olmo-1b-0724": ("allenai/OLMo-1B-0724-hf", "llama"),
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B", "llama"),
    "pythia-6.9b": ("EleutherAI/pythia-6.9b", "neox"),
    "llama-7b": ("huggyllama/llama-7b", "llama"),
}
BF16 = {"qwen2.5-7b", "pythia-6.9b", "llama-7b"}

from datasets import load_dataset
text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    if t.strip())

RES = {}
if os.path.exists(OUT):
    RES = json.load(open(OUT))


def save():
    with open(OUT, "w") as f:
        json.dump(RES, f, indent=1)


def pearson(a, b):
    a, b = a.float() - a.float().mean(), b.float() - b.float().mean()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-12)).item()


def omp(X, A, k, dev, batch=128):
    Ad = A.to(dev)
    N = X.shape[0]
    sel = torch.zeros(N, k, dtype=torch.long)
    coef = torch.zeros(N, k)
    resid = torch.zeros_like(X)
    for s in range(0, N, batch):
        x = X[s:s + batch].to(dev).float()
        r = x.clone()
        S = torch.zeros(len(x), 0, dtype=torch.long, device=dev)
        c = None
        for step in range(k):
            corr = r @ Ad.T
            if step:
                corr.scatter_(1, S, torch.zeros_like(S, dtype=corr.dtype))
            pick = corr.abs().argmax(1, keepdim=True)
            S = torch.cat([S, pick], 1)
            As = Ad[S]
            G = As @ As.transpose(1, 2) + 1e-5 * torch.eye(step + 1, device=dev)
            c = torch.linalg.solve(G, As @ x[:, :, None])
            r = x - (c.transpose(1, 2) @ As)[:, 0]
        sel[s:s + batch] = S.cpu()
        coef[s:s + batch] = c[:, :, 0].cpu()
        resid[s:s + batch] = r.cpu()
    return sel, coef, resid


for name, (hf, fam) in MODELS.items():
    if (name in RES and "error" not in RES.get(name, {})) or (ONLY and name not in ONLY):
        print(f"== {name} done or not selected, skip ==", flush=True)
        continue
    try:
        print(f"== {name} ==", flush=True)
        dt = (torch.bfloat16 if name in BF16 else
              torch.float16 if name == "olmo-1b-0724" else torch.float32)
        model = AutoModelForCausalLM.from_pretrained(
            hf, torch_dtype=dt, device_map="auto").eval()
        tok = AutoTokenizer.from_pretrained(hf)
        cfg = model.config
        D, NB = cfg.hidden_size, cfg.num_hidden_layers
        NH = cfg.num_attention_heads
        HD = D // NH
        L = NB // 2
        if fam == "llama":
            layers = model.model.layers
            emb_w = model.model.embed_tokens.weight
            mlp_lin = lambda l: l.mlp.down_proj
            att_w = lambda l: l.self_attn.o_proj.weight
        else:
            layers = model.gpt_neox.layers
            emb_w = model.gpt_neox.embed_in.weight
            mlp_lin = lambda l: l.mlp.dense_4h_to_h
            att_w = lambda l: l.attention.dense.weight
        DFF = mlp_lin(layers[0]).weight.shape[1]
        ids = tok(text, return_tensors="pt").input_ids[0][
            :(N_MEAN + N_EVAL) * CTX].view(-1, CTX)
        dev0 = next(model.parameters()).device

        acts = {b: [] for b in range(L + 1)}
        hooks = [mlp_lin(layers[b]).register_forward_pre_hook(
            (lambda b_: lambda m, inp: acts[b_].append(inp[0].half().cpu()))(b))
            for b in range(L + 1)]
        states = []
        for i in range(N_MEAN + N_EVAL):
            out = model(ids[i:i + 1].to(dev0), output_hidden_states=True)
            states.append(out.hidden_states[L + 1].float().cpu())
            del out
            torch.cuda.empty_cache()
        for h in hooks:
            h.remove()
        acts = {b: torch.cat(a).view(-1, DFF) for b, a in acts.items()}
        X = torch.cat(states).view(-1, D)
        n_mu = N_MEAN * CTX
        MU = X[:n_mu].mean(0)
        Xe = X[n_mu:] - MU
        acts = {b: a[n_mu:] for b, a in acts.items()}
        NT = Xe.shape[0]
        print(f"  {NT} eval states, d={D} dff={DFF} L={L}", flush=True)

        # dictionary: embeddings + per-block [mlp columns, head OV SVD, ]
        def unit(a):
            return a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        MLP_W = {b: mlp_lin(layers[b]).weight.detach().float().cpu()
                 for b in range(L + 1)}                      # [d, dff]
        atoms = [emb_w.detach().float().cpu()]
        for b in range(L + 1):
            atoms.append(MLP_W[b].T)                          # columns = writes
            aw = att_w(layers[b]).detach().float().cpu()      # [d, d]
            for h in range(NH):
                atoms.append(torch.linalg.svd(
                    aw[:, h * HD:(h + 1) * HD].T, full_matrices=False).Vh)
        A = unit(torch.cat(atoms))
        uniq = emb_w.shape[0]
        stride = DFF + NH * HD
        print(f"  dictionary {A.shape[0]} atoms", flush=True)

        WNb = {b: MLP_W[b].norm(dim=0) for b in MLP_W}        # column norms
        TRUE = torch.cat([acts[b].float() * WNb[b] for b in range(L + 1)],
                         dim=1)
        top3 = TRUE.abs().topk(3, dim=1).indices
        atom_idx = uniq + (top3 // DFF) * stride + (top3 % DFF)

        sel, coef, resid = omp(Xe, A, K, dev0)
        match = sel[:, None, :] == atom_idx[:, :, None]
        hit = match.any(-1)
        pos = match.float().argmax(-1)
        pred = coef.gather(1, pos) * hit
        true = TRUE.gather(1, top3)

        # one-shot recall
        Ad = A.to(dev0)
        os_hit = 0
        for s in range(0, NT, 256):
            x = Xe[s:s + 256].to(dev0).float()
            cr = (x @ Ad.T).abs()
            t64 = cr.topk(K, dim=1).indices.cpu()
            os_hit += (t64 == atom_idx[s:s + 256, 0:1]).any(-1).sum().item()
        del Ad

        pi, ti = pred[hit], true[hit]
        qs = ti.abs().quantile(torch.linspace(0, 1, 11))
        decs = []
        for d_ in range(10):
            m = (ti.abs() >= qs[d_]) & (ti.abs() <= qs[d_ + 1] + 1e-9)
            decs.append(round(pearson(pi[m], ti[m]), 3)
                        if m.sum() > 10 else None)

        # channel-robust FVU@32: exclude top-5 variance coordinates from metric
        var = Xe.var(0)
        topch = var.topk(5).indices
        keep = torch.ones(D, dtype=torch.bool)
        keep[topch] = False
        sel32, coef32, resid32 = omp(Xe, A, 32, dev0)
        Q = torch.linalg.qr(torch.randn(
            D, D, generator=torch.Generator().manual_seed(777)))[0]
        _, _, resid32r = omp(Xe, A @ Q, 32, dev0)
        fvu_full = (resid32 ** 2).sum().item() / (Xe ** 2).sum().item()
        fvu_keep = (resid32[:, keep] ** 2).sum().item() / \
            (Xe[:, keep] ** 2).sum().item()
        fvu_full_r = (resid32r ** 2).sum().item() / (Xe ** 2).sum().item()
        fvu_keep_r = (resid32r[:, keep] ** 2).sum().item() / \
            (Xe[:, keep] ** 2).sum().item()

        RES[name] = {
            "L": L, "n_atoms": int(A.shape[0]),
            "recall_top1_omp": round(hit[:, 0].float().mean().item(), 4),
            "recall_top1_oneshot": round(os_hit / NT, 4),
            "r_identified": round(pearson(pi, ti), 4),
            "r_unconditional": round(
                pearson(pred.flatten(), true.flatten()), 4),
            "r_identified_per_decile": decs,
            "fvu32": {"weight_full": round(fvu_full, 5),
                      "weight_excl_top5ch": round(fvu_keep, 5),
                      "rotated_full": round(fvu_full_r, 5),
                      "rotated_excl_top5ch": round(fvu_keep_r, 5),
                      "top5_var_coords": topch.tolist(),
                      "top5_var_share": round(
                          (var[topch].sum() / var.sum()).item(), 4)}}
        print(f"  {name}: {RES[name]}", flush=True)
        del model, acts, X, Xe, TRUE, A
        torch.cuda.empty_cache()
    except Exception:
        RES[name] = {"error": traceback.format_exc()}
        print(RES[name]["error"], flush=True)
    save()
print("DONE_X2", flush=True)
