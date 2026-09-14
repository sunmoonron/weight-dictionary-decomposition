"""Experiment H: cluster-level attribution.

Exp F showed neuron-level identification tops out ~65%, with near-duplicate
atoms (p99 pairwise |cos| = 0.997) as the suspected ceiling: the dictionary
contains functionally interchangeable directions, so demanding the exact
neuron may be the wrong target. Group atoms into equivalence classes by
residual-space similarity (union-find over |cos| >= tau on the full 78k
dictionary) and re-score: does WDD recover the correct computational CLASS
even when it misses the exact neuron? Also: cluster-level coefficient
estimation (signed sum over the class's selected members vs the true write).

Honesty guard: report cluster count/size stats -- if chaining creates a giant
component, cluster recall is trivially high and the result is void.
"""

import json
import os

_HF = "/data/mechinterp/hf"          # the original machine's cache; ignored elsewhere
if os.path.isdir(_HF):
    os.environ.setdefault("HF_HOME", _HF)
RESULTS_DIR = os.environ.get("WDD_RESULTS", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(0)
torch.set_grad_enabled(False)
DEV = "cuda:0"
OUT = RESULTS_DIR

CTX, N_EVAL, L, K, N_MLP = 512, 64, 6, 64, 3072
TAUS = [0.95, 0.9]

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
D = model.config.n_embd

text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
eval_ids = tok(text, return_tensors="pt").input_ids[0][:N_EVAL * CTX].view(N_EVAL, CTX)

acts = {b: [] for b in range(L + 1)}
hooks = [model.transformer.h[b].mlp.c_proj.register_forward_pre_hook(
    (lambda b_: lambda m, inp: acts[b_].append(inp[0].half().cpu()))(b))
    for b in range(L + 1)]
resid = []
for i in range(0, N_EVAL, 4):
    resid.append(model(eval_ids[i:i + 4].to(DEV), output_hidden_states=True)
                 .hidden_states[L + 1].float().cpu())
for h in hooks:
    h.remove()
acts = {b: torch.cat(a).view(-1, N_MLP) for b, a in acts.items()}
X = torch.cat(resid).view(-1, D)
X = X - X.mean(0)
NT = X.shape[0]

atoms, meta = [], []
def add(t, m):
    atoms.append(t.detach().float().cpu())
    meta.extend(m)
add(model.transformer.wte.weight, [None] * 50257)
add(model.transformer.wpe.weight, [None] * 1024)
for b in range(L + 1):
    blk = model.transformer.h[b]
    add(blk.mlp.c_proj.weight, [(b, j) for j in range(N_MLP)])
    w_o = blk.attn.c_proj.weight
    for hd in range(12):
        add(torch.linalg.svd(w_o[hd * 64:(hd + 1) * 64], full_matrices=False).Vh,
            [None] * 64)
    add(torch.stack([blk.attn.c_proj.bias, blk.mlp.c_proj.bias]), [None, None])
A = torch.cat(atoms)
NORMS = A.norm(dim=-1)
A = A / NORMS[:, None].clamp_min(1e-8)
NA = A.shape[0]
mlp_start = {b: 51281 + b * (N_MLP + 768 + 2) for b in range(L + 1)}
glob_of_col = torch.cat([torch.arange(mlp_start[b], mlp_start[b] + N_MLP)
                         for b in range(L + 1)])
wnorm = torch.cat([NORMS[mlp_start[b]:mlp_start[b] + N_MLP] for b in range(L + 1)])
TRUE = torch.cat([acts[b].float() for b in range(L + 1)], dim=1) * wnorm
print(f"{NT} states, {NA} atoms", flush=True)

# ---- similar-pair harvest (upper triangle, |cos| >= min(TAUS)) ----
A_dev = A.to(DEV)
tau_min = min(TAUS)
pairs_i, pairs_j, pairs_s = [], [], []
for s in range(0, NA, 2048):
    sims = (A_dev[s:s + 2048] @ A_dev.T).abs_()
    rows = torch.arange(s, min(s + 2048, NA), device=DEV)
    mask = sims >= tau_min
    mask &= torch.arange(NA, device=DEV)[None, :] > rows[:, None]   # j > i only
    idx = mask.nonzero()
    if len(idx):
        pairs_i.append((idx[:, 0] + s).cpu())
        pairs_j.append(idx[:, 1].cpu())
        pairs_s.append(sims[idx[:, 0], idx[:, 1]].cpu())
    del sims, mask, idx
pi = torch.cat(pairs_i); pj = torch.cat(pairs_j); ps = torch.cat(pairs_s)
print(f"{len(pi)} similar pairs at |cos|>={tau_min}", flush=True)


def unionfind(n, ii, jj):
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in zip(ii.tolist(), jj.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return torch.tensor([find(a) for a in range(n)])


# ---- OMP (same as exp F) + one-shot support ----
sels, cofs, oneshot = [], [], []
for s in range(0, NT, 768):
    x = X[s:s + 768].to(DEV)
    B = len(x)
    oneshot.append((x @ A_dev.T).abs_().topk(K, -1).indices.cpu())
    sel = torch.zeros(B, K, dtype=torch.long, device=DEV)
    taken = torch.zeros(B, NA, dtype=torch.bool, device=DEV)
    r = x.clone()
    for k in range(K):
        pick = (r @ A_dev.T).abs_().masked_fill_(taken, -1.0).argmax(-1)
        sel[:, k] = pick
        taken.scatter_(1, pick[:, None], True)
        A_S = A_dev[sel[:, :k + 1]]
        G = A_S @ A_S.transpose(1, 2) + 1e-5 * torch.eye(k + 1, device=DEV)
        c = torch.cholesky_solve(A_S @ x[:, :, None], torch.linalg.cholesky(G))
        r = x - (c.transpose(1, 2) @ A_S)[:, 0]
    sels.append(sel.cpu())
    cofs.append(c[:, :, 0].cpu())
sel = torch.cat(sels); cof = torch.cat(cofs); oneshot = torch.cat(oneshot)
print("OMP done", flush=True)

tops = []
for s0 in range(0, NT, 4096):
    tops.append(TRUE[s0:s0 + 4096].abs().to(DEV).topk(3, dim=1).indices.cpu())
topm_idx = torch.cat(tops)
topm_glob = glob_of_col[topm_idx]                       # [NT, 3]
true1 = TRUE.gather(1, topm_idx[:, :1])[:, 0]           # signed top-1 write


def pearson(a, b):
    a, b = a.float(), b.float()
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-9)).item()


res = {}
for tau in TAUS:
    keep = ps >= tau
    cl = unionfind(NA, pi[keep], pj[keep])
    sizes = torch.bincount(cl, minlength=NA)
    member_sizes = sizes[cl]
    stats = {"n_multi_clusters": int((torch.bincount(cl)[torch.bincount(cl) > 1]).shape[0]),
             "atoms_in_multi": int((member_sizes > 1).sum()),
             "largest_cluster": int(sizes.max()),
             "largest_frac_of_dict": float(sizes.max()) / NA}
    cl_sel, cl_one = cl[sel], cl[oneshot]
    out = {"cluster_stats": stats}
    for m in [1, 3]:
        ct = cl[topm_glob[:, :m]]
        out[f"recall_top{m}_cluster_omp64"] = \
            (ct[:, :, None] == cl_sel[:, None, :]).any(-1).float().mean().item()
        out[f"recall_top{m}_cluster_oneshot64"] = \
            (ct[:, :, None] == cl_one[:, None, :]).any(-1).float().mean().item()
    # cluster-level estimation for true top-1: signed sum over class members
    ct1 = cl[topm_glob[:, 0]]                            # [NT]
    match = cl_sel == ct1[:, None]                       # [NT, K]
    preds, truths = [], []
    for s0 in range(0, NT, 2048):
        mm = match[s0:s0 + 2048]
        any_hit = mm.any(-1)
        if not any_hit.any():
            continue
        dots = (A_dev[sel[s0:s0 + 2048]] *
                A_dev[topm_glob[s0:s0 + 2048, 0]][:, None, :]).sum(-1).cpu()
        pred = (mm * torch.sign(dots) * cof[s0:s0 + 2048]).sum(-1)
        preds.append(pred[any_hit])
        truths.append(true1[s0:s0 + 2048][any_hit])
    pp, tt = torch.cat(preds), torch.cat(truths)
    out["est_r_cluster_identified"] = pearson(pp, tt)
    res[f"tau_{tau}"] = out
    print(f"tau={tau}: clusters(multi) {stats['n_multi_clusters']} "
          f"covering {stats['atoms_in_multi']} atoms, largest {stats['largest_cluster']}"
          f" ({100*stats['largest_frac_of_dict']:.2f}% of dict)", flush=True)
    print(f"  cluster recall top1: omp {out['recall_top1_cluster_omp64']:.3f}  "
          f"oneshot {out['recall_top1_cluster_oneshot64']:.3f}   "
          f"top3: omp {out['recall_top3_cluster_omp64']:.3f}  "
          f"oneshot {out['recall_top3_cluster_oneshot64']:.3f}", flush=True)
    print(f"  cluster-level estimation r (identified): "
          f"{out['est_r_cluster_identified']:.3f}", flush=True)

# atom-level sanity (must match exp F)
atom_recall = (topm_glob[:, :1, None] == sel[:, None, :]).any(-1).float().mean().item()
res["sanity_atom_recall_top1"] = atom_recall
print(f"sanity atom-level top-1 recall: {atom_recall:.3f} (exp F: 0.647)", flush=True)

with open(f"{OUT}/exp_h_clusters.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE_H", flush=True)
