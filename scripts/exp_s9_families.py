"""Write Dynamics Ep.5 / generality sweep: WDD across model families.

Per model (SmolLM2-135M, Qwen2.5-0.5B, Pythia-410m, OLMo-1B-0724 -- gated
SwiGLU + rotary, NeoX parallel blocks, OLMo non-parametric LN):

  recon      FVU@32/64 at mid-layer for the weight dictionary vs the
             Gram-preserving ROTATED dictionary vs random -- the alignment
             signature. If weight ~ rotated, WDD is a GPT-2 artifact.
  attrib     top-1 true-MLP-write recall in the OMP support + r on identified.
  roles      per-block counter/novel/reinforcing (per-block top-3 frame).
  channels   unconditional counter-energy per block; top counter neuron and
             its dominant residual channel, checked against the superweight
             study's detection JSON for the same model.
  OLMo bonus the PUBLISHED super weight is layer 1, [row 1764, col 1710]:
             report atom mlp1#1710's dominant channel blind.

Uniformity notes: no bias atoms anywhere; embeddings restricted to tokens in
the corpus slice; mean from a disjoint slice; write dirs are down_proj/
dense_4h_to_h COLUMNS (nn.Linear is [out,in] -- transposed vs GPT-2 Conv1D).
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
import torch.nn.functional as Fn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(0)
torch.set_grad_enabled(False)
OUT = RESULTS_DIR
os.makedirs(OUT, exist_ok=True)
ONLY = [m for m in os.environ.get("WDD_MODELS", "").split(",") if m]
CTX, N_MEAN, N_EVAL, K = 512, 8, 16, 64

MODELS = {
    "smollm2-135m": ("HuggingFaceTB/SmolLM2-135M", "llama"),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", "llama"),
    "pythia-410m": ("EleutherAI/pythia-410m", "neox"),
    "olmo-1b-0724": ("allenai/OLMo-1B-0724-hf", "llama"),
    # 7B-class rows (added for the scale check; bf16 on an 80 GB GPU)
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B", "llama"),
    "pythia-6.9b": ("EleutherAI/pythia-6.9b", "neox"),
    "llama-7b": ("huggyllama/llama-7b", "llama"),
}
BF16 = {"qwen2.5-7b", "pythia-6.9b", "llama-7b"}
# optional: channel list from the separate super-weight study (not needed for the paper's numbers)
try:
    DET = json.load(open(os.environ.get(
        "WDD_SW_DETECTION", "/data/mechinterp/superweight/results/B_detection.json")))
except Exception:
    DET = {}

text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
g = torch.Generator().manual_seed(7)
try:
    ALL = json.load(open(f"{OUT}/exp_s9_families.json"))
except Exception:
    ALL = {}

for name, (hf, fam) in MODELS.items():
    if name in ALL or (ONLY and name not in ONLY):
        print(f"===== {name} already done or not selected, skipping =====", flush=True)
        continue
    print(f"===== {name} =====", flush=True)
    dt = (torch.bfloat16 if name in BF16 else
          torch.float16 if name == "olmo-1b-0724" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        hf, torch_dtype=dt, device_map="auto").eval()
    tok = AutoTokenizer.from_pretrained(hf)
    cfg = model.config
    D = cfg.hidden_size
    NB = cfg.num_hidden_layers
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
    ids = tok(text, return_tensors="pt").input_ids[0][:(N_MEAN + N_EVAL) * CTX]
    ids = ids.view(-1, CTX)
    dev0 = next(model.parameters()).device

    acts = {b: [] for b in range(NB)}
    hooks = [mlp_lin(layers[b]).register_forward_pre_hook(
        (lambda b_: lambda m, inp: acts[b_].append(inp[0].half().cpu()))(b))
        for b in range(NB)]
    levels = {lv: [] for lv in range(NB)}       # hs[lv], lv=0..NB-1 (raw only)
    ces = []
    for i in range(0, N_MEAN + N_EVAL, 1):
        out = model(ids[i:i + 1].to(dev0), output_hidden_states=True)
        for lv in levels:
            levels[lv].append(out.hidden_states[lv].float().cpu())
        lg = out.logits[:, :-1].float().cpu()
        ces.append(Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                    ids[i:i + 1, 1:].reshape(-1),
                                    reduction="none"))
        del out, lg
        torch.cuda.empty_cache()
    for h in hooks:
        h.remove()
    acts = {b: torch.cat(a).view(-1, DFF) for b, a in acts.items()}
    levels = {lv: torch.cat(v).view(-1, D) for lv, v in levels.items()}
    ce_mean = torch.cat(ces).mean().item()
    n_mean_tok = N_MEAN * CTX
    MUS = {lv: x[:n_mean_tok].mean(0) for lv, x in levels.items()}
    levels = {lv: x - MUS[lv] for lv, x in levels.items()}
    print(f"  captured; CE {ce_mean:.3f}  d={D} dff={DFF} NB={NB}", flush=True)

    # weights to CPU, then free the model
    MLP_W = {b: mlp_lin(layers[b]).weight.detach().float().cpu()      # [d, dff]
             for b in range(NB)}
    ATT_W = {b: att_w(layers[b]).detach().float().cpu() for b in range(L + 1)}
    EMB = emb_w.detach().float().cpu()
    del model
    torch.cuda.empty_cache()
    DEV = "cuda:0"

    dirs = {b: (MLP_W[b].T / MLP_W[b].T.norm(dim=-1, keepdim=True)
                .clamp_min(1e-8)) for b in range(NB)}
    wn = {b: MLP_W[b].T.norm(dim=-1) for b in range(NB)}

    # ---------------- dictionary at mid-layer L ----------------
    uniq = ids.flatten().unique()
    atoms, meta = [EMB[uniq]], [None] * len(uniq)
    for b in range(L + 1):
        atoms.append(MLP_W[b].T)
        meta += [(b, j) for j in range(DFF)]
        W_o = ATT_W[b]
        for h in range(NH):
            sl = W_o[:, h * HD:(h + 1) * HD]
            atoms.append(torch.linalg.svd(sl, full_matrices=False).U.T)
            meta += [None] * HD
    A = torch.cat(atoms)
    A = A / A.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    NA = A.shape[0]
    X = levels[L + 1][n_mean_tok:] if L + 1 < NB else None
    X = X.contiguous()
    NT = X.shape[0]
    denom = (X ** 2).sum().item()
    Q = torch.linalg.qr(torch.randn(D, D, generator=g))[0]
    R = torch.randn(NA, D, generator=g)
    R = R / R.norm(dim=-1, keepdim=True)
    print(f"  dict {NA} atoms; eval {NT} states", flush=True)

    def omp(Ad, want_sel=False):
        Ad = Ad.to(DEV)
        err = {32: 0.0, 64: 0.0}
        sels, cofs = [], []
        for s in range(0, NT, 512):
            x = X[s:s + 512].to(DEV)
            B = len(x)
            sel = torch.zeros(B, K, dtype=torch.long, device=DEV)
            taken = torch.zeros(B, NA, dtype=torch.bool, device=DEV)
            r = x.clone()
            for k in range(K):
                pick = (r @ Ad.T).abs_().masked_fill_(taken, -1.0).argmax(-1)
                sel[:, k] = pick
                taken.scatter_(1, pick[:, None], True)
                A_S = Ad[sel[:, :k + 1]]
                G = A_S @ A_S.transpose(1, 2) + 1e-5 * torch.eye(k + 1, device=DEV)
                c = torch.cholesky_solve(A_S @ x[:, :, None],
                                         torch.linalg.cholesky(G))
                r = x - (c.transpose(1, 2) @ A_S)[:, 0]
                if k + 1 in err:
                    err[k + 1] += (r ** 2).sum().item()
            if want_sel:
                sels.append(sel.cpu())
                cofs.append(c[:, :, 0].cpu())
        del Ad
        torch.cuda.empty_cache()
        fvu = {k: e / denom for k, e in err.items()}
        return (fvu, torch.cat(sels), torch.cat(cofs)) if want_sel else fvu

    fvu_w, sel, cof = omp(A, want_sel=True)
    fvu_rot = omp(A @ Q)
    fvu_rnd = omp(R)
    res = {"ce": round(ce_mean, 3), "d": D, "dff": DFF, "nb": NB, "L": L,
           "atoms": NA,
           "fvu": {"weight": {k: round(v, 3) for k, v in fvu_w.items()},
                   "rotated": {k: round(v, 3) for k, v in fvu_rot.items()},
                   "random": {k: round(v, 3) for k, v in fvu_rnd.items()}}}
    print(f"  FVU@32/64  weight {fvu_w[32]:.3f}/{fvu_w[64]:.3f}  "
          f"rotated {fvu_rot[32]:.3f}/{fvu_rot[64]:.3f}  "
          f"random {fvu_rnd[32]:.3f}/{fvu_rnd[64]:.3f}", flush=True)

    # ---------------- attribution: top-1 true write ----------------
    TRUE = torch.cat([acts[b][n_mean_tok:].float() * wn[b]
                      for b in range(L + 1)], dim=1)
    top1 = TRUE.abs().argmax(1)
    off = len(uniq)
    glob = torch.zeros_like(top1)
    per_blk = DFF + NH * HD
    b_of = top1 // DFF
    j_of = top1 % DFF
    glob = off + b_of * per_blk + j_of
    hit = (glob[:, None] == sel).any(-1)
    c_true = TRUE.gather(1, top1[:, None])[:, 0]
    pred = torch.where(hit, (torch.where(glob[:, None] == sel, cof,
                                         torch.zeros_like(cof))).sum(-1),
                       torch.zeros_like(c_true))
    pi, ti = pred[hit], c_true[hit]
    r_id = ((pi - pi.mean()) @ (ti - ti.mean())
            / ((pi - pi.mean()).norm() * (ti - ti.mean()).norm())
            .clamp_min(1e-9)).item()
    res["attrib"] = {"top1_recall": round(hit.float().mean().item(), 3),
                     "r_identified": round(r_id, 3)}
    print(f"  attrib: top1 recall {res['attrib']['top1_recall']}  "
          f"r|id {res['attrib']['r_identified']}", flush=True)

    # ---------------- role map + counter energy + channels ----------------
    known = set()
    if name in DET:
        pl = list(DET[name]["prompts"].values())[0]["per_layer"]
        known = {int(r_["resid_channel"]) for r_ in pl}
    roles, chans = {}, {}
    for b in range(NB - 1):
        Cb = acts[b].float() * wn[b]
        tk = Cb.abs().topk(3, dim=1).indices
        cs = Cb.gather(1, tk)
        dvec = dirs[b][tk]
        s = ((levels[b + 1][:, None, :] * dvec).sum(-1) / cs).clamp(-4, 5)
        roles[b] = {"counter": round((s < 0).float().mean().item(), 3),
                    "reinf": round((s >= 1.5).float().mean().item(), 3),
                    "median": round(s.median().item(), 2)}
        Sfull = (levels[b + 1].to(DEV) @ dirs[b].T.to(DEV)).cpu() / \
            Cb.where(Cb.abs() > 1e-6, torch.ones_like(Cb))
        mask = Cb.abs() > 0.5
        e_cnt = (Cb ** 2 * ((Sfull < 0) & mask)).sum(0)
        e_all = (Cb ** 2 * mask).sum().item()
        if b == 1:
            e_cnt_b1 = e_cnt.clone()
        jtop = int(e_cnt.argmax())
        dtop = dirs[b][jtop]
        ch = int(dtop.abs().argmax())
        chans[b] = {"counter_energy": round(e_cnt.sum().item() / max(e_all, 1e-9), 3),
                    "top_neuron": jtop, "channel": ch,
                    "ch_share": round((dtop[ch] ** 2).item(), 2),
                    "known_channel": ch in known}
        torch.cuda.empty_cache()
    res["roles"] = roles
    res["channels"] = chans
    hits = sum(v["known_channel"] for v in chans.values())
    print(f"  counter medians by block: "
          f"{[roles[b]['median'] for b in range(NB - 1)]}", flush=True)
    print(f"  top-counter channels matching superweight-study channels: "
          f"{hits}/{len(chans)}", flush=True)

    if name == "olmo-1b-0724":
        d_sw = dirs[1][1710]
        dom = int(d_sw.abs().argmax())
        res["olmo_published_sw"] = {
            "atom": "mlp1#1710", "dominant_channel": dom,
            "expected_channel": 1764,
            "channel_share": round((d_sw[dom] ** 2).item(), 3),
            "counter_energy_rank_in_b1": int(
                e_cnt_b1.argsort(descending=True).tolist().index(1710)),
            "atom_norm_rank_in_b1": int(
                wn[1].argsort(descending=True).tolist().index(1710))}
        print(f"  OLMo published SW check: {res['olmo_published_sw']}", flush=True)

    ALL[name] = res
    with open(f"{OUT}/exp_s9_families.json", "w") as f:
        json.dump(ALL, f, indent=1)
    del acts, levels, TRUE, A, X
    torch.cuda.empty_cache()

print("DONE_S9", flush=True)
