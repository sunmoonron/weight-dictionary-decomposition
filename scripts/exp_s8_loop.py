"""Write Dynamics Ep.4: closing one causal loop (GPT-2 small, channel 447).

Regulators (from S6): b3#614, b4#1894, b5#1790, b6#3039 (+ builder-side b2#2714
measured but not ablated).

B  channel -> regulator: scale the ch447 residual component at the block-2/3
   boundary (alpha 0.5 / 1.5) and measure each regulator's activation change,
   against matched random neurons. No response => regulation is open-loop
   (feed-forward), causally -- not just correlationally.
A  regulator -> channel: ablate the four regulators; track ch447 at every
   layer; compare the measured final shift against the EXACT predicted direct
   shift (-sum a_j * w_j[447]); the shortfall measures downstream
   compensation. Plus dCE of losing the regulation.
C  upstream ledger for the residue: for the top three non-channel
   counter-aligned neurons of block 5, decompose the opposing pre-projection
   <h_pre, d> into exact upstream contributions (embeddings, each block's MLP
   via a_i <w_i, d>, attention as the remainder) and name the top upstream
   neurons. A first circuit sketch for the unexplained population.
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
DEV = "cuda:0"
OUT = RESULTS_DIR
CTX, N_EVAL, N_MLP = 512, 32, 3072
CH = 447
REGS = {3: 614, 4: 1894, 5: 1790, 6: 3039}
BUILDER = (2, 2714)

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
D = model.config.n_embd

text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0][:N_EVAL * CTX].view(N_EVAL, CTX)

W = {b: model.transformer.h[b].mlp.c_proj.weight.detach().float().cpu()
     for b in range(7)}
WN = {b: W[b].norm(dim=-1) for b in W}
DIRS = {b: W[b] / WN[b][:, None].clamp_min(1e-8) for b in W}
watch = list(REGS.items()) + [BUILDER]
g = torch.Generator().manual_seed(41)
ctrl = [(b, int(torch.randint(0, N_MLP, (1,), generator=g))) for b, _ in watch]


def run(scale_alpha=None, ablate=False):
    """Forward pass; returns regulator/ctrl mean|activation|, ch447 per level,
    CE, and (if ablate) per-token L12 ch447 plus regulator acts for prediction."""
    grabbed = {k: [] for k in watch + ctrl}
    hooks = []
    for b in {b for b, _ in watch + ctrl}:
        def mk(b_):
            def f(m, inp):
                x = inp[0]
                for (bb, jj) in watch + ctrl:
                    if bb == b_:
                        grabbed[(bb, jj)].append(x[:, :, jj].float().cpu())
                if ablate and b_ in REGS:
                    x = x.clone()
                    x[:, :, REGS[b_]] = 0.0
                    return (x,)
            return f
        hooks.append(model.transformer.h[b].mlp.c_proj
                     .register_forward_pre_hook(mk(b)))
    if scale_alpha is not None:
        def bh(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            h[:, :, CH] = h[:, :, CH] * scale_alpha
            return (h,) + out[1:] if isinstance(out, tuple) else h
        hooks.append(model.transformer.h[2].register_forward_hook(bh))
    traj, ces, l12 = [], [], []
    for i in range(0, N_EVAL, 4):
        out = model(ids[i:i + 4].to(DEV), output_hidden_states=True)
        traj.append(torch.stack([out.hidden_states[lv][:, :, CH].float().cpu()
                                 for lv in range(13)]))
        l12.append(out.hidden_states[11][:, :, CH].float().cpu())
        lg = out.logits[:, :-1].float()
        ces.append(Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                    ids[i:i + 4, 1:].to(DEV).reshape(-1),
                                    reduction="none").cpu())
        del out
    for h in hooks:
        h.remove()
    acts = {k: torch.cat(v, dim=0).flatten() for k, v in grabbed.items()}
    return {"acts": acts,
            "traj": torch.cat(traj, dim=1).flatten(2).mean((1, 2)),
            "l12": torch.cat(l12).flatten(),
            "ce": torch.cat(ces).mean().item()}


base = run()
res = {"clean_ce": round(base["ce"], 4),
       "ch447_traj_clean": [round(v, 1) for v in base["traj"].tolist()]}
print("clean ch447 by level:", res["ch447_traj_clean"], flush=True)

# ---- B: channel -> regulator ----
respB = {}
for a in [0.5, 1.5]:
    pert = run(scale_alpha=a)
    row = {}
    for (b, j) in watch:
        b0 = base["acts"][(b, j)].abs().mean().item()
        row[f"b{b}#{j}"] = round((pert["acts"][(b, j)].abs().mean().item() - b0)
                                 / max(b0, 1e-6), 3)
    crow = []
    for (b, j) in ctrl:
        b0 = base["acts"][(b, j)].abs().mean().item()
        crow.append((pert["acts"][(b, j)].abs().mean().item() - b0)
                    / max(b0, 1e-6))
    row["ctrl_mean_absrel"] = round(float(torch.tensor(crow).abs().mean()), 3)
    respB[f"alpha_{a}"] = row
    print(f"B alpha={a}: {row}", flush=True)
res["B_channel_to_regulator"] = respB

# ---- A: regulator -> channel ----
abl = run(ablate=True)
res["A_dce"] = round(abl["ce"] - base["ce"], 4)
res["A_ch447_traj_ablated"] = [round(v, 1) for v in abl["traj"].tolist()]
# exact predicted direct shift at L12 from removing the four writes
pred = torch.zeros_like(base["l12"])
for b, j in REGS.items():
    pred -= base["acts"][(b, j)] * W[b][j, CH]
meas = abl["l12"] - base["l12"]
pm, mm = pred - pred.mean(), meas - meas.mean()
res["A_predicted_vs_measured"] = {
    "r": round((pm @ mm / (pm.norm() * mm.norm()).clamp_min(1e-9)).item(), 3),
    "slope_meas_over_pred": round(((pred @ meas) / (pred @ pred)).item(), 3),
    "mean_pred": round(pred.mean().item(), 2),
    "mean_meas": round(meas.mean().item(), 2)}
print(f"A: dCE {res['A_dce']}  pred-vs-meas {res['A_predicted_vs_measured']}",
      flush=True)
print("ablated ch447 traj:", res["A_ch447_traj_ablated"], flush=True)

# ---- C: upstream ledger for the non-channel counter residue (block 5) ----
acts_all = {b: [] for b in range(6)}
hooks = [model.transformer.h[b].mlp.c_proj.register_forward_pre_hook(
    (lambda b_: lambda m, inp: acts_all[b_].append(inp[0].half().cpu()))(b))
    for b in range(6)]
pre5, post5 = [], []
for i in range(0, N_EVAL, 4):
    out = model(ids[i:i + 4].to(DEV), output_hidden_states=True)
    pre5.append(out.hidden_states[5].float().cpu())
    post5.append(out.hidden_states[6].float().cpu())
    del out
for h in hooks:
    h.remove()
acts_all = {b: torch.cat(a).view(-1, N_MLP) for b, a in acts_all.items()}
PRE5 = torch.cat(pre5).view(-1, D)
POST5 = torch.cat(post5).view(-1, D)
mu = POST5.mean(0)
NT = PRE5.shape[0]

C5 = acts_all[5].float() * WN[5]
S5 = ((POST5 - mu) @ DIRS[5].T) / C5.where(C5.abs() > 1e-6, torch.ones_like(C5))
big = C5.abs() > 0.5
cnt_e = (C5 ** 2 * ((S5 < 0) & big)).sum(0)
chshare = DIRS[5][:, torch.tensor([138, 266, CH, 481, 496])].pow(2).sum(1)
resid_neurons = cnt_e.clone()
resid_neurons[chshare > 0.1] = 0.0
top_resid = resid_neurons.topk(3).indices.tolist()
res["C_neurons"] = {}
emb = model.transformer.wte.weight.detach().float().cpu()
wpe = model.transformer.wpe.weight.detach().float().cpu()
flat_ids = ids.flatten()
for j in top_resid:
    d = DIRS[5][j]
    m = ((S5[:, j] < 0) & big[:, j])
    tsel = m.nonzero()[:, 0]
    if len(tsel) > 4000:
        tsel = tsel[torch.randperm(len(tsel), generator=g)[:4000]]
    opp = (PRE5[tsel] - PRE5.mean(0)) @ d                 # opposing projection
    parts = {"embedding": ((emb[flat_ids[tsel]] + wpe[tsel % CTX]) @ d
                           - (emb[flat_ids] + wpe[torch.arange(NT) % CTX]).mean(0) @ d)}
    for b in range(5):
        contrib = acts_all[b][tsel].float() @ (W[b] @ d)  # sum_i a_i <w_i, d>
        parts[f"mlp{b}"] = contrib - (acts_all[b].float().mean(0) @ (W[b] @ d))
    known = sum(parts.values())
    parts["attn+rest"] = opp - known
    shares = {k: round((v.mean() / opp.mean().clamp(max=-1e-6)).item(), 2)
              if opp.mean() < 0 else 0.0 for k, v in parts.items()}
    top_up = {}
    for b in range(5):
        cvec = (acts_all[b][tsel].float() * (W[b] @ d)).mean(0)
        for i_ in cvec.abs().topk(2).indices.tolist():
            top_up[f"mlp{b}#{i_}"] = round(cvec[i_].item(), 2)
    res["C_neurons"][f"mlp5#{j}"] = {
        "n_counter_tokens": int(m.sum()),
        "mean_opposing_proj": round(opp.mean().item(), 2),
        "shares_of_opposition": shares,
        "top_upstream_writers": dict(sorted(top_up.items(),
                                            key=lambda kv: -abs(kv[1]))[:6])}
    print(f"mlp5#{j}: opp {opp.mean():.2f}  shares {shares}", flush=True)
    print(f"   top upstream: {res['C_neurons'][f'mlp5#{j}']['top_upstream_writers']}",
          flush=True)

with open(f"{OUT}/exp_s8_loop.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE_S8", flush=True)
