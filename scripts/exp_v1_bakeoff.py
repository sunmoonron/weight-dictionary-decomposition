"""Post-season audit: the nomination bake-off (pilot, N=3 per arm).

Question (GPT's, and the right one): does WDD's geometric screen nominate
anomalies that resolve into causal circuits at a higher rate than a standard
observational screen, when everything downstream of nomination is identical?

Arms (same universe: GPT-2 small, MLP neurons, blocks 2-8; both exclude every
season-studied neuron and any neuron with >0.15 direction-energy in the known
massive channels {138,266,447,481,496} -- fresh territory forced):
  WDD arm       top 3 by unconditional counter-alignment energy
  outlier arm   top 3 by activation excess-kurtosis x mean|write| (the classic
                "spiky interesting neuron" screen)

Identical battery per candidate j (pre-registered BEFORE any result existed):
  1. firing tokens = tokens with |c_j| in the neuron's own top 5% (min 200)
  2. ledger: top upstream MLP contributor u along d_j on those tokens
  3. intervention: ablate u; matched control: ablate a random neuron from u's
     block; measure on j's firing tokens:
       state:    d_proj = change in <h_post-b(j), d_j>; PREDICTED = -mean
                 contribution of u along d_j (direct part)
       response: relative change in |a_j|
  4. verdicts (mechanical):
       TIER1 (bookkeeping valid): sign(d_proj)==sign(pred), |d_proj| in
              [0.2, 2.0] x |pred|, and |d_proj| >= 3x control's |d_proj|
       TIER2 (live circuit): |d resp| >= 0.02 and >= 3x control's |d resp|
Score = tier hits per arm. Pilot scale; reported as such.
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
CTX, N_EVAL, N_MLP = 512, 32, 3072
BLOCKS = range(2, 9)
KNOWN_CH = torch.tensor([138, 266, 447, 481, 496])
EXCLUDE = {(2, 2714), (3, 614), (4, 1894), (5, 1790), (6, 3039), (5, 2070),
           (5, 1505), (5, 1856), (5, 1888), (6, 834), (7, 2402), (9, 840),
           (10, 900), (3, 1848), (3, 2614), (3, 1545), (3, 2664), (3, 442),
           (3, 2683), (3, 2885), (3, 2723), (3, 1995)}

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
D = model.config.n_embd

text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0][:N_EVAL * CTX].view(N_EVAL, CTX)

W = {b: model.transformer.h[b].mlp.c_proj.weight.detach().float().cpu()
     for b in range(12)}
WN = {b: W[b].norm(dim=-1) for b in W}
DIRS = {b: W[b] / WN[b][:, None].clamp_min(1e-8) for b in W}


def capture():
    acts = {b: [] for b in range(12)}
    hooks = [model.transformer.h[b].mlp.c_proj.register_forward_pre_hook(
        (lambda b_: lambda m, inp: acts[b_].append(inp[0].half().cpu()))(b))
        for b in range(12)]
    levels = {lv: [] for lv in range(1, 12)}
    for i in range(0, N_EVAL, 4):
        out = model(ids[i:i + 4].to(DEV), output_hidden_states=True)
        for lv in levels:
            levels[lv].append(out.hidden_states[lv].float().cpu())
        del out
    for h in hooks:
        h.remove()
    return ({b: torch.cat(a).view(-1, N_MLP) for b, a in acts.items()},
            {lv: torch.cat(v).view(-1, D) for lv, v in levels.items()})


acts, levels = capture()
MUS = {lv: x.mean(0) for lv, x in levels.items()}
NT = levels[1].shape[0]
print("captured", flush=True)

# ---------------- nominations ----------------
excl_mask = {b: torch.zeros(N_MLP, dtype=torch.bool) for b in BLOCKS}
for (b, j) in EXCLUDE:
    if b in excl_mask:
        excl_mask[b][j] = True
wdd_scores, out_scores = [], []
for b in BLOCKS:
    C = acts[b].float() * WN[b]
    S = ((levels[b + 1] - MUS[b + 1]) @ DIRS[b].T) / \
        C.where(C.abs() > 1e-6, torch.ones_like(C))
    big = C.abs() > 0.5
    e_cnt = (C ** 2 * ((S < 0) & big)).sum(0)
    a = acts[b].float()
    z = (a - a.mean(0)) / a.std(0).clamp_min(1e-6)
    kurt = (z ** 4).mean(0) - 3.0
    spik = kurt.clamp_min(0) * C.abs().mean(0)
    chsh = DIRS[b][:, KNOWN_CH].pow(2).sum(1)
    bad = excl_mask[b] | (chsh > 0.15)
    e_cnt[bad] = -1.0
    spik[bad] = -1.0
    for j in range(N_MLP):
        wdd_scores.append((e_cnt[j].item(), b, j))
        out_scores.append((spik[j].item(), b, j))
wdd_scores.sort(reverse=True)
out_scores.sort(reverse=True)
wdd_cand = [(b, j) for _, b, j in wdd_scores[:3]]
out_cand = []
for _, b, j in out_scores:
    if (b, j) not in wdd_cand:
        out_cand.append((b, j))
    if len(out_cand) == 3:
        break
print(f"WDD arm: {wdd_cand}", flush=True)
print(f"outlier arm: {out_cand}", flush=True)

g = torch.Generator().manual_seed(61)


def ablate_run(b_u, j_u, watch_b, watch_j):
    def hook(m, inp):
        x = inp[0].clone()
        x[:, :, j_u] = 0.0
        return (x,)
    h1 = model.transformer.h[b_u].mlp.c_proj.register_forward_pre_hook(hook)
    a_w, post = [], []
    h2 = model.transformer.h[watch_b].mlp.c_proj.register_forward_pre_hook(
        lambda m, inp: a_w.append(inp[0][:, :, watch_j].float().cpu()))
    for i in range(0, N_EVAL, 4):
        out = model(ids[i:i + 4].to(DEV), output_hidden_states=True)
        post.append(out.hidden_states[watch_b + 1].float().cpu())
        del out
    h1.remove(); h2.remove()
    return torch.cat(a_w).flatten(), torch.cat(post).view(-1, D)


def battery(b, j, arm):
    d = DIRS[b][j]
    c = acts[b][:, j].float() * WN[b][j]
    thr = c.abs().quantile(0.95)
    m = c.abs() >= max(thr.item(), 1e-3)
    if m.sum() < 200:
        m = c.abs() >= c.abs().quantile(0.90)
    tsel = m.nonzero()[:, 0]
    # top upstream MLP contributor along d on firing tokens
    best = (0.0, None)
    for bu in range(b):
        contrib = (acts[bu][tsel].float() * (W[bu] @ d)).mean(0)
        i_ = int(contrib.abs().argmax())
        if abs(contrib[i_].item()) > abs(best[0]):
            best = (contrib[i_].item(), (bu, i_))
    pred = -best[0]
    (bu, ju) = best[1]
    ctrl_j = int(torch.randint(0, N_MLP, (1,), generator=g))
    if ctrl_j == ju:
        ctrl_j = (ctrl_j + 1) % N_MLP
    base_proj = ((levels[b + 1][tsel] - MUS[b + 1]) @ d).mean().item()
    base_act = acts[b][tsel, j].float().abs().mean().item()

    def probe(jx):
        a_w, post = ablate_run(bu, jx, b, j)
        dproj = ((post[tsel] - MUS[b + 1]) @ d).mean().item() - base_proj
        dresp = (a_w[tsel].abs().mean().item() - base_act) / max(base_act, 1e-6)
        return dproj, dresp

    dproj, dresp = probe(ju)
    cproj, cresp = probe(ctrl_j)
    tier1 = (pred != 0 and dproj * pred > 0
             and 0.2 * abs(pred) <= abs(dproj) <= 2.0 * abs(pred)
             and abs(dproj) >= 3 * abs(cproj))
    tier2 = (abs(dresp) >= 0.02 and abs(dresp) >= 3 * abs(cresp))
    row = {"arm": arm, "upstream": f"mlp{bu}#{ju}",
           "upstream_share_pred": round(pred, 2),
           "dproj": round(dproj, 2), "ctrl_dproj": round(cproj, 2),
           "dresp": round(dresp, 3), "ctrl_dresp": round(cresp, 3),
           "TIER1": bool(tier1), "TIER2": bool(tier2)}
    print(f"[{arm}] mlp{b}#{j} <- {row['upstream']}: pred {pred:+.2f} "
          f"dproj {dproj:+.2f} (ctrl {cproj:+.2f}) dresp {dresp:+.3f} "
          f"(ctrl {cresp:+.3f}) T1={tier1} T2={tier2}", flush=True)
    return row


res = {"wdd_candidates": {}, "outlier_candidates": {}}
for b, j in wdd_cand:
    res["wdd_candidates"][f"mlp{b}#{j}"] = battery(b, j, "WDD")
for b, j in out_cand:
    res["outlier_candidates"][f"mlp{b}#{j}"] = battery(b, j, "outlier")

for arm, key in [("WDD", "wdd_candidates"), ("outlier", "outlier_candidates")]:
    rows = res[key].values()
    print(f"{arm}: TIER1 {sum(r['TIER1'] for r in rows)}/3  "
          f"TIER2 {sum(r['TIER2'] for r in rows)}/3", flush=True)
res["_note"] = ("pilot N=3/arm; criteria pre-registered in docstring before "
                "any battery ran; both arms share machinery, only nomination "
                "differs; season-studied neurons and massive channels excluded")
with open(f"{OUT}/exp_v1_bakeoff.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE_V1", flush=True)
