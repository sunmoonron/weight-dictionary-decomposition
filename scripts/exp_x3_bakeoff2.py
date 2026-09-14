"""X3: bake-off v2. Same machinery as exp_v1 (identical battery, identical
partner-assignment ledger, shared code path), scaled and hardened:
  N = 10 nominations per arm, THREE arms:
    WDD    top 10 by unconditional counter-alignment write energy
    outlier top 10 by activation excess-kurtosis x mean|write| (marginal)
    corr   top 10 by max |activation correlation| with any upstream MLP neuron
           (a RELATIONAL baseline, answering the endpoint-alignment objection)
  No deduplication rule: arms nominate independently; overlaps are reported
  and batteries run once per unique neuron. exp_v1's six nominees and all
  season-studied neurons/channel-heavy neurons are excluded from all arms.
Criteria identical to exp_v1 (TIER1 passive accounting, TIER2 causal
response), fixed here before execution. Writes results/exp_x3_bakeoff2.json."""

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
N_NOM = int(os.environ.get("WDD_N_NOM", 10))   # 10 = the run reported as bake-off v2; 30 = the enlarged rerun
RESFILE = f"{OUT}/exp_x3_bakeoff2.json" if N_NOM == 10 else f"{OUT}/exp_x3_bakeoff2_n{N_NOM}.json"
BLOCKS = range(2, 9)
KNOWN_CH = torch.tensor([138, 266, 447, 481, 496])
EXCLUDE = {(2, 2714), (3, 614), (4, 1894), (5, 1790), (6, 3039), (5, 2070),
           (5, 1505), (5, 1856), (5, 1888), (6, 834), (7, 2402), (9, 840),
           (10, 900), (3, 1848), (3, 2614), (3, 1545), (3, 2664), (3, 442),
           (3, 2683), (3, 2885), (3, 2723), (3, 1995)}
try:
    v1 = json.load(open(f"{OUT}/exp_v1_bakeoff.json"))
    for key in ("wdd_candidates", "outlier_candidates"):
        for nm in v1.get(key, {}):
            b, j = nm.replace("mlp", "").split("#")
            EXCLUDE.add((int(b), int(j)))
    print(f"excluding {len(EXCLUDE)} studied neurons (incl. v1 nominees)",
          flush=True)
except Exception as e:
    print(f"v1 json not loaded ({e}); using static exclusions", flush=True)

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
D = model.config.n_embd

text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0][:N_EVAL * CTX].view(
    N_EVAL, CTX)

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

# ---------------- nominations (three arms, no dedup) ----------------
excl_mask = {b: torch.zeros(N_MLP, dtype=torch.bool) for b in BLOCKS}
for (b, j) in EXCLUDE:
    if b in excl_mask:
        excl_mask[b][j] = True

Z = {}
for b in range(9):
    a = acts[b].float()
    Z[b] = ((a - a.mean(0)) / a.std(0).clamp_min(1e-6))

wdd_scores, out_scores, cor_scores = [], [], []
for b in BLOCKS:
    C = acts[b].float() * WN[b]
    S = ((levels[b + 1] - MUS[b + 1]) @ DIRS[b].T) / \
        C.where(C.abs() > 1e-6, torch.ones_like(C))
    big = C.abs() > 0.5
    e_cnt = (C ** 2 * ((S < 0) & big)).sum(0)
    z = Z[b]
    kurt = (z ** 4).mean(0) - 3.0
    spik = kurt.clamp_min(0) * C.abs().mean(0)
    # relational screen: max |corr| with any upstream MLP neuron
    zb = Z[b].T.to(DEV)                              # [3072, NT]
    maxcorr = torch.zeros(N_MLP)
    for bu in range(b):
        zu = Z[bu].to(DEV)                           # [NT, 3072]
        cr = (zb @ zu) / NT                          # [3072, 3072]
        maxcorr = torch.maximum(maxcorr, cr.abs().max(1).values.cpu())
        del zu, cr
    del zb
    chsh = DIRS[b][:, KNOWN_CH].pow(2).sum(1)
    bad = excl_mask[b] | (chsh > 0.15)
    e_cnt[bad] = -1.0
    spik[bad] = -1.0
    maxcorr[bad] = -1.0
    for j in range(N_MLP):
        wdd_scores.append((e_cnt[j].item(), b, j))
        out_scores.append((spik[j].item(), b, j))
        cor_scores.append((maxcorr[j].item(), b, j))
    print(f"  scored block {b}", flush=True)

arms = {}
for nm, scores in [("WDD", wdd_scores), ("outlier", out_scores),
                   ("corr", cor_scores)]:
    scores.sort(reverse=True)
    arms[nm] = [(b, j) for _, b, j in scores[:N_NOM]]
    print(f"{nm} arm: {arms[nm]}", flush=True)
uniq_noms = sorted(set(sum(arms.values(), [])))
overlaps = {f"{a}&{b}": sorted(set(arms[a]) & set(arms[b]))
            for a in arms for b in arms if a < b}
print(f"unique nominees: {len(uniq_noms)}; overlaps: {overlaps}", flush=True)

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


def battery(b, j):
    d = DIRS[b][j]
    c = acts[b][:, j].float() * WN[b][j]
    thr = c.abs().quantile(0.95)
    m = c.abs() >= max(thr.item(), 1e-3)
    if m.sum() < 200:
        m = c.abs() >= c.abs().quantile(0.90)
    tsel = m.nonzero()[:, 0]
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
    row = {"upstream": f"mlp{bu}#{ju}",
           "upstream_share_pred": round(pred, 2),
           "dproj": round(dproj, 2), "ctrl_dproj": round(cproj, 2),
           "dresp": round(dresp, 3), "ctrl_dresp": round(cresp, 3),
           "TIER1": bool(tier1), "TIER2": bool(tier2)}
    print(f"mlp{b}#{j} <- {row['upstream']}: pred {pred:+.2f} "
          f"dproj {dproj:+.2f} (ctrl {cproj:+.2f}) dresp {dresp:+.3f} "
          f"(ctrl {cresp:+.3f}) T1={row['TIER1']} T2={row['TIER2']}",
          flush=True)
    return row


res = {"arms": {nm: [f"mlp{b}#{j}" for b, j in lst]
                for nm, lst in arms.items()},
       "overlaps": {k: [f"mlp{b}#{j}" for b, j in v]
                    for k, v in overlaps.items()},
       "batteries": {}}
with open(RESFILE, "w") as f:
    json.dump(res, f, indent=1)
for (b, j) in uniq_noms:
    res["batteries"][f"mlp{b}#{j}"] = battery(b, j)
    with open(RESFILE, "w") as f:
        json.dump(res, f, indent=1)

for nm, lst in arms.items():
    rows = [res["batteries"][f"mlp{b}#{j}"] for b, j in lst]
    t1 = sum(r["TIER1"] for r in rows)
    t2 = sum(r["TIER2"] for r in rows)
    res[f"summary_{nm}"] = {"TIER1": f"{t1}/{N_NOM}", "TIER2": f"{t2}/{N_NOM}"}
    print(f"{nm}: TIER1 {t1}/{N_NOM}  TIER2 {t2}/{N_NOM}", flush=True)
res["_note"] = (f"bake-off v2: N={N_NOM}/arm, three arms (WDD counter-energy; "
                "marginal kurtosis-x-write outlier; relational max-upstream-"
                "correlation), no dedup rule, identical battery and partner "
                "ledger for all arms, criteria fixed before execution, "
                "v1 nominees and season-studied neurons excluded")
with open(RESFILE, "w") as f:
    json.dump(res, f, indent=1)
print("DONE_X3", flush=True)
