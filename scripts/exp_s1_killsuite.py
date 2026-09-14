"""Write Dynamics, Episode 1a: block-3 kill-suite on GPT-2 small.

  corpus     role-map (counter/novel/reinforcing per block) recomputed on
             OpenWebText -- does block 3's counter specialization survive a
             distribution shift away from Wikipedia?
  position   counter-fraction of block-3 writes by token position bucket
  unembed    is mlp3#614's function-word unembedding unusual, or do most
             block-3 neurons look like that? (early-unembedding artifact check)
  ablation   zero mlp3#614 everywhere: dCE overall, dCE on the tokens where it
             was a top-3 writer, and the shift in stream projection along its
             direction at L6 (absent counter-write should leave the opposed
             content standing -> more negative projection)
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
CTX, N_EVAL, L, N_MLP = 512, 64, 6, 3072
NEURON = 614

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
D = model.config.n_embd

W = {b: model.transformer.h[b].mlp.c_proj.weight.detach().float().cpu()
     for b in range(L + 1)}
WN = {b: W[b].norm(dim=-1) for b in W}
DIRS = {b: W[b] / WN[b][:, None].clamp_min(1e-8) for b in W}


def get_ids(name):
    if name == "wikitext":
        txt = "\n\n".join(t for t in load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
            if t.strip())
    else:
        try:
            ds = load_dataset("stas/openwebtext-10k", split="train")
            txt = "\n\n".join(ds["text"][:400])
        except Exception as e:
            print(f"owt fallback ({e}); using wikitext-103 train", flush=True)
            txt = "\n\n".join(t for t in load_dataset(
                "Salesforce/wikitext", "wikitext-103-raw-v1",
                split="train[:2%]")["text"] if t.strip())
    ids = tok(txt, return_tensors="pt").input_ids[0][:N_EVAL * CTX]
    return ids.view(N_EVAL, CTX)


def capture(ids):
    acts = {b: [] for b in range(L + 1)}
    hooks = [model.transformer.h[b].mlp.c_proj.register_forward_pre_hook(
        (lambda b_: lambda m, inp: acts[b_].append(inp[0].half().cpu()))(b))
        for b in range(L + 1)]
    levels = {lv: [] for lv in range(1, L + 2)}
    ces = []
    for i in range(0, N_EVAL, 4):
        out = model(ids[i:i + 4].to(DEV), output_hidden_states=True)
        for lv in levels:
            levels[lv].append(out.hidden_states[lv].float().cpu())
        lg = out.logits[:, :-1].float()
        ces.append(Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                    ids[i:i + 4, 1:].to(DEV).reshape(-1),
                                    reduction="none").cpu())
        del out
    for h in hooks:
        h.remove()
    acts = {b: torch.cat(a).view(-1, N_MLP) for b, a in acts.items()}
    levels = {lv: torch.cat(v).view(-1, D) for lv, v in levels.items()}
    mus = {lv: x.mean(0) for lv, x in levels.items()}
    levels = {lv: x - mus[lv] for lv, x in levels.items()}
    return acts, levels, torch.cat(ces), mus


def role_map(acts, levels):
    TRUE = torch.cat([acts[b].float() * WN[b] for b in range(L + 1)], dim=1)
    NT = TRUE.shape[0]
    tops = []
    for s0 in range(0, NT, 4096):
        tops.append(TRUE[s0:s0 + 4096].abs().to(DEV).topk(3, 1).indices.cpu())
    top3 = torch.cat(tops)
    out = {}
    cases = {}
    for b in range(L + 1):
        ts, js, cs = [], [], []
        for k in range(3):
            col = top3[:, k]
            m = (col // N_MLP) == b
            idx = m.nonzero()[:, 0]
            ts.append(idx); js.append(col[idx] % N_MLP)
            cs.append(TRUE[idx, col[idx]])
        ts, js, cs = torch.cat(ts), torch.cat(js), torch.cat(cs)
        s = ((levels[b + 1][ts] * DIRS[b][js]).sum(-1) / cs).clamp(-4, 5)
        out[b] = {"n": len(ts),
                  "counter": round((s < 0).float().mean().item(), 3),
                  "novel": round(((s >= 0.5) & (s < 1.5)).float().mean().item(), 3),
                  "reinf": round((s >= 1.5).float().mean().item(), 3),
                  "median": round(s.median().item(), 2)}
        cases[b] = (ts, js, cs, s)
    return out, cases


res = {}

# ---- corpus shift ----
ids_owt = get_ids("owt")
acts_o, levels_o, _, _ = capture(ids_owt)
rm_owt, _ = role_map(acts_o, levels_o)
res["rolemap_owt"] = rm_owt
print("OWT role map:", {b: (v["counter"], v["median"]) for b, v in rm_owt.items()},
      flush=True)
del acts_o, levels_o

ids_wt = get_ids("wikitext")
acts_w, levels_w, clean_ce, mus_w = capture(ids_wt)
rm_wt, cases_wt = role_map(acts_w, levels_w)
res["rolemap_wikitext"] = rm_wt

# ---- position curve (block 3, wikitext) ----
ts3, js3, cs3, s3 = cases_wt[3]
pos = ts3 % CTX
res["b3_counter_by_position"] = {
    f"{lo}-{lo+63}": round((s3[(pos >= lo) & (pos < lo + 64)] < 0)
                           .float().mean().item(), 3)
    for lo in range(0, CTX, 64)}
print("b3 counter by position:", res["b3_counter_by_position"], flush=True)

# ---- unembed base-rate check ----
FUNC = set(w.strip() for w in
           "the a an and or but of in on at to for with by from as is was were "
           "be been are am it its this that these those he she they we you i "
           ", . ; : ' \" ( ) - -- ? !".split())
W_U = model.transformer.wte.weight.detach().float().cpu()
def func_count(j):
    top = (DIRS[3][j] @ W_U.T).topk(8).indices.tolist()
    return sum(1 for t in top if tok.decode([t]).strip().lower() in FUNC)
g = torch.Generator().manual_seed(23)
rand_counts = [func_count(int(j)) for j in torch.randint(0, N_MLP, (64,), generator=g)]
res["unembed_func_words"] = {
    "mlp3_614": func_count(NEURON),
    "random_b3_mean": round(sum(rand_counts) / len(rand_counts), 2),
    "random_b3_frac_ge6": round(sum(c >= 6 for c in rand_counts) / len(rand_counts), 3)}
print("unembed func-word check:", res["unembed_func_words"], flush=True)

# ---- ablation of mlp3#614 ----
def abl_hook(m, inp):
    x = inp[0].clone()
    x[:, :, NEURON] = 0.0
    return (x,)
h = model.transformer.h[3].mlp.c_proj.register_forward_pre_hook(abl_hook)
abl_ce, abl_l6 = [], []
for i in range(0, N_EVAL, 4):
    out = model(ids_wt[i:i + 4].to(DEV), output_hidden_states=True)
    lg = out.logits[:, :-1].float()
    abl_ce.append(Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                   ids_wt[i:i + 4, 1:].to(DEV).reshape(-1),
                                   reduction="none").cpu())
    abl_l6.append(out.hidden_states[L + 1].float().cpu())
    del out
h.remove()
abl_ce = torch.cat(abl_ce)
abl_l6 = torch.cat(abl_l6).view(-1, D) - mus_w[L + 1]   # clean-mean centering

m614 = js3 == NEURON
t614 = ts3[m614]
d614 = DIRS[3][NEURON]
proj_clean = (levels_w[7][t614] @ d614)
proj_abl = (abl_l6[t614] @ d614)
tok_mask = torch.zeros(len(clean_ce), dtype=torch.bool)
valid = t614[t614 % CTX < CTX - 1]
ce_idx = (valid // CTX) * (CTX - 1) + (valid % CTX)
res["ablate_614"] = {
    "n_top3_tokens": int(m614.sum()),
    "dce_overall": round((abl_ce.mean() - clean_ce.mean()).item(), 4),
    "dce_on_614_tokens": round((abl_ce[ce_idx].mean()
                                - clean_ce[ce_idx].mean()).item(), 4),
    "proj_clean_mean": round(proj_clean.mean().item(), 2),
    "proj_ablated_mean": round(proj_abl.mean().item(), 2)}
print("ablation:", res["ablate_614"], flush=True)

with open(f"{OUT}/exp_s1_killsuite.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE_S1", flush=True)
