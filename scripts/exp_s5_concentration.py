"""Write Dynamics Ep.2c: concentration of counter-writing, all blocks, both
models. Separates "diffuse specialization" (many neurons) from "mega-channel
ritual" (one persistent neuron) -- the distinction exp_s3 exposed for b9.
Per block: among counter cases (birth survival < 0): n, distinct neurons,
top-1 and top-5 neuron share."""

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
CTX = 512

text = None
res = {}
for name, nev, batch in [("gpt2", 64, 4), ("gpt2-medium", 32, 2)]:
    model = AutoModelForCausalLM.from_pretrained(name).to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(name)
    D = model.config.n_embd
    NB = model.config.n_layer
    NM = 4 * D
    if text is None:
        text = "\n\n".join(t for t in load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
            if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0][:nev * CTX].view(nev, CTX)

    acts = {b: [] for b in range(NB)}
    hooks = [model.transformer.h[b].mlp.c_proj.register_forward_pre_hook(
        (lambda b_: lambda m, inp: acts[b_].append(inp[0].half().cpu()))(b))
        for b in range(NB)]
    levels = {lv: [] for lv in range(1, NB + 1)}
    for i in range(0, nev, batch):
        out = model(ids[i:i + batch].to(DEV), output_hidden_states=True)
        for lv in levels:
            levels[lv].append(out.hidden_states[lv].float().cpu())
        del out
    for h in hooks:
        h.remove()
    acts = {b: torch.cat(a).view(-1, NM) for b, a in acts.items()}
    levels = {lv: torch.cat(v).view(-1, D) for lv, v in levels.items()}
    levels = {lv: x - x.mean(0) for lv, x in levels.items()}
    NT = levels[1].shape[0]

    WN, DIRS = {}, {}
    for b in range(NB):
        w = model.transformer.h[b].mlp.c_proj.weight.detach().float().cpu()
        WN[b] = w.norm(dim=-1)
        DIRS[b] = w / WN[b][:, None].clamp_min(1e-8)
    TRUE = torch.cat([acts[b].float() * WN[b] for b in range(NB)], dim=1).half()
    del acts
    tops = []
    for s0 in range(0, NT, 2048):
        tops.append(TRUE[s0:s0 + 2048].float().to(DEV).abs()
                    .topk(3, 1).indices.cpu())
    top3 = torch.cat(tops)

    rows = {}
    for b in range(NB):
        col = top3.flatten()
        m = (col // NM) == b
        if m.sum() < 100:
            continue
        ts = torch.arange(NT)[:, None].expand(-1, 3).flatten()[m]
        js = (col % NM)[m]
        cs = TRUE[ts, col[m]].float()
        s = ((levels[b + 1][ts] * DIRS[b][js]).sum(-1) / cs).clamp(-4, 5)
        cm = s < 0
        if cm.sum() < 50:
            continue
        jc = js[cm]
        _, cnt = jc.unique(return_counts=True)
        sh = cnt.sort(descending=True).values.float() / len(jc)
        rows[b] = {"n_counter": int(cm.sum()),
                   "counter_frac": round(cm.float().mean().item(), 3),
                   "distinct": int(len(cnt)),
                   "top1": round(sh[0].item(), 3),
                   "top5": round(sh[:5].sum().item(), 3)}
        print(f"{name} b{b:>2}: counter n={rows[b]['n_counter']:>5} "
              f"({rows[b]['counter_frac']:.2f})  neurons={rows[b]['distinct']:>4} "
              f"top1={rows[b]['top1']:.2f} top5={rows[b]['top5']:.2f}", flush=True)
    res[name] = rows
    del model, TRUE, levels
    torch.cuda.empty_cache()

with open(f"{OUT}/exp_s5_concentration.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE_S5", flush=True)
