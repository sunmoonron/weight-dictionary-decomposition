"""Write Dynamics Ep.3b: does counter-writing survive removal of the massive
channels?

Massive residual channels (from the superweight study's detection JSON + S6
tops): gpt2 {138, 266, 447, 481, 496}; gpt2-medium {9, 265, 268, 590, 689,
909}. Zero those coordinates in both the state and each write direction, and
recompute the per-block-dominant-write role map in the complement space.
Cases whose write lives mostly in the massive channels (complement norm <=
0.3) are dropped and counted -- that drop-rate IS the "fraction of dominant
counter-writing that is channel regulation".

Collapse of the counter structure => the whole phenomenon is massive-channel
regulation. Persistence => a second, non-channel counter-writing population.
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
CTX = 512
CHANNELS = {"gpt2": [138, 266, 447, 481, 496],
            "gpt2-medium": [9, 265, 268, 590, 689, 909]}

text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())

res = {}
for name, nev, batch in [("gpt2", 64, 4), ("gpt2-medium", 32, 2)]:
    model = AutoModelForCausalLM.from_pretrained(name).to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(name)
    D = model.config.n_embd
    NB, NM = model.config.n_layer, 4 * D
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
    ch = torch.tensor(CHANNELS[name])

    rows = {}
    for b in range(NB):
        w = model.transformer.h[b].mlp.c_proj.weight.detach().float().cpu()
        wn = w.norm(dim=-1)
        Db = w / wn[:, None].clamp_min(1e-8)
        C = acts[b].float() * wn
        # per-block dominant frame: top-3 writes within this block per token
        tk = C.abs().topk(3, dim=1).indices                    # [NT, 3]
        cs = C.gather(1, tk)
        d = Db[tk]                                             # [NT, 3, D]
        x = levels[b + 1][:, None, :]                          # [NT, 1, D]
        s_before = ((x * d).sum(-1) / cs).clamp(-4, 5)
        # complement space
        d2 = d.clone()
        d2[:, :, ch] = 0.0
        n2 = d2.norm(dim=-1)                                   # [NT, 3]
        x2 = levels[b + 1].clone()
        x2[:, ch] = 0.0
        keep = n2 > 0.3
        s_after = ((x2[:, None, :] * d2).sum(-1)
                   / (cs * n2.clamp_min(1e-6) ** 2)).clamp(-4, 5)
        cb = (s_before < 0)
        ca = (s_after < 0) & keep
        rows[b] = {
            "counter_before": round(cb.float().mean().item(), 3),
            "dropped_frac": round((~keep).float().mean().item(), 3),
            "dropped_counter_frac": round(
                (cb & ~keep).float().sum().item()
                / max(cb.float().sum().item(), 1), 3),
            "counter_after": round(
                ca.float().sum().item() / max(keep.float().sum().item(), 1), 3)}
        print(f"{name} b{b:>2}: counter {rows[b]['counter_before']:.3f} -> "
              f"{rows[b]['counter_after']:.3f}  "
              f"(dropped {rows[b]['dropped_frac']:.2f} of writes; "
              f"{rows[b]['dropped_counter_frac']:.2f} of counter cases were "
              f"channel-writes)", flush=True)
    res[name] = rows
    del model, acts, levels
    torch.cuda.empty_cache()

with open(f"{OUT}/exp_s7_dechannel.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE_S7", flush=True)
