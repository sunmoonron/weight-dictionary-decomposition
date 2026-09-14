"""Write Dynamics Ep.3a: the UNCONDITIONAL oligarchy test + channel IDs +
superweight overlap + energy-matched ablation.

No dominant-write selection anywhere: per block, the full survival matrix
S = (post_state @ D^T) / C over every (token, neuron) write with |c| > 0.5.
Reported per block: counter fraction (count- and energy-weighted), and the
share of total counter-write ENERGY carried by the top 1/5/20 neurons.
Also: the named top counter channels; whether each writes predominantly into
one residual channel (super-activation overlap check; small's is ch447);
and for small b3, ablation of the 9 counter-neurons vs 9 ENERGY-matched
non-counter neurons (absolute dCE, not just the ratio-vs-random).
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
CTX = 512

text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())

res = {}
for name, nev, batch in [("gpt2", 64, 4), ("gpt2-medium", 32, 2)]:
    model = AutoModelForCausalLM.from_pretrained(name).to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(name)
    D = model.config.n_embd
    NB, NM = model.config.n_layer, 4 * model.config.n_embd
    ids = tok(text, return_tensors="pt").input_ids[0][:nev * CTX].view(nev, CTX)

    acts = {b: [] for b in range(NB)}
    hooks = [model.transformer.h[b].mlp.c_proj.register_forward_pre_hook(
        (lambda b_: lambda m, inp: acts[b_].append(inp[0].half().cpu()))(b))
        for b in range(NB)]
    levels = {lv: [] for lv in range(1, NB + 1)}
    ces = []
    for i in range(0, nev, batch):
        out = model(ids[i:i + batch].to(DEV), output_hidden_states=True)
        for lv in levels:
            levels[lv].append(out.hidden_states[lv].float().cpu())
        lg = out.logits[:, :-1].float()
        ces.append(Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                    ids[i:i + batch, 1:].to(DEV).reshape(-1),
                                    reduction="none").cpu())
        del out
    for h in hooks:
        h.remove()
    acts = {b: torch.cat(a).view(-1, NM) for b, a in acts.items()}
    levels = {lv: torch.cat(v).view(-1, D) for lv, v in levels.items()}
    levels = {lv: x - x.mean(0) for lv, x in levels.items()}
    clean_ce = torch.cat(ces)
    NT = levels[1].shape[0]

    rows = {}
    chan_info = {}
    for b in range(NB):
        w = model.transformer.h[b].mlp.c_proj.weight.detach().float()
        wn = w.norm(dim=-1)
        Db = (w / wn[:, None].clamp_min(1e-8)).to(DEV)
        C = (acts[b].float() * wn.cpu()).to(DEV)              # [NT, NM]
        S = (levels[b + 1].to(DEV) @ Db.T) / C.where(C.abs() > 1e-6,
                                                     torch.ones_like(C))
        mask = C.abs() > 0.5
        counter = (S < 0) & mask
        n_writes = mask.sum().item()
        e_all = ((C ** 2) * mask).sum().item()
        e_cnt_per = ((C ** 2) * counter).sum(0)               # [NM]
        e_cnt = e_cnt_per.sum().item()
        share = e_cnt_per.sort(descending=True).values / max(e_cnt, 1e-9)
        top_ids = e_cnt_per.topk(3).indices.tolist()
        rows[b] = {"writes": n_writes,
                   "counter_count": round((counter.sum() / max(n_writes, 1))
                                          .item(), 3),
                   "counter_energy": round(e_cnt / max(e_all, 1e-9), 3),
                   "top1_e": round(share[0].item(), 3),
                   "top5_e": round(share[:5].sum().item(), 3),
                   "top20_e": round(share[:20].sum().item(), 3),
                   "top_ids": top_ids}
        # channel spatial concentration: does the top counter neuron write
        # into one residual channel?
        d_top = Db[top_ids[0]].cpu()
        ch = int(d_top.abs().argmax())
        chan_info[b] = {"neuron": top_ids[0], "res_channel": ch,
                        "channel_share": round((d_top[ch] ** 2).item(), 3)}
        del Db, C, S, mask, counter
        torch.cuda.empty_cache()
    res[name] = {"blocks": rows, "top_channels": chan_info}
    for b, r in rows.items():
        print(f"{name} b{b:>2}: cnt {r['counter_count']:.3f} "
              f"energy {r['counter_energy']:.3f}  "
              f"topE 1/5/20: {r['top1_e']:.2f}/{r['top5_e']:.2f}/"
              f"{r['top20_e']:.2f}  top {r['top_ids'][0]} "
              f"(ch{chan_info[b]['res_channel']}, "
              f"{chan_info[b]['channel_share']:.2f})", flush=True)

    if name == "gpt2":
        # energy-matched ablation for the 9 known b3 counter-neurons
        b = 3
        w = model.transformer.h[b].mlp.c_proj.weight.detach().float().cpu()
        wn = w.norm(dim=-1)
        d3 = w / wn[:, None].clamp_min(1e-8)
        C3 = acts[b].float() * wn
        S3 = (levels[b + 1] @ d3.T) / C3.where(C3.abs() > 1e-6,
                                               torch.ones_like(C3))
        big = C3.abs() > 0.5
        cnt3 = (S3 < 0) & big
        per_n = cnt3.sum(0).float()
        per_cnt_rate = per_n / big.sum(0).clamp_min(1)
        per_energy = (C3 ** 2 * big).sum(0)
        counter_set = ((per_n >= 10) & (per_cnt_rate > 0.5)).nonzero()[:, 0]
        counter_set = counter_set[per_energy[counter_set]
                                  .topk(min(9, len(counter_set))).indices].tolist()
        # energy-matched non-counter picks
        noncounter = ((per_cnt_rate < 0.1).nonzero()[:, 0]).tolist()
        matched = []
        used = set()
        for j in counter_set:
            tgt = per_energy[j]
            best, bd = None, None
            for q in noncounter:
                if q in used:
                    continue
                dd = abs(per_energy[q].item() - tgt.item())
                if bd is None or dd < bd:
                    best, bd = q, dd
            matched.append(best)
            used.add(best)

        def ablate(neurons):
            idx = torch.tensor(neurons, device=DEV)

            def hook(m, inp):
                x = inp[0].clone()
                x[:, :, idx] = 0.0
                return (x,)

            hh = model.transformer.h[b].mlp.c_proj \
                .register_forward_pre_hook(hook)
            tot = []
            try:
                for i in range(0, nev, batch):
                    lg = model(ids[i:i + batch].to(DEV)).logits[:, :-1].float()
                    tot.append(Fn.cross_entropy(
                        lg.reshape(-1, lg.shape[-1]),
                        ids[i:i + batch, 1:].to(DEV).reshape(-1),
                        reduction="none").cpu())
            finally:
                hh.remove()
            return (torch.cat(tot).mean() - clean_ce.mean()).item()

        res["gpt2"]["b3_ablation"] = {
            "counter_set": counter_set,
            "matched_set": matched,
            "clean_ce": round(clean_ce.mean().item(), 4),
            "dce_counter": round(ablate(counter_set), 4),
            "dce_energy_matched": round(ablate(matched), 4),
            "counter_energy_sum": round(per_energy[counter_set].sum().item(), 0),
            "matched_energy_sum": round(per_energy[
                torch.tensor(matched)].sum().item(), 0)}
        print("b3 ablation:", res["gpt2"]["b3_ablation"], flush=True)
    del model, acts, levels
    torch.cuda.empty_cache()

with open(f"{OUT}/exp_s6_uncond.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE_S6", flush=True)
