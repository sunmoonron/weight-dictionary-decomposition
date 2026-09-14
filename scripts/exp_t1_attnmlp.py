"""Ep.6: the attention -> MLP counter circuit (GPT-2 small, mlp5#2070).

S8 found 61-64% of what mlp5#2070 (and #1505) oppose is attention-written.
Close the loop:
  attribute  per-head contribution along d_2070 (exact: head slice of the
             c_proj input dotted with W_O_h @ d), blocks 0-4, on the tokens
             where #2070 counter-writes -> name the source heads
  perturb    scale the top head's contribution (alpha 0 / 2) and measure:
             does #2070's activation respond (thermostat test)? does the
             opposition projection and #2070's own write move as predicted?
             dCE. Control: a random other head, same block, same alphas.
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
TGT_B, TGT_J = 5, 2070

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
D = model.config.n_embd

text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0][:N_EVAL * CTX].view(N_EVAL, CTX)

w5 = model.transformer.h[TGT_B].mlp.c_proj.weight.detach().float().cpu()
wn5 = w5.norm(dim=-1)
d = (w5[TGT_J] / wn5[TGT_J]).clone()


def run(head_scale=None):
    """head_scale = (block, head, alpha) or None. Captures target activation,
    pre/post-b5 projections on d, attn c_proj inputs blocks 0-4, CE."""
    attn_in = {b: [] for b in range(TGT_B)}
    hooks = []
    for b in range(TGT_B):
        def mk(b_):
            def f(m, inp):
                x = inp[0]
                if head_scale is not None and head_scale[0] == b_:
                    x = x.clone()
                    h_, al = head_scale[1], head_scale[2]
                    x[:, :, h_ * 64:(h_ + 1) * 64] *= al
                    attn_in[b_].append(x.half().cpu())
                    return (x,)
                attn_in[b_].append(x.half().cpu())
            return f
        hooks.append(model.transformer.h[b].attn.c_proj
                     .register_forward_pre_hook(mk(b)))
    a_t, pre_p, post_p, ces = [], [], [], []
    hooks.append(model.transformer.h[TGT_B].mlp.c_proj.register_forward_pre_hook(
        lambda m, inp: a_t.append(inp[0][:, :, TGT_J].float().cpu())))
    d_dev = d.to(DEV)
    for i in range(0, N_EVAL, 4):
        out = model(ids[i:i + 4].to(DEV), output_hidden_states=True)
        pre_p.append((out.hidden_states[TGT_B] @ d_dev).float().cpu())
        post_p.append((out.hidden_states[TGT_B + 1] @ d_dev).float().cpu())
        lg = out.logits[:, :-1].float()
        ces.append(Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                    ids[i:i + 4, 1:].to(DEV).reshape(-1),
                                    reduction="none").cpu())
        del out
    for h in hooks:
        h.remove()
    return {"attn_in": {b: torch.cat(v).view(-1, D) for b, v in attn_in.items()},
            "a": torch.cat(a_t).flatten(),
            "pre": torch.cat(pre_p).flatten(), "post": torch.cat(post_p).flatten(),
            "ce": torch.cat(ces).mean().item()}


base = run()
NT = len(base["a"])
pre_c = base["pre"] - base["pre"].mean()
write = base["a"] * wn5[TGT_J]
counter_tok = (write.abs() > 0.5) & ((base["post"] - base["post"].mean())
                                     * torch.sign(write) < 0)
print(f"target mlp5#{TGT_J}: {int(counter_tok.sum())} counter tokens; "
      f"clean CE {base['ce']:.4f}", flush=True)

# ---- per-head attribution along d, blocks 0-4 ----
contrib = {}
for b in range(TGT_B):
    W_O = model.transformer.h[b].attn.c_proj.weight.detach().float().cpu()
    x = base["attn_in"][b].float()
    for h in range(12):
        v = W_O[h * 64:(h + 1) * 64] @ d                 # [64]
        c = (x[:, h * 64:(h + 1) * 64] @ v)
        contrib[(b, h)] = round(c[counter_tok].mean().item(), 3)
top = sorted(contrib.items(), key=lambda kv: kv[1])[:5]
res = {"clean_ce": round(base["ce"], 4),
       "n_counter_tokens": int(counter_tok.sum()),
       "top_opposing_heads": {f"b{b}h{h}": v for (b, h), v in top}}
print("top opposition-building heads:", res["top_opposing_heads"], flush=True)

# ---- perturb the top head + a control head ----
(b_star, h_star), _ = top[0]
g = torch.Generator().manual_seed(13)
h_ctrl = int(torch.randint(0, 12, (1,), generator=g))
if h_ctrl == h_star:
    h_ctrl = (h_ctrl + 1) % 12
res["perturb"] = {}
for tag, (bb, hh) in [("top", (b_star, h_star)), ("control", (b_star, h_ctrl))]:
    for al in [0.0, 2.0]:
        p = run(head_scale=(bb, hh, al))
        da = (p["a"].abs().mean() - base["a"].abs().mean()) \
            / base["a"].abs().mean().clamp_min(1e-6)
        da_ct = (p["a"][counter_tok].abs().mean()
                 - base["a"][counter_tok].abs().mean()) \
            / base["a"][counter_tok].abs().mean().clamp_min(1e-6)
        dpre = (p["pre"][counter_tok] - base["pre"][counter_tok]).mean()
        res["perturb"][f"{tag}_b{bb}h{hh}_a{al}"] = {
            "d_act_all": round(da.item(), 3),
            "d_act_counter_tok": round(da_ct.item(), 3),
            "d_opposition_proj": round(dpre.item(), 3),
            "dce": round(p["ce"] - base["ce"], 4)}
        print(f"{tag} b{bb}h{hh} a={al}: dact {da:.3f} (ct {da_ct:.3f}) "
              f"dopp {dpre:.2f} dCE {p['ce']-base['ce']:+.4f}", flush=True)

with open(f"{OUT}/exp_t1_attnmlp.json", "w") as f:
    json.dump(res, f, indent=1)
print("DONE_T1", flush=True)
