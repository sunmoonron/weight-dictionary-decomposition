"""X9: concentration of the largest true write. For each model, how often is a
token's largest MLP write (blocks 0..L) the same neuron? Reports the share of
tokens taken by the most frequent top-1 neuron and by the top five, and that
neuron's dominant residual channel, so that top-1 recall can be read correctly:
if one neuron is the largest write on most tokens, top-1 recall measures one
atom. Same capture and ledger conventions as exp_s9_families and
exp_x2_families_attrib (post-nonlinearity activation times write-direction norm)."""
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

torch.set_grad_enabled(False)
OUT = os.path.join(RESULTS_DIR, "exp_x9_top1conc.json")
CTX, N_MEAN, N_EVAL = 512, 8, 16
MODELS = {
    "llama-7b": ("huggyllama/llama-7b", "llama"),
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B", "llama"),
    "pythia-6.9b": ("EleutherAI/pythia-6.9b", "neox"),
    "gpt2": ("gpt2", "gpt2"),
    "smollm2-135m": ("HuggingFaceTB/SmolLM2-135M", "llama"),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", "llama"),
    "pythia-410m": ("EleutherAI/pythia-410m", "neox"),
    "olmo-1b-0724": ("allenai/OLMo-1B-0724-hf", "llama"),
}
BF16 = {"qwen2.5-7b", "pythia-6.9b", "llama-7b"}
ONLY = [m for m in os.environ.get("WDD_MODELS", "").split(",") if m]
text = "\n\n".join(t for t in load_dataset(
    "Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
RES = json.load(open(OUT)) if os.path.exists(OUT) else {}

for name, (hf, fam) in MODELS.items():
    if name in RES or (ONLY and name not in ONLY):
        continue
    print(f"== {name} ==", flush=True)
    dt = (torch.bfloat16 if name in BF16 else
          torch.float16 if name == "olmo-1b-0724" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        hf, torch_dtype=dt, device_map="auto").eval()
    tok = AutoTokenizer.from_pretrained(hf)
    cfg = model.config
    if fam == "gpt2":
        layers, NB, L = model.transformer.h, cfg.n_layer, 6
        mlp_lin = lambda l: l.mlp.c_proj
        wdir = lambda l: l.mlp.c_proj.weight.detach().float().cpu()        # [dff, d]
    elif fam == "llama":
        layers, NB = model.model.layers, cfg.num_hidden_layers
        L = NB // 2
        mlp_lin = lambda l: l.mlp.down_proj
        wdir = lambda l: l.mlp.down_proj.weight.detach().float().cpu().T  # [dff, d]
    else:
        layers, NB = model.gpt_neox.layers, cfg.num_hidden_layers
        L = NB // 2
        mlp_lin = lambda l: l.mlp.dense_4h_to_h
        wdir = lambda l: l.mlp.dense_4h_to_h.weight.detach().float().cpu().T
    ids = tok(text, return_tensors="pt").input_ids[0][
        :(N_MEAN + N_EVAL) * CTX].view(-1, CTX)
    dev0 = next(model.parameters()).device
    acts = {b: [] for b in range(L + 1)}
    hooks = [mlp_lin(layers[b]).register_forward_pre_hook(
        (lambda b_: lambda m, inp: acts[b_].append(inp[0].half().cpu()))(b))
        for b in range(L + 1)]
    for i in range(N_MEAN + N_EVAL):
        model(ids[i:i + 1].to(dev0))
    for h in hooks:
        h.remove()
    n_mu = N_MEAN * CTX
    W = {b: wdir(layers[b]) for b in range(L + 1)}
    DFF = W[0].shape[0]
    TRUE = torch.cat([torch.cat(acts[b]).view(-1, DFF)[n_mu:].float()
                      * W[b].norm(dim=1) for b in range(L + 1)], dim=1)
    top1 = TRUE.abs().argmax(1)
    cnt = torch.bincount(top1, minlength=TRUE.shape[1]).float()
    share = cnt / cnt.sum()
    order = share.argsort(descending=True)
    rows = []
    for idx in order[:5].tolist():
        b, j = idx // DFF, idx % DFF
        d = W[b][j] / W[b][j].norm()
        ch = int(d.abs().argmax())
        rows.append({"neuron": f"mlp{b}#{j}",
                     "share_of_tokens": round(share[idx].item(), 4),
                     "dominant_channel": ch,
                     "channel_share": round((d[ch] ** 2).item(), 3)})
    p = share[share > 0]
    RES[name] = {"L": L, "n_tokens": int(TRUE.shape[0]),
                 "top1_neuron_rows": rows,
                 "distinct_top1_neurons": int((cnt > 0).sum()),
                 "top1_entropy_bits": round(float(-(p * p.log2()).sum()), 2),
                 "top5_share": round(sum(r["share_of_tokens"] for r in rows), 4)}
    print(RES[name], flush=True)
    json.dump(RES, open(OUT, "w"), indent=1)
    del model, acts, TRUE
    torch.cuda.empty_cache()
print("DONE_X9", flush=True)
