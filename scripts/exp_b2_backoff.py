"""Experiment B2: kNN lens CE with unigram backoff.

Raw 32-neighbor empirical distributions are spiky: any miss costs ~20 nats,
so exp B's CE column was dominated by smoothing artifacts. Standard fix from
the kNN-LM literature: back off to the bank's unigram distribution,
p = lam*empirical + (1-lam)*unigram. Still zero trained parameters.
Also reports the unigram-only CE floor (decode with NO state information),
which is the reference any lens must beat.
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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(0)
torch.set_grad_enabled(False)
DEV = "cuda:0"
OUT = RESULTS_DIR

CTX, N_BANK, N_QUERY, K, VOCAB = 512, 400, 16, 32, 50257
LAMBDAS = [0.5, 0.9]

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
D = model.config.n_embd


def chunks(split, n):
    text = "\n\n".join(t for t in load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"] if t.strip())
    return tok(text, return_tensors="pt").input_ids[0][:n * CTX].view(n, CTX)


def harvest(ids, batch=2):
    states = [[] for _ in range(13)]
    for i in range(0, len(ids), batch):
        out = model(ids[i:i + batch].to(DEV), output_hidden_states=True)
        for l in range(13):
            states[l].append(out.hidden_states[l][:, :-1].half().cpu())
        del out
    return [torch.cat(s).view(-1, D) for s in states], ids[:, 1:].reshape(-1)


bank_states, bank_labels = harvest(chunks("train", N_BANK))
q_states, q_labels = harvest(chunks("test", N_QUERY))
NQ = len(q_labels)
del model
torch.cuda.empty_cache()

unigram = torch.bincount(bank_labels, minlength=VOCAB).float()
unigram = (unigram + 0.5) / (unigram.sum() + 0.5 * VOCAB)     # tiny add-half
uni_ce = -unigram.log()[q_labels].mean().item()
print(f"unigram-only CE floor: {uni_ce:.4f}", flush=True)

bank_labels_dev = bank_labels.to(DEV)
uni_true = unigram[q_labels].to(DEV)                          # [NQ]
q_labels_dev = q_labels.to(DEV)

results = {"unigram_ce": uni_ce, "levels": {}}
for l in range(13):
    bank = F.normalize(bank_states[l].float(), dim=-1).to(DEV)
    ce = {lam: 0.0 for lam in LAMBDAS}
    for s in range(0, NQ, 512):
        q = F.normalize(q_states[l][s:s + 512].float().to(DEV), dim=-1)
        nb = bank_labels_dev[(q @ bank.T).topk(K, -1).indices]
        emp = (nb == q_labels_dev[s:s + 512, None]).float().mean(-1)
        for lam in LAMBDAS:
            p = lam * emp + (1 - lam) * uni_true[s:s + 512]
            ce[lam] += -p.log().sum().item()
        del q, nb
    del bank
    torch.cuda.empty_cache()
    row = {f"knn{K}_ce_lam{lam}": ce[lam] / NQ for lam in LAMBDAS}
    results["levels"][l] = row
    print(f"L{l:>2}  " + "  ".join(
        f"CE(lam={lam}) {ce[lam]/NQ:.3f}" for lam in LAMBDAS), flush=True)

with open(f"{OUT}/exp_b2_backoff.json", "w") as f:
    json.dump(results, f, indent=1)
print("DONE_B2", flush=True)
