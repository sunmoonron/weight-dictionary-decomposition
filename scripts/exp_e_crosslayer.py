"""Experiment E: is the kNN lens a LENS, or just kNN-LM in disguise?

Reviewer's test: decode layer-6 query states against banks that break the
same-space premise in different ways.
  bank6            same layer (the lens; replicates exp B)
  bank5 / bank7    adjacent layers -- the residual stream drifts slowly, so if
                   these work, the honest claim is "same coordinate SPACE",
                   not "same layer index"
  bank6_rotated    bank states through one fixed orthogonal rotation --
                   preserves the bank's internal geometry, destroys the shared
                   coordinate system (the analogue of Logit Lens basis
                   mismatch, induced deliberately)
  bank6_whitened   ZCA-whitened space (bank covariance), both sides
  random_neighbors 32 uniformly random bank rows per query (floor)
Metrics: top-1 accuracy vs truth, and CE with the lam=0.5 unigram backoff.
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

CTX, N_BANK, N_QUERY, K, VOCAB, LAM = 512, 400, 16, 32, 50257, 0.5
QL = 6                      # query layer
BANK_LEVELS = [5, 6, 7]

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
D = model.config.n_embd


def chunks(split, n):
    text = "\n\n".join(t for t in load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"] if t.strip())
    return tok(text, return_tensors="pt").input_ids[0][:n * CTX].view(n, CTX)


def harvest(ids, levels, batch=2):
    states = {l: [] for l in levels}
    for i in range(0, len(ids), batch):
        out = model(ids[i:i + batch].to(DEV), output_hidden_states=True)
        for l in levels:
            states[l].append(out.hidden_states[l][:, :-1].half().cpu())
        del out
    return ({l: torch.cat(s).view(-1, D) for l, s in states.items()},
            ids[:, 1:].reshape(-1))


bank_states, bank_labels = harvest(chunks("train", N_BANK), BANK_LEVELS)
q_states, q_labels = harvest(chunks("test", N_QUERY), [QL])
q6 = q_states[QL].float()
NQ = len(q_labels)
del model
torch.cuda.empty_cache()

unigram = torch.bincount(bank_labels, minlength=VOCAB).float()
unigram = (unigram + 0.5) / (unigram.sum() + 0.5 * VOCAB)
uni_true = unigram[q_labels].to(DEV)
bank_labels_dev = bank_labels.to(DEV)
q_labels_dev = q_labels.to(DEV)

g = torch.Generator().manual_seed(7)
Q_ROT = torch.linalg.qr(torch.randn(D, D, generator=g))[0]

b6 = bank_states[6].float()
mu6 = b6.mean(0)
cov = ((b6 - mu6).T @ (b6 - mu6)) / len(b6)
ev, U = torch.linalg.eigh(cov.to(DEV))
W_ZCA = (U @ torch.diag((ev + 1e-4).rsqrt()) @ U.T).cpu()


def evaluate(bank, queries, tag):
    bank_n = F.normalize(bank, dim=-1).to(DEV)
    ce_sum = 0.0
    hits = 0
    for s in range(0, NQ, 512):
        q = F.normalize(queries[s:s + 512], dim=-1).to(DEV)
        nb = bank_labels_dev[(q @ bank_n.T).topk(K, -1).indices]
        truth = q_labels_dev[s:s + 512]
        emp = (nb == truth[:, None]).float().mean(-1)
        p = LAM * emp + (1 - LAM) * uni_true[s:s + 512]
        ce_sum += -p.log().sum().item()
        hits += (nb.mode(-1).values == truth).sum().item()
        del q, nb
    del bank_n
    torch.cuda.empty_cache()
    row = {"top1": hits / NQ, "ce_lam05": ce_sum / NQ}
    print(f"{tag:>18}: top1 {row['top1']:.3f}  CE {row['ce_lam05']:.3f}", flush=True)
    return row


results = {"unigram_ce": -unigram.log()[q_labels].mean().item()}
print(f"unigram-only CE: {results['unigram_ce']:.3f}", flush=True)

results["bank6"] = evaluate(b6, q6, "bank6 (same layer)")
results["bank5"] = evaluate(bank_states[5].float(), q6, "bank5")
results["bank7"] = evaluate(bank_states[7].float(), q6, "bank7")
results["bank6_rotated"] = evaluate(b6 @ Q_ROT, q6, "bank6 rotated")
results["bank6_whitened"] = evaluate((b6 - mu6) @ W_ZCA, (q6 - mu6) @ W_ZCA,
                                     "bank6 whitened")

# random-neighbor floor: 32 uniform bank rows per query
idx = torch.randint(0, len(bank_labels), (NQ, K), generator=g)
nb = bank_labels[idx].to(DEV)
emp = (nb == q_labels_dev[:, None]).float().mean(-1)
p = LAM * emp + (1 - LAM) * uni_true
results["random_neighbors"] = {
    "top1": (nb.mode(-1).values == q_labels_dev).float().mean().item(),
    "ce_lam05": (-p.log()).mean().item()}
print(f"  random neighbors: top1 {results['random_neighbors']['top1']:.3f}  "
      f"CE {results['random_neighbors']['ce_lam05']:.3f}", flush=True)

with open(f"{OUT}/exp_e_crosslayer.json", "w") as f:
    json.dump(results, f, indent=1)
print("DONE_E", flush=True)
