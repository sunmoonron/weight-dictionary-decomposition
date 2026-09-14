"""Experiment B: the kNN lens -- a nonparametric, training-free Tuned Lens.

Hypothesis: you can decode an intermediate hidden state without ANY trained
translator by comparing it like-with-like: retrieve the k nearest layer-l bank
states from a corpus and read out their empirical next-token distribution.
Basis mismatch (Logit Lens's failure mode) vanishes by construction, since
layer-l states are only ever compared to layer-l states. The retrieved
exemplar contexts are the decode -- concepts need not be single tokens.

Levels: hidden_states[0] = embeddings, [l] = post block l-1 (l=1..11),
[12] = post-ln_f final. GPT-2 small, wikitext-2.
"""

import json
import os
import time

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

OUT = RESULTS_DIR
os.makedirs(OUT, exist_ok=True)
DEV = "cuda:0"

CTX = 512
N_BANK_CHUNKS = 400            # ~204k bank tokens (wikitext-2 train)
N_QUERY_CHUNKS = 16            # ~8k held-out query tokens (wikitext-2 test)
N_LEVELS = 13
KS = [8, 32]                   # neighbor counts to evaluate
EPS = 1e-4                     # smoothing mass for CE
VOCAB = 50257

model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
D = model.config.n_embd
W_U = model.transformer.wte.weight.detach().clone()
LN_W = model.transformer.ln_f.weight.detach().clone()
LN_B = model.transformer.ln_f.bias.detach().clone()


def token_chunks(split, n_chunks):
    text = "\n\n".join(t for t in load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0][: n_chunks * CTX]
    assert len(ids) == n_chunks * CTX
    return ids.view(n_chunks, CTX)


def harvest(ids, want_final=False, batch=2):
    """All-level states for positions 0..CTX-2 (fp16 CPU), next-token labels,
    and optionally the model's final top-1 prediction + final CE per position."""
    states = [[] for _ in range(N_LEVELS)]
    final_top1, final_ce = [], []
    for i in range(0, len(ids), batch):
        out = model(ids[i:i + batch].to(DEV), output_hidden_states=True)
        for l in range(N_LEVELS):
            states[l].append(out.hidden_states[l][:, :-1].half().cpu())
        if want_final:
            lg = out.logits[:, :-1].float()
            final_top1.append(lg.argmax(-1).cpu())
            tgt = ids[i:i + batch, 1:].to(DEV)
            final_ce.append(F.cross_entropy(
                lg.reshape(-1, VOCAB), tgt.reshape(-1),
                reduction="none").cpu())
        del out
    states = [torch.cat(s).view(-1, D) for s in states]
    labels = ids[:, 1:].reshape(-1)
    if want_final:
        return states, labels, torch.cat(final_top1).view(-1), torch.cat(final_ce)
    return states, labels


t0 = time.time()
print("building bank...", flush=True)
bank_states, bank_labels = harvest(token_chunks("train", N_BANK_CHUNKS))
print(f"  bank {bank_states[0].shape[0]} states/level in {time.time()-t0:.0f}s",
      flush=True)
q_states, q_labels, q_final_top1, q_final_ce = harvest(
    token_chunks("test", N_QUERY_CHUNKS), want_final=True)
NQ = q_labels.shape[0]
print(f"  queries {NQ}; model final CE {q_final_ce.mean().item():.4f}", flush=True)

# model no longer needed on GPU; free it for the search phase
del model
torch.cuda.empty_cache()

bank_labels_dev = bank_labels.to(DEV)
q_labels_dev = q_labels.to(DEV)
q_final_dev = q_final_top1.to(DEV)
results = {"final_ce": q_final_ce.mean().item(), "levels": {}}

for l in range(N_LEVELS):
    t0 = time.time()
    bank = F.normalize(bank_states[l].float(), dim=-1).to(DEV)      # [NB, D]
    ce_sum = {k: 0.0 for k in KS}
    top1_hits = {k: 0 for k in KS}
    agree_hits = {k: 0 for k in KS}
    lens_ce_sum = 0.0
    lens_top1 = lens_agree = 0
    for s in range(0, NQ, 512):
        q_raw = q_states[l][s:s + 512].float().to(DEV)
        q = F.normalize(q_raw, dim=-1)
        sims = q @ bank.T                                           # [b, NB]
        nb = bank_labels_dev[sims.topk(max(KS), -1).indices]        # [b, 32]
        del sims
        truth = q_labels_dev[s:s + 512]
        fin = q_final_dev[s:s + 512]
        for k in KS:
            top = nb[:, :k]
            # empirical next-token distribution over the k neighbors
            p_true = (top == truth[:, None]).float().mean(-1)
            p_true = p_true * (1 - EPS) + EPS / VOCAB
            ce_sum[k] += -p_true.log().sum().item()
            mode = top.mode(-1).values
            top1_hits[k] += (mode == truth).sum().item()
            agree_hits[k] += (mode == fin).sum().item()
        # logit lens on the same states
        if l < 12:
            h = F.layer_norm(q_raw, (D,), LN_W, LN_B)
        else:
            h = q_raw                                               # already ln_f'd
        lg = h @ W_U.T
        lens_ce_sum += F.cross_entropy(lg, truth, reduction="sum").item()
        pred = lg.argmax(-1)
        lens_top1 += (pred == truth).sum().item()
        lens_agree += (pred == fin).sum().item()
        del q_raw, q, nb, lg
    del bank
    torch.cuda.empty_cache()
    row = {"lens_ce": lens_ce_sum / NQ, "lens_top1": lens_top1 / NQ,
           "lens_agree": lens_agree / NQ}
    for k in KS:
        row[f"knn{k}_ce"] = ce_sum[k] / NQ
        row[f"knn{k}_top1"] = top1_hits[k] / NQ
        row[f"knn{k}_agree"] = agree_hits[k] / NQ
    results["levels"][l] = row
    print(f"L{l:>2}  knn32: CE {row['knn32_ce']:.3f} top1 {row['knn32_top1']:.3f} "
          f"agree {row['knn32_agree']:.3f}   lens: CE {row['lens_ce']:.3f} "
          f"top1 {row['lens_top1']:.3f} agree {row['lens_agree']:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)

# qualitative: what does the kNN lens SAY at a mid layer for a few queries?
print("== exemplar decodes at L6", flush=True)
bank = F.normalize(bank_states[6].float(), dim=-1).to(DEV)
flat_bank_ids = token_chunks("train", N_BANK_CHUNKS)
flat_q_ids = token_chunks("test", N_QUERY_CHUNKS)
qual = []
for qi in [500, 2000, 4000, 6000]:
    q = F.normalize(q_states[6][qi:qi + 1].float().to(DEV), dim=-1)
    idx = (q @ bank.T).topk(4, -1).indices[0].cpu()
    chunk_pos = [(i.item() // (CTX - 1), i.item() % (CTX - 1)) for i in idx]
    q_chunk, q_pos = qi // (CTX - 1), qi % (CTX - 1)
    entry = {
        "query_ctx": tok.decode(
            flat_q_ids[q_chunk, max(0, q_pos - 12):q_pos + 1].tolist()),
        "true_next": tok.decode([q_labels[qi].item()]),
        "neighbors": [
            {"ctx": tok.decode(
                flat_bank_ids[c, max(0, p - 12):p + 1].tolist()),
             "next": tok.decode([bank_labels[c * (CTX - 1) + p].item()])}
            for c, p in chunk_pos]}
    qual.append(entry)
    print(f"  Q: ...{entry['query_ctx']!r} -> {entry['true_next']!r}", flush=True)
    for nb_ in entry["neighbors"]:
        print(f"     N: ...{nb_['ctx'][-60:]!r} -> {nb_['next']!r}", flush=True)
results["qual_L6"] = qual

with open(f"{OUT}/exp_b_knn.json", "w") as f:
    json.dump(results, f, indent=1)
print("DONE_B", flush=True)
