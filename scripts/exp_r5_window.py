"""R5: continuation lens with SMALL LOCAL WINDOWS -- is the trained lens's remaining edge (levels 1-3 on the Pile) local context?
Variants added: sinkprev = attend to {0, t-1, self}; win4 = attend to {0, t-3..t}.
Derived from R4: the CONTINUATION LENS across architectures and on the tuned lens's own corpus.

R3 found, on GPT-2 small / WikiText-2, that running the model's remaining MLP
sublayers on an intermediate state (attention removed) beats the trained
Tuned Lens on KL at 11/12 levels, and that restoring attention's context-free
default (attend only to {position 0, self}) beats it at levels 3..11. Two
things could make that a fluke: the tuned lens was trained on the Pile, and
GPT-2 small is one architecture. This script runs the same bake-off on
  gpt2, gpt2-large            (tied unembedding, GELU, LayerNorm, learned pos)
  EleutherAI/pythia-410m-deduped (parallel attn+MLP, rotary, untied, tuned lens
                              available)
  HuggingFaceTB/SmolLM2-135M  (SwiGLU, RMSNorm, GQA, rotary; no lens)
on WikiText-2 test and on NeelNanda/pile-10k.
Variants: logit, mlp (MLP-only continuation), sink (attention masked to
{0, self} -- generic implementation via a 4D additive mask through the model's
own layer forward), tuned (where a pretrained lens exists), actual (full causal
mask through the same machinery: must reproduce the model's logits; the
sanity gate for the generic layer-call code).
Metrics: KL(final || lens), CE, top-1 agreement; positions 1..T-2; 32 chunks
of 512 tokens. Writes results/exp_r4_generality.json (merged across runs).
Usage: R_MODEL=<hf id> R_DATA=wikitext|pile R_DEV=cuda:0 python exp_r4_generality.py
"""
import json, os, sys, time, math
_HF = "/data/mechinterp/hf"          # the original machine's cache; ignored elsewhere
if os.path.isdir(_HF):
    os.environ.setdefault("HF_HOME", _HF)
RESULTS_DIR = os.environ.get("WDD_RESULTS", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(0)
torch.set_grad_enabled(False)
OUT = RESULTS_DIR
DEV = os.environ.get("R_DEV", "cuda:0")
MODEL = os.environ.get("R_MODEL", "gpt2")
DATA = os.environ.get("R_DATA", "wikitext")
CTX, N_EVAL, BS = 512, int(os.environ.get("R_NEVAL", "32")), 4
t0 = time.time()

model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
arch = cfg.model_type
print(f"{MODEL} ({arch}) on {DATA}, dev {DEV}", flush=True)

# ---- architecture adapters ------------------------------------------------------------
if arch == "gpt2":
    layers = model.transformer.h
    final_norm, head = model.transformer.ln_f, model.lm_head
    def mlp_only(layer, x): return x + layer.mlp(layer.ln_2(x))
    def rotary(x, pos): return None
    def call_layer(layer, x, mask, pe):
        out = layer(x, attention_mask=mask)
        return out[0] if isinstance(out, tuple) else out
elif arch == "gpt_neox":
    layers = model.gpt_neox.layers
    final_norm = model.gpt_neox.final_layer_norm
    head = model.lm_head if hasattr(model, "lm_head") else model.embed_out
    def mlp_only(layer, x): return x + layer.mlp(layer.post_attention_layernorm(x))
    def rotary(x, pos): return model.gpt_neox.rotary_emb(x, pos)
    def call_layer(layer, x, mask, pe):
        out = layer(x, attention_mask=mask, position_embeddings=pe)
        return out[0] if isinstance(out, tuple) else out
elif arch in ("llama", "smollm"):
    layers = model.model.layers
    final_norm, head = model.model.norm, model.lm_head
    def mlp_only(layer, x): return x + layer.mlp(layer.post_attention_layernorm(x))
    def rotary(x, pos): return model.model.rotary_emb(x, pos)
    def call_layer(layer, x, mask, pe):
        out = layer(x, attention_mask=mask, position_embeddings=pe)
        return out[0] if isinstance(out, tuple) else out
else:
    raise SystemExit(f"unsupported arch {arch}")
NL = len(layers)
D = cfg.hidden_size if hasattr(cfg, "hidden_size") else cfg.n_embd

# ---- data -----------------------------------------------------------------------------
def chunks_from_text(text, n_chunks):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0][: n_chunks * CTX]
    assert len(ids) == n_chunks * CTX, f"only {len(ids)} tokens"
    return ids.view(n_chunks, CTX)


if DATA == "wikitext":
    text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
else:
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    text = "\n\n".join(ds[i]["text"] for i in range(400))
eval_ids = chunks_from_text(text, N_EVAL)
print(f"eval tokens {eval_ids.numel()}", flush=True)

# ---- tuned lens ---------------------------------------------------------------------------
LENS_PATHS = {"gpt2": "lens/gpt2", "gpt2-large": "lens/gpt2-large",
              "EleutherAI/pythia-410m-deduped": "lens/EleutherAI/pythia-410m-deduped"}
tuned = None
if MODEL in LENS_PATHS:
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("AlignmentResearch/tuned-lens", LENS_PATHS[MODEL] + "/params.pt", repo_type="space")
        sd = torch.load(p, map_location="cpu", weights_only=True)
        tuned = {}
        for k, v in sd.items():
            parts = k.split(".")
            idx = [int(x) for x in parts if x.isdigit()]
            if idx:
                tuned.setdefault(idx[0], {})[parts[-1]] = v.to(DEV).float()
        assert all("weight" in tuned[i] for i in range(NL)), sorted(tuned)
        print(f"tuned lens: {len(tuned)} translators", flush=True)
    except Exception as e:
        print("TUNED LENS UNAVAILABLE:", repr(e), flush=True)
        tuned = None

# ---- capture layer inputs (level l = input to layer l) --------------------------------
stash = {}
rec = {"on": False}      # record layer inputs ONLY during the clean forward pass;
                         # continuation passes also fire these hooks and must not overwrite


def mk_pre(l):
    def f(mod, args, kwargs):
        if rec["on"]:
            stash[l] = args[0] if args else kwargs["hidden_states"]
    return f


for l, layer in enumerate(layers):
    layer.register_forward_pre_hook(mk_pre(l), with_kwargs=True)

NEG = torch.finfo(torch.float32).min
T = CTX
ar = torch.arange(T, device=DEV)
mask_causal = torch.where(ar[None, :] <= ar[:, None], 0.0, NEG).view(1, 1, T, T)
mask_sink = torch.where((ar[None, :] == ar[:, None]) | (ar[None, :] == 0), 0.0, NEG).view(1, 1, T, T)
mask_sinkprev = torch.where((ar[None, :] == ar[:, None]) | (ar[None, :] == ar[:, None] - 1) | (ar[None, :] == 0), 0.0, NEG).view(1, 1, T, T)
mask_win4 = torch.where(((ar[None, :] <= ar[:, None]) & (ar[None, :] >= ar[:, None] - 3)) | (ar[None, :] == 0), 0.0, NEG).view(1, 1, T, T)
pos_ids = ar[None, :]


def continue_from(l, mode):
    x = stash[l]
    pe = rotary(x, pos_ids)
    for m in range(l, NL):
        layer = layers[m]
        if mode == "mlp":
            x = mlp_only(layer, x)
        elif mode == "sink":
            x = call_layer(layer, x, mask_sink.expand(x.shape[0], 1, T, T), pe)
        elif mode == "sinkprev":
            x = call_layer(layer, x, mask_sinkprev.expand(x.shape[0], 1, T, T), pe)
        elif mode == "win4":
            x = call_layer(layer, x, mask_win4.expand(x.shape[0], 1, T, T), pe)
        elif mode == "actual":
            x = call_layer(layer, x, mask_causal.expand(x.shape[0], 1, T, T), pe)
        else:
            raise ValueError(mode)
    return x


def decode(l, var):
    h = stash[l]
    if var == "logit":
        x = h
    elif var == "tuned":
        x = h + h @ tuned[l]["weight"].T + tuned[l].get("bias", torch.zeros(D, device=DEV))
    else:
        x = continue_from(l, var)
    return head(final_norm(x))


VARIANTS = ["logit", "mlp", "sink", "sinkprev", "win4", "actual"] + (["tuned"] if tuned else [])
acc = {}


def add(key, ce, kl, agree, n):
    s = acc.setdefault(key, [0.0, 0.0, 0.0, 0])
    s[0] += ce; s[1] += kl; s[2] += agree; s[3] += n


def metrics(logits, final_lp, targets):
    lp = torch.log_softmax(logits.float(), -1)
    ce = -lp.gather(-1, targets[..., None]).squeeze(-1)
    kl = (final_lp.exp() * (final_lp - lp)).sum(-1)
    agree = (lp.argmax(-1) == final_lp.argmax(-1)).float()
    return float(ce.sum()), float(kl.sum()), float(agree.sum()), ce.numel()


sanity = {}
for bi in range(0, N_EVAL, BS):
    ids = eval_ids[bi:bi + BS].to(DEV)
    stash.clear()
    rec["on"] = True
    out = model(ids)
    rec["on"] = False
    snap = {l: stash[l].clone() for l in stash}
    final_lp = torch.log_softmax(out.logits.float(), -1)
    if bi == 0:
        # generic layer-call machinery must reproduce the model from several levels
        for l in [0, NL // 2, NL - 1]:
            d = float((decode(l, "actual") - out.logits).abs().max())
            sanity[f"actual_recon_L{l}"] = d
        print("sanity (max |logit diff| of full-mask continuation):", sanity, flush=True)
    for l in range(NL):
        for var in VARIANTS:
            if var == "actual" and bi > 0:
                continue
            logits = decode(l, var)
            add((l, var), *metrics(logits[:, 1:-1], final_lp[:, 1:-1], ids[:, 2:]))
    add(("final", "model"), *metrics(out.logits[:, 1:-1], final_lp[:, 1:-1], ids[:, 2:]))
    if bi == 0:
        drift = max(float((stash[l] - snap[l]).abs().max()) for l in snap)
        sanity["stash_immutable_max_drift"] = drift
        assert drift == 0.0, drift
        assert len(stash) == NL, len(stash)
    del out
    print(f"batch {bi // BS + 1}/{N_EVAL // BS} {time.time() - t0:.0f}s", flush=True)

M = {f"{k[0]}|{k[1]}": {"ce": v[0] / v[3], "kl": v[1] / v[3], "top1_agree": v[2] / v[3], "n": v[3]} for k, v in acc.items()}
ok = max(sanity.values()) < 5e-2
run = {"model": MODEL, "arch": arch, "data": DATA, "n_layers": NL, "variants": VARIANTS, "sanity": sanity,
       "machinery_ok": ok, "metrics": M, "elapsed_s": time.time() - t0}
tag = MODEL.split("/")[-1]
with open(f"{OUT}/exp_r5_{tag}_{DATA}.json", "w") as f:
    json.dump(run, f, indent=1)

print(f"\n{MODEL} / {DATA}: final CE {M['final|model']['ce']:.3f}; machinery_ok={ok}")
print("KL(final||lens): lvl " + " ".join(f"{v:>7s}" for v in VARIANTS))
for l in range(NL):
    print(f"{l:3d} " + " ".join(f"{M[f'{l}|{v}']['kl']:7.3f}" for v in VARIANTS))
print("CE: lvl " + " ".join(f"{v:>7s}" for v in VARIANTS))
for l in range(NL):
    print(f"{l:3d} " + " ".join(f"{M[f'{l}|{v}']['ce']:7.3f}" for v in VARIANTS))
for v in ["sinkprev", "win4"]:
    if tuned:
        print(f"{v} beats tuned at {sum(M[f'{l}|{v}']['kl'] < M[f'{l}|tuned']['kl'] for l in range(NL))}/{NL}")
w_mlp = sum(M[f"{l}|mlp"]["kl"] < M[f"{l}|logit"]["kl"] for l in range(NL))
w_sink = sum(M[f"{l}|sink"]["kl"] < M[f"{l}|logit"]["kl"] for l in range(NL))
print(f"mlp beats logit at {w_mlp}/{NL}; sink beats logit at {w_sink}/{NL}")
if tuned:
    print(f"mlp beats tuned at {sum(M[f'{l}|mlp']['kl'] < M[f'{l}|tuned']['kl'] for l in range(NL))}/{NL}; "
          f"sink beats tuned at {sum(M[f'{l}|sink']['kl'] < M[f'{l}|tuned']['kl'] for l in range(NL))}/{NL}")
print(f"done {run['elapsed_s']:.0f}s")
