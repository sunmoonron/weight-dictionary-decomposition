"""R6: give the trained lens its best shot -- train a Tuned Lens ON the evaluation
distribution and re-run the continuation bake-off.

Objection to R3/R4: the released tuned lenses were trained on the Pile, so on
WikiText-2 they are off-distribution, and even on the Pile the released
artifact is one training run of unknown budget. This script trains, per
level, the same affine translator h -> h + A h + b (identity-residual
parametrisation, as in the tuned-lens repo) to minimise KL(final || lens) on
the TRAIN split of the evaluation corpus (WikiText-2 train, 2.39M tokens; or
Pile-10k documents 400..1399, 1.85M tokens, disjoint from the eval documents
0..399), then evaluates logit / mlp / sink / released tuned / self-tuned on
exactly R4's eval chunks and positions.

If the self-trained lens beats the continuation, the R3/R4 headline was
distribution shift. If the continuation still wins, the trained lens's loss is
intrinsic to the affine form.
Usage: R_MODEL=gpt2 R_DATA=wikitext|pile R_DEV=cuda:0 python exp_r6_selftuned.py
Writes results/exp_r6_{tag}_{data}.json and the lens params .pt
"""
import json, os, sys, time, math
_HF = "/data/mechinterp/hf"          # the original machine's cache; ignored elsewhere
if os.path.isdir(_HF):
    os.environ.setdefault("HF_HOME", _HF)
RESULTS_DIR = os.environ.get("WDD_RESULTS", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(0)
OUT = RESULTS_DIR
DEV = os.environ.get("R_DEV", "cuda:0")
MODEL = os.environ.get("R_MODEL", "gpt2")
DATA = os.environ.get("R_DATA", "wikitext")
CTX, N_EVAL, BS = 512, 32, 4
EPOCHS = int(os.environ.get("R_EPOCHS", "1"))
LR = 1e-3
t0 = time.time()

model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
tok = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
arch = cfg.model_type
print(f"{MODEL} ({arch}) on {DATA}, dev {DEV}", flush=True)

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
else:
    raise SystemExit(f"unsupported arch {arch}")
NL = len(layers)
D = cfg.hidden_size if hasattr(cfg, "hidden_size") else cfg.n_embd


def chunks_from_text(text, n_chunks=None):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n = len(ids) // CTX if n_chunks is None else n_chunks
    ids = ids[: n * CTX]
    assert len(ids) == n * CTX, f"only {len(ids)} tokens"
    return ids.view(n, CTX)


if DATA == "wikitext":
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    eval_ids = chunks_from_text("\n\n".join(t for t in ds["test"]["text"] if t.strip()), N_EVAL)
    train_ids = chunks_from_text("\n\n".join(t for t in ds["train"]["text"] if t.strip()))
else:
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    eval_ids = chunks_from_text("\n\n".join(ds[i]["text"] for i in range(400)), N_EVAL)
    train_ids = chunks_from_text("\n\n".join(ds[i]["text"] for i in range(400, 1400)))
g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(train_ids), generator=g)
train_ids = train_ids[perm]
hold_ids, train_ids = train_ids[:16], train_ids[16:]
print(f"train chunks {len(train_ids)} ({train_ids.numel()} tokens), holdout 16, eval {N_EVAL}", flush=True)

# ---- released lens -------------------------------------------------------------------
LENS_PATHS = {"gpt2": "lens/gpt2", "gpt2-large": "lens/gpt2-large",
              "EleutherAI/pythia-410m-deduped": "lens/EleutherAI/pythia-410m-deduped"}
released = None
if MODEL in LENS_PATHS:
    from huggingface_hub import hf_hub_download
    sd = torch.load(hf_hub_download("AlignmentResearch/tuned-lens", LENS_PATHS[MODEL] + "/params.pt", repo_type="space"),
                    map_location="cpu", weights_only=True)
    released = {}
    for k, v in sd.items():
        parts = k.split("."); idx = [int(x) for x in parts if x.isdigit()]
        if idx:
            released.setdefault(idx[0], {})[parts[-1]] = v.to(DEV).float()

# ---- self-trained lens ------------------------------------------------------------------
lens = nn.ModuleList([nn.Linear(D, D) for _ in range(NL)]).to(DEV)
for lin in lens:
    nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
opt = torch.optim.AdamW(lens.parameters(), lr=LR, weight_decay=0.0)
steps_per_epoch = len(train_ids) // BS
total_steps = steps_per_epoch * EPOCHS
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)) * 0.99 + 0.01)


def lens_logits(h, l, which):
    if which == "self":
        return head(final_norm(h + lens[l](h)))
    w, b = released[l]["weight"], released[l].get("bias", torch.zeros(D, device=DEV))
    return head(final_norm(h + h @ w.T + b))


def holdout_kl():
    kls = torch.zeros(NL)
    with torch.no_grad():
        for bi in range(0, 16, BS):
            ids = hold_ids[bi:bi + BS].to(DEV)
            out = model(ids, output_hidden_states=True)
            flp = torch.log_softmax(out.logits[:, 1:].float(), -1)
            for l in range(NL):
                lp = torch.log_softmax(lens_logits(out.hidden_states[l][:, 1:], l, "self").float(), -1)
                kls[l] += (flp.exp() * (flp - lp)).sum(-1).mean().item()
    return (kls / (16 // BS)).tolist()


print("holdout KL at init (== logit lens):", " ".join(f"{v:.2f}" for v in holdout_kl()), flush=True)
step = 0
curve = []
for ep in range(EPOCHS):
    order = torch.randperm(len(train_ids), generator=g)
    for bi in range(0, steps_per_epoch * BS, BS):
        ids = train_ids[order[bi:bi + BS]].to(DEV)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
            flp = torch.log_softmax(out.logits[:, 1:].float(), -1)
            pf = flp.exp()
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for l in range(NL):
            h = out.hidden_states[l][:, 1:].detach()
            lp = torch.log_softmax(lens_logits(h, l, "self").float(), -1)
            loss = (pf * (flp - lp)).sum(-1).mean()
            loss.backward()
            tot += loss.item()
        torch.nn.utils.clip_grad_norm_(lens.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        if step % 100 == 0 or step == total_steps:
            hk = holdout_kl() if (step % 300 == 0 or step == total_steps) else None
            curve.append({"step": step, "train_kl_mean": tot / NL, "holdout_kl": hk})
            print(f"step {step}/{total_steps} train KL/level {tot / NL:.3f} lr {sched.get_last_lr()[0]:.2e} "
                  + (f"holdout {' '.join(f'{v:.2f}' for v in hk)}" if hk else "") + f"  {time.time() - t0:.0f}s", flush=True)
        del out, flp, pf
tag = MODEL.split("/")[-1]
torch.save(lens.state_dict(), f"{OUT}/exp_r6_lens_{tag}_{DATA}.pt")
lens.eval()
for p in lens.parameters():
    p.requires_grad_(False)

# ---- evaluation, R4 protocol ----------------------------------------------------------------
torch.set_grad_enabled(False)
stash, rec = {}, {"on": False}


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
mask_sink = torch.where((ar[None, :] == ar[:, None]) | (ar[None, :] == 0), 0.0, NEG).view(1, 1, T, T)
mask_causal = torch.where(ar[None, :] <= ar[:, None], 0.0, NEG).view(1, 1, T, T)
pos_ids = ar[None, :]


def continue_from(l, mode):
    x = stash[l]; pe = rotary(x, pos_ids)
    for m in range(l, NL):
        layer = layers[m]
        if mode == "mlp":
            x = mlp_only(layer, x)
        else:
            x = call_layer(layer, x, (mask_sink if mode == "sink" else mask_causal).expand(x.shape[0], 1, T, T), pe)
    return x


def decode(l, var):
    h = stash[l]
    if var == "logit":
        return head(final_norm(h))
    if var == "released":
        return lens_logits(h, l, "released")
    if var == "self":
        return lens_logits(h, l, "self")
    return head(final_norm(continue_from(l, var)))


VARIANTS = ["logit", "mlp", "sink", "self", "actual"] + (["released"] if released else [])
acc = {}


def add(key, ce, kl, agree, n):
    s = acc.setdefault(key, [0.0, 0.0, 0.0, 0]); s[0] += ce; s[1] += kl; s[2] += agree; s[3] += n


def metrics(logits, final_lp, targets):
    lp = torch.log_softmax(logits.float(), -1)
    ce = -lp.gather(-1, targets[..., None]).squeeze(-1)
    kl = (final_lp.exp() * (final_lp - lp)).sum(-1)
    agree = (lp.argmax(-1) == final_lp.argmax(-1)).float()
    return float(ce.sum()), float(kl.sum()), float(agree.sum()), ce.numel()


sanity = {}
for bi in range(0, N_EVAL, BS):
    ids = eval_ids[bi:bi + BS].to(DEV)
    stash.clear(); rec["on"] = True
    out = model(ids); rec["on"] = False
    final_lp = torch.log_softmax(out.logits.float(), -1)
    if bi == 0:
        for l in [0, NL // 2, NL - 1]:
            sanity[f"actual_recon_L{l}"] = float((decode(l, "actual") - out.logits).abs().max())
    for l in range(NL):
        for var in VARIANTS:
            if var == "actual" and bi > 0:
                continue
            add((l, var), *metrics(decode(l, var)[:, 1:-1], final_lp[:, 1:-1], ids[:, 2:]))
    add(("final", "model"), *metrics(out.logits[:, 1:-1], final_lp[:, 1:-1], ids[:, 2:]))
M = {f"{k[0]}|{k[1]}": {"ce": v[0] / v[3], "kl": v[1] / v[3], "top1_agree": v[2] / v[3], "n": v[3]} for k, v in acc.items()}
run = {"model": MODEL, "data": DATA, "n_layers": NL, "train_tokens": int(train_ids.numel()), "epochs": EPOCHS, "lr": LR,
       "steps": total_steps, "curve": curve, "variants": VARIANTS, "sanity": sanity, "metrics": M, "elapsed_s": time.time() - t0}
with open(f"{OUT}/exp_r6_{tag}_{DATA}.json", "w") as f:
    json.dump(run, f, indent=1)
print(f"\n{MODEL} / {DATA}: final CE {M['final|model']['ce']:.3f}; sanity {sanity}")
print("KL(final||lens): lvl " + " ".join(f"{v:>8s}" for v in VARIANTS))
for l in range(NL):
    print(f"{l:3d} " + " ".join(f"{M[f'{l}|{v}']['kl']:8.3f}" for v in VARIANTS))
print("CE: lvl " + " ".join(f"{v:>8s}" for v in VARIANTS))
for l in range(NL):
    print(f"{l:3d} " + " ".join(f"{M[f'{l}|{v}']['ce']:8.3f}" for v in VARIANTS))
w = lambda a, b: sum(M[f"{l}|{a}"]["kl"] < M[f"{l}|{b}"]["kl"] for l in range(NL))
print(f"self-tuned beats released at {w('self','released') if released else '-'}/{NL}; "
      f"mlp beats self-tuned at {w('mlp','self')}/{NL}; sink beats self-tuned at {w('sink','self')}/{NL}")
print(f"done {run['elapsed_s']:.0f}s")
