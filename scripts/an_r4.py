import json, glob, os
import os
_HF = "/data/mechinterp/hf"          # the original machine's cache; ignored elsewhere
if os.path.isdir(_HF):
    os.environ.setdefault("HF_HOME", _HF)
RESULTS_DIR = os.environ.get("WDD_RESULTS", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)
runs = {}
for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "exp_r4_*.json"))):
    r = json.load(open(p)); runs[(r["model"], r["data"])] = r
order = [("gpt2","wikitext"),("gpt2","pile"),("gpt2-large","wikitext"),("gpt2-large","pile"),
         ("EleutherAI/pythia-410m-deduped","wikitext"),("EleutherAI/pythia-410m-deduped","pile"),
         ("HuggingFaceTB/SmolLM2-135M","wikitext"),("HuggingFaceTB/SmolLM2-135M","pile")]
js = {}
print("model | data | NL | final CE | mlp>logit | sink>logit | mlp>tuned | sink>tuned | sink>tuned from level | KL L0 (logit/mlp/sink/tuned) | KL mid | KL last")
for key in order:
    if key not in runs: print(key, "MISSING"); continue
    r = runs[key]; M = r["metrics"]; NL = r["n_layers"]; V = [v for v in r["variants"] if v != "actual"]
    kl = {v: [M[f"{l}|{v}"]["kl"] for l in range(NL)] for v in V}
    ce = {v: [M[f"{l}|{v}"]["ce"] for l in range(NL)] for v in V}
    tuned = "tuned" in V
    w = lambda a,b: sum(x<y for x,y in zip(kl[a],kl[b]))
    frm = "-"
    if tuned:
        ok = [kl["sink"][l] < kl["tuned"][l] for l in range(NL)]
        # first level from which sink beats tuned at every later level
        frm = next((l for l in range(NL) if all(ok[l:])), NL)
    mid = NL//2
    f = lambda l: "/".join(f"{kl[v][l]:.2f}" for v in ["logit","mlp","sink"] + (["tuned"] if tuned else []))
    print(f"{key[0].split('/')[-1]} | {key[1]} | {NL} | {M['final|model']['ce']:.3f} | {w('mlp','logit')} | {w('sink','logit')} | {w('mlp','tuned') if tuned else '-'} | {w('sink','tuned') if tuned else '-'} | {frm} | {f(0)} | {f(mid)} | {f(NL-1)}")
    tag = key[0].split("/")[-1] + "|" + key[1]
    js[tag] = {"NL": NL, "final_ce": round(M["final|model"]["ce"],3), "sanity": r["sanity"],
               "kl": {v: [round(x,3) for x in kl[v]] for v in V}, "ce": {v: [round(x,3) for x in ce[v]] for v in V}}
print("\nJS:\nconst GEN=" + json.dumps(js, separators=(",",":")) + ";")
