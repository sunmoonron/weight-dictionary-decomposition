import json, numpy as np
from scipy.stats import spearmanr
R2=json.load(open("trainfree/results/exp_r2_depends.json"))
R1=json.load(open("trainfree/results/exp_r1_readiness_gpt2.json"))
NL=12
print("clean CE", round(R2["clean_ce"],4), "n_tok", R2["n_tokens"], "| cells: mlp", len(R2["mlp_edge"]), "attn", len(R2["attn_edge"]))
print("\n=== MLP-edge dCE: block m's MLP reads hidden_states[l] (rows m, cols l=0..m) ===")
for m in range(NL):
    print(f"m={m:2d} "+" ".join(f"{R2['mlp_edge'][f'{l}|{m}']['dce']:6.3f}" for l in range(m+1) if f'{l}|{m}' in R2['mlp_edge']))
print("\n=== attn-edge dCE: block m's attention reads hidden_states[l] (rows m, cols l=0..m-1) ===")
for m in range(1,NL):
    print(f"m={m:2d} "+" ".join(f"{R2['attn_edge'][f'{l}|{m}']['dce']:6.3f}" for l in range(m) if f'{l}|{m}' in R2['attn_edge']))
print("\n=== diagonal: MLP m without attention m (dCE) vs R1 frac needs-attn ===")
for m in range(NL):
    k=f"{m}|{m}"
    if k in R2["mlp_edge"]:
        print(f"m={m:2d} dCE {R2['mlp_edge'][k]['dce']:.4f}  KL {R2['mlp_edge'][k]['kl']:.4f}  R1 needs_attn(0.9) {R1['mlp_depth']['0.9'][str(m)]['frac_needs_attn']:.3f}  r_median(l=m) {R1['mlp']['r_median'][m][m]:.3f}")
print("\n=== one-layer-back: MLP m reads hidden_states[m-1] (dCE) and attention m reads hidden_states[m-1] ===")
for m in range(1,NL):
    a=R2["mlp_edge"].get(f"{m-1}|{m}",{}).get("dce",float('nan')); b=R2["attn_edge"].get(f"{m-1}|{m}",{}).get("dce",float('nan'))
    print(f"m={m:2d} mlp {a:.4f}  attn {b:.4f}")
# E4
xs,ys,dist=[],[],[]
for m in range(NL):
    for l in range(m+1):
        k=f"{l}|{m}"
        if k in R2["mlp_edge"]:
            xs.append(R1["mlp"]["r_wmean"][m][l]); ys.append(R2["mlp_edge"][k]["dce"]); dist.append(m-l)
xs,ys,dist=map(np.array,(xs,ys,dist))
print(f"\nE4 Spearman(readiness r_wmean, dCE) over {len(xs)} MLP cells: {spearmanr(xs,ys).correlation:+.3f}")
for d in [0,1,2,3,4]:
    sel=dist==d
    if sel.sum()>=4: print(f"  at fixed distance m-l={d} (n={sel.sum()}): Spearman {spearmanr(xs[sel],ys[sel]).correlation:+.3f}; dCE range {ys[sel].min():.3f}..{ys[sel].max():.3f}")
# null readiness vs dCE: does the random-direction null predict as well?
xn=[]
for m in range(NL):
    for l in range(m+1):
        if f"{l}|{m}" in R2["mlp_edge"]: xn.append(R1["mlp"]["null_iso_r_median"][m][l])
print(f"  same with null_iso median r: Spearman {spearmanr(np.array(xn),ys).correlation:+.3f}")
print("\n=== neuron groups: early-read only the ready (r>=0.9) / not-ready (r<0.6) / random subset of block m's neurons from level l ===")
for k,g in R2.get("groups",{}).items():
    if g.get("skipped"): print(k, "skipped", g); continue
    print(f"(l,m)=({k}) n={g['n']} [ready total {g['n_ready_total']}, notready total {g['n_notready_total']}]  dCE ready {g['ready']['dce']:+.4f} (imp {g['imp_ready']:.0f}) | notready {g['notready']['dce']:+.4f} (imp {g['imp_notready']:.0f}) | random {g['random']['dce']:+.4f} (imp {g['imp_random']:.0f})   per-imp ready {g['ready']['dce']/max(g['imp_ready'],1e-9)*100:.4f} notready {g['notready']['dce']/max(g['imp_notready'],1e-9)*100:.4f}")
print("\nE4_spearman in file:", R2.get("E4_spearman_readiness_vs_dce"))
