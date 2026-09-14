import json, numpy as np
R=json.load(open("trainfree/results/exp_r1_readiness_gpt2.json"))
Z=np.load("trainfree/results/exp_r1_readiness_gpt2.npz")
NL=12
print("=== per-head readiness levels (first l where TV<=0.2 and stays): rows=layer m, entries=ready_q/ready_k per head ===")
for m in range(1,NL):
    row=[]
    for h in range(12):
        d=R["heads"][f"L{m}H{h}"]; row.append(f"{d['ready_q']}/{d['ready_k']}")
    print(f"m={m:2d} "+" ".join(f"{x:>5s}" for x in row))
print()
for name in ["L4H11","L5H1","L5H5","L6H9","L7H10","L7H2","L3H0","L2H2","L1H0","L9H9","L9H6","L10H7","L11H10"]:
    d=R["heads"][name]
    print(name, "q:"," ".join(f"{x:.2f}" for x in d["tv_q"]), "| k:"," ".join(f"{x:.2f}" for x in d["tv_k"]))
print()
print("=== MLP readers vs nulls: quantiles [p5 p25 p50 p75 p95] of r at (l,m); frac readers below null p5 / above null p95 ===")
q=lambda a:" ".join(f"{np.quantile(a,p):.2f}" for p in [0.05,0.25,0.5,0.75,0.95])
for m in [2,4,6,8,10,11]:
    for l in [max(0,m-4), m-1, m]:
        r=Z[f"r_mlp_{m}"][l]; imp=Z[f"imp_{m}"]; live=imp>np.quantile(imp,0.05); r=r[live]
        ni=Z[f"r_iso_{m}"][l]; ns=Z[f"r_span_{m}"][l]
        lo=(r<np.quantile(ni,0.05)).mean(); hi=(r>np.quantile(ni,0.95)).mean()
        w=imp[live]/imp[live].sum(); rw=(r*w).sum()
        print(f"l={l:2d} m={m:2d} readers [{q(r)}] wmean {rw:.2f} | iso [{q(ni)}] | span [{q(ns)}] | below-p5 {lo:.2f} above-p95 {hi:.2f}")
print()
print("=== heaviest writers (top importance decile) vs lower half: median r at l=m-1 ===")
for m in range(1,NL):
    r=Z[f"r_mlp_{m}"][m-1]; imp=Z[f"imp_{m}"]; top=imp>=np.quantile(imp,0.9); bot=(imp<np.quantile(imp,0.5))&(imp>np.quantile(imp,0.05))
    print(f"m={m:2d} top-decile {np.median(r[top]):.3f}  lower-half {np.median(r[bot]):.3f}  null_iso {np.median(Z[f'r_iso_{m}'][m-1]):.3f}")
print()
print("=== attention unit-level r per head (Q / K / V) at l=m-1, head-averaged per layer; and at l=1 ===")
U=R["attn_unit_r"]
for m in range(1,NL):
    uq=np.array(U["q"][m]); uk=np.array(U["k"][m]); uv=np.array(U["v"][m])
    print(f"m={m:2d} l=m-1: q {uq[m-1].mean():.2f} k {uk[m-1].mean():.2f} v {uv[m-1].mean():.2f} | l=1: q {uq[1].mean() if m>1 else float('nan'):.2f} k {uk[1].mean() if m>1 else float('nan'):.2f} v {uv[1].mean() if m>1 else float('nan'):.2f}")
