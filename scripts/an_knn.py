import json
b=json.load(open("trainfree/results/exp_b_knn.json")); b2=None
try: b2=json.load(open("trainfree/results/exp_b2_backoff.json"))
except Exception as e: print("no b2", e)
print("knn final_ce", b["final_ce"])
lv=b["levels"]["6"]; print("level keys:", list(lv.keys()) if isinstance(lv,dict) else lv)
for l in range(12):
    d=b["levels"][str(l)]
    print(l, {k:(round(v,3) if isinstance(v,(int,float)) else str(v)[:60]) for k,v in d.items()} if isinstance(d,dict) else d)
if b2:
    print("b2 keys", list(b2.keys())[:20])
    for k,v in b2.items():
        print(k, str(v)[:400])
r3=json.load(open("trainfree/results/exp_r3_continuation.json"))
print("\nR3 sub16 (kNN protocol, positions 0..T-2 of 16 chunks): level: logit / mlp / sink / tuned  CE")
for l in range(12):
    g=lambda v: r3["sub16_knnprotocol"][f"{l}|{v}"]["ce"]
    print(l, f"{g('logit'):.3f} {g('mlp'):.3f} {g('sink'):.3f} {g('tuned'):.3f}")
print("final", r3["sub16_knnprotocol"]["final|model"]["ce"], "bigram sub16", r3["bigram_ce_sub16"])
