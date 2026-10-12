import pandas as pd, numpy as np, sys
b = pd.read_csv(sys.argv[1]+"/batches.tsv", sep="\t")
b["gain"] = b.hand_label - b.best_label
b["rel"] = b.gain / b.hand_label
for rung, g in b.groupby("rung"):
    tot = g.hand_label.sum()
    print(f"== {rung}: batches {len(g)}, label gain {g.gain.sum()/tot*100:.2f}% of hand seconds")
    for lo, hi in [(0,1e-9),(1e-9,.03),(.03,.10),(.10,.30),(.30,9)]:
        s = g[(g.rel>lo)&(g.rel<=hi)] if lo>0 else g[g.rel<=1e-9]
        print(f"  rel ({lo:.2f},{hi:.2f}]: {len(s):6d} batches ({len(s)/len(g)*100:5.1f}%), share of total gain {s.gain.sum()/max(g.gain.sum(),1e-9)*100:5.1f}%, median n {s.n.median():.0f}, median plans {s.plans.median():.0f}")
    big = g[g.rel>.10]; ok = g[g.rel<=1e-9]
    cols = ["n","plans","h_nodes","b_nodes","h_coloc","b_coloc","h_stack","b_stack","h_maxstack","b_maxstack","new_nodes_vs_hand","best_Srank"]
    print("  medians/means  >10% | no-gain")
    for c in cols:
        print(f"   {c:18s} {big[c].mean():7.2f} | {ok[c].mean():7.2f}")
    big = big.assign(dcoloc=big.b_coloc-big.h_coloc, dstack=big.b_stack-big.h_stack, dnodes=big.b_nodes-big.h_nodes)
    print("  >10%: oracle co-locates MORE pairs", (big.dcoloc>0).mean().round(3), "fewer", (big.dcoloc<0).mean().round(3), "same", (big.dcoloc==0).mean().round(3))
    print("        oracle stacks MORE on a platform", (big.dstack>0).mean().round(3), "fewer", (big.dstack<0).mean().round(3))
    print("        oracle uses a node hand did not", (big.new_nodes_vs_hand>0).mean().round(3))
    print("        S gap of oracle plan vs hand (rel) median", ((big.best_S-big.hand_S)/big.hand_S).median().round(3), "p90", ((big.best_S-big.hand_S)/big.hand_S).quantile(.9).round(3))
    # time in run
    q = pd.qcut(g.t, 4, labels=False)
    print("  gain share by run quarter:", [round(g[q==i].gain.sum()/g.gain.sum()*100,1) for i in range(4)], " rel>10% rate by quarter:", [round((g[q==i].rel>.1).mean()*100,1) for i in range(4)])
    print("  by cell rel>10% rate:", g.groupby("cell").rel.apply(lambda x: round((x>.1).mean()*100,1)).to_dict())
