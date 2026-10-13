import pandas as pd, numpy as np, sys
b = pd.read_csv(sys.argv[1]+"/batches.tsv", sep="\t")
b["gain"]=b.hand_label-b.best_label; b["rel"]=b.gain/b.hand_label
for rung,g in b.groupby("rung"):
    G=g.gain.sum(); tmax=g.groupby("cell").t.transform("max"); g=g.assign(tf=g.t/tmax)
    print(f"== {rung}  t max per cell (s):", g.groupby("cell").t.max().round(0).tolist())
    for a,c in [(0,.01),(.01,.05),(.05,.25),(.25,1.01)]:
        s=g[(g.tf>=a)&(g.tf<c)]
        print(f"  run fraction [{a},{c}): batches {len(s)}, gain share {s.gain.sum()/G*100:.1f}%, rel>10% rate {(s.rel>.1).mean()*100:.1f}%, mean hand label/task {(s.hand_label/s.n).mean():.1f}s")
    late=g[g.tf>=.05]; print(f"  after first 5% of run: label gain {late.gain.sum()/late.hand_label.sum()*100:.2f}% of hand seconds")
    top=g.sort_values("gain",ascending=False); c=top.gain.cumsum()/G
    print("  batches carrying 50/80% of gain:", int((c<.5).sum())+1, int((c<.8).sum())+1, "of", len(g))
    big=g[g.rel>.1]
    print("  >10%: hand stacked (h_stack>0)", round((big.h_stack>0).mean(),3), "base", round((g.h_stack>0).mean(),3), "| hand plan within one node", round((big.h_nodes==1).mean(),3), "base", round((g.h_nodes==1).mean(),3))
