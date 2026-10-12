import pandas as pd, numpy as np, sys
b = pd.read_csv(sys.argv[1]+"/batches.tsv", sep="\t")
b["gain"]=b.hand_label-b.best_label; b["rel"]=b.gain/b.hand_label
b["tf"]=b.t/b.groupby(["cell","rung"]).t.transform("max")
for rung,g in b.groupby("rung"):
  G=g.gain.sum()
  for ph,s in [("open <5%",g[g.tf<.05]),("steady",g[g.tf>=.05])]:
    big=s[s.rel>.1]
    ds=big.b_stack-big.h_stack; dc=big.b_coloc-big.h_coloc
    gl=lambda m: round(big[m].gain.sum()/G*100,1)
    print(f"{rung:8s} {ph:9s} >10% batches {len(big):5d} | oracle unstacks {round((ds<0).mean()*100,1)}% (gain {gl(ds<0)}% of rung total), stacks more {round((ds>0).mean()*100,1)}% | fewer co-located pairs {round((dc<0).mean()*100,1)}%, more {round((dc>0).mean()*100,1)}% | same stack&coloc but new node {round(((ds==0)&(dc==0)&(big.new_nodes_vs_hand>0)).mean()*100,1)}% (gain {gl((ds==0)&(dc==0)&(big.new_nodes_vs_hand>0))}%) | oracle S-rank median {big.best_Srank.median():.0f}")
