"""Size distribution of the Alibaba fan-outs the WF1 mint drew, BEFORE the --max-size 10 cut (size_raw).
Run where /share is mounted (datalab): python3 fanout_uncut.py [library dir]. Libraries only, no CSV.
"""
import gzip, json, sys
import numpy as np
D=sys.argv[1] if len(sys.argv)>1 else "/share/nikola.lukic/surrogate_gap/alibaba_fanout"
bins=[(1,1),(2,10),(11,20),(21,50),(51,10**9)]
tot=[]
print("shard  groups  max | groups% / tasks% in 2-10, 11-20, 21-50, 51+ | tasks%>10")
for s in (0,50,130,100):
    L=json.load(gzip.open(f"{D}/fanout_lib_{s}.json.gz")); sz=np.array([g["size_raw"] for g in L["groups"]]); tot.append(sz)
    assert L["rule"]["max_size"]==10
def rep(name,sz):
    n=len(sz);T=sz.sum();row=[]
    for lo,hi in bins:
        m=(sz>=lo)&(sz<=hi); row.append(f"{100*m.mean():.2f}/{100*sz[m].sum()/T:.2f}")
    print(f"{name:6} n={n} max={sz.max()} mean={sz.mean():.2f} | "+"  ".join(f"[{lo}-{hi if hi<1e9 else '+'}] {r}" for (lo,hi),r in zip(bins,row))+f" | tasks>10: {100*sz[sz>10].sum()/T:.2f}%  >20: {100*sz[sz>20].sum()/T:.2f}%")
for s,sz in zip((0,50,130,100),tot): rep(str(s),sz)
rep("pooled",np.concatenate(tot))
