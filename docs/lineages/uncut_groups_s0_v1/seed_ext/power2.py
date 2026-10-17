import json, os, glob, statistics as st, random, math, itertools
from collections import defaultdict
F="/share/nikola.lukic/uncut_groups"; FI=F+"/final"
TOP=list(range(19537,19549)); W=("g0","g1","g2","g3"); R=("moderate","heavy"); STATS=("whole","after360")
SRC={"bar":(F+"/arm1_k1_bar_d6ae87c2","cd_ib0_k100",(0,)),
 "A1g":(FI+"/cells_arm1","ra_gnn_eng_arm1_nosplit_k100",(1,2)),
 "A1bg":(FI+"/cells_arm1b","ra_gnn_eng_arm1b_nosplit_k100",(1,2)),"A1bt":(FI+"/cells_arm1b","ra_twin_eng_arm1b_nosplit_k100",(1,2)),
 "A3g":(FI+"/cells_arm3","ra_gnn_eng_arm3_nosplit_k100",(1,2)),
 "A2g":(FI+"/cells_arm2_g7","ra_gnn_eng_arm2_nosplit_k100",(1,2)),"A2t":(FI+"/cells_arm2_g7","ra_twin_eng_arm2_nosplit_k100",(1,2)),
 "K0g":(FI+"/cells_arm2k0_g7","ra_gnn_eng_arm2k0_nosplit_k100",(1,2)),"K0t":(FI+"/cells_arm2k0_g7","ra_twin_eng_arm2k0_nosplit_k100",(1,2))}
V={}
for a,(d,k,ss) in SRC.items():
  for t in TOP:
    for w in W:
      for r in R:
        for s in ss:
          p=f"{d}/cc40s{t}__{w}{r}__{k}_s{s}.summary.json"; j=json.load(open(p))
          V[(a,t,w,r,s)]={"whole":j["averageElapsedTime"],"after360":j["steady_state"]["after_360s"]["mean"]}
def y(a,b,t,r,stat,sa,sb):  # reader statistic for one topology, seeds sa of a, sb of b
  v=[]
  for w in W:
    ma=st.mean(V[(a,t,w,r,s)][stat] for s in sa); mb=st.mean(V[(b,t,w,r,s)][stat] for s in sb)
    v.append(100*(ma-mb)/mb)
  return st.median(v)
from functools import lru_cache
@lru_cache(None)
def nullcdf(n):
  cnt=defaultdict(int); cnt[0]=1
  for rk in range(1,n+1):
    nx=defaultdict(int)
    for s,c in cnt.items(): nx[s]+=c; nx[s+rk]+=c
    cnt=nx
  mx=n*(n+1)//2; acc=0; cdf=[]
  for s in range(mx+1): acc+=cnt[s]; cdf.append(acc)
  return cdf,2**n,mx
def pw(vals):
  n=len(vals); cdf,tot,mx=nullcdf(n)
  o=sorted(range(n),key=lambda i:abs(vals[i])); wp=sum(i+1 for i,ix in enumerate(o) if vals[ix]>0)
  return min(1.0,2*cdf[min(wp,mx-wp)]/tot)
# sign-test 95% CI order statistic (0-based lower index) per n
from math import comb
def lo_idx(n):
  k=0
  while True:
    cov=1-2*sum(comb(n,i) for i in range(k+1))/2**n
    if cov<0.95: return k-1
    k+=1
TESTS=[]
def add(fam,k,a,b,rungs,proxy=None,sign=-1): 
  for r in rungs:
    for s in STATS: TESTS.append((fam,k,a,b,r,s,proxy,sign))
add("E1 A3-vs-A1b",4,"A3g","A1bg",R)
add("E2 MP",8,"A1bg","A1bt",R); add("E2 MP",8,"A3g","A3t(proxy A1b)",R,proxy=("A1bg","A1bt"))
add("E3 heavy-vs-bar",6,"A3g","bar",("heavy",)); add("E3 heavy-vs-bar",6,"A1bg","bar",("heavy",)); add("E3 heavy-vs-bar",6,"K0g","bar",("heavy",))
add("E4 prefix/B1B2",4,"A2g","K0g",("heavy",),sign=+1); add("E4 prefix/B1B2",4,"K0g","A1g",("heavy",))
add("info MP K0",4,"K0g","K0t",R); add("info MP A2",4,"A2g","A2t",R)
random.seed(20261017); out=[]
SS=(2,4,6,8,10,12,16)
for fam,k,a,b,r,stat,proxy,sign in TESTS:
  aa,bb=(proxy if proxy else (a,b))
  sb=(0,) if bb=="bar" else None
  y1=[y(aa,bb,t,r,stat,(1,),sb or (1,)) for t in TOP]; y2=[y(aa,bb,t,r,stat,(2,),sb or (2,)) for t in TOP]
  yp=[y(aa,bb,t,r,stat,(1,2),sb or (1,2)) for t in TOP]
  se2=st.mean([(u-v)**2/2 for u,v in zip(y1,y2)])          # seed variance of one-seed topology value
  sa2=max(0.0,st.variance(yp)-se2/2)                        # topology x arm variance
  mu=st.mean(yp); med=st.median(yp)
  row={"family":fam,"k":k,"a":a,"b":b,"rung":r,"stat":stat,"median_pooled":round(med,2),"mean_pooled":round(mu,2),
       "sd_topo":round(math.sqrt(sa2),2),"sd_seed1":round(math.sqrt(se2),2),"p_current":round(pw(yp),4),"power":{},"neg_bound":{}}
  row["sign"]=sign; row["grid"]={}
  for eff,lab in ((mu,"obs"),(0.0,"null")):
    for n in (12,24,36,48):
      for S in (2,3,4,6,8,12,16):
        if n>12 and S not in (2,4): continue
        li=lo_idx(n); hit=nb=0; R=1500
        for _ in range(R):
          v=[eff+random.gauss(0,math.sqrt(sa2))+random.gauss(0,math.sqrt(se2/S)) for _ in range(n)]
          if pw(v)<0.05/k and sign*st.median(v)>0: hit+=1
          sv=sorted(sign*x for x in v)   # claimed direction positive
          if sv[n-1-li]<1.0: nb+=1        # upper CI end of the claimed-direction effect below 1 pct
        row["grid"][f"{lab}|n{n}|S{S}"]=(round(hit/R,2),round(nb/R,2))
  out.append(row); print(json.dumps(row))
json.dump(out,open(F+"/seeds_ext/power2.json","w"),indent=1)

