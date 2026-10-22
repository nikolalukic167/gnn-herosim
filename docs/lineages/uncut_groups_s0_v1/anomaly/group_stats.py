import json, numpy as np, os
U="/share/nikola.lukic/uncut_groups"; R=U+"/comb/cells_comb"; INP=U+"/comb/in_comball"
ARMS={"ref":("cd_ib0_inflightfull_k100",("s0",)),"gnn":("ra_gnn_eng_combfl_nosplit_k100",("s1","s2")),"mlp":("ra_mlp_same_combfl_nosplit_k100",("s1","s2"))}
TOPOS=range(19537,19549); W=("g0","g1","g2","g3"); T=360.0
wl={}
def groups(rung,w):
    if (rung,w) not in wl:
        ev=json.load(open(f"{INP}/wf1_{rung}/wl/grounded_{w}_n50000.json"))["events"]
        g=np.array([e["peer_group"] for e in ev]); u,inv,cnt=np.unique(g,return_inverse=True,return_counts=True)
        wl[(rung,w)]=(inv,cnt[inv],len(u))
    return wl[(rung,w)]
def cell(stem,rung,w):
    a=np.load(stem+".tasks.npy").astype(np.float64); inv,size,n=groups(rung,w)
    assert len(a)==len(inv)
    first=np.full(n,np.inf); last=np.full(n,-np.inf); mx=np.zeros(n)
    np.minimum.at(first,inv,a[:,0]); np.maximum.at(last,inv,a[:,1]); np.maximum.at(mx,inv,a[:,1]-a[:,0])
    sz=np.bincount(inv)
    return a, first, last-first, mx, sz
P=[50,90,99]
def stats(a,gc,mx,first,sz,mask):
    e=(a[:,1]-a[:,0])
    out={"task_mean":e.mean(),"task_p50":np.percentile(e,50),"task_p90":np.percentile(e,90),"task_p99":np.percentile(e,99)}
    g=gc[mask]; m=mx[mask]
    out.update({"grp_mean":g.mean(),"grp_p50":np.percentile(g,50),"grp_p90":np.percentile(g,90),"grp_p99":np.percentile(g,99),"slowest_member_mean":m.mean(),"n":mask.sum()})
    return out
res={}
for rung in ("moderate","heavy"):
  for scope in ("all","multi","after360"):
    rows={k:[] for k in ARMS}
    for t in TOPOS:
      for w in W:
        per={}
        for k,(kind,seeds) in ARMS.items():
            ss=[]
            for sd in seeds:
                a,first,gc,mx,sz=cell(f"{(U+'/comb/cells_bar_g8') if k=='ref' else R}/cc40s{t}__{w}{rung}__{kind}_{sd}",rung,w)
                mask={"all":sz>=1,"multi":sz>=2,"after360":(first>=T)&(sz>=2)}[scope]
                ss.append(stats(a,gc,mx,first,sz,mask)|{"_ref":np.sort(gc[mask])})
            per[k]=ss
        refp90=np.percentile(per["ref"][0]["_ref"],90); refp99=np.percentile(per["ref"][0]["_ref"],99)
        for k in ARMS:
            d={key:np.mean([s[key] for s in per[k]]) for key in per[k][0] if key!="_ref"}
            d["miss90"]=np.mean([(s["_ref"]>refp90).mean() for s in per[k]]); d["miss99"]=np.mean([(s["_ref"]>refp99).mean() for s in per[k]])
            rows[k].append(d)
    print(f"\n=== {rung}, scope={scope}: mean over 48 cells (12 topo x 4 windows); groups n/cell ~{np.mean([d['n'] for d in rows['ref']]):.0f}")
    keys=["task_mean","task_p50","task_p90","task_p99","grp_mean","grp_p50","grp_p90","grp_p99","slowest_member_mean","miss90","miss99"]
    print(f"{'':22s}"+"".join(f"{k:>10s}" for k in ("ref","gnn","mlp","gnn/ref%","mlp/ref%","gnn/mlp%","gnn<ref")))
    for k in keys:
        v={a:np.array([d[k] for d in rows[a]]) for a in ARMS}
        pr=lambda x,y:(np.mean(x/y)-1)*100 if not k.startswith("miss") else float("nan")
        print(f"{k:22s}{v['ref'].mean():10.3f}{v['gnn'].mean():10.3f}{v['mlp'].mean():10.3f}{pr(v['gnn'],v['ref']):10.2f}{pr(v['mlp'],v['ref']):10.2f}{pr(v['gnn'],v['mlp']):10.2f}{int((v['gnn']<v['ref']).sum()):>8d}/48")
