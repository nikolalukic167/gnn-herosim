"""state -> largest per-replica compute-lock backlog (queued service seconds) at the decision instant, from the live trace.
usage: bl.py trace out.json [--pick N] ; also prints ids of states above 100 s"""
import sys,json
import numpy as np
sys.path.insert(0,'.')
from scripts_cosim.physics_audit.i11_replay import live_states, read_jsonl, skey
tr=sys.argv[1]; out=sys.argv[2]; tmin=float(sys.argv[3]) if len(sys.argv)>3 else 0.0; cap=int(sys.argv[4]) if len(sys.argv)>4 else 40
svc=read_jsonl(tr,{"svc"})
by={}
for r in svc:
    if r.get("io_end") is None or r.get("compute_start") is None: continue
    by.setdefault(r["q"],[]).append((r["io_end"],r["compute_start"],r["done"]-r["compute_start"]))
arr={q:np.array(v) for q,v in by.items()}
res={}
for s in live_states(tr):
    t=s["t"]; best=0.0; oldest=0.0; nw=0
    for q,a in arr.items():
        m=(a[:,0]<=t)&(t<a[:,1])
        if m.any():
            w=a[m,2].sum()
            if w>best: best=w; oldest=float(t-a[m,0].min()); nw=int(m.sum())
    res[f"{s['batch'][0]}@{round(t,6)}"]={"t":t,"b0":s["batch"][0],"backlog_s":float(best),"oldest_wait_s":oldest,"waiters":nw}
json.dump(res,open(out,"w"))
v=sorted(res.values(),key=lambda r:r["t"]); hi=[r for r in v if r["backlog_s"]>100 and r["t"]>=tmin]
print(len(v),"states;",len(hi),"with backlog >100 s; max backlog",max(r["backlog_s"] for r in v))
step=max(1,len(hi)//cap)
open(out+".ids","w").write(",".join(str(r["b0"]) for r in hi[::step][:cap]))
