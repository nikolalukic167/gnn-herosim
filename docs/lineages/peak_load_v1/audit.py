"""peak_load_v1 independent audit of Amendment 1 (job 812291), recomputed from the raw summaries without the reader.
Usage: audit.py <dir with groundedladder *.summary.json/*.failed.json and inputs/selected.json> [out.json]"""
import json,glob,re,collections,os,sys
import numpy as np
from scipy.stats import wilcoxon
D=sys.argv[1]
topos=json.load(open(D+'/inputs/selected.json'))['topologies']
pat=re.compile(r'cc40s(\d+)__(g\d)(x\d+)__(.+)_s(\d+)\.(summary|failed)\.json')
R={}; fails=[]; integ=collections.Counter(); commits=collections.Counter(); dirty=0; ntasks=collections.Counter(); meta_mismatch=[]
for f in sorted(glob.glob(D+'/*.json')):
    m=pat.match(os.path.basename(f)); 
    if not m: continue
    t,g,x,pol,s,kind=m.groups(); t=int(t); s=int(s)
    if kind=='failed': fails.append(json.load(open(f))|{'file':os.path.basename(f)}); continue
    d=json.load(open(f))
    if d['topology']!=t or d['window']!=g+x or d['checkpoint_seed']!=s: meta_mismatch.append(f)
    commits[d['code']['commit']]+=1; dirty+=d['code']['dirty']; ntasks[d['num_tasks']]+=1
    n=d['num_tasks']
    R[(t,g,x,pol,s)]=dict(L=d['averageElapsedTime'],Q=d['averageQueueTime'],W=d['averageWaitTime'],
       X=d['totalPeerExchangeTime']/n,RV=d['totalPeerRendezvousWait']/n,qs=d.get('queue_share'),kind=d['policy_name'])
out={'integrity':{}}
G='xs1load_selfref'; rungs=['x20','x30','x50']; wins=['g0','g1','g2','g3']; seeds=[1,2,3,4]
learned=[k for k in R if k[3]==G]
missing=[(t,g,x,s) for t in topos for g in wins for x in rungs for s in seeds if (t,g,x,G,s) not in R]
out['integrity']=dict(learned_present=len(learned),learned_missing=missing,learned_failed=[f for f in fails if G in f['file']],
  failures=[(f['file'],f.get('why'),f.get('returncode'),f.get('wallclock_s')) for f in fails],
  commits=dict(commits),dirty_runs=dirty,num_tasks=dict(ntasks),meta_mismatch=meta_mismatch,
  topologies_in_data=sorted({k[0] for k in R}))
rules=['cd','batched','decima','reactive','random','selfpredict']
def pct(a,b): return 100*(a/b-1)
def contrast(x,ref,agg=np.median,seedset=seeds):
    pt={}
    for t in topos:
        vals=[]
        for g in wins:
            r=R.get((t,g,x,ref,0))
            if r is None: continue
            for s in seedset:
                a=R.get((t,g,x,G,s))
                if a: vals.append(pct(a['L'],r['L']))
        if vals: pt[t]=float(agg(vals))
    v=np.array(list(pt.values()))
    p=float(wilcoxon(v,method='exact').pvalue) if len(v)>1 else None
    return dict(median=float(np.median(v)),p=p,neg=int((v<0).sum()),n=len(v),per_topo=pt)
res={}
for x in rungs:
    res[x]={}
    for ref in rules:
        res[x][ref]=dict(main=contrast(x,ref),mean_over_seeds=contrast(x,ref,np.mean))
    # pooled cell-level
    cells=[]
    for t in topos:
        for g in wins:
            r=R.get((t,g,x,'cd',0)); 
            if not r: continue
            a=[R[(t,g,x,G,s)]['L'] for s in seeds if (t,g,x,G,s) in R]
            cells.append(pct(np.mean(a),r['L']))
    cells=np.array(cells)
    res[x]['cd']['cell_pooled']=dict(n=len(cells),median=float(np.median(cells)),neg=int((cells<0).sum()),
        p=float(wilcoxon(cells).pvalue),p_method='auto(exact if n<=50 no ties)')
    # geomean ratio per topology then bootstrap
    lr={}
    for t in topos:
        l=[np.log(R[(t,g,x,G,s)]['L']/R[(t,g,x,'cd',0)]['L']) for g in wins for s in seeds if (t,g,x,'cd',0) in R and (t,g,x,G,s) in R]
        lr[t]=np.mean(l)
    arr=np.array(list(lr.values())); rng=np.random.default_rng(0)
    boots=np.exp(arr[rng.integers(0,len(arr),(10000,len(arr)))].mean(1))
    res[x]['cd']['geomean_ratio']=dict(point=float(np.exp(arr.mean())),ci95=[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))])
    res[x]['cd']['per_seed']={s:{k:v for k,v in contrast(x,'cd',seedset=[s]).items() if k!='per_topo'} for s in seeds}
    # descriptive
    desc={}
    for pol in rules+[G]:
        rs=[v for k,v in R.items() if k[2]==x and k[3]==pol and k[0] in topos]
        desc[pol]=dict(n=len(rs),mean_L=float(np.mean([r['L'] for r in rs])),median_L=float(np.median([r['L'] for r in rs])),
          med_queue=float(np.median([r['Q'] for r in rs])),med_wait=float(np.median([r['W'] for r in rs])),
          med_exchange=float(np.median([r['X'] for r in rs])),med_rendezvous=float(np.median([r['RV'] for r in rs])),
          med_residual=float(np.median([r['L']-r['Q']-r['W']-r['X']-r['RV'] for r in rs])),
          med_queue_share=float(np.median([r['Q']/r['L'] for r in rs])))
    res[x]['descriptive']=desc
out['results']=res
json.dump(out,open(sys.argv[2] if len(sys.argv)>2 else 'audit.json','w'),indent=1,default=str)
print(json.dumps(out['integrity'],indent=0,default=str)[:3000])
for x in rungs:
    print('==',x)
    for ref in rules:
        m=res[x][ref]['main']; mm=res[x][ref]['mean_over_seeds']
        print(f"{ref:12s} med {m['median']:.4f} p {m['p']:.5f} {m['neg']}/{m['n']} | meanseeds {mm['median']:.4f} p {mm['p']:.5f} {mm['neg']}/{mm['n']}")
    c=res[x]['cd']; print('cell',c['cell_pooled'],'geo',c['geomean_ratio'])
    for s,v in c['per_seed'].items(): print(' seed',s,v)
    for pol,d in res[x]['descriptive'].items(): print(f" {pol:16s}"+' '.join(f"{k}={v:.3f}" if isinstance(v,float) else f"{k}={v}" for k,v in d.items()))
