import json,glob,os,sys
root=sys.argv[1]
def q(v,p): v=sorted(v); return v[min(len(v)-1,int(p*len(v)))] if v else None
rows=[]
for d in sorted(glob.glob(root+'/c*')):
    hf=f'{d}/headroom_k5.jsonl'
    if not os.path.exists(hf): continue
    for l in open(hf):
        r=json.loads(l); r['seed']=os.path.basename(d).split('s')[-1]; rows.append(r)
def line(R):
    ok=[r for r in R if r['status']=='ok']; allp=[r for r in R if r.get('n_plans',0)>=2]
    v=[r['regret_rel']*100 for r in ok]; g=[r['gap_rel']*100 for r in allp if r.get('gap_rel') is not None]
    if not ok: return "no scored batches"
    return f"n {len(ok)}/{len(R)} | regret median {q(v,.5):.2f} % p90 {q(v,.9):.0f} % | >1 %: {100*sum(1 for x in v if x>1)/len(v):.0f} % | gap<1 %: {100*sum(1 for x in g if x<1)/len(g):.0f} % (ties {100*sum(1 for x in g if x==0)/len(g):.0f} %) | multi-node {100*sum(1 for r in allp if r['multi_node_opt'])/len(allp):.0f} %"
seeds=sorted({r['seed'] for r in rows}); rungs=sorted({r['rung'] for r in rows})
print("per seed (both rungs):"); [print("  ",s,line([r for r in rows if r['seed']==s])) for s in seeds]
print("per rung (all seeds):"); [print("  ",g,line([r for r in rows if r['rung']==g])) for g in rungs]
print("per seed x rung:"); [print("  ",s,g,line([r for r in rows if r['seed']==s and r['rung']==g])) for s in seeds for g in rungs]
