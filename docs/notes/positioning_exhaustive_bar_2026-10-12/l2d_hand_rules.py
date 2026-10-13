import sys, time, copy
sys.path.insert(0, "/tmp/claude-0/-root-projects-my-herosim/b8711e56-08ac-40f3-86ba-fa809552eb5c/scratchpad/gnn_wins_repos/L2D")
sys.argv=['t','--device','cpu','--n_j','10','--n_m','10']
import numpy as np
from Params import configs
from JSSP_Env import SJSSP
data=np.load('./DataGen/generatedData10_10_Seed200.npy')
l2d=np.load('./drlResult_10x10_10x10_Seed200.npy')
env=SJSSP(n_j=10,n_m=10)
def run(rule):
    out=[]; t0=time.time()
    for i in range(data.shape[0]):
        adj,fea,cand,mask=env.reset((data[i][0],data[i][1]))
        while True:
            acts=[a for a,mk in zip(cand,mask) if not mk]
            if rule=='spt':
                a=min(acts,key=lambda a: env.dur.flatten()[a])
            elif rule=='mwkr':
                a=max(acts,key=lambda a: env.dur.flatten()[a//env.number_of_machines, :].sum() if False else env.dur[a//env.number_of_machines, a%env.number_of_machines:].sum())
            elif rule=='mopnr':
                a=max(acts,key=lambda a: env.number_of_machines-(a%env.number_of_machines))
            elif rule=='fddmwkr':
                a=min(acts,key=lambda a: env.dur[a//env.number_of_machines, :a%env.number_of_machines+1].sum()/env.dur[a//env.number_of_machines, a%env.number_of_machines:].sum())
            elif rule=='greedy_lb':
                # one-step lookahead on the env's own objective (LB of makespan), tie-break MWKR
                best=None
                for a in acts:
                    e=SJSSP(n_j=10,n_m=10); e.__dict__.update(copy.deepcopy(dict(env.__dict__))); _,_,r,_,_,_=e.step(a)
                    key=(e.LBs.max(), -env.dur[a//env.number_of_machines, a%env.number_of_machines:].sum())
                    if best is None or key<best[0]: best=(key,a)
                a=best[1]
            adj,fea,r,done,cand,mask=env.step(a)
            if done: break
        out.append(env.LBs.max())
    return np.array(out), time.time()-t0
for rule in ['mopnr','fddmwkr','greedy_lb']:
    ms,t=run(rule)
    print(f'{rule:10s} mean makespan {ms.mean():.1f}  L2D {l2d.mean():.1f}  rule/L2D-1 = {(ms.mean()/l2d.mean()-1)*100:+.2f}%  wins(rule<L2D) {(ms<l2d).sum()}/100 ties {(ms==l2d).sum()}  time {t:.1f}s', flush=True)
