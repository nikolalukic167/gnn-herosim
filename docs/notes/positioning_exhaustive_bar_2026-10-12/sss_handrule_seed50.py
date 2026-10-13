import sys
seed=int(sys.argv[1])
src=open('/tmp/claude-0/-root-projects-my-herosim/b8711e56-08ac-40f3-86ba-fa809552eb5c/scratchpad/gnn_wins_repos/sss_handrule_seed.py').read()
head=src.split('seed=int(sys.argv[1])')[0]
exec(head)
import random
# training config: config/decima_tpch.yaml env: num_executors 50, job_arrival_cap 200, rate 4e-5, moving 2000, warmup 1000
ENV_CFG.update({"num_executors":50,"job_arrival_cap":200})
NE=50; t0=time.time()
cfg=load(filename=osp.join("config","decima_tpch.yaml"))
dec=make_scheduler(cfg["agent"]|{"num_executors":NE,"state_dict_path":osp.join("models","decima","model.pt")})
fair=run(RoundRobinScheduler(NE,True),seed)
hand={c:run(SJFCap(c),seed) for c in [5,10,25,50]}
d=[]
for rep in range(3):
    torch.manual_seed(1000*seed+rep); random.seed(1000*seed+rep); np.random.seed(1000*seed+rep)
    d.append(run(dec,seed))
dm=float(np.mean(d))
print(f"seed {seed}: fair {fair:.0f} | decima mean {dm:.0f} (min {min(d):.0f} max {max(d):.0f}; runs {' '.join(f'{x:.0f}' for x in d)}) | " + " | ".join(f"cap{c} {v:.0f} ({(v/dm-1)*100:+.1f}%)" for c,v in hand.items()) + f" | wall {time.time()-t0:.0f}s", flush=True)
