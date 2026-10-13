import sys; sys.path.insert(0, "/tmp/claude-0/-root-projects-my-herosim/b8711e56-08ac-40f3-86ba-fa809552eb5c/scratchpad/gnn_wins_repos/shims")
sys.path.insert(0, "/tmp/claude-0/-root-projects-my-herosim/b8711e56-08ac-40f3-86ba-fa809552eb5c/scratchpad/gnn_wins_repos/spark-sched-sim")
import os; os.chdir("/tmp/claude-0/-root-projects-my-herosim/b8711e56-08ac-40f3-86ba-fa809552eb5c/scratchpad/gnn_wins_repos/spark-sched-sim")
import os.path as osp, time, numpy as np, torch
import gymnasium as gym
from cfg_loader import load
from schedulers import RoundRobinScheduler, make_scheduler
from schedulers.heuristics.utils import preprocess_obs, find_stage
from spark_sched_sim import metrics
ENV_CFG={"num_executors":10,"job_arrival_cap":50,"job_arrival_rate":4.0e-5,"moving_delay":2000.0,"warmup_delay":1000.0,"data_sampler_cls":"TPCHDataSampler"}
class SJFCap:
    env_wrapper_cls=None
    def __init__(self, cap): self.cap=cap
    def schedule(self, obs):
        preprocess_obs(obs); nodes=obs["dag_batch"].nodes; ptr=obs["dag_ptr"]; n=len(obs["exec_supplies"])
        work=[float((nodes[ptr[j]:ptr[j+1],0]*np.maximum(nodes[ptr[j]:ptr[j+1],1],1.0)).sum()) for j in range(n)]
        order=np.argsort(work, kind='stable'); sj=obs["source_job_idx"]
        if sj<n and obs["exec_supplies"][sj] < self.cap + obs["num_committable_execs"]:
            s=find_stage(obs,sj)
            if s!=-1: return {"stage_idx":s,"num_exec":min(obs["num_committable_execs"], max(1,self.cap-(obs["exec_supplies"][sj]-obs["num_committable_execs"])))},{}
        for j in order:
            if obs["exec_supplies"][j]>=self.cap or j==sj: continue
            s=find_stage(obs,j)
            if s==-1: continue
            return {"stage_idx":s,"num_exec":min(obs["num_committable_execs"], self.cap-obs["exec_supplies"][j])},{}
        return {"stage_idx":-1,"num_exec":obs["num_committable_execs"]},{}
def run(sched, seed):
    env=gym.make("spark_sched_sim:SparkSchedSimEnv-v0", env_cfg=ENV_CFG)
    if sched.env_wrapper_cls: env=sched.env_wrapper_cls(env)
    obs,_=env.reset(seed=seed, options=None); term=trunc=False
    while not (term or trunc):
        a,_=sched.schedule(obs); obs,_,term,trunc,_=env.step(a)
    v=metrics.avg_job_duration(env)*1e-3; env.close(); return v
seed=int(sys.argv[1]); t0=time.time()
cfg=load(filename=osp.join("config","decima_tpch.yaml"))
dec=make_scheduler(cfg["agent"]|{"num_executors":10,"state_dict_path":osp.join("models","decima","model.pt")})
fair=run(RoundRobinScheduler(10,True),seed); c5=run(SJFCap(5),seed); c10=run(SJFCap(10),seed)
d=[]
for rep in range(3):
    torch.manual_seed(1000*seed+rep); import random; random.seed(1000*seed+rep); np.random.seed(1000*seed+rep)
    d.append(run(dec,seed))
dm=float(np.mean(d))
print(f"seed {seed}: fair {fair:.0f} | decima mean {dm:.0f} (min {min(d):.0f} max {max(d):.0f}; runs {' '.join(f'{x:.0f}' for x in d)}) | cap5 {c5:.0f} ({(c5/dm-1)*100:+.1f}% vs decima mean) | cap10 {c10:.0f} ({(c10/dm-1)*100:+.1f}% vs decima mean) | wall {time.time()-t0:.0f}s", flush=True)
