import sys; sys.path.insert(0, "/tmp/claude-0/-root-projects-my-herosim/b8711e56-08ac-40f3-86ba-fa809552eb5c/scratchpad/gnn_wins_repos/shims")
sys.path.insert(0, "/tmp/claude-0/-root-projects-my-herosim/b8711e56-08ac-40f3-86ba-fa809552eb5c/scratchpad/gnn_wins_repos/spark-sched-sim")
import os.path as osp, time, numpy as np
import gymnasium as gym
from cfg_loader import load
from schedulers import RoundRobinScheduler, make_scheduler
from schedulers.heuristics.utils import preprocess_obs, find_stage
from spark_sched_sim import metrics
ENV_CFG={"num_executors":10,"job_arrival_cap":50,"job_arrival_rate":4.0e-5,"moving_delay":2000.0,"warmup_delay":1000.0,"data_sampler_cls":"TPCHDataSampler"}
class SJFCap:
    """Hand rule with the model's information: shortest-remaining-work job first, per-job executor cap."""
    env_wrapper_cls=None
    def __init__(self, cap): self.cap=cap; self.name=f'sjf_cap{cap}'
    def schedule(self, obs):
        preprocess_obs(obs); nodes=obs["dag_batch"].nodes; ptr=obs["dag_ptr"]; n=len(obs["exec_supplies"])
        work=[]
        for j in range(n):
            seg=nodes[ptr[j]:ptr[j+1]]
            work.append(float((seg[:,0]*np.maximum(seg[:,1],1.0)).sum()))
        order=np.argsort(work, kind='stable')
        # keep executors on the source job if it still has work and is under cap
        sj=obs["source_job_idx"]
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
cfg=load(filename=osp.join("config","decima_tpch.yaml"))
dec=make_scheduler(cfg["agent"]|{"num_executors":10,"state_dict_path":osp.join("models","decima","model.pt")})
arms=[('fair',RoundRobinScheduler(10,True)),('sjf_cap2',SJFCap(2)),('sjf_cap3',SJFCap(3)),('sjf_cap5',SJFCap(5)),('sjf_cap10',SJFCap(10)),('decima',dec)]
for seed in [1234,1,2,3]:
    t0=time.time(); r={k:run(s,seed) for k,s in arms}
    print(f"seed {seed}: "+"  ".join(f"{k} {v:.0f}" for k,v in r.items())+f"   best_hand/decima-1 = {(min(r[k] for k in r if k.startswith('sjf'))/r['decima']-1)*100:+.1f}%  wall {time.time()-t0:.0f}s", flush=True)
