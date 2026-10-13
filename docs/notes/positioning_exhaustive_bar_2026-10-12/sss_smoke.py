import sys; sys.path.insert(0, "/tmp/claude-0/-root-projects-my-herosim/b8711e56-08ac-40f3-86ba-fa809552eb5c/scratchpad/gnn_wins_repos/shims")
import sys; sys.path.insert(0, "/tmp/claude-0/-root-projects-my-herosim/b8711e56-08ac-40f3-86ba-fa809552eb5c/scratchpad/gnn_wins_repos/spark-sched-sim")
import os.path as osp, time, sys
import gymnasium as gym
from cfg_loader import load
from schedulers import RoundRobinScheduler, make_scheduler
from spark_sched_sim import metrics
ENV_CFG={"num_executors":10,"job_arrival_cap":50,"job_arrival_rate":4.0e-5,"moving_delay":2000.0,"warmup_delay":1000.0,"data_sampler_cls":"TPCHDataSampler"}
def run(sched, seed):
    env=gym.make("spark_sched_sim:SparkSchedSimEnv-v0", env_cfg=ENV_CFG)
    if sched.env_wrapper_cls: env=sched.env_wrapper_cls(env)
    obs,_=env.reset(seed=seed, options=None); term=trunc=False
    while not (term or trunc):
        a,_=sched.schedule(obs); obs,_,term,trunc,_=env.step(a)
    v=metrics.avg_job_duration(env)*1e-3; env.close(); return v
cfg=load(filename=osp.join("config","decima_tpch.yaml"))
dec=make_scheduler(cfg["agent"]|{"num_executors":10,"state_dict_path":osp.join("models","decima","model.pt")})
fair=RoundRobinScheduler(10, dynamic_partition=True); fifo=RoundRobinScheduler(10, dynamic_partition=False)
for seed in [1234, 1, 2, 3]:
    t0=time.time(); r={}
    for name,s in [('fifo',fifo),('fair',fair),('decima',dec)]:
        r[name]=run(s,seed)
    print(f"seed {seed}: " + "  ".join(f"{k} {v:.1f}s" for k,v in r.items()) + f"  decima/fair-1 = {(r['decima']/r['fair']-1)*100:+.1f}%  wall {time.time()-t0:.0f}s", flush=True)
