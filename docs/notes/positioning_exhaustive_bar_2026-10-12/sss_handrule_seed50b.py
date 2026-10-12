import sys
seed = int(sys.argv[1])
src = open('/tmp/claude-0/-root-projects-my-herosim/b8711e56-08ac-40f3-86ba-fa809552eb5c/scratchpad/gnn_wins_repos/sss_handrule_seed.py').read()
exec(src.split('seed=int(sys.argv[1])')[0])
ENV_CFG.update({"num_executors": 50, "job_arrival_cap": 200})


class SJF2(SJFCap):
    """Unknown stage durations <- job's mean known duration, else global mean of known durations."""

    def schedule(self, obs):
        preprocess_obs(obs)
        nodes = obs["dag_batch"].nodes
        ptr = obs["dag_ptr"]
        n = len(obs["exec_supplies"])
        known = nodes[:, 1][nodes[:, 1] > 0]
        g = float(known.mean()) if len(known) else 1.0
        work = []
        for j in range(n):
            seg = nodes[ptr[j]:ptr[j + 1]]
            kj = seg[:, 1][seg[:, 1] > 0]
            fill = float(kj.mean()) if len(kj) else g
            work.append(float((seg[:, 0] * np.where(seg[:, 1] > 0, seg[:, 1], fill)).sum()))
        order = np.argsort(work, kind='stable')
        sj = obs["source_job_idx"]
        if sj < n and obs["exec_supplies"][sj] < self.cap + obs["num_committable_execs"]:
            s = find_stage(obs, sj)
            if s != -1:
                return {"stage_idx": s, "num_exec": min(obs["num_committable_execs"], max(1, self.cap - (obs["exec_supplies"][sj] - obs["num_committable_execs"])))}, {}
        for j in order:
            if obs["exec_supplies"][j] >= self.cap or j == sj:
                continue
            s = find_stage(obs, j)
            if s == -1:
                continue
            return {"stage_idx": s, "num_exec": min(obs["num_committable_execs"], self.cap - obs["exec_supplies"][j])}, {}
        return {"stage_idx": -1, "num_exec": obs["num_committable_execs"]}, {}


t0 = time.time()
hand = {c: run(SJF2(c), seed) for c in [10, 25, 50]}
print(f"seed {seed} SJF2: " + " | ".join(f"cap{c} {v:.0f}" for c, v in hand.items()) + f" | wall {time.time() - t0:.0f}s", flush=True)
