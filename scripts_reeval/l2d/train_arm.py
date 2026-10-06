"""literature_reeval_v1 -- train one arm x seed x size of L2D with the upstream PPO loop.

This is upstream ``PPO_jssp_multiInstances.main`` with exactly four changes, all logged:
  1. the policy is fed ``arm_features.build_inputs(arm, ...)`` instead of the raw (adj, fea);
  2. seeds and the training-instance stream are CLI arguments (upstream hard-codes 200/600 and
     draws its first 100 training instances from the SAME RNG state as its validation set);
  3. both the best-validation checkpoint (upstream protocol) and the final checkpoint are saved,
     each with a JSON sidecar (arm, seeds, dims, upstream commit);
  4. curves go to W&B (repo rule 5) and to ``curves.json``.

Upstream ``PPO.update`` is reused verbatim (its ``aggr_obs`` block-diagonalises whatever adjacency
tensors were stored; for identity arms that is the identity, so the update sees the same inputs
the rollout saw).

Usage (from repo root, herosim pipenv):
  PIPENV_IGNORE_VIRTUALENVS=1 pipenv run python3 scripts_reeval/l2d/train_arm.py \
      --arm mlp_t1 --n_j 6 --n_m 6 --seed 0 --out simulation_data/literature_reeval_v1/l2d/runs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l2d_bridge import load_l2d, upstream_commit  # noqa: E402
from arm_features import ARMS, INPUT_DIM, USES_REAL_ADJ, build_inputs  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--n_j", type=int, required=True)
    p.add_argument("--n_m", type=int, required=True)
    p.add_argument("--seed", type=int, required=True, help="study seed s; torch_seed=600+s, np_seed_train=1000+s")
    p.add_argument("--max_updates", type=int, default=10000, help="upstream default 10000")
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--val_every", type=int, default=100)
    p.add_argument("--save_updates", default="",
                   help="comma-separated fixed update counts at which to save policy checkpoints")
    p.add_argument("--lr", type=float, default=None,
                   help="Adam learning rate; default = upstream configs.lr (2e-5). Amendment A1 sweeps this per arm.")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--wandb_project", default="literature-reeval-l2d")
    p.add_argument("--no_wandb", action="store_true", help="smoke tests only; every real run logs")
    return p.parse_args()


def make_policy(L, arm, n_j, n_m, device):
    c = L.configs
    return L.ActorCritic(
        n_j=n_j, n_m=n_m, num_layers=c.num_layers, learn_eps=False,
        neighbor_pooling_type=c.neighbor_pooling_type, input_dim=INPUT_DIM[arm], hidden_dim=c.hidden_dim,
        num_mlp_layers_feature_extract=c.num_mlp_layers_feature_extract,
        num_mlp_layers_actor=c.num_mlp_layers_actor, hidden_dim_actor=c.hidden_dim_actor,
        num_mlp_layers_critic=c.num_mlp_layers_critic, hidden_dim_critic=c.hidden_dim_critic, device=device,
    )


def greedy_eval(L, arm, policy, data_list, n_j, n_m, device):
    env = L.SJSSP(n_j=n_j, n_m=n_m)
    n = n_j * n_m
    g_pool = L.g_pool_cal(L.configs.graph_pool_type, torch.Size([1, n, n]), n, device)
    out = []
    for data in data_list:
        adj, fea, cand, mask = env.reset(data)
        total = -env.initQuality
        while True:
            adj_t, fea_t = build_inputs(arm, env, adj, fea, device)
            with torch.no_grad():
                pi, _ = policy(x=fea_t, graph_pool=g_pool, padded_nei=None, adj=adj_t,
                               candidate=torch.from_numpy(np.copy(cand)).to(device).unsqueeze(0),
                               mask=torch.from_numpy(np.copy(mask)).to(device).unsqueeze(0))
            a = L.greedy_select_action(pi, cand)
            adj, fea, r, done, cand, mask = env.step(a.item())
            total += r
            if done:
                break
        out.append(float(-(total - env.posRewards)))
    return np.array(out)


def save_ckpt(path, policy, meta):
    torch.save(policy.state_dict(), path)
    with open(path + ".contract.json", "w") as fh:
        json.dump(meta, fh, indent=1)


def main():
    a = parse()
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    L = load_l2d(a.n_j, a.n_m, a.device)
    c = L.configs
    torch_seed, np_seed_train = 600 + a.seed, 1000 + a.seed
    lr = c.lr if a.lr is None else a.lr
    save_updates = {int(v) for v in a.save_updates.split(",") if v}
    if any(v < 1 or v > a.max_updates for v in save_updates):
        raise ValueError("save_updates must be between 1 and max_updates")
    lr_tag = "" if a.lr is None else f"_lr{a.lr:g}"
    up_tag = "" if a.max_updates == 10000 else f"_up{a.max_updates}"
    run_name = f"l2d_{a.n_j}x{a.n_m}_{a.arm}{lr_tag}{up_tag}_s{a.seed}"
    out_dir = os.path.join(a.out, run_name)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "final.pth")):
        print(f"{run_name}: final.pth exists, refusing to overwrite a finished run", file=sys.stderr)
        sys.exit(3)

    meta = dict(lineage="literature_reeval_v1", target="L2D", upstream_commit=upstream_commit(),
                arm=a.arm, input_dim=INPUT_DIM[a.arm], real_adjacency=USES_REAL_ADJ[a.arm],
                n_j=a.n_j, n_m=a.n_m, study_seed=a.seed, torch_seed=torch_seed, np_seed_train=np_seed_train,
                val_set=f"DataGen/generatedData{a.n_j}_{a.n_m}_Seed200.npy", max_updates=a.max_updates,
                num_envs=a.num_envs, lr=lr, hidden_dim=c.hidden_dim, num_layers=c.num_layers,
                torch_version=torch.__version__, device=a.device)

    wandb = None
    if not a.no_wandb:
        import wandb as _w
        wandb = _w
        wandb.init(project=a.wandb_project, name=run_name, config=meta, dir=out_dir,
                   tags=[a.arm, f"{a.n_j}x{a.n_m}", "literature_reeval_v1"])

    vali = np.load(os.path.join(L.root, meta["val_set"]))
    vali_data = [(vali[i][0], vali[i][1]) for i in range(vali.shape[0])]

    torch.manual_seed(torch_seed)
    np.random.seed(np_seed_train)

    # upstream PPO object for its update(); swap in a policy with this arm's input_dim
    ppo = L.ppo_mod.PPO(lr, c.gamma, c.k_epochs, c.eps_clip, n_j=a.n_j, n_m=a.n_m,
                        num_layers=c.num_layers, neighbor_pooling_type=c.neighbor_pooling_type,
                        input_dim=INPUT_DIM[a.arm], hidden_dim=c.hidden_dim,
                        num_mlp_layers_feature_extract=c.num_mlp_layers_feature_extract,
                        num_mlp_layers_actor=c.num_mlp_layers_actor, hidden_dim_actor=c.hidden_dim_actor,
                        num_mlp_layers_critic=c.num_mlp_layers_critic, hidden_dim_critic=c.hidden_dim_critic)
    # upstream constructs on configs.device (== a.device here) -- assert rather than trust
    assert next(ppo.policy.parameters()).device.type == device.type
    n_params = sum(p.numel() for p in ppo.policy.parameters())
    meta["n_params"] = n_params
    print(f"{run_name}: params={n_params} upstream={meta['upstream_commit'][:8]}")

    envs = [L.SJSSP(n_j=a.n_j, n_m=a.n_m) for _ in range(a.num_envs)]
    memories = [L.ppo_mod.Memory() for _ in range(a.num_envs)]
    n = a.n_j * a.n_m
    g_pool_step = L.g_pool_cal(c.graph_pool_type, torch.Size([1, n, n]), n, device)

    curves = {"train_reward": [], "val": [], "loss": []}
    best_val, best_update = float("inf"), -1
    t_start = time.time()
    for i_update in range(a.max_updates):
        ep_rewards = [0.0] * a.num_envs
        states = []
        for i, env in enumerate(envs):
            adj, fea, cand, mask = env.reset(L.uni_instance_gen(n_j=a.n_j, n_m=a.n_m, low=c.low, high=c.high))
            states.append((adj, fea, cand, mask))
            ep_rewards[i] = -env.initQuality
        while True:
            tensors = []
            for i in range(a.num_envs):
                adj, fea, cand, mask = states[i]
                adj_t, fea_t = build_inputs(a.arm, envs[i], adj, fea, device)
                tensors.append((adj_t, fea_t, torch.from_numpy(np.copy(cand)).to(device),
                                torch.from_numpy(np.copy(mask)).to(device)))
            actions = []
            with torch.no_grad():
                for i in range(a.num_envs):
                    adj_t, fea_t, cand_t, mask_t = tensors[i]
                    pi, _ = ppo.policy_old(x=fea_t, graph_pool=g_pool_step, padded_nei=None, adj=adj_t,
                                           candidate=cand_t.unsqueeze(0), mask=mask_t.unsqueeze(0))
                    act, a_idx = L.select_action(pi, states[i][2], memories[i])
                    actions.append((act, a_idx))
            new_states = []
            for i in range(a.num_envs):
                adj_t, fea_t, cand_t, mask_t = tensors[i]
                m = memories[i]
                m.adj_mb.append(adj_t); m.fea_mb.append(fea_t); m.candidate_mb.append(cand_t)
                m.mask_mb.append(mask_t); m.a_mb.append(actions[i][1])
                adj, fea, r, done, cand, mask = envs[i].step(actions[i][0].item())
                new_states.append((adj, fea, cand, mask))
                ep_rewards[i] += r
                m.r_mb.append(r); m.done_mb.append(done)
            states = new_states
            if envs[0].done():
                break
        for j in range(a.num_envs):
            ep_rewards[j] -= envs[j].posRewards
        loss, v_loss = ppo.update(memories, n, c.graph_pool_type)
        for m in memories:
            m.clear_memory()
        mean_r = float(sum(ep_rewards) / len(ep_rewards))
        if not np.isfinite(loss):
            raise RuntimeError(f"{run_name}: non-finite loss at update {i_update}")
        curves["train_reward"].append(mean_r)
        curves["loss"].append([float(loss), float(v_loss)])
        log = {"update": i_update + 1, "train_makespan": -mean_r, "loss": float(loss), "v_loss": float(v_loss)}
        if i_update + 1 in save_updates:
            save_ckpt(os.path.join(out_dir, f"update_{i_update + 1}.pth"), ppo.policy,
                      {**meta, "checkpoint": "fixed_update", "selected_update": i_update + 1})
        if (i_update + 1) % a.val_every == 0:
            val_ms = float(greedy_eval(L, a.arm, ppo.policy, vali_data, a.n_j, a.n_m, device).mean())
            curves["val"].append([i_update + 1, val_ms])
            log["val_makespan"] = val_ms
            if val_ms < best_val:
                best_val, best_update = val_ms, i_update + 1
                save_ckpt(os.path.join(out_dir, "best_val.pth"), ppo.policy,
                          {**meta, "checkpoint": "best_val", "selected_update": best_update, "val_makespan": val_ms})
            elapsed = time.time() - t_start
            print(f"{run_name} upd {i_update+1} train_ms {-mean_r:.1f} val_ms {val_ms:.2f} best {best_val:.2f}@{best_update} "
                  f"{elapsed/60:.1f} min", flush=True)
            with open(os.path.join(out_dir, "curves.json"), "w") as fh:
                json.dump(curves, fh)
        if wandb is not None and ((i_update + 1) % 10 == 0 or "val_makespan" in log):
            wandb.log(log, step=i_update + 1)

    save_ckpt(os.path.join(out_dir, "final.pth"), ppo.policy,
              {**meta, "checkpoint": "final", "selected_update": a.max_updates, "best_val_makespan": best_val,
               "best_val_update": best_update, "wall_seconds": time.time() - t_start})
    with open(os.path.join(out_dir, "curves.json"), "w") as fh:
        json.dump(curves, fh)
    if wandb is not None:
        wandb.summary["best_val_makespan"] = best_val
        wandb.summary["best_val_update"] = best_update
        wandb.finish()
    print(f"{run_name}: done best_val {best_val:.2f}@{best_update} in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
