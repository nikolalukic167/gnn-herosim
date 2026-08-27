---
description: Run or analyze normal-sim policy RTT comparisons. Read docs/notes/compare.md first; ask user what to compare before executing.
---

# Policy Comparison Protocol (`/compare`)

You are running a **normal simulation policy comparison** (not co-sim). Read **`docs/notes/compare.md`** first — it lists **fixed benchmarks** (Knative, HRC, dim14-ce GNN anchor, tabular Regime A), configs, existing result paths, run commands, and the latest snapshot table.

**Framing:** `dim14-ce` is a **fixed GNN anchor benchmark** — same class as Knative or HRC, not the global winner table. New dim14 GNN variants are **experiments** that must **beat dim14-ce** on the 5-config sum (and not regress on `default` >+2%) to ship.

## Step 1 — Ask the user (required)

Do **not** assume defaults. Use `AskQuestion` or a short clarifying message. Collect:

1. **Mode**
   - `analyze` — compare existing result JSONs only (no new sim runs)
   - `run` — execute new simulation(s) then compare
   - `both` — run missing ones, then compare
   - `datalab` — **do not run locally**; sync code/artifacts to datalab and produce step-by-step instructions (and optional rsync) for the on-cluster agent or user

2. **Policies / models to include** (one or more)
   - **Fixed benchmarks** (stable anchors — include when building a scoreboard):
     - `knative_network`
     - `herocache_network`
     - `gnn` + `models/near-rtt-v2-dim14-ce-only.pt` (**GNN anchor**)
     - **Regime A tabular:** `xgboost_batch` + `models/tabular/batch_edge_ranker.json` (full workload-100-100 sweep planned)
   - **New dim14 GNN experiments** (must beat dim14-ce to ship):
     - `gnn` + checkpoint e.g. `models/near-rtt-v2-dim14-ce-init-regret*.pt`, `models/near-rtt-v2-dim14-1060.pt`
   - **Regime A (batch loop):** pair `xgboost_batch` + `knative_network_batch` with 14-dim `gnn` only — do not mix with per-arrival `knative_network` in the same table
   - Other: `roundrobin`, `random_network`, `offload_network`

3. **Configs** (one or more from standard set)
   - **Default: `standard5`** — `default_20_20_p50`, `01_balanced_40_40_p50`, `02_balanced_50_50_p60`, `03_client_heavy_50_35_p50`, `05_sparse_40_40_p25` (~57 min GNN GPU)
   - **Fast gate: `gate3`** — `default` + `01` + `05` only (Track B ablation, ~35 min GNN)
   - **Full validation: `all7`** — all 7 configs including `00` and `04` (~80 min GNN)
   - Or custom `--config` path(s)

4. **Workload** (default: `data/nofs-ids/traces/workload-100-100.json`)

5. **Execution** (if running or `datalab`)
   - **Host:** `local` (default) | `datalab` (HPC at `cluster.datalab.tuwien.ac.at`)
   - **Local only:** foreground vs `nohup` background
   - Seed (default `42`), timeout (default `3600` for GNN/MLP, `7200` for XGB batch)
   - **Output sweep dir** (default for Reviewer Triangle all7: `simulation_data/normal_sim_sweeps/reviewer_triangle_all7_YYYYMMDD/`)

If the user already specified some of these in the same message, confirm the rest — don't re-ask what they gave.

**When `mode=datalab` or `execution=datalab`:** skip local `pipenv run` sims unless user explicitly asks for both hosts. Proceed to **Step 2b — Datalab**.

## Step 2 — Execute

### Step 2b — Datalab (HPC)

Use when user picks **`mode=datalab`** or **`execution=datalab`**. Goal: same compare as local, but runs on the cluster repo **`/home/nikola.lukic/gnn-herosim`** with **micromamba `gnn`** (not pipenv).

#### What is / is not in git

- **In git:** code, `simulation_data/space_with_network.json`, sweep scripts under `scripts_cosim/`
- **NOT in git** (must rsync): `models/**`, `data/nofs-ids/traces/workload-100-100.json`, sweep configs under `simulation_data/normal_sim_sweeps/*/configs/`, result JSONs
- **Training parquet not needed** for inference-only sweeps

#### Agent workflow (mitrix → datalab)

1. **Verify local code** — commit/push branch with required policy wiring; note commit hash for datalab `git fetch && git checkout <branch>`
2. **Choose sweep dir** — e.g. `reviewer_triangle_all7_20260609` (co-locate MLP/XGB/GNN JSONs) or user-named dir
3. **Rsync artifacts** (from mitrix, unless already on cluster):
   - **GNN dim14-ce all7:** `bash scripts_cosim/transfer_gnn_dim14_ce_to_datalab.sh`
   - **Regime A tabular (MLP + XGB):** rsync `models/tabular/batch_edge_mlp.pt`, `batch_edge_mlp.pt.meta.json`, `batch_edge_ranker.json`, `batch_edge_ranker.meta.json`, `data/nofs-ids/traces/workload-100-100.json`, and `simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/` (6 variant JSONs)
   - Override env if needed: `SSH_KEY`, `REMOTE=nikola.lukic@cluster.datalab.tuwien.ac.at`, `REPO=/home/nikola.lukic/gnn-herosim`
4. **Emit instructions** for the datalab agent/user (copy-paste block below)
5. **After jobs finish:** pull results — `bash scripts_cosim/transfer_reviewer_triangle_from_datalab.sh` (or rsync the sweep dir back)
6. **Analyze** pulled JSONs per Step 2 analyze — note cross-host RTT may differ (SimPy trajectories, not wall clock; see `LINEAGES.md`)

#### Datalab run recipes (pick by policies)

| Policies | Parallelism | Submit on datalab |
|----------|-------------|-------------------|
| **GNN dim14-ce** (Regime A, 7 cfg) | SLURM array 0–6 | `cd /home/nikola.lukic/gnn-herosim && sbatch scripts_cosim/datalab/reviewer_triangle_gnn_dim14_ce_all7.sbatch` |
| **MLP + XGB batch** (7 cfg each) | 14 independent jobs or array | One config per job; `micromamba activate gnn`; see `run_reviewer_triangle_regime_a_all7_nohup.sh` pattern but **prefer parallel SLURM** over sequential bash |
| **Single smoke** | 1 job | `bash scripts_cosim/datalab/run_reviewer_triangle_gnn_dim14_ce_one.sh <cfg_name> <config_json_path>` |

**GNN one-config env (datalab):** `GNN_MODEL_PATH=models/near-rtt-v2-dim14-ce-only.pt`, `GNN_DECODE_MODE=argmax`, `python3 scripts_cosim/run_simulation.py --gnn ...`

**Tabular one-config env (datalab):** `MLP_MODEL_PATH=models/tabular/batch_edge_mlp.pt` or `XGB_MODEL_PATH=models/tabular/batch_edge_ranker.json`, `GNN_DECODE_MODE=argmax`

**Monitor:** `squeue -u nikola.lukic`; logs under `/home/nikola.lukic/gnn-herosim/logs/rev-tri-*`

#### Topology stress sweeps (125-225, GNN vs MLP)

For **tiered-hub / phase-boundary** compares (not the standard 7-config anchor), read `docs/notes/compare.md` § Tiered-Hub and use:

- **Workload:** `data/nofs-ids/traces/workload-125-225.json` (562k tasks)
- **Active sweep:** `simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1/` — k∈{4,6,8} × seek∈{35,50,65}%, **5/30ms asymmetric latency**, **GNN batch size fixed at 4** — **9/9 GNN+MLP+Knative complete** (see `docs/notes/compare.md` § Bipartite)
- **Policies:** `gnn_dim22` + `mlp_dim22` (Regime A, same batch loop) · Knative `--knative_network` is **Regime B** (per-arrival) — label separately, do not mix as batch-fair peers
- **Prepare / compare:** `bash scripts_cosim/important/prepare_bipartite_coordination_sweep.sh` · `pipenv run python3 scripts_cosim/important/compare_bipartite_coordination_sweep.py --sweep-dir …`
- **Datalab:** `transfer_bipartite_coordination_to_datalab.sh` · `submit_bipartite_coordination_partition.sh GPU-a40 gpu:a40:1 a40` (or keep `GPU-l40s`); if `GPU-l40s` is saturated, **GPU-a40** is fine — GNN inference is not FLOP-bound here
- **Local Knative (subset):** `bash scripts_cosim/run_bipartite_knative_local.sh` (phase anchors k4/k6/k8 @ seek50)
- **Legacy reference:** `tiered_hub_gnn_mlp_125225_20260610/` (symmetric 5ms, includes k=2)

#### Handover block for on-cluster agent (always include in report)

Produce this verbatim (fill `<...>` placeholders):

```
[DATALAB COMPARE HANDOVER]
Repo: /home/nikola.lukic/gnn-herosim
Git: checkout <branch_or_commit> (mitrix pushed <hash>)
Env: eval "$(micromamba shell hook --bash)" && micromamba activate gnn
Workload: data/nofs-ids/traces/workload-100-100.json · seed 42
Output dir: simulation_data/normal_sim_sweeps/<sweep_dir>/results/
Policies: <list policies + model paths>
Configs: <standard5 | all7 | list>
Submit:
  <exact sbatch / bash commands>
Skip if exists: result JSON with workload-100-100 and total_rtt > 0
When done: ls <output_dir>/*.json and print total_rtt per file
Pull back (mitrix): bash scripts_cosim/transfer_reviewer_triangle_from_datalab.sh
```

If user wants **you** to rsync from mitrix now, run the transfer script(s) and report what was sent + the handover block. Do not SSH-submit SLURM unless user explicitly asks you to operate the remote shell.

### Analyze existing results
- Load `total_rtt` from result JSONs per `docs/notes/compare.md`
- Build a table: configs × policies (fixed benchmarks + any new dim14 experiments)
- For **new dim14 variants**: delta % vs **dim14-ce anchor** per config; flag BEAT/LOSE; sum 5-config gate
- Also report delta % vs Knative (and HRC / tabular when those result dirs exist)
- Note which sweep dirs exist; use `baseline_default_100100/results/` for baseline Knative/GNN dg-26 files

### Run new simulations (local only)

Skip this subsection when `mode=datalab` or `execution=datalab`.
- Use `pipenv run python3 scripts_cosim/run_simulation.py` with `--config`, `--workload`, `--output`, `--seed`, `--timeout`
- GNN: `export GNN_MODEL_PATH=...` and `export CUDA_VISIBLE_DEVICES=0`
- Regime A XGB: `--xgboost_batch --xgb-model models/tabular/batch_edge_ranker.json` (or `XGB_MODEL_PATH`)
- Regime A batched Knative: `--knative_network_batch`
- Do **not** mix Regime A batch policies with per-arrival `knative_network` in the same fair-fight table
- Multi-config: **prefer 5-config** via `scripts_cosim/important/run_gnn_near_rtt_v2_5cfg_sweep_common.sh`; full 7 via `run_normal_sim_config_sweep.py --policy <policy>` or nohup scripts in `scripts_cosim/important/`
- Bash scripts from Windows: `sed -i 's/\r$//' <script.sh>` before `nohup`
- Log progress; report RTT + wall time per config

## Step 3 — Report

- Markdown table: **total_rtt** per config × policy (lower is better)
- Separate **fixed benchmarks** (Knative, HRC, dim14-ce, tabular) from **new dim14 experiments**
- For experiments: BEAT/LOSE vs dim14-ce per config; 5-config sum gate; `default` regression check (>+2% = fail)
- One-line winner per config across all policies; overall win count
- If GNN involved: note model path and GPU vs CPU
- If runs in background: give log path and `tail -f` command
- If **datalab**: include the handover block, rsync commands run (or pending), `squeue`/log paths, and pull-back command

## Constraints

- Do **not** write new `.md` explanation docs unless user asks — report in chat
- Do **not** edit `LINEAGES.md` or any `docs/lineages/` node except to record a gate outcome
- Prefer reusing configs from `simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/`
- Real sim path ignores preinit from config (cold autoscale) — same for all policies
- **Default sweep size is 5 configs** (`standard5`); skip `00` and `04` unless user asks for `all7` — see `docs/notes/compare.md` § GNN Regime Analysis for rationale
- **dim14-ce is a fixed GNN anchor**, not the experiment under test — new dim14 checkpoints must beat it to ship
- **Datalab:** code via git; binaries via rsync (`transfer_gnn_dim14_ce_to_datalab.sh`, tabular models, workload); micromamba `gnn` on cluster; co-locate triangle legs in one sweep dir for fair compare

Proceed: read `@docs/notes/compare.md`, ask the user what to compare, then run, send to datalab, or analyze.
