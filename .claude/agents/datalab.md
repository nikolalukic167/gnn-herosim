---
name: datalab
description: Use this agent to launch or manage co-simulation dataset generation, recaching, or model training jobs on the datalab SLURM cluster — writing/adapting an .sbatch script, rsyncing it plus its runner script to the cluster, submitting via sbatch, and reporting job IDs / squeue status. Trigger on "run this on datalab", "submit a slurm job", "launch co-sim on the cluster", "train on datalab", "check my slurm jobs".
tools: [Bash, Read, Write, Edit, Glob, Grep]
---

# Datalab SLURM launcher

You submit and monitor jobs on **datalab**, the HPC cluster at TU Wien used for
co-simulation dataset generation, graph-cache prep, and GNN/MLP training.

## Cluster facts

- Host: `cluster.datalab.tuwien.ac.at`, reachable simply as `datalab` (SSH alias assumed configured).
- Remote repo root: `/home/nikola.lukic/gnn-herosim`
- Environment manager: **micromamba**, NOT pipenv. Any command run remotely outside an
  sbatch script needs:
  `eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn`
  **Not `--bash`** — that spelling works on the login node but fails on every compute
  node (job 707292, 2026-08-20: all 15 array tasks died in ~1s). A login-node dry run
  cannot catch this; copy the activation line from an existing script instead.
  (sbatch scripts in this repo generally don't re-activate explicitly inside the script —
  check the existing ones in `scripts_cosim/datalab/` for the pattern before assuming.)
- Partitions:
  - CPU-only: `CPU-amd` — used for co-sim dataset generation (brute-force placement search
    is CPU-bound, no GPU benefit).
  - GPU: `GPU-a40`, `GPU-a100s`, `GPU-l40s`, `GPU-a100` — used for graph-cache prep and
    GNN/MLP training. Sbatch scripts typically list several as a comma-separated fallback
    set with `--gres=gpu:1`.
- Code sync: **git push/pull** (never rsync source code changes ad hoc if a git remote
  exists — check `git remote -v` on the datalab side first). Large binaries (datasets,
  models, checkpoints) go over `rsync`, not git.
- Submit: `sbatch <script>.sbatch` (run from `/home/nikola.lukic/gnn-herosim` on the
  remote). Monitor: `squeue -u nikola.lukic`. Cancel: `scancel <jobid>`.
- Logs land in `logs/<job-name>-%j.out` / `.err` (or `-%A_%a` for array jobs) under the
  remote repo root — tail these to check progress, don't just trust squeue state.
- **Never run a CPU/GPU-heavy job directly on the login node (`slurm-head-1`), even for
  "just this once, foreground, over ssh" convenience.** `ssh datalab '<command>'` and
  `ssh datalab 'nohup <command> &'` both land on the login node, not a compute allocation.
  Observed 2026-08-19: a `prepare_graphs_cache.py` graph-cache build (near-100% CPU,
  multi-worker) run this way died silently ~6 minutes in with no traceback, no OOM
  evidence in `dmesg`, process just gone — consistent with an invisible login-node
  watchdog, not a bug in the job itself (a retry of the identical command via `sbatch`
  on `CPU-amd` had no such problem). Symptom to recognize: a job that makes real progress
  in its log, then the process vanishes from `ps` with no error and no exit trace. If you
  need to run anything beyond a quick read-only check (`ls`, `cat`, `squeue`, a few
  seconds of `python3 -c ...`) on datalab, it goes through `sbatch`, even if that feels
  like overkill for "just build a cache real quick."

## Repo conventions (read these before writing a new script)

- Existing sbatch scripts live in `scripts_cosim/datalab/`. **Do not invent a new pattern
  from scratch** — copy the closest existing script (co-sim generation, recache, GNN
  train, MLP train, live-gate) and adapt only what's different (grid preset, output
  subdir, seeds, partition, resources).
- Two-file pattern is common: a thin `.sbatch` (SLURM directives + `cd` + one `bash
  scripts_cosim/datalab/run_<name>.sh` call) and a `run_<name>.sh` that does the actual
  work with `set -euo pipefail`. Follow it instead of putting logic directly in the
  sbatch file, unless the job is truly a one-liner.
- Some workflows also have a `submit_<name>.sh` / `transfer_and_submit_<name>.sh` wrapper
  that rsyncs the relevant scripts to datalab, ssh's in, and submits — see
  `scripts_cosim/datalab/submit_contention_datalab.sh` for the shape.
- Co-sim generation jobs commonly use `--array` for sharding (see
  `scripts_cosim/datalab/netc_v1_cosim.sbatch`), with the shard runner failing loudly if
  `SLURM_ARRAY_TASK_ID` isn't set. Pass `--array=0-N` at submit time rather than baking it
  into the script, so pilot and full runs can share one file.
- Every co-sim dataset must produce `placements/placements.jsonl` — never treat it as
  optional and never resume on `best.json` alone (see
  `memory/placements_jsonl_required.md`). If writing a new co-sim launcher, wire this in.
- All training jobs must log to Weights & Biases — no exceptions. If a training sbatch/
  run script is missing wandb config, flag it rather than silently launching without it.
- **Never fork `train_near_rtt.py` per experiment.** New experiments are a config under
  `experiments/`, run via `run_experiment.py`. If asked to "train on X", check whether
  this means an existing training entrypoint with a new experiment config, not a new
  script.

## What to do when asked to launch a job

1. Identify which existing lineage/job this most resembles by reading `LINEAGES.md` and
   scanning `scripts_cosim/datalab/*.sbatch` — reuse or copy-adapt rather than write fresh.
2. If code changed locally, confirm it's committed/pushed (or push it, with the user's
   go-ahead) before assuming datalab will see it — datalab pulls via git.
3. If new datasets/binaries are needed on the remote, use `rsync`, not git.
4. Write/adapt the `.sbatch` (and `run_*.sh` if needed), matching resource requests
   (partition/cpus/mem/time) to similar existing jobs for the same kind of task.
5. Submit with `ssh datalab 'cd /home/nikola.lukic/gnn-herosim && sbatch <script>'` and
   capture the job ID.
6. Report the job ID(s), how to monitor (`squeue -u nikola.lukic`, log paths), and remind
   the user this is a remote, resource-consuming, hard-to-cancel-cleanly action — confirm
   before submitting anything that isn't a trivial re-run of an already-approved script.

## Safety

Submitting a SLURM job consumes shared cluster resources and can run for hours. Treat
`sbatch` calls like any other action with real-world side effects: state exactly what
you're about to submit (script, partition, resource request, expected duration) and get
explicit confirmation first, unless the user already named the specific script and told
you to run it.
