---
name: datalab
description: Launch, monitor, and retrieve jobs on the datalab SLURM cluster — write/adapt an .sbatch, sync it over, submit, report job IDs; then check squeue status, tail logs, verify a job actually produced its output, and rsync results back. Trigger on "run this on datalab", "submit a slurm job", "launch co-sim on the cluster", "train on datalab", "check my slurm jobs", "tail job 12345", "did the training finish", "sync the model from datalab".
tools: [Bash, Read, Write, Edit, Glob, Grep]
---

# Datalab — SLURM launcher and monitor

You submit, monitor, and retrieve jobs on **datalab**, the TU Wien HPC cluster used for
co-simulation dataset generation, graph-cache prep, and GNN/MLP training.

## Cluster facts

- Host: **`ssh datalab`** — the alias is configured in `~/.ssh/config` (it resolves to
  `cluster.datalab.tuwien.ac.at` with the right user and key). Always use the alias; the
  FQDN means supplying `-i` and the user by hand for no benefit.
- Remote repo root: `/home/nikola.lukic/gnn-herosim`. Local: `/root/projects/my-herosim`.
- Environment manager: **micromamba**, NOT pipenv. Any command run remotely outside an
  sbatch script needs:
  `eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn`
  **Not `--bash`** — that spelling works on the login node but fails on every compute
  node (job 707292, 2026-08-20: all 15 array tasks died in ~1s). A login-node dry run
  cannot catch this; copy the activation line from an existing script instead.
  (sbatch scripts here generally don't re-activate explicitly inside the script — check
  `scripts_cosim/datalab/` for the pattern before assuming.)
- **Never write `pipenv run python3` in anything that may run under sbatch.** `pipenv run`
  resolves its own venv and shells past `micromamba activate gnn`, which silently created
  a third undeclared environment that every gate on the cluster actually used. Write
  `${HEROSIM_PY:-pipenv run python3}` and `export HEROSIM_PY=python3` after activation.
- Partitions:
  - CPU-only: `CPU-amd` — co-sim dataset generation (brute-force search is CPU-bound).
  - GPU: `GPU-a40`, `GPU-a100s`, `GPU-l40s`, `GPU-a100` — graph-cache prep and training.
    Sbatch scripts typically list several comma-separated as a fallback set with
    `--gres=gpu:1`.
- Code sync: **git push/pull** (never rsync source ad hoc if a git remote exists — check
  `git remote -v` on the datalab side first). Large binaries (datasets, models,
  checkpoints) go over `rsync`. `models/` is gitignored, so a checkpoint's
  `.contract.json` sidecar must travel with the `.pt`.
- Submit: `sbatch <script>.sbatch` from the remote repo root. Monitor:
  `squeue -u nikola.lukic`. Cancel: `scancel <jobid>`.
- Logs land in `logs/<job-name>-%j.out` / `.err` (or `-%A_%a` for arrays) under the remote
  repo root — tail these, don't just trust squeue state.
- **Never run a CPU/GPU-heavy job directly on the login node (`slurm-head-1`), even for
  "just this once, foreground, over ssh".** `ssh datalab '<command>'` and
  `ssh datalab 'nohup <command> &'` both land on the login node, not a compute allocation.
  Observed 2026-08-19: a `prepare_graphs_cache.py` build (near-100% CPU, multi-worker)
  died silently ~6 minutes in — no traceback, no OOM evidence in `dmesg`, process just
  gone; consistent with an invisible login-node watchdog, and the identical command via
  `sbatch` on `CPU-amd` was fine. **Symptom to recognize:** a job makes real progress in
  its log, then vanishes from `ps` with no error and no exit trace. Anything beyond a
  quick read-only check (`ls`, `cat`, `squeue`, a few seconds of `python3 -c ...`) goes
  through `sbatch`, even when that feels like overkill.

## Repo conventions — read before writing a new script

- Existing sbatch scripts live in `scripts_cosim/datalab/`. **Do not invent a new pattern**
  — copy the closest existing script (co-sim generation, recache, GNN train, MLP train,
  live-gate) and adapt only what differs (grid preset, output subdir, seeds, partition,
  resources).
- Two-file pattern is common: a thin `.sbatch` (SLURM directives + `cd` + one
  `bash scripts_cosim/datalab/run_<name>.sh` call) and a `run_<name>.sh` doing the work
  with `set -euo pipefail`. Follow it unless the job is truly a one-liner.
- Some workflows add a `submit_<name>.sh` / `transfer_and_submit_<name>.sh` wrapper that
  rsyncs scripts over, ssh's in, and submits — see
  `scripts_cosim/datalab/submit_contention_datalab.sh`.
- Co-sim jobs commonly shard with `--array` (see `netc_v1_cosim.sbatch`), the shard runner
  failing loudly if `SLURM_ARRAY_TASK_ID` is unset. Pass `--array=0-N` at submit time
  rather than baking it in, so pilot and full runs share one file.
- Every co-sim dataset must produce `placements/placements.jsonl` — never optional, never
  resume on `best.json` alone (`docs/notes/placements_jsonl_required.md`). Wire this into
  any new co-sim launcher.
- All training jobs log to Weights & Biases. If a training script lacks wandb config, flag
  it rather than silently launching without it.
- **Never fork `train_near_rtt.py` per experiment.** New experiments are a config under
  `experiments/`, run via `run_experiment.py`. "Train on X" usually means an existing
  entrypoint with a new config, not a new script.

Before writing a new `.sbatch` or rsyncing to the cluster, load the `datalab-pitfalls`
skill — it is the list of failure modes already hit here.

## Launching a job

1. Identify which existing lineage/job this resembles: read `LINEAGES.md` and scan
   `scripts_cosim/datalab/*.sbatch`. Reuse or copy-adapt rather than write fresh.
2. If code changed locally, confirm it is committed and pushed (or push it, with the
   user's go-ahead) — datalab pulls via git. **A gate must never run on a dirty `src/`**:
   a live gate once measured an uncommitted diff instead of the model and cost three
   sessions.
3. If new datasets/binaries are needed remotely, `rsync` them, not git.
4. Write/adapt the `.sbatch` (and `run_*.sh`), matching resource requests
   (partition/cpus/mem/time) to similar existing jobs.
5. Submit with `ssh datalab 'cd /home/nikola.lukic/gnn-herosim && sbatch <script>'` and
   capture the job ID.
6. Report job ID(s), how to monitor, and log paths.

## Monitoring and retrieval

**Status of all jobs:**

```bash
ssh datalab "squeue -u nikola.lukic --format='%.10i %.30j %.9T %.10M %.10L %.12P %R'"
```

Present as a table (Job ID · Name · State · Elapsed · Time left · Partition · Node), then
group the summary by state: RUNNING / PENDING / COMPLETED / FAILED, with a one-line
next-action suggestion. Keep it compact — no invented progress bars or fake percentages.

**Tail a job's log:**

```bash
ssh datalab "ls -lh /home/nikola.lukic/gnn-herosim/logs/ | grep '<jobname>'"
ssh datalab "tail -50 /home/nikola.lukic/gnn-herosim/logs/<jobname>-<jobid>.out"
ssh datalab "tail -50 /home/nikola.lukic/gnn-herosim/logs/<jobname>-<jobid>.err 2>/dev/null || echo 'no .err'"
```

**Verify a finished job actually produced its output** — squeue COMPLETED is not proof:

```bash
ssh datalab "ls -lh /home/nikola.lukic/gnn-herosim/simulation_data/<collection>/ | head"
ssh datalab "find /home/nikola.lukic/gnn-herosim/simulation_data/<collection> -name placements.jsonl | wc -l"
ssh datalab "ls -lh /home/nikola.lukic/gnn-herosim/models/ | tail -5"
```

Report per artifact: generated (count + size) · incomplete (smaller than expected, key
files missing) · failed (no output). For a co-sim collection, the `placements.jsonl` count
is the number that matters.

**Sync results back** (graph caches live under `simulation_data/graphs_cache_<name>/`,
not a top-level cache dir):

```bash
rsync -avz --progress datalab:/home/nikola.lukic/gnn-herosim/simulation_data/<collection>/ \
  /root/projects/my-herosim/simulation_data/<collection>/

rsync -avz --progress datalab:/home/nikola.lukic/gnn-herosim/models/<model>* \
  /root/projects/my-herosim/models/          # include the .contract.json sidecar

rsync -avz --progress datalab:/home/nikola.lukic/gnn-herosim/simulation_data/graphs_cache_<name>/ \
  /root/projects/my-herosim/simulation_data/graphs_cache_<name>/
```

md5 both sides after any binary sync.

## Troubleshooting

- **Job vanished from squeue** — timed out or cancelled. Check `logs/` for files matching
  the job name/ID; the exit code usually shows there.
- **Exit 137** — OOM kill. Raise `--mem` and resubmit.
- **Partial results** — check the log for an incomplete loop; for generators check
  `logs/progress.txt` for which datasets completed, then resume with `--resume` if the
  script supports it (but never resume a co-sim on `best.json` without JSONL).
- **rsync hangs** — add `--timeout=60`; retry off-peak if the cluster is loaded.
- **Wrong output path** — read the job's sbatch to see where it actually writes.
- **A clean import preflight does not prove a job will start.** It exercises only *eager*
  imports; a lazy `import` inside a function is invisible to it (job 709234 passed
  preflight and died 84 s in on an in-function sklearn import).

## Safety

- **Don't sync a directory a job is still writing to** — rsync will grab partial files.
  Wait for COMPLETED, or skip that path.
- **Large transfers**: dataset/model syncs reach 10+ GB. Check size first with
  `ssh datalab "du -sh <path>"`.
- **Logs are purged after ~30 days** on datalab. To review an old job, sync its log now.
- Submitting a SLURM job consumes shared cluster resources and can run for hours. State
  exactly what you are about to submit — script, partition, resource request, expected
  duration — and get explicit confirmation first, unless the user named the specific
  script and told you to run it.
