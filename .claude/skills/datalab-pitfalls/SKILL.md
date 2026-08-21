---
name: datalab-pitfalls
description: Preflight checklist of real, previously-hit failure modes on the datalab SLURM cluster — silent login-node deaths, sbatch env-activation bugs that pass locally and die remotely, dirty-working-tree overwrites, and rsync/git sync traps. Load before writing a new .sbatch, before rsyncing local changes to the cluster, or whenever the datalab agent is about to submit a job. Not a how-to for submitting jobs (see the datalab agent) — this is what has already gone wrong.
---

# Datalab pitfalls

Every item here cost a real job, a burned array, or a silent multi-minute death. Each has
a one-line check that would have caught it before submission. Run through this before
`sbatch`, not after a job fails mysteriously.

## 1. `micromamba shell hook --bash` fails only on compute nodes

Compute nodes run a newer micromamba that requires `--shell bash` and rejects the old
`--bash` spelling. **The login node still accepts `--bash`** — so a sanity check run there
passes while every array task dies in ~1 second on submission.

```bash
eval "$(micromamba shell hook --shell bash)"   # correct, every script in scripts_cosim/datalab/ uses this
eval "$(micromamba shell hook --bash)"         # WRONG — passes on login node, dies on compute nodes
```

**Check before submitting:** `grep -n "micromamba shell hook" your_new.sbatch` and confirm
it says `--shell bash`. Better: copy the activation preamble from an existing script in
`scripts_cosim/datalab/` instead of writing it from memory — do not trust a login-node dry
run to validate this line, it structurally cannot.

Cost: job 707292 (`siv1_full_corpus` live gate, 2026-08-20), all 15 array tasks dead in ~1s.

## 2. Never run a CPU/GPU-heavy command directly on the login node

`ssh datalab '<command>'` and `ssh datalab 'nohup <command> &'` both land on the login
node (`slurm-head-1`), not a compute allocation — even "just this once, foreground" or
"just a quick cache build."

**Symptom:** the job makes real progress in its log, then the process vanishes from `ps`
with no traceback, no OOM in `dmesg`, no exit trace — consistent with an invisible
login-node watchdog killing anything resource-heavy.

**Check before running anything remotely beyond `ls`/`cat`/`squeue`/a few seconds of
`python3 -c ...`:** it goes through `sbatch`, full stop. If it needs a CPU/GPU allocation
for real work, it is not a login-node command.

Cost: `prepare_graphs_cache.py` (near-100% CPU, multi-worker) died silently ~6 minutes in,
2026-08-19. Identical command via `sbatch` on `CPU-amd` had no problem.

## 3. Before overwriting a script on the cluster, diff it — don't assume local is newer

Both the local checkout and the cluster checkout can carry independent uncommitted edits
on top of the same commit. Blindly rsync-overwriting a script the cluster side has also
been editing silently drops whichever side loses.

**Check before rsyncing any file that isn't purely new:**
1. `ssh datalab 'md5sum <path>'` vs local `md5sum <path>` — if they already match, skip it.
2. If they differ, `ssh datalab 'git diff HEAD -- <path>'` (or fetch the remote copy and
   `diff` locally) to see what the cluster side actually changed, not just that it changed.
3. Enumerate lines present on the cluster side but *not* present anywhere in the local
   version. If that set is empty, local is a strict superset and the overwrite is safe —
   say so explicitly rather than assuming. If it isn't empty, stop and reconcile before
   overwriting; don't guess which side is "right."
4. Take a timestamped backup on the remote before overwriting
   (`cp -r target target.pre_<change>_<date>/`) even after step 3 says it's safe — it's
   cheap insurance against a diff blind spot.

This is not hypothetical caution: this exact shape (two independent uncommitted trees,
same base commit) has already cost a burned SLURM job once
(`gnn_necessity_ablation.py`, 2026-08-20 — see `LINEAGES.md` GATE TOOLS) from a merge that
picked one side as "the base" without diffing first.

## 4. A file's import can be untracked and missing from your rsync list

`git ls-files` won't surface a file someone created but never `git add`ed — and a module
you just wrote can import one of those without you noticing, because it works locally
where the file already exists. If it's absent on the cluster, the job dies with
`ImportError` on task 0 (or worse, on task 14 after 13 succeeded).

**Check before finalizing an rsync list:** for every new `.py` file being shipped, grep
its own `import` lines and confirm every local module it pulls in is either already on the
cluster or also in the transfer list — including ones `git status` doesn't show as
tracked. Don't rely on "it ran locally" as proof the import list is complete.

## 5. Code sync is git push/pull; only binaries go over rsync — ad hoc rsync of source is an exception, not the default

The repo convention is: source changes go to datalab via `git push` (local) /
`git pull` (remote), so there's a single commit history both sides agree on. Large
binaries (models, datasets, checkpoints) go over `rsync`, never git.

Rsyncing *source* files ad hoc (as in an urgent same-session fix) is sometimes the right
call when there's nothing to commit yet, but it means the two trees are no longer
git-synchronized — which is exactly the setup that makes pitfall #3 possible. If you rsync
source ad hoc, say so out loud and prefer committing + pushing the moment it's safe to,
so the next sync doesn't have to re-diff by hand.

## 6. Multi-GB result files: leave them on the cluster

Live-gate / large-sweep result JSONs can run ~100-150MB *each* — a modest gate (a handful
of cells × a few policies) is easily 1-2GB. Don't reflexively rsync results back to local;
score them on the cluster (or from a small extracted summary) and only pull back what's
actually needed for local analysis.

## 7. SLURM array jobs that share a generated artifact: elect one task to build it, make the rest wait

If N array tasks all need the same generated file (e.g. minted test cells, a shared
config) and each task tries to generate it independently, they race and can corrupt or
duplicate work. Pattern: have `SLURM_ARRAY_TASK_ID == 0` build it if absent, and every
other task poll (`sleep`-loop with a cap, then fail loud if it never appears) rather than
regenerating it itself.

## 8. `pipenv run` inside an sbatch silently ignores `micromamba activate`

`micromamba activate gnn` followed by `pipenv run python3` does **not** run in the `gnn` env.
`pipenv run` resolves its own project venv from `Pipfile` and execs that interpreter, shelling
straight past the activation. Worse, if no venv exists yet, pipenv *creates* one — so the
cluster silently grew a third, unmanaged, undeclared environment and every live gate that has
ever run there used it.

The `.sbatch` looks correct in review, because the activation line is right there. The
substitution happens one level down, inside the `.sh` the sbatch calls.

```bash
# in any .sh that may be invoked under sbatch
${HEROSIM_PY:-pipenv run python3} scripts_cosim/run_simulation.py ...   # correct

pipenv run python3 scripts_cosim/run_simulation.py ...                  # WRONG — ignores the activated env
```

```bash
# in the .sbatch, immediately after activation
micromamba activate gnn
export HEROSIM_PY=python3
```

**Check before submitting:** `grep -rn "pipenv run" <the .sh files your .sbatch calls>` — not
just the `.sbatch` itself, which is where reviews stop looking. Then prove which interpreter
actually served: `python3 -c "import sys,torch;print(sys.executable, torch.__version__)"`
through the *same call chain* the job uses, not from an interactive shell.

Cost: the cluster ran `torch 2.12.0+cu130` while CLAUDE.md, the sbatch header and `LINEAGES.md`
all asserted the `gnn` env's `torch 2.5.1+cu121`. Three sessions spent attributing an 11-26%
GNN gap to the environment. (Measured 2026-08-21: the version delta contributes *exactly zero*
— the gap was a code bug all along. See `PARITY.md`. The pitfall is that nothing recorded which
interpreter served, so it took three sessions to rule out.)
