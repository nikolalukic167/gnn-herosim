# 🚀 Session Handover (2026-08-22)

**Status:** `fix/deferred-gate-fixes` is **merged, pushed, and synced onto datalab** — no
unmerged work, no in-flight jobs, nothing pending. This session's headline result: **live
quality of the GNN training pipeline is a draw-to-draw lottery**, discovered via a variance
control that nobody had run before. That finding **confounds** the corrected-cache retrain's
0/15 FAIL from earlier the same session — read both together, not the FAIL alone.

> Read first: `LINEAGES.md` → search "The variance control answers" (siv1_full_corpus
> section, 2026-08-22). Then the two entries directly above it (the tempfix FAIL, and the
> matched MLP gates) for the full arc.

## 0. The one-paragraph story

Picked up from the 2026-08-21 handover: merged `fix/deferred-gate-fixes` (84 tests green),
gated the corrected-cache GNN retrain on all three real traces — it **FAILED 0/15**, uniformly
1.06–1.63× Knative, despite *better* val acc (70.7% vs deployed's 66.3%). That inversion
prompted a control nobody had run: retrain on the **same cache/pipeline/seed as the deployed
checkpoint**, differing only by GPU/dataloader nondeterminism. The control is the deployed
model's in-distribution twin (val 66.8%, greedy regret identical to 4 decimals) and **loses
4/5 cells live** where the deployed draw won 5/5, by +5.8% to +35.7% per cell — against a
0.1–0.4% simulation noise floor. Two val-acc-identical checkpoints differ by up to 36% of live
`total_rtt`. This is now the load-bearing fact for every future gate in this repo: **a
single-checkpoint live-gate verdict is a claim about one training draw, not about the recipe
that produced it.** Also landed: the local `a4_wl200200` trace (800k events) completed —
MLP sweeps 5/5 with zero collapse, GNN 3W/2L vs Knative, settling the trace-dependence
question; and `topology_transfer_v1`'s first-ever deployable checkpoints (10 `.pt` +
`.contract.json` sidecars, 5 seeds × `{pointwise, gnn_base}`) are on disk on datalab.

## 1. Nothing is in flight — start clean

```bash
ssh datalab 'squeue -u nikola.lukic'   # expect empty
git -C /root/projects/my-herosim status --porcelain   # expect clean
```

Both true as of this writing. No jobs to check on, no branch to merge.

## 2. What is now established (safe to build on)

- **Deployed checkpoint** (`near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt`) stays deployed —
  it is the best live artifact on disk, described honestly as *"the draw that won"*, not
  "what the pipeline produces." Recorded margins vs Knative (unchanged from 2026-08-21):
  2W/1T/2L on `workload-125-225`, **5/5 W** on `workload-150-100` and `workload-175-100`,
  **3W/2L** on `workload-200-200` (new this session).
- **Live quality is a training-draw lottery** — the session's central finding. In-distribution
  metrics (val acc, greedy regret) have **zero** discriminating power over which draw wins
  live. See `memory/herosim-live-quality-is-a-training-draw-lottery.md`.
- **The corrected-cache (tempfix) retrain is confounded, not cleanly falsified.** Cache
  ordering held on every cell (deployed < control < tempfix), which is *suggestive* the
  corrected cache is worse, but with one draw per cache the cache effect isn't separable from
  draw luck at this variance. Do not cite the tempfix 0/15 as "the corrected cache loses" —
  cite it as "the corrected cache lost, in a regime where a single draw can lose by 30%+ for
  no reason at all."
- **The MLP is much more draw-stable than the GNN.** Matched MLP retrain on the same
  corrected cache: −6.6% to +2.6% on 9 healthy cells (essentially inert), and on the 3
  collapse-implicated cells it *re-rolls* which cell collapses rather than fixing or uniformly
  worsening the tail. The GNN's systematic, one-directional degradation under the cache fix
  is a real GNN-specific signature — separate from and layered on top of the lottery finding.
- **`a4_wl200200` (workload-200-200, 800k events) settles trace-dependence.** MLP 5/5 paired
  wins, zero occupation collapse anywhere (bounds the collapse story — long-duration traces
  collapse it, this shorter/higher-rps one never does). GNN is 3W/2L, losing specifically on
  the two densest cells (opposite of its 125-225 losses on the two sparsest).
- **`topology_transfer_v1` has deployable checkpoints for the first time ever** — 5 seeds ×
  {`pointwise`, `gnn_base`}, each with a `.contract.json` recording split/held-out-sizes/
  `serving_port` (the verified `mp_residual=True` three-module rename). On
  `~/gnn-herosim/models/topo_transfer_v1_ckpts/` on datalab, **not yet rsynced to local**, and
  its own eval JSONs (`simulation_data/topo_transfer_v1_phase4_ckpt_seed{42..46}.json`) are
  untracked on datalab. This is co-sim eval only — no live cells have been minted at 60/80
  servers yet.

## 3. Open threads, prioritized

1. **A multi-draw re-gate, if the tempfix-cache question is worth resolving.** The confound in
   §2 can only be broken by training ≥3 draws per cache (pre-fix vs tempfix) and gating all of
   them — expensive (each draw is a ~25 min GPU train + a 15-task gate) but it's the only way
   to get a recipe-level verdict instead of a draw-level one. Given the lottery finding, this
   arguably has a stronger claim on the next session's GPU budget than any other open item.
2. **`topology_transfer_v1` — the live-gate steps are the only ones left.** Checkpoints exist;
   next needed: (a) `rsync` the 10 `.pt` + `.contract.json` files to local or leave them on
   datalab and work there, (b) mint parity-verified live cells at 60 and 80 servers (no
   existing script does this — `make_full_corpus_siv1_gate_cells.py` is siv1-specific), (c)
   run the 15-task-per-size gate, (d) **given §1's finding, gate more than one seed per arm
   before drawing a conclusion** — a single `gnn_base_seed42.pt` live result would repeat this
   session's original mistake.
3. **Root cause still open: the additive, queue-dominated co-sim target.** Untouched this
   session. `logit_tied_rate ≈ 0.54` / `confident_worse_queue_rate ≈ 0.8` remain unexplained —
   the tempfix retrain ruled out the dims 9-11 bug as the cause (fixing it didn't help any
   trace), so whatever is capping the model's decisiveness is elsewhere in
   `graph_structure_physics`.
4. **A6 (`soft_combo` live retest) is still dead as specified** — unchanged from 2026-08-21,
   both checkpoints sidecar-less and feature-dim-mismatched. No new information this session.
5. **Untracked artifacts on datalab, not yet committed anywhere:** `simulation_data/topo_
   transfer_v1_phase4_ckpt_seed{42..46}.json` and the `scratch/` dir (ad hoc sbatch scripts
   written this session — `topo_xfer_v1_partial_gate_train.sbatch`,
   `fc_siv1_mlp_tempfix.sbatch`). None are load-bearing for anything already recorded in
   `LINEAGES.md`, but if a future session wants to reproduce this session's launches exactly,
   they're there rather than lost.

## 4. Environment gotchas (unchanged)

```bash
PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=/root/projects/my-herosim \
  pipenv run python3 scripts_cosim/<script>.py ...
# datalab: eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
# pin OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_NUM_THREADS=4 for ML runs
# ssh alias `datalab` works; the raw hostname with BatchMode=yes does not (key is agent-loaded)
```

One new trap hit and fixed this session: `sbatch --wrap="..."` swallowed the MLP retrain
silently (0 s FAILED, empty logs, no error text anywhere) — resubmitting as a proper `.sbatch`
file worked first try. Prefer a real sbatch file over `--wrap` for anything beyond a one-liner.
Also: two of the five named real traces (`workload-150-100.json`, `workload-175-100.json`)
had **never existed on datalab** before this session — every prior datalab gate silently only
ever used `workload-125-225.json`. Both are now rsynced and md5-verified; check before
assuming a trace is present on the cluster.

## 5. Restore prompt for next session

```
[CONTEXT RESTORE] fix/deferred-gate-fixes is merged and synced to datalab, nothing in flight.
This session's finding: GNN live quality is a training-draw lottery -- a retrain on the exact
same cache/pipeline/seed as the deployed checkpoint is its in-distribution twin (val acc
66.8% vs 66.3%) but loses 4/5 cells live by up to 36% of total_rtt. This confounds the
corrected-cache (tempfix) retrain's earlier 0/15 FAIL -- cache ordering (deployed < control <
tempfix) is suggestive but not separable from draw luck with n=1 per cache. The deployed
checkpoint stays deployed as "the draw that won." Also landed: a4_wl200200 settles
trace-dependence (GNN 3W/2L, MLP 5/5 with zero collapse), and topology_transfer_v1 has its
first-ever deployable checkpoints (10 .pt + .contract.json on datalab, untested live). Read
LINEAGES.md's "variance control answers" entry first, then HANDOVER.md §3 for priorities --
top candidate is a multi-draw re-gate to get a recipe-level (not draw-level) verdict on the
cache question, or minting live cells for the topo_transfer_v1 checkpoints.
```
