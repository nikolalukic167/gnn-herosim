# 🚀 Session Handover (2026-08-23, later session)

**Status:** All work committed and pushed on `feat/network-contention-v1` (`e1443e5`); datalab
synced at the same commit, clean tree both sides, **nothing in flight**. This session closed
the two open MLP questions from the previous handover — both answered decisively, both
recorded in `LINEAGES.md`.

> Read first: `LINEAGES.md` → search **"The MLP collapse is ARCHITECTURAL"** and **"no
> STRUCTURE"** (`link_contention_v1`, 2026-08-23). Both sit directly under the "GNN's edge is
> RELIABILITY" subsection from the prior session, which they answer rather than revise.

## 0. The one-paragraph story

Task A: the corrected-cache MLP (`tempfix`) was run as a fifth arm on the same 30 cells.
**Exactly 7 of 30 collapse under each checkpoint — same count, a different set of cells.**
Retraining relocated the failure instead of reducing it, which is the signature of an
architectural failure, not a training-data artifact. Across 120 scheduler runs (2 MLP × 2–3
GNN arms × 30 cells) **all 14 collapse events are MLP arms and none is a GNN arm** — the
reliability claim from the prior session hardens.

Task B: the collapse cells share **no structural property**. Adjacency is byte-identical
across all four cell sets and no degree/HHI/choice-set statistic separates collapse from
healthy; the hogging platforms don't have unusually short initial queues; the trace that
flips a cell is a different random draw of the same distribution, not a different shape; and
the victim set moves when only the weights move. What actually separates them is
**dispersal** — how widely the scheduler spread load — and there are two distinct mechanisms
under one symptom: platform-side packing (12 of 14 events) and link-side starvation (2 of
14, where every platform sits at 2–6% utilisation and RTT still blows up).

## 1. What this session did, in order

1. **Task A infra.** Added `mlptempfix` to `score_live_gate_matrix.py`'s `ARM_SUFFIX`;
   wrote `scripts_cosim/datalab/mlp_tempfix_arm_all_gates.sbatch` (byte-identical to the
   `mlp` template except checkpoint + `SWEEP_DIR`, plus a guard asserting the sweep dir
   isn't shared with the first arm). Smoke-tested array index 4 (the +147.4% collapse cell)
   before submitting the other 29 — jobs 710656 + 710657, all green, verified `dim22` /
   non-zero `total_rtt` / clean tree at `98b41e9`.
2. **Re-scored all three gates** with the new arm. Verdict JSONs gained an `mlptempfix`
   column: **132 insertions, 0 deletions** — the integrity check that no GNN or Knative
   number moved.
3. **Task B tooling**, both reading bounded byte ranges instead of parsing ~80 MB result
   files (safe to run on the login node, per datalab pitfall #2):
   - `scripts_cosim/important/extract_gate_stats_summary.py` — pulls `stats` scalars +
     the response-time curve + the `.decode_stats.json` sidecar into one 700 KB summary
     (`simulation_data/gate_stats_summary.json`, 150 results).
   - `scripts_cosim/important/extract_platform_dispersal.py` — pairs each
     `idleProportion` in `nodeResults` with its `platformId` to get per-platform
     utilisation (`simulation_data/platform_dispersal.json`, 150 runs). This is what
     told the two collapse mechanisms apart.
4. **Structural checks that came up empty**, each recorded in `LINEAGES.md` and
   `memory/herosim-mlp-collapse-has-no-structural-signature.md`: adjacency diff across
   the four cell sets, initial-queue rank of the platforms that get hogged, and a
   statistical comparison of the two workload traces that flip cell05.
5. **`LINEAGES.md`** — two new subsections under the RELIABILITY finding, closing both
   open questions with numbers.
6. **Memory** — `herosim-mlp-collapse-is-occupation-collapse.md` updated with the
   retraining result and the better detector; new file
   `herosim-mlp-collapse-has-no-structural-signature.md` for the dispersal finding.

## 2. Numbers worth keeping close

**Collapse survives retraining, relocated:**

| gate / condition | mlp | mlptempfix |
|---|---|---|
| drawgate, no backbone | +85.1% · 4/5 | +133.4% · 3/5 |
| drawgate, backbone | +2.5% · 4/5 | +12.8% · 4/5 |
| promo175, no backbone | +53.4% · 4/5 | +98.2% · 4/5 |
| promo175, backbone | −35.1% · 5/5 | +4.3% · 4/5 |
| bbrob core8/bw1.5 | +38.5% · 3/5 | +28.6% · 4/5 |
| bbrob core4/bw0.5 | +11.3% · 3/5 | +10.5% · 4/5 |

Head-to-head: `tempfix` (GNN) beats `mlptempfix` on 23/30 cells (vs 17/30 against the
original MLP); `deployed` on 16/30 (vs 13/30).

**The best collapse detector found this session:** `chosen_queue_vs_min` **p95** from the
`.decode_stats.json` sidecar — collapse 13,485–23,866, healthy 449–1,387 across all 120
scheduler runs, a 9.7× gap with zero overlap. Better than the occupation-ratio rule from the
prior session (still valid, ≤0.33× vs ≥0.41×, but only a 1.24× gap) because the sidecar is
2.5 KB and nothing large needs parsing.

**Why the link-side mechanism was invisible until now:** `averageCommunicationsTime` sits at
~16.7 ms across all 150 runs (0.016662–0.016668) regardless of whether the backbone is
binding — it never measured link queueing. The wait is taken *inside* the replica's serving
loop (`src/placement/infrastructure.py:1082`, `with self.node.fabric.pipe(...).request()`),
so it blocks the replica and surfaces as queue time instead:
`averageQueueTime / averageElapsedTime` is 0.9990–1.0000 in every one of the 150 runs.
`link_wait_total` / `linkWaitTime` are computed there and never serialized (gate-tools row,
2026-08-21) — still open, still a reporting-only change, and now has a concrete payoff: it
would separate the two collapse mechanisms directly instead of by inference from
`max_busy_pct`.

## 3. Framing consequence for the paper

The GNN's advantage here is not "it reads a topological property the MLP is blind to" — no
such property distinguishes these cells, and a reviewer could check that in one pass over
the infrastructure JSONs. It is that the GNN **disperses** load and the MLP does not, and
dispersal is what keeps a metastable queueing instability from igniting. Still a legitimate
graph-aware advantage (a pointwise scorer cannot condition on where its peers in the same
batch are going), but state it as a **dispersal / reliability** argument, not as "the GNN
exploits topology P" — the data does not support the latter phrasing.

## 4. Open threads, not started this session

- **Serialize `link_wait_total`.** Cheap, reporting-only, and now has a specific use: split
  the 2 link-side collapse events cleanly from the 12 platform-side ones instead of by
  `max_busy_pct` inference. Would also let the two mechanisms be quoted separately in the
  paper rather than folded into one "collapse" bucket.
- **Everything in section 3 of the previous handover is still true and still open**: the GNN
  win itself is unchanged (scope: one topology family, 20c/20s sparse); `tempfix` is still
  the promotion candidate, still not gated on `workload-125-225` / `workload-200-200`;
  re-serve any GNN checkpoint with `inference_feature_layout: null` before citing it;
  `topology_transfer_v1` still has no live cells at 60/80 servers; root cause of the
  additive co-sim target is still open; `main` is still far behind this branch, unmerged.

## 5. Environment gotchas

```bash
PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=/root/projects/my-herosim \
  pipenv run python3 scripts_cosim/<script>.py ...
# datalab: eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
# in an .sbatch: export HEROSIM_PY=python3 right after activation
# pin OMP/MKL/OPENBLAS/TORCH_NUM_THREADS=4 for ML runs
```

- Live-gate result JSONs are ~80 MB each; **do not parse them wholesale**. `stats` is not a
  small header object — the per-task records are nested *inside* it (`stats.tasks` opens at
  byte ~2400, closes at ~79.87 MB in a typical file). Read a bounded prefix and pull fields
  by name/regex, or bracket-match only the specific array you need
  (`extract_gate_stats_summary.py`, `extract_platform_dispersal.py` are the reusable
  versions of this).
- **Never `[[ -d X ]] || cp -r`** to stage shared files in a SLURM array — not atomic, killed
  3 tasks of job 710315. Use `mkdir -p` + `rsync -a`.
- Running `pipenv run` from outside the repo creates a **new empty venv** and fails with a
  confusing `ModuleNotFoundError`. Always invoke from the repo root.
- Grep **both** pipenv spellings when auditing the env leak: the shell form and `"pipenv"` in
  Python argv lists.
- **Read `run_provenance.python_env` from the result JSON**, not the sbatch banner — the
  banner describes the process that printed it.

## 6. Restore prompt for next session

```
[CONTEXT RESTORE] feat/network-contention-v1 is pushed and synced to datalab, nothing in
flight, at e1443e5. This session closed both open MLP questions from the prior handover.
(A) Retrained the MLP on the corrected batch cache as a fifth arm (mlptempfix): exactly 7 of
30 cells still collapse, but a DIFFERENT 7 -- the corrected cache fixed cell03 under bbrob and
broke cell03 on drawgate/nobackbone and cell05 on promo175/backbone. Retraining relocated the
failure instead of reducing it, so it's architectural: across 120 scheduler runs all 14
collapse events are MLP, none GNN. (B) The collapse cells share NO structural property --
adjacency is identical across all four cell sets, the hogged platforms don't have short
initial queues, and the trace that flips a cell is just a different random draw. What
separates collapse from healthy is dispersal, in two mechanisms: platform-side packing (12 of
14 events) and link-side starvation (2 of 14, cell03 under bbrob, where every platform sits at
2-6% utilisation and RTT still blows up because link wait is charged inside the replica's
serving loop and surfaces as queue time, not comms time). Read LINEAGES.md's two new
subsections ("MLP collapse is ARCHITECTURAL" and "no STRUCTURE") for the full numbers. Best
new detector: chosen_queue_vs_min p95 from .decode_stats.json separates all 120 runs with a
9.7x gap and zero overlap. Open thread: serialize link_wait_total (src/placement/
infrastructure.py:1082) to split the two mechanisms directly instead of by max_busy_pct
inference. Section 4 of HANDOVER.md lists everything else still open from before this session
(GNN promotion gating, topology_transfer_v1, the additive co-sim target, unmerged main).
```
