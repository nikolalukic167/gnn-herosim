# siv1_full_corpus — ACTIVE

> **Status:** `ACTIVE` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-20 → 2026-08-21

**Outcome.** First real live-gate FAILED, then **SUPERSEDED the same day** — it measured an uncommitted code diff, not the model. The synced-code re-gate (job 709163) wins **5/5 on `workload-150-100` and `workload-175-100`**, 2W/1T/2L on `workload-125-225`. Corrected-cache retrain (job 709234) is ungated.

**Entry points:** `scripts_cosim/datalab/full_corpus_siv1_{recache,gnn_train,mlp_train}.sbatch` → `run_full_corpus_siv1_*.sh`

**Datasets:** whole `legacy_v0_node_disk_v2_4task` group

**Related:** [mp_parity](mp_parity.md) · [cache_live_divergence_audit](cache_live_divergence_audit.md) · `queue_feature_contract` (index only)

## Standing (from the index table)

Trains on the full corpus under `scale_invariant_v1`. GNN: `src/notebooks/train_near_rtt.py`. MLP: `src/policy/tabular/train_mlp_dim22_from_batch.py`. Recache: `src/notebooks/prepare_graphs_cache.py`. **Outcome 2026-08-17 — see `mp_parity` below.** **First live-gate on a real trace: FAILED (2026-08-21)** — `datalab/full_corpus_siv1_live_gate.sbatch` (array 0-14 = 3 policies × 5 cells) → `important/run_full_corpus_siv1_live_gate.sh`, cells minted by `important/make_full_corpus_siv1_gate_cells.py`, parity by `scripts_cosim/verify_live_infra_parity.py`, scored by `important/compare_sealed_live_holdout.py`. GNN loses to Knative on all 5 cells after fixing a `PYTHONHASHSEED`-dependent tie-break bug that had made the first attempt look like a sparse-topology win. See "the first real live-gate" below. **SUPERSEDED 2026-08-21 (same day) — do not cite the FAIL.** It was measured through a train/serve-divergent live feature path: the uncommitted dims 9-11 fix was absent on datalab. The **formal synced-code re-gate (job 709163, 15/15 COMPLETE)** reproduces the local re-grading on every cell to within +0.03%/+0.40%: **2W/1T/2L on `workload-125-225`**, and the same checkpoint **wins 5/5 on `workload-150-100` and 5/5 on `workload-175-100`**. `gnn/cell01` went 65.8M → 50.6M on a code change alone. Two follow-ups now open: the MLP's catastrophic tail is root-caused (occupation collapse, not collisions), and the deployed checkpoint is trained on a cache that disagrees with its serving features on **31.7% of platform rows** — corrected-cache retrain is job 709234, ungated. See the resolution subsection below.

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [siv1_full_corpus — re-testing the FAIL against untested real traces (2026-08-21, 🔄 IN PROGRESS)](#siv1-full-corpus-re-testing-the-fail-against-untested-real-traces-2026-08-21-in-progress)
- [siv1_full_corpus — the first real live-gate, and the infra-parity gap it closed (2026-08-20)](#siv1-full-corpus-the-first-real-live-gate-and-the-infra-parity-gap-it-closed-2026-08-20)

---

### siv1_full_corpus — the first real live-gate, and the infra-parity gap it closed (2026-08-20)

**Why this lineage went first.** Of the ACTIVE lineages, only this one is *deployable
today*: `models/near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt` exists with a contract
sidecar, is plain bipartite (no MP, no network entities), and loads through the production
`src/policy/gnn/scheduler.py` path with no porting work. `topology_transfer_v1` — the
lineage that most loudly asks for a live gate — cannot take the step: it has **zero saved
checkpoints**, no production code path for `use_network_entities`, and an already-`FAILED`
co-sim gate (see its scope caveat above). `link_contention_v1` and `contention_v4_v5` are
`FALSIFIED`.

**The gap this exposed: nothing tied a checkpoint to the infrastructure it trained on.**
`src/executesimulation.py` never loads a co-sim `infrastructure.json`; it *regenerates*
topology from a space config + seed (`prepare_infrastructure_for_real_simulation`). Three
concrete consequences, all measured, none previously recorded:

1. **The existing sealed-holdout gate was out-of-distribution and said nothing.**
   `important/run_contention_v2_873_sealed_holdout.sh` runs 40/40 p50 and 50/50 p60 configs;
   the corpus is **20/20** on every one of its 2,651 datasets. Now warns loudly with the
   numbers.
2. **`--seed` overrides the *topology* seed** (`executesimulation.py:842-854`), so a seed
   sweep is a topology sweep. Measured against `contention_v2/ds_00000`: at `--seed 42`,
   **144/210 directed edges differ and 64/64 shared edges disagree on latency.** The new
   gate passes no `--seed`; replication comes from distinct verified cells.
3. **`TOPOLOGY_FEATURE_CONTRACT` was enforced nowhere in `src/`.**
   `require_matching_topology_feature_contract` existed with only test callers, so a
   `size_invariant_v1` checkpoint served `src_index_v0` dim-2 features would have failed
   silently. Now raises, as does a warmth-physics mismatch and a feature-layout mismatch.

**The one legitimate divergence, characterized.** Live topology is always a strict
*subgraph* of the corpus topology, differing only by the replica-reachability repair
`generate_infrastructure.py` applies after replica placement (step 2b, ~lines 712-778) —
edges a live run cannot reproduce because it autoscales from zero and has no replicas.
Every such edge is client↔server at exactly `base_latency`, and zero live-only edges exist.
**Its size scales with sparsity**, which is worth knowing before choosing gate cells:

| connection_probability | repair edges (directed) |
|---|---|
| 0.15 | 34/174 (**19.5%**) |
| 0.20 | 12/172 (7.0%) |
| 0.25 | 4-14/182-226 (1.8-7.7%) |
| 0.35 | **0/282** |
| 0.50 | **0/380** |

At p=0.15 nearly a fifth of the corpus's edges do not exist live. That is not a bug, but a
model gated only at p=0.15 would be evaluated on a materially sparser graph than it trained
on. The gate spans p=0.15-0.50 deliberately so this is visible rather than averaged away.

**Cells: minted, not selected.** All 52 datasets on disk but absent from the training cache
share their exact topology cell with a trained dataset (**0/52 unseen**, across 268 distinct
trained cells) — they were dropped for data quality (`exclude_bad31`), not held out. So the
gate mints 5 fresh cells instead: 20/20 `sparse`, connection probabilities drawn from the
corpus's own six values, topology seeds 9001-9005 (the corpus used 142 seeds, max 609).
In-distribution on every axis the sidecar declares, memorized on none.

**Checkpoint sidecars now carry corpus provenance.** New
`src/placement/corpus_provenance.py` derives it from the cache's own `dataset_ids` rather
than a hand-written constant, so the record cannot drift from the data. Single-valued axes
are emitted as scalars, genuinely multi-valued ones as `<field>_values` sets tested for
membership (the full corpus spans six connection probabilities — equality would be wrong).
`train_near_rtt.py` writes it going forward; the three `full-corpus-siv1` sidecars were
backfilled through the same function. Note `models/` is gitignored and moves by rsync, so
these sidecars are **not** version-controlled and can drift between machines — the mitigation
is that `derive_corpus_provenance(cache_metadata)` regenerates them exactly from the cache,
so a suspect sidecar can always be re-derived rather than argued about.

Tools: `scripts_cosim/verify_live_infra_parity.py` (+ `test_live_infra_parity.py`, 13 tests
including a control for each fatal class and for the `--seed` divergence).

**Outcome (2026-08-21): FAILED — GNN loses to Knative on all 5 cells.** First gate attempt
(job 707307, 2026-08-20) reported GNN beating Knative by 17-26% at the two sparsest cells
(p=0.15, p=0.20) and losing narrowly elsewhere — a pattern that looked like real
sparse-topology structure worth investigating. It was not: it was measurement noise.

**Root cause: unpinned `PYTHONHASHSEED` made every placement tie-break non-reproducible.**
`system_state.replicas` (and related node/hardware pools) are built as Python `set()`s in
`src/policy/{knative,knative_network,gnn}/orchestrator.py`; the schedulers then resolve ties
— least-connected `min()`, most-available `max()`, first-match hardware iteration, and the
candidate order fed into the GNN's argmax decode — by iterating those sets directly. Python
randomizes string/object hashes per process by default, so **the exact same
config+workload+policy produces a different `total_rtt` every time it's run in a fresh
process**, with nothing to do with pipenv vs. micromamba. Measured directly: four separate
runs of `knative/cell01_p25_s9001` (identical inputs, only `PYTHONHASHSEED` varying) gave
`total_rtt` of 41.5M / 42.7M / 47.9M / 64.8M — a 56% spread from tie-break luck alone, the
same order of magnitude as the "GNN wins" margin the first gate reported.

**Fix**: sort every such tie-break deterministically on `(node.id, platform.id)` (or plain
`sorted()` for hardware-type strings) at the point a set feeds a decision — 6 files:
`{knative,knative_network,gnn}/scheduler.py` and `{knative,knative_network,gnn}/autoscaler.py`.
Also pinned `PYTHONHASHSEED=0` in the live-gate scripts as defense-in-depth. Verified: two
different `PYTHONHASHSEED` values now produce bit-identical `total_rtt` on the fixed code,
both locally (pipenv) and on datalab (micromamba) — the two independently landed on the exact
same value (`46,556,946.73649`) for `knative/cell01`, closing the pipenv/micromamba
cross-check that first surfaced the discrepancy.

**Re-gated (job 708549, 2026-08-21, all 15/15 COMPLETED, 0 failures) — clean result, no more
noise:**

| cell (density) | knative | mlp | gnn | gnn vs knative |
|---|---|---|---|---|
| cell01 (p=0.25) | 46.6M | 219.6M | 65.8M | +41.4% worse |
| cell02 (p=0.35) | 39.5M | 27.1M | 50.5M | +27.9% worse |
| cell03 (p=0.15) | 43.9M | 259.6M | 61.5M | +40.0% worse |
| cell04 (p=0.50) | 34.9M | 27.0M | 42.7M | +22.1% worse |
| cell05 (p=0.20) | 46.9M | 374.1M | 52.6M | +12.0% worse |

`compare_sealed_live_holdout.py` (paired, single seed per cell): **GNN wins 0/5 cells**
against Knative (Knative 3/5, MLP 2/5); GNN also loses on p90/p99 elapsed-time at every cell
where it isn't already losing on `total_rtt`. No sparse-topology advantage survives the fix —
the density-dependent pattern in the first attempt was entirely tie-break noise. MLP's
earlier collapse at low density (4.7-8.0x Knative at p=0.15/0.20/0.25, ~0.7x at p=0.35/0.50)
is unaffected by this bug and stands as a separate, real finding.

> **QUALIFIED 2026-08-21 — the reasoning above was unsound, but the conclusion survives
> re-measurement.** The dims 9-11 estimator bug lives in `build_inference_feature_bundle`,
> which serves the **MLP path as well as the GNN path**; only Knative never touches it. So
> "unaffected by this bug" was an assertion with nothing behind it — the MLP column was
> measured through exactly the same divergent features as the GNN column.
>
> Job 709163 re-ran these cells with the fix in place, same trace, same cells, same
> (leaked) environment — a clean code-only A/B. The MLP barely moved:
>
> | cell | 708549 (pre-fix) | 709163 (post-fix) | Δ |
> |---|---:|---:|---:|
> | cell01 (p=0.25) | 219.6M | 223,741,718 | +1.9% |
> | cell02 (p=0.35) | 27.1M | 27,037,296 | −0.2% |
> | cell03 (p=0.15) | 259.6M | 262,234,499 | +1.0% |
>
> (Knative reproduced bit-identically at cell01: 46,556,947 both runs.) So the collapse is
> **real and not an artifact of the bug** — but note the asymmetry that needs explaining: the
> same fix moved the GNN by −23.3% and the MLP by ~1%. The likely reading is that the MLP's
> fitted weights put little mass on dims 9-11 while the GNN's decode is highly sensitive to
> them; that is a hypothesis, not a measurement. cell04/cell05 MLP arms were still running
> when this was written. See also the partial walk-back later in this file ("A related MLP
> finding that corrects prior speculation"), which reached a compatible conclusion from the
> trace side.

**This is `siv1_full_corpus`'s first live gate on a real workload, and it FAILED.** The
deployable checkpoint (`near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt`) does not beat Knative
live. Per `topology_transfer_v1`'s entry above, an ablation-only or offline-greedy result is
not a substitute for this step — this *is* the step, and the checkpoint does not clear it.

### siv1_full_corpus — re-testing the FAIL against untested real traces (2026-08-21, 🔄 IN PROGRESS)

**Why.** The FAIL above rests on exactly one trace, `workload-125-225.json`. Five of this
repo's six lineage-level `FALSIFIED`/`FAILED` verdicts were decided entirely on 4-task
synthetic co-sim snapshots; this is the only one ever gated on a real trace, and n=1 trace is
the entire evidence base for its headline verdict. `data/nofs-ids/traces/` has 5 named
full-scale traces that have never been used in any gate
(`workload-{100-50,150-100,175-100,200-200}.json`, `workload-150-100-30k.json`) — this reuses
`run_full_corpus_siv1_live_gate.sh`/`.sbatch` unmodified (both already take `WORKLOAD` as an
env var) against the same 5 parity-verified cells and the same deployed checkpoint.

**⚠ Preliminary reversal found — GNN beats Knative 5/5 on `workload-150-100.json`, not 0/5.**
301,352 events (rps=150, dur=100), same cells, same checkpoint, run locally (pipenv, not
datalab), `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=4` pinned:

| cell (density) | knative | mlp | gnn | gnn vs knative | recorded on 125-225 |
|---|---:|---:|---:|---:|---:|
| cell03 (p=0.15) | 27,287,566 | 20,596,204 | 23,260,591 | **−14.8%** | +40.0% |
| cell05 (p=0.20) | 26,455,031 | 161,323,506 | 23,047,726 | **−12.9%** | +12.0% |
| cell01 (p=0.25) | 24,471,537 | 22,206,010 | 24,252,626 | **−0.9%** | +41.4% |
| cell02 (p=0.35) | 24,300,898 | 16,637,375 | 21,176,714 | **−12.9%** | +27.9% |
| cell04 (p=0.50) | 20,426,553 | 16,520,619 | 19,334,793 | **−5.3%** | +22.1% |

Margins 0.9–14.8%, against a measured local noise floor of 0.05% (see GATE TOOLS below) — 18×
to 296× the noise. MLP beats Knative 4/5, also a reversal from the recorded 2/5.

**Environment ruled out as the explanation — replication control PASSED exactly.** Re-ran the
*recorded* trace (`workload-125-225.json`) locally on the same cells as a control, since the
recorded gate ran on datalab/micromamba and this session runs on pipenv/local. Knative
reproduced the recorded numbers to 3 significant figures on all 5 cells, and matched the one
full-precision value `LINEAGES.md` already quotes (the pipenv/micromamba cross-check for
`knative/cell01`) **exactly**: `46,556,946.73649` both times. So the local harness is
validated; the reversal on `workload-150-100` is not a pipenv-vs-datalab artifact. The
control's GNN arm — the model-dependent half of the check — was still running when this entry
was written; the reversal above should be read as preliminary until that lands.

**What this would mean if it holds.** `workload-150-100` differs from `workload-125-225` in
both rps (150 vs 125) and duration (100 vs 225), so this shows trace-dependence without yet
isolating which property drives it. `workload-175-100.json` (351,767 events, rps=175,
duration=100 — holds duration fixed against `workload-150-100`) is queued to separate rps
from duration. `workload-200-200.json` (800,413 events — the trace this file has named three
times as an aspirational target and never run) is queued last, at reduced parallelism (GNN
measured ~2.9 GB RSS/worker on the 301k-event trace, so PAR was cut from 5 to 2 for the 800k
one to avoid OOM on a 32 GB box).

**Do not treat the FAIL above as retracted yet.** One trace reversing does not overturn a
result — it demonstrates the result is trace-dependent, which is itself the finding worth
having either way. If the GNN control arm also passes and `workload-175-100` /
`workload-200-200` corroborate, the honest updated claim is "the deployed siv1 checkpoint
beats Knative on some real traces and loses on others" — a materially different, more
interesting result than an unconditional FAIL, and one that would need its own root-cause
work (what about `workload-125-225` specifically makes Knative win there) before either verdict
is final. If they contradict `workload-150-100` instead, the original FAIL stands and
`workload-150-100`'s result becomes the thing needing an explanation.

**A related MLP finding that corrects prior speculation.** The recorded gate's open thread
(this file, "MLP's earlier collapse at low density") does not reproduce as a density effect on
`workload-150-100`: the *sparsest* cell (p=0.15) is fine at 0.75× Knative; only p=0.20 collapses
(6.10×); MLP beats Knative in 4/5 cells. The `decode_stats.json` sidecars (previously unread —
see GATE TOOLS) show the collapsing cell's `chosen_queue_vs_min` **median** is unremarkable
(119, in line with the other cells' 84–147) while its **p95 is 15,401**, ~30× the other cells'
~500 — a rare catastrophic-choice tail, not a systematically worse policy. Collision rates are
flat across all 5 cells (0.146–0.179), ruling out intra-batch collisions. Which cell exhibits
the tail moved between the two traces (p=0.15/0.20/0.25 on `workload-125-225`, only p=0.20 on
`workload-150-100`), so this reads as a trace-dependent instability, not a property of sparse
topology — revise the "sparse topologies force remote placements" hypothesis accordingly.

**Group A retest scope, decided this session.** `LINEAGES.md` records six lineage-level
`FALSIFIED`/`FAILED` rows; this siv1 retest and a `link_contention_v1` real-trace A/B (below)
are the two live-gate-shaped ones. The other four were evaluated for the same treatment and
found not to fit:
- **`contention_v4_v5`** — its failure is a ratio between two terms of the ECT cost formula
  (`depth × exec_time` grows with the lever, `added_in_batch × exec_time` does not), provable
  by reading `scheduling_cost.py` and confirmed on all 899 `contention_v2` datasets. A live
  trace reports `total_rtt`, which has no additive-R² to report — not retestable this way.
- **`topology_transfer_v1`** — is live-gate-shaped in principle (see its own §a/b/c list
  above) but needs checkpoint saving, a production `use_network_entities` serving path, and a
  multi-size GPU retrain (~14 GPU-hours at job 705834's pace) before a live cell can even be
  minted. Blocked this session on GPU availability (none locally; needs datalab).
- **`regime_b`** (RETIRED) — its 5 on-disk checkpoints have no `.contract.json` sidecar, so
  `load_gnn_model`'s warmth-physics guard (`executesimulation.py:486`) silently no-ops instead
  of raising, and its only real traces are `rps=0` cold-burst files (28–64 events). Also
  superseded per CLAUDE.md.
- **`soft_combo`** (RETIRED) — its matched checkpoint pair
  (`near-rtt-v2-regime-b-oracle-split-cosim-dim16-{ce-only,soft-combo-conc}.pt`) inherits every
  `regime_b` problem above, and its training collection has no `space_with_network.json` to
  mint parity-verified cells from at all.

#### Resolution (2026-08-21, later the same day): the replication control FAILED on its GNN arm — and the cause is an uncommitted code fix, not the trace and not the environment

**The control's knative arm passed bit-exactly; its GNN arm did not reproduce on any cell.**
Local re-run of the recorded trace (`workload-125-225.json`), same cells, same checkpoint
(sha256 `4df64b6a…` verified identical local↔datalab), same pinned env:

| cell (density) | recorded gnn (datalab) | local gnn | recorded margin vs kn | local margin vs kn |
|---|---:|---:|---:|---:|
| cell01 (p=0.25) | 65,822,323.78 | 50,407,465.33 | +41.4% | +8.3% |
| cell02 (p=0.35) | 50,469,878.45 | 37,358,224.81 | +27.9% | **−5.3% (win)** |
| cell03 (p=0.15) | 61,505,759.92 | 51,550,526.55 | +40.0% | +17.4% |
| cell04 (p=0.50) | 42,661,515.61 | 32,269,908.29 | +22.1% | **−7.6% (win)** |
| cell05 (p=0.20) | 52,580,830.03 | 46,905,755.20 | +12.0% | −0.06% (tie) |

**Root cause isolated by a 7-run probe matrix on `gnn/cell01`** (every value below is the
same cell, trace, checkpoint, and thread pinning unless noted):

| run | code | machine / device | total_rtt |
|---|---|---|---:|
| recorded gate (job 708549_10) | datalab `4db48d9` (clean) | datalab GPU (A40) | 65,822,323.78 |
| GPU repro ×2 (jobs 709154_0/1, same node) | datalab `4db48d9` | datalab GPU (A40) | 65,849,376.41 / 65,865,441.89 |
| CPU-forced (job 709155, `CUDA_VISIBLE_DEVICES=`) | datalab `4db48d9` | datalab CPU-amd | 65,806,356.23 |
| **clean worktree at `4db48d9`** | **committed tree** | **local CPU** | **65,795,161.49** |
| local working tree (repl + exact re-run) | uncommitted diff | local CPU | 50,407,465.33 / 50,358,532.75 |
| local working tree, `OMP_NUM_THREADS=1` | uncommitted diff | local CPU | 50,518,223.26 |

Two tight clusters, 23.3% apart, split **exactly on the code version**: committed tree
65.80–65.87M (spread 0.11% across two machines, two devices, two BLAS thread counts, two
numpy/pyg versions — numpy 1.26.4/pyg 2.7.0 on datalab vs 2.3.0/2.6.1 locally); working
tree 50.36–50.52M (spread 0.32%). Environment, GPU-vs-CPU numerics, CUDA scatter-add
atomics, and library versions are all **exonerated** — GPU run-to-run wobble is ±0.04%,
thread-count wobble ±0.2%, both the same order as the residual F1 tie-break noise
(0.05–0.1%, see GATE TOOLS), and none of it cascades.

**The responsible diff is the dims 9-11 live-feature fix that was never committed.**
`src/placement/temporal_features.py` (in the working tree since 2026-08-19, untracked) and
the `feature_builder.py` hunk that calls it fix the audit's Bug 1 (estimate gated per
snapshot vs per platform) and Bug 2 (live estimate averaged over ALL task types incl. the
9.5×-outlier `cnn`, serving 0.0815 where the vocab-restricted training-side formula gives
0.0086). Job 708549 ran on datalab's clean tree — i.e. **the recorded FAIL was measured
serving the known-divergent live features the audit had already flagged, two days after
those features were fixed locally.** Everything else in the working-tree diff is
eliminated by the knative arm's bit-exactness (`46,556,946.73649` reproduced exactly
across both trees and both machines — the shared physics/simulation path is provably
unchanged) plus code review of the remaining GNN-path hunks (`gnn_model.py` changes are
opt-in-gated or behavior-identical refactors; `topology_features` v0 is bit-exact by its
own 40/40 verification).

**Consequences.**
1. **The recorded 0/5 FAIL is re-graded, not merely trace-dependent**: it measured the
   checkpoint through a train/serve-divergent live feature path. With train-consistent
   features (the fixed path — strictly closer to what the checkpoint saw in training,
   where the affected cache rows were 0.0, not 0.0815), the same checkpoint on the same
   trace goes **2 wins / 1 tie / 2 losses**, and wins **5/5 on `workload-150-100`** and
   **5/5 on `workload-175-100`** (kn 34.19/33.21/35.26/31.11/36.63M vs gnn
   30.99/30.67/32.63/27.77/32.67M — margins −7.5% to −10.8%; MLP wins 4/5 by −19 to −30%
   but collapses at cell05, +366%). The trace-dependence that remains is real but mild:
   `workload-125-225` is the hardest of the three traces for the GNN, not a 0/5 outlier.
2. **The audit's "decision impact is unmeasured" thread is closed**: measured, it is
   **23.3% of live `total_rtt`** on `gnn/cell01`, enough to flip gate verdicts. The
   audit's "live-gate corpora unaffected" statement was about the *cache* side of those
   corpora and remains true; the *live* side of every gate run on the committed tree
   served Bug 2.
3. **Process fix, mandatory before the next datalab gate**: the temporal fix (and the rest
   of the reviewed working-tree diff) must be committed and pushed so datalab and local
   run the same code — the gate/live scripts sync `models/` by rsync but `src/` by git,
   and an uncommitted src fix silently splits the two sides. Additionally,
   `run_provenance` should record `git describe --dirty` + a working-tree diff hash in
   every live result JSON (small `executesimulation.py` addition; deferred only until the
   currently-running sweeps finish, same rule as the F1 fix).
4. Measured noise floors for future margins: GNN local run-to-run 0.1–0.3% (residual F1
   tie-break + thread wobble), datalab GPU run-to-run ±0.04%. The live margins above are
   25–300× these floors.

`workload-200-200` (800k events) was still running when this was written — its result
extends the trace ladder but cannot change the re-grading above, which rests on the
matched-code probe matrix.

#### The environment axis, measured directly and closed (2026-08-21)

The resolution above exonerated the environment *by inference* — two clusters splitting on
code version, with library differences absorbed inside a 0.11% within-cluster spread. That
left one axis genuinely untested, and one assertion in this file simply false.

**The false assertion.** `CLAUDE.md`, `full_corpus_siv1_live_gate.sbatch` and this file all
stated that datalab gates run in the micromamba `gnn` env. They never have.
`run_full_corpus_siv1_live_gate.sh` calls `pipenv run python3` (50 call sites across 18 shell
scripts), and `pipenv run` resolves its own venv and shells straight past
`micromamba activate gnn`. Every cluster gate has actually run in an undeclared third
environment, `~/.local/share/virtualenvs/gnn-herosim-2TQKssTQ` — **torch 2.12.0+cu130**, where
the `gnn` env and local both have torch 2.5.1+cu121. The probe matrix above therefore compared
across a torch major *and* a CUDA major without anyone knowing.

**The measurement.** New tool `scripts_cosim/verify_venue_parity.py --mode logits` forwards a
committed 64-graph fixture (`tests/fixtures/venue_parity/`, 256 decisions / 1,738 scored edges)
through `near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt` and diffs logits and per-decision
argmax against a committed reference. Run at `22e8f27` with md5-verified identical weights and
import closure on both sides:

| stack | python | torch | numpy | PyG | max\|Δlogit\| | argmax flips |
|---|---|---|---|---|---:|---:|
| local pipenv (reference) | 3.12.3 | 2.5.1+cu121 | 2.3.0 | 2.6.1 | — | — |
| cluster `gnn` env | 3.12.12 | 2.5.1+cu121 | 1.26.4 | 2.7.0 | **0.0** | **0 / 256** |
| **cluster pipenv venv (what gates actually used)** | 3.12.12 | **2.12.0+cu130** | 1.26.4 | 2.8.0 | **0.0** | **0 / 256** |
| cluster pipenv venv, 4 threads | " | " | " | " | **0.0** | **0 / 256** |
| local, `--device cuda` (negative control) | 3.12.3 | 2.5.1+cu121 | 2.3.0 | 2.6.1 | 1.9e-5 | **0 / 256** |

The last row is a deliberate negative control: it produces a nonzero delta, proving the probe
can detect a difference, so the zeros are a result rather than a broken comparison.

**Outcome: the library-version axis contributes exactly zero.** A torch major across a CUDA
major, a numpy major, and a PyG minor together move the logits by zero bits. Only the
accelerator moves them, at 1.9e-5, flipping no decisions — consistent with the ±0.09% GPU↔CPU
`total_rtt` figure already recorded above. **Keeping the repo in sync is sufficient for GNN
inference numerics; the environment leak is a provenance and governance defect, not a
numerical one.** It stays worth fixing (`${HEROSIM_PY:-pipenv run python3}` +
`export HEROSIM_PY=python3`) because a declared environment that is not the running
environment made this take three sessions to rule out — but it is hygiene, and it does not
block the lineage.

**The generalisable diagnostic, recorded so the next incident is cheaper.** Environment and
float drift perturb decisions *symmetrically* — a flip helps as often as it hurts. The
observed gap was one-directional on all five cells and never flipped sign, which is a
**biased-estimator signature**, not an environment signature. That pattern had already ruled
out the environment on day one. Diagnose sign pattern before venue. Full protocol and the
comparability checklist now live in **`PARITY.md`**; the hard rules are in `CLAUDE.md` and
datalab-pitfalls #8.

**Read-off for job 709163** (fixed code, still through the unfixed leak, GPU): with the env
axis measured at zero, 709163's GNN arm was predicted to land at the local ~50.4M cluster, not
the recorded 65.8M. If it landed at 65.8M instead, this measurement would be contradicted and
the `--mode logits` probe would not be capturing whatever the difference is.

#### ✅ The prediction is CONFIRMED — job 709163 complete, 15/15 (2026-08-21, 21:2x)

All 15 array tasks COMPLETED at datalab commit `6f24e36`. The GNN arm lands in the local
working-tree cluster on **every cell**, and the whole gate reproduces the local re-grading
cell for cell:

| cell (density) | knative | GNN (datalab 709163) | GNN vs kn | GNN (local worktree) | local vs kn | **datalab vs local** |
|---|---:|---:|---:|---:|---:|---:|
| cell01 (p=0.25) | 46,556,947 | 50,609,830 | +8.7% | 50,407,465 | +8.3% | **+0.40%** |
| cell02 (p=0.35) | 39,463,080 | 37,371,050 | −5.3% | 37,358,225 | −5.3% | **+0.03%** |
| cell03 (p=0.15) | 43,922,518 | 51,640,719 | +17.6% | 51,550,527 | +17.4% | **+0.17%** |
| cell04 (p=0.50) | 34,942,834 | 32,281,245 | −7.6% | 32,269,908 | −7.6% | **+0.04%** |
| cell05 (p=0.20) | 46,941,809 | 46,793,121 | −0.3% | 46,905,755 | −0.1% | **−0.24%** |

The recorded gate (708549) had `gnn/cell01` at **65.8M / +41.4%**. Same checkpoint, same cells,
same trace, same cluster, same GPU partition — **the only thing that changed is the committed
code**, and 15.2M of `total_rtt` went with it. Cross-venue agreement is now +0.03% to +0.40%,
i.e. inside the GNN's own 0.1–0.4% run-to-run floor on four of five cells and 0.40% on the
fifth. **The environment measurement is confirmed end-to-end at the `total_rtt` level, not
just at the logit level**, and nothing is left of the "datalab and local disagree" hypothesis.

Scored by `compare_sealed_live_holdout.py`: three-way paired cell wins **GNN 1/5 · MLP 2/5 ·
Knative 2/5**; head-to-head against Knative the GNN is **2W/1T/2L** (better on cells 02/04,
tie on 05 at −0.3%, worse on 01/03) — identical to the local verdict on this trace. The tool
prints *"GNN does not dominate MLP on sealed holdout — do not claim uniform live transfer"*,
which remains the correct reading **for this trace**; `workload-150-100` and `workload-175-100`
are where the checkpoint wins 5/5.

**The recorded "+12% to +41% worse, GNN 0/5" FAIL is now formally superseded**, not merely
re-graded: it was a measurement of the uncommitted dims 9-11 diff. Do not cite it.

**MLP arm, all 5 cells (final).** The catastrophic tail is real, survived the fix, and is
*worse* on this trace than on any other: 4.81× / 5.97× / 8.17× Knative on cells 01/03/05
(223.7M, 262.2M, 383.5M), against 0.69× and 0.77× on cells 02/04. Pre-fix vs post-fix the MLP
moved −0.2% to +1.9% — so the fix that moved the GNN 23.3% barely touches the MLP, confirming
the asymmetry hypothesised in the qualification box above. Root cause now measured; see the
next subsection.

#### The MLP catastrophic tail, root-caused (2026-08-21)

Standing open thread since the recorded gate ("why does the pointwise baseline blow up at
sparse topologies?"), guessed at twice as `intra_batch_platform_collisions`. It is not that.
Measured across 4 local sweeps + job 709163 — two traces, two venues, 8 collapse instances:

| stat (`stats` block of the result JSON) | collapsed MLP | healthy MLP | knative | gnn |
|---|---:|---:|---:|---:|
| `averageOccupation` | **1.0–1.5** | 13.6–13.8 | 8.7–10.6 | 5.1–5.6 |
| `endTime` | 5.1–7.8× knative | 0.4–0.5× | 1.0× | 1.4–1.7× |
| `averageQueueTime` | 467–535 s | 48–54 s | 78–88 s | 76–92 s |
| `chosen_queue_vs_min` p95 | **8,964–23,731** | 504–1,129 | — | 539–1,083 |
| `chosen_queue_vs_min` **median** | 4–129 | 69–192 | — | 8–163 |
| `intra_batch_platform_collisions` | 0.14–0.18 | 0.13–0.17 | — | 0.22–0.45 |

Every physics component (cold start, execution, communication, `offloadingRate`, cache hits)
is unchanged between collapsed and healthy runs, and the collision rate is *normal* — the
collision hypothesis is dead. What separates them is **occupation collapsing to ~1**: the
cluster idles while a few replicas hold runaway queues.

**The MLP wins by packing and loses by packing.** Its healthy occupation (13.6) is higher than
Knative's (10.6) and far above the GNN's (5.6) — aggressive packing is exactly why it beats
Knative on raw margin on most cells. Its pointwise score has no term that sees a platform's
queue *relative to the alternatives*, so when a packed platform cannot drain, the queue runs
away and `total_rtt` goes 4–8×. The median `chosen_queue_vs_min` is normal in every collapsed
run, so this is a compounding minority of decisions, not a systematic mis-ranking. Collapse
risk rises as topology gets sparser and durations get longer: only cell05 (p=0.20) collapses
on `workload-150-100`/`175-100`, while the longer-duration `workload-125-225` collapses the
three sparsest cells and spares p=0.35/0.50. **The GNN has never collapsed in 20 cell-runs**
(worst p95 1,083). `averageOccupation ≈ 1` is the cheap detector — no need to parse the 200 MB
result JSON. A fix belongs in the MLP decode (a queue-relative term, or a cap on
`chosen_queue_vs_min`), not in the simulator.

#### The corrected-cache retrain — the residual train/serve gap, measured (2026-08-21)

The dims 9-11 fix landed 2026-08-19; `graphs_cache_full_corpus_siv1_dim14` was built
2026-08-15. **The deployed checkpoint was therefore trained on pre-fix features and is now
served post-fix ones**, which is the same defect class as [[gnn-train-serve-mp-mismatch]].
Rebuilt the cache post-fix into a **new** directory (`..._tempfix`, job 709232, 2 min;
`CACHE_DIR` had to be made overridable first — the script `rm -rf`s it, and the default is the
deployed checkpoint's only training data) and diffed the two, 2,651 shared datasets:

| | siv1 full corpus | `shallow_v1` (for scale) |
|---|---:|---:|
| platform rows changed | **175,034 / 551,408 = 31.7%** | 18.7% |
| graphs with any change | **2,651 / 2,651 (100%)** | — |
| max abs delta | **2.775** | 0.0086 |
| changes escaping dims 9-11 | 2 graphs | 0 |
| labels `y` changed | 2 graphs (0.08%) | 0 |

**The perturbation is ~320× larger than the one measured on `shallow_v1`, and it touches every
graph in the corpus.** So the live wins on `workload-150-100`/`175-100` were achieved by a
checkpoint whose training inputs disagree with its serving inputs on a third of all platform
rows — the wins are real but they are not the ceiling. Retrain submitted as job **709234**
(`near-rtt-v2-full-corpus-siv1-dim14-ce-only-tempfix`, GPU-a40, corrected cache, 2,658 graphs;
`OUT_CKPT` is now derived from `WANDB_RUN_NAME` so it lands beside the deployed checkpoint
rather than on top of it). **Not yet gated — the new checkpoint means nothing until it runs
the same 5-cell live gate on all three traces.** The MLP has the same mismatch and moved only
~1% under the fix, so a matched MLP retrain is the fair comparison and has not been run.

#### `workload-200-200` — the fourth trace lands: MLP sweeps, GNN 3W/2L, and no MLP collapse (2026-08-22)

The queued 800k-event trace (`workload-200-200.json`, rps=200, dur=200 s — the trace this file
had named three times as an aspirational target) completed locally as sweep `a4_wl200200`
(PAR=2, 15/15 results, pre-merge working tree — same fixed dims 9-11 live path as every other
row in this retest; `fix/deferred-gate-fixes` was merged only after the sweep finished).
Deployed checkpoint, same 5 parity-verified cells:

| cell (density) | knative | mlp (vs kn) | gnn (vs kn) | occ mlp / gnn |
|---|---:|---:|---:|---:|
| cell01 (p=0.25) | 167,472,990 | 109,546,546 (**−34.6%**) | 130,089,834 (−22.3%) | 7.8 / 6.7 |
| cell02 (p=0.35) | 113,833,627 | 95,576,423 (**−16.0%**) | 117,217,941 (+3.0%) | 12.5 / 9.5 |
| cell03 (p=0.15) | 136,967,398 | 112,812,746 (**−17.6%**) | 126,429,628 (−7.7%) | 7.5 / 7.5 |
| cell04 (p=0.50) | 107,589,923 | 94,604,668 (**−12.1%**) | 114,534,379 (+6.5%) | 12.2 / 8.8 |
| cell05 (p=0.20) | 139,429,374 | 130,150,818 (**−6.7%**) | 135,213,788 (−3.0%) | 3.6 / 6.1 |

`compare_sealed_live_holdout.py`: paired cell wins **GNN 0/5 · MLP 5/5 · Knative 0/5**; MLP
also takes 4/5 cells on both p90 and p99 per-task tails. Three findings:

1. **The trace-dependence claim is now the settled reading.** Deployed checkpoint vs Knative
   across the four full-scale traces: 2W/1T/2L (125-225) · 5/5 W (150-100) · 5/5 W (175-100) ·
   **3W/2L (200-200)**. "Beats Knative on some real traces and loses on others" is confirmed;
   the losses concentrate on the two *densest* cells here (p=0.35/0.50, +3.0%/+6.5%), the
   opposite cells from the 125-225 losses (p=0.15/0.25).
2. **The MLP's catastrophic tail does not appear at the heaviest load.** No collapse on any
   cell (occupations 3.6–12.5, nothing near 1; worst p99 is cell05's 2,416 s, still ~2× not
   4–8×). This bounds the collapse subsection above: 225 s durations collapse three cells,
   200 s durations at *higher* rps collapse none — so duration alone is not the driver either;
   the instability needs both long durations and something this trace lacks. First trace of
   the four where the MLP dominates both baselines outright.
3. **The GNN has still never collapsed** (20 → 25 cell-runs), but at rps=200 its conservative
   low-occupation placement (5.6–9.5 vs MLP's 7.5–12.5) costs it every head-to-head vs the
   MLP — the packing that destroys the MLP on 125-225 is exactly what wins at sustained high
   throughput when queues do drain.

#### ❌ The corrected-cache retrain FAILS its live gate — 0/15, uniformly, on all three traces (2026-08-22) — **FALSIFIED 2026-08-23**

> **⚠ This FAIL was a serving artifact, not a model or cache result.** The `tempfix`
> checkpoint's sidecar declares `inference_feature_layout: null`, so it served **`atomic21`**
> while the deployed checkpoint it was compared against served **`dim22`** from its sidecar.
> Re-serving `tempfix` under `dim22` on the same cells improves `total_rtt` by **13.3–40.8%
> (mean −29.8%)**, and it then wins **5/5 vs Knative at −14.0% mean** without a backbone and
> **5/5 at −34.1%** with one — beating the deployed checkpoint in both conditions. The
> corrected cache is **better**, not worse. See *"The backbone win is REGIME-LEVEL"*
> (`link_contention_v1`, 2026-08-23). The table below is retained as the record of what was
> measured, not as a finding about the cache.

The retrain (job 709235, `near-rtt-v2-full-corpus-siv1-dim14-ce-only-tempfix.pt`, corrected
post-dims-9-11 cache, contract sidecar present) ran the full 15-task gate on all three traces
(jobs 709296 / 709435 / 709436, datalab commit `37e5004` — pre-merge, same code lineage as
709163; `workload-{150,175}-100.json` had to be rsynced first, md5-verified both sides — they
had **never existed on datalab**, which also retro-explains why every prior datalab gate used
125-225 only). Deployed-checkpoint margins vs Knative alongside, for contrast:

| trace | tempfix GNN vs kn (cells 01..05) | deployed GNN vs kn | verdict |
|---|---|---|---|
| workload-125-225 | +47.3 / +53.4 / +63.3 / +42.2 / +27.0 % | +8.7 / −5.3 / +17.6 / −7.6 / −0.3 % | **0/5** (deployed: 2W/1T/2L) |
| workload-150-100 | +9.3 / +34.7 / +13.5 / +43.8 / +19.8 % | −0.9 / −12.9 / −14.8 / −5.3 / −12.9 % | **0/5** (deployed: 5/5 W) |
| workload-175-100 | +14.6 / +36.7 / +31.7 / +39.0 / +5.6 % | −9.4 / −7.6 / −7.5 / −10.8 / −10.8 % | **0/5** (deployed: 5/5 W) |

The comparison is clean: the Knative and deployed-MLP arms in these sweeps reproduce the
recorded values **exactly** (e.g. `knative/cell01` 24,471,537 and `mlp/cell01` 22,206,010 on
150-100, digit for digit) — same cells, same traces, same code; only the GNN checkpoint
differs.

**The inversion is the finding.** In-distribution the retrain is the *better* model: best val
acc 70.7% vs the deployed 66.3% (different caches, same corpus/split/hyperparameters/seed —
the runner was identical except `CACHE_DIR`/`WANDB_RUN_NAME`). Live it is uniformly 1.06–1.63×
Knative and loses every cell to the deployed checkpoint, on the traces where the deployed one
wins 5/5. Making training features consistent with serving features — removing the 31.7%-of-rows
mismatch — made the live policy drastically worse while making the co-sim metric better.

**What this does and does not establish.**
- It *supports* the §"retrain's own risk" reading (HANDOVER 2026-08-21): some of the deployed
  checkpoint's live edge is attributable to the train/serve mismatch itself — training on
  pre-fix dims 9-11 acted as an accidental regularizer / decision-bias that happens to help
  live, and "fixing" it removed that.
- It does *not* yet separate that from **retrain variance**: this is one training run against
  one training run. The discriminating control is a retrain on the **pre-fix** cache with the
  identical pipeline (only `CACHE_DIR` differing) — if that lands near the deployed checkpoint,
  the cache is causal; if it also collapses live, the deployed checkpoint is a lucky draw and
  the corpus/live gap is wider than either cache. **Not run.**
- The `logit_tied_rate ≈ 0.54` thread gets its first answer: the dims 9-11 feature bug was
  not what was holding the model back — correcting it helps no trace.

**Decision: the deployed checkpoint stays deployed.** The tempfix checkpoint is evidence, not
a candidate. — **REVERSED 2026-08-23:** at a fixed `dim22` serving layout `tempfix` beats the
deployed checkpoint on mean margin in both conditions (5/5 each), so it is now the leading
*candidate*. Promotion still wants a re-gate on `workload-175-100` and `workload-200-200`,
since its advantage is so far measured on one trace.

**The matched MLP control landed the same day (jobs 709495-97, `..._tempfix.pt` MLP retrained
on the same corrected cache, MLP arm only, same cells/traces) and confirms the prediction with
one refinement.** On the 9 healthy cells across the three traces the tempfix MLP moves
**−6.6% to +2.6%** vs the deployed MLP (most within ±3%) — the feature fix is inert for the
pointwise model, consistent with its ~1% logit movement. On the collapse-implicated cells the
fix does not fix or worsen the tail so much as **re-roll it**: 150-100 cell03 (p=0.15,
previously healthy at 0.75× Knative) *newly* collapses (+281%, occupation 1.10) while 125-225
cell01 improves 23% yet stays collapsed (occ 0.98); the always-collapsing cell05 gets worse on
both 125-225 (+45%) and 175-100 (+48%). So which cell collapses is chaotic under small feature
perturbations — reinforcing that the MLP tail is a knife-edge packing instability
([[herosim-mlp-collapse-is-occupation-collapse]]), not a feature-quality effect. **The
contrast stands: the GNN degrades systematically and one-directionally under the corrected
cache (every cell, every trace); the MLP is indifferent except where it was already unstable.
The mismatch-sensitivity is GNN-specific.**

#### ⚠ The variance control answers: it is a lottery — the deployed checkpoint is a lucky draw (2026-08-22) — **PARTLY SUPERSEDED 2026-08-23**

> **⚠ Read the correction first.** This comparison is **confounded by the serving layout**:
> the deployed checkpoint served `INFERENCE_FEATURE_LAYOUT=dim22` (declared in its sidecar)
> while `prefixctl` and `tempfix` served `atomic21` (sidecar silent, and `load_gnn_model`
> defaulted). Re-serving `prefixctl` under `dim22` on these same cells improves it by
> 1.2–14.8%, and `tempfix` by 13.3–40.8%. At a fixed layout the draw spread is **much**
> smaller than the 5–36% below, `tempfix` becomes the best checkpoint on disk, and the
> backbone win is regime-level. See *"The backbone win is REGIME-LEVEL"*
> (`link_contention_v1`, 2026-08-23). The qualitative point — that a single-checkpoint gate
> is a claim about one draw — survives; the magnitudes do not.

`...-prefixctl.pt` (train job 709516, gate job 709534): **same pre-fix cache, same pipeline,
same seed** as the deployed checkpoint — the only difference is GPU/dataloader nondeterminism
in one training run. (`MIN_GRAPHS` overridden to 2651: the guard postdates the deployed run,
which itself trained on 2,651 graphs; the tempfix cache has 2,658 — the rebuild picked up 7
extra datasets, a second small train-set delta.) In-distribution it is the deployed model's
twin: val acc **66.8% vs 66.3%**, greedy regret identical to 4 decimals (0.6467s). Live, on
`workload-150-100` (same cells, GNN arm only):

| cell | deployed | **prefixctl (control)** | tempfix |
|---|---:|---:|---:|
| cell01 (p=0.25) | −0.9% | **+4.8%** | +9.3% |
| cell02 (p=0.35) | −12.9% | **+14.8%** | +34.7% |
| cell03 (p=0.15) | −14.8% | **−0.6%** | +13.5% |
| cell04 (p=0.50) | −5.3% | **+28.5%** | +43.8% |
| cell05 (p=0.20) | −12.9% | **+4.2%** | +19.8% |

**Control: 1/5 vs Knative (and that one at −0.6%, inside noise) where the deployed draw won
5/5.** Per cell the control is +5.8% to +35.7% worse than the deployed checkpoint — from
training nondeterminism alone, against a 0.1–0.4% run-to-run simulation floor.

Three conclusions, in order of weight:

1. **Live quality of this pipeline is a training-draw lottery.** Two checkpoints
   indistinguishable in-distribution (0.5pp val acc, identical greedy regret) differ by
   5–36% of live `total_rtt` per cell. In-distribution metrics have **zero** discriminating
   power over which draw wins live — the sharpest form yet of the co-sim/live disconnect
   (`graph_structure_physics` / `logit_tied_rate` thread).
2. **The tempfix FAIL above is therefore confounded.** Cache ordering is consistent
   (deployed < control < tempfix on every cell), so the corrected cache *may* still be worse —
   but with one draw per cache and within-cache variance this size, the cache effect is not
   separable from draw luck. The "mismatch was an accidental regularizer" reading is
   downgraded from supported to *possible*; the demonstrated effect is the variance.
3. **Any single-checkpoint live-gate verdict in this lineage is a claim about a draw, not
   about the recipe.** Every siv1 gate row above (including the 709163 CONFIRMED one, which
   compared the *same* checkpoint file across venues, and is unaffected as a venue
   measurement) generalizes to the training pipeline only up to this lottery. A future gate
   that wants a recipe-level claim needs multiple training draws per arm — which the ablation
   harness already does (5 seeds) and the production pipeline does not.

**The deployed checkpoint stays deployed** — it is demonstrably the best live artifact on
disk. But it should be described as "the checkpoint that won", not "what the pipeline
produces".
