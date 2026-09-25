# load_repr_v1 — does giving the GNN the CD greedy's load terms in seconds close the gap to CD?

**Status:** `REGISTERED` (2026-09-25). Every bar below was signed before its data.

**Parent:** [`cd_gap_v1`](cd_gap_v1.md) (`SCORE-EXPLAINS / CD-FASTER`). The burst-seat `gnnedge0` trails
CD +11–14 % live. The gap is all queue, and neither its label, its decode order, its candidate set nor
the load explains it. CD's score is queue drain in seconds plus service already committed to the same
replica in this batch. `gnnedge0` sees queue as a normalised task *count* frozen at batch start, and
batch-mates as per-*node* type counts (`partial_state_v3`), never per replica and never in seconds.
That close ended on a claim: "the fix is the representation". This lineage tests it.

## Design

- **The representation, `partial_state_v4`.** `v3`'s 22 prefix columns, unchanged, plus three per
  candidate replica:
  1. `log1p(backlog s)` at batch start;
  2. `log1p(service s committed to the same replica by batch-mates earlier in the decode)`;
  3. `log1p(their sum)`.
  - **Backlog** is the clock the co-sim replays for that replica, `live_snapshot_seed.seeded_backlog_seconds`
    (current task remaining + comm remaining + measured drain). Serving computes the same terms with
    `live_audit.candidate_backlog_seconds`.
  - **Service** is execution on the candidate's platform type plus the storage I/O approximation.
  - The columns enter at the EdgeScorer only, like the rest of the prefix block, so the cached-encode
    invariant holds.
- **Arms (4 seeds each, seeds 1–4):**
  - `v4load` is the jb2 `gnnedge0` recipe (corpus, split, alpha `inf`, labels, hyper-parameters) with
    `PARTIAL_STATE_CONTRACT=partial_state_v4`.
  - `v4twin` is the same with `PARTIAL_STATE_LOAD_SECONDS=0`: the three columns are zeroed, the width
    is the same, and it is trained separately. Both are recorded in the sidecar and adopted or
    verified at serving.
- **Cache:** `graphs_cache_joint_burst_v2_psv4_inf`, built from the same jb2 corpus and checked against
  the frozen `joint_burst_v2_split.json` (sha e54f3364…).
- **Offline (orders only):** held-out regret per `cd_gap_v1` A1's protocol, next to CD's 8.2 % and
  `gnnedge0`'s 11.7 %.
- **Live:** the `fresh_topo_burst_v1` study (12 topologies × 4 windows, base rate, its failure rule and
  statistic). New runs are `v4load` × 4 and `v4twin` × 4. CD, self-predict and jb2 `gnnedge0` are the
  existing 00dae37 gate summaries: the default serving path is bit-identical at this lineage's code,
  witnessed to the digit on CD 9119 w1 and `gnnedge0` s1 9119 w1.
- **Instruments, every learned run:** `v4_backlog_batches` > 0, and the served
  `PARTIAL_STATE_LOAD_SECONDS` equal to the sidecar's. A `v4load` run also needs
  `v4_backlog_nonzero` > 0.

## Bars (signed 2026-09-25, before any v4 datum)

| read | contrast | fires as |
|---|---|---|
| **L1** | `v4load` vs CD, live | `CLOSES-GAP` if `v4load` is not slower (median ≤ 0, or p ≥ 0.05); else `NARROWS` if L2 fires `LOAD-HELPS` (either form); else `NO-EFFECT` |
| **L2** | `v4load` vs `v4twin`, live (same seed paired) | `LOAD-HELPS` (≤ −5 %, p < 0.05) / `LOAD-HELPS (direction only)` (p < 0.05, \|median\| < 5 %) / `NOT-SEPARATED` / `LOAD-HURTS` (p < 0.05, `v4load` slower) |
| L3 | `v4load` vs jb2 `gnnedge0` (seeds 1–4 paired), vs self-predict | reported |
| O1 | offline held-out regret, both arms | reported; orders nothing that has not already been registered |

**Expectations, written before data.**
- **L2:** `LOAD-HELPS` 30 %, direction only 30 %, `NOT-SEPARATED` 35 %, `LOAD-HURTS` 5 %.
- **L1:** `CLOSES-GAP` 10 %, `NARROWS` 50 %, `NO-EFFECT` 40 %.
- **The standing risk:** offline, `gnnedge0` and CD tie in paired median regret (0.0 pp, `cd_gap_v1`
  D0). The jb2 training states may carry too little in-batch piling for the columns to be learned
  from, whatever they express.

**Rule 6.** O1 is offline and closes nothing; L1/L2 are the live gate.

## Entry points

- Contract: `src/policy/tabular/reduced_features.py` (`PARTIAL_STATE_CONTRACT_V4`, `partial_state_columns`),
  `tests/test_partial_state_v4.py`.
- Cache side: `src/notebooks/prepare_graphs_cache.py` (`_v4_load_seconds_block`).
- Serving side: `src/policy/gnn/prefix_serving.py` (`attach_live_prefix_block`) and
  `src/policy/gnn/scheduler.py` (`_v4_backlog_seconds`).
- Live gate: `scripts_cosim/fresh_topo_burst_v1_gate.py` phase `v4`.

## Record (newest first)

- 2026-09-25 — Registered. Code at the registration commit. The default path is bit-identical (the
  two witnesses above). `tests/test_partial_state_v4.py` 14/14 pass. The related suites show 100
  passed and 2 failures that also fail at the parent b053717 (a missing local smoke cache).
