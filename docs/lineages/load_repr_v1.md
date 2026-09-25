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
  - **Service** of a committed batch-mate is execution on the candidate's platform type, plus the
    storage I/O approximation, plus its peer transfer to committed partners on other nodes
    (Amendment 1).
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

- 2026-09-25 — **O1 read (offline; orders nothing): the load columns help offline.**
  - **Setup:** 480 held-out groups under A1's protocol. Training was datalab array 806789 at
    b449ebf, with determinism 17/17 at the same commit. All 8 sidecars passed their checks, and
    checkpoints were selected at epochs 104–132 (see the train logs).

  | arm | median regret | mean regret |
  |---|---|---|
  | CD | 8.2 % | 54.5 % |
  | **`v4load`** (median over 4 seeds; per seed 8.0–9.1 %) | **9.0 %** | **42.2 %** |
  | `v4twin` | 11.4 % | 54.0 % |
  | jb2 `gnnedge0` (13 seeds) | 11.7 % | 54.6 % |

  - **Pairwise:** `v4load` beats its twin on 190 groups and loses on 113. Paired median is 0.0 pp
    (most groups tie); paired mean is −11.8 pp vs the twin and −12.2 pp vs CD.
  - **Reading:** the representation does carry what the label needs, and CD's load terms are
    learnable from this corpus. This is the first learned arm with lower mean held-out regret than
    CD. Live is the test. [Read](load_repr_v1/o1_read.json); parity files in `load_repr_v1/`.

- 2026-09-25 — **Amendment 2 (signed before any live v4 read, and before any fixed-serving datum).**
  **The train/serve parity check found a serving defect in every burst-seat learned arm.**
  - **The check:** `peer_affinity_live_serve_check.py` on 60 held-out jb2 batches.
    - Pre-fix, the checkpoint behind every earlier burst-seat gate (`jb2` `gnnedge0` s1, `v3` cache)
      is clean on 0/60. Serving gave it a different node rank (`krank`, prefix columns 10–17) than
      training on 56/60, and decoded a different plan on 8/60.
    - `v4load` s1 and `v4twin` s1 show the same defect, 0/60 clean.
    - The new columns, `backlog_s` and `service_s`, matched on 60/60.
  - **Cause:** at the uncapped rung, serving built caps as `inf × peak demand for m > 0`. That omits
    nodes whose only candidates have zero memory demand (`pynqFpga` `dnn1` = 0.0) and ranks them first
    at cap 0.0. The cache's uncapped rung is `{}`, which puts every node at 0.0.
  - **Fixed at 721d44f:** serving uses `{}` at `alpha = inf`, and the decoder, which reads
    `caps.get(n, inf)`, is unchanged.
  - **After the fix:** `v4load` s1 is 60/60 clean. `v4twin` s1 and `jb2` `gnnedge0` s1 are 58/60,
    and the two leftovers are same-node sibling-platform tie-breaks whose live RTT equals the sweep
    row.
  - **Affected:** every live number for `gnnedge0` and MP-OFF in `joint_burst_v2`,
    `selfpredict_burst_v1`, `fresh_topo_burst_v1` and `cd_gap_v1`. Rule arms (CD, self-predict,
    1-pass, reactive) never touch this code.
  - **Design change:**
    - The running v4 gate (807111, b449ebf) is recorded **as served with the defect** and does not
      carry L1/L2.
    - Phase `fix` reruns the learned arms at the fix commit on the same fresh study: `v4load` ×4,
      `v4twin` ×4, `jb2` `gnnedge0` ×13, `jb2` MP-OFF ×13.
    - CD and self-predict are the existing 00dae37 runs.
    - **L1/L2 are read on fixed serving**, with bars and labels unchanged.
  - **New registered reads** (`load_repr_v1_read.py --old-gate`):

    | read | contrast | fires as |
    |---|---|---|
    | **R1** | `gnnedge0` fixed vs as served (fresh gate, 1ae90af) | `DEFECT-COST` (fixed ≤ −5 %, p < 0.05) / `DEFECT-COST (direction only)` / `NOT-SEPARATED` / `FIX-HURTS` |
    | **R2** | `gnnedge0` fixed vs CD | as D1's labels; replaces `cd_gap_v1`'s +11.4 % if it differs |
    | R3–R5 | MP direction fixed; MP-OFF fixed vs as served; `gnnedge0` fixed vs self-predict | reported |
  - **Expectations:**
    - R1 `NOT-SEPARATED` or direction only 70 %: rank changed the offline plan on ~10 % of batches.
    - R2 still `CD-FASTER` 85 %.

- 2026-09-25 — **Amendment 1 (signed before any v4 datum; no model trained yet).**
  - **The first cache job (806779) failed on every dataset.** `_v4_load_seconds_block` enumerated
    `task_logit_to_placement`, which is a dict, not a list. Fixed at ec702ba; a 3-dataset login-node
    smoke then built.
  - **The smoke showed a design gap.** Service as registered (execution + storage I/O) is ~0.003 s per
    task in this physics, so column 23 would carry almost nothing. CD's committed service is
    `exec + comm + exch`, and the backlog clock also charges each queued task its peer transfers.
  - **Column 23 now charges each committed batch-mate** its execution + I/O plus the peer transfer to
    every committed partner on another node, using column 7's formula. Bars and labels are unchanged.
  - **Corpus fact, measured from `infrastructure.json` before training.**
    - 1,284 of 2,516 jb2 groups have at least one replica with seeded backlog > 0.
    - Only 0.5 % of the 576,164 replica specs have backlog > 0 (median 0.75 s, p90 23.6 s,
      max 624 s).
    - The training states are far emptier than the live seat, where `gnnedge0` queues ~4 s per task.
      This is the standing risk named above, now measured, not an outcome.

- 2026-09-25 — Registered. Code at the registration commit. The default path is bit-identical (the
  two witnesses above). `tests/test_partial_state_v4.py` 14/14 pass. The related suites show 100
  passed and 2 failures that also fail at the parent b053717 (a missing local smoke cache).
