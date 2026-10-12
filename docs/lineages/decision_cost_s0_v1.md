# decision_cost_s0_v1 — when the scheduler's decision time is charged as simulated time, which search still pays, and does the GNN's cheaper search beat `cd_exactS`?

**Status:** `DRAFT` (2026-10-12, owner S4, for the coordinator's signature). Nothing launches before it is signed **and** `ib0_confirm_v1` is read.
Code and the identity smokes may start now.

## Why

Every latency in the program so far is simulated time with the scheduler's compute free: `decisionTiming` is wall-clock measurement only
(`src/placement/decision_timing.py`, "schedules nothing in SimPy"). `gate_79ea002c` summaries give the mean wall decision per task:

| arm | moderate | heavy | one-thread utilisation at heavy |
|---|---|---|---|
| `cd_exactS` | 0.061 s | 0.057 s | ~76 % |
| GNN self-search | 0.036 s | 0.035 s | ~45 % |
| plain GNN | 0.008 s | | ~10 % |
| `cdxapply` | 0.009 s | | |
| CD | 0.0002 s | | |

The naive per-task sum narrows the gaps by 1–2 pp and leaves `cd_exactS` ahead at heavy (+2.4 %). It ignores **scheduler queueing**: a batch that
arrives while the scheduler is still deciding waits, and at ~76 % utilisation that wait is large and bursty. This node charges it.

## Physics (to be fixed before any data)

One scheduler server, FIFO. In the batch decision path (`src/policy/gnn/scheduler.py`, the span that already runs `_prefix_inference` → `_cd_refine`;
all arms of the seat run through it), after the placements are computed:

    t = c × plans_evaluated        # seconds, c in ms per plan
    if t > 0: yield env.timeout(t) # then the placements are applied

- **Server and queue.** A `simpy.Resource(capacity=1)` (FIFO) is requested *before* the system-state mutex and released after the placements are
  applied, so batches queue behind the scheduler, not on the mutex. The mutex is held through the timeout (the decision is atomic in simulated time and the
  placement is applied to the state it was computed on, which keeps the "decoded replica is no longer a replica" check of
  `scheduler.py:1290–1296` valid). Disclosed side effect: the autoscaler and monitor wait for the mutex for `t` too.
  *Decision for the coordinator:* the alternative (release the mutex, apply on a stale snapshot) is realistic but re-opens the W4 eviction error; I
  recommend the held-mutex form.
- **`c = 0` is byte-identical to today:** no resource request, no timeout, no extra SimPy event (`HEROSIM_DECISION_COST_MS` unset or 0 takes the old code
  path). Required: the flag-off identity test (`r1a_gate_identity.py`) on the 24 identity cells, and the physics_audit_v1 replay rule (any extra SimPy
  event is a physics change) so the replay check must pass at `c = 0`.
- **`plans_evaluated`** — a new monotone counter `sched_plan_evals` (a counter read, no behaviour), defined per arm **now**:
  - `cd_exactS`: `pg_exact_plans` enumerated + `pg_expand_evals` (expansion fallback) + one per single-task candidate scored in the passes;
  - `cd` / hand seed: one per (task, candidate couple) scored in a sweep, i.e. the "choose" of a task counts 1 per candidate per sweep;
  - GNN self-search: `ss_scored` (plans scored); `cdxapply`: GNN seed forward (1 per task) + CD passes + `pg_expand_evals`;
  - plain GNN: 1 per task (one forward per task of the decode).
  - **Caveat stated up front:** a full-plan score and a single-move score are not the same work. A calibration table fixes this before the gate:
    on the smoke cells, `c_ref × plans_evaluated` is compared with the measured wall per batch of each arm; the ratio per arm is reported and the
    gate's `c` grid is read as a *reference-CPU model* (0.25–0.4 ms per plan is the measured cluster rate), never as measured wall.

## Arms and cells

Accel seat as in `ib0_confirm_v1` (160×24, p .6, `fastest_compatible`, KPA, pipelined, `HEROSIM_REPLICA_RELEASE=1`), rungs moderate / heavy, windows
**g0–g1**, 12 topologies **19405–19416** (reuse the `ib0_confirm_v1` inputs if byte-identical by md5; otherwise mint 12 fresh).

Arms (6, 7 if `ib0` confirms): `cd`, `cd_exactS`, `cd_exactS_ib0` (**only if `ib0_confirm_v1` reads WIN**), GNN self-search s1, `cdxapply` s1, plain
`nosplit` s1. c in {0, 0.1, 0.25, 0.5, 1.0} ms per plan.

## Hypotheses and bars

**Primary family (Holm over arm × rung at each c > 0):** GNN self-search vs `cd_exactS`; `cdxapply` vs `cd_exactS`. Statistic as in
`ib0_confirm_v1` (per topology median over windows of the paired %; median over topologies; sign count; exact Wilcoxon; Holm via `R.holm`).
**WIN** for an arm: median ≤ −2 % and Holm p < .05 at **both** rungs, for the same arm, at some c ≤ 0.5 ms. Otherwise TIES, or LOSES if `cd_exactS` is
Holm-faster at both rungs at every c ≤ 0.5.

**Also reported (no bar):** the break-even c per arm vs CD (the c at which the arm stops beating CD); scheduler utilisation and mean scheduler-queue wait
per arm and c; the latency breakdown (drain / cold / exec / latency / exchange / scheduler wait) in pp of the reference's mean latency; the
calibration table above; `c = 0` contrasts re-stated on these 12 topologies.

**Disclosed variant, not run unless the single-server result is a WIN:** a parallel scheduler with k servers.

## Predictions on record (coordinator's, written before data)

`cd_exactS` loses its heavy edge at c ≥ 0.25 ms; `cdxapply` beats it at heavy from c = 0.25 ms; the GNN self-search ties.
*(S4's addition:)* CD is flat in c (≤ 0.5 pp at c = 1.0); `cd_exactS` utilisation exceeds 1 at heavy for c ≥ 0.5 ms (unstable queue, so its curve is
steep rather than gradual) — if so, the sweep is reading an unstable regime and says so.

## Cost

Cells = arms × c × 12 topologies × 2 rungs × 2 windows = 48 cells per arm per c. Full grid (6 arms × 5 c) is **1,440 cells**; c = 0 is already
measured for `cd` and `cd_exactS` on these topologies (2 × 48 = 96), the other four arms' c = 0 cells are new, so **1,344 new cells**. At the measured
walls (`cdx_ib0` ≈ 80 s, `cdxib0` ≈ 290 s, and the `ib0_confirm_v1` average of ~1,150 s per exact-search cell) that is **≈ 430 cell-hours**, about 2 days at
8 jobs × 60 workers — over the ~1 day line. **Proposed cut: c ∈ {0, 0.25, 1.0}** first (the 0.1 and 0.5 points only if the 0.25/1.0 contrasts
straddle the bar): 6 × 3 × 48 = 864 cells, 768 new (96 c = 0 reused), **≈ 245 cell-hours, about 1 day**. The primary WIN rule above is evaluated on
c ≤ 0.5, so the cut reads it at c = 0.25 only; the 0.5 point is the follow-up. Queue ≤ 8 jobs for me (≤ 44 total).

## Record

2026-10-12 — drafted; code not yet written. Open points for the signature: held-mutex vs stale-snapshot form; the unit of `plans_evaluated` for
single-move scores; the c-grid cut.
