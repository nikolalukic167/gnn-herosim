# decision_cost_s0_v1 — when the scheduler's decision time is charged as simulated time, which search still pays, and does the GNN's cheaper search beat `cd_exactS`?

**Status:** `REGISTERED` (2026-10-12, owner S4, signed by the coordinator with the amendment below). Nothing launches before `ib0_confirm_v1` is read.
Code goes on branch `rp/decision-cost`, not `rp/plan-rerank`.

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

Arms (7: `ib0_confirm_v1` read PRIMARY WIN, so `cd_exactS_ib0` is in): `cd`, `cd_exactS`, `cd_exactS_ib0`, GNN self-search s1, `cdxapply` s1, plain
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

2026-10-12 — **signature amendment (coordinator rulings, supersede the draft where they differ):**
1. **Held-mutex form.** Disclosed side effect: the autoscaler and monitor wait `t` for the mutex. A counter records how long they waited (total and max
   mutex-wait seconds while the scheduler server is busy) so a reader can see whether the autoscaler was starved.
2. **`plans_evaluated` definitions stand**, plus: (a) the calibration table is read **before** the gate and recorded; if any arm's measured wall per
   plan differs from the others by more than 3×, say so and keep `c` as a common reference anyway (a plan score is the unit, disclosed);
   (b) per-cell counters for `plans_evaluated` and scheduler busy time must appear in the summaries, and the smoke must show them (instrument rule).
3. **Grid cut to c ∈ {0, 0.25, 1.0} ms** (864 cells, 768 new, ≈ 245 cell-hours). 0.1 and 0.5 ms run only as a registered follow-up if the primary
   contrasts straddle the bar at 0.25. The WIN rule's "some c ≤ 0.5" is therefore read at 0.25.
- **Launch order after `ib0_confirm_v1` is read:** identity test (c = 0 byte-identical on the 24 identity cells) → calibration → one-cell smoke with
  counters → gate, classical arms first.


2026-10-12 — drafted (558abed3); code not yet written.

2026-10-12 — `ib0_confirm_v1` classical stage read PRIMARY WIN (e7966896): `cd_exactS_ib0` is in the arm list (7 arms; cut grid 7 × 3 × 48 = 1,008 cells, 912 new since `c = 0` of `cd`
and `cd_exactS` is reused, ≈ 290 cell-hours, a little over a day). Heavy flag-off identity of the oracle-stage code closed IDENT (33dcfe1f).

2026-10-12 — **Amendment 2 (SCIENTIST's review, adopted by the coordinator; committed before any c > 0 data):**

1. **The bar is the hand anytime frontier at equal decision time**, not exact `ib0` paying full c × plans. Hand arms:
   (a) `cd_ib0` = CD greedy on the ib0 surrogate (`peer_greedy_network_cd`, `HEROSIM_PG_BATCH_ALPHA=0`, no exact search: a flag combination that exists, since the alpha check
   accepts plain CD; it needs a harness kind, additive); (b) `cd_exactS_ib0` (exact, already in); (c) `cd_exactS_ib0_prune` (ib0 exact with cost-ordered top-k
   pruning of over-cap batches): **NOT RUN.** The alpha check refuses `HEROSIM_PG_BATCH_ALPHA` together with `HEROSIM_PG_CD_EXACT_PRUNE`, so it is not an existing flag combination; lifting that
   refusal is a code change that would need its own identity test, and it is listed here as not run rather than added. (Coordinator: say if you want the relaxation as a registered change.)
   Read: for each c > 0 the best hand arm's latency (per topology, then as the paired contrast) is the bar; a GNN arm WINS only if it beats the best hand arm at that c
   with the same rule (median ≤ −2 %, Holm p < .05, both rungs, the same arm). The best hand arm is chosen per c at the arm level (the one with the lower
   median over topologies at that rung), not per topology. Arm list is therefore: `cd`, `cd_ib0`, `cd_exactS`, `cd_exactS_ib0`, GNN self-search s1, `cdxapply` s1, plain `nosplit` s1 (7 arms,
   `cd_exactS` kept as the old bar). Cut grid c ∈ {0, 0.25, 1.0}: 7 × 3 × 48 = 1,008 cells, 912 new.
2. **Ground c.** Registered before any read, in `read_plan` of this node:
   - **Measured:** per-plan wall from the calibration table (wall per batch ÷ `sched_plan_evals` per arm, on the smoke cells), reported per arm, with the 3× rule from the signature.
   - **External anchor 1 (Kubernetes default scheduler):** kube-scheduler is expected to sustain ≥ 100 pods/s in a 5,000-node cluster (the scheduler_perf target), i.e. ≤ 10 ms per pod,
     and about 1,000 pods/s (≈ 1 ms per pod) isolated from etcd (kubernetes/kubernetes issue 78208 and the Preferred Networks scheduler benchmark write-up, found by search 2026-10-12).
   - **External anchor 2 (a research cluster scheduler):** Firmament (Gog et al., OSDI 2016, "Firmament: Fast, Centralized Cluster Scheduling at Scale"): task placement latency about 5 ms on a 40-machine
     cluster and hundreds of ms at ten-thousand-machine scale.
   - **Conversion, fixed now:** an anchor is a per-task decision budget B. Our c is per plan, so the anchor in c is `c_anchor(arm) = B / (plans_evaluated per task of that arm at the crossover)`,
     with plans per task taken from the arm's own `sched_plan_evals / tasks` in the c = 0 cells. Both B values (1 ms and 10 ms from anchor 1, 5 ms from anchor 2) are reported; the decision line uses the
     largest. These are budgets of production schedulers on hardware and code unlike ours: a reference-CPU model, as already disclosed. **Before the read, each figure is re-read at its source
     (the figures above come from search result text, not yet from the papers' pages) and the node's numbers are corrected if they differ.**
   - **Crossover c per GNN arm** = the smallest c in the grid (interpolated linearly between grid points on the paired contrast vs the best hand arm) at which the arm WINS. **If the crossover lies above every
     anchor (c_anchor of the arm), the result is NO-GO regardless of the Holm result.** If no grid point shows a win, the crossover is "above 1.0 ms" and the result is NO-GO.
3. **Signed arms stay** (the existing checkpoints). An **ib0-labelled GNN** (labels = argmin of ib0's surrogate; cheap, can cover over-cap batches; built only on the uncapped candidate set) is a registered
   **SUCCESSOR**, run only if s0 shows a crossover at or below an anchor. Nothing is trained in this node.

2026-10-12 — **Calibration read and Amendment 3 (coordinator's ruling; committed before the smoke).**
- **Calibration (flag-off identity cells, 19405 and 19411, g0, new code 84415895; wall = `decisionTiming.total_s` ÷ `sched_plan_evals`).** 20/20 cells IDENT to the gate cells of d192081f on
  total_rtt, mean latency, queue, exchange, cold start, endTime, num_tasks and the search counters. Measured (pooled over the two topologies): `cd_exactS` 521 plans/task at 0.0665 ms/plan moderate,
  652 at 0.0586 heavy (34.6 / 38.2 ms per task); `cd_exactS_ib0` 499 / 625 plans at 0.0688 / 0.0617; `cd` 13–18 plans at 0.008–0.010; `cdx_ib0` 24–30 plans at 0.025–0.029;
  `ra_gnn_eng_cdxib0` 20–27 plans at 0.16 (heavy) / 0.26 (moderate) ms/plan, 4.3–5.2 ms per task. Decision wall over simulated run: `cd_exactS` 96 % at heavy (3,818 s of 3,985 s), 37 % at moderate.
  Wall per plan spans ~30× across arms: the 3× rule fires and is stated. The unit "plan" is not one unit of work (the GNN's wall is its forward pass, counted as one plan per task).
- **Amendment 3 (supersedes the common per-plan c).** A common per-plan c is unfair in both directions once per-plan wall spans 30×. The cost model is **arm-calibrated**:
  `t_batch = k × w[arm][rung] × plans_evaluated`, with `w` the measured ms per plan per arm and rung from the calibration table, fixed in `scripts_cosim/decision_cost_w.json` before any cell counts and
  recorded here. `k = 1` is "as measured on the cluster CPU"; k scales to a reference machine. The GNN's w is CPU forward-pass wall (conservative against the GNN; disclosed).
  The harness kind suffix is `_k025 / _k050 / _k100 / _k200` (`HEROSIM_DECISION_COST_MS = k × w` for that arm and rung). The per-plan c table above stays in the record for reference only.
- **Grid:** k ∈ {0, 0.25, 0.5, 1, 2}; 8 arms × 5 × 48 = 1,920 cells (k = 0 exists for `cd`, `cd_exactS`, `cd_exactS_ib0`, `cdx_ib0`-class cells already). The whole grid runs; k = 2 is cut for the
  classical arms only if cost forces it, and that is stated.
- **Anchors convert to k per arm:** `k_anchor(arm) = B / (w_arm × plans per task at k = 1)`. Exact search (35–38 ms/task) vs the 1–10 ms Kubernetes budget: k ≈ 0.03–0.3; the GNN at 4–5 ms/task: k ≈ 0.2–2.
  **Decision line:** a GNN WIN counts only if its crossover k is ≤ the largest anchor's k for the HAND arm it beats (the hand arm would have been within budget at that k); otherwise NO-GO as registered.
- **Utilisation:** reported per cell (`sched_busy_seconds` ÷ simulated end time); > 1 is flagged unstable and read as such. It is a result: exact search at k = 1 is ~0.96 utilisation at heavy on one core.
- WIN rule and families are unchanged, read per k. Smoke: one cell per GNN arm at k = 1, counters shown; launch when the oracle frees slots.
