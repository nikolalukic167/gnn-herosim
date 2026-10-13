# decision_cost_s0_v1 — when the scheduler's decision time is charged as simulated time, which search still pays, and does the GNN's cheaper search beat `cd_exactS`?

**Status:** `ACTIVE` (2026-10-13, owner S4). Interim whole-run read on the completed arms; prune incomplete; steady-state and attribution follow-ups running. Code on branch `rp/decision-cost` (follow-up pin `rp/dc-steady`).

**Registered frontier verdict (interim, WHOLE-RUN ONLY): `FRONTIER-WIN-AT-k=1`, `DECISION-LINE-AMBIGUOUS`, `ATTRIBUTION-PENDING`.**
- `cdxapply` (GNN seed + α-expansion) beats the best hand arm at k = 1 at both rungs (−3.57 % / −2.96 %, 12/12 and 11/12, Holm .002 / .024); no other learned arm wins at any k; `ra_gnn_eng_selfsearch` at k = 2 heavy is a **collapse** (util .995), not a number.
- **Decision line, both readings, as stated:** strict (each rung's beaten hand arm must be in budget at the crossover) → **NO-GO**; permissive (largest anchor over the arms beaten) → **GO**. We do not pick the convenient reading after the data.
- **Deployability caveat:** `cdxapply`'s own k_anchor is 1.41 (moderate) / 1.70 (heavy) at B = 10 ms, i.e. in budget at k = 1 only at the largest budget; at B = 5 ms (0.70 / 0.85) and 1 ms (0.14 / 0.17) it is out of budget. Against the best IN-BUDGET hand arm (`cd_ib0`, k_anchor 75 at B = 10) at k = 1 it is −3.00 % (11/12, p .0024) moderate and −2.96 % (11/12, p .012) heavy (Holm over the four primary tests ≈ .0096 / .037, by hand).
- Prune is incomplete and disclosed in every table; at moderate k = 0.25 it could still beat `cd_exactS_ib0` at k = 0 by > 2 % (unknown until its cells land).
- Whole-run includes the cold start from zero replicas; the steady-state lines (`after_360s`) from `steady_95202b74` sit beside each k = 1 number when they land.

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

2026-10-12 — **Anchors-table note (S7's surrogate-gap frontier, `rp/surrogate-gap` 066832e0, `/share/nikola.lukic/surrogate_gap/frontier_v1`; relayed by the coordinator, figures not re-read by S4).**
On n ≤ 6 full-candidate states: `ib0` S alone has mean regret 4.1 % heavy / 1.4 % moderate; `ib0` S plus 4 exact evaluations is within 1 % on ≥ 96 % of states; the GNN plan-score top-1 has mean
regret 9.8 % heavy / 27 % moderate (heavy-tailed) and needs k = 8 to match. Costs: S pass ≈ 0 s; GNN plan score 0.8 ms/plan; an exact replay 0.05 s each, **but building a replay for a fresh
state costs 5.6–7.7 s**, so a live oracle or exact-evaluation arm pays the build on every decision. That build cost is why "exact evaluation" is not a live option and is not an arm here
(the oracle in `ib0_confirm_v1` is an offline ceiling). Arm list unchanged. Before the read, the figures are re-read from the frontier file and corrected here if they differ.
- 2026-10-12 — **Held-out evaluation-budget frontier (S7, `/share/nikola.lukic/surrogate_gap/frontier_v1_heldout/read.{txt,json}`, code `rp/surrogate-gap` cfdec05d; quoted from S7's message, file not yet re-read by me).** 373 held-out states (heavy 191, moderate 182), pooled n = 2–6, mean regret % at k = 0/2/4/8/16/32 exact evaluations: heavy `ib0_s` 4.32/1.64/0.34/0/0/0, GNN 12.05/4.00/2.86/0.38/0.06/0.03; moderate `ib0_s` 1.73/0.75/0.04/0.01/0/0, GNN 25.84/22.63/16.90/3.11/2.25/2.24. `ib0_s` within 1 % at k = 4 on 96 % (heavy) / 99 % (moderate) of states. Caveat: 25 held-out n = 4 states errored (hook never ran) and are co-location-heavy (multi-node optimum 100 % / 88 % vs 21 % / 46 % scored), so n = 4 is optimistic; held-out n = 5/6 are thin. Supersedes nothing: the figures are re-read from the file before the final decision read.
- Evaluation-budget frontier of the hand surrogate (S7, `rp/surrogate-gap` cfdec05d; all-source 519 states and held-out 373): ib0 S + 4 exact replays puts 96 % (heavy) / 99 % (moderate) of held-out states within 1 % of the brute-force optimum, 8 replays 100 % / 99 %; GNN re-rank needs 8; ICM + alpha-expansion stalls at about 1–2 % mean regret.
- An exact evaluation is a lookup (replay == label bit for bit); wall per replay 0.05 s steady, 5.6–7.7 s first call on a fresh state. Caveat: 25 held-out n = 4 states (co-location-heavy) errored and are missing, so n = 4 is optimistic; n ≥ 7 not screened.
- See `docs/lineages/decision_cost_s0_v1/surrogate_frontier.md`.

2026-10-12 — **Amendment 4 (coordinator's rulings, registered before any grid read).**
1. **Arm list is final at 8:** `cd`, `cd_ib0`, `cd_exactS`, `cd_exactS_ib0`, `cd_exactS_ib0_prune`, GNN self-search s1, `cdxapply` s1, plain `nosplit` s1. This supersedes Amendment 2's "`cd_exactS_ib0_prune`: NOT RUN" (it was run, per the coordinator's later ruling, via the cost-ordered prune accepted by the alpha check) and the 7-arm grid text above. Grid launched at pin 03615acd: 12 topologies 19405–19416 × g0,g1 × moderate,heavy, k ∈ {0.25, 0.5, 1, 2} for all 8 arms and k = 0 for the five arms without a classical-gate cell (1,776 cells; jobs 864139–864146; OUT `/share/nikola.lukic/decision_cost/grid_03615acd`; reader `scripts_cosim/decision_cost_read.py`, written before the data).
2. **Whole-run statistic, disclosed.** The summaries carry no latency by arrival time (only whole-run means, percentiles and four-quarter queue means), so the decision_cost read is whole-run; no rerun. The live statistic includes the cold start from zero replicas.
3. **Steady-state follow-up, registered now:** k = 1 only, arms `cd_exactS_ib0`, `ra_gnn_eng_selfsearch`, `ra_gnn_eng_nosplit` (144 cells = 3 × 48) at a pin containing S6's summarizer c81f0e88 (`after_360s`, `after_p5`, per-task .npy), run after the grid read. Reason: the opening is where batches are largest and search is charged most, so the whole-run contrast may overweight it. Descriptive, no bar.
4. **Prune watchdog kills are a limit artifact.** The first kill (`cc40s19407 g0moderate cd_exactS_ib0_prune k050`, "projected 12,166 s > 1.5 × 7,200 s") came from my LIMIT = 7200, not from the physics. Killed prune cells are relaunched ONCE with LIMIT = 21600 into the same OUT after the prune job ends, counted and disclosed. Any prune cell (k = 2 in particular) still projected beyond 21600 s is recorded NOT-RUN with its projection and not raised again.

2026-10-13 — **INTERIM read on the completed arms (ordered by the coordinator before prune finished; whole-run; reader `decision_cost_read.py` 69abb8a2, job 864514, `/share/nikola.lukic/decision_cost/read_864514.log`).** Grid 1,593 of 1,776 cells at the time; `cd_exactS_ib0_prune` incomplete and therefore not in the best-hand-arm pick. Statistic as registered; negative = first arm faster; moderate / heavy.
- **Frontier vs the best hand arm** (bar moderate / heavy): k = 0 exactS_ib0 / exactS_ib0: self-search +3.15 / +7.72, cdxapply +5.88 / +3.97; k = 0.25 same bars: +2.13 / +4.70, +4.23 / +2.40; k = 0.5 exactS_ib0 / cd_ib0: +3.41 / −1.39, +5.40 / −4.30; **k = 1 exactS_ib0 / cd_ib0: self-search −5.24 (11/12, Holm .0044) / +1.70 (3/12, .077); cdxapply −3.57 (12/12, .002) / −2.96 (11/12, .024) = WIN at both rungs**; k = 2 cd_ib0 / cd_ib0: self-search +1.08 / +18,259 (heavy collapses), cdxapply −2.29 (11/12, .003) / −1.37 (10/12, .042) = no win (heavy above −2). Crossover: self-search none; cdxapply grid k = 1, interpolated 0.94, a one-point window.
- **Decision line is ambiguous and unruled:** cdxapply's moderate bar is exactS_ib0 (largest anchor k 0.236, B = 10 ms), its heavy bar is cd_ib0 (75.5). Strict (each rung's beaten arm in budget at the crossover): NO-GO. Permissive (largest anchor over arms beaten): GO. The node says "the hand arm it beats" (singular); the ruling is the coordinator's.
- **Kill check 2 (pick D), same arm k = 1 vs k = 0:** `cd_exactS` +9.53 % moderate (0/12) / +197.4 % heavy (0/12); `cd_exactS_ib0` +11.13 / +285.0 %. At its largest anchor k (0.24) `cd_exactS` is +1.76 / +0.77 % (k = 0.25), under 3 %. Utilisation `cd_exactS` heavy 0.25 (k .5), 0.70 (k 1), 0.995 (k 2). `sched_queue_wait_seconds` reads 0 at every utilisation because the scheduler loop already serialises decisions (waiting appears as latency); read from the numbers, not traced in code.
- **Prune:** 57 summaries (moderate only), 12/h over the last 3 h; 34 failed (32 timeouts at LIMIT 7200 s, 2 watchdog kills at k = 0.5); relaunch rule of Amendment 4 stands; whatever still exceeds 21600 s is NOT-RUN.
- **Steady-state follow-up launched** at pin `rp/dc-steady` 95202b74 (= `rp/decision-cost` + c81f0e88 + e3d5e069 + `run_provenance` python_env / env_fingerprint in the summary), k = 1, exactS_ib0 / self-search / nosplit, 144 cells, jobs 864511–864513, OUT `/share/nikola.lukic/decision_cost/steady_95202b74`.

2026-10-13 — **Coordinator's rulings on the interim read.**
1. **Decision line:** both readings recorded verbatim above (strict → NO-GO, permissive → GO); verdict `FRONTIER-WIN-AT-k=1 (whole-run), DECISION-LINE-AMBIGUOUS, ATTRIBUTION-PENDING`. A win by an out-of-budget arm over an out-of-budget arm is not a deployability result: `cdxapply` is in budget at k = 1 only under B = 10 ms (above), `cd_exactS_ib0` is out of budget from k = 0.24, so the moderate win is over an out-of-budget bar; the in-budget contrast (vs `cd_ib0`) holds at both rungs. In-budget bar at every k ≥ 0.25 is `cd_ib0` (cd also qualifies and is slower). `cdxapply` vs `cd_ib0`, moderate / heavy: k = 0.25 −4.83 / −4.23, k = 0.5 −3.44 / −4.30, k = 1 −3.00 / −2.96, k = 2 −2.29 / −1.37 (but at k = 2 `cdxapply` is itself out of budget, anchors 1.41 / 1.70).
2. **Attribution amendment (Amendment 5), written before launch.** `cdxapply` is a GNN seed followed by α-expansion; the grid has no hand-seeded expansion on the same surrogate and no twin-seeded expansion, so nothing yet says a win is the GNN's rather than the expansion's (cf. `prune_g8`: WIN-FOR-PRUNING; `cd_expand` beats `cd` at k = 0). The k = 1 follow-up gains `cdxapply`, `cd_ib0` and `cd_expand` (hand `cd` seed + the same expansion; harness kind `cd_expand`), same 48 cells each (12 topologies × g0,g1 × 2 rungs), plus the MP-OFF twin seed + the same expansion **if the harness has it: it does not** (`ra_twin_eng_cdxapply` is not a registered kind — `R1A_CDX` has `ra_twin_eng_cdxpra` and `ra_twin_eng_cdxib0` only, although the twin checkpoints exist), so it is **recorded as the missing control and not built**. Signed predictions (coordinator): `cd_expand` ties `cdxapply` within 1 pp at k = 1 at both rungs (the win is the search's); the twin seed would be within 1 pp of `cdxapply`. Labels: `cd_expand` within 1 pp or faster → `SEARCH-WIN-NOT-GNN`; `cdxapply` faster than `cd_expand` by ≥ 2 % with Holm p < .05 at a rung → `GNN-SEED-WIN` at that rung; MP only if `cdxapply` beats the twin seed by ≥ 2 % Holm (cannot be assigned without the twin arm). `cd_expand` has no row in `decision_cost_w.json`; it is calibrated first (k = 0, 19405 / 19411, both rungs, job 864515) and its w added in a data-only commit on `rp/dc-steady`; the follow-up for these three arms runs at that commit (= 95202b74 + the w key), declared here.
3. **Kill check 2 for pick D: PASSED.** `cd_exactS` vs itself at k = 1: +9.53 % moderate / +197.4 % heavy (0/12, p .0005). Onset (same arm, k vs 0, moderate / heavy): k = 0.25 +1.76 / +0.77 %, k = 0.5 +1.71 / +8.89 %, k = 1 +9.53 / +197.4 %, k = 2 +289 % / collapse (util .995). `cd_exactS_ib0` the same shape (k = 0.25 +0.95 / +2.39, k = 0.5 +0.82 / +9.97, k = 1 +11.13 / +285.0 %).
4. **Prune adopted as recommended:** the job drains; one relaunch at LIMIT = 21600 for the 2 watchdog kills and the k = 1 / k = 2 timeouts; anything still over is `NOT-RUN`.
5. **Instrument:** `sched_queue_wait_seconds` is 0 at every utilisation because the scheduler main loop processes one batch at a time (`gnn/scheduler.py:486–492`: `yield env.process(collect)` then `yield env.process(process)`), so the single-server resource never has a second requester; waiting shows up as the next batch starting its collection late. No k-table depends on it (latency is the primary). Entry in `docs/gates/gate-tools.md`. The counter is not changed.
6. Whole-run only in every headline until `steady_95202b74` lands; the k = 1 whole-run values there are first checked against the grid (a mismatch is a stop).
