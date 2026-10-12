# plan_rerank_v1 — can a learned plan-level scorer beat `cd_exactS` in the accel seat, and is there anything left to learn?

**Status:** `CLOSED` (2026-10-12) — **SUPERVISED-CEILING-AT-BAR**. Registered 2026-10-11 by S4 and signed by the coordinator, from the coordinator's brief and its amendment after SCIENTIST's review. Every bar was signed before its data.

> ⚠ **Declared-slate defect (caveat added 2026-10-12).** The GNN's declared top-5 slate charges no cold start on an initialized platform; the simulation does. Learned-vs-hand-search contrasts here (`cd_exactS`, `cd_expand`, `cdxapply`, label-optimal plans) are confounded until `slate_bar_v1` (branch `rp/slate-bar`) reads. Contrasts against plain `cd` are not affected in the GNN's favour.

**Outcome.** This was a live Phase 0 test (rule 6) on 6 topologies, window g0. It served a label-optimal oracle plan per batch in place of cd_exactS's plan, and gained little:
- moderate −0.92 % (5/6 topologies, p .22);
- heavy −4.17 % (6/6, p .031).

The GO bar needed ≤ −4 % and ≥ 5/6 at both rungs, so it fails at moderate. Phase 1, a learned selector, did not run. Its code is staged and parked unread on rp/plan-rerank-p1.

A batch-label learner therefore cannot clear cd_exactS by a useful margin. One caveat applies: the oracle served only 56–61 % of tasks, with the rest on the over-cap fallback, so the ceiling is a partial-coverage ceiling.

**Side finding, descriptive:** a one-scalar hand change to CD's surrogate beats cd_exactS. Zeroing its in-batch stacking term (`cd_exactS_ib0`) gives −4.60 % / −3.18 % (6/6 each). It needs its own registered live gate before it can be quoted as a bar.

## Why

`accel_nosplit_v1` (2026-10-11 09:00 ruling, offline, 413 held-out tables, g3) found the GNN's deficit is **selection among plans it
already proposes**, not coverage:
- the GNN plan score's top-1 has pooled regret 24.6 %, worse than the served decode (20.2 %);
- the oracle best of its top-k reaches 15.2 / 10.5 / 6.2 % at k = 2 / 5 / 10;
- argmin-S has 6.1 %.

So the candidate set is as good as S, and a learned reranker over the top-10 can at best tie S's offline regret. The amendment
therefore asks first whether anything beats `cd_exactS` in this seat at all.

`gnn_proposal_selector_v1` does **not** cover this seat (fixed replicas, a hand-scored 3-proposal selector, no exact-S plan).
It is cited for one reason: its gain was proposal diversity, and the all-MP-off lesion kept −3.0 of −3.9 %, so the **MP-OFF twin is
mandatory** for any message-passing claim here. A hand selector scored by S over a pool that contains the exact-S plan returns
`cd_exactS`, so it is not a control. The hand control is "S plus the missing term" from S6's miss analysis, if that names one.

## Phase 0 — live oracle screen (runs first; nothing trains before its read)

**Question.** If each batch is served its true single-batch optimum, is the result at least 4 % faster than `cd_exactS` live?
If not, no learner trained on batch labels can beat `cd_exactS` in this seat.

- **Arm O (oracle):** per batch, enumerate the declared slate's plans (≤ 625 plans, the self-search cap), score each by the
  real simulator's single-batch RTT, apply the argmin. The RTT is the co-sim label, defined as `cd_exactS`'s regret is
  defined (S6's fidelity labels, `--fidelity`, whole group). Above the cap, fall back to `cd_exactS` and count it.
  - Myopic by construction: it is the best plan for this batch, not for the run. The read says so.
  - **Reuse first.** Before writing any labeller: check the fidelity replay (`src/placement/fidelity_replay.py`,
    `scripts_cosim/rollout_replay.py`), S6's label tables and ILS reference, `lookahead_mp_v1`'s oracle and `rp/cd-exacts`.
    Whatever exists is reused and named here.
  - **Reuse answer (S6, 2026-10-11):** no live arm labels a batch in-process. `big_groups_search.py` (`rp/link-bound-s0` 66f4dad3) is an offline
    replay search (`FidelityReplay(spec, types, offered).run(plan)`, then `cut_to_batch`). So Arm O needs a new live hook: a fidelity snapshot of the
    live state at each batch, then a per-plan replay pool. Its per-plan seconds on heavy states are not yet measured (S6's reproduction job
    will time `FidelityReplay.run`); cells are sized from that number, not from a guess.
- **Cells:** 6 fresh topologies **19300–19305**, windows g0 and g1, both rungs, accel seat (the `accel_replica_v1` gate
  environment, no-split declared slate). 24 cells per arm.
  - **Range check (2026-10-11):** `git grep` over `rp/prune-g8`, `rp/cd-exacts`, `rp/link-bound-s0`, `rp/selfsearch-confirm`,
    `rp/cost-to-go-mlpv`; `REGISTRY.json`; `LINEAGES.md`; `docs`; `/share/nikola.lukic/**` and `~/gnn-herosim/simulation_data`
    names: no 1930x hit (one numeric false positive, a `total_rtt` of 19306.3). 1910x clean by the same check.
- **Bars (sealed; reader committed first):** O vs `cd_exactS` and O vs CD, per rung: per topology the median over its 2 windows of the
  paired % of mean latency; median over 6 topologies; sign count; exact Wilcoxon (n = 6 gives p ≥ 0.031, disclosed).
  Negative = O faster.
  - **GO:** O is ≥ 4 % faster than `cd_exactS` (median paired ≤ −4 %) at **both** rungs, **and** O is faster on ≥ 5 of 6 topologies at each rung
    (coordinator amendment, signed before data: one odd topology must not decide a lineage). p is reported, not part of the bar.
  - **Otherwise** plan_rerank is closed **`SUPERVISED-CEILING-AT-BAR`**, and the record says no batch-label learner can beat
    `cd_exactS` in this seat. Phase 1 and the gate are not run for the screen's sake; the lineage still gets its closing note.
- **Prediction, written before any data:** O is 1–3 % faster than `cd_exactS`, short of 4 %. `cd_exactS` reaches the label optimum in
  72.6–79 % of batches and its regret is tail-only (argmin-S 6.1 % pooled), and a myopic single-batch optimum need not be the
  run-optimal choice.
- **Descriptive arms (coordinator amendments, before any Phase 0 data; no bar, same 6 topologies × rungs × g0/g1, no replay):**
  - `cd_exactS_a0`: cd_exactS with S's whole queue term scaled to α = 0 (`HEROSIM_PG_DRAIN_ALPHA=0`): the platform's own drain plus the service of
    the batch-mates the plan stacks on it, scaled in the greedy pass (`_pg_choose`) and in the whole-plan cost the exact search minimises
    (`_pg_plan_cost`). This is S6's train-selected S_α (10c9081a, variant "drain", α = 0): offline it cuts held-out argmin-S regret from 6.09 % to
    0.75 % pooled, so on the batch label a learned residual has at most about 0.75 % left over S_0. The slate (standalone-cost ranking) is not scaled.
  - `cd_exactS_ib0`: only the in-batch stacking part zeroed, the platform's existing backlog kept (`HEROSIM_PG_BATCH_ALPHA=0`, α = 1). S6's variant
    "batch": held-out pooled regret 2.01 % (trimmed 0.28 %), 76 % of the S → S_0 drop. The safer live version: the backlog signal should prevent herding.
  - Flag off (both alphas 1, the default) is byte-identical to the registered S; identity on the same 2 cells as O's.
  - **Reading, written now.** If O wins live and a0 also wins, the batch-label direction (more stacking) is right live, S_0 is the hand bar for
    Phase 1, and a learned residual has ≤ 0.75 % label headroom over it. If a0 loses live (herding), the batch label misprices queue for later
    arrivals, which explains the self-search queue loss. ib0 is read next to a0, cd_exactS and O. Neither changes O's GO bar.
- **Per-cell readout:** the same buckets as S7's self-search-vs-`cd_exactS` decomposition at heavy (confirm gate c1c80205,
  `/share/nikola.lukic/link_bound/gap_decomp_*.json`): queue, cold start and exchange as pp of the paired gap, and the share of the gap in
  the first quarter of the run. Reported for O vs `cd_exactS` so the read shows whether the oracle fixes queue and cold or only exchange.
- **Decision cost:** reported per batch (plans evaluated, wall seconds, fallback share). Nothing is cut to fit the budget without the
  coordinator; the size that can be afforded is reported first. Cost smoke before the cells: one heavy cell on the first topology,
  completion, timing, instrument counters only.
- **Process rules:** pinned worktree per job, watchdog by progress rate, a killed or failed cell is counted and never rerun, raw results
  kept on `/share`. Queue ≤ 8 jobs.

## Phase 1 — conditional on Phase 0 GO

The model is SCIENTIST's version, not the top-10 reranker.
- **Scorer:** every enumerated plan of the batch is re-scored by S plus a learned residual Δ, target `RTT − S` in seconds, then the exact argmin
  is taken. Plan-level GNN with the committed-plan encoding (`plan_raw`), all tasks committed.
- **Twin:** the same scorer with message passing off, **the same S feature, pool and corpus**, trained separately with the same recipe.
- **Loss:** soft listwise targets (softmax of −RTT/τ over the plan set; τ fixed before training), batches weighted by RTT spread.
- **Data:** the candidate table from S6 (full enumerated plans, true RTT, S per plan), plus states served by `cd_exactS` live (heavy included).
  Split by topology and window. **Checkpoint selection on a topology set separate from the gate.**
- **Training:** `run_experiment.py` configs under `experiments/`, epoch cap 100 (not lifted), W&B, the determinism test first,
  `.contract.json` sidecars.
- **Offline screen** (orders the work, never closes it): pooled regret of the learned top-1 on held-out tables, GNN and twin; GO if below
  argmin-S's 6.1 % on the same tables.
- **Live gate (rule 6, whatever the screen says):** 24 fresh topologies 19100–19123, accel seat, both rungs, 4 windows, sealed.
  Arms: rerank GNN (s1, s2), rerank twin, plain self-search, `cd_exactS`, CD. Controls: `cdxapply` (GNN-seeded CD) and, if S6's miss analysis names a term,
  S plus that term.
  - **Primary:** rerank GNN vs `cd_exactS`, per rung, Holm over the two rungs.
  - **WIN:** median ≤ −2 % and Holm p < .05 at both rungs, and no rung where `cd_exactS` is Holm-faster.
  - **MP claim:** only if the rerank GNN beats the rerank twin.
  - **Prediction:** a tie, because S's regret is tail-only.
- **Queue:** ≤ 44 in total; this lineage gets at most 8 jobs.

## Record

- 2026-10-12 — **Phase 0 read (once, all 12 oracle g0 cells in, 0 failed; reader plan_oracle_read.py at b05e99f5, output `gate_c7ceec53/read.json`). NO-GO: `SUPERVISED-CEILING-AT-BAR`.**
  - **O vs cd_exactS (g0, paired, negative = O faster):** moderate **−0.92 %**, 5/6 faster, p = 0.219; heavy **−4.17 %**, 6/6, p = 0.031.
    The bar needs ≤ −4 % at both rungs, so moderate misses and GO is False. Per topology, moderate: −0.1 / −1.3 / −3.4 / +1.3 / −3.5 / −0.6; heavy: −5.1 / −1.4 / −1.1 / −4.4 / −5.7 / −3.9.
  - **Prediction (1–3 % faster, short of 4 %):** held at moderate, missed at heavy (−4.17 % clears the bar on its own; the lineage still closes because the bar is both rungs).
  - **O vs CD:** −9.98 % moderate, −9.60 % heavy (6/6 each); cd_exactS vs CD −9.01 / −6.04 %.
  - **Coverage:** oracle served 77 % (moderate) / 75 % (heavy) of batches but 61 % / 56 % of tasks; over-cap fallback 18 % / 23 % of batches (36 % / 43 % of tasks); unplaced-partner pass-through 4.5 % / 2.0 %. Books close. So O is a partial oracle: about 40–44 % of tasks ran cd_exactS unchanged, and the bars are on the arm as run.
  - **Cost:** ~80 / 105 plans per served batch, 0.82 / 0.72 s wall per served batch, 18 % of served batches changed, 0.51 / 0.88 s of single-batch RTT saved per served batch.
  - **Buckets (pp of E's latency):** O − E moderate: elapsed −0.92 (exchange −0.33, lock_wait −0.55, cold −0.15); heavy: elapsed −4.17 (lock_wait −2.10, exchange −0.62, cold +0.04). Oracle plans stack slightly more (share of tasks sharing a platform 0.45 vs 0.38 moderate, 0.40 vs 0.36 heavy).
  - **Descriptive arms (g0 vs cd_exactS):** a0 −4.76 % (6/6) moderate, −2.40 % (4/6) heavy; ib0 −4.60 % (6/6) moderate, −3.18 % (6/6) heavy. g0+g1: a0 −4.00 / −0.85 %, ib0 −3.72 / −2.34 %. Both hand variants are as fast as O at moderate and ib0 is 0.8 % behind it at heavy (ib0 vs O: −2.94 / −0.84 %, not significant); a0 loses to O at heavy (+1.83 %).
    Per the reading written before data: batch-label stacking is the right direction live (a0 wins at moderate), but a0 gives back queue at heavy (lock_wait +5.2 pp, exchange −7.9 pp), so the herding worry shows at heavy only; ib0, which keeps the backlog, is the safer version.
  - **Consequence:** no batch-label learner can reach the −4 % bar at both rungs against cd_exactS in this seat, since a label optimum served as far as it is enumerable moves moderate by 0.9 %. A cheap hand change of S (ib0, one scale factor, no replay) matches or beats the oracle. Phase 1 and its live gate are not run; the offline screen is not run. Hand-S, ib0 / a0, is the lead for any follow-up and would need its own registered live gate (not a rerank).
  - First-quarter share of the gap: not read, raw per-run files were node-local and not kept.
  - Disclosed: heavy cells 19301–19305 ran at 60 replay workers and moderate cells at 28–30; worker count changes no decision. Oracle g1 cells were not run.

- 2026-10-12 — **Amendment before any oracle read: every oracle heavy cell runs with >= 60 replay workers (coordinator ruling, launch-resource fault).** The g0
  relaunch (860732-860736) gave shards 1-5 28 replay workers per cell (two cells per 60-CPU node), about half the throughput the design was costed on;
  shard 0 (60 workers) passed the same watchdog. Shards 1 and 2 heavy cells were killed by the watchdog as "projected": 19301 heavy 67/3933 arrivals at
  765 s (finish 55,791 s > 1.5 x 36,000 s), 19302 heavy 74/3933 at 915 s (56,561 s). The kill came from the launch's worker count, not from the arm;
  a failed launch is not a rerun. This holds only because no output of these cells, or of any oracle cell, has been read, and none has been.
  - Heavy cells 19301-19305 are (re)launched at 60 workers, one per job. Shards 3-5 were cancelled a few minutes in, before any watchdog verdict, and
    their cells relaunched (heavy at 60; moderate 19303-19305 together in one job). Shards 1-2 moderate cells continue at 28.
  - If a cell at 60 workers is killed by the watchdog, that is a real failure: counted, not rerun, and heavy is read at the surviving n
    (NO-GO or INCONCLUSIVE-AT-HEAVY only; the n = 6 sign condition cannot be met).
  - Worker count changes no decision: replays are separate read-only simulations.

- 2026-10-11 — **Amendment before any read: Phase 0 oracle on g0 only (cost cut, coordinator).** Measured on the sealed launch (pin c7ceec53, job 859987,
  24 oracle cells, 4 per node): replay 0.63–0.90 s per plan in heavy (0.25 s in a single-cell smoke), 0.23–0.35 s in moderate; 3.2–5.3 s (heavy) and
  1.5–3.1 s (moderate) of wall per served batch at ~95–170 plans, ~11k batches per cell. Projection: heavy 12–20 h, moderate 5–8 h per cell, past the
  watchdog's 1.5 × LIMIT. 859987 was cancelled after 10 minutes and its output is not read; the oracle is relaunched at the same pin on g0 only
  (860018, 12 cells, 6 nodes × 2 cells, 60 replay workers). A cost cut before any read, not a rerun.
  - All O contrasts (O vs cd_exactS, the GO bar, ≥ 5 of 6 signs) pair on g0 cells only.
  - Classical g1 cells stay as descriptive. The a0 and ib0 contrasts vs cd_exactS are read on g0 for the headline and on g0+g1 as a sensitivity.
  - Oracle g1 cells, if run afterwards, are a sensitivity that cannot change the GO verdict.

- 2026-10-11 — Signed (53381522). Amendment before data: sign condition ≥ 5 of 6 topologies at each rung added to the Phase 0 GO bar. Overnight
  autonomy: the cost smoke needs no sign-off. It becomes a gate cell only on the gate code with the gate settings, otherwise it is labelled a smoke and kept
  out of the read. The coordinator is consulted only for the cell-size cut (with plans per core-hour measured) or a physics-touching hook change; a fidelity
  snapshot must add no SimPy event to the live run, and flag-off identity is shown on 2 cells.
- 2026-10-11 — Registered. Phase 0 added by the coordinator's amendment after SCIENTIST's review; the original top-10 reranker is replaced
  by the Phase 1 design above.
