# plan_rerank_v1 — can a learned plan-level scorer beat `cd_exactS` in the accel seat, and is there anything left to learn?

**Status:** `REGISTERED` (2026-10-11), pending the coordinator's signature. Owner S4. Drafted from the coordinator's brief and its
amendment after SCIENTIST's review. Nothing trains before the Phase 0 read.

**Now.** Node only. Phase 0 reader and harness are not yet committed; no live data.

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

- 2026-10-11 — Signed (53381522). Amendment before data: sign condition ≥ 5 of 6 topologies at each rung added to the Phase 0 GO bar. Overnight
  autonomy: the cost smoke needs no sign-off. It becomes a gate cell only on the gate code with the gate settings, otherwise it is labelled a smoke and kept
  out of the read. The coordinator is consulted only for the cell-size cut (with plans per core-hour measured) or a physics-touching hook change; a fidelity
  snapshot must add no SimPy event to the live run, and flag-off identity is shown on 2 cells.
- 2026-10-11 — Registered. Phase 0 added by the coordinator's amendment after SCIENTIST's review; the original top-10 reranker is replaced
  by the Phase 1 design above.
