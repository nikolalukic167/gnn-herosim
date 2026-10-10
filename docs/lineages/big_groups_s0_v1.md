# big_groups_s0_v1 — do bigger peer groups make exact search on CD's surrogate fail, at fixed load?

**Status:** `REGISTERED` (2026-10-11). Every bar below was signed before any data.

**Why.** In the accel seat the decision is too easy for a learned model to beat search (SCIENTIST, PUBLISHER, S4–S7,
2026-10-11; recorded in `selfsearch_confirm_v1` and `accel_nosplit_v1`).
- **Small batches:** median 3 tasks, max 10; 0 % of batches exceed the 100k cap after the top-5 slate.
- **The surrogate is nearly right:** argmin-S misses the label optimum in 21 % of 413 held-out tables, with pooled
  regret 6.1 %. The platforms are 95–99 % idle.
- **Exact search on S, the GNN-seeded search and GNN self-search all tie** at about −10 % vs CD.
- **S's misses grow with size and coupling:** the miss rate is 10 % at 2 tasks against 38–55 % at 7–10 tasks, and 10 %
  at 1–2 peer pairs against 44 % at ≥ 10 pairs. Pooled argmin-S regret is 1.5 % at 3 tasks against 11–13 % at ≥ 6.
- **What S misprices:** shared-link exchange (S prices each pair alone on its path) and cold starts. Queue never.
- This is the one non-additive coupling already inside validated physics. It rarely binds today only because groups
  are small.
- **Prior to disclose:**
  - at 6 servers, bursts acted as a load lever (`burst_groups_v1`, `burst_ladder_v1`), so this screen holds load fixed;
  - in the 32-task seat, route-aware hand search captured most of the headroom (`g32_*`).

## Design (fixed now)

- **Environment:** the accel seat, unchanged: R1.1, fastest_compatible, 160c × 24s, the same apps, payload model,
  pipelined transfers and KPA.
- **The one change:** the size of peer groups dispatched together, **G ∈ {current, 8, 12, 16}** tasks.
- **Load is held fixed:** the same total task arrival rate per rung (moderate, heavy), so a larger G means fewer,
  larger bursts.
  - The owner states how the workload generator implements G, and shows the task rate matches within 2 %.
- **Topologies:** 8 development ids from an unused range, disjoint from every gate pool (16251–16258, 16301–16399,
  16490–16523) and from the training corpora.
- **States:** decision states captured from CD runs at each (G, rung), about 50 per cell. Captured with replay G
  through the fidelity path, with the autoscaler-gate and monitor-phase fixes of rp/cost-to-go (daf74ffa, a96b8f27,
  8e19fcab).
- **Reference plan:** full enumeration is impossible at G ≥ 8. The reference is therefore the **best-known plan by true
  replayed batch latency**:
  - start from argmin-S, cd_exactS's plan and the GNN self-search plan;
  - apply exhaustive single-task moves and pair swaps over the top-5 slate, scored by replay, to a fixed budget per
    state that the owner sets before any read;
  - the budget is the same for every G.
- **Measures per (G, rung):**
  - **M1:** S-regret, the pooled (Σ latency of the argmin-S plan − Σ best-known) / Σ best-known. cd_exactS's served
    plan, including its fallback, is reported alongside.
  - **M2:** the top-2 gap, between the best and second-best distinct plans found. Report the median, and the share of
    states within 1 % and within 5 %.
  - **M3:** the cap rate, the share of batches whose slate plan count exceeds 100k.
  - **M4 (fair bar):** S′ regret. S′ is S plus a hand shared-link term: each pair's exchange is priced with the
    bandwidth divided by the number of the batch's pairs routed over the same link.
  - **Health:** CD's run end over the last arrival, p95 latency and effective queue share at each (G, rung).
- **GO bar (to a learned arm in the big-group regime):** at some G ≥ 8, at both rungs, all of:
  - M1 ≥ 12 % (twice today's 6.1 %);
  - M4 ≥ 8 %, so the obvious hand fix does not close it;
  - median M2 ≥ 2 %;
  - CD healthy: end ≤ 1.10 × the last arrival.
  - Otherwise **NO-GO**, and the regime is recorded as not separating search from learning.
- **Live gate (rule 6, it runs either way):** at the G closest to 12 that keeps CD healthy, on 12 fresh topologies × 4
  windows × both rungs:
  - the current GNN self-search (ra_gnn_eng s1, no retraining) vs CD and cd_exactS;
  - CD as the bar, read as `accel_nosplit_v1` was.
  - A GO adds a retrained arm on a big-group corpus as its own registered successor.
- **Owners:**
  - S6: capture, replay tool and reference search;
  - S7: the G generator, the load check and the analysis;
  - S5: the S′ hand term.
- **Cost, estimated:** capture about 1 h; reference search about 2–4 h (replay-scored moves; the budget sets it); the
  live gate about 3–4 h.

## Record (newest first)

- 2026-10-11 12:40 — **Search budget fixed before any data** (S6 plan, coordinator OK).
  - **Reference:** best-improvement local search over the top-5 slate, scored by the co-sim per-plan engine (the corpus
    label quantity, about 0.4 s per plan). Moves are single-task moves plus in-slate pair swaps.
  - Starts: argmin-S (S-only local search above the cap, marked), cd_exactS's plan, and GNN self-search if available.
  - **B = 600 replayed plans per state**, the same at every G. Every replayed plan is kept for M2.
  - **Safeguard:** if fewer than 50 % of states converge before B at any G ≥ 12, 20 seeded states rerun at B = 1,800.
    If M1 moves by more than 2 points, that G is labelled UNDER-SEARCHED and no NO-GO is drawn from it.
  - The scorer must reproduce held-out corpus labels within 0.1 % before use.
- 2026-10-11 12:00 — **Registered** (coordinator, on the user's "yes, please").
