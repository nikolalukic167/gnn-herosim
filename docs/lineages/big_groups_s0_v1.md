# big_groups_s0_v1 — do bigger peer groups make exact search on CD's surrogate fail, at fixed load?

**Status:** `CLOSED` (2026-10-11) — **NO-GO; live G8: HAND-SEARCH-FASTER.** Registered 2026-10-11; every bar below was
signed before its data.

**Outcome.**
- **Screen NO-GO as registered:** at G=8 the M2 clause fails in both conditions. M2 fails partly by construction, a
  recorded design flaw, not re-defined after the read.
- **Live G=8 gate** (856483, 12 fresh topologies × 4 windows × 2 rungs, 288/288 cells, reader 1a0babe7):
  - **cd_exactS beats CD −9.15 / −7.71 % (12/12 at both rungs)**, even though its fallback binds in 28 / 49 % of batches.
  - **GNN self-search** (ra_gnn_eng s1, no retraining):
    - beats CD at moderate (−9.70 %, 11/12, p .0015);
    - is only −1.46 % at heavy (p .11);
    - ties cd_exactS at moderate (+0.51 %);
    - is **slower than cd_exactS at heavy (+5.52 %, 2/12, p .0024)**, where its search is 90 % coordinate ascent.
- **Bigger groups did not open a gap for learning.** CD's own surrogate stays the better guide even when exact
  enumeration fails. The screen's large S-regret is tail-only (median 0), and live it does not cost cd_exactS.
- One checkpoint (s1), one environment; not a statement about other seeds or arms. No retrained big-group arm (that
  would need a GO).
- Stage 2 was not run. The S′ shared-link term did not help.

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

- 2026-10-11 03:30 — **Live gate read at G=8: HAND-SEARCH-FASTER; lineage CLOSED** (S7; array 856483, 288/288
  summaries, 0 failed, nothing killed or rerun; reader 1a0babe7, written before the data; harness 79ea002c; topologies
  16900–16911; `/share/nikola.lukic/big_groups/gate_G8_read.json`).
  - **Paired % of mean latency** (per-topology median over 4 windows, median over 12 topologies; exact two-sided
    Wilcoxon, uncorrected across the 6 rows; negative = first arm faster):

    | contrast | rung | median | wins | p |
    |---|---|---|---|---|
    | self-search vs CD | moderate | −9.70 % | 11/12 | .0015 |
    | self-search vs CD | heavy | −1.46 % | 10/12 | .110 |
    | self-search vs cd_exactS | moderate | +0.51 % | 4/12 | .73 |
    | self-search vs cd_exactS | heavy | +5.52 % | 2/12 | .0024 |
    | cd_exactS vs CD | moderate | −9.15 % | 12/12 | .0005 |
    | cd_exactS vs CD | heavy | −7.71 % | 12/12 | .0005 |

  - **Per-topology outliers, self-search vs CD:** 16904 is +4.6 % (moderate) and +7.8 % (heavy); 16909 is about 0 and
    +4.6 %. The trigger (≤ −5 % vs CD at both rungs) is False.
  - **Search counts:**
    - Self-search: 22 % exact / 78 % ascent at moderate (319,162 batches), 9.5 / 90.5 % at heavy (306,825). 43–44 % of
      batches change the decoded plan.
    - cd_exactS falls back in 28.1 / 49.2 % of batches, enumerating about 4.4 / 4.1 billion plans.
  - The three smoke cells (16900 g0 heavy, all arms) are part of the 288, as agreed before the read.
  - **Ruling (coordinator):** CLOSED. The screen stands NO-GO, and the live gate confirms there is no learned lever at
    G=8.
    - The hand search on S beats CD at both rungs even with its fallback binding in half the heavy batches.
    - The GNN's own search, mostly coordinate ascent at G=8, falls behind it at heavy.
    - No retrained big-group arm is registered (it needed a GO).
    - Group size is not the missing hardness. Larger groups make enumeration fail, but S-guided expansion recovers.
- 2026-10-11 16:50 — **Live gate at G=8 set up** (S7).
  - Topologies 16900–16911, from a clean scan; selection `selected_gate.json`, verdict asserted.
  - Inputs 856368: the rate matches the current G (moderate 5.3470, heavy 12.7173 tasks/s), 6,250 groups of 8 per
    window.
  - Reader rp/big-groups 1a0babe7, committed before the data.
  - Harness 79ea002c. Arms ra_gnn_eng_selfsearch s1, cd s0, cd_exactS s0: 288 cells, sealed, LIMIT 20,000 s.
  - Smoke 856369: one heavy cell per arm, written into the gate directory. Only completion and timing are read before
    the full read.
- 2026-10-11 16:20 — **Post-hoc, descriptive only (does not change the NO-GO): M2 over latency-distinct plans** (S6; from
  the same search rows; `search_a30a0102/m2_latency_distinct_posthoc.txt`).
  - **Median gap to the first distinct latency (share ≤ 1 % / share ≤ 5 %):**
    - G8 heavy: 11.0 % (30 / 46 %) none, 5.1 % (25 / 50 %) top5;
    - G8 moderate: 3.5 % (35 / 53 %) none, 4.9 % (35 / 50 %) top5;
    - current heavy: 34.0 / 39.2 %; current moderate: 35.3 / 35.3 %.
  - Every G8 cell would clear 2 % on this definition. The landscape is flatter at G8 than at the current G.
  - Recorded so a future screen registers the distinct-latency gap from the start.
- 2026-10-11 16:00 — **Stage 1 read: NO-GO as registered; ruling** (S6; search 856340 at a30a0102, ILS, B = 600; 448/448
  rows, 0 errors; read once at a30a0102).
  - **Per (G, rung, condition):** pooled M1 [cd_exactS served] | M4 | median M2 (share ≤ 1 %) | M3 | per-state
    argmin-S regret median / p75 / states over 5 %:

    | cell | M1 [cd_exactS] | M4 | M2 | M3 | per-state |
    |---|---|---|---|---|---|
    | G8 heavy, none | 40.8 [65.0] % | 42.3 % | 0.00 % (62 %) | 82 % | 0 / 5.9 % / 14 |
    | G8 heavy, top5 | 52.2 [55.4] % | 60.4 % | 0.00 % (59 %) | 75 % | 0 / 12.6 % / 17 |
    | G8 moderate, none | 15.5 [57.8] % | 33.6 % | 0.03 % (62 %) | 68 % | 0 / 8.0 % / 17 |
    | G8 moderate, top5 | 20.9 [57.9] % | 36.0 % | 0.17 % (59 %) | 48 % | 0 / 16.6 % / 22 |
    | current heavy, none | 7.0 [7.8] % | 7.0 % | 7.10 % (39 %) | 9 % | 0 / 0 % / 5 |
    | current heavy, top5 | 130.8 [8.9] % | 137.0 % | 9.80 % (30 %) | 7 % | 0 / 7.1 % / 15 |
    | current moderate, none | 39.8 [46.5] % | 31.9 % | 6.84 % (39 %) | 9 % | 0 / 0.1 % / 12 |
    | current moderate, top5 | 56.3 [37.0] % | 55.0 % | 7.23 % (39 %) | 7 % | 0 / 15.9 % / 16 |

  - **Caveats, to be read with the numbers:**
    - (1) Pooled M1 is tail-driven. Without the single worst state: G8 heavy 21.6 / 33.9 %, moderate 11.6 / 17.1 %;
      current heavy top5 130.8 → 53.7 %.
    - (2) The top-5 condition is confounded: in 10–12 states per cell, cd_exactS's plan uses a candidate outside the
      dataset's declared slate. No slate is the clean read of S.
    - (3) The current-G M1 is far above the corpus's 6.1 %, but its median per-state regret is 0. These are heavy tails
      on fresh topologies, not a typical-state shift.
    - (4) S′ does not fix S (M4 ≥ M1 in most cells).
    - Search adequacy: G8 restarts improved in the last 20 % of B in 2–9 % of states, below the 30 % safeguard trigger.
    - The GNN start is absent.
  - **Failed runs, disclosed:**
    - check 856288 (1 ms dispatch offset; fixed 5e80b996, recheck exact);
    - search 856303 (missing sweep file; fixed 2aa94043);
    - 856334 cancelled unread for the ILS amendment.
  - **Ruling (coordinator):**
    - **NO-GO stands as registered.** M2 is not re-defined after the read.
    - M2's failure is partly a design flaw: latency-identical equivalent placements make the top-2 gap 0 in 25–32 of
      56 G8 states. Recorded as a lesson for future screens: measure gaps over latency-distinct plans.
    - A latency-distinct M2 may be reported as a disclosed post-hoc descriptive, and it cannot change the verdict.
    - **Stage 2 (G 12/16) is not run:** M1 rises with G only at heavy, and M2 as registered would fail by the same
      construction.
    - **The live gate runs at G=8**, the only G with measured CD health, as the registered "closest to 12 that keeps
      CD healthy". Arms: the current GNN self-search (ra_gnn_eng s1, no retraining), CD and cd_exactS. 12 fresh
      topologies × 4 windows × both rungs.
    - Reason it is worth it: cd_exactS's served plan is 55–65 % above best-known at G8, because its fallback binds in
      48–82 % of batches. That is exactly where a learned search might differ.
- 2026-10-11 15:30 — **Amendment before any measure is read (coordinator):** iterated local search for the reference.
  - **Why:** the search smoke (856332; only errors, convergence and timing read) converged in 1 sweep after 33–43
    replays at G8. A one-descent local optimum would bias M1 low at G8.
  - **New rule:** after convergence, perturb the best plan (reassign ceil(n/4) seeded-random tasks to another
    candidate) and descend again, until B = 600 is spent; keep the overall best. The rule is the same at every G and in
    both conditions. Restart counts and restart-found shares are reported.
  - **Safeguard replaced:** if restarts still improve the best plan in the last 20 % of B for more than 30 % of G8
    states, rerun 20 seeded states at B = 1,800. If M1 moves by more than 2 points, that G is labelled UNDER-SEARCHED.
  - Failed runs so far:
    - search 856303 (dry-run datasets lacked the sweep file; fixed 2aa94043);
    - 856334 cancelled before any read for this amendment.
- 2026-10-11 15:10 — **Health: CD is healthy at every (G, rung)** (S7; 856296/856297, clean gate harness at ae113145,
  `cd` s0; 16700–16707 × 4 windows; 128/128 cells, 0 failed).
  - Run end over the last arrival: median 1.006–1.018, max 1.044 (bar ≤ 1.10). Effective queue share ≤ 0.67.
  - **Paired, G=8 vs current:**
    - moderate: mean RTT ratio median 1.13, p95 ratio 1.07;
    - heavy: RTT ratio 1.04, p95 ratio 1.03;
    - effective share −0.02 / −0.05.
  - Load did not move with G. A g0 cross-check against S6's capture raw is pending.
- 2026-10-11 14:40 — **States built; M3 and slate sizes read before the search** (S6; capture 856295 32/32; build
  856302 at 16778f28; label-reproduction check 856294 381/381 exact after the 1 ms dispatch fix 5e80b996; the first
  check 856288 failed on that offset).
  - **States:** 224 per condition, 56 per (G, rung). The no-slate set uses the same snapshots.
  - **M3, the share of batches over 100k plans:**

    | G, rung | top-5 slate | no slate |
    |---|---|---|
    | current, heavy | 7 % | 5/56 |
    | current, moderate | 7 % | 5/56 |
    | 8, heavy | **75 %** | 46/56 |
    | 8, moderate | **48 %** | 38/56 |

  - **No slate barely adds options:** median eligible candidates per task is 4–6 (p90 9–10, max 13–22). The 5-limit
    is near the natural candidate count.
  - The search (B = 600) was submitted on GO. The GNN start is absent and flagged.
- 2026-10-11 13:50 — **Inputs built** (S7; scan 856289 clean; builds 856291/856292 at rp/big-groups ae113145).
  - Topologies 16700–16707 × both rungs × g0–g3, on the accel base cfg (fastest_compatible asserted).
  - **Task rate identical across G:** moderate 5.3470, heavy 12.7173 tasks/s.
  - **G=current:** mean group size 3.91–4.09 (max 10), 1.30–1.33 pairs per task.
  - **G=8:** 6,250 groups of exactly 8, 1.71 pairs per task.
  - Cells list: `/share/nikola.lukic/big_groups/cells.txt`.
- 2026-10-11 13:25 — **G generator built and checked** (S7; `--group-size G` in `grounded_workload_v1_mint.py`,
  rp/big-groups 129099b3).
  - G=current re-mints the four ×1 windows byte-identical.
  - The realised task rate equals the base rate within 0.01 % at every G. G=8 gives a mean group size of exactly 8.00.
  - **Disclosed, these change together with G:**
    - (a) group spans: today's groups sometimes spread over up to 70–298 s (median 2 ms), while G=8 groups span
      ≤ 26 ms;
    - (b) partner pairs per task rise 29 % (1.71 vs 1.30–1.33), because each member still draws 2 partners.
  - So G=8 means bigger, simultaneous and denser groups. The screen reads them as one lever, and no attribution to
    size alone is made.
- 2026-10-11 12:50 — **Amendment before any data (coordinator, on the user's "isn't 8 enough? remove the 5 limit"):**
  - **Stage 1:** G ∈ {current, 8} only, each in two slate conditions:
    - (i) the declared top-5 slate;
    - (ii) no slate, where every eligible candidate per task is available to argmin-S, cd_exactS and the reference
      search. Report candidates per task.
  - **Stage 2:** G = 12 and 16 run only if Stage 1 is not GO and M1 rises from current to 8, or if G=8 is labelled
    UNDER-SEARCHED.
  - The GO bar applies unchanged at G = 8, under either slate condition, read separately and labelled.
  - The B = 600 cap and the safeguard apply to both conditions. Under no slate, a sweep has n × (c − 1) single moves.
  - The live gate uses the G and slate condition the screen selects, by the same rule.
- 2026-10-11 12:40 — **Search budget fixed before any data** (S6 plan, coordinator OK).
  - **Reference:** best-improvement local search over the top-5 slate, scored by the co-sim per-plan engine (the corpus
    label quantity, about 0.4 s per plan). Moves are single-task moves plus in-slate pair swaps.
  - Starts: argmin-S (S-only local search above the cap, marked), cd_exactS's plan, and GNN self-search if available.
  - **B = 600 replayed plans per state**, the same at every G. Every replayed plan is kept for M2.
  - **Safeguard:** if fewer than 50 % of states converge before B at any G ≥ 12, 20 seeded states rerun at B = 1,800.
    If M1 moves by more than 2 points, that G is labelled UNDER-SEARCHED and no NO-GO is drawn from it.
  - The scorer must reproduce held-out corpus labels within 0.1 % before use.
- 2026-10-11 12:00 — **Registered** (coordinator, on the user's "yes, please").
