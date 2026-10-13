# uncut_groups_s0_v1 — does a learned scorer beat the time-bounded hand bar when Alibaba peer groups are served uncut and decision time is charged?

**Status:** `REGISTERED` (2026-10-13). Designed by SCIENTIST (pick D; SCIENTIST keeps the name) and drafted by S6 on the expert's request. SCIENTIST closed the design on 2026-10-13; the expert authorised registration once it was closed. Every bar below was fixed before data.
- decision_cost kill check 2 PASSED (S4 interim read, job 864514): `cd_exactS` loses +9.53 % (moderate) and +197.4 % (heavy) to its own decision time at k = 1, 0/12 at both rungs.
- **Authorised now:** the mint of the 12 fresh topologies and the uncut libraries from the raw shards, identities (a)–(m), and the hand-arm cells (150–350 CPU-h).
- **Labels and training are gated by this node's own kills** (the expert, 2026-10-13, after S4's attribution read): labels start only after K0 and K1 are read and neither fires. If K1 fires, there are no labels, and the lineage closes on the hand read.
- **Scope (measured on the gate windows):** the gate serves groups up to about 230 tasks (window max 135–226). 31.8–43.2 % of tasks sit in groups > 10 per window, and 1.2–3.5 % in groups > 100. The 2,876-task and other giant groups lie past the windows' first 50k tasks and are **not tested**: no outcome of this node may be quoted as covering groups in the thousands.
- **Post-data amendments (2026-10-13, after k = 0 cells started returning):** WALL-INFEASIBLE is a third health count; K0's load question is read on `cd`; the k = 1 hand bar is `cd`, `cd_ib0` and the prune if it is wall-feasible. See [Post-data amendments](#post-data-amendments-2026-10-13-flagged-post-data). **Result:** hand expansion run to termination is wall-infeasible on uncut heavy groups (projected ~33,000 s per cell), measured before any learned arm ran.
- **Labels are ILS incumbents, not optima.** For groups above the 100,000-plan cap, label quality is bounded by the registered convergence check (ruling 3). `placements.jsonl` for those groups is a headed sample, not a sweep (F; the expert's exemption of 2026-10-13).

**Why.** Every grounded and accel gate so far served Alibaba peer groups cut to at most 10 tasks. The library builder keeps only the first `--max-size 10` siblings of a fan-out (`scripts_cosim/grounded_workload_v1_extract.py:74–76`), and the served batch is capped at the cell's `batch_size` 10 (code ceiling 16).

Context, trace-wide over the mint's draw ([grounded_workload_v1](grounded_workload_v1.md)'s 2026-10-12 cut audit):
- **37.47 % of member tasks** sit in groups larger than 10, though only **7.76 % of groups** exceed 10 (the audit text says 7.5 %; its own table sums to 7.77 %, and the rebuilt libraries give 7.76 %);
- 23.97 % of tasks sit in groups larger than 20;
- the largest group has **2,876** tasks, and the mean uncut size is 5.18.

The audit is over the mint's draw, not the whole trace. **What the gate windows hold** (uncut ×1 mint, 2026-10-13, see the record): max group 199 / 163 / 226 / 135 (g0–g3); tasks in groups > 10: 36.71 / 34.62 / 43.21 / 31.75 %; > 100: 1.84 / 1.39 / 3.52 / 1.15 %.

So the joint problem studied so far is the easy part of the workload. On the big groups, exact search cannot enumerate. With decision time charged to the clock ([decision_cost_s0_v1](decision_cost_s0_v1.md)), the hand bar pays for its fallback search.

**Not a revival.** The hard-stops entry "Two-level Alibaba fan-out" (2026-10-12) closes a hidden second level revealed when the parent completes. This node serves the **first-level** fan-out uncut, fully known at arrival. **It never reveals later levels.**

**Prior and hypothesis (SCIENTIST).** Registered at P(WIN) ≈ 0.10, revised to 0.05 before any D data, then to 0.03 before any k = 1 data (earlier wordings in the record). Current, verbatim: "2026-10-13, SCIENTIST, before any k = 1 data, after the wall smokes: P(WIN) ≈ 0.03 (was 0.05); P(MP-attributed win) ≈ 0.01. GNN self-search is WALL-INFEASIBLE and leaves F1; the claim now rests on a one-pass argmax decode (nosplit), which in the capped seat at k = 0 was E-FASTER behind cd_exactS_ib0 by +3.6 / +7.1 % (slate_fix_v2). The only feasible exact arm, cd_exactS_ib0, is charged ~1.5 % of latency at scalar w (more with the frozen per-bucket w(n)), against nosplit's 0.13 %. Closing a 3.6–7.1 % plan-quality gap needs D's uncut-ILS labels to buy what the capped labels did not, or the bucketed charge on 11–226-task batches to be several times the scalar figure."
- Coupling stays sparse. Each member draws `min(2, size − 1)` partners (`grounded_workload_v1_mint.py:13–14`, `:140`), so a 2,876-task group is a graph of mean degree about 4, not a clique.
- On states of ≤ 6 tasks, S plus CD is within about 2–4 % of the true optimum on average (frontier_v1: ib0_s mean regret 4.06 % heavy / 1.39 % moderate, cd_s 2.67 / 1.99 %). Whether S stays that good on groups > 10 is unmeasured; the K1b replay diagnostic measures it.
- α-expansion's exact-optimum hit rate of 67.2 % (ICM 55.3 %; 1,017 whole-batch datasets of 2–8 tasks, true replayed RTT; `/share/nikola.lukic/workload_fix_v1/accel_state_stats/expansion_summary.txt`) is **not** evidence that hand local search is near-optimal: its misses are large (median +44 %, p90 +748 %).
- **Registered hypothesis:** any learned edge comes from decision **time**, not plan quality.

**Parents to read:**
- [grounded_workload_v1](grounded_workload_v1.md): the cut audit and the origin caveat.
- [decision_cost_s0_v1](decision_cost_s0_v1.md): the clock, k, w, utilisation, Amendments 2–3.
- [big_groups_s0_v1](big_groups_s0_v1.md) and [prune_g8_s0_v1](prune_g8_s0_v1.md): the G=8 fallbacks.
- [ib0_confirm_v1](ib0_confirm_v1.md): the bar.
- [warm_surge_s0_v1](warm_surge_s0_v1.md): ib0 runs away under a warm heavy surge.
- [slate_bar_v1](slate_bar_v1.md): the slate handicap.
- [slate_fix_v2](slate_fix_v2.md): after-360 s and whole-run side by side.

## Design (SCIENTIST's rulings, 2026-10-13; registered)

Marks: [V] SCIENTIST read it on reference-physics 18161421 or at source on datalab. The five [M] items of the first draft were closed by SCIENTIST on 2026-10-13: four verified, one (α-expansion) corrected.

**1. The bar.**
- **(a) Enumeration limit:** the existing 100,000-plan cap (`HEROSIM_PG_CD_EXACT_MAX_PLANS`, `peer_greedy_network/scheduler.py:170`, `:356`) [V]. ib0 was confirmed there and decision_cost's w was calibrated there. On top-5 slates it means exact search up to n = 7.
- **(b) Fallback: both expansion and prune.** The bar is decision_cost Amendment 2's hand frontier: the best hand arm per rung, chosen at the arm level by median.
  - Candidates: `cd`, `cd_ib0`, `cd_exactS` (expansion), `cd_exactS_ib0` (expansion), `cd_exactS_ib0_prune`, plus the expert's 2026-10-13 additions:
    - **`cd_expand`**, a hand seed with the same α-expansion as `cdxapply`;
    - **a time-bounded `exactS` prune at the same expansion limits as `cdxapply`**.
    These two exist so that a learned win cannot be the search's.
    - **Ruled by the expert (2026-10-13), no re-sign either way:**
      - if `cd_expand` beats `cdxapply` by ≥ 2 % (Holm) at a rung, `cdxapply` moves to the descriptives and `cd_expand` stays in the bar;
      - if they are within 1 pp, or `cdxapply` is ahead, `cdxapply` stays an arm. decision_cost's interim read has `cdxapply` beating the hand frontier at k = 1, whole-run (−3.57 / −2.96 %), on a grid with no hand-seeded expansion.
  - The prune arm is in because cost-order top-k prune beat expansion −1.89 / −2.11 % vs `cd_exactS` at G=8 (`prune_g8_s0_v1`) [V]. It needs the alpha×prune refusal lifted as a registered change (the expert's ruling), with its own identity against the current prune arm at alpha = 1 (8c), before any cell of this node uses it. Amendment 2 lists it as NOT RUN. **No WIN is claimable without it.**
  - `cd` and `cd_exactS` are bar candidates, not reference lines: under a warm heavy surge ib0 collapsed on 3/12 topologies at ×2 (19422, 19425, 19430) and 5/12 at ×3 (19422, 19425, 19426, 19429, 19430); `cd_exactS` on 1/12 at ×3 (19427), `cd` on 0 (warm_surge 0b, `/share/nikola.lukic/warm_surge/collapse_table.md`) [V]. The "5/11" in warm_surge's record line 169 is a denominator error. Uncut giant groups are surges.
- **(c) Time bound: no wall-clock cap.** Decision time is charged per decision_cost Amendment 3: `t_batch = k × w[arm][rung] × plans_evaluated`, one FIFO scheduler server, mutex held [V].
  - Expansion and prune run to their own termination and pay for it on the clock. That is the bound.
  - k = 1 (as measured on the cluster CPU) is primary, and is where kill check 2 reads. k = 0.25 is a secondary family; k = 0 is a reference.
  - **w is recalibrated for size.** On the smoke, report ms per plan by batch size (≤ 10, 11–50, 51+). If any bucket differs from ≤ 10 by more than 1.5×, fit `w(n) = a + b·n` per arm and rung and freeze it in `decision_cost_w.json` under a new key, before any gate cell.
- **(d) Collapse:** warm_surge's rule, adopted (gate-tools 2026-10-13 section on `rp/warm-surge`, not yet on reference-physics). Its thresholds sit in the 2.4×–8.1× gap of 576 cells. Rewritten for a whole-run seat:
  - **COLLAPSE:** a cell's mean latency is more than 3× the same topology × window × rung's `cd` mean, tested separately on whole-run and after-360 s;
  - **TAIL FLAG:** end minus last arrival is more than 120 s, counted separately;
  - the collapse count and the tail-flag count per arm are registered lines beside every median. SCIENTIST's earlier fallback (300 s / 5× hand median) is dropped.

**2. Arms.** One clock for all: k × the arm's own calibrated w × its own plans. Each arm serves the slate it was trained on (named per arm; `slate_bar_v1`).

| arm | role |
|---|---|
| GNN self-search, no-split engineered, s1 | primary |
| its MP-OFF twin, same search | attribution (required) |
| `cdxapply` (GNN seed + CD + expansion) | primary; decision_cost's predicted winner |
| MLP same-input self-search | secondary pointwise control |
| MP-OFF twin seed + `cdxapply`'s expansion | **built only if F1 shows a learned edge over `cd_expand`**. It is required before any MP attribution of that edge: the harness has no twin-seed + expansion arm (S4, 2026-10-13). Cost: 96 cells at k = 1, about 50 CPU-h by `cdxapply`'s wall, plus the harness wiring; no extra training (the trained MP-OFF twin's checkpoint). |

- **`cdxapply` contingency resolved (2026-10-13):** S4's decision_cost attribution read has `cdxapply` tying `cd_expand` within 1 pp on every metric at k = 1 (−0.21 / −0.38 % whole-run, −0.25 / −0.99 % after 360 s, Holm 1.0; SEARCH-WIN-NOT-GNN). By the registered rule `cdxapply` stays an arm, with no re-sign. `cd_expand` is confirmed as the in-budget hand arm the learned side must beat at k = 1 (k_anchor about 14 / 12 at B = 10 ms).

- **Revised 2026-10-13, before any k = 1 data (SCIENTIST, accepted by the expert):** GNN self-search and its MP-OFF twin's self-search are WALL-INFEASIBLE and leave F1 and F2. The primary learned arm is now the one-pass no-split decode; `cdxapply` stays only if its heavy smoke is feasible. The table above is the registered form.
- Plain no-split was dropped at registration (E-FASTER in slate_fix_v2, +3.6 / +7.1 % [V]); it returns as the primary by the revision above.
- Search machinery is unchanged: self-search is exact up to `SELF_SEARCH_EXACT_MAX` = 625 slate plans, coordinate ascent above (`gnn/scheduler.py:174`, `:1066`) [V]. There is no hand expansion inside the pure GNN arms.

**3. Labels.** ILS on ib0's surrogate S, not exact replay: exact search is impossible above n = 7, and a replay build costs 5.59–7.73 s on its first call per state (raw range 4.98–7.96 s) plus **0.04–0.21 s per evaluation, rising with n** (0.05 s up to n = 4 heavy, 0.21 s at heavy n = 6, p90 about 0.22–0.25) (frontier_v1 `read.txt` §wall and `walls.jsonl`, 30 states, replay = label exactly) [V]. Replay-label costs at n > 10 are therefore lower bounds.
- **Budget:** α-expansion plus random-restart perturbation, capped at 200 × n × (mean candidates per task) evaluations per group.
- **Convergence (registered):** on a 200-state sample, doubling the budget moves the label's S by less than 0.1 %. Otherwise the budget is raised until it does.
- **Ties:** lexicographic on (node id, platform id) per task in task order, never set order. Seeded; two label runs are byte-identical (8h).
- **S-fidelity (kill K1b's input):** exact replay on 200 sampled groups > 10 (about 0.5 CPU-h) gives argmin-S's true regret. The reference is ib0's S mean regret at n = 2–6, k = 0: 4.06 % heavy (245 states), 1.39 % moderate (274); medians 0.00 (frontier_v1 `read.txt`; 27 states logged as errors are excluded) [V]. If the mean regret on groups > 10 exceeds 2× that (8.1 % heavy, 2.8 % moderate), S labels are wrong where this node lives: switch to replay-evaluated top-m labels (m = 8).

**4. Seat and corpus.**
- **Seat:** accel 160 × 24, WF1, p 0.6, `fastest_compatible`, KPA, pipelined, `HEROSIM_REPLICA_RELEASE=1`, 1 s window. That is where the bar, ib0 and decision_cost's w live.
- **Not the 6-server grounded seat:** a 2,876-task burst there is a pure load lever.
- **Gate topologies: 19461–19472**, disjoint from 19420–19431 (warm_surge), 19405–19416 (decision_cost, ib0), 19440–19452, 19206–19217, 18700–18711 and earlier pools. Picked by gnn_overcap's token-scan protocol (record, 2026-10-13); the prep job fails loud on an existing ROOT.
- **Corpus topologies:** fresh, and disjoint from the gate set.
- **Windows:** the gate runs g0–g3, one per shard. Shard 100 is the heaviest, 43.5 % of tasks in groups > 10 [V].
- **Corpus requests: option (a), ruled by the expert 2026-10-13** (reversing the expert's own earlier (b), on SCIENTIST's objection that (b) touches the extract under identity (i)):
  - The corpus uses the **same libraries**, starting at group G0 = 1 + the maximum group index any gate window consumes at either rung under either mint (cut or uncut), + 500. It runs to the library's end.
  - G0 per shard: 0 → **12,720**; 50 → **13,295**; 100 → **12,954**; 130 → **13,271**. The gate consumes groups 0–12,219 / 12,794 / 12,453 / 12,770, the larger of the cut and uncut mints.
  - Implemented as a mint option `--start-group G0`, bit-identical at 0 (added to identity (a)). A zero trace-id overlap between gate and corpus mints is asserted in the mint, fail loud.
  - Gate groups are never reused. A shortfall against the training plan is said before labels.
  - **(b) is the registered fallback:** `--skip-traces` in the extract, defaulting to 0. At skip 0, `--max-size 10` must still reproduce each library's md5.
- **Corpus vs gate group sizes** (uncut libraries from G0 to the end, task-weighted). The gate windows are g0–g3 = shards 0, 50, 100, 130.

  | shard | corpus tasks | max | > 10 | > 20 | > 100 | > 226 | > 256 (groups) | gate window > 10 / > 100 |
  |---|---|---|---|---|---|---|---|---|
  | 0 | 141,448 | 836 | 34.93 % | 23.66 % | 3.34 % | 1.17 % | 0.99 % (3) | 36.71 / 1.84 % |
  | 50 | 134,858 | 1,406 | 36.18 % | 24.41 % | 4.28 % | 2.51 % | 2.16 % (5) | 34.62 / 1.39 % |
  | 100 | 148,498 | 469 | 43.77 % | 24.36 % | 2.13 % | 0.74 % | 0.58 % (2) | 43.21 / 3.52 % |
  | 130 | 136,655 | 2,876 | 35.27 % | 22.93 % | 4.15 % | 2.47 % | 2.29 % (2) | 31.75 / 1.15 % |
  | pooled | 561,459 | 2,876 | 37.65 % | 23.85 % | 3.44 % | 1.70 % | 1.48 % (12) | |

- **Giant groups (SCIENTIST, 2026-10-13):** the corpus **workload** is never capped or cut, since truncation is the cut this node removes and would change the arrival stream. The **label set** is capped instead.
  - Snapshot states whose batch exceeds **256 tasks** (the gate windows' max of 226, plus margin; fixed now from the gate's size draw, not from any outcome) are not labelled or trained on. They are counted and reported.
  - The pre-gate forward-time and memory checks run on the largest corpus group either way (the expert).
- **Cut groups are not excluded;** the uncut groups are the point.
  - State sampling is stratified so batches > 10 are at least 30 % of training states. Unstratified, the loss would ignore them: they are 7.76 % of groups but 37.5 % of tasks.
  - Stratification is reported by size bucket (≤ 10 / 11–50 / 51–100 / 101–256): the corpus's state share and task share beside the gate windows' task-weighted shares. Within > 10, sampling keeps each bucket's task-weighted share within a factor of 2 of the gate's (SCIENTIST).
  - MIN_TIME stays 360 s, matching the after-360 s line.
- **Load:** the uncut workload is re-stretched so tasks per second match each rung within 2 % (heavy 12.7173, moderate 5.3470 tasks/s). Groups per second then fall about 23 % (mean group size about 4.0 → 5.18; disclosed). Without the re-stretch, uncut means more load, and the contrast becomes a load lever.

**5. Statistic.** Paired per topology (median over windows of the paired %), then the median over topologies, the sign count, an exact Wilcoxon and Holm (`R.holm`).
- **Whole-run and after-360 s are both reported, and a WIN must clear both.** Steady state as the sole primary is breaking change #4, still PROPOSED and the user's decision (confirmed by the expert, 2026-10-13). Requiring both also guards against a win that lives in the opening, the way slate_fix_v2's self-search deficit did.
- **F1, primary, k = 1** (Holm over 4). Revised 2026-10-13 before any k = 1 data (SCIENTIST, accepted by the expert; registered wording "self-search vs the bar and `cdxapply` vs the bar, × 2 rungs"). Verbatim: "A learned one-pass decoder with no search, trained on ILS labels over uncut groups (≤ 256 tasks), beats the best wall-feasible hand arm at k = 1 because (i) the hand arms' in-budget plan quality on > 10-task batches falls short of the ILS label while their charged decision time grows with n, and (ii) the decoder's charge stays ~0.1 % of latency. It predicts the gap to the hand bar concentrates in the 11–50 and 51–226 buckets."
  - F1 arms: nosplit (and `cdxapply` only if its heavy smoke is feasible) vs {`cd`, `cd_ib0`, `cd_exactS_ib0`, prune if feasible}, with the robust-bar clause.
  - **WIN:** median ≤ −2 % and Holm p < .05 at both rungs for the same arm, **and** that arm's collapse count ≤ the bar arm's, **and** decision_cost's anchor rule: the crossover k is ≤ the anchor k of the hand arm it beats, else NO-GO.
  - **Robust-bar clause (SCIENTIST's rule, 2026-10-13, before data; signed by the expert):** where the median-chosen bar arm has a collapse count > 0 in a condition, WIN also requires clearing the WIN rule against the **best zero-collapse hand arm** in that condition, by median, with `cd` always a candidate. If no hand arm has zero collapses, the read says so and is made against `cd`.
    - A WIN read under this clause is labelled **WIN-ROBUST**.
    - A WIN that clears only the median-chosen bar while that bar collapsed is labelled **WIN-OVER-BRITTLE-BAR** and is never quoted as WIN.
    - Why: the bar is chosen by median, and a median hides a collapse tail (warm_surge 0b/0c, ALPHA-FIXES-COLLAPSE-CELLS-BUT-NOT-A-SAFE-KNOB: exact batch search collapses on a topology-dependent subset at every alpha tried; one-pass `cd` never did).
- **F2, attribution, k = 1** (Holm over 4), × 2 rungs. Revised 2026-10-13 before any k = 1 data (registered wording: self-search vs its MP-OFF twin and vs MLP same-input). Verbatim: "nosplit vs its MP-OFF twin nosplit; nosplit vs MLP same-input one-pass. Self-search leaves F2."
- **K1 note (SCIENTIST, 2026-10-13, before any k = 1 data):** "K1 is unchanged and is now the decisive gate: it asks exactly (i), whether the best feasible hand arm is within 1 % of ILS on big states and charged < 3 %."
- **F3, secondary, k = 0.25** (Holm over 4): F1's tests, Holm separately. No WIN is claimed from F3 alone.
- **Labels:**
  - WIN / TIES / E-FASTER (the bar Holm-faster at both rungs);
  - MP: MP-WIN / DIRECTION-ONLY (right sign, not Holm) / NOT-SEPARATED;
  - **UNCUT-ATTRIBUTED** (registered, descriptive): at least 50 % of the pp gap sits in batches > 10, by gnn_overcap's by-batch bucket method on g0 cells, bucketed by the group's **original** size (a split big group still counts as > 10). A win carried by batches ≤ 10 is labelled **WIN-NOT-UNCUT**, not a result for this node.

**6. Kill rules.**
- **K0, admissibility, before any learned work** (hand arms at k = 0, 12 topologies, both rungs; health on g0): `cd` and `cd_exactS` must be healthy (no collapse, effective queue share ≤ 0.80) on ≥ 10/12 at heavy. Otherwise the giant groups are a load lever: fall back to the moderate rung only, disclosed. If moderate also fails, **STOP (NO-SEAT)**.
  - **Health for every hand arm** (SCIENTIST's rule, 2026-10-13, signed by the expert): K0 reports collapse and tail-flag counts at k = 0 for `cd`, `cd_expand`, `cd_exactS` and ib0, not only `cd` and `cd_exactS`.
  - **Uncut admissibility** (SCIENTIST, read fixed before data): at k = 0 on the bar arm, at least 90 % of the tasks in groups > 10 are served in a whole-group batch, **pooled over the four windows**.
    - Per-window shares are reported; g2 (shard 100, uncut sibling span p99 6.6 s) is expected low.
    - `batch_timeout` is never raised to rescue g2: that would change the seat and break identity (a).
    - If the pooled share fails: stop, the seat isn't serving uncut.
    - If only g2 fails its own 90 %: F1 still uses all four windows, with a descriptive g0/g1/g3-only F1 beside it.
- **K1, before any label or training** (live hand arms at k = 1, plus a replayed read on a fixed sample). Exact thresholds, written before the hand cells are read: S6's proposal as amended and confirmed by SCIENTIST, 2026-10-13.
  - **The hand arm** is the best hand frontier arm at k = 1 per rung, by F1's statistic (median over topologies of the per-topology median whole-run latency). `cd_expand` is expected.
    - **It must have zero collapses at k = 1 in that rung** (SCIENTIST's rule, 2026-10-13, signed by the expert). If the median-best arm collapses, the best non-collapsing arm decides, so a brittle arm never decides HAND-SUFFICES.
  - (a) **Decision-time headroom:** the ratio (mean scheduler-queue wait + mean charged decision time per task) ÷ `averageElapsedTime`, median over that arm's 48 cells, is **< 3 % on both whole-run and after-360 s, at both rungs**. The wait and the charged time are read from the decision_cost counters in the summaries (sched busy / queue-wait), never reconstructed.
  - (b) **Plan headroom, on the replayed true RTT:** a fixed sample of 200 batch states with > 10 tasks from that arm's k = 1 trajectories.
    - Sampling: 25 per window × rung, seeded, drawn before any label exists. The state ids are written to a file in `docs/lineages/uncut_groups_s0_v1/` **before any k = 1 hand cell is read**, with the seed and the file's hash recorded here, so the draw cannot see the cells (the expert). Within > 10, stratified over 11–50 / 51–100 / 101–226 in proportion to the gate's task-weighted bucket shares, and reported per bucket.
    - Each state's hand plan and ILS incumbent are both replayed (2 extra evaluations per state, about 0.5 CPU-h, on the same states as the K1b replay).
    - The rule is the **MEAN** over the 200 states of (RTT(hand) − RTT(ILS)) ÷ RTT(ILS) **< 1 %**. It is not a median: local-search regret is zero-inflated, with frontier medians at 0.00 % against means of 1.4–4 %.
    - The share within 1 % and the p90 are reported, with S-regret beside them. Where S and the replay disagree, **the replay rules**.
  - **Snapshots:** if the hand cells don't capture the 200 states, those cells are rerun with the snapshot hook at the seeded times. Identity (m) requires the rerun's summary to be byte-equal to the original cell's.
  - If **both** (a) and (b) hold, **STOP (HAND-SUFFICES)**: no time to save and no quality to gain. This is a live measurement of the hand arms, so rule 6 is met.
- **K2, before the live gate (offline):** the GNN plan's S-regret on held-out batches > 10 vs the hand fallback's, at equal charged time. A NO-GO is recorded and orders the work. It does not cancel the registered live gate (rule 6).

**7. Cost** (SCIENTIST's revision; refined from the smoke's wall per cell before launch).

| phase | CPU-h | notes |
|---|---|---|
| hand arms | 150–350 | k = 1, g0–g1, 3 exact-class arms. Per-cell wall is uncertain 3×: ib0_confirm averaged about 1,150 s per exact cell [V], but calibration shows `cd_exactS` decision wall of 3,818 s on a heavy cell [V]. 144 exact cells ≈ 45–160 CPU-h, plus the cd arms. k = 0.25 doubles it; K0 adds about 96 cells. Big batches inflate expansion by an unknown factor, the main uncertainty. |
| labels (S) | 150–300 | about 0.06 ms per S evaluation, about 60 s per big state at the cap, × 5–10k stratified big states, plus 0.5 h for the replay diagnostic. 0.5–1k only if K1b forces replay labels. |
| train + gate | about 150 | 4 learned arms × 48 cells at k = 1, plus GPU training of 2 arms and the MLP. |

**8. Identities before launch.**
- (a) **Cut path:** with every group ≤ 10, the uncut path is bit-equal to today's on grounded cells, and the mint at `--max-size 10` reproduces `workloads.sha256`.
- (b) **k = 0:** bit-equal to no charge. decision_cost has this 20/20 IDENT [V]; it is re-run on the new code.
- (c) **Alpha×prune:** the relaxation is a registered change. The relaxed arm at alpha = 1 is bit-equal to the current prune arm, and with the flag off it is bit-equal on the 24 identity cells.
- (d) **No silent cut:** a synthetic 3,000-task batch through every arm's serving path returns a plan of length n. Every gate cell asserts tasks served == tasks minted, and any internal size limit fails loud. This covers the list below.
- (e) **Determinism with a big group:** two runs at one seed give byte-identical summaries.
- (f) **Instrument rule:** the smoke shows `sched_plan_evals` and busy time non-zero on batches > 10, and the w(n) table, before the gate.
- (g) **Load check:** the uncut workload's tasks per second are within 2 % of each rung (a gate, not an identity).
- (h) **Label determinism:** two ILS runs are byte-identical.
- (i) **Library rebuild** (S6; condition by SCIENTIST): rebuilt at `--max-size 10`, each library reproduces the committed library's **md5**; if not, the extract has drifted and the node stops. The uncut rebuild reproduces each group's first 10 offsets and `size_raw`.
- (j) **Snapshot path** (the expert): the snapshot run with the cap removed reproduces the capped run bit-equal on a trace with no group > 10 (a window where that holds, or a filtered trace).
- (k) **Same batch formation** (SCIENTIST, B): one hand arm and one learned arm produce identical batch membership on a fixed arrival trace.
- (l) **Sampled `placements.jsonl` readers:** a sampled file without the header is rejected, one with it is reported as a sample, and no reader treats an S value as an RTT (tests in `tests/`).
- (m) **Snapshot hook** (SCIENTIST): a hand cell rerun with the snapshot hook at K1's seeded times gives a summary byte-equal to the original cell's.

**9. Origin.** Single origin per group, spread over the 160 clients: the accel seat's existing model (big_groups and prune_g8 designs) [V].
- A fan-out originates at one caller, so this is the realistic model.
- Scattered origins are what manufactured the earlier learned wins: with one origin no learned arm beats CD (`client_local_v1`, `small_batch_so_v1`; grounded_workload_v1 line 7) [V]. A result under scattered origins would be uninterpretable.
- Scattered origins are disclosed as NOT RUN.

## Serving and corpus rulings (SCIENTIST A–H and the expert, 2026-10-13)

**A. No bridge** on the plain mint (both). The batch is the trace group as the mint draws it, so identity (a) holds.
- The mint's rule, each member drawing `min(2, n − 1)` distinct partners (undirected union), gave **zero disconnected draws** in SCIENTIST's simulation. Seed 0; 2,000 draws per n ≤ 300 and 200 above; n = 3, 5, 10, 11, 15, 20, 30, 50, 100, 300, 1,000 and 2,876 [V]. The k-out literature is cited as background only.
- The inputs report gives the number of exchange-graph components per group (1, 2, 3+) by size bucket × shard.
- A bridged arm would be a separate registered condition, not the default.

**B. Late members:** today's rule stays exactly as written (`gnn/scheduler.py:578–601`). A batch waits up to `batch_timeout` (1.0 s), serves what has arrived, and late members go to later batches with their placed partners fixed. Changing it would be a breaking change to the seat. The sibling span is p50 2 ms, p99 430 ms over the **cut** libraries [V]. Over the **uncut** libraries it is p99 300 / 362 / 6,587 / 366 ms (shards 0 / 50 / 100 / 130), so shard 100 (g2) spreads its groups past the 1 s window.
- **Registered readouts per cell:** `peer_group_incomplete_batches`, and the share of tasks split off past the timeout by size bucket, per arm and rung, beside every table.
- **The split share is a mediator, not a confound** (SCIENTIST). Batch collection runs when the scheduler server is free, so at k > 0 a slower arm collects later and splits fewer groups: the split share is part of the arm's decision-time effect. It is reported per arm and condition, and is never removed. Above 1 %, the read says so.
- **Same batch formation for every arm (checked by S6):** the hand batch arms inherit `GNNScheduler._collect_task_batch` without overriding it. `PeerGreedyNetworkBatchScheduler(_PeerGreedyCore, GNNScheduler)` is at `peer_greedy_network/scheduler.py:896`, and `PeerGreedyNetworkCDScheduler` subclasses it at `:1577`. `:909` refuses to run without `GNN_DECODE_MODE=masked_topo` and `GNN_BATCH_BY_PEER_GROUP=1`. The tabular `XGBoostBatchScheduler(GNNScheduler)` doesn't override it either.
- A test that two arms form identical batch membership on a fixed arrival trace is still to be written (identity (k)).

**C. Batch ceiling:** an explicit whole-group mode (a cfg `batch_size: "group"` or a flag), not 10 → 3,000.
- Item 7's ignore-and-print becomes a **raise** (AGENTS rule 4).
- The 16 ceiling stays for the old modes and is lifted only under the whole-group flag.
- **Pre-flip scan (S6, 2026-10-13):** tracked repo JSON has one `batch_size` (10). The 3,661 cell cfgs under datalab `simulation_data` and `/share/nikola.lukic` (depth ≤ 6) hold 3,609 at 10 and 48 at 16, none above 16. So no prefix-mode gate relied on the ignore. Not checked: whether any argmax-mode arm (range [2, 4]) was served on a cfg of 10. If one was, it got the ignore and is a gate-tools correction.

**D. Declared slate:** the learned arms serve no-split (`serving_no_split`). A counter asserts `SUB_BATCH` never fires in any cell of this node, or the cell fails. The hand arms' 100,000 cap (item 10) is unchanged.

**E. Corpus:** the snapshot runs serve uncut, with whole-group batching, from the rebuilt libraries. Identity (j) applies.
- The rollout arm is `cd_exactS`, not ib0: ib0's runaway trajectories visit states the gate's healthy arms never do.
- Snapshots run at k = 1, so states carry the decision-time backlog.
- Stratification (≥ 30 % of states from batches > 10) is applied when sampling snapshots, never by duplicating states.
- Corpus topologies are disjoint from the gate's.

**F. Labels and `placements.jsonl`:** labels are ILS on S (ruling 3), never `executecosimulation`'s sweep, so the 250,000-combo skip is not in the label path. A state whose label fails fails loud and is never dropped.

**Exemption SIGNED by the expert (2026-10-13).** The terms are owned by [docs/notes/placements_jsonl_required.md](../notes/placements_jsonl_required.md), "Sampled files above the plan cap":
- the exemption is per dataset, by plan-space size: groups within the 100,000-plan cap keep the full sweep;
- a mandatory header line (`exhaustive`, `plan_space_size`, `eval_budget`, `evals_done`, `search`, `seed`, the S pin, the cap) on every file this lineage writes;
- every ILS-evaluated plan, with its S, its simulated label and an iteration index, with nothing deduplicated away;
- the metadata and validation readers report `exhaustive=false` and fail loud rather than read a sample as a sweep, with a test.

These reader changes and the test are pre-label code (identity (l)).
- **"Simulated label" clarified (SIGNED by the expert, 2026-10-13, as SCIENTIST recommended):**
  - S is on every row;
  - the replayed RTT is only on each new incumbent along the ILS trajectory, and on the top-m = 8 by S if K1b fires; it is null elsewhere;
  - the header states which rows carry it and their count;
  - never replay every evaluated plan.
  Replaying every plan would cost 0.5–3 h per 50-task group.

**G. Pre-gate checks** (from S6's unchecked items):
- **Feature ranges at large n.** Every per-batch or count feature (prefix block, partner counts, batch-size terms) has its max on batches > 10 compared with the checkpoint's training max (the queue_range_v1 / legacy_v0 lesson). Any feature beyond it is either trained on or reported as out of range.
- **Memory and forward time at n = 2,876.** Peak RSS and forward wall are measured in identity (d). An OOM fails loud. Forward time is charged through w(n), not capped. The gate's per-cell `--mem 8G` (`CELL_MEM`) is the binding limit to check. If a 2,876-task decode exceeds it, `CELL_MEM` is raised for **every** arm equally (not a physics change), disclosed, and never for the learned arms alone.
- **Merge order:** decision_cost reads first (kill check 2, done), then the w(n) key lands in `decision_cost_w.json`, then this node's code.

**H. Capacity structures** (SCIENTIST's question; S6 read at 18161421):
- **Not caps:**
  - KPA scale-up is bounded by `ceil(max_scale_up_rate × max(1, ready))` with rate 1,000 (`src/placement/scaleout.py:35`, `:172–174`), effectively unbounded.
  - Platform queues are unbounded `Store(env)` (`src/placement/infrastructure.py:778`).
  - The per-node `compute_slots` `Resource` (`infrastructure.py:2019`) is contention physics, not a size cap.
  - The orchestrator mutex (`orchestrator.py:118`) is capacity 1 by design.
- **Replica ceiling, CONFIRMED (S6, 2026-10-13):** `replica_count_series.peak` is exactly 120 in every accel summary read for slate_fix_v2. It is the max of a 121-point sampled total (`fresh_topo_burst_v1_gate.py:396–422`).
  - Topology 19420's infrastructure, built by `prepare_infrastructure_for_real_simulation` (`src/executesimulation.py:110`) from `inputs_12/inputs/wf1_heavy/cfg/cc40s19420.json` at f4f0453f, has **24 servers carrying exactly 120 platforms**: 8 servers × 4, 8 × 10 and 8 × 1. By type: rpiCpu 32, xavierCpu 64, xavierGpu 8, xavierDla 8, pynqFpga 8.
  - So the peak is every server platform holding a replica, i.e. full scale-out: a physical ceiling, not a code cap. A 2,876-task burst cannot get more replicas than this.
  - Checked on one topology. The other topologies use the same node-type percentages; identity (d)'s smoke reports the count per topology.
- **Keep-warm:** `GNN_REPLICA_KEEPWARM_MAX_REPLICAS` (default 4, `gnn/scheduler.py:212`) applies only when keep-warm is on, which it is not in this seat (counters 0).

**Live gate** (rule 6): F1–F3 above are the live gate and its closing step.

**Open before the work they gate:**
- `rp/decision-cost` is merged or pinned, and the w(n) key lands before any cell (G).
- The alpha×prune relaxation and the whole-group batch mode (C) are implemented with their identities before any cell that uses them.

Closed 2026-10-13:
- kill check 2 (passed);
- the collapse rule (fallback until warm_surge's lands);
- breaking change #4 (both lines reported);
- bridging (none);
- late members (today's rule);
- the code-path list (SCIENTIST spot-checked items 3, 6, 7 and 8).

## Post-data amendments (2026-10-13, flagged POST-DATA)

Written after k = 0 cells started returning. Signed by SCIENTIST and the expert, 2026-10-13. Trigger: S5's whole-group smoke at c671281c could not finish heavy `cd_exactS` or heavy `cd_expand` inside the 21,600 s watchdog (projected about 33,000 s of wall time per cell), and the k = 0 array had 29 watchdog kills, all in exact, expansion or prune arms; `cd` is healthy 12/12 at heavy.

- **(b) K0's load question is answered by `cd`** (healthy 12/12 at heavy). `cd_exactS` heavy health is WALL-INFEASIBLE, which is unknown, not healthy. Admissibility is read on `cd` when the bar arm has no finished cells. The moderate-only fallback is not applied by itself.
- **(c) WALL-INFEASIBLE is a third health count**, beside collapse and the tail flag.
  - Arms wall-infeasible at k = 0 are not launched at k = 1 at that rung (a known-failing launch).
  - Their k = 1 charge is derived from all k = 0 cells, including the killed cells' partial counters, as frozen w(n) × plan evaluations per bucket. It is a lower bound, and each cell reports the wall-versus-simulated-length inequality (utilisation > 1) as the basis for "out of budget".
  - The rule applies identically to the learned arms: one heavy uncut smoke each at k = 0 before any k = 1 cell.
- **The k = 1 hand bar** is `cd`, `cd_ib0` and `cd_exactS_ib0_prune` if the prune is wall-feasible (its smoke, below, decides). Otherwise it is `cd` and `cd_ib0`, and any learned win is labelled **WIN-VS-CD-ONLY**.
- **Finding:** `cd_expand`'s "in-budget bar at k = 1" status came from the capped seat ([decision_cost_s0_v1](decision_cost_s0_v1.md)). Under whole-group serving it is not in budget at heavy.
- **Result:** "hand expansion run to termination is wall-infeasible on uncut heavy groups (projected ~33,000 s per cell), measured before any learned arm ran".
- **Contingency arm `cd_expand_cap`**, defined now so it cannot be chosen from outcomes; built only if the prune's smoke is wall-infeasible at heavy. SCIENTIST's fixed form:
  - cap per batch = floor(B × n_tasks ÷ w(n_bucket)) evaluations, with B = 10 ms per task and w(n) the frozen `cd_expand` per-bucket ms per plan;
  - the expansion stops at the cap and serves its incumbent;
  - identity: on batches whose uncapped expansion finishes under the cap, the plan is bit-equal to `cd_expand`'s;
  - one heavy smoke for wall feasibility before k = 1.
  - Signed SCIENTIST + expert 2026-10-13, post-data, not built.
- **w(n):** per served-batch-size bucket for `cd_exactS`, `cd_exactS_ib0` and `cd_expand` (see the record). Scalar w stands for `cd` and `cd_ib0`. `cd_exactS` heavy and `cd_expand` heavy are **imputed**.

## Code paths that assume groups of at most 10 tasks (to change; not edited)

Read at reference-physics 18161421. "Cut" is where tasks are dropped; "cap" is where a larger group would be split or refused.

| # | Where | What it does | What uncut serving needs |
|---|---|---|---|
| 1 | `scripts_cosim/grounded_workload_v1_extract.py:40`, `:74–76` | **The cut.** `--max-size 10`; a fan-out larger than that keeps its first 10 siblings by time. `size_raw` is recorded (`:71`, `:77`), but the dropped offsets are not. | Rebuild the four libraries from the raw MSCallGraph shards (kept at `/share/nikola.lukic/surrogate_gap/alibaba_fanout/`, per hard-stops) with no cut. The committed libraries cannot be un-cut. |
| 2 | `scripts_cosim/fanout_uncut.py:12` | Asserts `rule.max_size == 10` when reading a library. | Relax it, or read both libraries side by side. |
| 3 | `scripts_cosim/grounded_workload_v1_mint.py:13–14`, `:140–145`, `:146–160` | Each member draws `min(2, size − 1)` random partners in its group (degree about 4). A bridging pair joining components is added **only** under `--merge-k` or `--group-size` (`:146`). So in the plain mint, a library group's peer closure is very likely, but not certainly, the whole group. | Decide whether uncut groups get the bridge (one closure = one group) or not. A 2,876-task group is then one peer component, and the peer-group batch is the whole group. The partner draw at `:142` is O(n²) per group, which is fine at 2,876. |
| 4 | `scripts_cosim/grounded_workload_v1_mint.py:54–77` (`merge_groups`, `regroup_to_size`) | `--merge-k` and `--group-size` (big_groups) regroup the already cut library tasks. | Not a substitute for uncut groups: they act on truncated offsets. |
| 5 | Cell configs: `inputs/<rung>/cfg/cc40s<topo>.json` `scheduler.batch_size: 10`, `batch_timeout: 1.0` (accel seat, read on `inputs_12`) | **Serving cap.** A peer-group batch stops collecting at `batch_size` tasks (`src/policy/gnn/scheduler.py:588`). | Raise per arm, or replace with an uncapped peer-group batch. |
| 6 | `src/policy/gnn/scheduler.py:117–118`, `:122–126`, `:137–143` | `MAX_BATCH_SIZE_FOR_PREFIX_GNN = 16`. `GNN_BATCH_SIZE` above it raises. | Lift the ceiling for the uncut arms (it is called "a guard … not a model limit" at `:112–116`). |
| 7 | `src/policy/gnn/orchestrator.py:72–84` | A cell `batch_size` outside `[batch_min, batch_max]` is ignored, with a print, and the env value is kept. | Same change as 6; otherwise an uncut config silently serves 16. |
| 8 | `src/policy/gnn/scheduler.py:578–601` (`_collect_task_batch_inner`, peer-group mode) | Collects the first task's peer group up to `batch_size`; members arriving later than `GNN_BATCH_TIMEOUT` are left to the next batch (`peer_group_incomplete_batches`). | For a 2,876-task group spread over the request's span, decide whether to wait for the whole group or decode incrementally. A design point for SCIENTIST. |
| 9 | `src/placement/declared_slate.py:30–32`, `:37`, `:68`, `:115–119`, `:154` | `TOP_K = 5`, `MAX_PLANS = 100,000`, `SUB_BATCH = 4`. A batch whose pruned plan space exceeds 100,000 is split into 4-task sub-batches that do not see each other's partners; `serving_no_split()` skips the split. | At the group sizes here, every batch over about 7 tasks sub-batches (5^7 = 78,125; 5^8 = 390,625). The no-split path, or a new one, is the only joint decode. |
| 10 | `src/policy/peer_greedy_network/scheduler.py:133`, `:170`, `:356`, `:1347–1350`, `:1488–1496` | The bar's exact search enumerates up to `HEROSIM_PG_CD_EXACT_MAX_PLANS` (default 100,000), with the expansion / prune fallback above it. | This is where the time-bounded fallback is defined. The cap and bound must be signed (bar item). |
| 11 | `src/policy/gnn/scheduler.py:174`, `:1038`, `:1066` | `SELF_SEARCH_EXACT_MAX = 625`: the GNN self-search enumerates up to 625 slate plans, with coordinate ascent above. | The learned arm's search size, and whether it is time-bounded like the bar. |
| 12 | `scripts_cosim/make_warm_corpus.py:866–867` | Each corpus dataset's batch is the live snapshot's batch, so the corpus inherits the serving cap of item 5. | Snapshots from an uncut serving run, or a group-level snapshot. |
| 13 | `scripts_cosim/generate_gnn_datasets_fast.py:87`; `src/executecosimulation.py:2218` | Exhaustive co-sim labels are skipped above `MAX_PLACEMENT_COMBINATIONS_SKIP` (default 250,000). | Exhaustive labels are infeasible for big groups, which is why the labels are ILS or time-bounded exact. |
| 14 | `rp/decision-cost` (ca927a5a, not on reference-physics) | The clock that charges decision time (`decision_cost_s0_v1`, Physics). | Merge it, or pin on it, before any uncut cell; registration waits on its kill check 2 anyway. |
| 15 | `scripts_cosim/datalab/r1a_gate.sbatch` (`--mem "${CELL_MEM:-8G}"`) | Each cell is a systemd scope with MemoryMax 8G. | A 2,876-task decode must fit, or CELL_MEM is raised for this node (G). |

Not checked: the feature contract for very large batches (the `partial_state_v3` block is size-free by design), the memory of a 2,876-task decode, and the per-batch time of the GNN forward at that size. These are smoke-test items.

## Record (newest first)

- 2026-10-13 — **`cdxapply` is wall-feasible where `cd_expand` is not** (S4, smoke 864642; 19461 g0, k = 0; slate_fix_v1 checkpoints trained on cut groups; descriptive, not a gate number; OUT `/share/nikola.lukic/uncut_groups/smoke_learned`).
  - `cdxapply` (GNN seed + α-expansion) finished at both rungs: 3,425 s (moderate) and 4,654 s (heavy) of wall time, `pgb_max_batch` 199, SUB_BATCH violation 0.
  - Hand `cd_expand` runs the same expansion from the hand seed and is wall-infeasible at heavy.
  - Decision wall 67.8 / 92.4 ms per task mean (median 4.2 / 4.6 ms). The largest single call took 186 / 215 s, on a 199-task group.
  - This is a **cost-side difference of the learned seed**, and the paper quotes it as that.
  - Mechanism unmeasured: S4's hypothesis is that a GNN seed nearer a local optimum leaves the expansion fewer moves. This is a hypothesis, not a claim.
  - Not run at k = 1 under the paper pivot (the user's decision, 2026-10-13) unless the user asks.
- 2026-10-13 — **Exact-search cap hits at k = 0** (S7, `scripts_cosim/uncut_caphit.py`, `/share/nikola.lukic/uncut_groups/caphit_k0.json`, read from the finished cells of `gate_k0`).
  - Counters per cell summary: `pg_exact_fallbacks` (batches over the 100,000-plan cap; the scheduler's own cap counter), `pg_exact_batches`, `pg_exact_plans`. Served batches by size are `pgb_whole_*` + `pgb_split_*`. Share = hits ÷ (exact + hits).

    | arm | rung | cells (failed) | exact batches | cap hits | share | plans / exact batch | served ≤ 10 / 11–50 / 51–100 / > 100 |
    |---|---|---|---|---|---|---|---|
    | `cd_exactS_ib0` | heavy | 36 (0) | 312,590 | 33,026 | 9.6 % | 2,431 | 1,154,495 / 496,171 / 134,553 / 35,692 |
    | `cd_exactS_ib0` | moderate | 47 (1) | 415,122 | 38,154 | 8.4 % | 1,869 | 1,526,808 / 655,960 / 178,247 / 48,110 |
    | `cd_exactS` | heavy | 1 (2) | 9,027 | 839 | 8.5 % | 2,486 | 33,031 / 12,321 / 4,407 / 693 |
    | `cd_exactS` | moderate | 34 (4) | 299,317 | 28,366 | 8.7 % | 2,064 | 1,103,110 / 470,346 / 130,471 / 35,640 |
    | `cd_exactS_ib0_prune` | heavy | 18 (2) | 156,631 | 16,749 | 9.7 % | 2,402 | 576,634 / 248,822 / 65,606 / 18,122 |
    | `cd_exactS_ib0_prune` | moderate | 38 (0) | 344,540 | 28,647 | 7.7 % | 1,820 | 1,268,943 / 493,351 / 154,511 / 30,881 |

  - **The summary has no per-bucket cap counter**, so cap hits cannot be split by served-size bucket. None is added now, since K1 does not need it (the expert).
  - `cd_exactS` heavy rests on one finished cell. The table is re-run when S5's `gate_k0_completion` cells land.
  - The expert's earlier expectation that "almost every big batch hits the cap" is contradicted: cap hits are 8–10 % of exact batches, and at most about 5 % of 11+ batches.
- 2026-10-13 — **Sealed capture lost 3 k = 1 states (launch defect, fixed at 1929fd3c); recaptured as a code fix.**
  - Cause:
    - The gate sets `LIVE_AUDIT_MAX_SNAPSHOTS` to the number of sealed ids in the cell.
    - A postponed task's retry can lead a second batch with a trigger already captured, and the hook captured it again.
    - That capture took the last slot, so a later sealed trigger could not be captured whether or not it fired.
    - Evidence from summary counts only: each of the 3 cells with a missing id has written = wanted (19463 g1 moderate `cd_ib0_k100` 3/3, missing 45393; 19463 g3 moderate `cd_k100` 2/2, missing 46539; 19465 g1 moderate `cd_ib0_k100` 3/3, missing 41168).
  - Fix (1929fd3c): a sealed trigger is captured once; the retry batch it leads takes no slot. K1(b) already takes the first capture per id, so nothing it reads changes.
  - The same pin writes the served plan of each sealed batch to `<snapshots>.chosen.jsonl` (`live_audit.load_chosen`), so K1(b) can check re-decided = served.
  - S5's line, corrected: the 3 missing ids are "missing: cap exhausted by a retry capture; whether the id triggered is unknown (launch defect, fixed 1929fd3c)", not "never a trigger at k = 1".
  - **Ruled (the expert):** recapture, not impute. The 3 cells run at 1929fd3c (jobs 864752–864754, OUT `gate_k1_recap_1929fd3c`). Acceptance: SIM-IDENTICAL summaries, and byte-equal snapshots for the ids already captured. Imputation at the bucket max (signed rule (1)) is the fallback only if that identity fails. Imputation was written for a trigger that never fired; here the state can be measured.
  - **Read: PASS** (ruled by the expert). The summaries are SIM-IDENTICAL to the gate_k1 originals on 451–452 shared keys, except `sealed_snapshots.missing` (now []). All 3 ids are recovered (3/3, 2/2, 3/3 written).
    - Old capture order [28090, 28090, 45008], [11754, 11754], [24890, 37730, 37730]; new order [28090, 45008, 45393], [11754, 46539], [24890, 37730, 41168].
    - The first capture of each already-captured id is byte-equal, except 45008 in 19463 g1 `cd_ib0_k100`, which differs only in `snapshot_id` (old 2, new 1). That is the file's write counter, moved because the dropped duplicate came first.
    - The recapture ran twice at 1929fd3c with the same result. **S5's OUT `gate_k1_recapture` (jobs 864750/864751) is canonical for K1(b).** S6's OUT `gate_k1_recap_1929fd3c` (jobs 864752–864754) is an independent confirmation: SIM-IDENTICAL summaries and byte-equal snapshots against S5's, with the same 3 ids recovered. It is kept on disk and is not read into K1.
- 2026-10-13 — **K1(b) tool and label engine built** (S7, `rp/uncut-ils` dee6e4eb, `scripts_cosim/uncut_ils.py`, tests `tests/test_uncut_ils.py`).
  - ILS runs on the scheduler's own `_pg_plan_cost` (ib0). Replay runs only on the hand plan and on each new incumbent. The K1(b) reader works per bucket.
  - **Smoke 2 (validation):** 20 frontier states (n ≤ 6). ILS reaches the exhaustive S-optimum 20/20, replay equals the stored table RTT to within 7e-15, and ILS is at the label optimum on 17/20.
  - **Smoke 1:** the identity-(m) snapshots (n = 7 / 9 / 10; 16490 g0 heavy, `cd_exactS`; config `slate_fix/smoke_inputs/wf1_heavy/cfg/cc40s16490.json`, md5 9af73d93…, read from 864574's launch). One state (idm_1, n = 9) has a replay regret of 421 %. That is a **one-state observation only**.
  - Re-deciding the 3 identity-(m) states twice gives identical hand plans, ILS bests, evaluation logs and replayed RTTs.
  - **Big-n cost is measured only in the sealed run** (ruled by the expert): no gate_k1 snapshot is touched before the read. Nothing is measured yet at n > 10.
  - Outputs: `/share/nikola.lukic/uncut_ils/{smoke1,smoke2}`.
  - Sealed snapshots carried `chosen: null`, because the capture runs before the decision. The expert ruled for a served-plan sidecar (`<snapshots>.chosen.jsonl`), to land before the remaining k = 1 snapshot cells. The `cd` and `cd_ib0` k = 1 cells already running keep the re-decision check, disclosed.
- 2026-10-13 — **Prune smoke: `cd_exactS_ib0_prune` is wall-feasible at both rungs; it is the k = 1 hand bar's third arm** (job 864605 at c671281c; 19461 g0, k = 0, `inputs_group`, LIMIT 21,600 s; `/share/nikola.lukic/uncut_groups/wnprune_c671281c`).
  - Both cells finished all 50,000 tasks: 4,551 s (heavy) and 3,900 s (moderate) of wall time, `pgb_max_batch` 199.
  - Ms per plan by served bucket (≤ 10 / 11–50 / 51+): heavy 0.0549 / 0.1520 / 0.3003 (ratio 1 / 2.77 / 5.48); moderate 0.0573 / 0.1536 / 0.3453 (1 / 2.68 / 6.03). Both exceed 1.5×, so w(n) is frozen by the same rule: heavy 0.09034 / 0.25040 / 0.49467, moderate 0.09134 / 0.24498 / 0.55072.
  - Descriptive: this cell's ≤ 10 rate is about 0.6× the calibrated scalar (0.0549 against 0.09034 heavy). The frozen values keep ≤ 10 at the scalar.
- 2026-10-13 — **Prior, F1, F2 and K1 note revised before any k = 1 data** (SCIENTIST's verbatim text, accepted by the expert; in the head and § Statistic). Replaced prior text, verbatim: "2026-10-13, SCIENTIST, before any D data: P(WIN) ≈ 0.05 (was 0.10); P(MP-attributed win) ≈ 0.02. Revised after decision_cost's attribution read: in the capped seat at k = 1, cdxapply ties cd_expand within 1 pp (SEARCH-WIN-NOT-GNN), so the learned seed's edge there is zero, and cd_expand is within budget at k_anchor ≈ 12–14. D adds only bigger batches (to ~230 tasks). A learned edge needs the hand expansion's cost or quality to degrade faster with n than the GNN's; nothing measured yet says it does. K1 measures it." (registered at P(WIN) ≈ 0.10). The nosplit-split item from the learned-arm smoke is resolved (below).
- 2026-10-13 — **Learned-arm wall smoke** (S4, job 864633; 19461 g0, k = 0, pin f36f293f; slate_fix_v1 checkpoints seed 1, trained on cut groups; descriptive, not a gate number; OUT `/share/nikola.lukic/uncut_groups/smoke_learned`).
  - `ra_gnn_eng_selfsearch` is **WALL-INFEASIBLE at both rungs**: killed at about 600 s of wall time, projected 32,584 s (moderate) and 72,480 s (heavy). It leaves F1 by the symmetry rule (post-data amendment (c)).
  - `ra_gnn_eng_nosplit` **finished** in 200 s (moderate) and 193 s (heavy) of wall time, with `pgb_max_batch` 199.
    - Decision wall: 3.23 / 3.06 ms per task, which is 0.13 % of mean latency (2.55 / 2.11 s).
    - The forward pass on a 199-task group fits inside the 8 GB cell cap. The job-level sacct MaxRSS is 3.10 GB for 4 cells, not per cell.
  - **SUB_BATCH assertion, defined:** the violation counter is `slate_declared_sub_batched − slate_declared_unsplit` and must be 0. It is 0 here (1,225 and 1,641, each equal to its pair).
  - **Resolved (S4's split analysis; ruled by the expert):** the nosplit cells split more peer groups than `cd` in the heavy > 100 bucket (45 against 0), but this is a trajectory effect, not a defect.
    - nosplit's whole-group task share is inside the hand spread: heavy 99.0 % against `cd` 99.4, `cd_ib0` 99.2 and `cd_exactS_ib0` 99.3 %; moderate 97.7 % against 97.2–98.0 %.
    - Hand arms also split groups > 100 at moderate (`cd` 115, `cd_exactS` 72, `cd_exactS_ib0` 62 tasks).
    - `peer_group_incomplete_batches` is identical across all arms (11 heavy, 15–16 moderate).
    - At k = 0 the decode wall never enters the simulated clock (`scheduler.py:1401`: the charge is the only advance of simulated time). So a split reflects which bucket an incomplete batch lands in, not the decision wall.
    - Ruled: split groups are counted at their original size, as registered. At k = 1 the serialisation effect becomes live for every arm and is read there, not instrumented now.
  - F1's learned slate is now `nosplit`, plus `cdxapply` only if its feasibility smoke (running next) passes at a rung.
- 2026-10-13 — **Pause rule and NOT-RUN cells.**
  - The harness's 5 % pause line (`scripts_cosim/datalab/r1a_gate.sbatch:4`): "production pauses for a rung when more than 5 % of its decided non-Knative cells hang". In code (`scripts_cosim/progress_watchdog.py:21–22, :96–97`), it acts once at least 40 cells are decided, and a timeout, watchdog kill or unreadable result counts as hung.
  - In the k = 0 array, the pause left 110 cells **NOT-RUN**, a fourth category that is neither failed nor healthy.
  - S5 launches completion arrays for the never-run cells of feasible arms (`cd_exactS_ib0` heavy 12; prune moderate 48; prune heavy 28 if the prune smoke is wall-feasible). **Completion is not a rerun:** these cells never started, and no failed or killed cell is run again.
  - The pause line pools hung cells across every arm in a rung's OUT directory (`progress_watchdog.py`: hung cells over decided cells, per rung). So S5 launches the never-run `cd_exactS_ib0` heavy and prune heavy k = 0 cells into a **sibling OUT**. There, the exact and expansion arms' watchdog kills do not count against them. Both OUT directories are read together as one k = 0 grid.
- 2026-10-13 — **w(n) frozen per bucket; the clock charges it** (rp/uncut-wn, from 3a59a70c).
  - Source: S5's whole-group smoke at c671281c, one cell per arm × rung, `/share/nikola.lukic/uncut_groups/wn_smoke.json` (sha256 6b2c22d4…), script `scripts_cosim/uncut_groups_wn_table.py` (rp/uncut-hand 92dc22d3).
  - Ms per plan, ratio to the arm's own ≤ 10 bucket (11–50 / 51+): `cd_exactS` moderate 4.2 / 8.6×; `cd_exactS_ib0` heavy 3.6 / 6.0×, moderate 3.4 / 6.9×; `cd_expand` moderate 12.2 / 22.5×; `cd` ≤ 1.17×; `cd_ib0` ≤ 1.11×.
  - **Scalar w stands for `cd` and `cd_ib0`.**
  - `scripts_cosim/decision_cost_w.json` key `w_ms_per_plan_by_bucket`: value = calibrated scalar w × measured ratio to ≤ 10, so ≤ 10 equals the scalar. `cd_exactS` heavy and `cd_expand` heavy are **imputed** (their heavy smoke cells were watchdog-killed): the arm's heavy scalar × its own moderate ratio.
  - The clock: `HEROSIM_DECISION_COST_MS_BY_BUCKET` (JSON, `src/placement/decision_cost.py`) charges the rate of the decided batch's bucket; the gate serves it for arms with a bucket entry and keeps `HEROSIM_DECISION_COST_MS` for every other arm. Tests in `tests/test_decision_cost_by_bucket.py`.
- 2026-10-13 — **Registered.** SCIENTIST closed the five [M] items (four verified, α-expansion corrected) and ruled on B–H follow-ups. The expert signed the `placements.jsonl` exemption (owned by `docs/notes/placements_jsonl_required.md`) and the bar additions (`cd_expand`, a time-bounded prune at `cdxapply`'s limits). Kill check 2 PASSED. The replica ceiling of 120 is confirmed as full scale-out (24 servers, 120 platforms). Labels and training are on hold.
- 2026-10-13 — **Per-size decision counters for w(n)** (S5's request, approved by the expert as measurement only).
  - c671281c on eb22be53 adds `decisionTiming.by_bucket.{le10,11_50,gt50}.{calls,total_s,plan_evals}`, by served batch size.
  - Identity job 864590 (19461 g0 heavy `cd`, `inputs_group`, k = 0) is SIM-IDENTICAL to S5's f36f293f cell (`smoke/`, job 864578) on all 441 shared keys. The only new keys are the 9 bucket counters.
  - The buckets sum to the totals: calls 9,553, decision wall 9.81339 s, plan evaluations 1,353,318 = `sched_plan_evals`.
  - This cell's ms per plan is 0.0068 / 0.0078 / 0.0079 (≤ 10 / 11–50 / 51+). That is descriptive; the registered w(n) table comes from S5's five-arm smoke.
- 2026-10-13 — **Identities (1), (3), (m) PASSED; hand cells handed to S5** (split confirmed by the expert).
  - All runs are the smoke cells 16490 g0 heavy; summaries are compared on every shared key, with timing and provenance excluded.
  - (1) whole-group mode + raise, 4523dc56 (jobs 864543 / 864544): `cd_exactS` s0 and `ra_gnn_eng_nosplit` s1 are SIM-IDENTICAL to `v2_identity_f4f0453f` (427 / 369 keys).
  - (3) dc-steady merge, 4e028f44 (864547 / 864548): SIM-IDENTICAL; the only new keys are provenance.
  - (m) snapshot hook:
    - The first run at f20bfaab (864559) wrote 3/3 snapshots, but 8 `peerExchangeByAccessClass` counters moved. Everything else, including the trajectory, was unchanged.
    - Diagnostics 864566 (fidelity, no capture) and 864567 (capture, no fidelity) were each identical.
    - Cause: the capture's input estimate booked priced exchanges. Fixed at 6d1764bd, as a launch fault ruled by the expert; the gate-tools entry is dated 2026-10-13.
    - Rerun 864574 at 6d1764bd, hook on, is SIM-IDENTICAL to the hook-off cell 864558 (445 keys), with 3/3 written.
  - (4) prune at alpha = 1 **PASSED**: job 864550 at 4e028f44 (`ra_gnn_eng_cdxprb` 18700 g0 moderate, 3 h 10 min) is SIM-IDENTICAL to prune_g8's recorded cell (8301f267) on 397 keys. The only new keys are provenance and `env.HEROSIM_WARMTH_PHYSICS`. So S4's alpha×prune relaxation leaves the alpha = 1 prune path unchanged, and the `cd_exactS_ib0_prune` arrays are unblocked.
  - S5 owns the hand cells, the K0/K1 reads and the readers (`rp/uncut-hand`, `scripts_cosim/uncut_groups_hand_read.py`). Read outputs come into this node's attachments through S6.
- 2026-10-13 — **K1(b) sample SEALED before any k = 1 hand cell** (none has run).
  - `scripts_cosim/uncut_groups_k1_draw.py` at 983361cc, seed **20261013**, on `gate_inputs_e2738e37/inputs_group`, topologies 19461–19472. 200 states: 150 in 11–50, 40 in 51–100, 10 in 101+.
  - [k1_sealed_states.json](uncut_groups_s0_v1/k1_sealed_states.json), sha256 `755ace0276b4a41c2e12dcb54b51e334cf1e9c058257c0b3b5427e21039a3e7f`.
  - [k1_snapshot_plan.json](uncut_groups_s0_v1/k1_snapshot_plan.json) (the gate's `R1A_SNAPSHOT_PLAN`), sha256 `f8b555550071cc5e212c02c86484d72c2b7088e8fc2cd4a76d4c42e58d6b28fc`.
  - A state is (topology, window, rung, trigger task id = the group's first member). Every hand cell of that topology × window × rung captures it, so the arm picked after the cells run has the same states.
- 2026-10-13 — **Code for the hand cells** (rp/uncut-groups), each with tests:
  - whole-group mode with the raise (4523dc56);
  - `--start-group` (2ed779d8);
  - decision_cost merge (4e028f44);
  - peer-group batch counters `pgb_whole_*` / `pgb_split_*` / `pgb_max_batch` (K0's uncut share);
  - whole-group inputs `inputs_group` with MD5SUMS in both trees (8787907f; job 864557: 24 cfgs rewritten, 10 untouched files byte-identical; MD5SUMS md5 4954f384 / 0ad1ea61);
  - sealed snapshot hook `R1A_SNAPSHOT_PLAN` / `LIVE_AUDIT_TRIGGER_IDS` (f20bfaab).
  - Identities (1), (3), (4) and (m) are running.
- 2026-10-13 — **Robust-bar rules, before data** (SCIENTIST's, signed by the expert as rules): WIN-ROBUST / WIN-OVER-BRITTLE-BAR clause; collapse and tail-flag health at K0 for every hand arm; K1 decided only by a non-collapsing arm. Prompted by warm_surge 0c.
- 2026-10-13 — **SCIENTIST's prior revision (P(WIN) 0.10 → 0.05; MP-attributed 0.02) and K1 amendments:** (a) on both whole-run and after-360 s; (b) is a mean, not a median, on the replayed RTT, with bucket-stratified sampling; identity (m) added for the snapshot hook. All written before any hand cell.
- 2026-10-13 — **Gate inputs built** (job 864541 at pin e2738e37, 35 s; `big_groups_inputs.sbatch`, X1 = `x1_uncut`, M 11.6139 / 27.622665025860204, REPLICA_RULE `fastest_compatible`, BASECFG `/share/nikola.lukic/accel_nosplit/inputs/cfg_src/cc40s16345.json` as ib0_confirm_v1 used). Output in `/share/nikola.lukic/uncut_groups/gate_inputs_e2738e37/inputs`.
  - The first submission, 864539, failed the job's own check: the default BASECFG carries no replica rule. Its partial ROOT was removed.
  - Check: 12 topologies × moderate/heavy at 160 / 24 / p 0.6, topology seed = id, rule `fastest_compatible`. Single origin spread over the 160 clients.
  - **Rate (identity (g)) PASSED:** moderate 5.3470 / 5.3469 / 5.3470 / 5.3417 tasks/s and heavy 12.7173 / 12.7172 / 12.7173 / 12.7048 (g0–g3), all within 0.10 % of 5.3470 / 12.7173. Groups 9,443 / 10,058 / 9,385 / 10,428; max 199 / 163 / 226 / 135.
  - The cell cfgs still carry `scheduler.batch_size` 10. The whole-group mode (C) is code still to be written, and no cell runs before it and identities (a)–(l).
- 2026-10-13 — **Library rebuild and ×1 mint.**
  - Identity (i) PASSED: job 864527 at e2738e37. `--max-size 10` rebuilds reproduce each committed library's JSON md5. All 160,000 uncut groups match the committed first 10 offsets, `size_raw` and t0. Uncut md5s: 0 69185e63, 50 47e70a26, 100 13d5069e, 130 1f2452f0, in `/share/nikola.lukic/uncut_groups/libs/`.
  - Identity (a), mint half, PASSED: job 864535. The K = 1 mint from the committed libraries reproduces `/share/nikola.lukic/big_groups/x1_G0` byte for byte in all 4 windows.
  - Uncut ×1 windows are in `/share/nikola.lukic/uncut_groups/x1_uncut`. Task rate 0.4604/s (g3 0.4599, −0.11 %), so the registered multipliers stand. Groups fall 21–25 %. Every group is one exchange component.
  - Corpus ruled (a); giant-group label cap 256; K0 pooled read; K1 thresholds written.
  - `cdxapply` contingency resolved as a tie (S4). The twin-seed + expansion arm is added, built only on an F1 edge.
  - Gate topologies 19461–19472 (token scan: 194xx–199xx in use on any branch are 19400–19432, 19440–19452, 19490, 19500, 19516 and up; no `/share` name or selection JSON holds 19461–19472). Gate inputs job 864539.
- 2026-10-13 — **Label term and `cdxapply` contingency ruled** by the expert: S on every row, a replayed RTT on incumbents (and the top-8 if K1b fires); `cdxapply` leaves F1 only if `cd_expand` beats it by ≥ 2 % (Holm) at a rung.
- 2026-10-13 — Drafted by S6 on the expert's request. Design items 1–9 are SCIENTIST's rulings of the same day; the code-path list is S6's (read only, nothing edited). Not registered; no index row.
