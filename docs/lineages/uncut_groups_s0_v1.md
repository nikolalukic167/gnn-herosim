# uncut_groups_s0_v1 — does a learned scorer beat the time-bounded hand bar when Alibaba peer groups are served uncut and decision time is charged?

**Status:** `DRAFT — NOT REGISTERED` (2026-10-13). Drafted by S6 on the expert's request; the design is SCIENTIST's (pick D), and SCIENTIST keeps the name.
- decision_cost kill check 2 PASSED (S4 interim read, job 864514): `cd_exactS` loses +9.53 % (moderate) and +197.4 % (heavy) to its own decision time at k = 1, 0/12 at both rungs.
- Registration waits on SCIENTIST closing the [M] items, and on the expert's ruling on the `placements.jsonl` exemption (Serving and corpus, F).
- Authorised by the expert once registered: the topology and library mint, identities (a)–(j), and the hand-arm cells.
- **On hold:** ILS labels and all training wait for S4's decision_cost attribution read (`cd_expand` vs `cdxapply`).

**Why.** Every grounded and accel gate so far served Alibaba peer groups cut to at most 10 tasks. The library builder keeps only the first `--max-size 10` siblings of a fan-out (`scripts_cosim/grounded_workload_v1_extract.py:74–76`), and the served batch is capped at the cell's `batch_size` 10 (code ceiling 16).

[grounded_workload_v1](grounded_workload_v1.md)'s 2026-10-12 cut audit:
- **37.47 % of member tasks** sit in groups larger than 10, though only **7.5 % of groups** exceed 10;
- 23.97 % of tasks sit in groups larger than 20;
- the largest group has **2,876** tasks, and the mean uncut size is 5.18.

The audit is over the mint's draw, not the whole trace.

So the joint problem studied so far is the easy part of the workload. On the big groups, exact search cannot enumerate. With decision time charged to the clock ([decision_cost_s0_v1](decision_cost_s0_v1.md)), the hand bar pays for its fallback search.

**Not a revival.** The hard-stops entry "Two-level Alibaba fan-out" (2026-10-12) closes a hidden second level revealed when the parent completes. This node serves the **first-level** fan-out uncut, fully known at arrival. **It never reveals later levels.**

**Prior and hypothesis (SCIENTIST): P(WIN) ≈ 0.10.**
- Coupling stays sparse. Each member draws `min(2, size − 1)` partners (`grounded_workload_v1_mint.py:13–14`, `:140`), so a 2,876-task group is a graph of mean degree about 4, not a clique.
- Hand local search is strong on sparse pairwise objectives: α-expansion reaches the exact optimum on 67 % of tables (`cd_expand`; from memory, not re-read).
- **Registered hypothesis:** any learned edge comes from decision **time**, not plan quality.

**Parents to read:**
- [grounded_workload_v1](grounded_workload_v1.md): the cut audit and the origin caveat.
- [decision_cost_s0_v1](decision_cost_s0_v1.md): the clock, k, w, utilisation, Amendments 2–3.
- [big_groups_s0_v1](big_groups_s0_v1.md) and [prune_g8_s0_v1](prune_g8_s0_v1.md): the G=8 fallbacks.
- [ib0_confirm_v1](ib0_confirm_v1.md): the bar.
- [warm_surge_s0_v1](warm_surge_s0_v1.md): ib0 runs away under a warm heavy surge.
- [slate_bar_v1](slate_bar_v1.md): the slate handicap.
- [slate_fix_v2](slate_fix_v2.md): after-360 s and whole-run side by side.

## Design (SCIENTIST's rulings, 2026-10-13; not signed)

Marks: [V] SCIENTIST read it on reference-physics 18161421; [M] from memory, not re-read. Every [M] item is checked before registration.

**1. The bar.**
- **(a) Enumeration limit:** the existing 100,000-plan cap (`HEROSIM_PG_CD_EXACT_MAX_PLANS`, `peer_greedy_network/scheduler.py:170`, `:356`) [V]. ib0 was confirmed there and decision_cost's w was calibrated there. On top-5 slates it means exact search up to n = 7.
- **(b) Fallback: both expansion and prune.** The bar is decision_cost Amendment 2's hand frontier: the best hand arm per rung, chosen at the arm level by median.
  - Candidates: `cd`, `cd_ib0`, `cd_exactS` (expansion), `cd_exactS_ib0` (expansion), `cd_exactS_ib0_prune`, plus the expert's 2026-10-13 additions:
    - **`cd_expand`**, a hand seed with the same α-expansion as `cdxapply`;
    - **a time-bounded `exactS` prune at the same expansion limits as `cdxapply`**.
    These two exist so that a learned win cannot be the search's. decision_cost's interim read has `cdxapply` beating the hand frontier at k = 1, whole-run (−3.57 / −2.96 %), on a grid with no hand-seeded expansion.
  - The prune arm is in because cost-order top-k prune beat expansion −1.89 / −2.11 % vs `cd_exactS` at G=8 (`prune_g8_s0_v1`) [V]. It needs the alpha×prune refusal lifted as a registered change (the expert's ruling), with its own identity against the current prune arm at alpha = 1 (8c), before any cell of this node uses it. Amendment 2 lists it as NOT RUN. **No WIN is claimable without it.**
  - `cd` and `cd_exactS` are bar candidates, not reference lines: under a warm heavy surge ib0 ran away on 3/12 topologies (×2) and 5/11 (×3) while they held (warm_surge 0b) [M]. Uncut giant groups are surges.
- **(c) Time bound: no wall-clock cap.** Decision time is charged per decision_cost Amendment 3: `t_batch = k × w[arm][rung] × plans_evaluated`, one FIFO scheduler server, mutex held [V].
  - Expansion and prune run to their own termination and pay for it on the clock. That is the bound.
  - k = 1 (as measured on the cluster CPU) is primary, and is where kill check 2 reads. k = 0.25 is a secondary family; k = 0 is a reference.
  - **w is recalibrated for size.** On the smoke, report ms per plan by batch size (≤ 10, 11–50, 51+). If any bucket differs from ≤ 10 by more than 1.5×, fit `w(n) = a + b·n` per arm and rung and freeze it in `decision_cost_w.json` under a new key, before any gate cell.
- **(d) Collapse:** warm_surge's collapse rule has not landed; S5 writes it after the 0c diagnostic (the expert, 2026-10-13). Until it lands, this extension of gate-tools 2026-10-08 is registered, and the node switches to warm_surge's rule verbatim when it does:
  - a cell collapses if its end minus its last arrival is more than 300 s (the request timeout), **or** its mean latency is more than 5× the median of the hand arms' means on that topology × window × rung;
  - the collapse count per arm is a registered line beside every median.

**2. Arms.** One clock for all: k × the arm's own calibrated w × its own plans. Each arm serves the slate it was trained on (named per arm; `slate_bar_v1`).

| arm | role |
|---|---|
| GNN self-search, no-split engineered, s1 | primary |
| its MP-OFF twin, same search | attribution (required) |
| `cdxapply` (GNN seed + CD + expansion) | primary; decision_cost's predicted winner |
| MLP same-input self-search | secondary pointwise control |

- Plain no-split is dropped: it is E-FASTER in slate_fix_v2 (+3.6 / +7.1 %) [V].
- Search machinery is unchanged: self-search is exact up to `SELF_SEARCH_EXACT_MAX` = 625 slate plans, coordinate ascent above (`gnn/scheduler.py:174`, `:1066`) [V]. There is no hand expansion inside the pure GNN arms.

**3. Labels.** ILS on ib0's surrogate S, not exact replay: exact search is impossible above n = 7, and a replay build costs 5.6–7.7 s per state plus 0.05 s per evaluation (S7's frontier figure as recorded in decision_cost; not re-read).
- **Budget:** α-expansion plus random-restart perturbation, capped at 200 × n × (mean candidates per task) evaluations per group.
- **Convergence (registered):** on a 200-state sample, doubling the budget moves the label's S by less than 0.1 %. Otherwise the budget is raised until it does.
- **Ties:** lexicographic on (node id, platform id) per task in task order, never set order. Seeded; two label runs are byte-identical (8h).
- **S-fidelity (kill K1b's input):** exact replay on 200 sampled groups > 10 (about 0.5 CPU-h) gives argmin-S's true regret. The reference is ib0's S regret at n ≤ 6, 4.1 % heavy / 1.4 % moderate (as relayed). If the mean regret on groups > 10 exceeds 2× that (8 % heavy), S labels are wrong where this node lives: switch to replay-evaluated top-m labels (m = 8).

**4. Seat and corpus.**
- **Seat:** accel 160 × 24, WF1, p 0.6, `fastest_compatible`, KPA, pipelined, `HEROSIM_REPLICA_RELEASE=1`, 1 s window. That is where the bar, ib0 and decision_cost's w live.
- **Not the 6-server grounded seat:** a 2,876-task burst there is a pure load lever.
- **Gate topologies:** 12 fresh ids, disjoint from 19420–19431 (warm_surge), 19405–19416 (decision_cost, ib0), 19206–19217, 18700–18711 and earlier pools. Picked by gnn_overcap's token-scan protocol; the prep job fails loud on existing files.
- **Corpus topologies:** fresh, and disjoint from the gate set.
- **Windows:** the gate runs g0–g3, one per shard. Shard 100 is the heaviest, 43.5 % of tasks in groups > 10 [V].
- **No overlap:** the corpus draws from the same libraries but from requests **after** the gate's first 40,000. Overlap is reported as 0 by trace id.
- **Cut groups are not excluded;** the uncut groups are the point.
  - State sampling is stratified so batches > 10 are at least 30 % of training states. Unstratified, the loss would ignore them: they are 7.5 % of groups but 37.5 % of tasks.
  - MIN_TIME stays 360 s, matching the after-360 s line.
- **Load:** the uncut workload is re-stretched so tasks per second match each rung within 2 % (heavy 12.7173, moderate 5.3470 tasks/s). Groups per second then fall about 23 % (mean group size about 4.0 → 5.18; disclosed). Without the re-stretch, uncut means more load, and the contrast becomes a load lever.

**5. Statistic.** Paired per topology (median over windows of the paired %), then the median over topologies, the sign count, an exact Wilcoxon and Holm (`R.holm`).
- **Whole-run and after-360 s are both reported, and a WIN must clear both.** Steady state as the sole primary is breaking change #4, still PROPOSED and the user's decision (confirmed by the expert, 2026-10-13). Requiring both also guards against a win that lives in the opening, the way slate_fix_v2's self-search deficit did.
- **F1, primary, k = 1** (Holm over 4): self-search vs the bar and `cdxapply` vs the bar, × 2 rungs.
  - **WIN:** median ≤ −2 % and Holm p < .05 at both rungs for the same arm, **and** that arm's collapse count ≤ the bar arm's, **and** decision_cost's anchor rule: the crossover k is ≤ the anchor k of the hand arm it beats, else NO-GO.
- **F2, attribution, k = 1** (Holm over 4): self-search vs its MP-OFF twin and vs MLP same-input, × 2 rungs.
- **F3, secondary, k = 0.25** (Holm over 4): F1's tests, Holm separately. No WIN is claimed from F3 alone.
- **Labels:**
  - WIN / TIES / E-FASTER (the bar Holm-faster at both rungs);
  - MP: MP-WIN / DIRECTION-ONLY (right sign, not Holm) / NOT-SEPARATED;
  - **UNCUT-ATTRIBUTED** (registered, descriptive): at least 50 % of the pp gap sits in batches > 10, by gnn_overcap's by-batch bucket method on g0 cells. A win carried by batches ≤ 10 is labelled **WIN-NOT-UNCUT**, not a result for this node.

**6. Kill rules.**
- **K0, admissibility, before any learned work** (hand arms at k = 0, g0, 12 topologies, both rungs): `cd` and `cd_exactS` must be healthy (no collapse, effective queue share ≤ 0.80) on ≥ 10/12 at heavy. Otherwise the giant groups are a load lever: fall back to the moderate rung only, disclosed. If moderate also fails, **STOP (NO-SEAT)**.
- **K1, before training** (live hand arms at k = 1, plus offline labels):
  - (a) Decision-time headroom: the best hand arm's scheduler wait plus decision share is under 3 % of its mean latency (kill check 2, restated on uncut groups).
  - (b) Plan headroom: on batches > 10, the hand fallback's plan is within 1 % of the ILS label on S (median over batches).
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

**9. Origin.** Single origin per group, spread over the 160 clients: the accel seat's existing model (big_groups and prune_g8 designs) [V].
- A fan-out originates at one caller, so this is the realistic model.
- Scattered origins are what manufactured the earlier learned wins: with one origin no learned arm beats CD (`client_local_v1`, `small_batch_so_v1`; grounded_workload_v1 line 7) [V]. A result under scattered origins would be uninterpretable.
- Scattered origins are disclosed as NOT RUN.

## Serving and corpus rulings (SCIENTIST A–H and the expert, 2026-10-13)

**A. No bridge** on the plain mint (both). The batch is the trace group as the mint draws it, so identity (a) holds.
- A random 2-out graph is connected with high probability at large n [M].
- The inputs report gives the number of exchange-graph components per group (1, 2, 3+) by size bucket × shard.
- A bridged arm would be a separate registered condition, not the default.

**B. Late members:** today's rule stays exactly as written (`gnn/scheduler.py:578–601`). A batch waits up to `batch_timeout` (1.0 s), serves what has arrived, and late members go to later batches with their placed partners fixed. Changing it would be a breaking change to the seat. The sibling span is p50 2 ms, p99 430 ms [V].
- **Registered readouts per cell:** `peer_group_incomplete_batches`, and the share of tasks split off past the timeout by size bucket, per arm and rung, beside every table.
- Above 1 %, the read says so and SCIENTIST rules whether it is a confound.
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
- This conflicts with AGENTS' mandatory full-sweep `placements/placements.jsonl`.
- SCIENTIST's proposal: write every ILS-evaluated plan with its S and the label, under a header with `exhaustive=false` and the eval budget, and state in the node that it is a sample, not a sweep.
- **Open ruling, the expert's.**

**G. Pre-gate checks** (from S6's unchecked items):
- **Feature ranges at large n.** Every per-batch or count feature (prefix block, partner counts, batch-size terms) has its max on batches > 10 compared with the checkpoint's training max (the queue_range_v1 / legacy_v0 lesson). Any feature beyond it is either trained on or reported as out of range.
- **Memory and forward time at n = 2,876.** Peak RSS and forward wall are measured in identity (d). An OOM fails loud. Forward time is charged through w(n), not capped. The gate's per-cell `--mem 8G` (`CELL_MEM`) is the binding limit to check.
- **Merge order:** decision_cost reads first (kill check 2, done), then the w(n) key lands in `decision_cost_w.json`, then this node's code.

**H. Capacity structures** (SCIENTIST's question; S6 read at 18161421):
- **Not caps:**
  - KPA scale-up is bounded by `ceil(max_scale_up_rate × max(1, ready))` with rate 1,000 (`src/placement/scaleout.py:35`, `:172–174`), effectively unbounded.
  - Platform queues are unbounded `Store(env)` (`src/placement/infrastructure.py:778`).
  - The per-node `compute_slots` `Resource` (`infrastructure.py:2019`) is contention physics, not a size cap.
  - The orchestrator mutex (`orchestrator.py:118`) is capacity 1 by design.
- **Unverified ceiling:** `replica_count_series.peak` is exactly 120 in every accel summary read for slate_fix_v2. It is the max of a 121-point sampled total (`fresh_topo_burst_v1_gate.py:396–422`). It is likely full scale-out onto every eligible (node, platform) pair, a physical limit and not a code cap, but that is not confirmed from the cfg. A 2,876-task burst cannot get more replicas than that, so identity (d)'s smoke reports it.
- **Keep-warm:** `GNN_REPLICA_KEEPWARM_MAX_REPLICAS` (default 4, `gnn/scheduler.py:212`) applies only when keep-warm is on, which it is not in this seat (counters 0).

**Live gate** (rule 6): F1–F3 above are the live gate and its closing step.

**Open points before registration:**
- SCIENTIST verifies the [M] items: the warm_surge 0b counts, α-expansion's 67 %, the ib0 S-regret and replay-cost figures as relayed, and random 2-out connectivity.
- The expert rules on the `placements.jsonl` exemption (F).
- `rp/decision-cost` is merged or pinned, and the w(n) key lands before any cell (G).
- The alpha×prune relaxation and the whole-group batch mode (C) are implemented with their identities before any cell that uses them.

Closed 2026-10-13:
- kill check 2 (passed);
- the collapse rule (fallback until warm_surge's lands);
- breaking change #4 (both lines reported);
- bridging (none);
- late members (today's rule);
- the code-path list (SCIENTIST spot-checked items 3, 6, 7 and 8).

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

- 2026-10-13 — Drafted by S6 on the expert's request. Design items 1–9 are SCIENTIST's rulings of the same day; the code-path list is S6's (read only, nothing edited). Not registered; no index row.
