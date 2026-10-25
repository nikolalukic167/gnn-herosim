# mp_confirm_v1 — on fresh topologies, does the static link-feature GNN beat its separately trained MP-OFF twin under option D?

**Status:** `REGISTERED` (2026-10-25). Drafted by S7 at the expert's request with the user's go, for SCIENTIST's check; nothing below is signed until SCIENTIST files it. Written before any mint, any twin checkpoint at seeds 4–6, or any cell.
- **Question.** Under option D (fair share plus TCP), does the epoch-40 static link-feature GNN `uncut-linkstatic-gnn-eng-e40` beat its MP-OFF twin `uncut-linkstatic-twin-e40`, trained separately on the same corpus, labels and contract, on 24 topologies on which neither has been served? Seeds 1–6 of each arm, moderate and heavy.
- **Why.** `live_link_features_v1` closed TIE on this contrast with a moderate direction (−6.90 %, 11/12 without runaways, Holm .068, 3/3 seeds negative) and no heavy direction. `llf_cd_confirm_v1` then confirmed the gnn against the bar and the parameter-matched wide MLP with no twin in the design, and says why it wins is not settled. This node is the missing contrast, on topologies whose gnn values have not been read, at six seeds.
- **Outcome.** Not run.
- **What this is not.** Not a Knative claim (reactive is not run: under D it link-saturates, `llf_cd_confirm_v1` head). Not a rerun of the parent's sealed 12 (spent). **A win here is not by itself "message passing helps":** the gnn has 507,524 parameters and the twin 257,156 (seed-1 `.pt`), so a gnn win mixes message passing with capacity. The parameter-matched wide MLP is the capacity control and is not run here; its result is in `llf_cd_confirm_v1` (gnn −9.3 % moderate, −33.2 % heavy). This node answers only "gnn vs its twin".

## Motivation (parent primary, whole run, checked by SCIENTIST in `live_link_features_v1`)

- gnn vs twin, moderate: −6.90 %, faster on 10/12 with all cells (Holm .30) and on 11/12 without runaways (raw p .034, Holm .068). Seeds alone: −7.7, −4.6 to −5.0, −8.2 %.
- Heavy: +13.0 % with all cells (4/12), −6.7 % without runaways (8/12, p .85). **No direction survives at heavy**; heavy is runaway-dominated (191 runaway flags across all arms in the parent primary).
- Raw p, chosen among the parent's secondary rows. That is why the fresh topologies are needed. Winner's curse applies to the moderate magnitude: quote the direction, not −6.9 %, as the expectation.
- The best-val wave's MP-DIRECTION (moderate −6.58 %) is not evidence here (parent caveat: a seed-1 twin served at its first evaluation).

## Arms

| Arm | Seeds | Checkpoint / pin | Role |
|---|---|---|---|
| static gnn `ra_gnn_eng_combfl_linkstatic_e40_nosplit_k100` | 1–6 | `uncut-linkstatic-gnn-eng-e40-seed{1..6}-final.pt`, the same files and md5s as `llf_cd_confirm_v1` (lists `xs/llf_conf_models_final.md5` md5 `a126f46b5f30dd88ba43795d91c20ab3` and `xs/llf_conf_models_final_s456.md5` sha256 `ae613e9843b84aadf0f6f1deab79278662e21fa5abcfb5f5457ab2f5a1ab9b47`); pin `02668ecf` | treatment |
| MP-OFF twin `ra_twin_eng_combfl_linkstatic_e40_nosplit_k100` | 1–6 | `uncut-linkstatic-twin-e40-seed{1..6}-final.pt`; s1–3 md5 `056c4843…`, `b44e374b…`, `ca369012…` (`xs/llf_conf_models_final.md5`); s4–6 trained by S6 and filed (below); pin `02668ecf` | control |
| bar `cd_ib0_inflightfull_k100` | 0 | pin `0ec7cdcb` | admission and context row |

- **Twin seeds 4–6** are trained with the parent's A6 recipe (40 epochs, min-epochs 40, no early stop; same corpus, labels and contract, wandb logging), after `tests/test_trainer_determinism.py` passes. **The md5s of all twelve served `.pt` and their `.contract.json` are filed before the first cell of either arm**; SCIENTIST re-hashes them. If a twin seed fails to train, the expert rules before any cell, and the seed set is not changed after the first cell.
- **Decision charge.** Every cell carries its arm's measured per-(arm, rung) charge, as in the parent. The gnn's is the `llf_cd_confirm_v1` table. The twin's is the parent's `d_conf_final` value, identical on seeds 1–3 and applied to seeds 4–6 (ms, le10 / 11_50 / gt50):

  | arm | moderate | heavy |
  |---|---|---|
  | gnn | 5.01465 / 2.254181 / 2.191336 | 6.151998 / 3.103575 / 3.678685 |
  | twin | 4.323723 / 1.878764 / 2.129705 | 4.826589 / 2.525467 / 2.983758 |
  | bar | 0.0116 | 0.01098 |

  The gnn is charged more than the twin in every bucket; that is part of every gnn-vs-twin number. The free-decisions row is reported beside it, descriptive.
- **Physics, slate and recipe as `llf_cd_confirm_v1`:** option D (TCP on, fair share serial, default cellular uplink), cap200 inputs, `declared_pruning_v1_uncapped`, no split, k100, 24 servers, pinned worktrees.

## Topologies

- **24 new ids at 24 servers**, the third set of S5's joint rescan (after `knative_stable_v1`'s 24 and `scale_d_v1`'s). **Not the 30327–30350 of `llf_cd_confirm_v1`:** their gnn values have been read. Disjoint also from every list in `live_link_features_v1`, `llf_cd_confirm_v1`, `knative_stable_v1`, `scale_d_v1` and `uncut_groups_s0_v1`, from fresh6, the sealed 30285–30297 and the accel ids. S5 mints them and files the ids, their sha256 and the input md5s before any cell.
- **Mint** as the parent's confirmation recipe (xs x1 mint at `224dd02e`, then the cap200 clip).
- **Admission is by the bar alone, on all four of its cells** (g0 and g1 × moderate and heavy):
  1. `queue_share` ≤ 0.80 on each;
  2. completion on each: the summary exists with `num_tasks` = 50,000 and a finite `endTime`, there is no `.failed.json`, and no watchdog stall or wall kill. Unknown is a fail.
  - A topology that fails is replaced by the next free id in the rescan, and every replacement is reported. A killed bar cell is rerun once with a longer wall first.
  - **Reactive is not run.** The bar cells are both the admission cells and the context row. The learned cells are minted and run only for admitted topologies, so admission cannot depend on gnn or twin values.

## Cells

Option D, windows **g0 + g1**, both rungs. gnn and twin at seeds 1–6, bar at s0: 2 × 6 × 24 × 2 × 2 = **1,152 learned cells** plus 24 × 2 × 2 = **96 bar cells**. Cells live in their own roots, separate from every other study's.

## Statistic, primary family and labels

- **Statistic.** `live_link_features_v1/confirm_rulings.json` (md5 `b609acef33b10e7369a0f5aa58b42477`), unchanged: whole-run task mean; a learned cell is the mean over its six seeds; per topology the median over g0 and g1 of the paired % (gnn / twin − 1); then the median over topologies, the count faster, and the exact two-sided Wilcoxon over topologies (zeros dropped). A cell is a runaway if it exceeds 10× its own arm's median in the rung (seeds pooled; any seed flags the cell); a pair with a flagged cell is dropped; a topology with no window left is dropped.
- **Primary family: 2 tests, Holm over 2.** (1) gnn vs twin, moderate; (2) gnn vs twin, heavy.
- **Reading: without runaways, with the all-cells sign guard.** Without runaways, a gnn runaway hides inside a dropped pair, so a test cannot be CONFIRMED while its all-cells median is ≥ 0.
- **Labels, per test.**
  - **CONFIRMED:** median < 0, Holm p < .05, and the all-cells median < 0.
  - **DIRECTION:** median < 0, not CONFIRMED.
  - **NOT-FASTER:** median ≥ 0, or no topology left.
  - **UNDERPOWERED:** as `scale_d_v1` (v3, item 4, "Primary family" there): CONFIRMED stands at any n; a test with n_eff < 8 topologies after the drops that is not CONFIRMED is labelled UNDERPOWERED in place of DIRECTION or NOT-FASTER. The Holm family stays m = 2.
- **Failed learned cells:** the `scale_d_v1` v3 rule ("Failed learned cells" there), applied to the gnn and twin kinds. A cell that fails twice drops its (topology, window, rung) pair from both readings and is counted per arm; any gnn failure at a rung blocks CONFIRMED for the test at that rung; any other missing cell fails loud.
- **Power.** The exact Wilcoxon with n topologies all one sign has p = 2/2ⁿ: n = 24 → 1.2e-7, n = 12 → .0005, n = 8 → .0078. With Holm over 2 the smaller p must be < .025 and the larger < .05. In `llf_cd_confirm_v1` the gnn alone lost 16 (topology, window) cells to runaways at heavy, and any-seed-flags-cell drops more at six seeds, so **heavy may well end UNDERPOWERED**; that is a registered outcome, not a failure of the design.
- **Descriptive beside each test:** the all-cells row; seeds 1–3, seeds 4–6 and each seed alone; the after-360 s row; free-decisions row; gnn vs bar and twin vs bar (context); the runaway list per arm; per-task and request p50/p99 and the link-load rows, with the fresh-24 reviewer-defence script's definitions; the measured/charged decision-time ratios.
- **Not run, not quoted:** reactive and random.

## What each outcome means (fixed now)

| Reading | Statement allowed |
|---|---|
| moderate CONFIRMED | the gnn beats its twin at moderate on fresh topologies (the parent's direction holds); not a message-passing attribution on its own (capacity: 507,524 vs 257,156 parameters) |
| moderate CONFIRMED and `llf_cd_confirm_v1`'s gnn vs the parameter-matched wide MLP (506k parameters; CONFIRMED there at both rungs) | together the strongest MP-adjacent claim this design allows: the gnn beats its twin and beats a same-size pointwise model. Still not an attribution to message passing alone: the two contrasts come from different topology sets and the wide MLP is not run here |
| moderate DIRECTION | the sign repeats; no claim |
| moderate NOT-FASTER | the parent's moderate direction did not repeat; the TIE of `live_link_features_v1` stands |
| heavy CONFIRMED | stated as a direction plus Holm, not a magnitude (runaway-dominated) |
| heavy UNDERPOWERED / DIRECTION / NOT-FASTER | reported as labelled; no heavy claim |

## Reader

- `llf_confirm_read.py`, mode `cd`, extended with a `twin` arm and the 2-test family: the tests above, Holm over 2, the labels including UNDERPOWERED and the gnn-failure block, the failed-learned-cell rule with an allow-list of learned kinds, and the seed sets 1–6. It checks the id-list sha256; every cell's pin per group (static `02668ecf`, bar `0ec7cdcb`); each learned cell's `env.GNN_MODEL_PATH` basename against arm and seed (`uncut-linkstatic-gnn-eng-e40-seed{N}.pt` / `uncut-linkstatic-twin-e40-seed{N}.pt`); each learned cell's charge string against the table above; the bar's per-rung charge. It fails loud on any mismatch.
- **The default mode and both `llf_cd_confirm_v1` invocations must reproduce their filed outputs except `reader_sha256`** (tests run on the parent's sealed cells or a fixture, and on the fresh-24 `read_primary_cd.json` inputs).
- SCIENTIST freezes the blob, with the rulings file, before the first cell is read. **The read runs once**, after all 1,248 cells are in or the failed-cell rule has been applied.

## Prior (SCIENTIST to file at signing)

Not set by the drafter. Inputs for it: the moderate direction is 3/3 seeds negative with raw p .034 at n = 12; the heavy parent rows show no direction; the gnn-vs-twin gap sits inside the best-val wave's seed-to-seed spread, which includes a positive seed.

## Live gate

This node is itself the live gate: every number is a served simulation cell. Wall and queue estimates come from `llf_cd_confirm_v1` (2,016 cells) and are refreshed by S5/S6 before launch.

## Record

- 2026-10-25 — **Drafted** (S7, at the expert's request; the user wants the MP test). Unsigned. Sources read: `llf_cd_confirm_v1` (design, arms, charges, reader, labels), `live_link_features_v1` (parent outcome), `scale_d_v1` v3 (failed-learned-cell rule, UNDERPOWERED precedence, completion definition; merged at `47466dc1`), `knative_stable_v1` (admission, id rule), the twin charge strings and md5s from `d_conf_final` and `xs/llf_conf_models_final.md5` on datalab.
