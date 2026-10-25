# mp_confirm_v1 — on fresh topologies, does the static link-feature GNN beat its separately trained MP-OFF twin under option D?

**Status:** `CLOSED` (2026-10-25) — **CONFIRMED (2/2)**. Registered 2026-10-25, before any mint or cell, any twin checkpoint at seeds 4–6, or any read. Drafted by S7 at the expert's request with the user's go; signed by SCIENTIST with four amendments the expert accepted. The primary (seeds 1–6 pooled) was read once with the frozen reader (`llf_confirm_read.py` blob `cbf7fb08`, `confirm_rulings.json` md5 `b609acef`).
- **Question.** Under option D (fair share plus TCP), does the epoch-40 static link-feature GNN `uncut-linkstatic-gnn-eng-e40` beat its MP-OFF twin `uncut-linkstatic-twin-e40`, trained separately on the same corpus, labels and contract, on 24 topologies on which neither has been served? Seeds 1–6 of each arm, moderate and heavy.
- **Why.** `live_link_features_v1` closed TIE on this contrast with a moderate direction (−6.90 %, 11/12 without runaways, Holm .068, 3/3 seeds negative) and no heavy direction. `llf_cd_confirm_v1` then confirmed the gnn against the bar and the parameter-matched wide MLP with no twin in the design, and says why it wins is not settled. This node is the missing contrast, on topologies whose gnn values have not been read, at six seeds.
- **Outcome.** **The gnn beats its separately trained MP-OFF twin at both rungs on 24 fresh topologies (seeds 1–6 pooled, Holm over 2).** Moderate −4.25 % without runaways (21/24, Holm .0036; all cells −3.41 %, 19/24), 6/6 seeds negative. Heavy CONFIRMED as a direction plus Holm, not a magnitude (12/16, Holm .0155; all cells 14/24, p .58).
  - **Capacity control (descriptive, same 24 ids).** The 257k twin beats the 506k parameter-matched wide MLP: −6.7 % moderate (19/23), −40.6 % heavy (13/16). The gnn beats the wide MLP −10.2 % (22/23) and −44.0 % (15/15). Parameter count alone does not explain the gnn's lead. This is still not an attribution to message passing alone: no 507k MP-OFF twin was trained, and gnn vs wide is outside the Holm family.
  - **Heavy is runaway-dominated.** 77 gnn and 73 twin heavy cells are runaways, so 8 of 24 topologies drop and n = 16. The seed blocks disagree on strength: s1–3 −5.5 % (14/18, p .018), s4–6 −11.2 % (13/20, p .23). Seed 4 alone is +3.5 %.
  - **Moderate seeds 4–6 are weaker** than seeds 1–3: −3.2 % (18/24) against −6.2 % (21/24).
  - It confirms, on topologies whose gnn values had not been read, the moderate direction that `live_link_features_v1` left at TIE on the same contrast.
- **What this is not.** Not a Knative claim (reactive is not run: under D it link-saturates, `llf_cd_confirm_v1` head). Not a rerun of the parent's sealed 12 (spent). **A win here is not by itself "message passing helps":** the gnn has 507,524 parameters and the twin 257,156 (seed-1 `.pt`), so a gnn win mixes message passing with capacity. The parameter-matched wide MLP (506k) is the capacity control and is **served here as a descriptive arm on the same 24 ids** (amended), so gnn vs wide and gnn vs twin come from one topology set; `llf_cd_confirm_v1` has gnn vs wide CONFIRMED on another set (−9.3 % moderate, −33.2 % heavy). The primary answers only "gnn vs its twin".

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
| wide MLP `ra_mlp_wide_combfl_linkstatic_e40_nosplit_k100` | 1–6 | `uncut-linkstatic-mlp-wide-e40-seed{1..6}-final.pt`, already trained, the same files and md5s as `llf_cd_confirm_v1` (s1–3 `23498572…`, `f4f01f8e…`, `cc654108…`; s4–6 `04357770…`, `7db997a7…`, `c670b6ae…`); pin `02668ecf` | descriptive capacity control, same 24 ids (amended) |
| bar `cd_ib0_inflightfull_k100` | 0 | pin `0ec7cdcb` | admission and context row |

- **Twin seeds 4–6** are trained by S6 with the parent's A6 recipe (40 epochs, min-epochs 40, no early stop; same corpus, labels and contract, wandb logging), after `tests/test_trainer_determinism.py` passes. **Md5 rule (amended 2026-10-25):** the md5s of the seeds 1–3 `.pt` and `.contract.json` (both arms) are filed before the first cell; the md5s of the seeds 4–6 files (both arms) are filed before the first seeds 4–6 cell. SCIENTIST re-hashes each set. If a twin seed fails to train, the expert rules before any seeds 4–6 cell is served; the primary seed set 1–6 is not changed after the first cell, and no interim value may be used in that ruling.
  - **Twin s4–6 validity (signed).** When the md5s are filed, SCIENTIST checks each new twin `.contract.json`: `disable_message_passing` is true, and every key equals the s1–3 twin contract's except the seed and run identity. `load_state_dict` cannot see MP-OFF, so only the contract carries it.
- **Decision charge.** Every cell carries its arm's measured per-(arm, rung) charge, as in the parent. The gnn's is the `llf_cd_confirm_v1` table. The twin's is the parent's `d_conf_final` value, identical on seeds 1–3 and applied to seeds 4–6 (ms, le10 / 11_50 / gt50):

  | arm | moderate | heavy |
  |---|---|---|
  | gnn | 5.01465 / 2.254181 / 2.191336 | 6.151998 / 3.103575 / 3.678685 |
  | twin | 4.323723 / 1.878764 / 2.129705 | 4.826589 / 2.525467 / 2.983758 |
  | wide MLP | 4.527073 / 3.585821 / 7.915672 | 5.008803 / 5.134591 / 10.944349 |
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

Option D, windows **g0 + g1**, both rungs. gnn and twin at seeds 1–6, bar at s0: 2 × 6 × 24 × 2 × 2 = **1,152 primary learned cells**, **576 descriptive wide-MLP cells** (seeds 1–6) and 24 × 2 × 2 = **96 bar cells**, 1,824 in all. Cells live in their own roots, separate from every other study's.
- **Serving order (amended).** Right after admission, seeds 1–3 of the gnn, the twin and the wide MLP are served, with the bar (the bar cells are the admission cells). Seeds 4–6 are served when S6 files their md5s. Seeds 1–3 and seeds 4–6 go in separate cell roots.

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
- **Descriptive beside each test:** gnn vs wide MLP and twin vs wide MLP on the same 24 ids (raw p, no label, outside the Holm family: the primary stays gnn vs twin, Holm over 2); the all-cells row; seeds 1–3, seeds 4–6 and each seed alone; the after-360 s row; free-decisions row; gnn vs bar and twin vs bar (context); the runaway list per arm; per-task and request p50/p99 and the link-load rows, with the definitions of S7's `scripts_cosim/llf_reviewer_fresh24.py` (blob `c062ed0ec942475ee1d1836f4d680fa81152c023`, which is fixed to the `llf_cd_confirm_v1` ids; its extension for these ids is filed by blob with the rows, and these rows stay outside the reader freeze); the measured/charged decision-time ratios.
- **Not run, not quoted:** reactive and random. The wide MLP is served and read descriptively only; a wide-MLP cell that fails twice is absent: its pairs drop from the wide rows, it is counted, and it never blocks a label.

## What each outcome means (fixed now)

| Reading | Statement allowed |
|---|---|
| moderate CONFIRMED | the gnn beats its twin at moderate on fresh topologies (the parent's direction holds); not a message-passing attribution on its own (capacity: 507,524 vs 257,156 parameters) |
| moderate CONFIRMED and `llf_cd_confirm_v1`'s gnn vs the parameter-matched wide MLP (506k parameters; CONFIRMED there at both rungs) | together the strongest MP-adjacent claim this design allows: the gnn beats its twin and beats a same-size pointwise model. Still not an attribution to message passing alone: gnn vs wide is read here only descriptively (no label, outside the Holm family) |
| moderate DIRECTION | the sign repeats; no claim |
| moderate NOT-FASTER | the parent's moderate direction did not repeat; the TIE of `live_link_features_v1` stands |
| heavy CONFIRMED | stated as a direction plus Holm, not a magnitude (runaway-dominated) |
| heavy UNDERPOWERED / DIRECTION / NOT-FASTER | reported as labelled; no heavy claim |

## Reader

- `llf_confirm_read.py`, mode `cd`, extended with a `twin` arm and the 2-test family: the tests above, Holm over 2, the labels including UNDERPOWERED and the gnn-failure block, the failed-learned-cell rule with an allow-list of learned kinds, and the seed sets 1–6. It checks the id-list sha256; every cell's pin per group (static `02668ecf`, bar `0ec7cdcb`); each learned cell's `env.GNN_MODEL_PATH` basename against arm and seed (`uncut-linkstatic-gnn-eng-e40-seed{N}.pt` / `uncut-linkstatic-twin-e40-seed{N}.pt`); each learned cell's charge string against the table above (the wide MLP's `uncut-linkstatic-mlp-wide-e40-seed{N}.pt` and charge included); the bar's per-rung charge. It fails loud on any mismatch.
- **The default mode and both `llf_cd_confirm_v1` invocations must reproduce their filed outputs except `reader_sha256`** (tests run on the parent's sealed cells or a fixture, and on the fresh-24 `read_primary_cd.json` inputs).
- SCIENTIST freezes the blob, with the rulings file, before the first cell is read. The reader takes the seed set as a parameter (`--seeds`, as `scale_d_v1` v3), and **one frozen blob serves both reads below**.
- **Primary read (fixed now): seeds 1–6 pooled, run once**, after all 1,824 cells are in or the failed-cell rule has been applied. **Its verdict is the lineage's verdict, whatever the interim says.**
- **Interim read (amended 2026-10-25, the user wants results faster): gnn vs twin at seeds 1–3**, run once as soon as the 864 seeds 1–3 learned cells (gnn, twin, wide) and the bar cells exist (or the failed-cell rule has been applied to them). Same statistic, family (2 tests, Holm over 2), reading, guard and labels, reported as `INTERIM` (the reader's `--tag INTERIM` prefixes every label, for example INTERIM-DIRECTION) in its own output file. The invocation is `--seeds 1,2,3` on the seeds 1–3 root; the primary is the default seeds 1–6 on both roots, with the same blob. **The node head and any paper text quote the primary (seeds 1–6) only.** The interim appears only beside it, labelled INTERIM, and until the primary is filed it is reported to the user as interim and nothing else. The interim is descriptive for the lineage: **it cannot stop, extend or change anything** (not the seed set, the serving of seeds 4–6, the topologies, the labels or the primary), and no ruling made after it may cite its values. It looks at a subset of the cells the primary will pool, so its p-values are not independent of the primary's and are never combined with them.

## Prior (SCIENTIST)

P(moderate CONFIRMED) ≈ 0.45. The parent's −6.9 % was selected by raw p, so a smaller effect is expected, but n = 24 at six seeds detects about −3 to −4 % if the sign holds. P(heavy CONFIRMED) ≈ 0.1; P(heavy UNDERPOWERED) ≈ 0.35. The inputs: the moderate direction is 3/3 seeds negative at raw p .034, n = 12; the heavy parent rows show no direction; the best-val wave's spread includes a positive seed.

## Live gate

This node is itself the live gate: every number is a served simulation cell. Wall and queue estimates come from `llf_cd_confirm_v1` (2,016 cells) and are refreshed by S5/S6 before launch.

## Record

- 2026-10-25 — **PRIMARY read, seeds 1–6 pooled: CONFIRMED at both rungs** (SCIENTIST, run once; the lineage's verdict). Frozen blob `cbf7fb08`, the filed primary invocation on `cells_s123`, `cells_s456` and the bar's `cells_adm`. 1,824 cells as registered (1,152 gnn/twin, 576 wide, 96 bar), pins `02668ecf` / `0ec7cdcb`, ids sha `3d92a583…`, no failed cell, no gnn block. Output `mp_confirm_v1/reads/read_primary.json` (md5 `b8c8b75f5808850eba6ede9bc5e583ec`).
  - Seeds 4–6 inputs: S6 trained the twin s4–6 (jobs 906451–3, 40 epochs, W&B). SCIENTIST re-hashed the `-final` `.pt` and `.contract.json` against `xs/llf_conf_models_final_s456_twin.md5` (sha256 `24349aa1…3605`). The contracts differ from the s1 twin only in `train_seed`, with `disable_message_passing` true. S4 served the wave as jobs 907214–907225 (864/864, 0 failed).
  - **Primary, gnn vs twin, without runaways (all cells):**
    - Moderate −4.25 %, 21/24, p .0018, Holm .0036 (all cells −3.41 %, 19/24, p .084). **CONFIRMED.**
    - Heavy −12.73 %, 12/16, p .0155, Holm .0155 (all cells −11.47 %, 14/24, p .58). **CONFIRMED** (stated as a direction plus Holm).
  - **Seeds, without runaways.** Moderate: −2.8, −6.5, −9.0, −0.8, −3.8, −1.9 %. Heavy: −3.9, −4.9, −0.1, +3.5, −6.2, −9.3 %.
  - **Seed blocks, without runaways.**
    - Moderate: s1–3 −6.16 % (21/24, p 1.0e-5); s4–6 −3.18 % (18/24, p .027).
    - Heavy: s1–3 −5.52 % (14/18, p .018); s4–6 −11.2 % (13/20, p .23).
  - **After 360 s, without runaways:** −5.06 % moderate (21/24), −13.6 % heavy (13/14).
  - **Runaways (whole run, cells).**
    - Heavy: gnn 77 on 16 topologies, twin 73 on 16, wide 90 on 22, bar 20.
    - Moderate: gnn 6, twin 1, wide 10, bar 13.
  - **Descriptive, without runaways (raw p, outside the family).**
    - gnn vs wide: −10.23 % (22/23) moderate, −43.99 % (15/15) heavy.
    - twin vs wide: −6.74 % (19/23), −40.56 % (13/16).
    - gnn vs bar: −19.29 % (23/23), −25.83 % (15/18).
    - twin vs bar: −12.76 % (20/23), −14.12 % (14/20).
  - The interim (seeds 1–3, below) is a subset of these cells and is not combined with them.

- 2026-10-25 — **INTERIM read, seeds 1–3** (SCIENTIST, run once; descriptive for the lineage, the primary seeds 1–6 read is the verdict). Frozen blob `cbf7fb08`, the filed interim invocation on `cells_s123` plus the bar's `cells_adm` (960 cells, pins `02668ecf` / `0ec7cdcb`, ids sha `3d92a583…`). Output `mp_confirm_v1/reads/read_interim_s123.json` (md5 `f0a17e560514d1c69f071201a6ae2da3`). Labels: moderate **INTERIM-CONFIRMED** (−6.16 %, 21/24, Holm 2.1e-5; all cells −5.82 %, 20/24); heavy **INTERIM-CONFIRMED** (−5.52 %, 14/18, Holm .018; all cells −13.1 %, 17/24, p .75). Heavy rests on the reading without runaways: 23 of 48 (topology, window) pairs drop (gnn 17 cells flagged, twin 19).
- 2026-10-25 — **Reader frozen: `scripts_cosim/llf_confirm_read.py` blob `cbf7fb08e7c60ea228c2a33959f746d521eeb2c4`** (SCIENTIST, before any study cell is read; one freeze for `knative_stable_v1`, `scale_d_v1` and `mp_confirm_v1`). The rulings file is `live_link_features_v1/confirm_rulings.json` (md5 `b609acef33b10e7369a0f5aa58b42477`). Both filed reads are reproduced in every key but `reader_sha256`: `read_primary_A_epoch40.json` (default mode) and `llf_cd_confirm_v1/reads/read_primary_cd.json` (its frozen cd invocation). Tests: 40 passed, 1 skipped (that regression, run directly). An extended read records its options in `options`; a default read carries no new key. Invocations: the primary is `--mode cd --arms gnn,twin,wide,bar --primary gnn:twin --min-n 8` on both seed roots (seeds 1–6 by default); the interim is the same with `--seeds 1,2,3 --tag INTERIM` on the seeds 1–3 root. Add `[--failed-learned <cells>]`, `--cells`, `--ids`, `--ids-sha256` and `--pin static=02668ecf,bar=0ec7cdcb`. The twin is checked against `uncut-linkstatic-twin-e40-seed<N>.pt` and its d_conf_final charge.
- 2026-10-25 — **Signed** (SCIENTIST; four amendments the expert accepted, before any mint or cell): (1) a wide-MLP cell that fails twice is absent and counted, never fail-loud and never blocking; (2) the twin s4–6 contract check (`disable_message_passing` true, the rest equal to the s1–3 twin contract except identity); (3) quoting: the head and the paper use the primary only, and the interim appears beside it labelled INTERIM; (4) the reader's `--tag INTERIM` and the two invocations. The reviewer-defence rows are tied to `llf_reviewer_fresh24.py` by blob. The prior is filed.
- 2026-10-25 — **Amended before signing** (S7, at the expert's request; the user wants results faster): added the INTERIM read at seeds 1–3, kept the primary at the pooled seeds 1–6 read, serving in two steps, and the md5 rule per seed set. No family, label or statistic changed. Also added (the expert, same amendment): the wide MLP -final seeds 1–6 (already trained, md5s filed) as a served descriptive arm on the same 24 ids, +576 cells (1,824 in all), seeds 1–3 in the first wave; the outcome-table row no longer says the wide MLP is not run.
- 2026-10-25 — **Drafted** (S7, at the expert's request; the user wants the MP test). Unsigned. Sources read: `llf_cd_confirm_v1` (design, arms, charges, reader, labels), `live_link_features_v1` (parent outcome), `scale_d_v1` v3 (failed-learned-cell rule, UNDERPOWERED precedence, completion definition; merged at `47466dc1`), `knative_stable_v1` (admission, id rule), the twin charge strings and md5s from `d_conf_final` and `xs/llf_conf_models_final.md5` on datalab.
