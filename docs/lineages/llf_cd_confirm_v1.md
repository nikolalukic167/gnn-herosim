# llf_cd_confirm_v1 — does the static link-feature GNN beat the CD bar and the parameter-matched MLP on fresh topologies?

**Status:** `REGISTERED` (2026-10-24; amendment A1, n = 24, filed the same day before any mint). Signed by SCIENTIST at the expert's request with the user's go, before any mint or cell. Nothing below is read until the reader is frozen by blob.
- **Question.** Under option D, do the epoch-40 static link-feature GNN checkpoints of `live_link_features_v1` beat (a) the CD bar `cd_ib0_inflightfull_k100` and (b) the parameter-matched wide MLP, on 24 topologies none of them has been read on?
- **Parent.** `live_link_features_v1` (CLOSED TIE). Its primary file, read on the sealed 12, is the motivation. Those 12 ids are spent and are not reused here.
- **No training.** The arms are the parent's filed checkpoints, served as they were.
- **Reader frozen** (2026-10-24): `llf_confirm_read.py` blob `d6e186b5`, `--mode cd`. **Descriptive additions** (2026-10-24, the user's request): gnn/MLP seeds 4–6 and the (b) re-encode checkpoint. Neither enters the registered family unless the user pools seeds 4–6 before the read. Both are in the Record.

## Motivation (parent primary file, `read_primary_A_epoch40.json` md5 `7f882e8cd0c6de1bd1bdba750739a1be`, whole run; checked by SCIENTIST)

| contrast | moderate, without runaways | moderate, all cells | heavy, without runaways | heavy, all cells |
|---|---|---|---|---|
| gnn vs bar | −15.67 % (10/12, p .034) | −15.67 % (9/12, p .20) | −22.89 % (9/12, p .34) | −42.80 % (9/12, p .38) |
| gnn vs wide MLP | −10.35 % (12/12, p .0005) | −10.35 % (11/12, p .034) | −5.13 % (7/10, p .70) | −17.62 % (7/12, p .57) |
| gnn vs same-input MLP | −9.68 % (11/12, p .034) | −9.68 % (10/12, p .23) | −4.20 % (5/9, p .91) | −22.48 % (8/12, p .11) |

- Raw p, chosen after seeing six secondary rows. That is why this needs fresh topologies.
- **Parameters** (seed-1 `.pt`, the expert's count): gnn 507,524, wide MLP 506,251, MP-OFF twin 257,156, same-input MLP 25,985.
- **Heavy load is runaway-dominated.** In the parent primary, the heavy runaway flags were: wide 19, MLP 15, gnn 11, bar 7. So the heavy all-cells rows mostly measure who ran away.

## Arms (served exactly as in the parent primary)

| Arm | Seeds | Checkpoint / pin | Role |
|---|---|---|---|
| static gnn `ra_gnn_eng_combfl_linkstatic_e40_nosplit_k100` | 1–3 | `uncut-linkstatic-gnn-eng-e40-seed{1,2,3}.pt` md5 `e801af4d…`, `ab24d859…`, `a2f98156…`; pin `02668ecf` | treatment |
| wide MLP `ra_mlp_wide_combfl_linkstatic_e40_nosplit_k100` | 1–3 | `uncut-linkstatic-mlp-wide-e40-seed{1,2,3}.pt` md5 `23498572…`, `f4f01f8e…`, `cc654108…`; pin `02668ecf` | primary control (parameter-matched) |
| bar `cd_ib0_inflightfull_k100` | 0 | pin `0ec7cdcb` | primary control (the hand bar) |
| same-input MLP `ra_mlp_same_combfl_linkstatic_e40_nosplit_k100` | 1–3 | `uncut-linkstatic-mlp-same-e40-seed{1,2,3}.pt` md5 `94eda9fd…`, `d1454ae6…`, `ee8202d5…`; pin `02668ecf` | descriptive |
| `reactive`, `random` | 0 | pin `0ec7cdcb` | descriptive |

- **Checkpoints.** The files are in `/share/nikola.lukic/uncut_groups/xs/in_llf_conf_final/models/`, listed in `xs/llf_conf_models_final.md5` (md5 `a126f46b5f30dd88ba43795d91c20ab3`; SCIENTIST re-hashed all nine served `.pt`, and they match). `llf_cd_in.sh` links them into each new IN dir. The launch md5-checks every served `.pt` against that list before the first job.
- **No twin.** This is the user's call: this study asks about the bar and the MLPs, not message passing. **No result here is a message-passing claim.**
- **Decision charge.** Every cell carries the parent primary's charge for its (arm, rung). The bar is `HEROSIM_DECISION_COST_MS` 0.01098. Learned arms use `HEROSIM_DECISION_COST_MS_BY_BUCKET` (le10 / 11_50 / gt50, ms):

  | arm | moderate | heavy |
  |---|---|---|
  | gnn | 5.01465 / 2.254181 / 2.191336 | 6.151998 / 3.103575 / 3.678685 |
  | wide MLP | 4.527073 / 3.585821 / 7.915672 | 5.008803 / 5.134591 / 10.944349 |
  | same-input MLP | 4.275423 / 1.91878 / 2.003687 | 5.181011 / 3.17074 / 3.602202 |

  The wide MLP is charged up to 3× the gnn on groups over 50 tasks. These are measured values, not a choice, and they are part of every gnn-vs-wide number.
- **Physics, slate and recipe as the parent cells.** Option D: TCP on, fair share serial, default cellular uplink. cap200 inputs; `declared_pruning_v1_uncapped`, no split, k100. Cells run through `llf_cd_launch.sh` → `r1a_gate.sbatch` at the pinned worktrees.

## Topologies

- **Ids.** 24 new ids (amendment A1), the lowest free run of 24 ≥ 30304 under TE3's rescan rule. They are rescanned just before the mint.
- **Disjoint from:**
  - every id list in `live_link_features_v1.md` and `uncut_groups_s0_v1.md`;
  - fresh6 30298–30303;
  - the sealed 30285–30297 (30288 included, though it is not in the sealed set);
  - the 57 accel ids.
- **S5 mints them and files** the ids, their sha256 and the input md5s before any cell.
- **Mint and admission.** These follow the parent's confirmation recipe verbatim (`live_link_features_v1.md`, "Confirmation (read once)", items 1–5, and the 2026-10-23 kill ruling):
  - xs x1 mint at `224dd02e`, then the cap200 clip;
  - reactive s0 under D on g0–g3 × both rungs;
  - a topology is admitted only if all 8 of its reactive cells have `queue_share` ≤ 0.80. Unknown is not a pass, and only `queue_share` is read;
  - a killed cell is rerun once with a longer wall;
  - a topology that fails is replaced by the next free id, and every replacement is reported.

## Cells

- Option D, windows **g0 + g1**, both rungs. Learned arms at seeds 1–3; bar, reactive and random at s0.
- That is 12 runs × 24 topologies × 2 windows × 2 rungs = 1,152 cells.
- The descriptive reactive cells are the admission cells for g0 and g1.

## Statistic and primary family

- **Statistic.** The parent's, with the rulings of `live_link_features_v1/confirm_rulings.json` (md5 `b609acef33b10e7369a0f5aa58b42477`):
  - primary metric `whole`; runaway metric `same_as_stat`; arm medians pooled over seeds; a learned cell is flagged if any of its seeds is; a topology with no window left is dropped;
  - per cell, the paired % A vs B; a learned cell is the mean over seeds;
  - per topology, the median over g0 and g1; then the median over topologies, the count faster, and the exact two-sided Wilcoxon over topologies, zeros dropped.
- **The primary family has 4 tests**, Holm over the 4:
  1. gnn vs bar, moderate;
  2. gnn vs bar, heavy;
  3. gnn vs wide MLP, moderate;
  4. gnn vs wide MLP, heavy.
- **Primary reading: without runaways** (a pair is dropped when either cell is flagged), since that is the reading that motivated the study.
  - The all-cells reading is a guard, not a second test. Without runaways, a gnn runaway hides inside a dropped pair, so a test cannot be CONFIRMED while its all-cells median is ≥ 0.

## Labels (per test)

- **CONFIRMED:** median < 0, Holm p < .05, and the all-cells median < 0.
- **DIRECTION:** median < 0, not CONFIRMED.
- **NOT-FASTER:** median ≥ 0, or no topology left.
- **Descriptive, reported beside each test:**
  - the all-cells row, the per-seed medians and the after-360 s row;
  - gnn vs same-input MLP, gnn vs reactive and gnn vs random;
  - the runaway list per arm.

## Reader

- S5 extends `llf_confirm_read.py` with an `--ids/--arms` mode that implements this family and these labels.
- **The new mode must:**
  - check the id-list sha256 filed at the mint;
  - check every cell's pin per group (static `02668ecf`; bar, reactive and random `0ec7cdcb`);
  - check each learned cell's `env.GNN_MODEL_PATH` basename against the arm and seed table above, and its decision-charge string against the table above.
  - It fails loud on any mismatch.
- **The default mode must stay byte-for-byte.** A test reruns the default mode on the parent's sealed cells, or a fixture of them, and must reproduce `read_primary_A_epoch40.json` except `reader_sha256`.
- SCIENTIST freezes the blob, with this rulings file, before the first cell is read. The read runs once.

## Prior (SCIENTIST)

- P(CONFIRMED), per test:
  - gnn vs wide MLP, moderate: ≈ 0.7;
  - gnn vs bar, moderate: ≈ 0.5;
  - each heavy test: ≈ 0.15.
- **Why n = 24:** at n = 12 and Holm over 4, the bar test would likely need p < .0167 against a motivating raw p of .034. Doubling the topologies makes that reachable at the motivating effect size, if the effect holds on fresh topologies (winner's curse applies).

## Record

- 2026-10-24 — **Descriptive additions, filed before any fresh-24 cell is read** (the user's request, relayed by the expert; written by SCIENTIST). Neither changes the registered family or labels.
  - **(a) Seeds 4–6.** The static gnn, the wide MLP and the same-input MLP each train at seeds 4–6, with the parent's A6 recipe (40 epochs, min-epochs 40, no early stop; same corpus, labels and contract) and wandb logging. They serve their `-final.pt` on the fresh 24 (g0 + g1, both rungs) at the registered (arm, rung) decision charges.
    - Run `tests/test_trainer_determinism.py` before training. File the nine checkpoint md5s before the first cell.
    - Their cells go in a cells root separate from the primary's. The reader refuses unexpected cells and reads seeds 1–3 only.
    - **Default: descriptive**, reported as a seeds 4–6 row and a seeds 1–6 row beside each primary test.
    - Pooling them into the primary is the user's call, and it must be made before the primary read. Pooling changes the reader's seed set, so it also needs a re-freeze. If no pooling decision is filed by the time of the read, the primary is read on seeds 1–3 as registered.
  - **(b) Re-encode variant, as a descriptive arm.** Its only checkpoint is `uncut-linkconv-gnn-eng-seed1.pt` (md5 `f177c7c27748b7e82448e912d9b70074`; contract `6e09cd98…`, staged in `xs/in_link_conv`, `xs/in_llf_conf_reenc` and `xs/in_llf_conf_r1_reenc`, all identical).
    - It is the 20-epoch, seed-1, best-val checkpoint from the A1 selection. No retraining.
    - It is served on the fresh 24 and on the parent's sealed 12, at its re-measured charge (`live_link_features_v1`, 2026-10-24 (b) w entry: by bucket ≤10 / 11–50 / >50, moderate 8.050 / 6.554 / 9.366 ms, heavy 8.288 / 6.904 / 10.745 ms).
    - Caveats for every row: one seed; 20 epochs, not 40; selected on a validation metric that scores the sweep labels; seen on the development topologies in the selection. Never a label, never pooled.
- 2026-10-24 — **Reader frozen: `scripts_cosim/llf_confirm_read.py` blob `d6e186b504e652ea9750fadb15c3ee15cfc17438`** (S5's `--mode cd`, merged at `b91cdb33`; frozen by SCIENTIST before any fresh-24 cell is read). The rulings file is `live_link_features_v1/confirm_rulings.json`, md5 `b609acef33b10e7369a0f5aa58b42477`, unchanged.
  - **Invocation:** `--mode cd --ids <the 24 filed ids> --ids-sha256 <as filed> --pin static=02668ecf,bar=0ec7cdcb`. The `bar` group holds bar, reactive and random, so the descriptive reactive g0/g1 cells must run at `0ec7cdcb`.
  - **Checks by SCIENTIST:**
    - **Default mode unchanged.** Rerun on the parent's 1,200 cells (`d_conf_primaryA`), it reproduces `read_primary_A_epoch40.json` in every key but `reader_sha256`.
    - **Cells are checked against the registration.** cd mode checks the id sha256 and the per-group pins. It also checks each learned cell's `GNN_MODEL_PATH` basename against arm and seed, each learned cell's per-(arm, rung) charge against the Arms table, and the bar's 0.01098 ms.
    - **The family and labels are as registered:** four tests, Holm over the 4 within each reading. The label reads without runaways, with the all-cells guard: CONFIRMED / DIRECTION / NOT-FASTER.
    - **Tests pass:** 28 passed, 1 skipped (the parent regression, which SCIENTIST ran directly above).
- 2026-10-24 — **Amendment A1: n = 24 topologies, not 12** (the expert's decision; filed by SCIENTIST before any mint or cell). The 24 ids are the lowest free run of 24 ≥ 30304, under the same exclusions and the same mint and admission rule; replacements are made as before. The cells grow to 1,152 at g0 + g1. Everything else is as registered.
- 2026-10-24 — **Registered** (SCIENTIST, at the expert's request with the user's go; before any mint or cell). The motivation table was recomputed from the parent's primary file. The checkpoint md5s and the per-(arm, rung) decision charges were read from the served files and from the cells of `d_conf_final` on datalab.
  - Changes from the expert's draft:
    - the primary reading is fixed as without runaways, with the all-cells sign guard;
    - the windows are fixed at g0 + g1, the parent's statistic;
    - the reader checks checkpoints and charges.
