# llf_cd_confirm_v1 — does the static link-feature GNN beat the CD bar and the parameter-matched MLP on fresh topologies?

**Status:** `CLOSED` (2026-10-25) — **CONFIRMED (4/4)**. Registered 2026-10-24, before any mint or cell. A1 (n = 24) and A2 (seeds 1–6, the user's POOL) were both filed before the primary read. The read ran once (`llf_confirm_read.py` blob `e7cd3aae`, `confirm_rulings.json` md5 `b609acef`).
- **Question.** Under option D, do the epoch-40 static link-feature GNN checkpoints of `live_link_features_v1` beat (a) the CD bar `cd_ib0_inflightfull_k100` and (b) the parameter-matched wide MLP, on 24 fresh topologies? There is no training here: the parent's checkpoints are served as filed, at seeds 1–6.
- **Outcome (2026-10-25).** **A graph model beats the CD bar and a parameter-matched MLP**, on 24 fresh topologies (30327–30350) with 6 seeds. Without runaways, Holm over 4:
  - **vs the bar:** moderate −21.1 % (21/23 topologies faster, Holm .003); heavy −26.1 % (18/20, Holm .003).
  - **vs the wide MLP:** moderate −9.3 % (22/23, Holm .0005); heavy −33.2 % (14/15, Holm .0005).
  - **Consistency.** Seeds 1–3 and 4–6 each favour the gnn on all four tests, and so does every seed alone.
- **Caveats a reader must not quote without.**
  - **Not a message-passing claim.** The design has no MP-OFF twin. The parent's twin tied the gnn (TIE), and at moderate it sat close to the MLP; that is a separate result on another topology set.
  - **Heavy is runaway-dominated.** Quote heavy as a direction plus Holm, not as a magnitude. Runaway (topology, window) cells at heavy: wide 31, MLP 27, bar 17, and **the gnn itself 16**. Heavy vs wide keeps 15 of 24 topologies; the heavy all-cells medians (−53 to −57 %) mostly measure the controls' runaways.
  - **The all-cells guard is a sign check only.** It holds on all four tests, but gnn vs wide at moderate with all cells is −9.5 % at p .065.
  - **Do not quote the reactive or random rows** (about −99.9 %). Under D these arms saturate the links. S7's diagnosis: about 124k transfers per cell at about 2,000 s each, a peak of 5,783 flows on one link, queue share about 1e-4. The request timeout covers only the rendezvous wait, not transfers (`src/placement/request_timeout.py:1–3`). The constraint regime for them is pending the user's call.
  - **One physics state** (option D, cap200). The decision charge is per arm and rung as measured; the wide MLP's charge is up to 3× the gnn's on groups over 50 tasks.
- **Path.** The registration, A1, A2, the reader freezes and the read are in the Record. The parent's motivation numbers are below.

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
| static gnn `ra_gnn_eng_combfl_linkstatic_e40_nosplit_k100` | 1–6 | `uncut-linkstatic-gnn-eng-e40-seed{1..6}.pt` md5 s1–3 `e801af4d…`, `ab24d859…`, `a2f98156…`, s4–6 `2c3a6f85…`, `a2d144ef…`, `7253890d…`; pin `02668ecf` | treatment |
| wide MLP `ra_mlp_wide_combfl_linkstatic_e40_nosplit_k100` | 1–6 | `uncut-linkstatic-mlp-wide-e40-seed{1..6}.pt` md5 s1–3 `23498572…`, `f4f01f8e…`, `cc654108…`, s4–6 `04357770…`, `7db997a7…`, `c670b6ae…`; pin `02668ecf` | primary control (parameter-matched) |
| bar `cd_ib0_inflightfull_k100` | 0 | pin `0ec7cdcb` | primary control (the hand bar) |
| same-input MLP `ra_mlp_same_combfl_linkstatic_e40_nosplit_k100` | 1–6 | `uncut-linkstatic-mlp-same-e40-seed{1..6}.pt` md5 s1–3 `94eda9fd…`, `d1454ae6…`, `ee8202d5…`, s4–6 `1f53ed26…`, `4e23da1f…`, `fc55f173…`; pin `02668ecf` | descriptive |
| `reactive`, `random` | 0 | pin `0ec7cdcb` | descriptive |

- **Checkpoints.** The files are in `/share/nikola.lukic/uncut_groups/xs/in_llf_conf_final/models/`, listed in `xs/llf_conf_models_final.md5` (md5 `a126f46b5f30dd88ba43795d91c20ab3`; SCIENTIST re-hashed all nine served `.pt`, and they match). `llf_cd_in.sh` links them into each new IN dir. The launch md5-checks every served `.pt` against that list before the first job.
- **No twin.** This is the user's call: this study asks about the bar and the MLPs, not message passing. **No result here is a message-passing claim.**
- **Decision charge.** Every cell carries the parent primary's charge for its (arm, rung). The bar is `HEROSIM_DECISION_COST_MS` 0.0116 (moderate) / 0.01098 (heavy), its filed per-rung w in `decision_cost_w.json`. Learned arms use `HEROSIM_DECISION_COST_MS_BY_BUCKET` (le10 / 11_50 / gt50, ms):

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

- Option D, windows **g0 + g1**, both rungs. Learned arms at seeds 1–6 (A2); bar, reactive and random at s0.
- That is 21 runs × 24 topologies × 2 windows × 2 rungs = 2,016 cells.
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

- 2026-10-25 — **Fixed-bar descriptive read** (SCIENTIST; not registered, no label). The bar priced only inbound exchange, so a cellular server's 4 MB/s uplink went unpriced. The fixed bar `cd_ib0_inflightfull_sym_k100` (s0, `HEROSIM_PG_EXCHANGE_SYMMETRIC=1`, pin `e616a946`, 96 cells in `llf_cd_conf/d_sym`, the user's go) also charges each remote partner's pull from the candidate. It is read against this lineage's filed cells and ids by the frozen reader `cbf7fb08`, imported unchanged by the wrapper `reads/sym_bar_read.py` (md5 `2aa674b2b18fa0ea8d6f8aec80b7d952`), which only registers the second bar kind with its own pin and checks. The wrapper reproduces all 80 filed secondary rows and the filed runaway list exactly. Output `reads/read_symbar_descriptive.json` (md5 `bdf19e76dc1d8b6dd6023005ce3bc6cd`). Without runaways (all cells), raw p:
  - fixed vs old bar: moderate −28.0 % (−34.9 %), 24/24, p 1.2e-7; heavy −25.9 % (−32.8 %), 20/24, p 3.2e-4;
  - gnn vs fixed bar: moderate **+11.9 %** (+12.5 %), 1/24 faster, p 2.4e-7; heavy −5.4 % (−0.1 %), 12/21, p .54. Every seed is +8.4 to +11.3 % at moderate and −3.3 to −9.7 % at heavy;
  - wide MLP vs fixed bar: +24.0 % (0/24) and +52.6 % (3/16). The same-input MLP: +21.8 % (0/24) and +3.5 % (7/17, p .31).
  - Runaway (topology, window) cells: fixed bar 0 moderate and 11 heavy; old bar 10 and 17.
  - **So under option D the CONFIRMED gnn-vs-bar margin depends on the bar's inbound-only pricing:** with the uplink priced, the hand rule is ahead at moderate and the heavy rung is not separated.

- 2026-10-25 — **CLOSED: CONFIRMED (4/4)** (SCIENTIST; the read is the entry below, `read_primary_cd.json` md5 `05d3cbe5c196ca0a5163408eabc13208`).

  | test | without runaways (primary) | Holm (4) | all cells (guard) | seeds 1–3 / 4–6, without runaways |
  |---|---|---|---|---|
  | gnn vs bar, moderate | −21.08 % (21/23, p .0031) | .0031 | −23.42 % (22/24, p .0022) | −20.3 % (24/24) / −19.2 % (21/23) |
  | gnn vs bar, heavy | −26.11 % (18/20, p .0014) | .0029 | −56.58 % (20/24, p .025) | −31.9 % (20/23) / −28.9 % (20/21) |
  | gnn vs wide, moderate | −9.27 % (22/23, p .00015) | .00049 | −9.47 % (20/24, p .065) | −8.4 % (23/24) / −10.7 % (22/23) |
  | gnn vs wide, heavy | −33.22 % (14/15, p .00012) | .00049 | −53.16 % (22/24, p .00085) | −25.9 % (16/19) / −27.2 % (15/18) |

  - **Per seed.** All six seeds are negative on all four tests in both readings. gnn vs wide spreads by seed: moderate −4.3 to −15.7 %; heavy without runaways −4.8 to −34.4 %.
  - **Runaways.** Distinct (topology, window) cells flagged at heavy: wide 31, mlp 27, bar 17, gnn 16. At moderate: bar 10, wide 5, gnn 4, mlp 1. Under the any-seed-flags-cell ruling, six seeds drop more pairs than three.
  - **Descriptive, without runaways.** After 360 s, gnn vs bar −17.0 % (21/23) moderate and −18.0 % (17/19) heavy; gnn vs wide −9.5 % (21/22) and −44.1 % (11/11). gnn vs same-input MLP −7.3 % (22/23) moderate and −6.9 % (11/15) heavy. Wide vs bar −10.8 % (22/24) moderate and +17.9 % (7/16) heavy. MLP vs bar −15.0 % (21/23) moderate.
  - **Not quoted.** The reactive and random rows: see the head.
  - **What it settles, and what it does not.**
    - Settled: under option D the static link-feature gnn beats the CD bar and a parameter-matched pointwise model on fresh topologies.
    - Not settled: why it wins. With no twin, model class, message passing and features are not separated.
    - Not covered: Knative under D, whose regime is pending.
- 2026-10-25 — **Primary read, run once** (SCIENTIST, with reader blob `e7cd3aae` at `d658321d`; the output's `reader_sha256` `947886fca295…` is that file's). The invocation is the frozen one in the A2 entry. Output: `llf_cd_confirm_v1/reads/read_primary_cd.json`, md5 `05d3cbe5c196ca0a5163408eabc13208`, 2,014 cells; `allowed_missing` holds the two random cells. Labels: gnn vs bar moderate CONFIRMED, heavy CONFIRMED; gnn vs wide moderate CONFIRMED, heavy CONFIRMED. Interpretation pending.
- 2026-10-25 — **Amendment A2: the primary learned arms run at seeds 1–6, and the reader is frozen at blob `e7cd3aae6c29f35981cf5c7f85a0bb068de010de`** (the user's POOL decision, relayed by the expert; filed by SCIENTIST before any primary cell is read). It replaces the default "descriptive" for (a) in the 2026-10-24 additions entry. The family, readings, guard and labels are unchanged.
  - **Seeds 4–6.** These are the static e40 `-final.pt` gnn, wide MLP and same-input MLP. The served files are in `xs/in_llf_cd_s456/models/`, listed in `xs/llf_conf_models_final_s456.md5` (sha256 `ae613e9843b84aadf0f6f1deab79278662e21fa5abcfb5f5457ab2f5a1ab9b47`). SCIENTIST re-hashed all nine, and they match.
  - **Cells.** The cells are in `llf_cd_conf/d_cd_s456`. Its 864 summaries are all at `02668ecf`, clean, and serve from `in_llf_cd_s456` (checked from the summaries' provenance fields only).
  - **Seeds and runaways.** A learned cell is the mean over its six seeds, and the runaway arm median pools all six. Seed blocks s1–3 and s4–6 are reported alone, as descriptive rows.
  - **Reader changes (SCIENTIST, at the expert's request).** cd mode now does four new things:
    - reads the learned arms at seeds 1–6;
    - takes several `--cells` roots;
    - takes the reactive cells from the flat admission directory via `--react-cells`, reading only `reactive_s0` cells of g0/g1, so the g2/g3 admission cells there are ignored;
    - writes `seeds` and `per_seed_block`;
    - with `--allow-missing`, accepts only named random-arm cells as absent and records them in `allowed_missing`. A named cell that exists, a non-random cell, or any other missing cell still fails loud. The random rows of an affected topology use its remaining window.
    - The default mode is untouched. Rerun on the parent's 1,200 cells, it reproduces `read_primary_A_epoch40.json` in every key but `reader_sha256`. Tests: 32 passed, 1 skipped (that regression, run directly).
  - **Random, descriptive.** 94 of 96 cells exist. 30327 and 30344 g0 moderate are deterministic watchdog stalls: the 30327 rerun in `d_cd_rr` stalled at the same simulated instant, 5,359 of 9,354 arrivals.
  - **Invocation:** `--mode cd --cells llf_cd_conf/d_cd,llf_cd_conf/d_cd_s456 --react-cells llf_cd_conf/cells_react --allow-missing cc40s30327__g0moderate__random_s0,cc40s30344__g0moderate__random_s0 --ids 30327,…,30350 --ids-sha256 8d9d17a1b7e5c45ea4eb8fa99e11ddca46ff3f32248f7ce1e727d44424d1c563 --pin static=02668ecf,bar=0ec7cdcb`, with `live_link_features_v1/confirm_rulings.json` (md5 `b609acef33b10e7369a0f5aa58b42477`). It is read once.
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
    - **Cells are checked against the registration.** cd mode checks the id sha256 and the per-group pins. It also checks each learned cell's `GNN_MODEL_PATH` basename against arm and seed, each learned cell's per-(arm, rung) charge against the Arms table, and the bar's per-rung charge (0.0116 / 0.01098 ms).
    - **The family and labels are as registered:** four tests, Holm over the 4 within each reading. The label reads without runaways, with the all-cells guard: CONFIRMED / DIRECTION / NOT-FASTER.
    - **Tests pass:** 28 passed, 1 skipped (the parent regression, which SCIENTIST ran directly above).
- 2026-10-24 — **Amendment A1: n = 24 topologies, not 12** (the expert's decision; filed by SCIENTIST before any mint or cell). The 24 ids are the lowest free run of 24 ≥ 30304, under the same exclusions and the same mint and admission rule; replacements are made as before. The cells grow to 1,152 at g0 + g1. Everything else is as registered.
- 2026-10-24 — **Registered** (SCIENTIST, at the expert's request with the user's go; before any mint or cell). The motivation table was recomputed from the parent's primary file. The checkpoint md5s and the per-(arm, rung) decision charges were read from the served files and from the cells of `d_conf_final` on datalab.
  - Changes from the expert's draft:
    - the primary reading is fixed as without runaways, with the all-cells sign guard;
    - the windows are fixed at g0 + g1, the parent's statistic;
    - the reader checks checkpoints and charges.
