# outbound_link_features_v1 — does a GNN that also sees the outbound side of the exchange beat the bar, its MP-OFF twin and the MLPs under option D?

**Status:** `REGISTERED` (2026-10-25) — registered before any v9 checkpoint, mint or cell. Drafted by S7 at the expert's request with the user's ruling on the primary contrast. Signed by SCIENTIST (2026-10-26) with nine amendments, before any v9 checkpoint, mint or cell, and before the counted probe run. The registered bar is `cd_ib0_inflightfull_k100` (pin `0ec7cdcb`).
- **Question.** Under option D (fair share plus TCP), do the `partial_state_v9` GNN and its separately trained MP-OFF twin, which add the outbound direction of the peer exchange to `partial_state_v8`, beat (a) the registered bar, (b) the v9 MP-OFF twin, (c) the parameter-matched wide MLP and (d) the same-input MLP, on fresh topologies?
- **Caveat on the bar (quote with every bar number).** vs the registered bar (`0ec7cdcb`), which prices peer exchange inbound only and so under-prices servers with a slow (cellular) uplink; not a comparison with a sound hand rule.
- **Why.** `llf_cd_confirm_v1` confirmed the static link-feature gnn against the bar and the wide MLP; `mp_confirm_v1` confirmed it against its MP-OFF twin (the parent's TIE in `live_link_features_v1` is superseded). Every link column of v7/v8 prices the partner → candidate direction. The pair table is symmetric, so each remote partner also pulls from the candidate over the candidate's own access uplink (cellular out 4 / in 75 MB/s). v5+ already carries the push direction at the solo rate (columns 25–26); no column prices it at the live fair-share rate, and none counts the candidate's own outbound flows sharing its uplink.
- **What this is not.** Not a comparison with a sound hand rule (caveat above). Not a message-passing claim unless the twin contrast passes on its own. v9 vs v8 isolates the new columns (secondary below); the primary family does not.

## Contract `partial_state_v9`

v8 byte for byte (columns 0–32), plus `LINK_OUT_DIM` = 4 scorer columns (33–36), one definition `placement/link_features.out_edge_columns`, over the task's committed partners and the active plus pending flows (the committed prefix refreshes them, as E1–E3):
- **R1–R3**, the mirror of E1–E3 on the route candidate → partner: log1p of the fair-share seconds, log ratio to solo, log1p of the busiest hop count.
- **U1**, log1p of the seconds the candidate's uplink needs for this task's k remote outbound flows (B bytes in all) when they share it with the c flows already on it: B (c + k) / (bw · MiB). Zero for k = 0.

The conv's edge attribute stays v8's, so the gnn and its twin read the same columns through the scorer. Code: merged on reference-physics as `eaca0224` (branch `s7/partial-state-v9`, commit `4326555c`), tests `tests/test_link_out.py` (definition against hand values, sequential columns against the batched loss on random plans, masks and prefixes, v8 columns untouched, v7/v8/v9 sidecar refusals). Serving refuses a checkpoint of another contract; a sidecar-less checkpoint is not served.

## Precondition: the label probe (before any recache or training)

The labels are the fixed fair-share replay of `live_link_features_v1`, reused unchanged (`label_override_union_flows.json` sha256 `11532616732908455841291c94ff822b5675fa44e0bc3a576ef4d2d7111fc066`). They price the reverse pull only if the replay charges a partner's pull from the candidate.
- **The probe** replays, at reference-physics `95293af0` with stored flows on, the served plan of a captured decision with one task A moved between a wired and a cellular candidate of the same platform type, both idle and initialised, every other placement held. Δ = cellular − wired. Δ_in is the change of A's own exchange time, Δ_out the change of its remote in-batch partners' summed exchange time, Δ_rtt the change of the batch's summed RTT (sum of doneTime − scheduledTime, the label). Predicted Δ_out is the solo transfer time over the cell's link topology.
- **Frozen files** (`docs/lineages/outbound_link_features_v1/`), md5 `84992aaaf26df10a8168b8c7e941f7f2` (`label_probe.py`), `4250862d1d5c61e74021ecb762b09a2b` (`label_probe.sbatch`), `4689c6451adc8b45e1f0dc621a52af1d` (`label_probe_read.py`). The sbatch refuses to run unless the script's md5 matches.
- **State set.** The union-corpus bar snapshots (`flowcap/collect_union`), every topology with a cell config, windows g0–g3, moderate rung, batches of 3–24 tasks, up to 120 evenly spaced whole-batch decisions per cell, up to 3 tasks per decision.
- **Criterion** (`label_probe_read.py`), on the rows with predicted Δ_out ≥ 0.5 s: at least 20 such rows; Δ_out > 0 on at least 90 % of them; the median Δ_out / predicted Δ_out in [0.5, 2.0]; the median Δ_rtt > 0. Anything else, fewer rows included, is FAIL and nothing trains.
- **One run counts:** a single run of the frozen sbatch after this node merges. Its rows and read are filed in the Record with their md5.

## Arms (one corpus, labels, recipe, epochs and seeds)

| Arm | Role |
|---|---|
| v9 gnn | treatment |
| v9 MP-OFF twin (`disable_message_passing`, the same columns and prefix refresh) | primary control |
| v9 wide MLP (hidden width matched to the v9 gnn's parameter count, within 1 %) | primary control (capacity) |
| v9 same-input MLP | primary control (pointwise) |
| bar `cd_ib0_inflightfull_k100` | primary control, with the caveat above |
| v8 static e40 gnn (`02668ecf`), seeds 1–6 from the filed lists `xs/llf_conf_models_final.md5` (md5 `a126f46b`) and `xs/llf_conf_models_final_s456.md5` (sha256 `ae613e98…`) | secondary: v9 vs v8, same corpus and labels |

- **Seeds and parameters.** All four v9 learned arms are trained at seeds 1–6 (24 trainings). The v9 gnn and wide-MLP parameter counts (seed-1 `.pt`) are filed in the Record before the wide MLP is trained for confirmation; the wide MLP's width is set so its count is within 1 % of the gnn's.
- **Recipe.** The parent's A6: 40 epochs, minimum 40, no early stop, wandb on; training configs `experiments/uncut_linkout_*.yaml` are added with the training run, after `tests/test_trainer_determinism.py` passes. Recache and training start only after the probe passes and this node has merged.
- **Decision charge.** Each v9 arm carries its own measured per-(arm, rung) charge, taken on the v9 code (the extra columns cost time). Never aliased to a v8 value. The bar keeps 0.0116 / 0.01098 ms per plan. The v8 secondary arm carries its `llf_cd_confirm_v1` charge. **Calibration runaway rule (e40 set):** a calibration cell whose averageElapsedTime exceeds 10× the median of the other kinds' cells at that rung is not used. That kind's w at that rung is re-measured on the next calibration topology, and the replacement is reported in its `_source`.
- **Cross-pin identity (before any cell is read).** The study is served at a v9 serving pin, the v8 arm and the bar included. A bar s0 cell and a v8 gnn s1 parent cell are rerun at that pin and must reproduce their filed summaries, as the standing identity check does.
- **Development (direction only).** Topologies 30176–30181 (already read under D, so post-look), g0 + g1, both rungs, seeds 1–3, 20 epochs, four learned arms plus the bar: 12 arm-seeds × 24 cells + 24 bar cells = 312 cells. It may change the recipe only; each change is filed before the confirmation mint. The early collapse check of the parent applies (hold at 2× the v8 gnn cell or 57 s per task on cc40s30176 g0 moderate, 30179 and 30181 g0 heavy).
- **Topologies.** 24 new ids at 24 servers from S5's joint rescan, disjoint from 30327–30350, 30505–30528, 30542–30565, the `knative_stable_v1` ids, the sealed 30285–30297, fresh6, the accel ids and every list in `live_link_features_v1`, `llf_cd_confirm_v1`, `mp_confirm_v1`, `scale_d_v1` and `uncut_groups_s0_v1`. S5 mints them (the parent's recipe: xs x1 mint at `224dd02e`, then the cap200 clip) and files the ids, their sha256 (of the comma-joined list) and the input md5s **before any cell**.
- **Admission: the bar alone, as `mp_confirm_v1` ("Admission is by the bar alone").** All four bar cells (g0 and g1 × moderate and heavy) have `queue_share` ≤ 0.80 and complete (`scale_d_v1` ruling 7: summary with `num_tasks` = 50,000 and a finite `endTime`, no `.failed.json`, no watchdog stall or wall kill; unknown is a fail). Reactive is not run; under D its queue share carries no information. The bar cells are the comparison cells. A topology that fails is replaced by the next free id, every replacement is reported, and a killed bar cell is rerun once with a longer wall first. The learned cells run only for admitted topologies.
- **Cells.** Option D, g0 + g1, both rungs, 96 cells per arm-seed: 4 learned arms × 6 seeds = 2,304, the v8 secondary 6 × 96 = 576, the bar 96: **2,976 confirmation cells**, plus the 312 development cells and the identity and smoke cells. Cells live in their own roots.

## Statistic and primary family

The parent's, with `confirm_rulings.json` (md5 `b609acef33b10e7369a0f5aa58b42477`) unchanged: metric `whole`, paired % per cell, a learned cell the mean over seeds, per topology the median over g0 and g1, then the median over topologies and the exact two-sided Wilcoxon, zeros dropped; runaway > 10× the arm's rung median.
- **Runaways and failures.** Seeds pooled; a learned cell is flagged if any of its seeds is; a flagged pair is dropped from the without-runaways reading; a topology with no window left is dropped. A learned cell that fails twice (killed, rerun once with a longer wall; a watchdog stall counts) drops its (topology, window, rung) pair from both readings and is counted per arm (`scale_d_v1` v3, "Failed learned cells"); **any gnn failure at a rung blocks CONFIRMED for the tests at that rung**; a control arm's failed cell never blocks; any other missing cell fails loud.
- **UNDERPOWERED is adopted** (`scale_d_v1`, item 4): CONFIRMED stands at any n; a test with n_eff < 8 topologies after the drops that is not CONFIRMED is labelled UNDERPOWERED in place of DIRECTION or NOT-FASTER. Holm stays m = 8 whatever the labels.
- **Primary family: 8 tests, Holm over the 8.** gnn v9 vs {bar, v9 twin, v9 wide MLP, v9 same-input MLP} × {moderate, heavy}. Primary reading without runaways; the all-cells reading is a guard (a test cannot be CONFIRMED while its all-cells median is ≥ 0).
- **Labels per test.** CONFIRMED: median < 0, Holm p < .05, all-cells median < 0. DIRECTION: median < 0, not CONFIRMED. NOT-FASTER: median ≥ 0 or no topology left. A contrast is CONFIRMED when both rungs are.
- **Secondary, Holm over its own pair:** v9 gnn vs v8 static gnn, moderate and heavy. Descriptive beside every test: the all-cells row, per-seed medians, the after-360 s row, the runaway list per arm. Heavy is runaway-dominated in this physics; quote it as a direction plus Holm, not a magnitude.
- **Reader.** `llf_confirm_read.py`, mode `cd`, extended for the v9 arm names, the 8-test family, the secondary pair and the rules above; it checks the id-list sha256, every cell's pin per group, each learned cell's checkpoint basename and decision-charge string against the arm and seed table. Its default mode must stay byte for byte: a test reproduces the default read and the filed cd reads (`llf_cd_confirm_v1`, `mp_confirm_v1` `read_primary.json` md5 `b8c8b75f…`, `scale_d_v1` read A `12f5c485…` and read B `b818cd18…`) in every key except `reader_sha256`. SCIENTIST freezes the blob before the first cell is read; one read. **Frozen** at blob `55429041` (merged `d7421382`); see the Record.

## Cost (rule 6: the live gate is part of the plan)

- **Cells:** 2,976 confirmation + 312 development + identity and smoke. mp_confirm_v1 served 864 learned cells as 12 jobs (907214–907225), about 72 cells per job, so the 2,880 learned confirmation cells are about 40 jobs, the bar and v8 cells included in that count only roughly; S4 refreshes the per-job wall from those jobs before launch (no wall figure is claimed here).
- **Training:** 24 CPU trainings at about 9 min per epoch × 40 epochs, about 6 h each, in parallel when the queue allows; the development trainings are 12 of 20 epochs.
- **44-job plan (cap shared by all sessions; count `squeue` before every submit):** recache (1 job) → development trainings (12) → development cells (about 5) → confirmation trainings (24, in waves if the cap binds) → confirmation cells in waves of at most 20 jobs, seeds 1–3 first.

## Prior (S7, rough; SCIENTIST sets theirs)

The bar contrast is expected to confirm: the v8 gnn already beat the same bar by −21 % / −26 % and the bar prices the outbound side wrongly. The informative contrasts are the twin and the v8 secondary. P(gnn v9 CONFIRMED vs v8 at either rung) about 0.3. P(gnn v9 CONFIRMED vs the v9 twin, both rungs) about 0.5, given `mp_confirm_v1`.

## Record

- 2026-10-26 — **Dev w filed and dev cells launched.** S6 measured the dev w on the seed-1 20-epoch checkpoints (job 907967, series `series_v9_d7421382_dev_r3`, H3 drift 0.10 / 0.31 %), merged as 1a1bb8fb; `scripts_cosim/v9_charges.json` md5 b58396dc (k = 1.0). w ms per plan, moderate / heavy: gnn 3.994 / 4.325, twin 3.438 / 3.991, mlp 3.457 / 8.503, wide 4.633 / 6.417. The 312 dev cells (jobs 908004–908019) run at serving pin 1a1bb8fb for every arm, bar included.
- 2026-10-26 — **Dev w note and e40 calibration rule** (SCIENTIST; filed before any dev cell is read and before the e40 calibration).
  - The MLP's heavy dev w (8.50 ms per plan, job 907967) was measured on a calibration cell that ran away: cc40s19461 g0 heavy, 881 s per task against 3.1–7.8 s for the other kinds, link capture 102 s against 5–7 s. The dev w stands as filed, so gnn vs mlp heavy in the dev read is slightly flattered (under 1 % of a heavy latency at first order).
  - The e40 calibration runaway rule is added under Decision charge above.
- 2026-10-26 — **Cross-pin identity REPRODUCES** at the v9 serving pin `d7421382` (S7, jobs 907808 and 907809, clean, cc40s30327 g0 moderate, option D; outputs `flowcap/v9_identity/`; checked independently by SCIENTIST). The bar s0 cell matches its filed `0ec7cdcb` cell (averageElapsedTime 628.6594533664218), and the v8 gnn s1 cell matches its filed `02668ecf` cell (4.666787813514169, `uncut-linkstatic-gnn-eng-e40-seed1.pt`, the same charge). Every simulated field and endTime is identical, and the tasks, batch-size and lock `.npy` files are byte-identical. Only wall-time and host fields differ (decisionTiming, total_rtt_plus_inference, wallclock_s, host), plus one added zero-valued counter in the bar summary.
- 2026-10-26 — **Reader frozen** (SCIENTIST, before any v9 cell): `scripts_cosim/llf_confirm_read.py` blob `5542904145e7f1647a9785ff0c81f658feeafc5e` (S7's `f4d5f582`, merged as `d7421382`), with `r1_attribution_v1_read.py` md5 `1cd0bfb6` and `confirm_rulings.json` md5 `b609acef`.
  - **Confirmation read:** `--mode cd --arm-set v9 --min-n 8 --charges <file> --pin <v9 serving pin>` plus `[--failed-learned <cells>]`, `--cells`, `--ids` and `--ids-sha256`. One pin for every arm; the group form is refused.
  - **Development read:** `--arm-set v9dev --tag DEV`, seeds 1–3, the 20-epoch checkpoints `uncut-linkout-*-seed<N>.pt`, direction only.
  - **Charges:** the `--charges` file is generated from the arm's own `decision_cost_w.json` entries (the dev `_linkout_nosplit` kinds for the development read, the `_linkout_e40_nosplit` kinds for the confirmation read), and its md5 is filed before that read. The reader records its sha256 and refuses a v8 value; the gate refuses an aliased v9 w.
  - **Reproduced** with this blob, in every key except `reader_sha256`: `mp_confirm_v1` `read_primary.json` (`b8c8b75f`), `scale_d_v1` `read_A.json` (`12f5c485`) and `read_B.json` (`b818cd18`), `llf_cd_confirm_v1` `read_primary_cd.json` (`05d3cbe5`), and the default-mode `read_primary_A_epoch40.json` (`7f882e8c`).
  - **Still before cells:** the identity rerun of a bar s0 cell and a v8 gnn s1 parent cell at the v9 serving pin.
- 2026-10-26 — **Signed** (SCIENTIST, before any v9 checkpoint, mint or cell, and before the counted probe run).
  - **Amendments, applied by S7 in `ba792fa6`:**
    - the frozen probe files, with the md5 gate in the sbatch and the ≥ 20-row clause;
    - bar-only admission as `mp_confirm_v1`, completion as `scale_d_v1` ruling 7, and the id sha256 filed before any cell;
    - the runaway, failed-cell and UNDERPOWERED rules;
    - the cross-pin identity check;
    - seeds 1–6 for all four v9 arms, and the 1 % parameter tolerance;
    - the cell count and job plan;
    - the merge sha `eaca0224`;
    - the reader reproduction set;
    - the status wording.
  - Checked: the three probe md5s match the files; the sbatch refuses a mismatched script and an existing output dir. The node names no other bar.
  - The probe's per-cell failures go to `fails.txt`, and the read does not gate on them. Their count is filed with the counted run.
  - **SCIENTIST's prior:**
    - P(probe PASS) ≈ 0.6.
    - P(CONFIRMED at both rungs): vs the bar ≈ 0.7; vs the v9 twin ≈ 0.3; vs the v9 wide MLP ≈ 0.45; vs the v9 same-input MLP ≈ 0.45.
    - P(v9 vs v8 CONFIRMED at either rung) ≈ 0.2.
    - The twin reads the new columns through the same scorer, so the new columns should help both arms. The gnn-vs-twin margin should stay near `mp_confirm_v1`'s, and heavy is runaway-dominated.

- 2026-10-25 — **REGISTERED.** Contract and tests merged as `eaca0224`. The probe files are frozen by md5 above; the counted probe run follows the merge of this node. No v9 checkpoint, mint or cell exists.
- 2026-10-26 — Counted label probe 907604: FAIL on the ratio clause (median 3.14, band [0.5, 2.0]); share 96.7 %, Δ_rtt +28.6 s, n 1,862, 40 topologies. Read: `outbound_link_features_v1/read.json` (md5 `1e70085083a8e8e10eb3cf3875d9f6fe`).
- 2026-10-26 — Probe FAIL on the ratio clause, explained by background uplink contention; overridden by the user (2026-10-26): discriminator 907654 shows the excess is fair-share contention from flows already on the cellular uplink (both-off ratio 1.00); labels unchanged. The five variants' rows were written inside 907654's run window (01:28:32–01:28:55); 907653 (31 s, 01:27) is the earlier v9-disc submission. Fixed state list `k1_states.json` (md5 `4937b553337fcd8bc164ffa7508785a7`), 20 states per variant, median Δ_out / predicted Δ_out (median Δ_rtt s): as-is 4.35 (42.9), fair-share off 0.93 (9.2), TCP off 4.46 (41.5), both off 1.00 (9.2), stored flows off 1.47 (11.0). The probe is not PASSED.
