# live_link_features_v1 — do live link-sharing features let message passing beat its MP-OFF twin under fair-share physics?

**Status:** `CLOSED` (2026-10-24) — **TIE**. Registered 2026-10-23, before any data. Every amendment (A1–A6) and the reader rulings were filed before the confirmation read. The confirmation was read once with the frozen reader (`llf_confirm_read.py` blob `bdce5ed3`, `confirm_rulings.json` md5 `b609acef`).
- **Question.** Under option D (fair share plus TCP), does the GNN beat its own MP-OFF twin when both see the same live link-sharing columns and train on labels from a replay that keeps in-flight flows?
- **Superseded on the gnn-vs-twin contrast by `mp_confirm_v1` (CONFIRMED, 2026-10-25).** On 24 fresh topologies at seeds 1–6, the same static e40 gnn beats a separately trained MP-OFF twin at moderate (−4.25 %, 21/24, Holm .0036) and, as a direction, at heavy. The TIE below is this node's own read on its 12 topologies at seeds 1–3, and still stands as that read. It is no longer the answer to the contrast.
- **Outcome (2026-10-24).** Primary (epoch-40 wave, the user's option A): **TIE**, the weaker of all cells TIE and without runaways MP-DIRECTION.
  - **Moderate:** gnn vs twin −6.90 %. It is faster on 10/12 topologies with all cells (Holm .30) and on 11/12 without runaways (p .034, Holm .068). All three seeds alone are negative (−7.7, −4.6 to −5.0, −8.2 %).
  - **Heavy:** +13.0 % with all cells (4/12), −6.7 % without runaways (8/12, p .85).
  - **Not MP-ATTRIBUTED:** the Holm bar fails at moderate, and heavy has no direction that survives the runaway reading.
  - **The moderate direction is descriptive.** The registered labels have no per-rung label, so it is reported as a seed-consistent direction at moderate load only.
- **Caveats a reader must not quote without.**
  - One physics state (option D, cap200).
  - The variant (static) was picked at seed 1 and 20 epochs on development topologies, then carried at 40 epochs.
  - Heavy load is runaway-dominated: 191 runaway flags across all arms in the primary, the bar included. Runaways are real overloads of cellular uplinks (ρ > 1), not an artefact of the equal-split fair-share rule; max-min moves exchange by ≤ ~7 % (2026-10-25, under Motivation → Option D).
  - The best-val checkpoint metric scores the sweep labels, not the training labels, and selects at noise level. The secondary best-val wave reads MP-DIRECTION (moderate −6.58 %, 12/12, Holm .001), but it rests on a seed-1 twin served at its first evaluation (seed 1 −18 % / −80 %), and seed 3 is positive (+6.1 % / +3.8 %). It is not evidence for message passing.
- **Model class, descriptive (primary file).**
  - At moderate the static gnn is ahead of its MLP (−9.7 %) and of the parameter-matched wide MLP (−10.4 %, 12/12 without runaways). Its MP-OFF twin is close to the MLP (−1.8 %), so at moderate the gnn's edge over the pointwise arms is mostly the gnn-vs-twin difference.
  - Against the v6 fslabel gnn it is −1.4 % at moderate and +21.7 % at heavy (all cells).
- **Path.** The variant selection (A1 rule; STATIC, 2026-10-24), the A2–A6 training recipe, the admission and kill rules, and the reader rulings are all in the Record.

## Motivation (sources checked by SCIENTIST, 2026-10-23)

- **Option D** is TCP slow start (`slow_start_v1`, per partner) plus fair-share peer exchange (`processor_sharing_v1`, serial partners, directional links), with the default cellular uplink and no payload override. Source: the filed bar cell `d_cap200/heavy/cc40s30176__g0heavy__cd_ib0_inflightfull_k100_s0.summary.json` (blocks `peerExchangeTcp`, `peerExchangeFairshare`; `accessCellularOutMbps` null; pin `5e595e52`). Paths are under `/share/nikola.lukic/uncut_groups/anomaly/fairshare/`.
  - **The fair-share rule is equal-split, and the runaways are real overloads of cellular uplinks** (SCIENTIST, 2026-10-25, descriptive). `FairShare` gives each flow the minimum over its links of capacity / active flows. That is not max-min fair: a flow held down by a narrow link keeps its unused share on wide links. Five cells were rerun with `R1A_FLOW_LOG`, all fresh g0 moderate, and match their filed twins in mean and transfer count: 30327 heft, bar and gnn s1, and 30328 bar and gnn s1, in `/share/nikola.lukic/uncut_groups/diag_flowlog_30327/`. An offline replay of each flow log under the engine's rule reproduces the summary's `extra_seconds` to a relative error of ≤ 1.2e-14 and the peak flows exactly. Every runaway cell has server cellular uplinks (4 MB/s) offered ρ > 1 over the arrival window: bar 30327 has 6, at most 2.10; heft 30327 has 5, at most 4.04. No healthy cell has any link above ρ = 1; the highest is 0.72. The huge peak flow counts sit on 1,000 MB/s core links at 0.7–3 % utilisation and do not drive the runaways. Max-min sharing was replayed open-loop, with the logged placements and flow starts held fixed and no decision reacting. It cuts exchange seconds by ≤ ~7 % (runaways 3–5 %): the bar on 30327 goes 628 → 606 s per task and heft 1,980 → 1,873 s, so neither runaway disappears. On the healthy pair 30328, gnn vs bar mean latency moves −25.9 → −24.9 %. Scripts and results are in `live_link_features_v1/flow_replay_2026-10-25/`.
- **The fair-share label read** (S6; `d_fsl/read.txt` md5 `df11a84758441703703615f7e50798dc`). It used one seed (s1) and 20 epochs, on 30176–30181, g0+g1, both rungs, pin `d3b3de5a`. It is direction-only by its own header.
  - The fslabel gnn beats the fslabel MLP by a pooled −12.31 %, faster on 12/12 (moderate −10.89 % 6/6, heavy −28.65 % 6/6; each p = .0312, the n = 6 floor).
  - The fslabel gnn ties its fslabel twin: pooled −1.17 % (7/12, p = .73); moderate −2.36 % (4/6), heavy +1.16 % (3/6).
  - **"12/12" is 6 topologies × 2 rungs, not 12 topologies**, and the pooled p treats the two rungs of a topology as independent. Quote it as 6/6 at each rung.
- **The audit (verified in code).** The fidelity snapshot keeps in-flight tasks as ghosts, each with a fixed `io_remaining` taken from the static `_peer_exchange_time` estimate (`src/placement/snapshot_fidelity.py:131`, `record=False`). It captures no `FairShare` flow state (`network_fabric.py:206–260`). So a label replay under fair share (`uncut_ils.replay_rtts` → `plan_oracle._label_chunk`) prices sharing only among the batch's own transfers. This is why `uncut_label_screen.py:189` lets `--record-pair` keep the stored and replayed RTT apart instead of failing.
- **The collapse loop.** The first contention-aware hand rule (`cd_ib0_inflightfull_fsaware`, which reads live flow counts) ran at **127.5 s per task vs the bar's 7.31 s** on cc40s30176 g0 moderate (`d_cap200/diag_herd_moderate/*.tasks.npy`, mean of column 1 − column 0, SCIENTIST recomputed it; corrected 2026-10-23 from "mean of column 1"). Two fixes each removed it: counting the batch's own committed flows (`_commit`, 6.18 s), and pricing the drain walk solo (`_nodrain`, 6.05 s). The mechanism is herding: every task in a batch sees the same currently empty link and picks it. A learned scorer that reads live flows can herd the same way.
- **The D grid read** (`d_cap200/d_read_main.txt` md5 `9b7fd746db5dc7237a62e7cca298b634`, rules `d_read_rules.txt` md5 `cba0597e967169cf184c3f11618633a0`) holds the statistic and runaway rule used here. Its 192-cell block flags 30 runaway cells, 22 of them at heavy g0, including the bar itself at 862 s on 30179 (corrected 2026-10-23 from "19 at heavy g0 alone"). The direction-only reads under D that preceded this registration are filed in `uncut_groups_s0_v1.md` (Record, 2026-10-23, "Option D direction-only reads").

## Hypothesis

Message passing has something to act on only when (a) the scorer sees which links are already carrying flows and (b) the labels reward avoiding them. With (a) and (b) both given, the v7 GNN beats its v7 twin, which sees the same columns. If it ties, the fair-share gain over the MLP is information plus model capacity, not message passing.

## Arms (all trained on one v7 corpus with one label set; recipe fixed before confirmation)

| Arm | Role |
|---|---|
| v7 gnn | treatment |
| v7 twin | **primary control.** MP-OFF (`disable_message_passing`), the same `partial_state_v7` columns, corpus, labels, epochs and seeds |
| v7 MLP | pointwise control, same columns |
| v7 wide MLP | parameter-matched pointwise control (hidden width matched to the v7 gnn's parameter count, as `uncut_comb_fl_mlp_wide_fslabel.yaml` did for v6) |
| v6 fslabel gnn / twin / MLP | references only (filed checkpoints, `experiments/uncut_comb_fl_*_fslabel.yaml`) |
| bar `cd_ib0_inflightfull_k100` | reference, at its frozen w |

Conditions on the arms:
- **The v7 columns refresh with the committed prefix.** Within a batch, the link columns a task is scored on must include the flows of the batch-mates already decoded (the `_commit` fix, through the sequential decode's committed prefix, `src/policy/gnn/seq_decode.py:834–838`). This must be identical for the gnn, twin and MLP arms.
  - Without this, the twin is blind to its own batch and loses to the herding loop: that would be an information gap, not a message-passing result.
  - What message passing alone adds is foresight about partners not yet decoded.
- **The parity test (S4)** shows two things on captured states:
  - the columns built at capture equal the columns built at serve, byte for byte;
  - the gnn and twin graphs carry byte-identical v7 columns.
- **Decision charge.** Each v7 arm is charged its own w, measured on the v7 code before confirmation (the link columns cost time). It is never aliased to a v6 w, since an alias would favour v7.
- **Comparisons with v6 change labels and features together.** Any v7 vs v6 contrast is reported as that pair of changes, never attributed to either one.

## Corpus and labels

- **States.** 16 Arm 1 training topologies captured under D (S6), with in-flight flows stored in each state. The capturing arm, windows and state sampling are named by S6 and filed before capture. They are the same for every v7 arm.
- **Labels.** The fixed fair-share replay (S7), which re-creates the stored flows. Search as the fslabel labels used it, with only the replay changed.
- **Replay identity (S7), filed before any label is used:**
  1. **Off-switch identity:** with fair share off, the fixed replay is byte-identical to the current replay on the filed fslabel test states.
  2. **Flow identity:** a state replayed with no plan applied reproduces the stored flows' remaining bytes and completion times.
  3. **Live fidelity of the served plan:** replay-vs-live gap on ≥ 200 captured states, reported before and after the fix.
     - Equality within `REL_TOL` is required where no flow that starts after capture touches the batch's links.
     - Elsewhere the median absolute gap must shrink.
- **Reported once, descriptive:** the share of states whose label plan changes against the fslabel labels.

## Early live check for the collapse loop (before the development grid)

- **What runs.** v7 gnn and v7 twin s1, one cell each, on the cells where loops are known: cc40s30176 g0 moderate (the 127 s fsaware cell) and cc40s30179 / cc40s30181 g0 heavy (bar runaways at 862 s and 677 s).
- **Hold rule.** Hold, and diagnose before going on, if any cell's mean per-task latency is above either of:
  - **2×** the filed fslabel gnn s1 cell on the same (topology, window, rung);
  - **57 s** (10× the bar's D-grid arm median).
- **Doesn't count as evidence.** These cells are not used in any read.

## Development (direction only)

- **Grid.** Option D, 30176–30181, windows g0+g1, both rungs, seeds 1–3, 20 epochs. All v7 arms plus the references.
- **These topologies have already been read under D** (`d_cap200`, `d_fsl`), so development is post-look. At n = 6 the smallest exact two-sided p is .03125, so development gives a direction and never a label.
- **What it may change:** the recipe only (epochs, checkpoint selection, learning rate, column scaling). Each change is filed before the confirmation mint.
- **The confirmation still runs whatever development shows** (AGENTS.md rule 6). Only a failed collapse check, or a failed identity or parity check, holds it.

## Confirmation (read once)

- **Topologies.** 12 newly minted, by the lowest-free-run rule rescanned just before the mint and disjoint from every list in use. The admission rule is the one 30176–30181 passed under D, filed verbatim before the mint.
  - **Admission rule as found (S5, 2026-10-23, filed before the mint; sources checked).** There is no rule written for D. 30176–30181 were admitted once, with the other xs ids 30182–30187, by the exchange-sensitivity rule, under the default physics at the uncapped x1 payload, not under D:
    - Registration, `uncut_groups_s0_v1.md` amendment (c): "Reactive (`reactive` at `a459dd98`) runs first. A topology is admitted for a variant only if all 8 of its reactive cells have queue share ≤ 0.80; unknown is not a pass."
    - Reader `docs/lineages/uncut_groups_s0_v1/xs/xs_react_read.py` (md5 `ec08d4f8a5805dfa5633b076c3a1a608`), header: "A topology is admitted iff all 8 of its reactive cells (4 windows x 2 rungs) have queue_share <= 0.80; unknown (missing, failed, no field) is not a pass." It reads only the `queue_share` field of the 96 `reactive_s0` summaries. Result for x1: 12 of 12 admitted, max queue share 0.1666 (`xs/admission_summary.txt`, selection `xs/xs_selected.json` md5 `75e7fef5833415ad4421bd241b828d35`).
    - So "the rule 30176–30181 passed under D" does not exist. What those ids did under D is the D grid read: the bar itself ran away on 30177–30181 at heavy g0 (102, 244, 862, 58 and 677 s per task; `d_read_rules.txt`), cells that the xs rule, run on the default physics, did not see.
  - **Proposed rule for the 12 new ids (needs the expert's and SCIENTIST's OK before the mint; S5 mints nothing until then).**
    1. Mint like xs x1 (`big_groups_inputs.sbatch` at `224dd02e`, 160 clients × 24 servers, p 0.6, single origin `x1_uncut`, unscaled M, `fastest_compatible`, `cc40s16345.json`), then clip every peer-exchange payload at 200 MB with the `cap200_build.sbatch` recipe (md5 `e05fbf3d1647037564bd9fe6e90532aa`; cellular 4 MB/s unchanged). The clipped tree is the cells' input.
    2. Reactive admission as the xs rule, on the cells' own inputs: `reactive` s0, 4 windows × 2 rungs = 96 cells, run under option D (TCP and fair-share on, the D pin) so that the load it measures is the load the cells carry. A topology is admitted only if all 8 of its reactive cells have `queue_share` ≤ 0.80; unknown (missing, failed, no field) is not a pass. Only `queue_share` is read.
    3. Nothing else gates admission. In particular the bar's per-task mean is **not** read before the screen: on 30176–30181 it ran away under D in 5 of 6 topologies at heavy g0, so a bar-runaway exclusion would remove most of the development set and would be a rule chosen after seeing outcomes. Runaways are handled by the registered all-cells and without-runaways readings (> 10× the arm's rung median).
    4. A topology that fails is not replaced by hand: S5 mints the next free id by the same rule and reports the replacement before any cell; the id list and its sha256 are re-filed. Count of failures reported either way, even if 0.
    5. **Kills (2026-10-23 ruling, Record):** a cell killed by the per-cell LIMIT or the job wall is rerun once with a longer wall only; if it fails twice, its topology is replaced as in 4.
    5. Sealed: inputs md5s and the id-list sha256 filed in the mint line; nobody reads outcomes before the reader is frozen.
  - **Confirmation ids, filed (S5, 2026-10-23; the sealed set from now on; only `queue_share` was read).** The 12 ids are **30285, 30286, 30287, 30289, 30290, 30291, 30292, 30293, 30294, 30295, 30296, 30297**; sha256 of the comma-joined list `f165536615a17be19f9bb5a01736dacb7f30255f85370d32846b3b177dbf179a`. **30288 → 30297:** 30288's g3 heavy reactive cell was killed at LIMIT 14,400 s and again on its rerun at 28,800 s (unknown, not a pass), so 30288 was replaced by the next free id under the filed rule (30297, checked against every local and remote ref, the share and simdata names, the sh/sbatch text and the provenance seeds).
    - **Admission** (reactive `queue_share` ≤ 0.80 on all 8 cells under D at `a9733951`; unknown is not a pass): 12 of 12 admitted, none failing, none unknown. Max `queue_share` 0.0025 over the 11 ids of the first mint, 0.00025 for 30297. Five reactive cells were killed at the first limit (30288 g0 heavy, g0 moderate, g3 heavy; 30289 g0 moderate, g0 heavy), four completed on rerun at LIMIT 28,800 (jobs 902250, 902252, 902441), and 30288 g3 heavy (902251) did not. Their `.failed.json` files are in `llf_conf/cells_react_killed/`. 30297's 8 cells ran at LIMIT 43,200 (jobs 903233–903240, all COMPLETED).
    - **Mint** (recipe commit `224dd02e`, worktree `uh_xs_224dd02e`; 160 clients × 24 servers, p 0.6, single origin `x1_uncut`, unscaled `M_MODERATE` 11.6139 / `M_HEAVY` 27.622665025860204, `fastest_compatible`, `cc40s16345.json`; every peer-exchange payload clipped at 200 MB by the `cap200_build` recipe, md5 `e05fbf3d1647037564bd9fe6e90532aa`; cellular 4 MB/s unchanged). Roots under `/share/nikola.lukic/uncut_groups/`: `llf_conf/` (ids 30285–30296; `ids.txt` md5 `4e6c9a3154be49df0686448b34b90033`; x1 `inputs/MD5SUMS` `9ccfcae3710742c76a1169b90a80f4f6`, `inputs_group/MD5SUMS` `21e6834574289c54c677cca4505b93da`, cap200 `inputs_group/MD5SUMS` `320a8b08a63e3ed8d31f2fb7eb04eb90`; cfg md5 `c07a5424b89456ce2c241acda06305ac`, identical in x1 and cap200) and `llf_conf_r1/` (id 30297; `ids.txt` md5 `5c916bd2c1ace501cfd52264e38ea159`; x1 `inputs/MD5SUMS` `51c2734b37ee6f2d1bc5196e20296986`, `inputs_group/MD5SUMS` `61bd52f04207a2382fd4e00589b68259`, cap200 `inputs_group/MD5SUMS` `7993ce18bc5e566381cb5222f4e6da2f`; cfg md5 `c13e44490c818a8f182913d733e3f889`). The cap200 inputs the cells use are `llf_conf/in_cap200` and `llf_conf_r1/in_cap200`. 30288's other cells exist in `llf_conf/` but 30288 is not in the set.
    - **Files** (`docs/lineages/live_link_features_v1/confirmation_ids/`): `llf_mint.sh` md5 `b8d666feb2959703d1c325f25e9a0a86`, `llf_mint1.sh` `63a57d11e628254c7baa8aef1541f7e8`, `llf_react.sh` `65f194a1dbd26f944921635fdc08de6d`, `llf_react1.sh` `9c9ad33333f903221df52b2441b3c669`, `llf_rerun.sh` `2ee32c35901344c6e64680daf09d30ea`, `llf_react_read.py` `88fc3224de0055724c70ff726d42884c`, `llf_react_read1.py` `f55de7cd9fe7777c132d7598fc3ac907`, `admission_12.json` (the first 12-id read: 11 admitted, 30288 failing) `c2da4a73cf2906d69cd47b2d499d32f8`, `admission_final.json` (the final 12) `77c3a2afbf6df13d730aee7aae9ee902`.
- **Cells.** Option D, g0+g1, both rungs, the development recipe as frozen, seeds 1–3, all arms.
- **Statistic (as `d_read`).**
  - The paired % per cell, A vs B at the same topology and window.
  - A learned cell is the mean over seeds of the per-seed cell value.
  - Per topology, the median over the 2 windows; then the median over 12 topologies, and an exact two-sided Wilcoxon over topologies (zeros dropped).
  - A cell is a runaway if it is > 10× its arm's median in that rung. Every contrast is reported with all cells and without runaways (dropping a pair when either cell is flagged).
- **Labels, primary (v7 gnn vs v7 twin)**, read on 2 tests (moderate, heavy) with Holm over the 2. The label is the weaker of the all-cells and the without-runaways reading. The first that holds applies, in this order:
  1. **TWIN-FASTER:** on either rung, the median is ≥ +2 % and Holm p < .05.
  2. **MP-ATTRIBUTED:** on both rungs, all of these hold:
     - the median is ≤ −2 % and Holm p < .05;
     - each seed alone gives a median ≤ −1 %;
     - the v7 gnn is not slower than the v7 MLP on either rung.
  3. **MP-DIRECTION:** both medians are < 0, with ≥ 8/12 topologies faster on at least one rung.
  4. **TIE:** anything else.
- **Secondary, descriptive.** v7 gnn vs v7 MLP, v7 wide MLP, the bar, and each v6 fslabel arm; v7 twin vs v7 MLP. Each is reported at both rungs, both ways for runaways, with the after-360 s row beside the whole run.
- **Prior (SCIENTIST).** P(MP-ATTRIBUTED) ≈ 0.15.
  - Against it: the fslabel twin tie (−1.17 %); the 6/6 dispersion at heavy (per-topology −93 % to +84 %); and two earlier closures where message passing tied pointwise (`link_mp_v1`, `gnn_seeded_cd_mp_ablation_v1`).
  - For it: this is the first set of labels that prices contention from outside the batch.

## Record

- 2026-10-24 — **CLOSED: TIE (confirmation, read once).** Reads by S5 at reader `f16c359a` (blob `bdce5ed3`; the output's reader sha256 `699326f575f5…` equals that blob's file). Outputs are in `live_link_features_v1/confirmation_reads/`: primary `read_primary_A_epoch40.json` md5 `7f882e8cd0c6de1bd1bdba750739a1be` (pins static `02668ecf`, refs and bar `0ec7cdcb`; 1,200 cells); secondary `read_secondary_bestval.json` md5 `35815889e8e59509199dc5d2ea721e2e` (all at `0ec7cdcb`; 1,200 cells). Interpretation by SCIENTIST.
  - **Primary, gnn vs twin (whole run):**

    | reading | moderate | heavy | label |
    |---|---|---|---|
    | all cells | −6.90 % (10/12, p .23, Holm .30) | +13.01 % (4/12, p .15, Holm .30) | TIE |
    | without runaways | −6.90 % (11/12, p .034, Holm .068) | −6.68 % (8/12, p .85) | MP-DIRECTION |

    - **Per seed (moderate / heavy).** All cells: s1 −7.66 / −13.57, s2 −4.62 / −0.34, s3 −8.24 / +3.15. Without runaways: s1 −7.66 / −10.71, s2 −5.03 / −9.15, s3 −8.24 / −7.67.
    - **gnn vs mlp_static:** −9.68 / −22.48 % (all cells).
    - **Label: TIE**, the weaker of the two readings. MP-DIRECTION fails with all cells because the heavy median is positive. MP-ATTRIBUTED fails on Holm at moderate (.068) and on heavy.
  - **Secondary, best-val wave (descriptive):** MP-DIRECTION on both readings, moderate −6.58 % (12/12, Holm .001) and heavy −62.4 % (all cells).
    - It is driven by seed 1, −18.1 / −79.6 %, where the twin is served at its first evaluation (A6 notes). Seed 3 is +6.07 / +3.78 %, so the seed clause fails.
    - This is what the checkpoint-metric finding predicts. It is not quoted as evidence for message passing.
  - **Secondary rows, primary file (whole run).**
    - **All cells:**
      - moderate: gnn vs wide static −10.35 % (11/12), twin vs mlp static −1.80 % (8/12), gnn vs bar −15.67 % (9/12);
      - heavy: gnn vs bar −42.80 % (9/12);
      - gnn vs the v6 fslabel gnn: −1.40 % moderate, +21.73 % heavy.
    - **Without runaways:**
      - moderate: gnn vs wide static −10.35 % (12/12), gnn vs bar −15.67 % (10/12);
      - gnn vs the fslabel gnn: −1.40 % moderate, −1.76 % heavy.
    - The after-360 rows have the same signs at moderate (gnn vs twin −6.88 %).
  - **What it rules out, and what it leaves.**
    - Rules out: a message-passing win under option D with these features and labels at this power (12 topologies, 3 seeds, 40 epochs).
    - Leaves open: a moderate-load direction, seed-consistent and Holm .068. Any follow-up needs its own registration and fresh topologies, and must not reuse these 12 as confirmation.

- 2026-10-24 — **Confirmation reader re-frozen: `scripts_cosim/llf_confirm_read.py` blob `bdce5ed36eacbadeed84dca764a4ba33192f1a79`** (was `eca50ed9`; S5's per-group pin check, merged at `65bcf62b`; checked by SCIENTIST: only `load()`/`run()`/`main()` pin handling and the docstring change, and the statistic, rulings and labels are untouched; before any confirmation cell is read). The primary mixes static cells at `02668ecf` with reference and bar cells at `0ec7cdcb`. `git diff 0ec7cdcb 02668ecf` changes only `scripts_cosim/datalab/llf_confirm_in.sh` and `llf_confirm_launch.sh` (nothing in `src/` or `scripts_cosim/*.py`), and `0ec7cdcb` is an ancestor of `02668ecf`. Same `confirm_rulings.json` (md5 `b609acef33b10e7369a0f5aa58b42477`). Primary invocation: `--pin static=02668ecf,refs=0ec7cdcb,bar=0ec7cdcb`.

- 2026-10-24 — **Confirmation reader frozen: `scripts_cosim/llf_confirm_read.py` blob `eca50ed97ef63debff4a03baf508905700886ea8`** (S5 wrote it from the registered text, merged at `7dcac079`; frozen by SCIENTIST before any confirmation cell is read). The confirmation is read once with this blob and `live_link_features_v1/confirm_rulings.json` (md5 `b609acef33b10e7369a0f5aa58b42477`).
  - **Rulings, from the registered text.** Primary on the whole run. The runaway test uses the statistic being read, with the arm median pooled over seeds. A seed-mean cell is flagged if any seed is, and its pair is dropped. A topology with no window left is dropped. "Weaker" follows the P9 order (MP-ATTRIBUTED > MP-DIRECTION > TIE > TWIN-FASTER). The MLP clause is a gnn-vs-MLP median ≤ 0 % on both rungs. The seed clause is seed-matched within the reading. MP-DIRECTION needs a count of 8 topologies faster. Reasons per point are in the file.
  - **Arms.** The registered "v7" arms are the selected STATIC variant's 40-epoch arms (gnn, twin, MLP, wide static), with the v6 fslabel arms and the bar as references. The v7 MLP and v7 wide MLP references of A3 are not in this read.
  - **Two waves.** `d_conf` serves the best-validation checkpoints (A6). `d_conf_final` (pin `02668ecf`) serves the epoch-40 `-final.pt` checkpoints and is read by the same blob and rulings. Both were decided before any confirmation cell was read. **Which wave is primary is the user's choice, pending.** The other wave is a disclosed secondary, never pooled.
    - **Primary (the user, option A, decided before any confirmation cell was read):** the epoch-40 wave, `d_conf_final` static cells at `02668ecf` with the shared reference and bar cells from `d_conf` at `0ec7cdcb`. **Secondary:** the best-val wave (`d_conf`), never pooled. The reason: the best-val checkpoint metric scores the sweep labels, not the training labels, and selects at noise level (the seed-1 twin is served at its first evaluation).
  - **A6 checkpoint notes (training logs only).**
    - The best-val epoch is ≥ 35 for twin s3 (38) and MLP s1 (39).
    - The best-val checkpoint is the first evaluation for twin s1, wide s1 and wide s3 (val CE 41.1, 40.7, 39.0, against 33–35 for the other seeds; train CE falls to about 17 by epoch 40).
    - The checkpoint metric, val/regret_masked_topo, is computed on validation labels that keep the sweep label ("val/test keep the sweep label"), not the flow-replay labels the arms train on. Every arm's best is 18.48–19.02 s, against a random-plan regret of 9.9 s and a worst-plan regret of 19.7 s on that set, with about 20 % of decodes mapped.
    - So best-val selection is at noise level here, and the best-val wave serves a seed-1 twin taken at its first evaluation.

- 2026-10-24 — **A1 selection read: STATIC is carried to seeds 1–3 (40 epochs, A6) and confirmation** (SCIENTIST ran `llf_select.py` blob `54367447` once, 10:42Z, on the expert's relay; all flags TRUE; output `live_link_features_v1/selection_s1_e20.json` md5 `9de058fac7e656bac899990b002b1574`, flags `selection_flags.json` md5 `b05fa09ee422d684389b199c8de6cd7a`).
  - **Inputs.** Cells: as built `anomaly/fairshare/d_link_asbuilt/asb_*`, static `d_link/gw_*` (gnn) and `d_link/tm_*` (twin), (b) `d_link_conv/conv_*`; 48 per variant.
  - **Scores** (gnn vs its own control twin, pooled median over (topology, rung), seed 1, 20 epochs; positive means the gnn is slower):

    | variant | without runaways | n | all cells | runaway cells |
    |---|---|---|---|---|
    | as built | +2.47 % | 12 | +2.85 % | 3 |
    | static | **+0.97 %** | 12 | −0.91 % | 7 |
    | (b) re-encode | +14.89 % | 11 | +25.84 % | 4 |

  - **Choice:** static, by the lowest score without runaways.
  - **Read as a direction only (A1).** In every variant the gnn is not ahead of its own twin at seed 1. Per topology at moderate load: the as-built gnn is ahead on 5 of 6 topologies (−12.7 to +5.6 %); static is within ±5.2 %; (b) is slower on 5 of 6 (+10.9 to +15.2 %). Heavy load is dominated by single-cell swings (−86 % to +566 %) that the 10× runaway rule does not flag.
  - **Collapse labels, reported beside the selection (ruling i, not a gate):** environment on 30181 g0 heavy for the as-built gnn and twin, the static gnn and twin and the (b) twin; static also on 30179 g0 heavy (S6: no herd excess, and not separable from backlog growth).

- 2026-10-23 — **Amendment A5b: one final LayerNorm for (b), filed after the A5 gate data; the gate rule is unchanged** (the user's ruling, option 1, relayed by the expert; written by SCIENTIST; filed before any (b) selection or e100 training).
  - **The A5 gate data (S4), seeds 1–3 to step ~300.**
    - Criteria (a) (finite loss to step 300) and (b) (the loss bound never trips) pass on all three seeds.
    - **Criterion (c) fails on seeds 2 and 3:** the maximum per-layer activation L2 is 12.6× and 10.9× the as-built v7 gnn's at the same seed and step, always at the last layer.
    - **What the failure is:** the activations are flat at 15–31 over the steps, not growing. With no final norm, the residual stream carries a scale offset; it is not a divergence.
    - Under A5 a failed gate goes back to the user. It did, and the user chose this change.
  - **The change.** (b)'s gnn adds one final affine LayerNorm after the last pre-norm residual layer. It has no running statistics, and its affine weights are in the state dict, so a strict load fails across.
    - It is recorded in the contract sidecar as `mp_conv_prenorm_final_norm` (name as built by S4). The loaders and serving read it from the contract, never from a default.
    - It is valid only with `mp_conv_prenorm_residual`, and fails loud otherwise.
    - The default is off. With it off, the A5 pre-norm code is byte-unchanged.
    - **Disclosed with every (b) row:** the scorer's input is now layer-normalised, which the as-built and static gnns' are not. This replaces A5's "no final norm" disclosure.
  - **The gate is rerun unchanged:** seeds 1–3 to step ~300, the same criteria (a)–(c), with (c) at ≤ 10× the as-built v7 gnn at the same seed and step.
    - **(c), current definition:** activation L2 ≤ 10× the as-built v7 gnn at the same seed and step; if that fails, per seed and layer the activation at the last logged step ≤ 2× its value at step 50; plus (a) and (b).
    - **(c) is read on the same tensors as in the A5 gate,** the per-layer outputs up to and including the last residual layer, before the new final norm.
    - The final norm's output is reported beside them but not used for (c). Measured after a LayerNorm, (c) would pass by construction.
    - Then come equivalence, determinism and the full suite at the fix pin, as in A5.
    - A failed rerun goes back to the user again, with no retry on other seeds.
  - **Disclosed:** A5b was filed after the A5 gate data was seen, and it changes the architecture, not the gate. The rule, its threshold and its tensors are as filed in A5. (b)'s w is measured on this code.

- 2026-10-23 — **Pre-data amendment A6: after the selection, the winner trains 40 epochs, not 100; the 100-epoch runs become a disclosed sensitivity** (the user's decision, relayed by the expert; written by SCIENTIST; filed before the selection read). Amends A2 and A3.
  - **Recipe (replaces A2's 100 epochs).** The winner's arms, as listed per outcome in A3 (gnn, control twin, MLP, wide MLP), train at seeds 1–3:
    - 40 epochs, min-epochs 40, no early stop;
    - the checkpoint is the best validation epoch within the 40;
    - everything else as A2: corpus, labels, contract, venue, its own measured w.
    - Seed 1 is retrained from scratch at 40 epochs. No 20- or 100-epoch run is continued or reused.
  - **These runs serve** the registered development grid (30176–30181) and the confirmation.
  - **Why: the time budget.** At about 23:00 the user asked for "proper results and a full paper by end of Monday". A (b) win trained at 100 epochs on GPU (about 19 min per epoch, so about 32 h per run) could not reach confirmation by Monday, so the expert proposed the cap. The cap is filed before the selection read, with no live cell, no e100 curve and no static or (b) metric seen by the person who proposed it.
  - **A separate fact, disclosed:** before this amendment was drafted, SCIENTIST, who wrote it and runs the selection, had seen S7's as-built 20-epoch validation metrics (seal disclosure, 20:12Z). The expert, who proposed the cap, had not seen them.
  - **The 100-epoch runs (A3) become a sensitivity.** They cover every variant at seed 1, plus the v7 MLP and wide MLP at seeds 2–3.
    - **Cancelled (the user, 2026-10-23):** all 12 A3 100-epoch runs (as built: gnn, twin, MLP s1–s3, wide s1–s3; static: gnn, twin, MLP, wide) are cancelled, so no 100-epoch sensitivity exists. Across 80 uncut-program runs, none gained ≥ 1 % val regret after epoch 40 (median 0.0 %, as reported by the user).
    - **Caveat (the user):** the link features may keep the GNN learning past 40 epochs. When the winner's 40-epoch runs finish, each seed's best-val epoch is reported; any seed with a best at epoch ≥ 35 is flagged to the expert for the user's decision (extend or not). This is a training-curve flag only, never a live result.
    - They stay sealed until the selection is filed. After that they are read as a disclosed **100-epoch sensitivity**, labelled so in every row.
    - They are never pooled with the 40-epoch runs, and never used for the selection, a label or a confirmation cell.
    - The winner's 100-epoch seeds 2–3 are not launched, unless the user asks.
    - A3's "the winner's 100-epoch seed-1 runs become A2's seed 1" no longer applies.
  - **Disclosed:** the selection was made at 20 epochs and the winner is carried at 40, so the ordering at 20 may not hold at 40. The choice is not revisited.
  - **Compute.** 4 arms × 3 seeds = 12 jobs.
    - CPU: 40 × about 9 min, so about 6 h per job, and about 6 h wall in parallel.
    - If (b) wins: its gnn and GPU twin at the measured GPU rate on the pre-norm code (19 min per epoch was the pre-A5 rate, so about 12.7 h at 40 epochs).
    - Count the 44-job cap before each submit.

- 2026-10-23 — **Seal disclosure: the selection runner saw as-built 20-epoch validation metrics** (SCIENTIST; agreed with the expert).
  - **What happened.** At 20:12Z SCIENTIST, who runs `llf_select.py`, received S7's as-built CURVES flag (TRUE) with metric values attached, not TRUE/FALSE alone. The values were the gnn's and twin's 20-epoch seed-1 val/ce, task_acc_choice and plan_exact (gnn job 902811, run `m0dh6bkt`; twin job 902812, run `kety0rcz`).
  - **What was not seen:** no live cell, no 100-epoch run, and no static or (b) metric.
  - **Why it cannot change the choice:** the selection score is live latency from the fixed script (blob `54367447`), with no discretion left in it. The values seen are an offline proxy for one variant.
  - **Fix.** S7 now sends only TRUE/FALSE, run ids and checkpoint md5s for the static and (b) flags. The A4 wording on who reads the 20-epoch curves is corrected above.

- 2026-10-23 — **Pre-data amendment A5: (b)'s gnn gets pre-norm residual message passing** (the user's decision, relayed by the expert; written by SCIENTIST; filed before any (b) selection or e100 training, as the expert reports).
  - **Evidence: S4's robustness table** (diagnostic runs, not selection runs; they read only (b)'s own training metrics).
    - No init or scaling fix is robust. Zero-init diverges on seed 3, rescaling on seeds 2 and 3, and both together on seed 3.
    - All of them show a depth-compounding activation signature (activation L2 up to 1.2e6).
    - Zero-init does use E: with E zeroed, val CE per task is +2.0 % at seed 1.
    - **History:** the unnormalised conv was never a problem while every conv input was fixed per batch. (b) is the first conv whose inputs change within a batch, by being re-encoded per committed set.
  - **The change.** (b)'s gnn only: h_{l+1} = h_l + Conv_l(LayerNorm_l(h_l)). Random init, with the same recipe, corpus, labels, venue (GPU) and seeds as filed.
    - **Design facts (S4).**
      - "Stateless" means no running statistics, not no parameters. The LayerNorm has an elementwise affine (`bip_norms.{l}.weight` / `.bias`): one LayerNorm per layer, shared over the task and platform rows.
      - There is no linear skip: every `BipartiteEdgeConv` maps emb → emb, so the residual is the identity.
      - There is no final norm after layer 3. So the scorer's input scale differs from the as-built gnn's, and this is disclosed with every (b) row. *Superseded by A5b (2026-10-23): one final affine LayerNorm is added after layer 3.*
    - **The flag is weight-visible and in the contract.**
      - The affine parameters change the state dict, so a strict load fails across a pre-norm and a plain checkpoint.
      - The flag is also in the sidecar (`mp_conv_prenorm_residual`). The loaders and serving read it from the contract, never from a default (the `mp_residual` lesson).
      - It fails loud combined with `mp_bipartite_hetero`, `plan_raw` or `mp_residual`.
    - The default is off, and everything else is byte-unchanged.
  - **The controls.**
    - (b)'s control twin has no conv, so it is unchanged, and the (b) gnn-vs-twin contrast still isolates message passing.
    - The as-built and static variants keep their architecture. So (b) differs from them by the re-encode **and** by this block.
    - **Disclosed with every (b) row:** if (b) is selected, its message-passing result is attributed to "per-placement re-encode with pre-norm residual MP". Neither part alone is credited, and no cross-variant contrast attributes a difference to re-encoding.
  - **The gate, in order, before any (b) training (selection or e100). Each result is filed:**
    1. **Step-debug stability on seeds 1–3 to step ~300.** The pass criterion is fixed now:
       - every seed reaches step 300 with a finite loss;
       - the per-graph loss bound (defect ruling iii) never trips;
       - the maximum per-layer activation L2 stays within 10× that of the as-built v7 gnn at the same seed and step.
       Any seed failing any of these fails the gate; this goes back to the user, and is not retried with another seed.
    2. **Equivalence:** with the flag off, the (b) gnn's forward pass and gradients are byte-identical to the current conv's on fixed batches. Every existing arm is unchanged.
    3. **Determinism:** `tests/test_trainer_determinism.py` (CPU), plus the A1 one-repeat GPU weight-difference report for (b).
    4. **Full suite green** at the fix pin.
  - **Unchanged:** the selection rule and script (blob `54367447`), the A4 wait, the A3 seal, (b)'s eligibility tests and its own measured w. The w is re-measured on the pre-norm code, since the block costs time.

- 2026-10-23 — **A1 selection script filed before the first selection cell: `scripts_cosim/llf_select.py`, blob `54367447467e1fb8a4d97a2f8f5bfea0d641c153` (file md5 `1bc243c9cd2c1bc5a370235ede4aefd8`)** (SCIENTIST, at the expert's request; tests `tests/test_llf_select.py`). The selection of A1/A4 runs this blob once, by SCIENTIST, after (b)'s last 20-epoch seed-1 selection cell.
  - **What it reads.** Per variant, its gnn and control twin at seed 1: as built `ra_gnn_eng_combfl_link` / `ra_twin_eng_combfl_link`; (a) `ra_gnn_eng_combfl_linkstatic` / `ra_twin_eng_combfl_linkstatic`; (b) `ra_gnn_eng_combfl_linkconv` / `ra_twin_eng_combfl_link_gpu` (the test asserts these are the gate's registered kinds). Cells `<kind>_nosplit_k100_s1` on 30176–30181, g0+g1, both rungs; the cell value is whole-run `averageElapsedTime`, the uncut reader's statistic of record.
  - **What it refuses.** It refuses to compute unless all 144 cells exist with a finite latency. It also refuses if any cell is not option D (fair share serial, TCP on, default cellular), if a gnn and its twin served different task counts, or if its `--out` exists.
  - **Eligibility.** Eligibility comes from a flags file: parity, collapse and curves per variant, each literally `true`. Missing or non-boolean is not a pass.
  - **Scoring, as A1.** The runaway rule and the pair drop as `d_read`; per topology the median over windows; the score is the median over the (topology, rung) values without runaways; lowest wins.
  - **Ties (interpretation, filed now).** The 0.5 pp ties are measured from the best score. The all-cells stage compares only the variants tied on the primary score, with the same 0.5 pp tie, and its ties go to as built < (a) < (b).
  - **Hold.** If no variant is eligible, or an eligible one has no score, it prints HOLD and exits 2.
  - **Output.** It prints every variant's score, n, all-cells score, runaway count and flags, then the choice and its reason. The JSON holds the per-topology values.

- 2026-10-23 — **Pre-data amendment A4: the selection waits for (b); (b) is fixed, not marked ineligible** (the user's decision, relayed by the expert, overriding ruling (ii) of the defect entry; written by SCIENTIST; filed before any selection-run score is read, as the expert reports).
  1. **The selection read runs only when all three variants' 20-epoch seed-1 selection cells are complete:** as built, (a) and (b). The rule, its eligibility tests and its script are unchanged.
  2. **The wait is not outcome-dependent.** It was decided now, when no one had read any CPU variant's selection score.
     - Until the selection read runs, the as-built and (a) selection cells stay unread. Their 20-epoch curves are read only by the curves flag's owner (S7, who sees no live cell), who passes only TRUE/FALSE with run ids to the selection runner; everyone else leaves them unread. *(Wording fixed 2026-10-23: it first said the curves stay unread, which contradicted the curves flag.)* The only checks are completion and training health (crash, NaN/inf, job state), reported pass/fail with no metric value.
     - The selection script is run once, by one person, after the last cell completes.
  3. **The sealed 100-epoch runs of every variant (A3) continue meanwhile, sealed as filed.**
  4. **Pin.** (b) and its GPU twin train together at the fix pin, which is later than `10f3a9c4`, carries the fix merge, and has the full suite green. Disclosed on every (b) row.
     - The per-graph loss bound (defect ruling iii) applies, and the defect entry's byte-identity test decides whether any started arm keeps its pin.
     - A fixed (b) still faces the rule's eligibility tests (parity, collapse check, curves above the chance floor). Failing one is ordinary ineligibility, not a deadline.
  5. **If (b) proves unfixable,** the user decides then what happens, and the record names that decision and its date. The decision is made, and filed, before the selection read runs and without any CPU-variant score.
  - **Cost.** The selection read, and with it the winner's seeds 2–3 and the confirmation, slip by (b)'s fix time plus about 6–7 h of GPU training: 20 epochs × about 19 min per epoch on an L40S is about 6.3 h. Then come (b)'s selection cells under D: about 1 h, somewhat slower than the other arms because of (b)'s 2.1× decision charge. *Corrected 2026-10-23: the 11,675 s heavy median first quoted here is the reactive admission screen's. The learned-arm grids under D ran 15–26 min per job (sacct: d_fsl 901504–901507, d_fsl3 901873, 901874, 901970, 901971).*

- 2026-10-23 — **Variant (b) training defect, and three rulings** (the expert's rulings; filed by SCIENTIST before any selection-run data exists, as the expert reports).
  - **The defect.** S4's 1-epoch smoke of (b) on the real corpus has a train CE of about 3e7, against about 34 for its GPU v7 twin on the same corpus. Some per-graph losses are huge but finite. (b) does not train until it is fixed.
  - **(i) Pin.** (b) and its GPU twin train together at a pin later than `10f3a9c4` that carries the fix merge, with the full suite green there. The twin is retrained at that pin too, so the pair keeps one pin and one venue (A1). The pin and the fix are disclosed on every (b) row.
  - **(ii) Eligibility deadline.** *Superseded by A4 (2026-10-23, the user's decision) before any selection score was read: the selection waits for (b).* The selection read runs as soon as the 20-epoch selection cells of every CPU variant (as built, (a)) are complete.
    - It never waits for (b). No one reads any CPU variant's selection score before it runs.
    - (b) takes part only if its 20-epoch runs passed the eligibility tests and its selection cells are complete by then. Otherwise it is marked **'not eligible: training defect'**, and the rule picks among the eligible variants.
    - Any later (b) run is its own amendment, and is exploratory against the selected variant. It never re-opens the selection.
  - **(iii) Per-graph loss bound (fail loud) in every arm's loss path.**
    - The bound is a fixed constant in code, set before any selection run and recorded with the pin. It is never tuned on a selection run.
    - A trip stops the run. It is a crash, fixed and rerun, never a skipped graph and never an excluded data point.
    - "Changes no as-built arm" means: with and without the bound, a 1-epoch run of each arm already started (as built, (a), and the early 100-epoch runs of A3) gives byte-identical weights. Each such arm keeps its pin.
    - An arm whose weights differ, or whose run trips the bound, is retrained at the pin with the bound, and this is disclosed.

- 2026-10-24 — **Decision charge w of variant (b) re-measured on the A5/A5b architecture (S4; code at `b0abbecc`, job 903323).** The pre-norm residual conv blocks and the final LayerNorm cost time, so (b)'s w is measured again on its serving code, as A5 requires; this entry supersedes the 2026-10-23 (b) w below for every (b) cell.
  - **Series.** Pin `b0abbecc` (reference-physics, the (b) training pin), os-cpu-slurm-17 (the other 16/17 allowed node; not the filed node B), AMD EPYC 9335, SMT off, exclusive, PAR 1, 1 torch thread, 19461 g0 both rungs, k = 0, seed 1, option D's exchange physics, the recipe of `cot_link_smt.sbatch` / `cot_link_steps.sh`. Weights: the filed v6 FL gnn's with the new input columns zeroed and the LayerNorms at identity init (`v7_untrained_arm_checkpoints.py --kinds ra_gnn_eng_combfl_linkconv --prenorm`, job 903319, `link_calib/in_link5`). One run, no void: H3 drift start to end 0.22 % moderate / 0.08 % heavy (stop at 2 %, not fired); against the filed node-B H3 at most 0.57 % / 0.12 % (void at 2 %, not void).
  - **Frozen w (ms per plan).** moderate 7.804, heavy 8.199; by bucket ≤10 / 11–50 / >50 tasks: moderate 8.050 / 6.554 / 9.366, heavy 8.288 / 6.904 / 10.745. The link capture is 1.40 % (moderate) and 0.97 % (heavy) of the decision seconds. The table is `link_calib/freeze_w_link_conv_pn_17.json` md5 `e733a8c637f287162945fd395a08bbb4`, written by `scripts_cosim/v7_freeze_w.py --kinds conv`; `scripts_cosim/decision_cost_w.json` carries the values for `ra_gnn_eng_combfl_linkconv_nosplit` and its `_e100_` copy (moderate 7.451649 -> 7.803501, heavy 8.26126 -> 8.198949).
  - **Disclosures.** The same as the 2026-10-23 (b) entry: a different node than the filed w's (checked by H3), zero-extended v6 weights, two cells on one topology. The e40 kinds do not exist yet; they are added only if (b) wins, with this w copied (a serving cost).
- 2026-10-23 — **Decision charge w of variant (b), the re-encode-per-placement gnn, measured on its own serving code (S4; the w-only pin is `s4/v7-w-conv` 5d662369, a child of `e668f9d3`).** A calibration, not a result.
  - **Series.** Pin `e668f9d3` (reference-physics, with the gate kinds of `s4/link-kinds`), **os-cpu-slurm-17** (a different node than the filed w's, node B = os-cpu-slurm-22, and than the v7 kinds' os-cpu-slurm-16; the expert allowed nodes 16 and 17 for the w calibrations on 2026-10-23), AMD EPYC 9335 32-Core, SMT off, exclusive, PAR 1, 1 torch thread, 19461 g0, both rungs, k = 0, seed 1, 2 cells, option D's exchange physics. The same driver as the v7 series (`cot_link_steps.sh`): H3pre, the kind, H3end. Weights: the filed v6 FL gnn's with the new input columns zeroed, including the conv message weights of the 3 layers (`v7_untrained_arm_checkpoints.py --kinds ra_gnn_eng_combfl_linkconv`, `link_calib/in_link4`, .pt md5 `47afe62b…`), so the cell makes the filed arm's decisions while paying the re-encode.
  - **Two runs, one void.** Run 1 (job 902235) drifted 2.99 % at the moderate rung between H3pre and H3end, over the 2 % stop, so it is void and not used (its table, `freeze_w_link_conv_17.json`, md5 `2e6ac127…`, stays on disk). Run 2 (job 902243, same node, same pin) is the series used: H3 drift start to end **1.54 % moderate / 0.04 % heavy**, and against the filed node-B H3 at most 0.88 % / 0.13 %, under the 2 % stop and void rules. Table `link_calib/freeze_w_link_conv_17_r2.json` md5 `bb7e685e265cc85c82dd5f182bef314a`, written by `v7_freeze_w.py --kinds conv` (this commit's script).
  - **Frozen w (ms per plan).** moderate: w 7.452 ms/plan (2.11x the v7 gnn's 3.532); by bucket ≤10 / 11–50 / >50 tasks 7.678 / 6.168 / 9.245 (1.64x / 4.13x / 5.45x the v7 gnn's); heavy: w 8.261 ms/plan (2.14x the v7 gnn's 3.853); by bucket ≤10 / 11–50 / >50 tasks 8.130 / 7.018 / 11.948 (1.62x / 4.11x / 5.61x the v7 gnn's).
  - **Reading.** The re-encode costs about 2.1x the as-built v7 gnn per plan, less than the microbenchmark's 2-3x because the decode's other costs are shared. The link capture is 1.4 % / 1.8 % of the decision seconds, so it is small beside the re-encode itself.
  - **The same-architecture kinds.** The 100-epoch kinds and the GPU-trained twin are charged the measured w of the arm they are, copied values with a source line each (`link_e100` ← `link`, `link_gpu` and `_e100` twins ← the as-built twin, `linkconv_e100` ← `linkconv`): the decision charge is a serving cost, unaffected by epochs or by the venue a checkpoint was trained on.
  - **Disclosures.** A different node than the filed w's (checked by the H3 control, run 2); zero-extended v6 weights, not trained ones, so a trained model's decode length may differ (the by-bucket w's carry that); 2 cells, one topology, seed 1, as the filed series.
- 2026-10-23 — **D admission screen: a cell killed by a wall limit is an infrastructure failure, rerun with a longer wall** (the expert's ruling; filed by SCIENTIST after the coordinator read `queue_share` on the 68 completed cells (0 fails) at 15:18 and passed the count to S5; no killed cell's output and no other field was read. The ruling is not conditioned on that count: every killed cell is rerun whatever it produced). The screen is reactive s0, 96 cells on 30285–30296, pin `a9733951`. Some cells near the 14,400 s per-cell LIMIT: heavy cells take a median of 11,675 s under D.
  - **The ruling.**
    - A cell killed by the per-cell LIMIT or the job wall is an infrastructure failure. It is rerun alone at the same pin, inputs and D flags, with LIMIT 28,800 s and a job wall of at least 9 h.
    - The rerun counts only if it completes. Unknown is not a pass.
    - A cell that fails twice sends its topology to the next free id, by the filed rule (admission item 4). The topology is counted as an admission failure ("did not complete"), not dropped silently.
    - The rerun changes only the wall limit, never the physics, pin or inputs, and it is not conditioned on any outcome.
  - **Clarifications (SCIENTIST).**
    - Every killed cell is rerun, whatever it had produced. No partial output of a killed cell is read: no `queue_share` from a partial summary, and no log line.
    - Wall time is not an admission criterion. A rerun that completes is read by the admission rule like any other cell, on `queue_share` only.
    - **The admission report lists:** every kill (cell, which limit, elapsed s, job id), every rerun (job id, elapsed s, completed or not), and each replaced topology with its replacement id.

- 2026-10-23 — **Pre-data amendment A3: the 100-epoch seed-1 runs of every variant start early, sealed until the selection is filed** (the user's decision, relayed by the expert; written by SCIENTIST; filed before any v7 training or cell exists). Amends A2's "seed 1 retrained from scratch after selection".
  - **What starts early** (each fresh from scratch, never continued from a 20-epoch run). These run in parallel with A1's 20-epoch selection runs, on the A2 recipe: 100 epochs, min-epochs 100, no early stop, best validation checkpoint within 100. Each starts as soon as its code and the corpus exist.
    - **Seed 1, every variant:**
      - as built: v7 gnn, v7 twin, v7 MLP, v7 wide MLP;
      - (a) static (the `v8` contract): `gnn_static`, `twin_static`, `mlp_static`;
      - (b) re-encode: its gnn and its GPU-trained v7 twin, both on GPU, with the A1 nondeterminism disclosure.
    - **Seeds 2–3 of the v7 MLP and v7 wide MLP.** They are A2 arms under every outcome (below).
  - **The seal (until the selection read of A1 is filed, from the 20-epoch live cells, with the rule unchanged):**
    - no 100-epoch checkpoint is served in any live cell;
    - no one who runs or applies the selection rule reads their W&B curves, validation metrics or selected epoch;
    - the only checks allowed are training health: crash, NaN or inf loss, and job state. They are reported as pass or fail, with no metric value.
    - The selection rule's eligibility test ("its gnn run is not untrained") reads the 20-epoch runs only.
  - **After the selection is filed.**
    - The winner's 100-epoch seed-1 runs become A2's seed 1, and are not retrained. Its seeds 2–3 launch immediately.
    - The seed-1 runs stand only if they were trained on the same code pin, config, corpus and venue that seeds 2–3 then use. If any of these changes after a seed-1 launch, that arm's seed 1 is retrained with seeds 2–3.
    - The other variants' 100-epoch runs are reported as **exploratory**, labelled so in every row. They are never pooled with A2's runs, never used for the selection or a label, and never served in a confirmation cell.
  - **A2's arms per outcome** (resolving A2's "all in the same contract"):
    - **As built or (b) wins:** the gnn, its control twin, the v7 MLP and the v7 wide MLP.
    - **(a) wins:** `gnn_static`, `twin_static` and `mlp_static`, plus the v7 MLP and v7 wide MLP as pointwise references without the static block. `mlp_static` seeds 2–3 then launch with the rest.
    - **Gap closed (the expert, 2026-10-23, before any training):** S6 adds `wide_static`, a parameter-matched MLP with the static block. It is in the early set at seed 1 (20 and 100 epochs) and in the w series. Under (a), the wide-MLP control is `wide_static`, its seeds 2–3 launch with the rest, and the v7 wide MLP stays a reference.
  - **Compute.** The early set is 14 training jobs on top of the 20-epoch selection runs:
    - 9 at seed 1 (2 of them on GPU), plus `wide_static` at seed 1 (S6), so 14 in all;
    - 4 for the v7 MLP and wide MLP at seeds 2–3.
    - Each CPU job takes about 15 h (A2's rate). Count the 44-job cap before each submit.

- 2026-10-23 — **Pre-data amendment A2: after selection, the chosen variant trains at 100 epochs, seeds 1–3 from scratch** (the user's decision, relayed by the expert; written by SCIENTIST; filed before any v7 training or cell exists).
  - **Arms.** The selected variant's gnn, its control twin (for (b), the GPU-trained twin of A1), the v7 MLP and the v7 wide MLP, all in the same contract.
  - **Recipe.**
    - 100 epochs: min-epochs 100, no early stop.
    - The checkpoint is the best validation epoch within the 100, as for the filed arms.
    - 100 is also the cap: `run_experiment.py` clamps epochs to 100.
    - Seeds 1–3, each retrained from scratch. Seed 1 is not continued from its 20-epoch run.
    - Corpus, labels, contract and venue as at selection.
  - **Used for both** the registered development grid (30176–30181) and the confirmation. They replace the registration's 20-epoch development recipe.
  - **The seed-1 20-epoch selection read stays as filed.** It is never pooled, averaged or tabulated with the 100-epoch runs, and it is quoted only as the selection read.
  - **Disclosed.** The variant is chosen at 20 epochs and carried at 100, so the ordering at 20 epochs may not hold at 100. The 100-epoch development rows are reported for the chosen variant only, and the choice is not revisited.
  - **Compute cost.** 4 arms × 3 seeds = 12 training jobs.
    - CPU arms: about 9 min per epoch, so about 15 h per job and about 15 h wall with the jobs in parallel (about 180 job-hours).
    - Variant (b): its gnn and GPU twin run at their measured GPU rate, filed with the jobs.
    - Count the 44-job cap before submitting.
    - The decision charge w is a serving cost, unaffected by epochs; each arm keeps its own measured w.

- 2026-10-23 — **Pre-data amendment A1: two GNN variants put the link columns in the message passing, and a fixed rule picks one for confirmation** (the user's decision, relayed by the expert; written by SCIENTIST; filed before any v7 training or cell exists).
  - **Why.** S4 confirmed that in `partial_state_v7` as built, the edge columns E1–E3 enter only the EdgeScorer, which the MP-OFF twin shares. The bipartite conv gets only the at-capture platform columns P1–P3. So the as-built gnn vs twin contrast tests message passing over P1–P3 alone: the live, prefix-refreshed link information reaches both arms equally, and through no message.
  - **The development-1 arms** (seed 1, 20 epochs, the v7 corpus and labels as filed, W&B on every run):

    | Variant | gnn | its control twin | MLPs |
    |---|---|---|---|
    | as built (S4) | v7 gnn | v7 twin | v7 MLP, v7 wide MLP |
    | (a) static link block (S6) | `gnn_static`: link columns against the partners' candidate nodes, in the conv once per batch | `twin_static`: the same block in its scorer rows, MP off | `mlp_static`: the same block in its rows |
    | (b) re-encode per placement (S4) | E1–E3 in the conv, re-encoded for each committed set; trained on GPU | a GPU-trained v7 twin, the same venue and recipe as (b) (its scorer sees the same E1–E3 per placement) | the as-built v7 MLP and wide MLP |

    Plus the references as registered (v6 fslabel arms, the bar).
  - **Conditions on the variants.**
    - (a) Parity: `twin_static` and `mlp_static` carry the static block byte-identical to `gnn_static`'s, on captured states. Columns built at capture equal those built at serve (extending `tests/test_link_features.py`).
    - (b) Parity: the re-encoded E1–E3 for a committed set equal the scorer's E1–E3 for that set, so the twin's information is the same. Serving is on CPU, and (b) is charged its own CPU w, measured on its serving code like the other v7 arms.
    - **GPU nondeterminism (b), disclosed.** (b) is the only arm trained on GPU. Its weights are not bit-reproducible across runs (`tests/test_trainer_determinism.py` covers CPU), and the contrast with a CPU-trained twin also changes the training venue. Disclosed with every (b) row.
      - SCIENTIST's condition: if (b) is selected, its twin for seeds 2–3 and confirmation is retrained on the same GPU venue and recipe, so venue is not confounded with message passing.
      - Extension (the expert, 2026-10-23, before any v7 training): the confound also reaches the seed-1 score the selection rule reads. So (b)'s control twin at seed 1 is also a GPU-trained v7 twin, the same venue and recipe (S4), not the CPU-trained as-built twin.
      - Each (b) run records the two-run weight difference at seed 1 (one repeat run) for the record.
    - Every arm is charged its own measured w (as registered); none is aliased.
  - **What runs first.** The early collapse check (as registered) runs for `gnn_static`, `twin_static` and (b) as well as the as-built gnn and twin, before the development-1 grid. The development-1 grid is option D on 30176–30181, g0+g1, both rungs, seed 1, every arm above.
  - **Curves.** Each run is read with `scripts_cosim/read_training_curves.py` against its chance floor, and the selected epoch is named. A run whose live curves do not leave the chance floor is reported as untrained.
  - **Selection rule (fixed now, before any v7 training).** Exactly one of {as built, (a), (b)} is carried to seeds 2–3 and confirmation.
    1. **Eligible:** the variant passed its parity test and the collapse check, and its gnn run is not untrained.
       - **Collapse (the user, 2026-10-23, before any selection score):** a collapse-check fail diagnosed as an environment runaway counts as a pass, for every variant alike, and all three variants (as built, static, (b)) enter the selection. The collapse results and S6's herding/environment labels are reported beside the selection and exclude no variant.
    2. **Score:** the pooled median, over the 12 (topology, rung) values without runaways (statistic and runaway rule as `d_read`, seed 1), of the variant's gnn vs its own control twin as listed above. Lower is better.
    3. **Ties** within 0.5 pp go to the all-cells pooled median, then to the simpler variant (as built < (a) < (b)).
    4. **If no variant is eligible,** hold, and report to the user.
    - The score is read once, by one script whose blob is filed before the first development-1 cell. Nothing else (MLP contrasts, bar contrasts, curves) enters the choice, and the choice is not revisited after seeds 2–3.
  - **After the choice.** Seeds 2–3 run for the selected variant's gnn, control twin and MLPs only. They form the registered development grid with seed 1, still direction only. Confirmation then runs the selected variant's arms, as registered.
  - **What this stage cannot give.** The development-1 read cannot give a label of any kind:
    - it is one seed;
    - n = 6 topologies, with an exact two-sided floor of p .03125;
    - all 6 topologies have been read before under D;
    - it chooses among three variants, so the chosen score is biased downward (winner's curse).
    The registered confirmation labels apply unchanged to the selected variant's gnn vs its control twin. Its secondary rows name the variant.
  - **Unchanged:** the corpus, labels, replay identity, confirmation topologies and statistic, the label order, and the report-both-ways rule.

- 2026-10-23 — **Decision charge w of the four v7 arms, measured per arm (S4; the w-only pin is `s4/v7-w` f283b8e2, a child of `23d5c99b`).** A calibration, not a result: no outcome is read. Each v7 arm is charged its own w, never an alias of a v6 key.
  - **Series.** Job 902034 at `23d5c99b` (reference-physics merge of `s4/v7-arms`), **os-cpu-slurm-16, a different node than the filed w's** (those are node B = os-cpu-slurm-22; the node was held by another user's job for days, and the expert ruled node 16 on 2026-10-23, node 17 the fallback). AMD EPYC 9335 32-Core, SMT off, exclusive, PAR 1, 1 torch thread, 19461 g0, both rungs, k = 0, seed 1, 2 cells per kind, the exchange physics of option D (`R1A_EXCHANGE_TCP=1`, `R1A_EXCHANGE_FAIRSHARE=1`). Recipe of series 884584 through `cot_link_smt.sbatch` / `cot_link_steps.sh` (`cotenancy/`): H3pre (`cd_ib0` at `114cc0e7`), the four kinds, H3end. 32 min 22 s, 0 failed cells.
  - **Weights.** Each kind is its filed v6 FL counterpart's weights with the new input columns zeroed (`scripts_cosim/v7_untrained_arm_checkpoints.py`, `link_calib/in_link2/models`, .pt md5s gnn `50a0ead3…`, twin `1e6c3a8c…`, MLP `4f9a6482…`, wide MLP `5b010303…`). A decision-time w is seconds of scoring per plan, which depends on the architecture, the widths and the code path, not on fitted weights. The first try with random weights ran the same cell 5x+ slower (job 901738 killed by the watchdog at 18 min with a projected 10,900 s; job 901772 was still running at 19 min and was cancelled; the zero-extended gnn cell took 4 min 13 s, job 901789), which is why the zero extension was used: it gives the filed arm's decisions until trained.
  - **The clock.** The decision seconds of a v7 call INCLUDE the live link capture (`flows_live`, `node_table`, the per-platform lookups), metered into the decision span for the v7 kinds only (expert ruling, `s4/v7-arms` 2517393b); v6 and the bar are unchanged. The capture seconds are a separate field.
  - **H3 control on this node** (`cd_ib0`, no exchange switches), pooled w ms per plan: start 7.914 / 7.440, end 7.896 / 7.388 (moderate / heavy; the unit is 10^-3 ms). Drift start to end **0.23 % / 0.69 %**, under the 2 % stop. Against the filed node-B H3 (series 884584): start 0.35 % / 0.78 %, end 1.11 % / 0.35 %, under the 2 % void rule, so node 16 reads the same as node B on the control and the series stands.
  - **Frozen w** (ms per plan; `w_ms_per_plan` moderate / heavy; `w_ms_per_plan_by_bucket` ≤ 10 / 11–50 / > 50 tasks), with the link capture's share of the timed decision seconds:

    | Kind | w moderate / heavy | by bucket, moderate | by bucket, heavy | capture share, mod. / heavy | capture per call, mean |
    |---|---|---|---|---|---|
    | v7 gnn | 3.532 / 3.853 | 4.672 / 1.493 / 1.697 | 5.018 / 1.710 / 2.131 | 3.8 % / 3.8 % | 0.70 / 0.77 ms (max 0.66 / 0.64 s) |
    | v7 twin | 2.918 / 3.150 | 3.822 / 1.298 / 1.461 | 4.016 / 1.542 / 1.927 | 4.2 % / 3.5 % | 0.63 / 0.59 ms (max 0.58 / 0.67 s) |
    | v7 MLP | 2.816 / 3.175 | 3.700 / 1.201 / 1.451 | 4.015 / 1.559 / 2.110 | 3.6 % / 4.7 % | 0.52 / 0.79 ms (max 0.59 / 0.62 s) |
    | v7 wide MLP | 4.133 / 4.837 | 4.081 / 3.042 / 7.038 | 4.375 / 3.919 / 9.681 | 3.0 % / 1.9 % | 0.65 / 0.48 ms (max 0.62 / 0.58 s) |

    For scale, the filed FL gnn is 3.020 / 3.107 and the FL MLP 2.379 / 2.429 (series 884584, node B). The v7 gnn's w is 17 % / 24 % above it and the v7 MLP's 18 % / 31 %: the capture is 2–5 % of it, the rest is the wider inputs and the routes and edge columns in the decode. The v7 twin is cheaper than the v7 gnn (no message passing) and the wide MLP costs the most, as the filed pair does.
  - **Disclosures.** (1) A different node than the filed w's, checked by H3 (above). (2) Zero-extended v6 weights, not trained v7 weights; the live cost does not depend on the values, but a trained model's decode may differ in length (the by-bucket w's carry that). (3) 2 cells per kind, one topology (19461), seed 1, as the filed series. (4) The table is `link_calib/freeze_w_link_16.json` md5 `7e27788375913b0a536ba6fdbfcba383`, written by `scripts_cosim/v7_freeze_w.py` (this commit), which also fails on a drift or a void H3.
- 2026-10-23 — **Capture amendment: extend to the remaining training topologies and recapture with `unplanned_detail` (S6; filed before any cell of either).** Two reasons, both found after the first capture (`collect_bar`, pin `a794431c`, 128 cells, 3,880 states, all drawn triggers captured):
  - **Corpus size.** 3,880 states against the 11,073 the v6 arms trained on would put a 3× corpus-size confound under every v7-vs-v6 and v7-vs-v6-MLP contrast. The v7 gnn vs v7 twin contrast shares a corpus and is unaffected.
  - **Replay gap (S7's finding).** 25.1 % of captured states carry a ghost the replay cannot plan the exchange of (rendezvous, or admitted with an unplaced partner), and the snapshot held no partner information for it. The flows block gains an additive key `unplanned_detail` (schema v1.1, same version; `validate()` tolerates its absence): per such ghost its node, stage, the full pre-exchange read, and all its peers with the node each runs on or is planned for (null when not placed). The `a794431c` corpus lacks it, so it is superseded for labelling by the recapture below; it stays on disk as captured.
  - **Extension topologies (24):** the Arm 1 training ids (`arm1_training_topologies.json`, md5 `5fcfb60e866e0bb7adf83d6c1d1dcb54`, 44 ids) minus the four pilot ids 16201–16204 and minus the 16 already captured: **16223, 16224, 16225, 16227, 16228, 16229, 16231–16239, 16241–16249.** Same arm (the bar `cd_ib0_inflightfull_k100` s0 under D), cap200 inputs (`flowcap/in_flowcap`), windows g0–g3 and rungs moderate and heavy: 24 × 8 = 192 cells.
  - **Trigger plan for them:** `dagger_corpus.py plan --total 7500 --seed 20261024` (a fresh seed; same stratification and rules as the first plan; groups over 256 excluded, no top-up). Drawn **7,245** (shortfall 255, reported): quotas ≤ 10 / 11–50 / 51–100 / 101–256 = 3,000 / 2,875 / 1,250 / 375; moderate 3,610, heavy 3,635; early share 0.2716. The plan keys are the 192 cells. `flowcap/plan_flowcap2.json` md5 `cb33985b24f8208615ab85eba0002e55`, sha256 `af81b795de492a046ad7079054ebe0a53f35e117931738feeb8ba33afb686b4b`. Union with the first plan: 3,880 + 7,245 = **11,125 drawn**, against 11,073 in the v6 corpus.
  - **The early share of the extension is 0.2716, below the 0.30 target and above the 0.25 floor.** The union is (0.3168 × 3,880 + 0.2716 × 7,245) / 11,125 = 0.287.
  - **Recapture:** all 128 first-plan cells (plan md5 `f387e648790c15d7a8fc7af3b60cea91`) and the smoke cell are captured again with `unplanned_detail`, so the whole corpus is one pin. Everything else is unchanged: arm, seed, inputs, plan, windows, rungs. The capture only reads, so the latencies must equal the `a794431c` cells' to the last digit; I check that cell by cell.
  - **Pin of the amended capture:** the reference-physics merge of `s6/unplanned-detail`; its sha is in every captured cell's `code.commit` and is filed with the first job.
- 2026-10-23 — **`partial_state_v7` spec and parity test, filed before any data (S4).** A feature contract: no capture, label or training is read from it. Code on `s4/v7-features` (head `0515ed05`, on reference-physics `12f96536`; full suite 3,287 passed, 115 skipped, 0 failed, `-n auto`).
  - **Contract.** `partial_state_v7` = v6 byte for byte plus 3 platform columns and 3 partial-state (edge) columns, so the partial-state width is 30 (v6: 27). It is in the peer, load-seconds, both-direction, four-type and in-flight families of `reduced_features.py`, and in the new `LINK_CONTRACTS`. A v6 checkpoint under a v7 run (and the reverse) is refused by `require_matching_partial_state_contract`. Needs `HEROSIM_EXCHANGE_FAIRSHARE=1`.
  - **One definition.** `src/placement/link_features.py` computes every column. Two front ends only build its input, the active transfers in `FairShare.flows` order as (remaining bytes, ((directed link id, MB/s), ...)): `flows_live(share, now)` from `fabric.fairshare.flows`, and `flows_stored(block)` from `fidelity["flows"]` (flow_snapshot schema v1, validated by `flow_snapshot.validate`). Only `status == "active"` transfers hold a link, so only they count; startup, latency_tail and pending ones are in neither front end. The directed route table comes from one function, `network_fabric.directed_hops_from_topology`, which `NetworkFabric.directed_hops` now calls (live) and the cache builder calls on the dataset's `link_topology` (stored).
  - **Remaining bytes.** The live front end never calls `FairShare._advance` (it would split the float subtraction of `rem` and change later bits of the run). It evaluates `rem - rate * (now - _last)`, the very expression `_advance` computes, which is what the capture stores ("`rem` after `_advance`"), so the two are bit-equal; a test advances the real object after reading and compares.
  - **Platform columns (appended after the v5 four-type and v6 in-flight columns), per platform, from the flows crossing its node's access links.** Hop ids are directed `a>b`. P1 = log1p(number of active flows with a hop id starting `"{node}>"`); P2 = log1p(number with a hop id ending `">{node}"`); P3 = log1p(max of the two sides, each the sum over its flows of `rem / (MB/s * 1024**2)`, the seconds to drain that link at full capacity). Not divided by the queue norm: counts of flows are not queue lengths.
  - **Edge columns (appended after the v5 push columns, indices 27-29), per (task, candidate node), over the task's committed partners** (in-batch peers already placed; a co-located partner moves nothing). The pull direction, partner node -> candidate node, is the one the platform charges. With `n_h` the flows on hop h and `bw_h` its MB/s: the shared price of one transfer of `b` bytes is `b / min_h(bw_h * 1024**2 / (n_h + 1)) + latency`; with no flow on any hop it is the solo price `b * pb + latency` of `_xchg` (the price columns 7-8 and 25-26 already use). E1 = log1p(sum of the shared prices). E2 = log(sum of shared prices / sum of solo prices), 0 on an empty fabric. E3 = log1p(largest `n_h` over every hop of every one of those routes, before adding this flow).
    - **E2 is S6's ratio on a log scale** (S6 proposed the ratio itself, which is 1 on an empty fabric); the log makes the quiet-fabric value 0 like the other two and keeps a 10x slowdown at 2.3. E1 uses the exchange tables' solo price `b * pb` rather than `_payload_transfer_time`; they are equal up to float rounding for the pipelined transfer model, and the contract needs one number both sides compute the same way.
  - **The edge columns refresh with the committed prefix** (the `_commit` fix of the fair-share-aware rule, `_pg_remote_edges`): before E1-E3 are priced, the flows the already-placed batch-mates will run are added to the active counts. For every placed task `u` and every placed peer `v` of `u` on another node, one flow `v -> u` over `routes[(node_v, node_u)]`; both directions of a pair are counted, as `_pg_remote_edges` does. This runs at every decode step and in every refine pass (a refine pass scores a task with every other task committed; the task itself is never in its own prefix), and is the same code for gnn, twin and MLP, because it sits in `partial_state_columns`. Peers outside the batch are invisible to the context by contract (the cache never saw one) and are not counted. **P1-P3 are the at-capture state and do not move with the prefix:** the platform block is encoded once per batch (the shared-encode assertion of `make_partial_state_score_fn`), and re-encoding per commit is the Arm 2 committed-inputs path, not used by these arms. So the prefix-refreshing link information the twin and the MLP see is E1-E3, the same as the gnn.
  - **Stored-record path.** `prepare_graphs_cache._flow_block_of_dataset` reads `warm_snapshot.json` -> `snapshot.fidelity.flows`, validates it, and refuses a state without it (every state captured before option B) with a FAIL LOUD; it never serves the link columns as zeros. `build_graph` writes `link_platform_extra` from `node_raw(flows_stored(...), node_name)`, `widen_graph` appends it after the in-flight columns, and `attach_dag_partial_state_block` writes `link_counts` and `link_routes` (candidate-node pairs) into `partial_state_ctx`. The live batch writes the same two keys in `attach_live_prefix_block` and the same three platform columns through `StateCaptureHelper` (`link_node` in the temporal state) and `build_pyg_inference_graph`. `PartialStateContext` refuses a v7 context without the two tables. The serving loader subtracts `LINK_PLATFORM_DIM` when it recovers the layout from the platform width.
  - **Parity test (`tests/test_link_features.py`, 16 tests).** The columns: P1-P3 and E1-E3 are checked against hand-computed values on a real `FairShare` (three servers, directional access links, three flows). The live front end leaves the fabric unchanged, and the stored front end of a block written by the capture's own `_advance` equals the live one bit for bit, node by node. The gate: one state through the trainer path (stored block, the dataset's routes and links, `build_partial_state_context_from_graph`) and the live path (running fabric, `fabric.link_topology`), each through a full `masked_topo` decode plus two refine passes, for the gnn, its MP-OFF twin (`GNN_DISABLE_MESSAGE_PASSING`) and the same-input MLP, CPU: the platform block, the node embeddings (the conv inputs), the edge block at every step and the logits are `torch.equal` live vs cache for each arm; every step's edge block equals the single definition recomputed from (task, prefix); and the gnn, twin and MLP edge blocks are byte-identical at every (task, prefix) they share. The new columns are inputs, not constants: a quiet fabric gives different edge columns for the same prefix.
  - **What this does not yet cover.** No real recaptured state has been run through `prepare_graphs_cache` (none exists until S6's capture), so the cache wiring is exercised through its loader and the shared functions, not a full cache build; the first recaptured dataset gets the full build and the live-vs-cache check (`wf1_fidelity_parity.py`'s pattern) before any label is used. The arm kinds, sidecar checks and gate kind lists for v7 are not added here; they come with the arm configs.
  - **Decision charge: a measured w per v7 arm, calibration planned now (as the filed P7 recipe, job 884584).** One cell per arm, the v7 gnn, v7 twin, v7 MLP and v7 wide MLP, on one exclusive node (SMT off, `PAR 1`, 1 torch thread, the node class the other w's were measured on), 19461 g0 both rungs, k = 0, seed 1, 2 cells per arm, at the pin that runs the arms. The scalar is the sum of decision seconds over the sum of plan evaluations, and the per-bucket values are the pooled w by served-batch bucket (`uncut_charge_share.pooled_w`); the H3 drift stop (2 %) applies. The cells need a v7-capable live path under option D and any checkpoint of the arm's architecture and contract (an untrained one times the same code), so they can run as soon as the capture pin exists, before any trained v7 weight; the w's go into `decision_cost_w.json` under new `ra_{gnn,twin,mlp,mlpwide}_eng_combfl_link` keys and are never aliased to a v6 key.
- 2026-10-23 — **Capture design, filed before any capture cell (S6).** A data step: no outcome, label or training is read from it, and the plan draw reads no latency.
  - **Capturing arm:** the bar `cd_ib0_inflightfull_k100` (`cd_ib0_inflightfull` at `_k100`, seed 0) served under option D (`HEROSIM_EXCHANGE_TCP=1`, `HEROSIM_EXCHANGE_FAIRSHARE=1` serial, default cellular uplink, `R1A_SLATE=declared_pruning_v1_uncapped`), with `HEROSIM_SNAPSHOT_FIDELITY=1` and the sealed-snapshot hook (`R1A_SNAPSHOT_PLAN`). It is one arm and one seed: the states do not depend on a learned arm, as in Arm 1. It departs from Arm 1's capture arm (`cd_ib0` k = 0 hand) in running under D and at `_k100`; the states are the bar's under the physics the v7 arms are served in.
  - **Payloads:** `cap200`, the rule of the filed D inputs: `min(bytes, 200 MB)` per pair, nothing else changed (`payload_cap_v1`). I minted it from the Arm 1 inputs (`/share/nikola.lukic/uncut_arm1/inputsA/inputs_group`, same workloads and configs) with S6's `mint_flowcap.py` (`~/s6_scripts`) into `/share/nikola.lukic/uncut_groups/flowcap/in_flowcap`: 4 % of pairs clipped (moderate g0 3,021 of 74,312; g1 2,903 of 72,184; g2 3,044 of 75,233; g3 2,904 of 71,949), against 4.1 % in the D inputs (3,017 of 74,147). Each file records its source sha256.
  - **Windows and rungs:** g0, g1, g2, g3 × moderate and heavy.
  - **Topologies (16):** the lowest 16 ids of `arm1_training_topologies.json` (md5 `5fcfb60e866e0bb7adf83d6c1d1dcb54`, 44 training ids) that are not the four own-state pilot ids 16201–16204: **16205, 16206, 16207, 16208, 16209, 16211, 16212, 16213, 16214, 16215, 16216, 16217, 16218, 16219, 16221, 16222.** None is 16226, a held-out id (16210, 16220, ...), a gate, TE3 or accel id, or 30176–30181. 16 × 4 windows × 2 rungs = 128 cells.
  - **Trigger sampling:** the pre-drawn plan of amendment A2 (`dagger_corpus.py plan`, arm-independent): `--total 4000 --seed 20261023`, Arm 1's size-bucket stratification, 30 % target at t < 360 s, groups over 256 tasks excluded, no top-up. Drawn **3,880** (shortfall 120 reported, not topped up): quotas ≤ 10 / 11–50 / 51–100 / 101–256 = 1,600 / 1,533 / 667 / 200; moderate 1,931, heavy 1,949; early share 0.3168 (floor 0.25). Plan `/share/nikola.lukic/uncut_groups/flowcap/plan_flowcap.json`, md5 `f387e648790c15d7a8fc7af3b60cea91`, sha256 `da49987a4ac00fb666ea3bf0963e6cdb8939b4c38f1ffc86680931672b27fe2b`. The A2 stop rule applies: a cell capturing under 80 % of its drawn triggers stops labelling.
  - **What a state stores:** the existing fidelity block plus `fidelity["flows"]` (schema v1, `src/placement/flow_snapshot.py`, commit `c0a6c3a7`; capture `7e4b54e3`): per ghost the ordered partner transfers with status, remaining bytes and rate; FairShare's active flows in order; the pre-exchange read remaining. Ghosts whose exchange cannot be planned at the instant are listed in `unplanned`. The capture only reads: it does not advance FairShare.
  - **Smoke (one cell, 16201 g0 moderate, bar s0, plan = every 3rd group; pin `7e4b54e3`):** `averageElapsedTime` 11.348520884232148 and `totalPeerExchangeTime` 522143.87876118824 with and without capture, identical; all 3,103 captured snapshots carry a flows block that passes `validate`. Active flows per state: median 28, p90 107, max 774. Exchange records per state: median 30, p90 110. Unplanned ghosts per state: median 0, max 102. Records are 99.6 % `input_io` (planned = False). The 16201 cell is a pilot topology and is not part of the capture list.
  - **Pin of the capture:** the reference-physics merge of `s6/flow-capture`; its sha is in every captured cell's `code.commit` and is filed with the first capture job.
- 2026-10-23 — **Replay identity (1)–(3), filed before any label is used (S7).** Code `s7/flow-replay`, merged on reference-physics `a9733951` (replay `src/placement/flow_replay.py`, hook in `snapshot_fidelity.apply_platforms/_serve`, relabel switch `uncut_label_screen relabel --stored-flows`; tooling `scripts_cosim/flow_replay_identity.py`). Switch `HEROSIM_REPLAY_STORED_FLOWS=1`, default off; it needs `HEROSIM_EXCHANGE_FAIRSHARE=1` and a `fidelity["flows"]` block, else FAIL LOUD.
  - **What the replay does.** It re-creates every stored active `FairShare` flow in the stored `active` order (hops, remaining bytes, scale, owner) in one reschedule, and resumes each ghost's serial exchange from its stored status (read, TCP start-up holding no link, shared bytes, latency tail, pending partners) instead of `timeout(io_remaining)`. Ghosts with `planned=True` (cold/ingress) run the plan evaluated at the snapshot instant; `unplanned` ghosts take the old path (no exchange). Fabric exchange tallies (`record_exchange*`) are telemetry and are not re-booked.
  - **Check (1), switch off is byte-identical.** Job 901688 (head `affea5fd`, same replay path as the merge): 24 of 24 stored states (12 twin-corpus, 12 FL-corpus; 3 per size bucket, `101_256` included) relabelled with the switch unset give rows byte-identical to the filed `relabel20` rows. The full suite was green on the merge.
  - **Check (2), the stored flows reproduce.** `start_active` recomputes the rate of every re-created flow and raises when it differs from the stored `rate` by more than 1e-9 relative. It ran without raising on all 240 states below (median 28, p90 112, max 774 active flows per state). Synthetic identity (`tests/test_flow_replay.py`, 27 tests): a live multi-task exchange sharing one narrow link, snapshotted at 30+ instants (read, TCP start-up, shared bytes, latency tail, done, a task that begins after the instant), TCP on and off, replays to the live end times within 1e-9.
  - **Check (3), served-plan replay vs live.** Cell: the bar `cd_ib0_inflightfull_k100` s0, 16201 g0 moderate, option D with cap200 payloads (S6's smoke cell, `flowcap/smoke_flowlog`, pin `deb1f04d`; same run as `smoke_cap`). 240 of its 3,103 decisions, spread evenly in time, with the served plan placing the whole batch (batch median 3, p90 6, max 147). Replay: the snapshot with the SERVED plan from `chosen.jsonl`; live: the sum over the batch of (completion time − snapshot time) from `tasks.npy` column 1. The live run charges the decision time to every task of the batch (`plan_evals × 0.0116 ms`); replay and labels do not, so it is removed from the live figure (raw numbers are in the rows). Rows `flowcap/smoke_flowlog/replay_identity_rows.jsonl` md5 `61abbadb85997b9089266fe1aa8b7bde`, read `…read.json` md5 `e2947c6689ba9deedc52ed3d3411260a`.
    A later flow is one that begins after the capture, inside the batch's live window, from a task the snapshot does not represent (not in the batch, no stored exchange record), joined from S6's `HEROSIM_FLOW_LOG` (24,952 flow starts).

    | gap \|replay − live\|/live | n | switch | median | p90 | max | within 1 % |
    |---|---|---|---|---|---|---|
    | all decisions | 240 | off | 0.061 % | 45.0 % | 1207 % | 69 % |
    | all decisions | 240 | **on** | 0.022 % | 1.07 % | 49.0 % | 90 % |
    | no later flow on the batch's links | 211 | off | 0.040 % | 14.0 % | 1207 % | 78 % |
    | no later flow on the batch's links | 211 | **on** | **0.018 %** | **0.14 %** | 9.1 % | 98 % |
    | a later flow on the batch's links | 29 | off | 45.0 % | 87.1 % | 94.0 % | 0 % |
    | a later flow on the batch's links | 29 | **on** | 3.7 % | 36.6 % | 49.0 % | 31 % |
    | no later flow on the batch's or the stored flows' links | 202 | **on** | 0.018 % | 0.13 % | 9.1 % | 99 % |

    Errors are two-sided (78 states replay faster than live by more than 0.01 %, 82 slower). No state is exact at 1e-9 (the median residual is 0.02 %).
  - **Ghost classes per state (240 states).** Stored exchange records (`input_io`): median 29, p90 114, max 786. `planned` (cold/ingress): median 0, max 10, in 7.5 % of states. `unplanned`: max 1, in 0.4 % (one state). Rendezvous ghosts: 0 in every state. Gap with the switch on: states with no planned or unplanned ghost (221) median 0.020 %, p90 0.72 %; states with a planned ghost (18) median 0.041 %, p90 1.8 %, max 2.5 %; the one unplanned state 9.1 %. This sizes the planned/unplanned approximation as small on this cell; the full capture (g1–g3, heavy) is rechecked below.
  - **The worst state, 2374 (trigger task 38672, t = 7169.35 s, 5 tasks): CONFIRMED.** Live 211.6 s against 107.8 s replayed (−49.0 %). One batch task spends 82.8 s in its exchange in the replay and 182.8 s live; the flow log shows 472 flows from tasks the snapshot does not hold, 367 of them (3.77 GB) on the batch's links inside its 183 s window. The replay has no post-capture arrivals; this is the named cause of the residual.
  - **Three clean-group states above 5 %: 531, 907, 971.** Not explained by a later flow on the batch's links. 531 (gap +9.1 %, 4 tasks) holds the one unplanned ghost; 907 (−6.0 %) and 971 (−7.7 %) have 3 and 2 later flows on the stored active flows' links and large stored sets (137, 141 active flows); with the switch off the replay gives the same number, so the stored flows did not move them. These are candidate causes, not shown.
  - **Undisproven hypothesis for the 0.02 % median.** I subtract `plan_evals × 0.0116 ms` per task of the batch; live may charge the decision time differently (once per batch, or per task in another order). Not isolated; it does not change plan ranks.
  - **Disclosed properties, not defects.** (i) The label is the batch optimum from the snapshot state with later arrivals absent, the same definition as every filed label; feeding later tasks at their live placements would change the label's meaning and leak the bar's later decisions (expert ruling), so it is not built. Replay-versus-live: overall p90 1.1 %, max 49 %; with a later flow on the batch's links median 3.7 %, p90 36.6 %. (ii) `planned`/`unplanned` ghosts are approximated as above. (iii) Per-task pairing of `tasks.npy` rows with batch gids can swap for tasks of one application (rows 38675/38676 are swapped in `tasks_lock`); batch sums are unaffected.
  - **Still to file.** The full-corpus (128 cells) planned/unplanned recheck, then the label file of the `--stored-flows` relabel (sha).
- 2026-10-23 — **Union corpus: build chain and the label pool, filed before the relabel runs (S7; expert ruling for the pool).** No labelled union corpus existed (`collect_union` holds snapshots only), so the chain runs at one pin in `llf_corpus/`: (1) `dagger_corpus.py states` (arm `bar`, token `cd_ib0_inflightfull_k100`, seed 0, plan `plan_flowcap_union.json`): 11,125 of 11,125 drawn, 0 missing, 0 over 256 tasks; buckets ≤ 10 / 11–50 / 51–100 / 101–256 = 4,618 / 4,396 / 1,797 / 314; `states.json` md5 `221441b041af436f6392ecd34087fe76`. (2) first-pass labels, `uncut_arm1.py label --per 50`, K1 `ca47cb62`, uncontended S search as the twin, FL and fslabel labels (job 902046). (3) replica check, (4) build into `dagger_bar`, (5) new-only split. (6) the relabel under option D (`HEROSIM_EXCHANGE_FAIRSHARE=1`, `HEROSIM_EXCHANGE_TCP=1`) with `--stored-flows --record-pair --union-sfs`.
  - **The candidate pool of step (6) is a union (expert ruling).** For plan spaces ≤ 600 the enumeration, replayed in full. Otherwise, deduplicated by plan and all replayed with the stored flows: the first-pass label, the top 8 by S of the stored first-pass ILS, and the top 8 by S_fs of an ILS run with the shared-rate surrogate (`pg_sfs`, `fs_sfs_study`) under the stored flows at the same per-50 budget. The fastest replayed plan is the label (strict improvement over the first-pass label, `decide()`). Because the label is the minimum over the pool, the union is never worse than either pool on the replayed RTT; the S_fs study found S_fs better on 30.8 % of states and worse on 20.0 % (`dagger/fs_label/sfs/read.json` md5 `443d302bdeff9abd78009085cf7dd261`), and the union keeps both. Each row records the pool of the winner (`first_pass`, `S`, `S_fs`, `both`, `exhaustive`); the shares by bucket are reported with the label file.
  - **Why `--record-pair`.** The first-pass label's stored RTT was made under uncontended physics, so its replayed RTT under option D with the stored flows legitimately differs; the pair is kept in the rows and summarised in the stats, and not in the label file.
- 2026-10-23 — **Union labels filed (S7): `llf_corpus/label_override_union_flows.json`, sha256 `11532616732908455841291c94ff822b5675fa44e0bc3a576ef4d2d7111fc066`, md5 `d97d173b9a77567469b5f52950018b6e`; 10,720 entries over the 10,720 built datasets of `llf_corpus/corpus/dagger_bar` (split `corpus/split_newonly.json` md5 `593b0a401ed048b777305fc3b608e19b`).**
  - **Chain and pins.** States, first-pass labels, replica check and build at reference-physics `1efc59a0`; the relabel at `f9cd34fd` (`1efc59a0` + `--union-sfs` + the hand-replan record; merged with its ancestors); merge by `llf_label_merge.py`. States 11,125 → built 10,720 (exact 3,608, hand 4,134, ils 2,978; 269 rejected, 136 dropped at label); replica check 10,989 states, 0 failed.
  - **Relabel.** Option D with the stored flows, `--stored-flows --record-pair --union-sfs`: the pool is the first-pass label, the top 8 by S of the stored first-pass ILS and the top 8 by S_fs of an ILS under the stored flows (per-50 budget); plan spaces ≤ 600 exhaustive; the fastest replayed plan wins. Hand-labelled states: the hand rule's re-decision under the stored flows differs from the first-pass hand label on 63.5 % of them (`hand_replan_differs`), by bucket ≤ 10 / 11–50 / 51–100 / 101–256 = 63 / 60 / 70 / 75 %; the first-pass label stays a candidate. The first-pass label's replayed RTT under flows over its stored uncontended RTT: median 1.0, p10 0.67, p90 7.5.
  - **How the rows were assembled.** 9,295 rows from the 24-shard run (job 902201, which lost four shards to the 3,600 s per-state timeout), 1,305 from the re-split of the remaining 1,425 states over 20 tasks by size (job 902401, timeout 21 600 s, no code change: 20 split files), and 118 from a second re-split of tasks 9 and 15 (job 902772). Duplicate rows are accepted only when equal but for wall times. Two sbatch defects found on the way: a state timeout left the job COMPLETED (fixed by `llf_relabel.sbatch`, non-zero exit unless one row per state), and the first pin's check 'the engine's hand plan is the stored hand label' failed under stored flows (recorded instead, `hand_replan_differs`).
  - **Disclosure: 2 of 10,720 states carry their FIRST-PASS label** (`relabel_unfinished: true`, `method: first_pass_carried`, named in `_meta.relabel_unfinished`): `gnn_datasets_wf1_dagger_bar/ds_02312` (`dagger_bar:16215:g1:heavy:46605`, 3-task batch) and `ds_05537` (`dagger_bar:16229:g1:heavy:45753`, 2-task batch). Cause (faulthandler, job 902792): not a hang; both are late heavy instants (t = 3,678 s and 3,630 s) with 6,166 and 6,149 active flows and 14,503 and 14,264 transfers, `FairShare._reschedule` costs O(active flows) per event, and with a batch of 2–3 tasks the pool is exhaustive (hundreds of plans), each replay minutes to hours; the first run stopped on each at the 3,600 s timeout and the re-split tasks were cancelled after more than 2 h. Their `cd_rtt` is the first-pass RTT under uncontended physics. (Two more heavy states of the same kind, `ds_01158` and `ds_05532`, with 3,533 and 5,317 active flows, finished in the re-split.)
  - **Label movement (final label vs the first-pass label, and vs the plan the bar SERVED, compared on the tasks the live decision placed).**

    | bucket | n | final ≠ first-pass | first-pass ≠ served | final ≠ served |
    |---|---|---|---|---|
    | ≤ 10 | 4,407 | 40.4 % | 51.3 % | 67.6 % |
    | 11–50 | 4,283 | 79.5 % | 45.7 % | 87.7 % |
    | 51–100 | 1,736 | 90.6 % | 49.6 % | 94.6 % |
    | 101–256 | 294 | 94.2 % | 42.5 % | 96.6 % |
    | all | 10,720 | 65.6 % | 48.5 % | 80.8 % |

  - **Winner pool** (all 10,720): first-pass label 3,685 (34.4 %), S only 1,927 (18.0 %), S_fs only 2,795 (26.1 %), in both pools 1,787 (16.7 %), exhaustive 524 (4.9 %). By bucket (first_pass / S / S_fs / both / exhaustive): ≤ 10: 2,626 / 161 / 265 / 834 / 519; 11–50: 879 / 1,070 / 1,552 / 777 / 5; 51–100: 163 / 596 / 832 / 145 / 0; 101–256: 17 / 100 / 146 / 31 / 0. S_fs alone supplies the winner on about half of the 51–256 states; its ILS cost 127,875 cpu-s.
- 2026-10-23 — **Registered** (SCIENTIST, at the expert's request with the user's go). The motivation figures were checked against the datalab files named above. The audit claim was checked in code, and the 127.5 / 7.31 s pair was recomputed from the cells. Open before data:
  - S6 names the capturing arm, windows and state sampling;
  - S7 files the replay identity (1)–(3);
  - S4 files the `partial_state_v7` spec and parity test;
  - the D admission rule is filed verbatim.
