# live_link_features_v1 — do live link-sharing features let message passing beat its MP-OFF twin under fair-share physics?

**Status:** `REGISTERED` (2026-10-23; no capture, label, training or cell yet). The user's go is relayed by the expert. SCIENTIST wrote the registration, and the expert coordinates. Tracks: S6 captures under option D, S7 owns the replay fix and its identity check, S4 owns `partial_state_v7` and its train/serve parity test.
- **Question.** Under option D, can the GNN beat its own MP-OFF twin once both see the network flows that share each link? Both are given the same new columns, so a difference is message passing, not information.
- **Why now.** On fair-share labels the GNN beats the MLP but ties its twin. The audit found the cause in the labels: they priced only the contention inside a batch, because the replay drops the flows already in flight (see Motivation).
- **Primary contrast:** v7 gnn vs v7 twin, read once on 12 newly minted topologies. Development on 30176–30181 gives a direction only and can't produce a label.
- **Standing answer (2026-10-23):** none yet. Until the confirmation read is filed, no message-passing claim may be made from this lineage.

## Motivation (sources checked by SCIENTIST, 2026-10-23)

- **Option D** is TCP slow start (`slow_start_v1`, per partner) plus fair-share peer exchange (`processor_sharing_v1`, serial partners, directional links), with the default cellular uplink and no payload override. Source: the filed bar cell `d_cap200/heavy/cc40s30176__g0heavy__cd_ib0_inflightfull_k100_s0.summary.json` (blocks `peerExchangeTcp`, `peerExchangeFairshare`; `accessCellularOutMbps` null; pin `5e595e52`). Paths are under `/share/nikola.lukic/uncut_groups/anomaly/fairshare/`.
- **The fair-share label read** (S6; `d_fsl/read.txt` md5 `df11a84758441703703615f7e50798dc`). It used one seed (s1) and 20 epochs, on 30176–30181, g0+g1, both rungs, pin `d3b3de5a`. It is direction-only by its own header.
  - The fslabel gnn beats the fslabel MLP by a pooled −12.31 %, faster on 12/12 (moderate −10.89 % 6/6, heavy −28.65 % 6/6; each p = .0312, the n = 6 floor).
  - The fslabel gnn ties its fslabel twin: pooled −1.17 % (7/12, p = .73); moderate −2.36 % (4/6), heavy +1.16 % (3/6).
  - **"12/12" is 6 topologies × 2 rungs, not 12 topologies**, and the pooled p treats the two rungs of a topology as independent. Quote it as 6/6 at each rung.
- **The audit (verified in code).** The fidelity snapshot keeps in-flight tasks as ghosts, each with a fixed `io_remaining` taken from the static `_peer_exchange_time` estimate (`src/placement/snapshot_fidelity.py:131`, `record=False`). It captures no `FairShare` flow state (`network_fabric.py:206–260`). So a label replay under fair share (`uncut_ils.replay_rtts` → `plan_oracle._label_chunk`) prices sharing only among the batch's own transfers. This is why `uncut_label_screen.py:189` lets `--record-pair` keep the stored and replayed RTT apart instead of failing.
- **The collapse loop.** The first contention-aware hand rule (`cd_ib0_inflightfull_fsaware`, which reads live flow counts) ran at **127.5 s per task vs the bar's 7.31 s** on cc40s30176 g0 moderate (`d_cap200/diag_herd_moderate/*.tasks.npy`, mean of column 1, SCIENTIST recomputed it). Two fixes each removed it: counting the batch's own committed flows (`_commit`, 6.18 s), and pricing the drain walk solo (`_nodrain`, 6.05 s). The mechanism is herding: every task in a batch sees the same currently empty link and picks it. A learned scorer that reads live flows can herd the same way.
- **The D grid read** (`d_cap200/d_read_main.txt` md5 `9b7fd746db5dc7237a62e7cca298b634`, rules `d_read_rules.txt` md5 `cba0597e967169cf184c3f11618633a0`) holds the statistic and runaway rule used here. It has 19 runaway cells at heavy g0 alone, including the bar itself at 862 s on 30179.

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

- 2026-10-23 — **Registered** (SCIENTIST, at the expert's request with the user's go). The motivation figures were checked against the datalab files named above. The audit claim was checked in code, and the 127.5 / 7.31 s pair was recomputed from the cells. Open before data:
  - S6 names the capturing arm, windows and state sampling;
  - S7 files the replay identity (1)–(3);
  - S4 files the `partial_state_v7` spec and parity test;
  - the D admission rule is filed verbatim.
