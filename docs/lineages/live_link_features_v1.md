# live_link_features_v1 — do live link-sharing features let message passing beat its MP-OFF twin under fair-share physics?

**Status:** `REGISTERED` (2026-10-23; no capture, label, training or cell yet). The user's go is relayed by the expert. SCIENTIST wrote the registration, and the expert coordinates. Tracks: S6 captures under option D, S7 owns the replay fix and its identity check, S4 owns `partial_state_v7` and its train/serve parity test.
- **Question.** Under option D, can the GNN beat its own MP-OFF twin once both see the network flows that share each link? Both are given the same new columns, so a difference is message passing, not information.
- **Why now.** On fair-share labels the GNN beats the MLP but ties its twin. The audit found the cause in the labels: they priced only the contention inside a batch, because the replay drops the flows already in flight (see Motivation).
- **Primary contrast:** v7 gnn vs v7 twin, read once on 12 newly minted topologies. Development on 30176–30181 gives a direction only and can't produce a label.
- **Pre-data amendment A3 (2026-10-23; the user's decision; Record):** seed-1 100-epoch runs of every variant start now, in parallel with the 20-epoch selection runs. They are sealed (no live cell, curves unread by the selectors) until the selection is filed. The winner's runs become A2's seed 1; the others are exploratory only.
- **Pre-data amendment A2 (2026-10-23; the user's decision; Record):** after the seed-1 selection, the chosen variant's four arms train at 100 epochs (the cap), seeds 1–3 from scratch, for development and confirmation. The 20-epoch selection read is never pooled with them.
- **Pre-data amendment A1 (2026-10-23; the user's decision; Record):** as built, v7's link columns E1–E3 reach only the EdgeScorer, which the twin shares, so message passing sees only the at-capture P1–P3. Two GNN variants, (a) a static link block in the conv and (b) re-encoding per placement, run beside the as-built arms at seed 1, 20 epochs, on 30176–30181. A rule fixed now picks the one variant carried to seeds 2–3 and confirmation. This stage gives no label.
- **Standing answer (2026-10-23):** none yet. Until the confirmation read is filed, no message-passing claim may be made from this lineage.

## Motivation (sources checked by SCIENTIST, 2026-10-23)

- **Option D** is TCP slow start (`slow_start_v1`, per partner) plus fair-share peer exchange (`processor_sharing_v1`, serial partners, directional links), with the default cellular uplink and no payload override. Source: the filed bar cell `d_cap200/heavy/cc40s30176__g0heavy__cd_ib0_inflightfull_k100_s0.summary.json` (blocks `peerExchangeTcp`, `peerExchangeFairshare`; `accessCellularOutMbps` null; pin `5e595e52`). Paths are under `/share/nikola.lukic/uncut_groups/anomaly/fairshare/`.
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
    5. Sealed: inputs md5s and the id-list sha256 filed in the mint line; nobody reads outcomes before the reader is frozen.
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
    - **Gap:** no parameter-matched MLP with the static block exists. Under (a), the wide-MLP contrast also changes the block, and is reported so.
  - **Compute.** The early set is 13 training jobs on top of the 20-epoch selection runs:
    - 9 at seed 1 (2 of them on GPU);
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
- 2026-10-23 — **Registered** (SCIENTIST, at the expert's request with the user's go). The motivation figures were checked against the datalab files named above. The audit claim was checked in code, and the 127.5 / 7.31 s pair was recomputed from the cells. Open before data:
  - S6 names the capturing arm, windows and state sampling;
  - S7 files the replay identity (1)–(3);
  - S4 files the `partial_state_v7` spec and parity test;
  - the D admission rule is filed verbatim.
