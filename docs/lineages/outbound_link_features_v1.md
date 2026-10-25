# outbound_link_features_v1 — does a GNN that also sees the outbound side of the exchange beat the bar, its MP-OFF twin and the MLPs under option D?

**Status:** `REGISTERED` (2026-10-25) — registered before any v9 checkpoint, any mint, any cell or any read. Drafted by S7 at the expert's request with the user's ruling on the primary contrast. Not yet signed by SCIENTIST. The registered bar is `cd_ib0_inflightfull_k100` (pin `0ec7cdcb`).
- **Question.** Under option D (fair share plus TCP), do the `partial_state_v9` GNN and its separately trained MP-OFF twin, which add the outbound direction of the peer exchange to `partial_state_v8`, beat (a) the registered bar, (b) the v9 MP-OFF twin, (c) the parameter-matched wide MLP and (d) the same-input MLP, on fresh topologies?
- **Caveat on the bar (quote with every bar number).** vs the registered bar (`0ec7cdcb`), which prices peer exchange inbound only and so under-prices servers with a slow (cellular) uplink; not a comparison with a sound hand rule.
- **Why.** `llf_cd_confirm_v1` confirmed the static link-feature gnn against the bar and the wide MLP; `mp_confirm_v1` confirmed it against its MP-OFF twin (the parent's TIE in `live_link_features_v1` is superseded). Every link column of v7/v8 prices the partner → candidate direction. The pair table is symmetric, so each remote partner also pulls from the candidate over the candidate's own access uplink (cellular out 4 / in 75 MB/s). v5+ already carries the push direction at the solo rate (columns 25–26); no column prices it at the live fair-share rate, and none counts the candidate's own outbound flows sharing its uplink.
- **What this is not.** Not a comparison with a sound hand rule (caveat above). Not a message-passing claim unless the twin contrast passes on its own. v9 vs v8 isolates the new columns (secondary below); the primary family does not.

## Contract `partial_state_v9`

v8 byte for byte (columns 0–32), plus `LINK_OUT_DIM` = 4 scorer columns (33–36), one definition `placement/link_features.out_edge_columns`, over the task's committed partners and the active plus pending flows (the committed prefix refreshes them, as E1–E3):
- **R1–R3**, the mirror of E1–E3 on the route candidate → partner: log1p of the fair-share seconds, log ratio to solo, log1p of the busiest hop count.
- **U1**, log1p of the seconds the candidate's uplink needs for this task's k remote outbound flows (B bytes in all) when they share it with the c flows already on it: B (c + k) / (bw · MiB). Zero for k = 0.

The conv's edge attribute stays v8's, so the gnn and its twin read the same columns through the scorer. Code: branch `s7/partial-state-v9`, commit `4326555c` (merge sha to be filed in the Record), tests `tests/test_link_out.py` (definition against hand values, sequential columns against the batched loss on random plans, masks and prefixes, v8 columns untouched, v7/v8/v9 sidecar refusals). Serving refuses a checkpoint of another contract; a sidecar-less checkpoint is not served.

## Precondition: the label probe (before any training)

The labels are the fixed fair-share replay of `live_link_features_v1`, reused unchanged (`label_override_union_flows.json` sha256 `11532616732908455841291c94ff822b5675fa44e0bc3a576ef4d2d7111fc066`). They price the reverse pull only if the replay charges a partner's pull from the candidate. The probe (datalab job 907548, replay at `95293af0`, stored flows on, bar states of cc40s16205–16211 g0 moderate, batches 3–14) replays the served plan with one task A moved between a wired and a cellular candidate of the same platform type, both idle and initialised, and reads the per-task exchange times: Δ_in (A's own), Δ_out (its remote in-batch partners'), against the solo prediction from the cell's link topology.
- **Rows used:** those with predicted Δ_out ≥ 0.5 s.
- **PASS:** Δ_out > 0 on ≥ 90 % of the rows, the median Δ_out / predicted Δ_out in [0.5, 2.0], and the median Δ_rtt > 0.
- **Otherwise** the probe is reported as is and nothing trains: the labels are fixed before v9 features are introduced.
- The probe script is scratch; its output rows and their md5 are filed in the Record.

## Arms (one corpus, labels, recipe, epochs and seeds)

| Arm | Role |
|---|---|
| v9 gnn | treatment |
| v9 MP-OFF twin (`disable_message_passing`, the same columns and prefix refresh) | primary control |
| v9 wide MLP (hidden width matched to the v9 gnn's parameter count) | primary control (capacity) |
| v9 same-input MLP | primary control (pointwise) |
| bar `cd_ib0_inflightfull_k100` | primary control, with the caveat above |
| v8 static e40 gnn (`02668ecf`, the filed checkpoints) | secondary: v9 vs v8, same corpus and labels |

- **Recipe.** The parent's A6: 40 epochs, minimum 40, no early stop, wandb on; training configs `experiments/uncut_linkout_*.yaml` are added with the training run, after `tests/test_trainer_determinism.py` passes.
- **Decision charge.** Each v9 arm carries its own measured per-(arm, rung) charge, taken on the v9 code (the extra columns cost time). Never aliased to a v8 value. The bar keeps 0.0116 / 0.01098 ms per plan.
- **Development (direction only).** Topologies 30176–30181 (already read under D, so post-look), g0 + g1, both rungs, seeds 1–3, 20 epochs. It may change the recipe only; each change is filed before the confirmation mint. The early collapse check of the parent applies (hold at 2× the v8 gnn cell or 57 s per task on cc40s30176 g0 moderate, 30179 and 30181 g0 heavy).
- **Confirmation.** A fresh mint of 24 topologies by the parent's recipe and admission rule (`live_link_features_v1.md`, "Confirmation"), disjoint from every list in use including 30327–30350. Option D, g0 + g1, both rungs; learned arms at seeds 1–6, bar at s0. Per the parent, the confirmation runs whatever development shows.

## Statistic and primary family

The parent's, with `confirm_rulings.json` (md5 `b609acef33b10e7369a0f5aa58b42477`) unchanged: metric `whole`, paired % per cell, a learned cell the mean over seeds, per topology the median over g0 and g1, then the median over topologies and the exact two-sided Wilcoxon, zeros dropped; runaway > 10× the arm's rung median.
- **Primary family: 8 tests, Holm over the 8.** gnn v9 vs {bar, v9 twin, v9 wide MLP, v9 same-input MLP} × {moderate, heavy}. Primary reading without runaways; the all-cells reading is a guard (a test cannot be CONFIRMED while its all-cells median is ≥ 0).
- **Labels per test.** CONFIRMED: median < 0, Holm p < .05, all-cells median < 0. DIRECTION: median < 0, not CONFIRMED. NOT-FASTER: median ≥ 0 or no topology left. A contrast is CONFIRMED when both rungs are.
- **Secondary, Holm over its own pair:** v9 gnn vs v8 static gnn, moderate and heavy. Descriptive beside every test: the all-cells row, per-seed medians, the after-360 s row, the runaway list per arm. Heavy is runaway-dominated in this physics; quote it as a direction plus Holm, not a magnitude.
- **Reader.** `llf_confirm_read.py` in its `--ids/--arms` mode, extended for the v9 arm names; the default mode stays byte for byte. SCIENTIST freezes the blob before the first cell is read; one read.

## Prior (S7, rough; SCIENTIST sets theirs)

The bar contrast is expected to confirm: the v8 gnn already beat the same bar by −21 % / −26 % and the bar prices the outbound side wrongly. The informative contrasts are the twin and the v8 secondary. P(gnn v9 CONFIRMED vs v8 at either rung) about 0.3. P(gnn v9 CONFIRMED vs the v9 twin, both rungs) about 0.5, given `mp_confirm_v1`.

## Record

- 2026-10-25 — **REGISTERED.** Contract and tests on `s7/partial-state-v9` (`4326555c`), label probe submitted (job 907548); nothing has been read.
