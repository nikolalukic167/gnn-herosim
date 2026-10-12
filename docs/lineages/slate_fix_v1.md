# slate_fix_v1 — does the GNN trained and served on the uncapped candidate set close its heavy gap to `cd_exactS`?

**Status:** `REGISTERED` (signed by the expert 2026-10-12 at d66ab923; drafted by S6). Launched 2026-10-12; no gate cell, no data yet.

**Now (2026-10-12).** Code and configs are on `rp/slate-fix`; the jobs run at the pinned code 4d90119c (438fa161 plus the eval `ulimit` fix; later commits are this node only). Launch order:
1. S7's recache and the cache check (`scripts_cosim/slate_fix_v1_cache_check.py`): PASSED.
2. **User-ordered amendment (2026-10-12, relayed by the expert):** training starts now and no longer waits for the `slate_bar_v1` read. The gate order is unchanged.
3. Eval-only identity task (job 863950, task 8; the first try 863945 died on a launch defect, see Record) → training (863951, gnn_eng / twin_eng, seeds 1–2; `afterok` on the identity) → offline screen (863952, tasks 0–7; `afterok` on training) → identity and smoke on 16490 → gate A and B → single read. A failed identity is fixed in code, never rerun.

**What the fix does not touch: read this first if the gap does not move.**
- **54 over-cap train datasets** (> 100,000 full plans) keep their capped top-5 labels: `untouched_over_cap` in `extend_manifest.*.jsonl`.
  - Source: S7's census of `gnn_datasets_wf1_train` before launch, as relayed by the expert on 2026-10-12. Of 2,731 datasets, 1,767 were to be extended, 910 were never pruned and 54 are over the cap.
  - The held-out manifest is complete with 9 over-cap.
- **648 sub-batched datasets** are tables cut from a batch whose pruned plan space exceeded the cap. Each is extended within its sub-batch of ≤ 4 tasks, but the peer cut across sub-batches stays in the label.
  - Source: S5's accel_replica_v1 corpus read on 2026-10-10 (jobs 855783 / 855785 / 855788; `/share/nikola.lukic/workload_fix_v1/accel_state_stats/summary.json`): 648 of 3,169 datasets, 20.4 %, 4.3 % of plans.
  - Split by `sub` in `full_rank_rows.jsonl` in the same directory (S6, 2026-10-12): **556 train** (heavy 364 of 1,415 = 25.7 %; moderate 192 of 1,316 = 14.6 %) and **92 held-out** (heavy 54, moderate 38).
- Both are re-counted from the final extend manifests when the build completes. On 2026-10-12 it was partial: 1,609 train rows, of which 1,067 extended, 507 already full and 35 `untouched_over_cap`.
- The heavy deficit to `cd_exactS` sits in exactly these large, split groups:
  - the accel GNN's loss is on split 9–10-task groups, about 22 % of tasks;
  - `selfsearch_gap_v1` found self-search loses on batches ≤ 8 and wins on 9+;
  - `slate_bar_v1`'s recall falls from 96.7 % (n = 2) to 56.7 % (n = 10) at heavy.
- So the fix reaches the small and mid batches, where the declared slate's recall was already 85–97 %, and leaves the part of the corpus where the gap lives.
- A gap that does not move says that part, not the cold-cost defect, is the binding constraint.

## Question and why

`selfsearch_confirm_v1` (CLOSED DIRECTION-ONLY) left the learned self-search arm behind the hand exact search at heavy.
- On g0–g3: gnn_eng self-search vs `cd_exactS` was −0.16 % moderate / **+3.66 % heavy** (2 of 24 topologies faster).
- Read again on g0–g1 only, the windows this gate uses (same 24 topologies, descriptive): self-search −1.03 / +3.51 %; no-split +0.35 / +5.44 %.

`slate_bar_v1` (S6 audit, 2026-10-12) found one cause that sits in the data and not in the model.
- The GNN's declared top-5 slate charges no cold start on an initialized platform; the simulation does. The fault is at
  `src/placement/declared_slate.py:46`, recorded in `docs/gates/gate-tools.md`.
- So the corpus labels and the served slate miss plans that `cd_exactS` (its own top 5, correct cold cost) can pick.
- G8 `cd_exactS` batch recall: declared top 5 79.5 % vs `cd_exactS`'s own top 5 89.3 %. At heavy, recall falls by batch size, from 96.7 % (n = 2) to 56.7 % (n = 10).

S7's `accel_uncap` (`scripts_cosim/extend_corpus_slate.py`, 4f41b771, rule `declared_pruning_v1_uncapped`) removes the cap without re-simulating.
- It keeps every candidate and scores only the plans the wider slate adds.
- It covers about 98 % of tables. The rest are over 100,000 full plans and keep the capped labels.

This gate asks whether the same architecture, retrained and served on the full candidate set, closes the heavy gap.

## Design (fixed at signature)

**Code.** `rp/slate-fix` = `rp/corpus-uncap` (S7: uncapped rule, serving mode, cache's one allowed mixed corpus) plus a merge of 2c84ab4e (the
`selfsearch_confirm_v1` gate harness: `ra_gnn_eng_selfsearch`, `ra_twin_eng_selfsearch`, `cdxapply`, `cd_exactS`). Two add/add conflicts took
`rp/corpus-uncap`'s side, which is a strict superset. Changes of mine, all in the commit after the merge:
- **Serving slate from the sidecar.** `scripts_cosim/fresh_topo_burst_v1_gate.py` (`ra_serving_slate`, `r1a_slate`) exports
  `GNN_SERVE_CANDIDATE_SLATE` as the rule the checkpoint's `.contract.json` records.
  - `R1A_SLATE` names the rule a gate submission expects; unset means `declared_pruning_v1`, so every earlier gate is unchanged.
  - A sidecar with another rule fails the cell before it runs.
  - The per-cell instrument check compares the served env to the same rule. The loader's `require_matching_slate` (`src/executesimulation.py:802`,
    `src/policy/gnn/prefix_serving.py:213`) is the second, independent check.
- **Sidecheck.** `scripts_cosim/joint_burst_v2_sidecheck.py` accepts `candidate_slate` = `R1A_SLATE` (default `declared_pruning_v1`).
- **New harness kind.** `ra_twin_eng_nosplit`, the twin's no-split decode, existed only as a self-search kind before.
- **Offline eval path.** `src/notebooks/train_near_rtt.py` has an eval-only path: `NEAR_RTT_EVAL_ONLY_CHECKPOINT` + `NEAR_RTT_EVAL_ONLY_JSON`.
  - It scores a checkpoint on the cache's val split through the same `evaluate()` the training loop selects on, then exits before any gradient step or
    checkpoint write.
  - It is env-gated, so training is unchanged; `tests/test_trainer_determinism.py` passes, 12 passed and 8 skipped as on the parent.
- **Two merged tests fixed** (`tests/test_slate_no_split.py`, test-only): one monkeypatch lambda now takes S7's new `rule` argument, and two source-string assertions were stale against the selfsearch line's own harness.
- **New files:**
  - `scripts_cosim/slate_fix_v1_cache_check.py`, `scripts_cosim/slate_fix_v1_gate_read.py`, `scripts_cosim/slate_fix_v1_selected.json`;
  - `scripts_cosim/datalab/slate_fix_v1_train.sbatch`, `scripts_cosim/datalab/slate_fix_v1_eval.sbatch`,
    `scripts_cosim/datalab/slate_fix_v1_gate_models.sh`;
  - `tests/test_slate_fix_v1.py` (13 tests).

**Corpus and cache.**
- Corpus: `/share/nikola.lukic/accel_uncap/4f41b771/{train,heldout}`. The over-cap tables are symlinked in untouched (`--link-untouched`).
- Cache: S7's recache under `partial_state_v5`, alpha `inf`, written 2026-10-12 by rp/corpus-uncap 836e469d (merged into this branch): `/share/nikola.lukic/accel_uncap/4f41b771/cache_run/cache`. Both configs name it as an absolute `cache_dir`. The training sbatch still refuses a placeholder.
- **Pre-training check** (`slate_fix_v1_cache_check.py`, metadata only, login-node safe). Against the capped cache
  (`/share/nikola.lukic/accel_prod/20261010_061132/cache_run/cache`, 3,169 graphs):
  - `candidate_slate` = `declared_pruning_v1_uncapped`, and `candidate_slate_rules` is absent or exactly both rules;
  - 13 contract keys are unchanged (`partial_state_contract`, `physics_env`, `training_contract`, ...);
  - the same 3,169 parent ids;
  - the split is `experiments/accel_replica_v1_split.json`, sha256 f5bc4f51…, which the g4 and twin checkpoints record. It has train 2,674 / val 438
    (wf1_heldout) / test 57 placeholder. The 57 test ids are never scored: `NEAR_RTT_SKIP_FINAL_TEST=1`.

**Training** (`scripts_cosim/datalab/slate_fix_v1_train.sbatch`, array 0–1).
- Configs: `experiments/slate_fix_v1_gnn_eng.yaml` and `experiments/slate_fix_v1_twin_eng.yaml`.
- They are the accel_replica_v1 g4 recipe and its MP-OFF twin byte-for-byte: lr 1e-3, width 128, `ce_only` + full-context CE, 100-epoch cap,
  patience 60, min 100. Only `lineage`, `cache_dir` and the W&B names differ; a test enforces this.
- Seeds 1 and 2. W&B project `gnn-slate-fix-v1`.
- Each checkpoint is sidechecked as `ra_gnn_eng` / `ra_twin_eng` with `R1A_SLATE=declared_pruning_v1_uncapped`, and its md5 is logged.
- **Deviation from the request, decided here.** You asked for four files, `slate_fix_v1_{nosplit,selfsearch}[_twin].yaml`. No-split and self-search are
  two serving modes of one checkpoint, as in `selfsearch_confirm_v1`, where `ra_gnn_eng_selfsearch` and `ra_gnn_eng_nosplit` served the same g4 weights.
  Four configs would train identical recipes twice and unpair the "search adds" contrast, so there are two configs and one checkpoint per arm and seed.

**Arms.** A cell is arm × topology × window × rung; 48 per arm and seed.

| arm | submission | stem / seeds | cells |
|---|---|---|---|
| **new self-search** `ra_gnn_eng_selfsearch` | A (`R1A_SLATE=…_uncapped`) | `slate-fix-v1`, 1–2 | 96 |
| **new no-split** `ra_gnn_eng_nosplit` | A | `slate-fix-v1`, 1–2 | 96 |
| new twin self-search / no-split (`ra_twin_eng_*`) | A | `slate-fix-v1`, 1–2 | 192 |
| new `cdxapply` (`ra_gnn_eng_cdxapply`: no-split plan, CD refine with expansion) | A | `slate-fix-v1`, 1–2 | 96 |
| old self-search / no-split, capped g4 (md5 0f915a36 / 446f99fd) | B (`R1A_SLATE` unset) | `accel-replica-v1`, 1–2 | 192 |
| `cd_exactS`, `cd` | B | seed 0 | 96 |

- Total: **768 cells**.
- The old checkpoints are served capped (their sidecar says `declared_pruning_v1`), exactly as every earlier gate served them. They are the "same model, old data" control.
- The old twin is not in the gate. Its offline numbers are in the screen.

**Offline screen (ordering only, never a bar).** `scripts_cosim/datalab/slate_fix_v1_eval.sbatch` scores val `regret_masked_topo` on the uncapped cache's
438 held-out tables for 8 checkpoints:
- new gnn and twin, seeds 1 and 2;
- old gnn (above) and old twin (bc44f9c2 / 7798e128), seeds 1 and 2.
- Disclosed: the old checkpoints are scored decoding over the full candidate set, not the capped slate they are served on.
- **Identity first (task 8):** the old gnn seed 1 through this path on its own capped cache must reproduce its recorded best_val 0.7807019727098458
  within 1e-6, or the screen is not read.

**Identity and smoke before the gate** (two submissions, old/hand and new, as the gate's A and B; on 16490 g0, both rungs; `SMOKE_IN` reuses selfsearch_confirm_v1's inputs; smokes are not read as results):
- **Flag-off identity.** `cd_exactS` and old `ra_gnn_eng_selfsearch` s1 reproduce the recorded `selfsearch_confirm_v1` cells
  (`/share/nikola.lukic/selfsearch_confirm/gate_79ea002c`) to the digit: mean latency and every scheduler counter. The only serving-code changes
  since 79ea002c are S7's uncapped branch (`declared_slate.py`, `src/policy/gnn/scheduler.py:855`), and the capped path is byte-identical per S7's tests.
- **Uncapped serving smoke:** the 5 new kinds at s1.
  - The sidecar round-trips: the harness serves `declared_pruning_v1_uncapped` and `require_matching_slate` passes in the loader.
  - `slate_declared_pruned` = 0 on every batch, `ss_batches` = `slate_declared_batches` on the self-search kinds, and unsplit = sub-batched on the no-split kinds.
  - Walls are measured here. They set `LIMIT` for submission A = max(8,100 s, 2.5 × the slowest smoke cell).

**Gate.**
- Seat: the accel seat, unchanged — 160 × 24, p 0.6, `fastest_compatible`, WF1 windows **g0–g1**, moderate (×11.6139) and heavy
  (×27.622665025860204), pipelined, `HEROSIM_REPLICA_RELEASE=1`, KPA.
- Inputs: `scripts_cosim/datalab/s160_gate_inputs.sbatch` with ib0_confirm_v1's arguments (base cfg `accel_nosplit/inputs/cfg_src/cc40s16345.json`) and
  `TOPOS` = 19420–19431. `scripts_cosim/datalab/slate_fix_v1_gate_models.sh` wires `models/`, checking md5 and sidecheck.
- Harness: phase `r1a` via `scripts_cosim/datalab/r1a_gate.sbatch` at the pinned worktree with `SELECT=scripts_cosim/slate_fix_v1_selected.json`.
  - A watchdog kill or a failed cell is counted and reported, never rerun.
- **Topologies 19420–19431.**
  - `git grep` of 194xx tokens over every local and remote branch finds only 19400, 19404–19417 and 19490.
  - A `/share/nikola.lukic` name scan for 1942x–1949x topology tokens (excluding the 6-digit `ds_` dataset dirs) finds none. `truth_19439.json` is a batch index, not a topology.
  - Disjoint from:
    - ib0's 19405–19416; overcap's 19206–19217 and the 19400–19404 prefixes;
    - 19300–19305 (never read);
    - selfsearch_confirm_v1's 16490–16513, L24's 16900;
    - the corpus topologies 16201–16258 (train, val and test 16226).

**Statistic** (`scripts_cosim/slate_fix_v1_gate_read.py`, committed now, before data).
- Per topology: the median over (window, seed) of the paired % of mean latency. The reference is its own seed when seeded, else seed 0; new vs old pairs seed s with seed s.
- Then the median over 12 topologies, the sign count and the exact two-sided Wilcoxon (n = 12: p floor 0.00049, Holm over 4 floor 0.002).
- Negative = the first arm is faster.
- The reader renames learned kinds by directory (`new:` / `old:`), refuses topologies outside 19420–19431, windows other than g0–g1, and a hand cell present in both directories.
- A second pass repeats every family without the topologies that have a failure.

**Registered families (Holm within each).**
- **PRIMARY (Holm over 4):** new self-search vs `cd_exactS` and new no-split vs `cd_exactS`, × moderate and heavy.
- **SECONDARY, data effect (Holm over 4):** new vs old checkpoint, self-search and no-split, × 2 rungs.
- **SECONDARY, MP (Holm over 4):** new gnn vs new twin, self-search and no-split, × 2 rungs.
- **Descriptive, untested:**
  - the gap change, i.e. (new vs `cd_exactS`) − (old vs `cd_exactS`) in pp, per topology then the median. This is the quantity both predictions are about;
  - old arms vs `cd_exactS`; new `cdxapply` vs `cd_exactS` and vs CD; new self-search vs new no-split and vs CD;
  - per-arm slate counters: pruned, sub-batched, unsplit, exact vs ascent batches.

**Labels (standard, fixed now).**
- **WIN** for an arm: median ≤ −2 % with Holm p < .05 at **both** rungs.
- **E-FASTER:** `cd_exactS` faster with Holm p < .05 at either rung (a LOSS).
- Otherwise **DIRECTION-ONLY** (median < 0 at both rungs) or **NOT-SEPARATED**, reported with both medians and p-values.
- Per rung: CONFIRMED / E-FASTER / DIRECTION-ONLY / NOT-SEPARATED.
- **NOT-COMPARABLE** if `cd_exactS` is not faster than CD at both rungs (the seat is then not the one selfsearch_confirm_v1 measured).
- Attribution:
  - a WIN with the MP family NOT-SEPARATED is a learned-scorer result, not a message-passing one;
  - a data-effect family not significantly negative means "uncapping did not help this model", whatever the primary says.

**Predictions, written before any data.**
- **Expert:** the heavy gap shrinks by 2–4 pp and ends NOT-SEPARATED.
- **S6:** the heavy gap shrinks less, by **0.5–2 pp**. New self-search lands at **+1.5 to +3 %** vs `cd_exactS` at heavy and stays **E-FASTER**; new no-split stays E-FASTER (+3.5 to +5 %).
  Moderate stays tied: |median| < 1.5 %, NOT-SEPARATED. **No WIN** for either arm (P(WIN) ≈ 0.05). Data effect: heavy −0.5 to −2 %,
  DIRECTION-ONLY or NOT-SEPARATED; moderate within ±1 %. MP: NOT-SEPARATED, as in selfsearch_confirm_v1.
  - **Reason:** the heavy deficit lives in the large groups. The sub-batch peer cut on split 9–10-task groups is ~22 % of tasks, and over-cap batches are ~half of heavy tasks at G8.
  - Those are exactly the tables `accel_uncap` leaves capped (> 100,000 full plans), and the batches where self-search falls back to coordinate ascent.
  - Uncapping widens per-task candidates 5 → 7–13 and mainly fixes small and mid batches, where recall was already 85–97 %.
  - A wider slate also pushes more batches past self-search's 625-plan exact limit, which can cost some of the gain.

**Not claimed:** a message-passing win; any seat other than the accel seat; other recipes or seeds; G8; "beats the best hand rule" unless the primary is a WIN.

## Cost and queue plan

Walls are measured on `selfsearch_confirm_v1`'s g0–g1 cells (`wallclock_s`, median / max).
- Capped: self-search 1.85–1.92 ks (max 2.3 ks), twin self-search 1.45–1.84 ks, no-split 0.46–0.50 ks, `cdxapply` 0.54–0.56 ks, `cd_exactS` 3.1 ks (max 3.9 ks), CD 0.09 ks.
- Uncapped learned cells are **estimated** at 1.5–3× those (7–13 candidates per task instead of 5); the smoke fixes the number.

| step | jobs (queue entries) | resources | wall (est.) |
|---|---|---|---|
| inputs, 19420–19431 (`s160_gate_inputs.sbatch`) | 1 | CPU-amd | ~2 min (ib0's took 72 s) |
| cache check (login node, metadata only) | 0 | — | seconds |
| training, 2 arms × 2 seeds in parallel | 2 | CPU-amd, 18 CPU, 150 G each | 3–5 h (twin: 1.6 h capped, 25 GB RSS per seed; × 2–3 for the wider graphs) |
| offline identity, then screen | 1 + 8 | CPU-amd, 8 CPU, 96 G | 0.5–1 h each |
| identity + smoke (16490 g0: `cd_exactS` + old self-search, and the 5 new kinds, × 2 rungs = 14 cells) | 2 | 64 CPU | 1–2 h |
| gate A (new, 480 cells) | 6 | 64 CPU, 240 G, PAR 24 | |
| gate B (old + hand, 288 cells) | 2 | 64 CPU, 240 G, PAR 24 | |
| reader | 1 | CPU-amd, 2 CPU | minutes |

- **Gate compute:** A ≈ 190–390 cell-h, B ≈ 105 cell-h; 768 cells over 192 concurrent slots is about 4 waves, **≈ 4–7 h** wall.
- **Total:** ≈ 0.4–0.6 M core-s training plus 1.1–1.8 M core-s gate; about 10–15 h end to end after the recache.
- **Queue:** 23 entries in all (limit 44). At most 8 CPU jobs at once (my share), so the gate arrays are throttled (`%6` and `%2`) and launch only when L24 and my
  other jobs have drained.
- **Order (amended 2026-10-12, see head):** S7 recache → cache check → eval identity → training → screen → smoke → gate A and B together → single read.

## Record (newest first)

- 2026-10-12 — **Smokes PASS; gate A and B launched.**
  - Smoke B 864098 (32 m; 4/4, 0 failed): **flag-off identity PASS.** `cd_exactS` and old `ra_gnn_eng_selfsearch` s1 on 16490 g0 at both rungs equal `selfsearch_confirm/gate_79ea002c` in every simulated key (mean latency bit-equal: heavy 2.2544 / 2.2196, moderate 2.6100 / 2.5902; every scheduler counter). The only differing keys are measured timing (`decisionTiming.*`, `ss_seconds`, `total_rtt_plus_inference`), code provenance and the model path (same md5).
  - Smoke A 864099 (21 m; 10/10, 0 failed): served `GNN_SERVE_CANDIDATE_SLATE=declared_pruning_v1_uncapped` and the loader's slate check passed; `slate_declared_pruned` = 0 on every cell; `ss_batches` = `slate_declared_batches` on both self-search kinds; `slate_declared_unsplit` = `slate_declared_sub_batched` on every cell. Slowest cell 1,268 s (heavy gnn self-search), so `LIMIT` = max(8,100, 2.5 × 1,268) = 8,100 s. Smokes are not results.
  - Gate A 864114 (`--array=0-5%6`, PAR 24, LIMIT 8,100; 5 new kinds × seeds 1–2 under the uncapped rule; 480 cells) and gate B 864115 (`--array=0-1%2`, PAR 24, LIMIT 8,100; `cd_exactS`, `cd` and old self-search / no-split seeds 1–2, `R1A_SLATE` unset; 288 cells), at 4d90119c on 19420–19431 × g0–g1 × moderate/heavy. OUT `/share/nikola.lukic/slate_fix/gate_{A,B}_4d90119c`. Watchdog kills and failed cells are counted, never rerun. Then the single read with `scripts_cosim/slate_fix_v1_gate_read.py --new gate_A… --old gate_B…`.
- 2026-10-12 — **Offline screen (863952; ordering only, never a bar) and smokes submitted.**
  - Val `regret_masked_topo` (s) on the uncapped cache's 438 held-out tables, seeds 1 / 2 (mean); plan_exact in brackets:

    | arm | new (uncapped-trained) | old (capped g4, scored over the full set) |
    |---|---|---|
    | gnn_eng | 0.9633 / 0.7720 (0.868) [0.591 / 0.605] | 1.0625 / 0.8560 (0.959) [0.594 / 0.568] |
    | twin_eng | 0.9163 / 1.0645 (0.990) [0.580 / 0.580] | 0.8380 / 0.7824 (0.810) [0.573 / 0.559] |

  - Directions only (2 seeds): the new GNN is ahead of the old GNN on the new labels by ~0.09 s on both seeds; the new twin is behind the old twin on both seeds; the new GNN is ahead of the new twin on the mean, while the old twin was ahead of the old GNN. The new checkpoints reproduce their training best_val exactly through this path.
  - Smokes on 16490 g0, both rungs, at 4d90119c, `SMOKE_IN=/share/nikola.lukic/slate_fix/smoke_inputs`: **B** 864098 (`cd_exactS` + old `ra_gnn_eng_selfsearch` s1, `R1A_SLATE` unset, 4 cells; identity against `selfsearch_confirm/gate_79ea002c`) and **A** 864099 (the 5 new kinds s1 under `declared_pruning_v1_uncapped`, 10 cells).
- 2026-10-12 — **Training done; gate inputs built and wired.**
  - Training 863951: gnn_eng 2 h 04 m, twin_eng 1 h 19 m; 0 failures; 100/100 epochs each; post-training sidechecks PASS.
    - best_val regret_masked_topo on the UNCAPPED val labels (not comparable to the capped 0.7807; the screen compares like with like): gnn_eng 0.9633 / 0.7720, twin_eng 0.9163 / 1.0645 (seeds 1 / 2); last improvement at epochs 96 (gnn s2), 90 / 82 (twin s1 / s2).
    - md5 (.pt): gnn s1 8bbaead3, s2 ff7b4946; twin s1 2c609b75, s2 7aa3a51c.
  - Screen 863952 started on the `afterok` (tasks 0–7, 4 at a time).
  - Gate inputs 864094 (34 s): `s160_gate_inputs.sbatch` with ib0_confirm_v1's arguments (M_MODERATE 11.6139, M_HEAVY 27.622665025860204, REPLICA_RULE fastest_compatible, BASECFG `accel_nosplit/inputs/cfg_src/cc40s16345.json`) and `SELECTION=scripts_cosim/slate_fix_v1_selected.json`, into `/share/nikola.lukic/slate_fix/inputs_12/inputs`; split copied in (sha f5bc4f51). No overlap with sparse_obs_s0_v1's 19440–19451.
  - `slate_fix_v1_gate_models.sh` wired 12 model links (old gnn md5 0f915a36 / 446f99fd confirmed; 6 sidechecks PASS) and `SMOKE_IN=/share/nikola.lukic/slate_fix/smoke_inputs`.
- 2026-10-12 — **Identity PASS** (job 863950, 43 s): the old gnn_eng seed-1 checkpoint through the eval-only path on its own capped g4 cache gives regret_masked_topo **0.7807019727098458**, bit-equal to its training's recorded best_val (n = 438 val datasets; plan_exact 0.626). Training 863951 started on the `afterok`; both tasks' cache checks PASSED at 4d90119c (3,169 graphs; RTT rows 10,668,160 vs 5,227,936; split 2,674 / 438 / 57).
- 2026-10-12 — **Identity task 863945 FAILED on a launch defect, before any number:** `OSError: [Errno 24] Too many open files` in the eval DataLoader 33 s in. `slate_fix_v1_eval.sbatch` lacked `ulimit -n 65536` (datalab-pitfalls #11; the training sbatch had it). The `afterok` chain dropped 863946 and 863947, so nothing trained. Fixed in code at 4d90119c and relaunched in worktree `~/gnn-herosim-wt/sf_4d90119c`: identity 863950, training 863951, screen 863952 (`/share/nikola.lukic/slate_fix/screen_4d90119c`), same `afterok` chain.
- 2026-10-12 — **Launched, by user-ordered amendment** (relayed by the expert): training no longer waits for the `slate_bar_v1` read; the gate order is unchanged.
  - Datalab worktree `~/gnn-herosim-wt/sf_438fa161` (code 438fa161), with `models/` → `~/gnn-herosim/models` and the g4 cache path → `accel_prod/20261010_061132/cache_run/cache` (the identity task's own capped cache).
  - Jobs: identity 863945 (task 8); training 863946 (tasks 0–1, `afterok` on 863945); screen 863947 (tasks 0–7, throttle 4, `afterok` on 863946). Screen output `/share/nikola.lukic/slate_fix/screen_438fa161`.
- 2026-10-12 — **What drives the 417 + 49 improved labels** (S7; `scripts_cosim/extend_corpus_drivers.py` on rp/corpus-uncap 8e0b8803, job 863777; `accel_uncap/4f41b771/drivers_{train,heldout}.json`). A classification, not a causal test.
  - Class of the candidates the new best plan uses that the old slate dropped (train / held-out):
    - cold-mispricing, meaning every such candidate would be top-5 under the simulation's cold charge: 61.6 % (257) / 49.0 % (24);
    - mixed: 9.8 % (41) / 12.2 % (6);
    - cap, meaning outside the top 5 even when correctly priced: 28.5 % (119) / 38.8 % (19).
  - Per dropped candidate, the simulation-cost rank is ≤ 5 for 410 of 621 / 45 of 78, and 6–10 for the rest. None is beyond 10, so a top-10 slate at the correct cost would hold every new-best candidate.
  - The new best moves at least one task to another node in 95.4 % / 89.8 % of cases; the rest swap platform on the same node.
  - Co-located pairs go up in 51.3 % / 44.9 %, stay the same in 46.0 % / 46.9 %, and go down in 2.6 % / 8.2 %.
  - Label regret of the old best (old minus new, s): median 0.285 / 0.309, mean 3.02 / 3.18, p90 7.19 / 7.61, max 104.9 / 43.3.
  - Caveat: the simulation cost applies the previous-task rule only and ignores CD's "current task present → cold 0" rule (`scheduling_cost.py:73`).
  - So most of the label gain is the cold-cost defect `slate_bar_v1` named (a top-5 slate priced with the simulation's cold charge would have reached the new best in ~62 % of train cases; `declared_pruning_v2` is the partner rule, not top-5, and was not scored here). The rest needs the wider slate itself.
- 2026-10-12 — **Uncapped corpus and cache written** (S7). The fix's denominators are here.
  - Corpus: `/share/nikola.lukic/accel_uncap/4f41b771/{train,heldout}`, from `gnn_datasets_wf1_{train,heldout}`; rule `declared_pruning_v1_uncapped`, max-plans 100,000, `--link-untouched`.
  - Runs: extension at 4f41b771; check job 863723 (`scripts_cosim/extend_corpus_check.py`, `accel_uncap/4f41b771/check_{train,heldout}.json`); recache job 863752 at rp/corpus-uncap 836e469d.
  - Cache `cache_run/cache`, split `cache_run/split.json` (sha f5bc4f51, equal to production).
  - Census, train / held-out: extended 1,767 / 258; already full 910 / 171; **over-cap untouched 54 / 9** (the pre-launch 54 is confirmed); plans 1,490,170 → 6,152,545 and 347,885 → 1,125,734; cache RTT rows 5,227,936 → 10,668,160.
  - Checks: re-scored stored plans differ by 0.0 on every extended dataset; old rows are a subset of new rows everywhere; 0 problems.
  - The old best was strictly beaten on **417 of 1,767 train datasets (23.6 %)** and 49 of 258 held-out (19.0 %). S7 is running a diagnostic of what drives these (cold-defect rank change vs co-location).
  - The 556 / 92 sub-batched count is S5's (see head), not S7's.
  - This branch merged 836e469d, which adds `replica_placement_rule` and in-flight capture to the physics contract. Both configs now name the cache.
- 2026-10-12 — **Signed** by the expert at d66ab923.
  - The two-config deviation is confirmed: one checkpoint, two serving modes, as the parent did; the byte-equality test is the guard.
  - Rulings: (1) the launch gating above; (2) the untouched corpus goes in the head; (3) the eval identity task is a prerequisite and a failure is fixed in code; (4) the expert writes the LINEAGES.md row.
- 2026-10-12 — Drafted (S6) on the expert's request. Code, configs, reader and tests are on `rp/slate-fix`; nothing has run.
