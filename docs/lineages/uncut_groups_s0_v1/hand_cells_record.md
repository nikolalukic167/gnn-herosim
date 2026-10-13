# uncut_groups_s0_v1 — hand-cell record (S5; S6 files this into the node's hand-cells section)

## Smoke, 2026-10-13 (totals only)

- Pin f36f293f (rp/uncut-groups, code identical to 6d1764bd), inputs `/share/nikola.lukic/uncut_groups/gate_inputs_e2738e37/inputs_group` (cfg `scheduler.batch_size: "group"`), topology 19461, window g0, k = 0, seed 0. Job 864578; scratch `/share/nikola.lukic/uncut_groups/smoke`. Env set explicitly: `R1A_TOPOS`, `R1A_WINDOWS=g0`, `WF1_RUNGS=moderate,heavy`, `R1A_SEEDS=0`; `R1A_SLATE` / `R1A_STEM` / `R1A_SPLIT` / `R1A_SNAPSHOT_PLAN` unset.
- `cd` cells (heavy / moderate): the uncut path serves whole groups.
  - `pgb_whole` 11_50 / 51_100 / gt100 = 12,852 / 4,534 / 922 (heavy) and 12,840 / 4,534 / 922 (moderate); `pgb_split` = 98 / 16 / 0 and 302 / 114 / 115; `pgb_max_batch` = 199, the gate window g0's largest group.
  - Whole-group share of tasks in groups > 10: 99.5 % (heavy), 98.4 % (moderate) on this cell, `cd` arm (K0's bar-arm share is read on the k = 0 array, not here).
  - `averageElapsedTime` 2.163 / 2.702 s; `peer_group_incomplete_batches` 11 / 15.
- Decision wall per plan, **cell totals only** (`decisionTiming.total_s` ÷ `sched_plan_evals`): `cd` 0.0074 ms (heavy) and 0.0080 ms (moderate) against the calibrated `w` 0.00822 / 0.00971. **This is not the registered table**: the summaries carry no per-size-bucket decision time. A per-bucket counter (calls, decision wall, plan evals by served batch size ≤ 10 / 11–50 / 51+) is requested from S6; the registered 1.5× check on w(n) runs on it.
- `cd_ib0` cells finished; the exact-class cells (`cd_expand`, `cd_exactS`, `cd_exactS_ib0`) were still running when this was written.

## Rules for the hand cells (ruled by the expert, 2026-10-13)

- k = 0 array at pin f36f293f; k = 1 array only after the per-bucket counter pin and the w(n) table are seen. If a bucket exceeds 1.5× the calibrated `w`, S6 freezes `w(n)` per bucket in `decision_cost_w.json` before any k = 1 cell; otherwise the scalar `w` stands and that is recorded.
- The two pins differ (k = 0: f36f293f; k = 1: the counter pin) only because the counter is measurement-only. One smoke cell at the counter pin must be SIM-IDENTICAL on every non-counter key to f36f293f before the pin is used; both pins and that identity are recorded here.
- The prune arm waits for S6's report that identity (4) (job 864550) passed.
