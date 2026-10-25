# knative_stable_v1 — at an arrival rate where reactive Knative is stable under option D, does the static link-feature GNN beat reactive, the CD bar and the parameter-matched MLP?

**Status:** `CLOSED` (2026-10-25) — **STOP-NO-STABLE-RATE**. Registered 2026-10-25 (SCIENTIST, the user's "go A", before any calibration or study cell); the rule was fixed before the data and not edited. The user ruled STOP: no payload fallback, no 1/64 rung.
- **Outcome.** Under option D, reactive Knative is not stable at any rate from moderate down to 1/32 of it. Topologies passing (of 12, g0 and g1 both stable): f0.5 0, f0.25 0, f0.125 0, f0.0625 6, f0.03125 10; the rule needs 11. The failure is the transfer clause (fair-share slowdown > 3×) with drain ≈ 0 and queue_share ≤ 0.1 at the lightest rates; at f0.03125 two cells miss by a hair (30289 g0 3.2×, 30292 g0 3.1×), every other cell sits at 2.3–2.9×. No study was minted or run; ids 30353–30376 were never minted and are released. f0.5 was cancelled at 6 of 24 cells; its unfinished cells count as fail (unknown is not a pass) and the verdict does not depend on them, since 0.5 could at best add one passing level and f0.25 and f0.125 pass none.
- **Table.** `knative_stable_v1/kstab_rule_final.json` (per cell: drain, slowdown, queue_share) and `.txt`; reader `kstab_rule.py` (reads only the three rule fields).
- **Question.** Under option D, `reactive` saturates the links at the moderate and heavy rungs. Find the highest lighter arrival rate at which reactive is stable. At that rate, does the static e40 gnn beat reactive, the CD bar `cd_ib0_inflightfull_k100` and the wide MLP?
- **Why.** In `llf_cd_confirm_v1` the reactive and random rows read about −99.9 % and are not quotable. S7's diagnosis: about 124k transfers per cell at about 2,000 s each; a peak of 5,783 flows on one link; queue share about 1e-4; the request timeout does not cover transfers (`src/placement/request_timeout.py:1–3`).
  - SCIENTIST checked one reactive cell, 30327 g0 moderate:
    - drain ratio (end − last arrival) / last arrival = 5.11;
    - fair-share slowdown 621× (`flow_seconds` 280,186,053 s over a solo time of 451,182 s);
    - `peak_flows_on_a_link` 5,954.
- **Held-out load.** The trained models never saw this rate. It is a load outside their training range, and every row says so.

## Lever

- **Lever: a lighter arrival rate.** Payloads, links, physics and features stay as served in `llf_cd_confirm_v1`. Every column the models read stays in its trained kind.
- **The knob.** The mint's rate multiplier: `scripts_cosim/datalab/big_groups_inputs.sbatch:45–47`, `--rung <tag>=1/m` into `workload_fix_v1_build.py`. The window timestamps are divided by m. A rung at fraction f of moderate is m = f × 11.6139, tagged `f<f>` (for example `f0.125`).
  - The script hard-codes the moderate and heavy tags at :45 and in its checks. S5 parameterises the tag list; nothing else changes.
  - `r1a_gate.sbatch` takes any tag through `WF1_RUNGS` (`fresh_topo_burst_v1_gate.py:551`).
- **Fallback, only on STOP-NO-STABLE-RATE: a payload scale.** `PAYLOAD_SCALE` at `big_groups_inputs.sbatch:41` → `--payload-scale` of `workload_fix_v1_build.py`, applied before the cap200 clip. It moves the payload columns out of their trained kind, so it needs its own amendment.
- **Not already answered.** `payload_scale_v1` (archive, CLOSED 2026-09-20) ran at 6 servers and 0.46 arrivals/s, with no fair share or TCP. It does not answer this, and it is not quoted (no cross-era comparisons).

## Stability rule (reactive, per cell; fixed now)

A reactive s0 cell is **stable** only if all three hold. Every value is read from the cell's summary:
1. **Drain:** (`endTime` − `arrival_end.last_arrival_s`) / `arrival_end.last_arrival_s` ≤ 0.10.
   - The span is taken from t = 0. The first arrival is at 2.5 s against a span of 9,354 s at moderate g0 (0.03 %), and it scales with the rate.
2. **Transfers:** `peerExchangeFairshare.flow_seconds` / (`flow_seconds` − `extra_seconds`) ≤ 3.
   - This is the seconds-weighted mean slowdown over the cell's remote exchanges.
   - The denominator is the summed **unshared (solo) time**: for each exchange, the TCP start-up delay + payload / (route bottleneck × 1024²) + route latency, as booked by `record_exchange_fairshare` (`src/placement/infrastructure.py:1389–1392`, `network_fabric.py:514–521`).
   - The expert's draft asked for a median over transfers. The summary carries only these sums, so the mean ratio is the rule.
3. **Admission:** `queue_share` ≤ 0.80 (the parent's rule).

Unknown (missing, failed, killed twice, a missing field) is not a pass. A killed cell is rerun once with a longer wall, as in the parent.

## Calibration (on the spent sealed 12; reactive only)

- **Cells.**
  - `reactive` s0 at pin `0ec7cdcb`, the D flags, cap200 inputs reminted at each rate.
  - The sealed 12 of `live_link_features_v1` (30285–30287, 30289–30297).
  - g0 + g1, f ∈ {0.5, 0.25, 0.125, 0.0625, 0.03125} of moderate, so 120 cells.
  - f = 0.03125 is added to the expert's four. It costs 24 cells and saves a round trip if 0.0625 fails.
- **Rule.**
  - A topology passes a rung if both its g0 and g1 cells are stable.
  - **The chosen rung is the highest f at which ≥ 11 of 12 topologies pass.**
  - If none does: **STOP-NO-STABLE-RATE**, and the payload fallback needs an amendment.
- **What is read.** Only the three rule fields of the reactive summaries; no other arm runs. SCIENTIST checks the read before the study mint.

## Study

- **Topologies: 24 newly minted ids**, the lowest free run of 24 ≥ 30351. They are disjoint from every list in `live_link_features_v1`, `llf_cd_confirm_v1` and `uncut_groups_s0_v1`, from fresh6, from the sealed 30285–30297 and from the accel ids. The mint follows the parent recipe at the chosen rung only, and the ids and sha256 are filed before any cell.
  - *Changed from the expert's draft:* 30327–30350 are not reused. Their primary was read on 2026-10-25, so a topology's lean toward one arm is already seen.
- **Cells.** Option D, g0 + g1, the chosen rung:
  - the static e40 `-final.pt` gnn, wide MLP and same-input MLP at seeds 1–6, the same files and md5s as `llf_cd_confirm_v1` (Arms and A2);
  - `cd_ib0_inflightfull_k100`, `reactive`, `random` and `heft` at s0, pins static `02668ecf` and s0 `0ec7cdcb`;
  - 22 runs × 24 × 2 = 1,056 cells.
- **Cell admission.** A (topology, window) enters every contrast only if that window's reactive study cell is stable by the rule above. A topology with no admitted window is dropped, and the count is reported.
  - This conditions on the reactive arm. It removes the cells where reactive saturates, so it can only shrink the gnn's margin over reactive.
- **Decision charge.** No arm has a w at the new rung. Each arm is charged its filed **moderate** w at the new rung:
  - learned arms by bucket, as in the `llf_cd_confirm_v1` table;
  - bar 0.0116 ms; heft 0.003035 ms;
  - reactive and random carry none.
  - S4 files these as named copies in `decision_cost_w.json` before any study cell. A disclosed copy, not a measurement.

## Statistic, family and labels

- **Statistic.** As `llf_cd_confirm_v1`: `confirm_rulings.json` (md5 `b609acef33b10e7369a0f5aa58b42477`), a seed-mean learned cell over seeds 1–6, the per-topology median over the admitted windows, the median and count over topologies, and the exact two-sided Wilcoxon.
- **Primary family:** 3 tests at the chosen rung, Holm over 3:
  1. gnn vs reactive;
  2. gnn vs bar;
  3. gnn vs wide MLP.
- **Reading.** Read without runaways, with the all-cells sign guard.
- **Labels per test:**
  - **CONFIRMED:** median < 0, Holm p < .05, and the all-cells median < 0.
  - **DIRECTION:** median < 0, not CONFIRMED.
  - **NOT-FASTER:** median ≥ 0, or no topology left.
- **Descriptive:** gnn vs same-input MLP, heft and random; bar vs reactive; seed blocks s1–3 and s4–6; the after-360 s row; runaway and admission counts per arm.
- **Not an MP claim.** There is no twin.

## Reader

- S5 extends `llf_confirm_read.py` cd mode:
  - a `--rung` parameter: one rung, and the family of 3;
  - the new rung's charge table (the moderate copies above);
  - `heft` as a descriptive s0 arm;
  - cell admission computed by the reader from the reactive summaries, by the rule above, with the admitted list recorded in the output.
- The default mode and the `llf_cd_confirm_v1` invocation must reproduce their filed outputs except `reader_sha256`.
- SCIENTIST freezes the blob before the study read. The read runs once.

## Prior (SCIENTIST)

- P(the calibration finds a stable rung) ≈ 0.6. Reactive's transfers sit at 621× solo at moderate, so the rate has to cut concurrency by roughly two orders of magnitude.
- **Given a rung, P(CONFIRMED):**
  - gnn vs reactive ≈ 0.7;
  - gnn vs bar ≈ 0.4: at a light load the arms' choices matter less, and the margins shrink;
  - gnn vs wide ≈ 0.4.

## Record

- 2026-10-25 — **Reader frozen: `scripts_cosim/llf_confirm_read.py` blob `cbf7fb08e7c60ea228c2a33959f746d521eeb2c4`** (SCIENTIST, before any study cell is read; one freeze for `knative_stable_v1`, `scale_d_v1` and `mp_confirm_v1`). The rulings file is `live_link_features_v1/confirm_rulings.json` (md5 `b609acef33b10e7369a0f5aa58b42477`). Both filed reads are reproduced in every key but `reader_sha256`: `read_primary_A_epoch40.json` (default mode) and `llf_cd_confirm_v1/reads/read_primary_cd.json` (its frozen cd invocation). Tests: 40 passed, 1 skipped (that regression, run directly). An extended read records its options in `options`; a default read carries no new key. Invocation: `--mode cd --arms gnn,wide,mlp,bar,reactive,random,heft --primary gnn:reactive,gnn:bar,gnn:wide --rungs <chosen f tag> --charge-rung <chosen f tag>=moderate --admit-reactive` plus `--cells`, `--ids`, `--ids-sha256` and `--pin static=02668ecf,bar=0ec7cdcb`. The admission is computed in the reader by the stability rule, and the admitted and excluded (topology, window) lists are written to the output. The HEFT cell kind is `heft_k100`; a different name fails loud as an unexpected cell.
- 2026-10-25 — **CLOSED: STOP-NO-STABLE-RATE** (S5 ran the calibration, 5 levels × 12 sealed topologies × g0+g1 = 120 reactive s0 cells, pin `0ec7cdcb`, option D flags, cap200 inputs reminted at each rate: minted rates 2.6735, 1.3367, 0.6684, 0.3342 and 0.1671 tasks/s against moderate 5.347; jobs 906424–906428, mints 906409–906423). The rule was applied once to the files (`kstab_rule.py` md5 `d9e429442c2d546f2fb9b34a08a43360`, output `kstab_rule_final.json` md5 `ac343725ebef6971a5448e320f46126b`): 0 / 0 / 0 / 6 / 10 of 12 topologies stable at f = 0.5 / 0.25 / 0.125 / 0.0625 / 0.03125; no level reaches 11. At f0.125 and above the slowdown is 3–265×; at f0.0625 six topologies sit at 3.4–4.8×. The user ruled STOP and recorded as a finding: no payload-scale fallback and no 1/64 rung. f0.5 (906424) was cancelled at 30 min with 6 of 24 cells done; the unfinished cells are recorded as fail. Study ids 30353–30376 are released, never minted. The reader and study arms below were never used.
- 2026-10-25 — **Registered** (SCIENTIST; the user's "go A", relayed by the expert; before any calibration or study cell).
  - **Changes from the expert's draft:**
    - the stability transfer clause is a mean ratio from the summary sums, not a median;
    - queue share ≤ 0.80 is kept in the rule;
    - f = 0.03125 is added to the calibration;
    - the study gets 24 new ids, not 30327–30350;
    - new-rung charges are moderate-w copies;
    - the reader applies admission itself.
  - **Checked by SCIENTIST:** the fields and their definitions in code at the lines cited; one reactive summary (30327 g0 moderate) for the field names and the numbers in "Why"; the first arrival time in the moderate g0 workload.
