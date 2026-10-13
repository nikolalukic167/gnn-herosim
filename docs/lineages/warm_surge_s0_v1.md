# warm_surge_s0_v1 — does a surge onto a warm, loaded accel cluster move the hand headroom or the GNN gap?

**Status:** `ACTIVE` (2026-10-13) — Phase 0b read; 0c (alpha diagnostic) and the uncapped-arm read pending, then a single close. Owner S5. **Standing answer so far:** the registered primary does NOT move (PRIMARY_MOVED = False at all four conditions; the oracle stays unrun), but `cd_exactS_ib0` is brittle under a warm heavy surge: 3/12 topologies at ×2 and 5/12 at ×3 run away (post-switch mean 5–55 s against 0.5–1.0 s) where `cd_exactS` collapses on 1/12 and `cd` on 0, so heavy-rung medians against ib0 are collapse artefacts. A ×2 surge also saturates reactive Knative at both rungs (47/48 cells, share ≈ 0.99). The g4 GNN falls further behind ib0 on moderate (+9.9 / +8.8 %). Registered 2026-10-12 (SCIENTIST design, signed by the coordinator with two notes applied below).
Parents read: `ib0_confirm_v1` (the bar, and its post-close time split
`ib0_confirm_v1/oracle_loss_diag.md` § Time split), `peak_load_v1`, `cd_gap_v1`, `sparse_obs_s0_v1`, ADR 0004.

## Why

Every surge rung so far started from an empty cluster. `cd_gap_v1_build_b.py:47–49` multiplies every timestamp from t = 0, and the live path autoscales from zero with
no warm-up discard (`src/executesimulation.py:5`, `:116`, `:202`; `src/placement/orchestrator.py:431`). So in `peak_load_v1` and every ×k rung since, the surge and the
scale-from-zero opening were one event. The `ib0_confirm_v1` time split puts 84 % (heavy) / 68 % (moderate) of the oracle's label room over the hand bar in that opening.
This node asks what is left when ×k arrives after the cluster has already run warm at ×1.

**Primary question.** Hand headroom under a warm surge:
- (i) post-switch `cd_exactS_ib0` vs `cd_exactS`;
- (ii) a 4-cell label oracle vs ib0, run only if (i) shows that the surge moves anything.

The current learned arms are descriptive.

## Seat and cells

- **Seat:** the accel seat, 160 clients × 24 servers.
- **Physics:** R1.1 (pipelined transfers, replica release, KPA). `GATE_FIXED_POLICY_TIME_SCALE=1.0` (`scripts_cosim/fresh_topo_burst_v1_gate.py:389`, `:765–766`); batch window fixed at 1.0 s (`inputs_12` manifests).
- **Cells:** topologies 19420–19431, window g0.
- **Base rungs:** exact factors, never roundings.
  - heavy: m = 27.622665025860204, factor 0.03620215497178874;
  - moderate: m = 11.6139, factor 0.08610372054176461.
- **Source files:** `/share/nikola.lukic/slate_fix/inputs_12/inputs/wf1_{heavy,moderate}/wl/grounded_g0_n50000.json`.

**k ∈ {1, 2, 3}, and k = 1 is required.** Every `gate_B` / `gate_B2` summary on 19420–19431 predates the c81f0e88 summarizer. `steady_state` is None in all of them, and the
per-task raw files are deleted, so no ×1 steady-state number exists on this seat. The k = 1 cells measure it on the same post-switch task window.

## Warm-up W = 1200 s on both rungs (wall time, not a task share)

- **Replica settle** ([V], `replica_count_series` of ib0 and exactS at g0, 24 cells per rung). Settled means the total first stays within max(3, 2σ) of the second-half median for 6 grid points.
  - heavy: median ~365 s, max 498 s;
  - moderate: median ~470 s, max 1,564 s (one cell; the 78 s grid is coarse).
  - KPA overshoots to a 120-replica peak in the first grid points, then decays to a second-half median of 28–40, with σ ≈ 8–11. The replica series is therefore a poor clock.
- **Latency settle** ([V], 16490 cd, c81f0e88 `.tasks.npy`, 60 s bins).
  - heavy reaches its second-half level (1.35 s) by 240–300 s. The bins read 56.9 s at 0–30 s, 2.3 s at 180–240 s and 1.05 s at 300–360 s.
  - moderate settles by ~180–360 s (noisy).
- **Why 1200 s.** The rule "settle + 360 s" gives 858 s (heavy, max) and 830 s (moderate, median). 1200 s covers every cell except the one moderate outlier and leaves 20 KPA stable windows of warm history.
- **Task counts.** About 15.3k tasks arrive before the switch at heavy (12.71/s) and about 6.4k at moderate (5.35/s).
- **Post-switch spans:**

  | rung | post-switch tasks | ×2 | ×3 |
  |---|---|---|---|
  | heavy | 34.7k | 1,366 s | 911 s |
  | moderate | 43.6k | 4,074 s | 2,716 s |

  This is a sustained step, like Shahrad's peak hour, not Luo's minute surge. The onset secondary covers the minute scale.

## Time constants

Time constants are unchanged at the switch and across rungs.
- R1 already pins policy time scale 1.0 on every rung (`physics_audit_v1` B1). The gate fails loud otherwise (`fresh_topo_burst_v1_gate.py:985–988`).
- KPA keeps stable window 60 s, panic window 6 s, target 0.7. The batch window stays at 1.0 s.
- `cd_gap_v1_build_b.py`'s convention, which scales `batch_timeout` with the factor, is NOT applied here. A real cluster does not retune at a surge.

## Workload change

`scripts_cosim/grounded_workload_v1_mint.py` stays frozen: its ×1 stretch is the grounded part.
- **Code.** Add `surge_after(wl, t_switch_s, k)` and `--surge-at-s` / `--surge-k` (default off) to `scripts_cosim/cd_gap_v1_build_b.py`, applied as a post-pass to the frozen rung files.
- **Rule.** Per peer group, take the anchor a = the earliest member timestamp.
  - If a < t_s, the group is unchanged.
  - Otherwise every member moves to t' = t_s + (t − t_s) / k. Offsets compress as build_b compresses them, and no group straddles the switch.
- **Workload file.** Recompute `duration` and record `warm_surge_s0_v1` = {t_switch_s, k, n_pre, n_post, source_sha256}.
- **cfg.** Copy it unchanged. In particular, do not write `cd_gap_v1_rate_scale`: the gate reads that factor (`fresh_topo_burst_v1_gate.py:762`) and derives the time scale and `HEROSIM_PG_EXT_RATE` from it.
- **Output.** The output directories are `wf1_heavyk{1,2,3}` and `wf1_moderatek{1,2,3}`, served with `WF1_RUNGS=heavyk1,…`. The ladder maps g{i}{tag} to `wf1_{tag}` (`fresh_topo_burst_v1_gate.py:184–186`). k = 1 is a byte copy of the source.

## Statistic

**Primary:** mean latency (done − dispatched) over the tasks with dispatchedTime ≥ t_s, read from `<cell>.tasks.npy`.
- dispatchedTime is the event timestamp (`src/placement/orchestrator.py:841`), so this means the tasks that arrived after the switch.
- `after_360s` and `after_p5_dispatch` are fixed cuts unrelated to the switch, so neither is the primary.
- Optional: a one-line summarizer key `steady_state.after_switch` that reads t_switch_s from the workload.

**Secondaries:**
- onset, [t_s, t_s + 300);
- settled, [t_s + 360, end);
- pre-switch warm, [600, t_s);
- post-switch split of cold start, pull and queue.

**Contrasts** are paired per topology: the median of the per-topology % and the sign count n/12. Each is quoted as Δ(k) − Δ(k = 1) on the same task window.

## Arms

- **Classical, seed 0:** `cd`, `cd_exactS`, `cd_exactS_ib0`.
- **Descriptive, checkpoint seeds 1–2:** the new SS, NS and twin-NS arms (S5 maps them to arm keys).
- That makes 9 arm-runs per cell.

## Phase 0a: admissibility calibration

- **Cells:** cd plus reactive, on 12 topologies × the 4 surge conditions = 96 cells, about 1.5 CPU-h.
- **Rule:** a (rung, k) condition enters Phase 0b if at least 10 of its 12 cells pass all of:
  - reactive post-switch queue share ≤ 0.80 (the record's rule; unknown is not a pass);
  - cd post-switch backlog not growing: the mean in-system count over the last quarter of the post-switch window is ≤ 1.5× the second quarter (computed from the npy);
  - end time − last arrival ≤ 120 s;
  - `requestFailures` = 0.
- **Load at heavy ×1** (the nearest [V] measures):
  - effective queue share 0.47–0.62 (ib0) and 0.35–0.57 (cd); queue share 0.14–0.25;
  - 19420 ib0 alone logs 766 failed scale-ups with cause "load" and peaks at 120 replicas, so platform capacity already binds at times at ×1.
- **The "76 %" figure** is `decision_cost_s0_v1`'s one-thread decision-wall utilisation of `cd_exactS` at heavy (decision wall time ÷ simulated time). It is not a queue share and does not enter admissibility.
- **Phase 0b runs without ×3 heavy unless 0a admits it** (coordinator's note b).

## Phase 0b: gate and cost

- **Cells:** 2 rungs × k ∈ {1, 2, 3} × 12 topologies × 9 arm-runs = 648 cells, or 540 without ×3 heavy.
- **Wall time per cell** ([V], ×1, g0, median, heavy / moderate):

  | arm | heavy | moderate |
  |---|---|---|
  | ib0 | 2,057 s | 1,782 s |
  | exactS | 1,960 s | 1,759 s |
  | cd | 46 s | 46 s |
  | SS | 1,142 s | 1,077 s |
  | NS | 251 s | 266 s |
  | twin-NS | 224 s | 254 s |

- **Total:** about 2.0 CPU-h per (topology, condition), so about 146 CPU-h for all six conditions and about 120 CPU-h without ×3 heavy. Allow a ×1.5 margin. [M] Because batches are per peer group, `cd_exactS` / ib0 cost per batch should scale with group size rather than rate; smoke one ×3 cell first to check. That is one night on the CPU partition.

## Kill rule

The surge is NOT a lever, and the lineage stops, if both of these hold at every admissible (rung, k):
1. |Δ ib0-vs-exactS (k) − Δ (k = 1)| ≤ 1 pp;
2. |GNN deficit vs ib0 (k) − (k = 1)| ≤ 1 pp, using the better of SS and NS (seed mean).

A move counts only if it exceeds 1 pp AND is sign-consistent in at least 9 of 12 topologies, because 1 pp is near paired noise. [M] There is no ×1 paired SD on this window yet.

If (1) moves, run the 4-cell label oracle vs ib0 on the condition that moved (`oracle_diag` on post-switch batches only). That orders the work; any learned follow-up ends with a live gate on this same surge seat.

## Priors, written before data

| contrast | prior |
|---|---|
| (i) ib0-vs-exactS lead moves > 1 pp from k = 1. Zero stacking weight should matter more when queues build, but ib0 was picked at ×1 heavy, so the direction is open. | ~45 % |
| GNN deficit vs ib0 moves > 1 pp | ~50 % |
| GNN beats ib0 post-switch at any admissible k | ~15 % |
| The surge passes the kill rule (something moves, sign-consistent) | ~55 % |
| Warm-surge label-oracle headroom ≥ 5 % over ib0, if run | ~15 % |
| twin-NS vs NS (the MP contrast) moves > 1 pp | ~30 % |
| ×3 heavy inadmissible | ~75 % |
| ×2 heavy inadmissible | ~30 % |
| ×3 moderate inadmissible (16.1/s, above heavy ×1's 12.7/s) | ~25 % |

## Leak checks (fail loud)

1. **Workload.** Events whose group anchor is before t_s are byte-identical to the source. Post-switch inter-arrival times equal the source's ÷ k (max abs error ≤ 1e-9). The type, QoS, client, `peer_group` and `peer_exchange` sequences are identical. No group straddles t_s. The sha chain is recorded.
2. **Pre-switch identity.** For each deterministic arm, the per-task (dispatched, done) of every task with done < t_s is bit-identical across k = 1, 2, 3. This proves the transform touched nothing before the switch.
3. **k = 1 whole-run parity.** k = 1 `averageElapsedTime` for ib0, exactS and cd equals the `gate_B2` / `gate_B` value per topology. If it does not, physics moved between d192081f / 4d90119c and HEAD: report it and do not proceed.
4. **Time constants.**
   - `policy_time_scale` is 1.0 and `HEROSIM_POLICY_TIME_SCALE` is unset or 1.0;
   - `scaleOut` keeps stable 60 s and panic 6 s; `batch_timeout` is 1.0;
   - the cfg sha equals the source's;
   - no served arm reads the rung multiplier. Grep the served arms for `rate_scale` and `HEROSIM_PG_EXT_RATE`; the gate sets the latter only for cdext / cdextr (`fresh_topo_burst_v1_gate.py:869`), which are not in this design.
5. **Warm at the switch.** Total replicas over [t_s − 300, t_s) lie within the ×1 second-half band, and the backlog at t_s is ≤ the ×1 median. If not, W was too short for that cell: flag the cell, do not drop it.
6. **Post-switch cold starts are physics, not a leak.** Report the cold and pull share of the onset window separately, so that an onset gain is attributed to new-replica colds or to queueing.
7. **Completeness.** `num_tasks` = 50,000, `requestFailures` = 0 and `task_times_rows` = 50,000 on every cell.

## Record

- 2026-10-12 — Registered (SCIENTIST design; signed by the coordinator as drafted). Coordinator's notes applied: the "76 %" is `decision_cost_s0_v1`'s decision-wall utilisation, not a queue share; Phase 0b runs without ×3 heavy unless Phase 0a admits it. Not yet implemented.
- 2026-10-12 — **Implemented** (S5, rp/warm-surge). Mint: `surge_after` and `--surge-at-s/--surge-k` in `cd_gap_v1_build_b.py`, t_s = 1200 s, inputs `/share/nikola.lukic/warm_surge/inputs/wf1_{heavy,moderate}k{1,2,3}` (heavy n_pre/n_post 15,281/34,719; moderate 6,568/43,432). Leak checks 1–3 passed (`warm_surge_checks.py`): workload (no straddling group, offset error 9e-13), pre-switch identity for `cd` on 19420–19422 (ib0/exactS repeated in 0b), k = 1 `cd` equal to `gate_B` on 12 topologies, both rungs. A harness fault was found and fixed on the way (`run_one` rebound the cell name; `docs/gates/gate-tools.md`, e3d5e069); the 36 check cells that died on it were relaunched into a fresh OUT.
- 2026-10-12 — **Phase 0a read** (job 864316, e3d5e069, 96/96, 0 failed): every condition INADMISSIBLE under the registered rule because reactive saturates — whole-run effective queue share ≈ 0.99 in 47 of 48 cells (heavy ×2, ×3 and moderate ×3 0/12; moderate ×2 1/12). cd is healthy (backlog ratio ≤ 1.5 in 47/48, tail ≤ 120 s in 48/48, 0 failures). **Result in its own right: a ×2 surge saturates reactive Knative at both rungs.**
- 2026-10-12 — **Amendment (coordinator's ruling): Phase 0b admission is cd-healthy.** Knative is in neither the primary (ib0 vs exactS) nor the secondary (GNN vs ib0) family, so the reactive criterion guards nothing in 0b. Per cell: cd post-switch backlog ratio ≤ 1.5, end − last arrival ≤ 120 s, requestFailures 0; a topology enters a condition only if healthy at that k and at k = 1; a condition stays in the read only with ≥ 10/12 topologies. Knative is dropped from 0b. All four conditions are run (including ×3 heavy).
- 2026-10-12 — **Amendment (coordinator's ruling): learned arms are the accel g4 checkpoints on `declared_pruning_v1`** (`ra_gnn_eng_selfsearch`, `ra_gnn_eng_nosplit`, `ra_twin_eng_nosplit`, seeds 1–2), not the slate-fix-v1 uncapped retrains. Reason: `slate_fix_v1` showed the uncapped retrain does not change the GNN's standing (+2.6 / +5.3 % vs ib0 either way), and the g4 arms have a k = 1 reference in `gate_B`. If the secondary contrast moves > 1 pp on any condition, the slate-fix-v1 uncapped arms are rerun on that condition only (seeds 1–2) before any claim.
- 2026-10-12 — **Phase 0b launched** (array 864320, pin e3d5e069, 576 cells: `cd_exactS_ib0`, `cd_exactS`, SS, NS, twin-NS; `cd` reused from 0a and the check run, same pin; out `/share/nikola.lukic/warm_surge/gate_0b`). First ×3 heavy ib0 cell wall 1,987 s. Reader `scripts_cosim/warm_surge_0b_read.py` written before the data lands.
- 2026-10-13 — **Phase 0b read** (arrays 864320 / 864452 / 864474, pin e3d5e069, 576/576, 0 failed; reader `warm_surge_0b_read.py` cbdf7866, `read_0b.json`). Leak check 2 (ib0, exactS, cd and the learned arms across k): 216 series, 0 failures. Only 19424 at heavy ×3 fails cd health. **Registered outcome: PRIMARY_MOVED = False** at all four conditions (median Δ(k) − Δ(1) of ib0 vs exactS: heavy ×2 +5.57, ×3 −0.11; moderate ×2 −1.95, ×3 +4.06 pp, none sign-consistent in ≥ 9/12). The oracle is not run: (i) did not move by the signed rule, and the rule is not relaxed after the data. The GNN secondary moves on moderate ×2 and ×3 (best GNN deficit vs ib0 +9.91 / +8.79 %, Δ +5.05 / +3.48 pp).
- 2026-10-13 — **Headline: `cd_exactS_ib0` is brittle under a warm heavy surge.** 3/12 topologies at ×2 and 5/12 at ×3 run away (post-switch mean 6–55 s against 0.6–1.0 s; end − last arrival up to +1,094 s) where `cd_exactS` collapses on 1/12 and `cd` on 0 (the collapse denominators are all 12 topologies; the 11 of the heavy ×3 median read is the cd-health admission count, which drops 19424 for the Δ(k) − Δ(1) statistic and is a different denominator) (full per-cell table: `docs/lineages/warm_surge_s0_v1/collapse_table.md`). Moderate: no collapse. Wherever ib0 does not collapse it is the fastest arm, as at ×1. **The heavy-rung medians against ib0 (cd −32 %, SS −39 % at ×3) are collapse artefacts; they are never quoted as a GNN or cd win.** Because the bar is one arm and it breaks, every heavy-surge GNN contrast is also read against `cd_exactS` and `cd` (descriptive, data already in hand). The median paired rule is blind to a 25–45 % collapse tail: `docs/gates/gate-tools.md` carries the collapse-count rule.
- 2026-10-13 — **Phase 0c registered before launch (coordinator's ruling): is it the batch alpha?** Mechanism guess, untested: ib0 zeroes the in-batch stacking weight (`HEROSIM_PG_BATCH_ALPHA = 0`), and under a long queue its batch piles onto the platform that looks cheapest. Arms: `cd_exactS` at batch alpha 0 (`cd_exactS_ib0`), 0.25 (`cd_exactS_a025`, new kind), 0.5 (`cd_exactS_a05`) and 1.0 (`cd_exactS`) on heavy ×2, topologies 19422, 19425, 19430 (the collapse cells) and 19420 (the control: the median non-collapsing topology by ib0 post-switch mean, 0.736 s, picked from `read_0b.json` before launch). Same warm ×1 → ×2 protocol, pin e3d5e069 + the new kind.
  - **Deviation, disclosed before launch:** the ruling says seed 1; every classical arm in this lineage runs at seed 0 (the scheduler does not read the seed), so alpha 0 and 1.0 are the existing 0b cells and only alpha 0.25 and 0.5 are new: 8 cells, not 16.
  - **Prediction, signed now:** the collapse disappears (no COLLAPSE flag: post-switch mean ≤ 3× the cell's cd mean, per the gate-tools rule) at some alpha < 1 on all three collapse cells, and the control's post-switch mean at alpha 0.25 and 0.5 is not more than 1 pp above its alpha 0 value.
  - **Kill:** if alpha 0.25 and 0.5 each still collapse on ≥ 2 of 3 cells, the mechanism guess is wrong; record BRITTLE-CAUSE-UNKNOWN and stop. No further diagnostic.
  - **Per-platform dispatch table (first 60 s post-switch, alpha 0 and 1):** the summary and the per-task npy carry no per-platform assignment (the raw result is deleted with the cell), and no new instrumentation is added to the live path, so the table is skipped.
  - Reader `scripts_cosim/warm_surge_extra_read.py alpha`, written before the data.
- 2026-10-13 — Slate-fix-v1 uncapped arms (SS, NS, twin-NS, seeds 1–2, `R1A_SLATE=declared_pruning_v1_uncapped`, `R1A_STEM=slate-fix-v1`) on moderate ×2 and ×3 only, 144 cells (array 864497), approved by the coordinator; read with the same three hand references (ib0, exactS, cd) by `warm_surge_extra_read.py collapse --uncapped`.
