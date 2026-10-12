# warm_surge_s0_v1 — does a surge onto a warm, loaded accel cluster move the hand headroom or the GNN gap?

**Status:** `REGISTERED` (2026-10-12; designed by SCIENTIST, signed by the coordinator as drafted, with the two notes applied below). Phase 0, live, no training. Owner S5;
S5 implements it after the `sparse_obs_s0_v1` stage-2 read. Parents read: `ib0_confirm_v1` (the bar, and its post-close time split
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
