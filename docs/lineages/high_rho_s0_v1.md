# high_rho_s0_v1 — is there a high-but-stable utilisation rung above `heavy`, and how much live headroom does CD leave there?

**Status:** `REGISTERED-DRAFT` (2026-10-11). Draft by S5 for the coordinator to sign. Every bar below is fixed before any data. This node is the rung search only; the oracle step is S4's and pre-registered below.

**Why.** The user asks for a shot at a ~20 % win over the hand bar. At `heavy` (accel seat, 160 clients × 24 servers) CD runs with end/last-arrival 1.008 and queue share 0.20 (smoke cell cc40s18700 g0 heavy, `prune_g8_s0_v1`), so the cluster is far from its capacity and the hand rules leave little to win. The one defensible regime is a *high-but-stable* utilisation rung: load high enough that placement quality moves the queue, low enough that CD does not diverge. This is not the ×3/×5 overload regime, where reactive saturates and every number is load-contingent; an unstable rung is inadmissible.

**Prior, disclosed.** Expect the stable window to be narrow: CD's queue share rises steeply near capacity and the first unstable rung can be one multiplier step above the last stable one. If no rung with bottleneck ρ ≥ 0.70 is stable, that is the result (`NO-STABLE-HIGH-RHO`), and the oracle step does not run.

## Design (fixed now)

- **Seat:** the `prune_g8_s0_v1` environment, unchanged except the multipliers and topologies: 160 clients × 24 servers, p = 0.6, WF1, G = 8 groups, single origin spread over the 160 clients, `fastest_compatible`, pipelined transfers, `HEROSIM_REPLICA_RELEASE=1`, KPA; moderate = ×11.6139, heavy = ×27.622665 over the x1 windows. Same code (pinned worktree of the rp/high-rho head), inputs minted with `big_groups_inputs.sbatch` as for `prune_g8_s0_v1`.
- **Topologies:** 6 fresh ids, **19310–19315**. Disjoint from S4's 19300–19305 by the `/share` directory names (the only `193xx` directory names are 19300–19305), from 18700–18711, 17007–17018, 16900–16911, and by `git grep` over every branch (only unrelated counters hit) and `REGISTRY.json` (no hit). The method is a name and token check, not a full content scan.
- **Utilisation ρ (defined before data).** From the kept raw per-task results (`KEEP_RAW_DIR` on `/share`), with `scripts_cosim/physics_audit/descriptives.py` (`util_by_platform_type`, occupied fraction of replica life and compute-lock fraction): **ρ = time-weighted mean occupied fraction of the busiest platform type**, over the arrival span. Reported next to it: ρ of every platform type, the compute-lock fraction, and the share of replica lives above 0.8. Every latency number in this lineage carries its ρ.
- **Stable (all three, on the cell, CD):** `arrival_end.end_over_last_arrival` ≤ 1.05; `queue_share` ≤ 0.80; no growing queue, `queue_drift`: the last-quarter mean queue ≤ 1.5 × the mean of the middle two quarters (the first quarter is the warm-up and excluded). A rung is stable only if the *median over the 6 topologies of each statistic* passes **and** at most 1 of its 24 cells fails; unknown is not a pass.
- **Stage 1, calibration (cheap):** CD, topology 19310, window g0, multipliers = heavy × {1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3, 4, 6, 8} (extended upward by the coordinator's amendment before data: platforms are 95–99 % idle at heavy, so ×2.5 may not bracket ρ 0.70). The cells run in parallel because they are cheap; the ladder is read as if stopped at the **first multiplier where CD is unstable**, and cells above it are reported but not used. Reads ρ and stability. One cell each, never rerun. From these ρ(m) is interpolated linearly in m to the targets 0.70 / 0.80 / 0.85 (if a target is still not bracketed below the first unstable multiplier, it is recorded as **"unreachable stably"**, which is itself the answer, and that rung is not run).
- **Stage 2, ladder:** CD and `cd_exactS` (seed 0) at each interpolated multiplier on the 6 topologies × windows g0–g3 (24 cells per arm per rung). ρ is **re-measured on the stage-2 cells**; the rung is labelled by its measured ρ, not the target. A rung whose measured ρ is more than 0.05 from its target is labelled with the measured value.
- **Reported per rung:** ρ (busiest type, all types), the three stability statistics and the cell count that fail, CD vs `cd_exactS` as the paired % of mean latency per topology (median over windows), median over 6 topologies, sign count, exact Wilcoxon (descriptive only: n = 6 cannot reach p < 0.05 two-sided with fewer than 6/6, disclosed), and the latency breakdown from the summaries (queue `averageQueueTime`, cold `averageColdStartTime` + init/pull, exec `averageExecutionTime`, exchange `totalPeerExchangeTime`/tasks + latency).
- **Watchdog:** progress watchdog as in `r1a_gate.sbatch` (`LIMIT` 30000); a killed or failed cell is counted and reported, never rerun, and counts as *unstable* for its rung (it is not dropped).
- **Cost:** stage 1: 6 cells (CD, minutes each). Stage 2: at most 3 rungs × 2 arms × 24 cells = 144 cells; `cd_exactS` cells were 6.0–6.8 ks at heavy and will be longer under load; at most 6 jobs alongside other gates.
- **Registered next step, not this node's (S4):** at the highest *stable* rung, S4's oracle harness (`cd_oracle`) runs against `cd_exactS`. If the oracle's live headroom there is under 20 %, no learned arm can reach 20 % and the programme stops. S5 reports the rungs; S4 owns that run.
- **Not claimed:** any learned-arm result; any ρ not measured here; stability on another seat.

## Reading rules
- **HIGH-RHO-RUNG-FOUND:** at least one rung with measured bottleneck ρ ≥ 0.70 is stable; the highest such rung is handed to S4.
- **NO-STABLE-HIGH-RHO:** no rung with ρ ≥ 0.70 is stable; the oracle step does not run.
- A rung with ρ below 0.70 is reported but is not the oracle rung.

## Record (newest first)
- 2026-10-11 — Signed (bda1e9d2) with the calibration extension above, before data.
- 2026-10-11 — Drafted (S5) on the coordinator's request.
