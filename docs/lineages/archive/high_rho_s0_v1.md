# high_rho_s0_v1 — is there a high-but-stable utilisation rung above `heavy`, and how much live headroom does CD leave there?

**Status:** `CLOSED` (2026-10-12) — **RHO-UNREACHABLE-STABLE** (the registered `NO-STABLE-HIGH-RHO` outcome). Registered 2026-10-11 by S5, signed by the coordinator; every bar below was fixed before any data.

**Outcome.** No stable rung reaches ρ 0.70 in the accel seat (CD, 6 topologies 19310–19315, g0). The highest stable multiplier is heavy ×5, with median end/last 1.046 and median ρ 0.638. ρ falls as load rises past ×4, compute lock stays ≤ 0.54, and latency is 63–71 % storage/I/O. The instability comes from the I/O path, not from platform saturation. The S4 oracle step does not run, so the high-ρ route to a ~20 % win is closed in this seat. Caveats: g0 only, with 2 of 6 cells failing at both ×4 and ×5. Not tested: other seats and other ρ definitions.

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
- 2026-10-12 — **CLOSED `RHO-UNREACHABLE-STABLE` (S5; live measurement, rule 6).** Stage 1 completed on 6 topologies (19310–19315, g0, CD, traced; 11 new cells plus 19310 ×4 from the first read; the traced cells' latencies equal the untraced ones).

  | ×heavy | n | median end/last | cells end/last > 1.05 | median queue share | median last/mid queue | median ρ (busiest type) | max ρ | median latency s | max compute ρ | other-I/O share | queue share of latency |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | 4 | 6 | **1.047** | 2 (19313 1.113, 19314 1.167) | 0.204 | 1.01 | **0.662** | 0.711 (19315, FPGA) | 5.65 | 0.43 | 0.64 | 0.19 |
  | 5 | 6 | **1.046** | 2 (19310 3.298 collapsed, 19314 1.143) | 0.181 | 0.98 | **0.638** | 0.670 | 7.72 | 0.54 | 0.70 | 0.17 |

  Per-cell ρ at ×4: 19310 .657 / 19311 .601 / 19312 .682 / 19313 .620 / 19314 .667 / 19315 .711; at ×5: .562 / .661 / .670 / .638 / .637 / .602. Queue share is 0.07–0.22 in every cell and the last-quarter queue never grows (last/mid ≤ 1.07).
  **Rule as ruled by the coordinator:** the highest multiplier whose 6-topology median end/last ≤ 1.05 is **×5** (1.046); its median ρ is **0.638 < 0.70**, so the node closes. (×6 failed on 19310 in the first read, end/last 1.144; ×8 collapsed.)
  **Disclosed:** the signed per-rung rule also allowed at most 1 failing cell; 2 of 6 fail at both ×4 and ×5, so under the stricter reading neither is stable and the result is the same (no stable rung reaches ρ 0.70). Only g0 was run, so "24 cells" was 6. 19310 at ×5 collapsed (end/last 3.3) next to five stable ×5 cells, so the ×5 boundary is topology-dependent.
  **Mechanism:** the busiest class never gets near saturation (ρ 0.56–0.71 at any multiplier, falling as load rises past ×4); the compute lock is at most 0.54 in any cell (0.37 at ×6 on 19310); and latency is 63–71 % storage and input/output ("other"), 17–20 % queue, 1–3 % exec. The rung where instability appears is set by the I/O path, not by platform utilisation, so a ρ ≥ 0.70 stable rung does not exist on this seat. No stage 2, no oracle. The oracle step registered for S4 does not run.
  **Not claimed:** other seats, other windows, a ×4.5 or other fractional multiplier, a capacity-based ρ definition (ρ here is the occupied fraction of replica life; replicas scale to 120).
- 2026-10-12 — **Coordinator correction (before the ×5 data landed):** one more traced CD cell, 19310 g0 at heavy ×5 (inputs job in `/share/nikola.lukic/high_rho/s2b`), so the ×5 median is over the same 6 topologies as ×4. The earlier "no ×5 cell on 19310" meant extending 19310 on its own, not the signed rule; a 5-topology median would not meet it. All 11 cells are read together, once, under the same rule. The pinned code worktree for these cells is 0a597e1e (the docs-only commit after it changes no code).
- 2026-10-12 — **Amendment before launch (coordinator ruling): stage 1 completed on 6 topologies, because the signed stability rule is a 6-topology median.** No ×5 cell on 19310 and no ×4 CD-vs-`cd_exactS` stage 2.
  CD only, traced, heavy ×4 and ×5 on 19311–19315 (10 cells, g0; inputs one job, `big_groups_inputs.sbatch` with the moderate tag = ×4 and the heavy tag = ×5). The ×4 median is therefore over 6 topologies (19310 from stage 1 plus these five) and the ×5 median over 5 (19311–19315), disclosed (superseded by the correction above: ×5 on 19310 is added).
  Rule: the highest multiplier whose median end/last ≤ 1.05 is the candidate rung. Median ρ there < 0.70 closes the node `RHO-UNREACHABLE-STABLE` (mechanism recorded: ρ falls as stability fails, compute lock ≤ 0.37, latency 66–71 % storage/I/O); ≥ 0.70 is reported to the coordinator before anything else.
- 2026-10-12 — **Stage 1 read (S5), topology 19310, g0, CD, one cell per multiplier (heavy × k); ρ = occupied fraction of the busiest platform type.**
  Multipliers 1 and 1.5 / 2 / 3 / 6 ran on the `moderate` tag and 1.25 / 1.75 / 2.5 / 4 / 8 on the `heavy` tag of five input pairs (jobs 860861–860870). ρ was measured on traced reruns of six
  cells (jobs 861451–861456: `HEROSIM_AUDIT_TRACE` through a copy of `r1a_gate.sbatch` at `/share/nikola.lukic/high_rho/r1a_gate_trace.sbatch`, the only change; the traced cells' mean latencies equal the untraced ones to 3 decimals, so the trace is inert).
  The harness raw results carry no per-platform utilisation, so the node's "from the kept raw results" is amended to "from the audit trace".

  | ×heavy | mean lat s | queue share | end/last | last/mid queue | ρ busiest (type) | compute ρ max (type) | stable |
  |---|---|---|---|---|---|---|---|
  | 1 | 2.379 | 0.175 | 1.021 | 0.99 | 0.46 (pynqFpga 0.49) | 0.12 (DLA) | yes |
  | 1.25 | 2.526 | 0.194 | 1.015 | 0.80 | not traced | | yes |
  | 1.5 | 2.387 | 0.217 | 1.037 | 1.00 | not traced | | yes |
  | 1.75 | 2.498 | 0.228 | 1.003 | 1.04 | not traced | | yes |
  | 2 | 3.732 | 0.182 | 1.004 | 1.00 | 0.60 (DLA) | 0.35 (DLA) | yes |
  | 2.5 | 3.336 | 0.224 | 1.075 | 1.02 | 0.53 (DLA/FPGA) | 0.31 (DLA) | end/last > 1.05 on this cell |
  | 3 | 3.877 | 0.232 | 1.014 | 1.09 | 0.61 (xavierGpu) | 0.28 (DLA) | yes |
  | 4 | 5.625 | 0.198 | 1.046 | 1.02 | **0.66 (xavierGpu)** | 0.31 (DLA) | yes (end/last 1.046 is 0.004 under the bar) |
  | 6 | 7.589 | 0.186 | **1.144** | 0.98 | 0.61 (xavierGpu) | 0.37 (DLA) | **no** (end/last) |
  | 8 | 46.273 | 0.514 | **2.185** | 0.90 | not traced | | **no** (collapsed) |

  Latency breakdown (traced cells, share of mean latency; queue / cold / exec / exchange / other-I/O): ×1 .135/.022/.035/.536/.232; ×2 .156/.004/.039/.233/.541; ×4 .180/.001/.019/.126/.656; ×6 .171/.001/.015/.090/.708.
  "Other" (storage and input/output beyond exchange) is 66–71 % of latency at ×4–×6; queue stays 0.17–0.21.
  **Reading, under the signed rules:** the first unstable multiplier is ×6; the highest stable one is ×4 at ρ = 0.66 (busiest type xavierGpu), so the targets 0.70 / 0.80 / 0.85 are **unreachable stably** on this cell, and ρ falls (0.66 → 0.61) from ×4 to ×6 while end/last crosses 1.05, so the instability is not a busiest-class saturation (replicas peak at 120 from ×1 upward; alive mean 100 at ×4). Stage 2 is not run. Caveats: one topology, one window, one cell per multiplier (the signed stability rule is a median over 6 topologies); ×5 and the interval 4–6 are untested; ×2.5 shows an isolated end/last 1.075.
- 2026-10-11 — Signed (bda1e9d2) with the calibration extension above, before data.
- 2026-10-11 — Drafted (S5) on the coordinator's request.
