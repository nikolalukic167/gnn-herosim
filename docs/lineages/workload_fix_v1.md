# workload_fix_v1 — payloads, access-link classes and task types on R1 (freeze workload WF1)

**Status:** `ACTIVE` (2026-10-08) — W2 read; W2+W3 and W2+W3+W4 cells running (not opened); CD's batching window fixed by amendment before W3 is read. Depends on: `physics_audit_v1` (R1 frozen). Plan: [`reference_physics_programme.md`](reference_physics_programme.md).
Created 2026-10-08 from W2–W4 of the withdrawn draft `workload_redesign_v1` (never committed) (W1 moved to `call_graph_pairing_v1`).


**Pre-run amendments (2026-10-07, decided by the coordinator from a code check).**
- **W3 applies to every node's access links, servers included.** Peer exchange runs server to server
  (`Platform._peer_exchange_time`, `src/placement/infrastructure.py`), and replicas are server-only, so a client's
  access link carries only the ingress input (12.8–153.6 KB per task, `data/nofs-ids/task-types.json`): at 4 MB/s
  about 0.04 s, against about 1 s of latency. Classes on client links alone could not move exchange. Each node
  (server or client) draws its class once per topology, mix 40 / 40 / 20 %. Our servers are Raspberry Pi and Jetson
  Xavier edge devices, so wireless edge servers are in scope.
- **Cellular is directional:** 4 MB/s out of the node, 75 MB/s into it. Links today carry one bandwidth for both
  directions (`network_fabric.link_key`), so the fabric gains a per-direction bandwidth; uniform-link topologies are
  unchanged.
- **Prediction 2 reworded:** exchange is under 10 % of latency for pairs whose two servers are both wired, and not
  when either server is Wi-Fi or cellular.
- **W2 scale, stated before the read:** mean payload per pair falls from about 430 MB to about 27 MB, and about 72 %
  of the new bytes sit in the heavy tier, so co-location value will come mostly from heavy pairs and route latency.
- **W4 cause found:** the generator's reachability repair skips any type with no initial replica placement
  (`src/generate_infrastructure.py`, `if not replica_servers: continue`), so `rf` / `cnn` can be unreachable. The fix
  repairs every type the workload uses; the static check then runs over all test and calibration topologies.


**Implementation and W4 precondition (2026-10-08, before any W4 run).** Code at `28411eb3` (all opt-in; with the
flags off, 96 of 96 configs and the minter replay byte for byte): `src/placement/workload_payloads.py` (W2,
`--payload-sampler wf1_v1` on `grounded_workload_v1_mint.py` only), per-node access classes and direction-aware
bandwidth in `src/placement/network_fabric.py` (W3; every bandwidth reader, including CD's cost model and the GNN
features, goes through `fabric.hops` or `route_hops_and_bottleneck`), and `network.reachability_repair` (W4), which
fails loud on a used type with no server replica instead of skipping it. Static check
(`scripts_cosim/workload_fix_v1_reachability_check.py`): **19 / 19 test topologies pass without the repair**
([`reach_19.json`](workload_fix_v1/reach_19.json)) and 96 / 96 configs of the pool pass with it
([`reach_all_repair.json`](workload_fix_v1/reach_all_repair.json)), so the `rf` / `cnn` defect is latent in these
configs; the 4 calibration topologies are checked when minted. W3 draws on the 96 configs: 40.1 / 40.5 / 19.4 %.
**Calibration topologies: ids 9601–9604**, minted with the same generator and config template as the 9473–9568
pool (unused anywhere in `simulation_data/` or the record, checked 2026-10-08). The 77 non-admitted ids of that pool
are not used: they failed admission under old physics.


**Pre-run amendment (2026-10-08, coordinator, before any W2+W3 or W2+W3+W4 cell): stages run in parallel, and
are read in order.** The W2+W3 and W2+W3+W4 cells may run while W2 is running. The registered order still governs
reading. No arm comparison of a stage is opened before the previous stage is read. Before that, only completion,
failure and hang counts may be looked at. If reading an earlier stage forces a change to shared code, configs or rungs,
every later stage's cells are discarded and rerun. All stages use the same provisional rungs (from the W2 bisection) and
the same 19 test topologies. Each stage writes to its own directory under `simulation_data/workload_fix_v1/`.
Reason: W3 and W4 add new code paths (direction-aware links; `rf`/`cnn`), and hangs surfaced as stuck cells on
calibration topologies 9603/9604. Finding such failures while W2 runs costs nothing in attribution, since each
stage still differs from the previous one by one factor and is read after it.

## Question
Which classical rankings on R1 survive replacing the synthetic payloads, uniform links and two-type task mix with
grounded or explicitly labelled values? The result fixes workload **WF1**, on which all learned arms are trained.

## Factors (introduced one at a time, then combined)
| Factor | Current | Fixed value | Source class |
|---|---|---|---|
| W2 Payload | 200 MB × 10^U(−1,1) (20 MB–2 GB) per partner pair | log-normal per pair with median 4 MB (σ = 1.2 in natural-log space), plus a labelled heavy tier: 10 % of pairs at 50–500 MB log-uniform. Overall ≈ 70 % of pairs under 10 MB (0.9 × Φ(ln 2.5 / 1.2) ≈ 0.70), approximately matching Eismann's 69 % (theirs is data volume per application, not per pair; stated as an approximation) | cited: Eismann et al., IEEE Software 2021 (69 % of apps under 10 MB); platform caps (Step Functions ≤ 256 KB direct, Lambda 6 MB); heavy tier = assumed and swept |
| W3 Access links | every link 1000 MB/s | per-client class drawn once per topology: wired 117 MB/s; Wi-Fi 7–14 MB/s; cellular 4 MB/s up and 75 MB/s down. Mix 40 / 40 / 20 %. Core links unchanged. | weak: practitioner Pi benchmarks; one 5G measurement (Muzaffar et al., 2020) |
| W4 Task types | 2 occurring (`dnn1`, `dnn2`) | all 4 defined types, equal mix | HeROsim task profiles (Lannurien et al., IEEE Internet Computing 2024) |

All rates in MB/s (unit documented in `gate-tools.md`). Values are fixed in this node and will not change after any read.

**W4 precondition (code, before any W4 run).** The generator must guarantee that every task type has at least one
reachable eligible server in every topology (today `rf`/`cnn` can starve and hang). Verified by a static check over
all test and calibration topologies, recorded here before W4 runs.

## Design
- Physics: R1. Policy time scale fixed as frozen in R1.
- Provisional load: two multipliers found on the 4 held-out calibration topologies (protocol of
  `load_recalibration_v1`, max 8 bisection steps), targeting CD queue share ≈ 0.1 and ≈ 0.3 on the current workload.
  Final rungs come from `load_recalibration_v1` on WF1.
- Arms: classical only (CD, locality-first, one-pass greedy, self-predict, Knative as context). **No learned arms.**
- Stages: W2 → W2+W3 → W2+W3+W4. Each stage is read before the next runs.
- Output: WF1 = the final stage's configuration (committed hash), plus a parameter-provenance table for the paper,
  with every parameter marked trace / measured / cited / assumed-and-swept.

## Primary family
CD vs each other classical arm at each stage and provisional rung (Holm within stage).

## Predictions
1. CD remains first at every stage.
2. W2 makes exchange a small share of latency (< 10 %) for clients on wired links, but not on Wi-Fi/cellular links.
3. W3 widens the spread of per-task cost; locality-first's gap to CD grows (hop distance stops proxying cost).
4. W4 increases cold-start share (more types → more sandboxes).

## Outcomes
Rankings that survive all stages are reported as robust; flips are sensitivity findings. WF1 is frozen regardless
of which arm benefits.


## Record (newest first)

### 2026-10-08 — stage W2 read; calibration set; provisional rungs; amendment WB (batching window)

Run code `f22b2e7e` (`rp/wf1-run`; `src/` identical to the replay-tested `f44cf9cf`), R1 with
`GATE_FIXED_POLICY_TIME_SCALE=1.0`. Default path: 5 R1 cells identical to `690ba384` on every simulated field; the
legacy build chain reproduces `grounded_x20` byte for byte. Report: [`w2_report.md`](workload_fix_v1/w2_report.md),
read: [`w2_read.json`](workload_fix_v1/w2_read.json).

**Calibration set: 9601, 9602, 9607, 9608.** 9603 and 9604 time out on every cell at ×2 for CD and Knative ("No
compatible hardware available" for one client). That's a runtime capacity hang that the static W4 check passes. Replaced
by the first ids from 9605–9610 that finish all 8 screen cells; 9605 and 9606 hang too (4 of 10 minted ids). The screen
covers ×2 only.

**Provisional rungs** (CD median queue share, 4 calibration topologies × g0/g1, legacy payloads): **×11.61 → 0.3007**
(8 steps); **×0.2666 → 0.1001**. The registered bracket [×0.5, ×24] did not reach 0.1 (0.117 at ×0.5), so a labelled
8-step extension on [×0.1, ×0.5] found it. On W2 payloads CD reads 0.09 / 0.37, as expected of provisional rungs.

**Stage W2** (`wf1_v1` payloads: mean per pair 27.2 MB against 429.2, median 4.8 MB, 70 % of pairs < 10 MB; pair graph
and events unchanged): 19 topologies × 2 rungs × 4 windows × 5 arms, 760/760, 0 failed.

| rung | CD latency | self-predict vs CD | locality vs CD | one-pass greedy vs CD | reactive (context) |
|---|---|---|---|---|---|
| ×0.2666 | 2.85 s | −14.9 % (19/19, Holm 2e-5) | +0.6 % (CD-FASTER) | +0.1 % (CD-FASTER) | −14.9 % |
| ×11.61 | 0.50 s | −6.1 % (19/19, Holm 2e-5) | +1.4 % (CD-FASTER) | +0.9 % (CD-FASTER) | +4.4 % |

Prediction 1 (CD first at every stage) **fails at W2**: self-predict is first at both rungs. Exchange is 3.9 % / 10.5 %
of CD's latency; the wired/wireless split of prediction 2 needs W3.

**Amendment WB (coordinator, 2026-10-08, after the W2 read: post-data, labelled).** The ladder protocol scales the
cells' `batch_timeout` with the rung: 16 s × 3.75 = **60 s** at ×0.2666 and 1.4 s at ×11.61. That's a 43× change in a
policy constant across rungs. B1 (`physics_audit_v1`) froze one set of policy constants in wall-clock seconds on every
rung, and the batching window is one of them; I12 didn't list it, so it slipped through. CD's latency is 2.85 s at the
light rung against 0.50 s at the heavy one, and reactive (no batching) ties self-predict there, so the 60 s window
plausibly explains much of CD's light-rung loss. For WF1 and every later node, **`batch_timeout` is 16 s (its ×1 value) at
every rung**. The fix isn't obviously in CD's favour: at ×11.61 the window grows from 1.4 s to 16 s. The W2 read above
stays recorded as run. The batching arms (CD, one-pass greedy, and any other arm that reads `batch_timeout`) are rerun on
W2, W3 and W4 with the fixed window; arms that don't read it stand. The W2 primary family is re-read on the rerun, with a
decomposition of CD's latency (batching wait, queue, exchange, cold start) under both windows.
