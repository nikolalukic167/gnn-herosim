# workload_fix_v1 — payloads, access-link classes and task types on R1 (freeze workload WF1)

**Status:** `REGISTERED` (no runs). Depends on: `physics_audit_v1` (R1 frozen). Plan: [`reference_physics_programme.md`](reference_physics_programme.md).
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
