# grounded_workload_v1 — does the learned burst-seat arm beat every strategy on a workload whose group structure is measured, not assumed?

**Status:** `CLOSED` (2026-09-27) — **G1 `NOT-SEPARATED`: on the grounded workload the learned arm ties CD.**
Registered 2026-09-27; every bar below was signed before any gate datum existed (the only runs seen before
signing were a 3,000-event smoke of three rules on one cell, disclosed below).

**Outcome** (job 810898, 1,046/1,058 runs, 43/48 cells admissible, witness passed, no reliability flag):
- **G1** `xs1load_selfref` vs CD **+0.43 %**, p = 0.57, 6/12 — a tight tie (per topology −5.6 to +3.0 %).
- **G2** vs self-predict −0.91 %, p = 0.76 — tie.
- **G3 `CONFIRMED`** vs Knative **−16.2 %**, random **−39.8 %**, one-pass greedy **−7.1 %**, Decima **−20.5 %**
  (each p ≤ 0.001, 11–12/12).
- **G4** `gnnedge0` vs its MP-OFF twin −1.41 %, 12/12, p = 0.0005 — direction only; the MP twins both trail
  CD (+5.4 / +6.6 %).
- So on a trace-grounded workload the learned arm is in the top tier with CD and self-predict and beats every
  industry-style baseline; **it is not a win over CD** and must not be quoted as one. All 12 failures were
  rule runs timing out (11 on 9466, 1 CD on 9423/g2); no learned run failed and 9434 did not collapse.

**Why.** Every burst-seat result so far (`joint_burst_v2`, `fresh_topo_burst_v1`, `burst_ladder_v1`,
`x11_confirm_v1`) runs one invented group structure: 10 consecutive arrivals form a peer group, either spread
over 13.8 s (median; the ×4000 time stretch of `workload-150-150.json`) or collapsed to one instant (the burst
lever). Neither the group size nor the sibling spacing was ever measured, and the base trace's provenance is
not recorded in the repo (`peer_affinity_v1.md` calls it "the real production trace" without a source). A
paper needs a workload whose structure is taken from a published trace.

## What was measured (2026-09-27, before registration)

Source: Alibaba `cluster-trace-microservices-v2021`, MSCallGraph shards 0, 50, 100, 130 (Luo et al., SoCC'21;
md5 in each library's `source`). A request is a `traceid`; a fan-out is the calls of one trace that share an
rpcid parent.

- **Sibling spacing** (600 k fan-outs, shards 0/50/100): span p50 **2 ms**, p90 94 ms, p99 430 ms; 23 % exactly
  simultaneous; 90 % within 100 ms. The three shards agree to within a few ms. Real fan-out is close to the
  burst lever, not to the 13.8 s stretch.
- **Fan-out size** (one fan-out per request, sizes ≥ 11 cut to 10): mean ~4.0, mode 3; 1.3–2.0 % of requests
  have no fan-out; 6.8–10.3 % were larger than 10.
- **Request arrivals** (5-minute shards): squared coefficient of variation of inter-arrival times 1.04–1.07,
  i.e. Poisson at fleet scale; the rate drifts 435 → 488 → 620 req/s over the day. The trace is a 0.5 %
  per-trace sample, and thinning pushes a point process towards Poisson — disclosed.
- **Domain-matched rate check (UNSW-NB15)**: ~22.8 flows/s over the documented 31 h capture, and 96 % of
  sorted flows share their 1-second timestamp. The rate is a testbed rate, not one sized to 6 servers, and
  the timestamps are too coarse for sibling timing; it is context, not an input.

## Design

- **Workloads** (`scripts_cosim/grounded_workload_v1_extract.py` → `fanout_lib_<shard>.json.gz` →
  `grounded_workload_v1_mint.py`). Window `g<i>` takes shard (0, 50, 100, 130)[i] and the study's pre-burst
  window (w0, w1, w2, w3)[i]:
  - groups in request-arrival order take the library's fan-out sizes until 50,000 tasks;
  - a group dispatches at its request's arrival, with inter-request times stretched by one factor
    (3,680–5,675×) so the task rate equals the study's **×1, 0.4604 tasks/s** (the admissible load);
    whole-millisecond arrivals are spread uniformly inside their millisecond first (seeded), else the
    stretch puts them on a ~4 s lattice;
  - members keep their **real** sibling offsets (never stretched);
  - each task takes the base window's k-th task type, demand scale, QoS and client;
  - peers: the study rule (2 partners in-group, or size − 1; 200 MB × 10^U(−1,1)).
  - Checksums: `grounded_workload_v1/workloads.sha256`. Datalab mints its own copies from the committed
    libraries and must reproduce every sha256, or the job stops.
  - Per window: 12,220–12,795 groups, mean size 3.91–4.09; group-arrival C² 1.02 (old w0 1.56, old w1 0.10).
- **Topologies:** the 12 `fresh_topo_burst_v1` study topologies. Cells = 12 × 4 = 48.
- **Arms** (phase `grounded` of `backlog_corpus_v1_gate.sbatch`, 1,058 runs):
  - rules, one run per cell: `random` (`random_network`), `reactive` (Knative), `batched` (one-pass greedy),
    `selfpredict`, `cd` (coordinate descent), `decima` (tuned weighted fair, α = +1 from `decima_rule_v1`);
  - learned, seeds 1–4 per cell: **`xs1load_selfref`** (the strongest learned arm, `exchange_seconds_v1`
    checkpoints + 3 self-refine passes), `xs1load` (the same weights, plain decode), `gnnedge0` and `mpoff`
    (`joint_burst_v2`'s MP / MP-OFF twins; the manuscript's arm).
  - All checkpoints are unchanged; none saw a group of any size but 10 (disclosed risk).
- **Witness:** `xs1load_selfref` seed 1 on `w0x11d1` for 9119 and 9420 must equal `capacity_sweep_v1`'s runs
  to the digit, proving the served path did not move.

## Statistic and bars (signed 2026-09-27, before any datum)

- **Admissible cell:** reactive completed and its queue share ≤ 0.80. Missing or failed is not a pass. Only
  admissible cells enter any contrast.
- **Statistic** (`scripts_cosim/grounded_workload_v1_read.py`): learned vs rule, per topology the median over
  (admissible window, seed) of the paired % on the same cell. Learned vs learned pairs the seed; rule vs rule
  is one run per cell. Exact two-sided Wilcoxon over topologies. A pair with a missing run is dropped by name.
  Fewer than 8 topologies → `DESIGN-SHORT`.
- **Labels:** `CONFIRMED` (median ≤ −5 %, p < 0.05) / `DIRECTION-ONLY` (median < 0, p < 0.05) /
  `REF-FASTER` (median > 0, p < 0.05) / `NOT-SEPARATED`.

| read | contrast | role |
|---|---|---|
| **G1** | `xs1load_selfref` vs `cd` | **the primary** — the only contrast that licenses "the learned arm wins" |
| G2 | `xs1load_selfref` vs `selfpredict` | the per-arrival hand-rule bar |
| G3 | `xs1load_selfref` vs `reactive`, `random`, `batched`, `decima` | reported with the same labels |
| G4 | `gnnedge0` vs `mpoff` (same seed) | the message-passing question; a win here is not G1 |
| — | `xs1load`, `gnnedge0`, `mpoff` vs every rule; every rule vs `reactive`; failures per arm | reported |

- **Reliability flag:** if `xs1load_selfref` fails on more than one admissible cell, every G-read is quoted
  with the flag.
- **What may be quoted in the paper:** G1 `CONFIRMED` → "beats every strategy including CD on the grounded
  workload (12 topologies, 4 windows, 4 seeds, one admissible load)". G1 `DIRECTION-ONLY` → a direction, no
  magnitude. G1 `NOT-SEPARATED` or `REF-FASTER` → no win claim over CD; G2/G3 are then quoted as they read.
  No re-draw, re-seed, re-window or re-load of this gate is licensed by its outcome.
- **Expectations:** G1 `CONFIRMED` 10 %, `DIRECTION-ONLY` 20 %, `NOT-SEPARATED` 55 %, `REF-FASTER` 15 %.
  At ×1 the w0 lead dissolved (−11.9 %, p = 0.15) and w1–w3 tied; here groups are ~4 tasks, not the 10 every
  checkpoint was trained on, which should cost the learned arms more than the rules. G2 `CONFIRMED` 55 %.
  G3 vs `reactive` and `random` `CONFIRMED` 85 %. G4 `DIRECTION-ONLY` or better 45 %.
- **Standing risks:**
  - Train/serve shift in group size (10 → mean 4, 2 % singletons).
  - Cross-domain grounding: RPC microservice fan-out, not the dnn1/dnn2 inference chain; the physics
    (exec, cold start, transfer) stays the hardware-profiled one.
  - One load (×1); the testbed rates of UNSW-NB15 are not transferable to 6 servers.
  - The 9434 replica-expiry collapse (`replica_guard_v1`) may recur.
  - Rule 6 is met: every read is live.

## Entry points

- Builder: `scripts_cosim/grounded_workload_v1_extract.py`, `scripts_cosim/grounded_workload_v1_mint.py`.
- Libraries and checksums: `docs/lineages/grounded_workload_v1/`.
- Driver: `scripts_cosim/fresh_topo_burst_v1_gate.py` phase `grounded`; sbatch
  `scripts_cosim/datalab/backlog_corpus_v1_gate.sbatch` (`PHASE=grounded`).
- Reader: `scripts_cosim/grounded_workload_v1_read.py`. Tests: `tests/test_grounded_workload_v1.py`.

## Record (newest first)

- 2026-09-27 — **Gate read** (job 810898 on os-cpu-slurm-5, `gnn` env torch 2.5.1, at f31deaf; datalab's own
  mint of g0–g3 matched every local sha256). Attachment: [`grounded_read.json`](grounded_workload_v1/grounded_read.json).
  - **Runs:** 1,046/1,058. 12 timeouts at 1,800 s, all rules: 9466 reactive g0/g1/g3, random g0–g3,
    self-predict g0–g3 (the starved-client spin); CD on 9423/g2 (stalled at event 40,001). Learned arms: 0.
  - **Admissible:** 43/48. Out: 9466 g0/g1/g3 (reactive failed), 9461 g1 (0.809) and g2 (0.808).
  - **Witness passes:** 9119 8.726741 s and 9420 5.741893 s equal `capacity_sweep_v1` to the digit.
  - **G1 `NOT-SEPARATED`:** +0.43 %, p = 0.569, 6/12; 9423/g2 dropped by name (CD failed). Per topology (%):
    9119 −4.0, 9414 −0.0, 9420 +1.7, 9423 −4.0, 9434 +1.4, 9435 −5.6, 9444 −2.3, 9446 +2.7, 9456 +3.0,
    9461 +1.3, 9466 +0.9, 9469 −2.1.
  - **G2 `NOT-SEPARATED`:** −0.91 %, p = 0.765, 6/11 (9466 has no admissible self-predict run).
  - **G3 `CONFIRMED`:** reactive −16.15 %, random −39.75 %, batched −7.14 %, decima −20.47 %.
  - **G4 `DIRECTION-ONLY`:** `gnnedge0` vs `mpoff` −1.41 %, 12/12, p = 0.0005.
  - **Reported:**

    | vs → | random | reactive | batched | selfpredict | cd | decima |
    |---|---|---|---|---|---|---|
    | `xs1load` (plain decode) | −38.8 | −14.4 | −5.6 | +1.7 ns | +2.6 ns | −17.9 |
    | `gnnedge0` | −36.7 | −12.3 | −3.0 dir | +4.9 | +5.4 | −15.7 |
    | `mpoff` | −35.9 | −10.0 | −0.9 ns | +6.9 | +6.6 | −14.3 |

    Rules vs reactive: CD −14.8 %, self-predict −16.8 %, one-pass −10.3 %, Decima +4.4 %, random +46.8 %.
  - **Reading:** the measured group structure (≈4 tasks, ms spacing) does not change the ranking. The best
    learned arm sits exactly at CD's level with a narrow spread, i.e. this is a well-measured tie, not an
    underpowered miss. Self-refine is worth ~2 pp over plain decode (+0.4 vs +2.6 % vs CD).
- 2026-09-27 — Registered. Smoke seen before signing: 3,000 events of `g0` on 9119, `random_network`,
  `knative_network`, `peer_greedy_network_cd` all complete (smoke only; not a read).
