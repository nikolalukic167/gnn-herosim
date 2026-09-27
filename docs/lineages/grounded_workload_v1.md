# grounded_workload_v1 — does the learned burst-seat arm beat every strategy on a workload whose group structure is measured, not assumed?

**Status:** `REGISTERED` (2026-09-27). Every bar below was signed before any gate datum existed. The only
runs seen before signing were a 3,000-event smoke of three rules on one cell (disclosed below).

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

- 2026-09-27 — Registered. Smoke seen before signing: 3,000 events of `g0` on 9119, `random_network`,
  `knative_network`, `peer_greedy_network_cd` all complete (smoke only; not a read).
