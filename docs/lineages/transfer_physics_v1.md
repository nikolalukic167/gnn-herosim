# transfer_physics_v1 — do the transfer model and replica holding decide the single-origin ranking?

**Status:** `REGISTERED` (2026-10-07). Exploratory, with no verdict bars. The conditions, arms, cells, statistic and
expected directions below were written before any condition run was read; only the 24-run replay check and three
smoke runs had been submitted.

**Outcome.** None yet.

**Why.** Two modelling choices sit under every grounded number, and neither was stated:
- **Transfers are charged store-and-forward.** `Platform._payload_transfer_time` charges hops × size ÷ slowest link.
  On `small_batch_confirm_v1`'s 19 topologies a server-to-server route is 2–8 hops, mean **4.96** (measured
  2026-10-07). Every link runs at 1000 MB/s (`bandwidth_mbps` is MB/s; `docs/gates/gate-tools.md`). So a peer
  exchange costs about 5× what a pipelined transfer (size ÷ bottleneck) would. Rules that compute distance exactly
  benefit most from that weighting.
- **A replica is held from cold start to output.** That includes the peer rendezvous and the exchange
  (`infrastructure.py`, `platform_process`), which matches Knative `containerConcurrency: 1`. Transfer time is
  therefore service time, and queues form behind other tasks' transfers.

`client_local_v1` found the single-origin ranking puts CD, locality-first and the one-pass greedy ahead of every
learned arm. This lineage asks whether that ranking survives when either choice changes, before any redesigned
workload is registered.

**Conditions** (a 2×2; all opt-in, defaults replay every earlier run):

| | replica held (default) | replica released (`HEROSIM_REPLICA_RELEASE=1`) |
|---|---|---|
| **store-and-forward** (default) | existing runs: `client_local_v1/gate_so_server`, `small_batch_so_v1/gate` | `release` |
| **pipelined** (`HEROSIM_TRANSFER_MODEL=pipelined`) | `pipe` | `pipe_release` |

- **Pipelined** charges size ÷ bottleneck plus the route's latency, everywhere data moves:
  - the peer exchange;
  - parent→child transfers (inert here, single-task applications);
  - task ingress over the link pipes, which holds every link on its route at once for one transmission.

  The same switch changes the rules' estimate (they call the same function) and the GNN's exchange feature
  (`prefix_serving`), so no arm reads different physics from the physics charged.
- **Released replica:** the rendezvous wait and the exchange no longer hold the replica. Execution and output
  take a per-replica compute lock, so I/O overlaps across tasks while compute stays serial. Overall this
  resembles Knative concurrency above 1 for waiting requests.
  - The sandbox stays warm: after its cold start the replica holds the function, and tasks popped while an
    earlier one is still in flight start warm.
  - The autoscaler neither scales down nor drains a replica with tasks in flight.
  - A starved-peer rendezvous is released through the task's own process.

**Arms.** The five rules (Knative `reactive`, `selfpredict`, `locality`, one-pass `batched`, `cd`, one run each) and
`so1load_selfref` seeds 1–2. **`so1load` is zero-shot here**: it was trained on store-and-forward, held-replica
labels. A loss under corrected physics may be distribution shift, so nothing in this lineage concludes anything
about the GNN; that needs a retrain on the chosen physics.

**Cells.** `client_local_v1/inputs_so_server`: `small_batch_confirm_v1`'s 19 topologies × g0–g3 × {×2, ×3, ×5},
single origin, server-only. Gate phase `tp1` of `scripts_cosim/fresh_topo_burst_v1_gate.py`, sbatch
`scripts_cosim/datalab/transfer_physics_v1_gate.sbatch` (`TP1_COND`).

**Statistic.** As `small_batch_so_v1`: per topology, the median paired % over (window, seed). Then the median over
19 topologies, with an exact Wilcoxon reported for direction only. Reported per condition and rung:
- each arm's latency;
- queue share (mean queue time ÷ mean latency);
- exchange per task;
- the ranking of the six arms;
- every arm against CD.

**Failures** (fixed 2026-10-07, before any condition result was read):
- A run that fails gets one rerun at 3× the timeout (8,100 s).
- A run that still fails counts as a failure **for its arm**. It is reported as a count per arm and condition, and
  that (topology, window, seed) cell drops out of **that arm's** paired tests only. Other arms keep the cell; nothing
  is imputed.
- Sensitivity: every test is re-read with any topology that has a remaining failure excluded for **every** arm.
- The failures so far are arm-specific, not topology-specific. All 16 first-try failures were Knative or
  self-predict on topology 9485. Both run on `KnativeNetworkAutoscaler`, which lacks the starved-type eviction the
  other arms' autoscaler has, and the same cell hung for the same arms under the default physics
  (`client_local_v1`).

**Replay check.** `TP1_COND=replay` reruns 24 runs (4 topologies × g0/g1 ×2 × CD, one-pass, Knative) at default
physics on this commit. Each must reproduce its `gate_so_server` summary exactly, or nothing below is read.

**Expected directions** (written 2026-10-07, before any condition run):
1. **`pipe`.** The exchange transmission term falls about 5× (latency unchanged), so exchange per task drops
   roughly 3–4×. Replicas are held through the exchange, so service time falls and **queue share falls clearly at
   every rung**. The rules' distance advantage shrinks:
   - the spread between CD, locality-first and the one-pass greedy narrows;
   - `so1load`'s deficit to CD narrows from +13 % at ×2 toward zero, since its weak point was exchange.

   Knative stays last.
2. **`release`.** **Queue share falls sharply**, from about 0.8 to well under half, because the replica no longer
   waits out other tasks' 2–4 s transfers. Latency becomes mostly exchange plus rendezvous, which rewards
   minimising exchange over balancing queues:
   - CD and locality-first gain relative to the others;
   - `so1load`'s queue-balancing edge stops mattering, so it falls further behind CD;
   - Knative, which has no exchange term, falls further behind every rule in relative terms.
3. **`pipe_release`.** Queue is nearly gone and exchange is small, so absolute latencies converge and arm
   differences shrink in seconds. Ranking is set by exchange plus latency: CD and locality-first ahead, Knative last.

**What this lineage cannot settle.** It changes physics under fixed workloads, payloads (200 MB × 10^U(−1,1)) and
1000 MB/s links; the redesign changes those. Link contention for the exchange stays off: exchange never takes a
link pipe. Testing contention is a separate later lineage, one change at a time.

## Record

- 2026-10-07 — **Conditions run** (jobs 838361–838363, 1,596 runs each): 16 first-try failures, all Knative or
  self-predict on 9485 (timeouts), rerun at 8,100 s (jobs 838555–838557). Replay 838353: 23 / 24 identical, field
  for field; the 24th (Knative, 9485 g1 ×2) times out as it did in the original gate. Failure handling fixed above
  before any read.
- 2026-10-07 — Registered. Hop counts measured on the 19 topologies: server-to-server 2–8 hops, mean 4.96
  (client-to-server mean 4.98). Code: opt-in flags `HEROSIM_TRANSFER_MODEL` and `HEROSIM_REPLICA_RELEASE` (both
  recorded in `run_provenance.env`; the gate fails a run whose recorded physics differs from the driver's), tests
  `tests/test_transfer_model.py`. Smoke (job 838349) and replay (job 838350) submitted before this node was written;
  no condition run submitted.
