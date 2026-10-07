# transfer_physics_v1 — do the transfer model and replica holding decide the single-origin ranking?

**Status:** `ACTIVE` (2026-10-07). The 2×2 is read; the third factor (autoscaler scale-out) is registered as its own node,
[`kpa_scaleout_v1`](kpa_scaleout_v1.md), node 1 of [`reference_physics_programme`](reference_physics_programme.md). Registered 2026-10-07, exploratory with no verdict bars. Conditions, arms, cells, statistic and expected
directions were written before any condition run was read.

**Outcome (2×2, read 2026-10-07).**
- **CD is fastest in every condition, at every rung, on every topology.** No other arm is faster than CD on any of
  the 19 topologies in any of the 9 new condition × rung cells.
- **The rule order holds across physics:** CD, then locality-first and the one-pass greedy (+2.6 to +12 %), then
  self-predict, then Knative.
- **Replica holding is what made queues dominate:**
  - releasing the replica cuts queue share from 0.78–0.96 to **0.05–0.08**;
  - pipelining alone leaves it at **0.69–0.78** (absolute queue about 2 s instead of 6.7–55 s).
- **Absolute latency falls 4–70×** (CD at ×2: 8.62 s → 2.89 s pipelined, 1.99 s released, 0.97 s both).
- **Knative's gap to CD shrinks** from +90 / +126 / +151 % to +15–72 %.
- **`so1load`, zero-shot, falls further behind CD** (+12.7 / +7.4 / −3.3 % → +17 to +27 %); it exchanges more than
  every rule except Knative.
- **The load ladder no longer orders load:** under the new physics latency *falls* from ×2 to ×5 (pipelined CD 2.89 →
  2.18 s), so the ×N multipliers must be recalibrated before any comparison under new physics.
- Excluding 9485 for every arm moves every new-condition paired median by ≤ 2.1 points. In the default-physics reference, Knative at ×5 moves from +151 to +163 % and self-predict at ×5 from +8 to +12 %. No sign or ranking changes. All of this is under the default scale-out
  target (100 per replica), which `transfer_physics_v1` has not yet varied.

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

- 2026-10-07 — **Read (2×2).** Reader `scripts_cosim/transfer_physics_v1_read.py`; output
  `simulation_data/transfer_physics_v1/tp1_read.json` (and `tp1_read_excl.json` with `TP1_EXCLUDE=9485`) on datalab.
  - **Failures.** The 16 Knative / self-predict runs on 9485 were marked hung after about 57 min of their 8,100 s
    rerun (jobs cancelled on the user's instruction, not at the registered 3× limit) and enter as arm failures under
    the rule above.
  - **CD median latency (s) and queue share, ×2 / ×3 / ×5:**

    | condition | ×2 | ×3 | ×5 |
    |---|---|---|---|
    | store-and-forward, held | 8.62 (0.78) | 13.32 (0.84) | 56.99 (0.96) |
    | pipelined | 2.89 (0.69) | 2.38 (0.70) | 2.18 (0.74) |
    | released | 1.99 (0.07) | 1.92 (0.07) | 1.94 (0.08) |
    | both | 0.97 (0.15) | 0.86 (0.17) | 0.78 (0.20) |

  - **Paired % vs CD** (median over 19 topologies; 0/19 faster than CD in every new cell):

    | arm | pipelined | released | both |
    |---|---|---|---|
    | one-pass greedy | +2.6 / +3.6 / +5.8 | +11.8 / +11.2 / +11.2 | +4.4 / +5.4 / +5.6 |
    | locality-first | +6.9 / +7.4 / +8.4 | +8.0 / +8.3 / +6.5 | +3.6 / +3.6 / +4.6 |
    | `so1load` (zero-shot) | +16.6 / +19.1 / +22.8 | +23.9 / +26.7 / +23.2 | +18.3 / +22.6 / +26.6 |
    | self-predict | +28.2 / +34.7 / +43.8 | +35.4 / +39.1 / +35.2 | +8.5 / +15.1 / +19.7 |
    | Knative | +29.6 / +38.3 / +52.4 | +67.3 / +71.7 / +70.0 | +15.0 / +22.5 / +29.1 |

    Default physics, for reference: greedy +7.5 / +6.3 / −3.3, locality +9.8 / +7.3 / +1.0, `so1load`
    +12.7 / +7.4 / −3.3, self-predict +52.2 / +53.6 / +8.1, Knative +90.4 / +126.2 / +151.4.
  - **Against the expected directions:**
    1. `pipe`: queue share falls (0.78 → 0.69 at ×2) but only modestly (**weaker than expected**). The spread among
       CD / locality / greedy narrows at ×2–×3 but **widens at ×5**. `so1load`'s deficit **grows** (expected to
       narrow). Knative is last, as expected.
    2. `release`: queue share collapses to 0.05–0.08 (**as expected**). CD and locality-first lead (as expected).
       `so1load` falls further behind (as expected). Knative's relative gap **shrinks** from +90–151 % to +67–72 %
       (expected to grow).
    3. `pipe_release`: latencies converge to 0.8–1.2 s and CD / locality-first lead (as expected). Queue share is
       0.15–0.20, not "nearly gone": with exchange small, the remaining queue is a larger share.
  - **Sensitivity** (9485 excluded for every arm, n = 18):
    - new conditions: every paired median moves by ≤ 2.1 points;
    - default-physics reference: Knative ×5 +151.4 → +162.6 %, self-predict ×5 +8.1 → +12.4 %;
    - no sign or ranking changes.
- 2026-10-07 — **Conditions run** (jobs 838361–838363, 1,596 runs each): 16 first-try failures, all Knative or
  self-predict on 9485 (timeouts), rerun at 8,100 s (jobs 838555–838557). Replay 838353: 23 / 24 identical, field
  for field; the 24th (Knative, 9485 g1 ×2) times out as it did in the original gate. Failure handling fixed above
  before any read.
- 2026-10-07 — Registered. Hop counts measured on the 19 topologies: server-to-server 2–8 hops, mean 4.96
  (client-to-server mean 4.98). Code: opt-in flags `HEROSIM_TRANSFER_MODEL` and `HEROSIM_REPLICA_RELEASE` (both
  recorded in `run_provenance.env`; the gate fails a run whose recorded physics differs from the driver's), tests
  `tests/test_transfer_model.py`. Smoke (job 838349) and replay (job 838350) submitted before this node was written;
  no condition run submitted.
