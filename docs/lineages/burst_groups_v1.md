# burst_groups_v1 — peer groups that arrive together

**Status:** `CLOSED` (2026-09-20) — **`WAIT-COLLAPSED` · `NOT-SEPARATED` (`gnnedge0` vs
reactive) · `RULE-FASTER-THAN-GRAPH-ARM`; registered verdict `TOO-FEW-UNSATURATED-ENVIRONMENTS`,
study read under two signed amendments on 8 environments, disclosed.** Closed on a live gate
(rule 6): 48 screen arms + 280 study arms (3 `mpoff` checkpoints hang, below). Registered
2026-09-20; bars, reader, 6 tests and the expectation committed before any levered arm ran; two
local smoke runs disclosed. **The registered expectation was right on L4, L2, L3 and L5 and
called L1 at 45 % BEATS — it read NOT-SEPARATED.** Shares its apparatus with `payload_scale_v1`
and `backbone_sparsity_v1` (`scripts_cosim/env_lever_v1_*.py`, `scripts_cosim/datalab/env_lever_v1*.sbatch`).

**Outcome. Removing the wait from the environment does not rescue the learned arm; it makes the
rule's margin twice as large.** With every peer group dispatched as one burst the batching arms'
scheduler wait goes from 7.1 s to **0.001 s** per task (L4 `WAIT-COLLAPSED`) and reactive Knative
itself gets 54 % faster (17.9 → 8.29 s per task on the study environments: rendezvous 3.7 → 0.00 s,
queue 8.7 → 3.9 s), because partners placed within milliseconds of each other are never waited
for. On the 8 environments both serving paths can serve (9101, 9106 × 4 windows; reads at n = 8
disclosed, checkpoint-unit reads at n = 16):

| read | verdict | median | p | ahead |
|---|---|---|---|---|
| L1 `gnnedge0` vs reactive (ckpt) | `NOT-SEPARATED` | **−2.17 %** | 0.196 | 11/16 |
| `mpoff` vs reactive (ckpt, 13 complete) | `REACTIVE-FASTER` | +7.95 % | 0.0015 | 0/13 |
| L2 immediate rule vs reactive (8 env) | `RULE-BEATS-REACTIVE` | **−26.42 %** | 0.012 | 8/8 |
| batched rule vs reactive (8 env) | `NOT-SEPARATED` at n = 8 | −20.19 % | 0.21 | 6/8 |
| random vs reactive (8 env) | `REACTIVE-FASTER` | +22.46 % | 0.012 | 0/8 |
| L3 `gnnedge0` vs immediate rule (ckpt) | `RULE-FASTER-THAN-GRAPH-ARM` | **+25.01 %** | 0.0004 | 0/16 |
| L5 `gnnedge0` vs `mpoff` (ckpt, 13 complete) | `GRAPH-FASTER-THAN-TWIN` | −9.79 % | 0.0024 | 12/13 |

Per task (medians, s: queue / exchange / rendezvous / elapsed): Knative 3.86 / 4.34 / 0 / 8.29;
random 5.01 / 4.43 / 0 / 9.37; **rule 2.89 / 3.22 / 0 / 6.24**; batched rule 2.99 / 3.26 / 0 / 6.56;
`gnnedge0` 4.04 / 3.92 / 0 / 7.96; `mpoff` 4.63 / 4.11 / 0 / 8.96. The graph arm, with no wait at
all, saves 0.4 s of exchange and gives 0.2 s back in queue — a 2 % tie with Knative — while the
rule saves 1.1 s of exchange **and** 1.0 s of queue. The model-class contrast survives the lever
(`gnnedge0` beats its twin −9.8 %) and is irrelevant to the ranking: both learned arms sit between
random and the rule.

**What this settles.** (1) The 7 s wait was never the learned arm's whole deficit: with the wait
at zero it still does not beat Knative, and the no-wait decoder inherits this prediction — its bar
is the rule at −26 %, not Knative. (2) Bursts are a load lever at 6 servers before they are a
structure lever: reactive hangs on 22 of 48 candidate cells (starved-client spin) and the batch
path hangs on a topology reactive survives (Amendment 2). (3) Everything peer-aware pays more
under bursts (rule −13 → −26 %), because with rendezvous gone the exchange term is the whole
peer cost.

**Carry.** 8 environments, two topologies: a DIRECTION read, not a magnitude; `mpoff` at 13 of
16 checkpoints (3 hang at the end of the trace on 9101 w1; 48 and 96 GB). The registered rule's
verdict stays `TOO-FEW-UNSATURATED-ENVIRONMENTS`; every number above is quoted with "8
environments, disclosed".

**Parents:** [`batch_window_edge_v1`](batch_window_edge_v1.md) (the wait is intrinsic to
peer-group placement when the group arrives over 13.8 s), [`peer_greedy_live_v1`](peer_greedy_live_v1.md)
(the rule arms served here), [`unsaturated_edge_v1`](unsaturated_edge_v1.md) (statistic, baselines,
checkpoints).

**Question.** A "peer group" in the study trace is 10 consecutive independent arrivals of a
0.46/s stream, so its members span **13.8 s (median; p90 32 s)** and a scheduler that wants to
see the group before deciding pays ~7 s per task for it. A real fan-out stage dispatches its
tasks together. If every group arrives **as one burst**, the wait collapses toward zero by
construction — does the learned arm then keep its co-location gain and beat reactive Knative?
This is the environmental twin of a no-wait decoder: the same 7 s removed from the environment
instead of from the policy.

**The lever** (`env_lever_v1_mint.py burst`). Every event's timestamp becomes the earliest
timestamp of its `peer_group`; task types, source clients, QoS, peer pairs and payloads are
untouched, and the mean rate is unchanged (0.460 arrivals/s). Nothing is retrained. **Caveat
registered in advance:** a burst of 10 raises the instantaneous load on 6 servers, so the
lever gets its **own admissibility screen** (L0: reactive on 12 candidate topologies × 4
windows at C40; a cell enters the study at reactive queue share ≤ 0.80; **unknown is not a
pass**) and its study runs on the 4 lowest-numbered topologies admissible on all 4 windows —
which need not be the parent study's four.

**Arms** at C40 on the lever's 16 environments: reactive (the screen), `random_network`,
`peer_greedy_network`, `peer_greedy_network_batch`, `be1670_gnnedge0` × 16 checkpoints,
`1670_mpoff` × 16 (the model-class twin, because with the wait gone the E2 contrast can be
read where it matters). 608 runs.

**Bars** (`scripts_cosim/env_lever_v1_read.py`; chain's: |median| ≥ 5 %, p < 0.05, signed-rank;
learned arm unit = checkpoint n = 16, rule unit = environment n = 16):

| read | what | fires as |
|---|---|---|
| L0 | the lever's screen | `DESIGN-READY` / `TOO-FEW-UNSATURATED-ENVIRONMENTS` (then nothing below is run) |
| **L1** | `gnnedge0` vs reactive | `ARM-BEATS-REACTIVE-UNDER-THE-LEVER` / `REACTIVE-FASTER-UNDER-THE-LEVER` / `NOT-SEPARATED` |
| L2 | `peer_greedy_network` vs reactive (env) | `RULE-BEATS-REACTIVE-UNDER-THE-LEVER` / … |
| L3 | `gnnedge0` vs `peer_greedy_network`, GRAPH FIRST, paired on env and checkpoint | `GRAPH-ARM-FASTER-THAN-RULE` / `RULE-FASTER-THAN-GRAPH-ARM` / `GRAPH-ARM-MATCHES-RULE` |
| **L4** | did the lever do what its name says: `gnnedge0` median scheduler wait per task < 1.0 s | `WAIT-COLLAPSED` / `WAIT-DID-NOT-COLLAPSE` — if it did not, L1 is not a burst result |
| L5 | `gnnedge0` vs `mpoff`, GRAPH FIRST | `GRAPH-FASTER-THAN-TWIN-UNDER-THE-LEVER` / `POINTWISE-FASTER…` / `NOT-SEPARATED` |

Hung checkpoints make a learned-arm read UNREADABLE by the letter; the read on the complete
checkpoints is printed as **disclosed**.

**Registered expectation (signed 2026-09-20; two smoke runs seen).** Locally on `cc40s9001` w0
with the burst workload: Knative **10.68 s** per task (from 18.12 s unburst: rendezvous 3.3 → 0,
queue 10.0 → 5.6) and the immediate rule **7.85 s** (−26.5 %). So the lever helps *everyone*:
Knative loses its rendezvous too, because partners placed within milliseconds of each other are
never waited for. **L4 WAIT-COLLAPSED 95 %.** **L1: BEATS 45 %, NOT-SEP 30 %, REACTIVE-FASTER
25 %** — with the wait gone the arm's margin is the exchange it saves minus the queue its
concentration adds, against a Knative that is itself 40 % faster. L2 BEATS 75 %. L3 MATCHES
45 %, RULE-FASTER 35 %, GRAPH-FASTER 20 %. L5 GRAPH-FASTER 60 %. L0 DESIGN-READY 75 % (the
smoke cell's queue share is 0.53).

**Consequences, signed in advance.**
- L1 fires: the first learned-arm win over healthy reactive in the programme, and it is
  conditional on the arrival structure — the paper's environment claim becomes "fan-out
  arrivals", and the no-wait decoder is unnecessary for it.
- L1 does not fire while L4 fires: removing the wait from the environment does not rescue the
  arm; its deficit is not the wait alone (concentration remains), and the no-wait decoder
  inherits that prediction.
- L3 MATCHES or RULE-FASTER: the burst regime is one where a rule does what the model does.
- L0 fails: the lever is a load lever at 6 servers, recorded as such; the study does not run.

**Cost.** 48 screen + 560 study ≈ 608 runs (~1 h at 48-wide). Results under
`simulation_data/peer_affinity_live_gate/results/el_v1/burst/`; selection
`simulation_data/env_lever_v1/selected_burst.json`.

**Datasets.** None; nothing trained.

**Amendment 1 (signed 2026-09-20, after L0 and before any study arm).** L0 read
`TOO-FEW-UNSATURATED-ENVIRONMENTS`: bursts of 10 hang reactive on 22 of 48 cells (the
starved-client spin on 9002/9005/9102/9103/9107/9108 and on 9001 w1, 9104 w0) and saturate 9002 w0
(0.894) and 9105 w0 (0.871); only **9003, 9101, 9106** are admissible on all four windows, one short
of the registered four. By the registration the study does not run and the lever is recorded as a
load lever at 6 servers. **This amendment runs the study on the 12 admissible environments
anyway** (3 topologies × 4 windows; `selected_burst_a1.json`), with every rule-vs-baseline read
printed at n = 12 as **DISCLOSED**, never in the registered slot — the n = 16 bar is a power bar,
not an admissibility bar, and all 12 cells clear the admissibility bar (shares 0.43–0.55). The
checkpoint-unit reads (L1, L3, L5) keep n = 16 checkpoints over 12 environments. 420 runs. The
lineage's registered verdict stays L0's; the amendment's reads are quoted with "12 environments,
disclosed" attached, always.

**Amendment 2 (signed 2026-09-20, after the first study block of Amendment 1).** Under bursts the
**batch-path arms cannot be served on topology 9003**: all 12 `gnnedge0` checkpoints on 9003 w0
and the batched rule on 9003 w0 and w1 enter the starved-client spin early in the trace
(`No compatible hardware available for dnn2 on nodes with connectivity to client_node8`, from
sim-time 2,078–19,794 s onward), while reactive, random and the per-arrival rule finish on all 12
cells. A burst of 10 leaves the batch path's deferred tasks with no replica it can create; the
per-arrival path places them one at a time as replicas appear. The 14 hung arms were cancelled
(job 793450). **The study narrows to the 8 environments both paths can serve (9101, 9106 × 4
windows; `selected_burst_a2.json`)**; the per-arrival results already on disk for 9003 stay in
the result directory and are reported alongside, and every environment-unit read is DISCLOSED at
n = 8. The registered verdict stays L0's.

## Record (newest first)

- 2026-09-20 — **CLOSED under Amendment 2** (jobs 793604 / 793853-class blocks / 794190 / 794454:
  280 arms, 277 completed, 3 `mpoff` OOM at 48 GB on 9101 w1, re-run at 96 GB cancelled by the 20-minute watchdog at the same point, job 794555 (deterministic; disclosed)).
  Read `env_lever_v1_gate_read.py study burst --selection selected_burst_a2.json`: as in the head.
  L4 median wait 0.001 s.

- 2026-09-20 — Amendment 1's first block (job 793450, tasks 0–47): random 12/12, immediate rule
  12/12, batched rule 10/12 (9003 w0, w1 hang), `gnnedge0` seeds 1–12 on 9003 w0 0/12 — all 14
  hangs are the starved-client spin on 9003. Amendment 2 signed (above).

- 2026-09-20 — **L0: `TOO-FEW-UNSATURATED-ENVIRONMENTS`** (job 793207, 48 reactive arms: 25 completed,
  22 cancelled at 20 min, 1 OOM). Screen table in `simulation_data/env_lever_v1/selected_burst.json`.
  Amendment 1 signed (above); the 12-environment study runs next.

- 2026-09-20 — Registered; local smoke as disclosed above.
