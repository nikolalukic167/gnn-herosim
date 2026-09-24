# fresh_topo_burst_v1 — does the burst-seat message-passing edge hold on topologies never studied?

**Status:** `REGISTERED` (2026-09-24). Every bar below was signed before the screen or gate produced
any data; the one prior read of these statistics (the old 4-topology study, exploratory) is disclosed.

**Parents:** [`selfpredict_burst_v1`](selfpredict_burst_v1.md) (`gnnedge0` beats self-predict −7.25 %,
13/13 checkpoints, on 4 topologies), [`joint_burst_v2`](joint_burst_v2.md) (the checkpoints; K4
`gnnedge0` vs its MP-OFF twin −4.4 %, under the 5 % bar).

## Why this lineage exists

A post-close exploratory read of the existing burst gates (2026-09-24, no bars; the `mpoff` arm from
the `joint_burst_v2` gate, comparable because all 208 fresh `gnnedge0` cells reproduce that gate to the
digit) found:

- `gnnedge0` vs MP-OFF, same seed: −4.72 % per environment, **faster on 16/16 environments and on all 4
  topologies** (−3.4 to −5.5 %; leave-one-topology-out −4.4 to −5.1 %).
- MP-OFF vs self-predict: −2.0 %, p = 0.71, 9/16, so a learned arm WITHOUT message passing does
  not clearly beat the per-arrival rule.
- `gnnedge0` vs self-predict depends on topology: −11.5 / −22.4 / +2.4 / −7.6 % by topology, and it
  loses on 9114.

That is the first live contrast in the programme where the message-passing direction held everywhere
it was measured. But it covers 4 topologies, and the 13 checkpoints are training draws, not new
infrastructure. This lineage asks whether it holds on topologies neither the model nor any study has
seen. Nothing is trained: the gate reuses the 13 jb2 `gnnedge0` and 13 jb2 MP-OFF checkpoints.

## Design

- **Candidate pool (48 topologies, 6 servers / 40 clients, burst workloads, 4 windows):** the 24
  `joint_burst_v1` pool members that were never a study topology (9001–9005, 9102–9105, 9107–9113,
  9115, 9117–9124, which were screened before the 2026-09-24 scale-down fix, and 23 of 24 failed on a
  hang). Plus 24 newly minted cells, 9401–9424, from the same base with the same generator
  (`cluster_scale_v1_cells.py`); a re-mint of 9109 is byte-identical to datalab's. None is a training
  topology (9201–9224). 9001/9003/9005 were unburst study cells elsewhere, never under bursts.
- **Screen (F0, `joint_burst_v1`'s rule, unchanged):** a topology qualifies iff reactive
  `knative_network`'s queue share ≤ 0.80 AND the batch path (`peer_greedy_network_batch`) finishes,
  on all four windows; unknown is not a pass. **Study = the 12 lowest qualified ids**; fewer than 8
  qualified is `DESIGN-SHORT`.
- **Gate arms, fresh at one commit on the study environments (topology × 4 windows):** self-predict
  (per arrival), CD greedy, 1-pass greedy (batched seat), `knative_network`, and jb2 `gnnedge0` and jb2
  MP-OFF UNCAPPED × the 13 shared seeds (1–5, 7–10, 12, 14–16), served exactly as `joint_burst_v2` did
  (masked_topo, peer-group batching, the cell's 16 s window, `GNN_PREFIX_ALPHA_KEY=inf`; each
  sidecar checked by `joint_burst_v2_sidecheck.py`). 120 runs per topology.
- **Venue:** local, one commit, 12 runs in parallel, each in its own 2.5 GB cgroup with a 1800 s
  timeout. Venue parity is measured, not assumed: on 9101 w1, self-predict, CD, `gnnedge0` s1 and
  MP-OFF s1 run locally reproduce the datalab gates' `total_rtt` **to the digit, 4/4**. All inputs
  (cells, workloads, 26 checkpoints + sidecars, the jb2 split) are md5-identical to datalab.
- **Failure rule:** any study run that times out, hits its memory cap or fails an instrument check
  drops its whole topology from every contrast, reported by name. Fewer than 8 complete is `DESIGN-SHORT`.

## Bars (signed 2026-09-24, before any screen or gate data)

Unit = topology. Per environment, a learned arm = the median over its 13 checkpoints of the paired %
against the reference (`gnnedge0` vs MP-OFF pairs the same seed). Per topology, the median over its 4
windows. Two-sided exact Wilcoxon signed-rank over topologies, α = 0.05. Negative = first arm faster.

| read | contrast | fires as |
|---|---|---|
| **F1 (primary)** | `gnnedge0` vs MP-OFF | `GNNEDGE0-FASTER` (≤ −5 %) / `GNNEDGE0-FASTER (direction only)` (p < 0.05, \|median\| < 5 %) / `MPOFF-FASTER…` / `NOT-SEPARATED` |
| **F2** | `gnnedge0` vs self-predict | same labels, `SELFPREDICT` in place of `MPOFF` |
| **F3** | MP-OFF vs self-predict | same labels |
| — | disclosed: `gnnedge0` vs CD, self-predict vs CD, `gnnedge0` vs 1-pass, `gnnedge0` vs reactive; per-topology tables | — |

**Verdict (from F1):** `MP-EDGE-GENERALISES` (qualified `(direction only)` if |median| < 5 %);
`MPOFF-FASTER-ON-FRESH-TOPOLOGIES`; `MP-EDGE-DOES-NOT-GENERALISE` (not separated). F2 and F3 are
reported beside it and never upgrade it. A direction is quotable from this design; a magnitude only
if it clears 5 %.

**Disclosed before data: the old study under THIS statistic** (reader fixture, 3 complete topologies;
9116 dropped by the failure rule for two missing MP-OFF cells): F1 −4.1 / −4.6 / −4.1 %; F2 −10.7 /
+0.4 / −0.5 % (median −0.5 %). The median over windows discounts the saturated 9106 w0 cell, so
most of the old "beats self-predict" lead comes from one topology.

**Expectations, written before data:** F1: direction replicates 55 %, of which magnitude ≥ 5 % in
~20 %; not separated 40 %; MP-OFF faster 5 %. F2: `gnnedge0` beats self-predict by ≥ 5 % 20 %, not
separated 65 %, self-predict faster 15 %. F3: not separated 70 %. CD ahead of every arm: 90 %.

## Entry points

- Driver: `scripts_cosim/fresh_topo_burst_v1_gate.py` (`screen` / `parity` / `gate`).
- Reader: `scripts_cosim/fresh_topo_burst_v1_read.py` (`select` / `gate`).

## Record (newest first)

- 2026-09-24 17:01 — **Amendment A2 (contingency), signed while 7 candidates' batch-path runs were
  still unread.** At this point 1 topology had qualified (9119) and 7 new cells had passed reactive on
  all windows, with their batch path pending; the other 40 candidates were out. Most old-pool cells spin
  even after the fix. If the screen ends with fewer than 12 qualified, the pool extends with **48
  newly minted cells, 9425–9472** (same base, same generator, range checked unused), screened under
  the same rule and A1's timeout. The study is still the 12 lowest qualified ids of the combined pool,
  and fewer than 8 after the extension is final `DESIGN-SHORT`. The extension runs to reach 12, not
  just 8, so a single gate-time drop cannot sink the design.
- 2026-09-24 — **Amendment A1 (screen only), signed before any screen result was read.** The first
  screen attempt (1800 s timeout, 12 parallel) finished 10/384 runs in 12 min: every slot was held by
  reactive runs on old-pool cells (9001 w1, 9002 w1–w3, 9005, 9102, 9103) spinning at 100 % CPU with
  flat ~0.8 GB memory, i.e. the known starved-client spin, not the fixed stranding bug. These same
  cells had no reactive result in the pre-fix screen either. The screen now uses a **600 s timeout
  and 20 parallel runs**. A normal screen run takes ~60 s. A run that would finish between 600 and
  1800 s is saturated far past the 0.80 queue-share filter, so the admission rule is unchanged: a
  timed-out run is still a failure, and unknown is not a pass. The 10 finished summaries are kept.
  **The gate keeps its 1800 s timeout.**
- 2026-09-24 — **Registered.** Inputs md5-verified against datalab; mint determinism and venue
  parity (4/4 to the digit) checked; reader tested on the old-study fixture (drop rule fires).
