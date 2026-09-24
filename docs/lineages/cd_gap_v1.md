# cd_gap_v1 — why does the burst-seat GNN lose to the CD greedy, and can imitating CD or changing load close it?

**Status:** `REGISTERED` (2026-09-25). Every bar below was signed before its data.

**Parents:** [`fresh_topo_burst_v1`](fresh_topo_burst_v1.md) (on 11 fresh topologies `gnnedge0` is
+11.4 % behind CD, 0/11), [`joint_burst_v2`](joint_burst_v2.md) (the checkpoints and the corpus; label =
the group sweep optimum, which offline strictly dominates CD).

## Why

The one learned arm that reaches the self-predict rule in the burst seat still loses to the CD greedy
by ~11–12 % everywhere. **The decomposition of the fresh gate (medians over runs, per task) puts the
whole gap in queue, not co-location:**

| arm | elapsed | queue | exchange |
|---|---|---|---|
| CD greedy | 7.62 s | 2.89 s | 4.19 s |
| `gnnedge0` | 8.64 s | 4.13 s | 4.23 s |
| self-predict | 8.54 s | 4.22 s | 4.14 s |
| MP-OFF | 8.97 s | 4.45 s | 4.35 s |

About 10 % of partner links fall outside the served batch (`prefix_peers_outside_batch` 18,050 vs
79,928 in-batch pairs per `gnnedge0` run). The record's leads for the gap are (a) the model does not
reach its label (v1 `gnnedge0` offline regret 27.5 s vs CD 11.2 s on 48 groups; never measured for
v2), (b) it is blind to partners outside its batch, (c) the offline→live gap: training states are
the 1-pass rule's, labels are group-local. None of them has been isolated.

## Design

Nothing here is a new model until A. Probes run on existing checkpoints and code, ordered cheapest-first.

- **D0, fit vs gap (offline, orders the work).** On the 480 `joint_burst_v2` held-out groups (cells
  9222–9224, never trained on): per group, regret % = 100 × (rtt − sweep optimum) / optimum, for CD and
  the 1-pass greedy (`joint_burst_v1_offline_rule_regret.py`, the engine replay) and for the 13 jb2
  `gnnedge0` and 13 jb2 MP-OFF checkpoints (`eval_route_b_stage2_arm.py`, alpha key `inf`, the decode
  `peer_affinity_live_serve_check.py` proved identical to serving). Learned arm per group = median over
  checkpoints. Unit for direction = cell × window; groups are described, not treated as independent.
- **D1, out-of-batch information (live).** `peer_greedy_network_cd` with `HEROSIM_PG_BATCH_BLIND=1`:
  identical except a partner outside the current batch is never priced, exactly what the served GNN
  sees. Paired against CD on the `fresh_topo_burst_v1` study (11 topologies × 4 windows), statistic
  as that lineage's (per topology median over windows, exact Wilcoxon over topologies).
- **A, CD imitation (trained, live-gated).** jb2 `gnnedge0` recipe, same corpus and split, label =
  CD's plan on each snapshot (via the engine replay) instead of the sweep optimum. 4 seeds. Offline:
  regret vs the sweep optimum and plan agreement with CD on held-out. Live: on the fresh study, vs CD
  and vs jb2 `gnnedge0`. Its ceiling is a tie with CD; it is a diagnostic of the offline→live gap
  (does a model that reproduces CD offline reproduce it live?), not a route to beat CD.
- **B, burst load ladder (live).** The fresh study at a lighter admissible rate (arrival factor
  ×2 slower than the current burst workloads) with every policy time constant scaled by the same
  factor (hard-stop rule, `drainable_regime_v1`). Arms: CD, self-predict, jb2 `gnnedge0` × 13, jb2
  MP-OFF × 13. A topology whose reactive screen fails at that rate is dropped by name.

## Bars (signed 2026-09-25, before any D0/D1/A/B datum)

| read | contrast | fires as |
|---|---|---|
| **D0** | `gnnedge0` vs CD, offline regret % on held-out | `FIT-GAP` if `gnnedge0`'s median regret ≥ CD's (it does not reach CD even offline); `OFFLINE-LIVE-GAP` if it is below CD's by ≥ 1 pp median and in every cell × window; else `MIXED` |
| D0b | `gnnedge0` vs MP-OFF, offline | reported (does MP's live direction exist offline?) |
| **D1** | blind-CD vs CD, live | `OUT-OF-BATCH-MATTERS` (≤ −5 % for CD, p < 0.05) / `(direction only)` (p < 0.05, \|median\| < 5 %) / `NOT-SEPARATED` |
| **A1** | CD-imitator vs CD, offline | `IMITATES-CD` if its median regret is within 1 pp of CD's; else `CANNOT-FIT-CD` |
| **A2** | CD-imitator vs CD, live (fresh study) | as D1's labels; with A1 = `IMITATES-CD`, a live loss ≥ 5 % is `OFFLINE-LIVE-GAP-CONFIRMED` |
| A3 | CD-imitator vs jb2 `gnnedge0`, live | reported |
| **B1** | `gnnedge0` vs CD at the lighter rate | as D1's labels |
| B2 | `gnnedge0` vs MP-OFF, `gnnedge0` vs self-predict at the lighter rate | reported, with the rate named |

**Expectations, written before data.** D0: `FIT-GAP` 60 % (v1's 27.5 vs 11.2 s), `MIXED` 25 %,
`OFFLINE-LIVE-GAP` 15 %. D1: `NOT-SEPARATED` or direction only 75 % (only ~10 % of partner links are
outside and CD's lead is queue, not exchange). A1: `IMITATES-CD` 45 %. A2 given A1: CD faster ≥ 5 %
40 %. B1: CD still faster 85 %.

**Rule 6.** D0 is offline and only orders the work; it closes nothing. D1, A and B are live.

## Entry points

- D0: `scripts_cosim/joint_burst_v1_offline_rule_regret.py`, `scripts_cosim/eval_route_b_stage2_arm.py`.
- D1/A/B live: `scripts_cosim/fresh_topo_burst_v1_gate.py` (its study selection and failure rule).

## Record (newest first)

- 2026-09-25 — **Registered.**
