# exchange_seconds_v1 — does pricing peer exchange in seconds, not per-batch-normalised, close the rest of the CD gap?

**Status:** `ACTIVE`. Registered 2026-09-26; the live gate was read 2026-09-26.
- **X1 `CLOSES-GAP`:** `xs1load_selfref` vs CD −0.94 %, p = 0.92, faster on 5/10 topologies.
- **X2 `NOT-SEPARATED`:** vs `fc1load_selfref` −1.34 %, p = 0.054, faster on 10/11.

The seconds encoding is a consistent small gain everywhere except the topology it targeted: 9461 got
worse (+3.9 % vs `fc1load_selfref`, +31.6 % vs CD). So the per-batch normaliser is not what separates
the learned arm from CD on 9461.

**Parent:** [`fullctx_refine_v1`](fullctx_refine_v1.md). Its F1 read `fc1load_selfref` vs CD at −1.76 %
(p = 0.76, 6/11), a tie. The loss is concentrated on 9461 (+24 %), 9434 (+8 %) and 9456 (+8 %).
[`backlog_corpus_v1`](backlog_corpus_v1.md)'s slow-topology study found that on 9461 w1–w3, CD gets
both lower exchange (1.3–1.6 s per task) and lower queue. Self-refine does not help there, which points at
the score, not the decode.

**The suspect.** Peer-block columns 7–8 (committed-peer exchange, peer-mass lookahead) enter as
`seconds / peer_norm`. Here `peer_norm = max pair bytes × max per-byte cost + max latency`, taken over the
batch. The same exchange cost therefore reads differently from batch to batch, and differently from the
load columns 22–24 (log1p seconds) it has to be traded against. The design review's synthetic test
measured the mis-weighting at 0.04 s per decision.

## Design

- **Arm `xs1load`** (4 seeds): `fc1load`'s recipe verbatim (the `backlog_corpus_v1` cache and split,
  `partial_state_v4`, the `gnnedge0` architecture, full-context CE weight 1.0) plus
  `PARTIAL_STATE_EXCHANGE_SECONDS=1`.
  - Columns 7–8 become `log1p(seconds)`. No other column changes, and the cache is unchanged: the
    columns are computed from the cached raw ingredients (`peer_pairs`, `node_exchange`).
  - The flag defaults to 0, so every earlier checkpoint is unaffected. It is valid only under
    `partial_state_v4`.
  - The sidecar records `exchange_seconds`. The live loader adopts or verifies it, and a sidecar without
    the key means 0.
  - Config `experiments/exchange_seconds_v1_xs1load.yaml`, jobs
    `scripts_cosim/datalab/exchange_seconds_v1_{train,eval,servecheck}.sbatch`.
- **Served** with `service_end_v1`, as `xs1load_selfref` (3 self-refine passes, the registered arm) and
  plain `xs1load`, on the `fresh_topo_burst_v1` study (phase `xs1` of `backlog_corpus_v1_gate.sbatch`).
  - CD, self-predict, `fc1load` and `fc1load_selfref` are the existing runs.
- **Before the gate:** P1-style parity (`peer_affinity_live_serve_check.py`, 60 held-out batches, `xs1load`
  s1) must be clean.

## Bars (signed 2026-09-26, before any datum)

| read | contrast | fires as |
|---|---|---|
| **X1** | `xs1load_selfref` vs CD | `BEATS-CD` (median < 0, p < 0.05) / `CLOSES-GAP` (median ≤ 0 or p ≥ 0.05) / `CD-FASTER` |
| **X2** | `xs1load_selfref` vs `fc1load_selfref` (same seed) | `SECONDS-HELP` (≤ −5 %, p < 0.05) / `SECONDS-HELP (direction only)` / `NOT-SEPARATED` / `SECONDS-HURT` |
| X3 | `xs1load` vs `fc1load` (one-pass decode) | reported |
| X4 | `xs1load_selfref` vs self-predict | reported |

- Also reported: the per-topology X2 on 9461, 9456 and 9434, the three the change targets. This is not a bar.
- The reader is `scripts_cosim/backlog_corpus_v1_read.py`, with the same statistic and drop rule as the
  parents.
- **Expectations:**
  - X1: `BEATS-CD` 20 %, `CLOSES-GAP` 60 %, `CD-FASTER` 20 %.
  - X2: `SECONDS-HELP` (either form) 40 %, `NOT-SEPARATED` 50 %, `SECONDS-HURT` 10 %.
- **Standing risks:**
  - The 0.04 s-per-decision synthetic cost is small against a ~0.2 s queue gap. 9461's loss may be CD's
    refine passes reaching a joint move that no per-task score expresses.
  - log1p compresses large exchanges. A mis-weighting that comes from scale rather than from the
    per-batch normaliser survives this change.
  - It is one seed-set of 4 on 11 topologies: quote a direction, not a magnitude.

## Entry points

- Columns: `src/policy/tabular/reduced_features.py` (`partial_state_columns`, `exchange_seconds_enabled`).
  Tests: `tests/test_partial_state_v4.py::test_exchange_seconds_*`.
- Serving: `src/policy/gnn/prefix_serving.py` (`load_prefix_conditioned_gnn`, the `exchange_seconds`
  sidecar check).

## Record (newest first)

- 2026-09-26 — **9461 study** (two agents).
  - **Forensics** on the existing gate summaries. **Instrumented reruns** on 9461 w0–w3 of CD,
    `xs1load_selfref` s1 and `xs1load` under CD-refine shadow, on datalab at 1c72012, with raw per-task
    results kept (since deleted).
  - **The loss is w1–w3 only.** In w0, CD collapses (node5 queue 19.5 s) and the learned arm is 61 %
    faster. The window median discards that.
  - **9461 is the lightest topology.** CD's w1–w3 elapsed is 2.5–3.1 s, against 4.9–9.0 s elsewhere.
    The absolute gap (about +1 s) is still the largest of the 11.
  - **Mechanism: node choice, not sibling spread and not the normaliser.**
    - CD puts 77–79 % of tasks on node0 (4 `rpiCpu` siblings, p199–202), against 60–66 % for the GNN.
    - It keeps 76–80 % of peer bytes on one node, against 66–72 %.
    - The GNN sends `dnn1` to node5's FPGA (22.8 % vs CD's 10.9 %) and `dnn2` to the Xavier nodes
      (31 % vs 15 %). That saves 12 ms of execution and pays about 1.2 s of exchange plus queue per task.
    - Node0's queue is *lower* under CD despite more load (1.15 vs 1.46 s): the exchange stage occupies
      the platform, so co-location shortens service and with it the queue behind batch-mates.
    - Batch-mates per platform are equal (mean maximum 4.8 vs 4.3). Sibling-spread ratios match.
  - **Counterfactuals:**
    - `GNN_PREFIX_SIBLING_SPREAD=1` on `xs1load_selfref` w1/w2 changes nothing (4.274 vs 4.271 s,
      3.580 vs 3.554 s).
    - CD-refine shadow on the GNN trajectory would move 14.7 % of tasks; 75 % of those moves change node.
    - CD refine seeded by any learned plan (`gnnedge0` or MP-OFF, `cd_gap_v1` D5) matches CD on 9461.
  - **Reading:** a per-task score can express this, since CD's own greedy is one. The learned score
    over-weights device speed against exchange bytes on a topology where one many-sibling node is the
    natural exchange hub.
    - The earlier reading, CD spreading partners across siblings, is refuted by the counterfactual.
    - The seconds encoding did not fix the weighting (9461 +3.9 %).
    - A likely cause is too few hub-shaped topologies in the training corpus. That is untested.

- 2026-09-26 — **Live gate read** (job 808426, phase `xs1`, 48 min, at cca3533).
  - **Runs:** 378 of 384 wrote summaries. The misses are all 1800 s timeouts on `cc40s9423__w3`: plain
    `xs1load` s1–4 (as for plain `bc1load` and `fc1load`) and `xs1load_selfref` s1–2. By the drop rule,
    X1 loses 9423 (and 9466, CD w3), and X2 loses 9423.
  - **Attachment:** [`xs1_read.json`](exchange_seconds_v1/xs1_read.json).
  - **X1 `CLOSES-GAP`:** vs CD −0.94 %, p = 0.92, 5/10.
    - Per topology (%): 9119 −4.7, 9414 −2.7, 9420 +1.1, 9434 +5.3, 9435 −6.7, 9444 −3.9, 9446 +0.8,
      9456 +7.8, 9461 +31.6, 9469 −4.0.
    - Pooled means: elapsed 7.25 vs 7.46 s; queue 3.09 vs 2.99 s; exchange 3.93 vs 4.11 s per task.
    - Seven topologies are now faster than or within ~1 % of CD. The median is held down by the tail
      topologies, not by typical ones.
  - **X2 `NOT-SEPARATED`:** vs `fc1load_selfref` −1.34 %, p = 0.054, 10/11 faster.
    - Per topology (%): 9119 −2.4, 9414 −1.0, 9420 −0.1, 9434 −2.4, 9435 −1.3, 9444 −2.3, 9446 −1.3,
      9456 −0.0, 9461 +3.9, 9466 −2.6, 9469 −1.5.
    - Pooled: exchange 4.09 → 3.96 s, queue 3.17 → 3.09 s.
  - **X3 (reported), plain `xs1load` vs `fc1load`:** −1.02 %, p = 0.21, 8/11.
  - **X4 (reported), vs self-predict:** −14.59 %, p = 0.001, 11/11. The fastest learned arm against
    the burst-seat bar.
  - **Reading:** removing the per-batch normaliser buys a small, consistent exchange saving on 10 of 11
    topologies but does not clear the 5 % magnitude bar.
    - It does not rescue 9461, where it is worse. The registered risk holds: 9461's gap is not the
      normaliser.
    - CD's lower queue *and* lower exchange there more likely comes from joint moves in its refine
      passes, which no per-task score expresses.
    - Offline one-pass regret improved (median 4.5–5.4 vs 5.3–5.7 %); live the plain decode moved
      −1.0 %. Offline again overstates the live effect.

- 2026-09-26 — Trained (job 808272; 2 h 03 min to 5 h 17 min per seed). All 4 sidecars passed the
  `exchange_seconds` check. Best val `regret_masked_topo` was 6.49 / 6.61 / 6.72 / 6.44 s, against
  `fc1load`'s 7.23 / 7.07 / 6.93 / 7.23 s.
  - **Offline one-pass decode** (orders nothing): median regret 4.50 / 5.44 / 4.96 / 4.94 %, against
    `fc1load`'s 5.74 / 5.25 / 5.47 / 5.68 %.
  - **P1 parity** (`xs1_servecheck_xs1load_s1.json`): every field matches on 60/60 and the plan on
    59/60. The leftover is `ds_08828`, where one task got platform 199 live and 200 offline. Both are
    node0 `rpiCpu` with an empty queue and zero backlog, and both plans have sweep RTT 117.006 s.
    That is the same-node sibling tie at equal RTT that `load_repr_v1`'s standard admits (plan
    ≥ 58/60). Job 808274 exits 1 on any listed mismatch, by design.
  - Live gate phase `xs1` submitted as job 808426.
- 2026-09-26 — Registered.
