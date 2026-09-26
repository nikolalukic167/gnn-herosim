# exchange_seconds_v1 — does pricing peer exchange in seconds, not per-batch-normalised, close the rest of the CD gap?

**Status:** `ACTIVE` (2026-09-26). Registered and training. Every bar below was signed before any datum
of this arm existed.

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

- 2026-09-26 — Registered.
