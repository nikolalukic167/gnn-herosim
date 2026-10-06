# hetero_conv_v1 — do relation-specific weights let the raw-plan GNN close its gap to CD?

**Status:** `REGISTERED` (2026-10-06). Arms, topologies, statistic and verdict below were fixed before any training run.

**Outcome.** None yet.

**Question.** `lf1gnn` ([`local_features_v1`](local_features_v1.md)) is the one arm where message passing clearly beats
the same-input MLP (−11 / −24 / −20 %) and its no-conv twin (−13 / −17 / −8 %), but CD is faster at every rung
(+14 / +34 / +6 %). Its bipartite stack (`BipartiteEdgeConv`) runs **one** message MLP over three kinds of edges —
task→replica, replica→task (the candidate edges are made undirected) and replica↔replica on the same node (the raw-plan
path) — and **one** update MLP for tasks and replicas alike. Does giving each relation and each node type its own weights,
as a heterogeneous GNN does (R-GCN / HeteroConv), close that gap?

Already tested, so not part of this lineage: edge physics inside the convs (`raw_plan_v2` `rawE` / `rawES`, both worse
than `rawS`), sum instead of mean aggregation, and edge conditioning as such (`bipartite_edge_v1`).

**The change** (`HeteroBipartiteEdgeConv`, `src/policy/gnn/gnn_model.py`; env `NEAR_RTT_MP_BIPARTITE_HETERO`, sidecar key
`mp_bipartite_hetero`, weight-visible): a message MLP per relation (t2p, p2t, p2p), mean over each relation's edges,
the per-relation means summed at each node (PyG HeteroConv's default), then a task update MLP and a replica update MLP.
Same message inputs, depth (3), width (64) and everything else as `lf1gnn`. Tests: `tests/test_hetero_conv.py` (each
weight block reaches only its own node type; hetero and plain checkpoints refuse to load into each other).

**Arms** (small-batch corpus, cache and split of `small_batch_v1`; 100 epochs, best validation checkpoint; 8 seeds):
- `lf1het`: `experiments/hetero_conv_v1_lf1het.yaml` — `lf1gnn`'s config verbatim plus the flag.
- References, reused from their gates on the same cells: `lf1gnn`, `lf1twin` (the MP-OFF twin — with every conv skipped
  `lf1het` and `lf1gnn` are the same model, so `lf1twin` is the twin of both), `lf1mlp`, CD, `cdextr`, `sb1load`,
  `sb1mpoff`.

**Topologies and windows.** `small_batch_confirm_v1`'s 19 unseen topologies × grounded windows g0–g3 × {×2, ×3, ×5},
phase `het1` of `scripts_cosim/fresh_topo_burst_v1_gate.py`, original (scattered-origin) workload, server-only cells.
Disclosure: these topologies have been read several times before (`small_batch_confirm_v1`, `local_features_v1`,
`client_local_v1` among them), so they are no longer fresh; a win here is a candidate for confirmation on new topologies,
not a confirmed result.

**Statistic.** Per topology, the median paired % over (window, seed); learned arms pair on the seed, rules on their single
run. Median over the 19 topologies, exact two-sided Wilcoxon over topologies.

**Primary family** (Holm across 9): `lf1het` vs {`lf1twin`, `lf1gnn`, CD} × {×2, ×3, ×5}. CONFIRMED = median ≤ −5 % and
Holm p < 0.05. Verdict, first that holds:
- **HETERO-BEATS-CD-AND-TWIN** — the same ≥ 2 rungs at which vs CD and vs `lf1twin` are both CONFIRMED, and no reference
  Holm-confirmed faster at any rung;
- **HETERO-HELPS** — vs `lf1gnn` CONFIRMED at ≥ 2 rungs and `lf1gnn` Holm-confirmed faster at none;
- **NO-GAIN** — anything else;
- **INCOMPLETE** — a primary test short of 19 topologies, or a primary run missing without a failure record from one
  rerun at 3× the timeout (as `local_features_v1` Amendment 2).

Descriptive only: `lf1het` vs `lf1mlp`, `cdextr`, `sb1load`, `sb1mpoff`.

**Entry points.** Training `scripts_cosim/datalab/hetero_conv_v1_train.sbatch`; gate phase `het1`; reader
`scripts_cosim/hetero_conv_v1_read.py`. Cost: 8 training jobs (16 CPU each, as `lf1gnn`), then 1,824 gate runs.

## Record

- 2026-10-06 — Registered. Code, config, training script, gate phase and reader written; unit tests and
  `test_trainer_determinism.py` pass. Nothing trained yet.
