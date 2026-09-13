# mp_parity — FALSIFIED

> **Status:** `FALSIFIED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-17 → 2026-08-17

**Outcome.** Both arms falsified, gate FAILED on both pre-registered criteria. The residual pays only where interaction exists and costs RTT where the target is additive — which is `graph_structure_physics`' finding arriving from the model side.

**Entry points:** `scripts_cosim/test_train_serve_mp_parity.py`, `experiments/full_corpus_siv1_gnn_mp_residual{,_node_edges}.yaml`, `datalab/mp_arm_gnn_train.sbatch`

**Datasets:** full corpus siv1

**Related:** [siv1_full_corpus](siv1_full_corpus.md) · [graph_structure_physics](graph_structure_physics.md)

## Standing (from the index table)

Train/serve message-passing parity, and what to do about it. Outcomes below.

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [mp_parity — outcomes (2026-08-17)](#mp-parity-outcomes-2026-08-17)

---

### mp_parity — outcomes (2026-08-17)

**Root cause.** `train_near_rtt.py` fitted `self.gin(x, data.edge_index)` (bipartite only)
while the serving copy in `src/policy/gnn/gnn_model.py` concatenated every same-node
platform↔platform edge — ~26:1 more edges than bipartite on the full-corpus cache. The
served model ran message passing on a graph its weights had never seen. Fixed by making
same-node edges opt-in, and structurally by deleting the second copy of the model: the
trainer now imports the one definition.

**Baseline gate** (`normal_sim_sweeps/gnn_mp_parity_gate_20260816`, deployed checkpoint
with parity fix, 3 configs × 5 seeds):

| config | GNN/Kn | MLP/Kn | GNN cell wins | p99 winner |
|---|---|---|---|---|
| sparse_p25 | 1.14x | 0.83x | 0/5 | mlp |
| sparse_p25_skew | 0.84x | 2.27x | 3/5 | **gnn** (71.0s vs MLP 498.5s) |
| sparse_p35 | 1.02x | 0.77x | 0/5 | mlp |

Pre-registered PRIMARY (GNN > MLP on total_rtt in ≥2 of 3 configs) = 1/3 **FAILED**.
TAIL (same on p99) = 1/3 **FAILED**. The parity fix removes the 12.4x catastrophe but the
fixed baseline still loses to MLP on the two large-RTT configs. It does reproduce the
pre-registered *collision cliff* on `sparse_p25_skew`, where the MLP is catastrophically
unreliable (2.27x Knative) and the GNN is not — a bounded claim, not a general win.

**`FALSIFIED` — same-node edges.** Arm B (`full_corpus_siv1_gnn_mp_residual_node_edges`)
trained *with* candidate-restricted same-node edges (0.37x bipartite, present on 80% of
graphs, recorded in the checkpoint sidecar) and was worse than Arm A on every metric:
val acc 62.6% vs 65.6%, test greedy regret 0.4944s vs 0.2621s. Co-location coupling is
not the signal the GNN was missing. Do not re-try this without new evidence.

**`FALSIFIED` — the GIN residual, with one instructive exception.** Arm A
(`full_corpus_siv1_gnn_mp_residual`) more than halves offline greedy regret vs the
deployed baseline (**0.5682s → 0.2621s, −54%**; top-5 0.0346 → 0.0239) and learns
`mp_gate` = 1.08, i.e. it leans on message passing *more* once MP augments rather than
replaces the per-node encoding. **None of that transferred.** Live re-gate
(`normal_sim_sweeps/mp_residual_gate_20260817`, 15/15, `compare.json` + `manifest.json`):

| config | baseline GNN/Kn | Arm A GNN/Kn | delta | MLP/Kn |
|---|---|---|---|---|
| sparse_p25 | 1.14x | 1.18x | +3.5% | 0.83x |
| sparse_p35 | 1.02x | 1.12x | +9.9% | 0.77x |
| sparse_p25_skew | 0.84x | **0.80x** | **−4.3%** | 2.27x |

PRIMARY **1/3**, TAIL **1/3** — both still FAILED, paired wins identical to baseline
(GNN 3/15 · MLP 10/15 · Kn 2/15), SUM 1.05x → 1.12x. **The sign flips with coupling:**
the residual costs RTT on the two near-pointwise configs and pays only on
`sparse_p25_skew` (also p99 71.0s → 63.1s), the one config with real interaction. That is
the `graph_structure_physics` prediction measured directly — graph capacity is wasted, and
actively harmful, wherever the target is additive. A mid-session read that
`logit_tied_rate` was the discriminator is **wrong**: it rose on all three configs,
including the one that improved.

⇒ Model-side work on this corpus is closed. The next lever is physics (Phase 1
`node_contention_v3`), not architecture.

**Two reproducibility traps found, both still open.**
1. `run_provenance` records neither the git commit nor `OMP_NUM_THREADS`. The
   2026-08-16 ablation figure of 0.88x Knative on `sparse_p35/s42` is **not reproducible**
   — the current gate gives 1.04x for the same cell/seed/model/config. That arm ran with
   `GNN_DROP_NODE_EDGES=1`, a variable implemented nowhere in the tree today, so the code
   that produced it no longer exists. Treat the gate as the baseline of record.
2. `logit_tied_rate ≈ 0.54` — the scoring head's top-2 margin is under 0.1 on half of all
   live decisions. A model that indifferent is sensitive to FP reduction order, which is
   why thread count matters. If the residual does not move this, the next lever is the
   ranking loss or edge features, **not** the encoder.

---
