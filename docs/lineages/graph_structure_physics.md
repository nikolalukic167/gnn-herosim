# graph_structure_physics — ACTIVE

> **Status:** `ACTIVE` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-17 → 2026-08-17

**Outcome.** **The co-sim target is pointwise-separable, so a pointwise MLP is the correctly specified model class and the GNN cannot beat it by training.** Additive R² 0.988 → 1.00000 across collections. Deep queues as a coupling lever: FALSIFIED — the lever runs backwards.

**Entry points:** `scripts_cosim/separability_diagnostic.py` (M4 + `--gate-additive-r2`)

**Datasets:** all co-sim collections

**Related:** [throughline](throughline.md) · [shallow_v1](shallow_v1.md) · [route_a_v1](route_a_v1.md) · [program_verdict_v1](program_verdict_v1.md)

## Standing (from the index table)

Does the simulator produce a target a GNN could ever beat a pointwise MLP on? **Outcome 2026-08-17 below: no, not today.** Phases 1-4 (node contention, congestible links, fan-out DAGs, batch size) planned against that measurement.

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [graph_structure_physics — outcomes (2026-08-17)](#graph-structure-physics-outcomes-2026-08-17)

---

### graph_structure_physics — outcomes (2026-08-17)

**The co-sim target is pointwise-separable, so a pointwise MLP is the correctly specified
model class and the GNN cannot beat it by training.** This reframes `mp_parity`: the GNN
did not lose to the MLP because of a bug, it lost because there is nothing non-pointwise
left to learn.

Measured by the new **M4** block in `separability_diagnostic.py`, which fits
`rtt(plan) ≈ μ + Σ_t f_t(plan[t])` — exactly what `PointwiseEdgeMLP` can express — by least
squares over every plan in `placements.jsonl`:

| collection | additive R² | additive-fit argmin regret | datasets where additive = optimum |
|---|---|---|---|
| contention_v2 (80 ds) | 0.98812 | 0.29% mean | 81% |
| contention_v4_pilot (27 ds) | 0.99973 | 0.00% | **100%** |
| contention_v5_quick_test (35 ds) | 0.99973 | 0.02% | 97% |
| highq_safe_20260606 (25 ds) | 1.00000 | 0.00% | **100%** |

**`FALSIFIED` — deep queues as a coupling lever, confirmed on the full 899-dataset
contention_v2 corpus.** Queue depth *predicts* separability, monotonically:

| quartile | mean queue depth | additive R² | collision R² gain |
|---|---|---|---|
| shallowest 25% | 27.6 | **0.97822** | **+1.986 pp** |
| middle 50% | 35.5 | 0.99211 | +0.672 pp |
| deepest 25% | 50.8 | 0.99803 | +0.181 pp |

`corr(depth, additive_r2) = +0.256`, `corr(depth, collision_gain) = −0.259` (n=899). The
collision term is **11× weaker** in the deepest quartile than the shallowest.

The mechanism is arithmetic: queue work is `depth × exec_time` and grows with depth, while
the interaction term `added_in_batch × exec_time` does **not** — so deeper queues dilute
the only coupling the corpus has. contention_v4/v5 deepened queues further and landed at
R² 0.9997. **The series' core lever is backwards.** Do not extend it.

**The lever that follows: shallow queues plus long-exec task types** (cnn is 3.09 s on
rpiCpu vs dnn2's 0.024 s), making `added_in_batch × exec_time` dominate rather than
vanish. A grid-preset change, not a physics change. Note the corpus only spans depth
26–56.5, so depths below ~26 are an extrapolation — but the trend is monotone across all
899 datasets and the mechanism explains it.

Two grids added: **`shallow_v1`** (shallow queues, stock dnn1/dnn2 — isolates the measured
lever) and **`shallow_longexec_v1`** (adds the `("cnn", "rf")` task pair via the new
`task_type_pair` preset key).
