# bipartite_aggr_v1 — is the bipartite penalty a `sum` over a candidate set that grows with the cluster?

**Status:** `CLOSED` (2026-09-18) — **`SUM-COSTS-ONLY-WHERE-CANDIDATES-ARE-MANY`. The prediction
held.** Closed on a live gate at both ends of the server ladder (rule 6). Registered
2026-09-18; every bar is a module constant in `scripts_cosim/bipartite_aggr_v1_read.py`,
committed before any arm was trained.

**Outcome. The bipartite penalty is the `sum` aggregation, and it is scale-dependent exactly as
predicted.** This is the **first mechanism in the programme that predicted an existing null and
was confirmed by it.** `peer_only_v1` measured the penalty at −18.94 % on 80 servers and
**absent at 6** (p = 0.61), and that absence sat unexplained through C3, C4, C5, C6 and D1–D4.
A `sum` over a task's candidate platforms scales with the size of that set — **3.55
candidates/task at 6 servers, 47.92 at 80** (`cluster_scale_v1` S0.d, a 13.5× ratio) — so the
aggregator hypothesis predicted the penalty must appear at 80 and not at 6. It does.

**The 2×2, one value per checkpoint, all 16, both rungs.**

| contrast | what it isolates | R3 — 47.92 cand/task | R0 — 3.55 cand/task |
|---|---|---|---|
| `peeronly` vs `GIN` | **the original penalty** | **−18.94 %**, 16/16, p = 0.0004 | −4.98 %, 9/16, p = 0.6051 — **absent** |
| mean-conv vs `GIN` | aggregation + MLP shape | **−15.72 %**, 15/16, p = 0.0011 | +7.43 %, 6/16, p = 0.1089 |
| sum-conv vs `GIN` | **MLP shape ALONE** | −5.86 %, 10/16, p = 0.2343 | −3.86 %, 9/16, p = 0.9176 |
| **sum vs mean (E1/E2)** | **AGGREGATION ALONE** | **+13.26 %**, 4/16, **p = 0.0052** | −8.30 %, 10/16, p = 0.1089 |

**Read the table by column.** At 80 servers the aggregation contrast is significant and the
MLP-shape contrast is not: give the `BipartiteEdgeConv` the `GIN`'s `sum` and its −15.72 %
advantage **collapses to −5.86 % and stops being significant**. At 6 servers **nothing is
separated at all** — not the original penalty, not the conv, not the aggregator. That is the
predicted pattern, and no other explanation this programme has offered produces it.

**Three clauses that must travel with it.**

1. **E2 is a non-separation on SIGNIFICANCE, not on magnitude, and that is the weakest link in
   the chain.** Its median is **−8.30 %**, which is *outside* the ±5 % tie band; it reads
   `AGGREGATION-NOT-SEPARATED` because p = 0.1089. The point estimate at 6 servers actually
   favours **sum**. So the honest statement is *"sum is not shown to cost where candidates are
   few"*, **not** *"sum is harmless there"* — and a higher-powered E2 could move this. The
   verdict is as signed; the caveat is not optional.
2. **At 6 servers every learned arm loses to reactive Knative** (`gnnedgesum` +85.95 %, 0/16),
   as the record already says. Nothing at R0 is a claim about beating the baseline; it is a
   claim about a null between arms.
3. **R3 is SATURATED** (reactive queue 99 % of elapsed). `gnnedgesum` beats reactive there by
   −31.92 % and so does every other arm; that is a ranking, not a result.

**Registered expectation, scored: RIGHT, on all three bars.** E1 `SUM-COSTS`, E2
`AGGREGATION-NOT-SEPARATED`, E3 `SUM-COSTS-ONLY-WHERE-CANDIDATES-ARE-MANY`. Recorded as a
positive prediction rather than a hedge, and it is the first of this programme's last four
registered expectations to hold (`peer_only_v1` C1's magnitude and `bipartite_edge_v1`'s D3
were both wrong).

**Verified before the numbers were written down:** **0 of 64 (cell, seed) pairs are
bit-identical** between the sum and mean arms at *each* rung. `mp_bipartite_aggr` is
weight-invisible, so a flag that silently failed to apply would look exactly like E2's null.

**What this closes.** The chain `peer_only_v1` → `bipartite_edge_v1` → here is complete: the
bipartite stage cost 18.94 % live because a `GIN` **sums** over a candidate set that is 9.58×
larger than anything the corpus contained, and the fix is one word. **The practical
recommendation is `mean` aggregation for any bipartite stage over a variable-sized candidate
set** — filed in `docs/lessons.md`, because it is not specific to this model.

**What it does NOT license.** No GNN-vs-MLP claim — both arms are message-passing arms. No
claim about edge-conditioning, settled negative by `bipartite_edge_v1`. The residual MLP-shape
difference (−5.86 % at R3) is **not established** and is not worth a lineage. One corpus, one
physics, two rungs.

---

## Record (newest first)

### 2026-09-18 — CLOSED on the live gate: the predicted pattern, at both rungs

**Execution.** 16 checkpoints of `gnnedgesum`, then 192 live arms — `gnnedgesum` at R3 and R0,
plus `gnnedge0` at R0, which E2 had no comparison arm without (it had only ever been served at
80 servers and at 40 clients). All arms that ran, completed; the failures below were
submission-time refusals, never wrong measurements.

**The read** (`scripts_cosim/bipartite_aggr_v1_read.py`, checkpoint-level, n = 16 throughout):
E1 `SUM-COSTS` **+13.26 %, 4/16, p = 0.0052**; E2 `AGGREGATION-NOT-SEPARATED` −8.30 %, 10/16,
p = 0.1089; **E3 `SUM-COSTS-ONLY-WHERE-CANDIDATES-ARE-MANY`** — the registered prediction, on
all three bars. The full 2×2 is in the head; its load-bearing row is that the **MLP-shape
contrast (sum-conv vs `GIN`) is not significant at either rung**, so aggregation is carrying
the conv's advantage and the rest of the module is not.

**Three apparatus defects, all caught loudly, none of which touched a number.**

1. **The gate task table outgrew `MaxArraySize`.** It has been appended to by five lineages and
   reached 1,108 entries; this cluster's limit is **1001**, so `--array=964-1011` was rejected
   at submission. Fixed with `PO_TASK_OFFSET` rather than a second gate script — two gates that
   share cells, workloads, physics and a summary schema drift apart in one constant, and then a
   paired comparison is quietly a cross-gate one. Filed in `docs/gates/gate-tools.md`.
2. **The sidecar pin demanded `mp_bipartite_aggr` from checkpoints trained before it existed**,
   failing all 64 `gnnedge0` arms at R0. The guard was right to refuse; the rule was too strict.
   Absent is *provably* `mean` — the parameter did not exist before this lineage and the conv
   hardcoded `mean`.
3. **The first fix for (2) was wrong and cost another 48 arms.** It set the *expected value* to
   `"mean"` for a sidecar with no such key, but the check loop compares `sc.get(k)` with no
   default, so `None != "mean"` failed identically. The key must not be **pinned at all** when
   absent. The corrected logic was proven against all five real cases — including both
   refusals (`gnnedgesum` missing the key, a non-sum arm declaring sum) — **before**
   resubmitting rather than after.

**Verification before the numbers were written down:** 0 of 64 (cell, seed) pairs bit-identical
between the sum and mean arms at *each* rung, and all 16 sidecars declare
`conv=True attr_zero=True aggr='sum'`, each checked against its own state dict at training time.
`mp_bipartite_aggr` is weight-invisible, so a flag that silently failed to apply would have
looked exactly like E2's null.

### 2026-09-18 — REGISTERED, flag built and its mechanism verified before any arm

`mp_bipartite_aggr` exists as a single string flag on the unchanged `BipartiteEdgeConv`, wired
through the trainer, the sidecar, `checkpoint_mp_config`'s **string** path,
`load_prefix_conditioned_gnn` and `load_gnn_model`. Four tests in
`tests/test_bipartite_edge_arm.py` pin it: the flag is weight-invisible and byte-compatible in
both load directions but **does** change the embedding (or the arm is a no-op that would read
as a clean null); **`sum` grows faster with candidate count than `mean`**, holding weights
fixed, which is the mechanism the lineage claims; the string survives the serving whitelist as
a string and a typo (`"summ"`) fails loud rather than defaulting; and setting it without the
conv is refused as meaningless.

Default `"mean"` everywhere, so every existing `bipartite_edge_v1` checkpoint is unaffected.
