# bipartite_edge_v1 — can a bipartite graph work in this environment, if its messages can see the edge?

**Status:** `CLOSED` (2026-09-18) — **BIPARTITE-WORKS-BUT-NOT-BY-EDGE-CONDITIONING.** Closed on
a live gate at two rungs (rule 6). Registered 2026-09-18; every bar below is a module constant
in `scripts_cosim/bipartite_edge_v1_read.py`, committed before any arm was trained.

**Outcome, both halves, neither quotable alone.** **A bipartite message-passing stage CAN work in
this environment — it is now the best arm the programme has at an unsaturated rung — and
edge-conditioning, the mechanism this lineage was built to test, is NOT what makes it work.**

`gnnedge` (the bipartite `GIN` replaced by a `BipartiteEdgeConv` conditioned on the 5-column
`edge_attr`) beats the `gnn` arm by **−20.37 %** at 80 servers (16/16, p = 0.0004) and
**−28.70 %** at 40 clients (13/16, p = 0.0032) — **D1 fires at both rungs**, and the −18.94 %
bipartite penalty `peer_only_v1` measured is not merely erased but inverted. At 40 clients,
**where the rung is UNSATURATED** (reactive queue 77 % against the 90 % bar), it beats reactive
Knative by **−20.82 % on 15/16** where the `gnn` arm **loses** to reactive by +8.44 %. **D3 reads
`BIPARTITE-REPAIRED` at both rungs**: `gnnedge` is no longer behind `peeronly` (−3.06 %,
p = 0.1627 at R3; −6.93 %, p = 0.0627 and directionally ahead at 40 clients).

**And D2, the control, says the attributes are not the reason.** The decomposition, at both rungs,
one value per checkpoint over all 16:

| | 40 clients (unsaturated) | 80 servers (saturated) |
|---|---|---|
| conv alone (`gnnedge0` vs `gnn` — **both blind to `edge_attr`**) | **−24.71 %**, 15/16, p = 0.0005 | **−15.72 %**, 15/16, p = 0.0011 |
| attributes alone (`gnnedge` vs `gnnedge0` — **D2**) | **+0.44 %**, 8/16, p = 0.8361 | −7.49 %, 11/16, p = 0.0437 |
| both together (**D1**) | −28.70 %, 13/16, p = 0.0032 | −20.37 %, 16/16, p = 0.0004 |

**The conv architecture carries the whole result — 15/16 at both rungs.** Edge-conditioning adds
a marginal extra that clears its bar at exactly one rung, at **p = 0.0437 on 11/16**, and reads a
**flat null at the other** (8/16 is a coin flip). It does not replicate, so **`ATTRIBUTES-ARE-THE-LEVER`
is NOT an established finding** — the registered D2 verdict is reported per rung, as signed, and the
lineage-level reading is `CONV-IS-THE-LEVER`. The sharpest single fact: at 40 clients the **zeroed
control** beats reactive on **16/16 (−20.30 %)**, one seed *better* than the treatment's 15/16.

**Four clauses that must travel with the headline.**

1. **Without `gnnedge0` this would have been written up as "edge-conditioning repairs the
   bipartite stage, −20 to −29 %", and that would have been wrong.** `GIN → BipartiteEdgeConv`
   changes the aggregation (`sum` → `mean`) and the MLP shape as well as edge-awareness. The
   control was trained for exactly this and it earned its cost. Filed in `docs/lessons.md`.
2. **The 80-server rung is SATURATED** (reactive queue 99 % of elapsed). Its −42.20 % vs reactive
   is a ranking among arms, not a result — every arm beats reactive there. **The 40-client rung
   is the one that carries weight**, and it is the rung where D2 is a flat null.
3. **The mechanism is now differently unexplained, not explained.** `peer_only_v1` could not say
   why the bipartite GIN cost 18.94 %; this lineage can say that replacing it with a
   mean-aggregating, 2-MLP conv recovers that and more, and **cannot say why that conv is
   better.** `sum` vs `mean` aggregation over a task's candidate platforms is the obvious
   suspect and is untested here.
4. **`peer_only_v1`'s B8 clause is now in question, on a POST-HOC contrast with no registered
   bar.** At 40 clients `gnnedge` beats the best pointwise arm `mpoff_516` by **−12.96 %
   (15/16, p = 0.0061)** and `gnnedge0` by **−14.76 % (16/16, p = 0.0004)** — where B8 read
   `BEST-ARM-NOT-ESTABLISHED`. At the saturated 80-server rung `mpoff_516` still wins
   (+8.27 %, +19.50 %). **This was not registered and must not be quoted as a finding until it
   is**; it is recorded because it is the successor's reason to exist.

**Registered expectations, scored.** D1 UNCERTAIN → fired positive. D2 UNCERTAIN → split and
not established. **D3 NEGATIVE (`PENALTY-SURVIVES`) → WRONG**: it read `BIPARTITE-REPAIRED` at
both rungs. The kill condition did not fire. The lineage-level label
`BIPARTITE-WORKS-BUT-NOT-BY-EDGE-CONDITIONING` was **not** pre-registered — only the four bar
verdicts and the kill condition were — and is a composite of the signed bars, stated as such.

**What this does NOT license.** It is not a GNN-vs-MLP claim: `gnnedge0` carries the result and
is as much "message passing" as `gnnedge`, while the comparison against the pointwise arm is
post-hoc. It is not a claim that peer reasoning pays. And it is one corpus, one physics, two
rungs.

## Record (newest first)

### 2026-09-18 — CLOSED on the live gate: D1 fires, D2 does not replicate

**Execution.** 32 checkpoints (16 per arm) on the 1,670-dataset corpus, then 256 live arms —
128 at R3 (80 servers) and 128 at 40 clients — **all COMPLETED, zero lost to OOM**, so both
registered reads are at the full n = 16 and no disclosed read was needed. Training and gates
ran unattended; the D arms were run inside `peer_only_v1`'s two gate scripts so they share
cells, workloads, window and summary schema with every arm they are compared against.

**Read, `scripts_cosim/bipartite_edge_v1_gate_read.py`, checkpoint-level throughout.**

| bar | contrast | R3 (80 srv, SATURATED) | 40 clients (unsaturated) |
|---|---|---|---|
| D1 | `gnnedge` vs `gnn` | **−20.37 %**, 16/16, p = 0.0004 → `EDGE-CONDITIONING-HELPS` | **−28.70 %**, 13/16, p = 0.0032 → `EDGE-CONDITIONING-HELPS` |
| D2 | `gnnedge` vs `gnnedge0` | −7.49 %, 11/16, p = 0.0437 → `ATTRIBUTES-ARE-THE-LEVER` | **+0.44 %, 8/16, p = 0.8361 → `CONV-IS-THE-LEVER`** |
| D3 | `gnnedge` vs `peeronly` | −3.06 %, 9/16, p = 0.1627 → `BIPARTITE-REPAIRED` | −6.93 %, 11/16, p = 0.0627 → `BIPARTITE-REPAIRED` |
| D4 | `gnnedge` vs reactive | −42.20 %, 16/16, p = 0.0004 (queue 99 %, **saturated**) | **−20.82 %**, 15/16, p = 0.0011 (queue 77 %, **unsaturated**) |

**D2 is why this lineage exists and it is the finding.** The two rungs disagree, and the
disagreement is not symmetric: the positive reading is marginal on both axes (p = 0.0437,
11/16) while the null is flat (8/16 is a coin flip). Decomposed, the conv architecture alone —
`gnnedge0` vs `gnn`, both blind to `edge_attr` — carries **−24.71 % (15/16)** at 40 clients and
**−15.72 % (15/16)** at R3, i.e. essentially all of D1 at the unsaturated rung and the bulk of it
at the saturated one. **Edge-conditioning is not established as the mechanism.**

**Verification done before any of this was written down**, because the control flag is
weight-invisible and a flag that silently failed to apply would look exactly like a null:

* **0 of 64 (cell, seed) pairs at each rung are bit-identical** between `gnnedge` and
  `gnnedge0`. Median |Δ| is 1.16 s of ~19 s at 40 clients and 38.87 s of ~345 s at R3. The two
  arms are genuinely different measurements; the D2 null is real, not plumbing.
* **Every sidecar declares its flags** — 16/16 `conv=True, attr_zero=False` and 16/16
  `conv=True, attr_zero=True` — and each was checked against its own state dict at training
  time (`bip_convs weights=24`), because the sidecar is a claim and the weights are the fact.
* **The read reproduces `peer_only_v1`'s published numbers on the same cells**: `gnn` +8.44 %,
  `peeronly` −14.93 %, `mpoff_516` −7.24 % vs reactive at 40 clients. Same footing, not a
  parallel measurement.

**Post-hoc, no registered bar, recorded so the successor has a reason.** At 40 clients the two
new arms beat the best pointwise arm `mpoff_516` by −12.96 % (15/16, p = 0.0061) and −14.76 %
(16/16, p = 0.0004); at the saturated R3 rung `mpoff_516` still wins (+8.27 %, +19.50 %). That
bears on `peer_only_v1`'s B8 `BEST-ARM-NOT-ESTABLISHED` clause and **must not be quoted as a
finding until it is registered and re-run**.

**Successor, not started.** The open question is now *why the conv is better*, and the obvious
suspect is weight-invisible and cheap to test: `GIN` aggregates with `sum` over a task's
candidate platforms, `BipartiteEdgeConv` with `mean`. A sum over a variable-sized candidate set
scales with cluster size; a mean does not — which would also explain why the penalty
`peer_only_v1` measured was **absent at 6 servers** and present at 80. That is one flag, one
arm, and it is the first mechanism this programme has had that predicts an existing null.

**One defect found and fixed mid-read**, filed in `docs/gates/gate-tools.md`: the reader printed
`None/16` for every ahead-count, because `paired_tie` names it `v3_ahead`. The medians were
right and the counts were absent — and the counts are exactly what turned D2 at R3 from
"clears" into "clears at 11/16". It now refuses rather than printing `None`.

### 2026-09-18 — REGISTERED, apparatus built and proven inert

The conv, both flags, the trainer plumbing, the sidecar keys, the serving whitelist entry and
the parity tests exist and pass **before** any arm is trained. `tests/test_bipartite_edge_arm.py`
(11 tests) pins, in the order it can break: the conv is weight-visible in both load directions;
the zero control is weight-invisible so the sidecar is its only record; both keys survive
`checkpoint_mp_config`'s whitelist; the control flag reaches the model through
`load_prefix_conditioned_gnn`, which is the path the gates actually decode on (**not**
`load_gnn_model`); the control is *exactly* the treatment with zeroed attributes, bit-identical
on an already-zero graph and separated on a real one; and every combination the conv cannot
honour fails loud — extra node/dag/net edges (which carry no attribute, so padding them with
zeros would make "no attribute" indistinguishable from the control), an absent attribute, a
misaligned one, and the two nonsense flag combinations.

Default off everywhere: with both flags unset the model is byte-identical to the one that
produced every existing checkpoint.
