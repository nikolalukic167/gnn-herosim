# best_arm_v1 — is the best scheduler this programme has still a pointwise one?

**Status:** `REGISTERED` (2026-09-18) — bars signed before any arm is served. This lineage
exists to settle **half of CLAUDE.md's standing answer**, which has read *"the best scheduler
the program has is still a pointwise one"* for months.

## Why now

That clause rests on `peer_only_v1`'s **B4**: `peeronly_1670` loses to `mpoff_516` by
**+11.17 %, 0/16, p = 0.0004** at 80 servers / 20 clients. **B8** then scoped it to *saturated*
rungs — at the unsaturated client rungs the two are **not separated** (−9.20 %, p = 0.5349;
−4.14 %, p = 0.1089) ⇒ `BEST-ARM-NOT-ESTABLISHED`, neither shown better.

Since then the graph arm has changed. `bipartite_edge_v1` and `bipartite_aggr_v1` between them
found that the bipartite stage cost 18.94 % because `GIN` **sums** over a candidate set that is
9.58× wider than anything the corpus contained, and that a mean-aggregating conv repairs it.
The repaired arm beats reactive Knative by **−20.82 % (15/16) at the unsaturated 40-client
rung**, where the old `gnn` arm *loses* by +8.44 %.

And — **post-hoc, with no registered bar** — at that rung it also beats `mpoff_516` by
**−12.96 % (15/16, p = 0.0061)**, with the zeroed control at **−14.76 % (16/16, p = 0.0004)**.
**That number is not evidence.** It was computed after the fact, on a contrast nothing had
signed, in a lineage whose registered question was about aggregation. This lineage is what
turns it into evidence or refutes it.

## The arms

| arm | what it is | why it is here |
|---|---|---|
| `gnnedge0` | the repaired bipartite arm — `BipartiteEdgeConv`, **mean** aggregation, attributes zeroed | it is what actually carried `bipartite_edge_v1`'s result; the attributes contribute nothing (D2) |
| `mpoff_516` | the best pointwise arm | the arm B4/B8 crowned; `partial_state_v3`'s `mpoff`, re-served, and A0 proves that re-serve bit-identical |
| reactive | Knative | the floor, and the saturation classification |

**`gnnedge0`, not `gnnedge`.** Reporting the arm with the extra machinery that was measured to
do nothing would overstate what is needed to get the result.

## Rungs — the whole client ladder, and only its unsaturated part

20 / 40 / 80 clients at 6 servers, reactive queue share **63 / 73 / 73 %** against the
registered 90 % saturation bar (`peer_only_v1` B7). **The 20-client rung IS the R0 server cell**
(6 servers, 20 clients) and is read from the server gate's results — the same cells, not a
parallel measurement. Two arm-sets are missing and are what this lineage runs: `mpoff_516` at
R0 (never served there) and `gnnedge0` at 80 clients (served only at 40 and at the two server
rungs). 40 clients is already complete on both arms.

**Saturated rungs are deliberately out of scope.** B4's clause was *scoped* to them by B8 and
this lineage does not revisit it; at 80 servers `mpoff_516` still beats both new arms (+8.27 %,
+19.50 %, post-hoc). The question here is what happens at defensible load.

## Registered bars — signed 2026-09-18, before either missing arm-set is served

Module constants in `scripts_cosim/best_arm_v1_read.py`. Bars reused **unchanged** from B4/B8
(`5.0 %`, `0.05`, `16` checkpoints) so this reads against the clause it is testing rather than
against a bar of its own choosing. Unit: the **checkpoint**. Orientation-neutral verdicts via
`read_pair_pct`; negative median = the graph arm faster.

- **F1 — `gnnedge0` vs `mpoff_516`, per rung.** `GRAPH-ARM-FASTER` / `POINTWISE-FASTER` /
  `NOT-SEPARATED` at each of 20, 40, 80 clients.
- **F2 — the composite, and the consequence for the standing answer.**
  - **`GRAPH-ARM-IS-BEST`** — `GRAPH-ARM-FASTER` at **≥ 2 of 3** unsaturated rungs and
    `POINTWISE-FASTER` at none. **Signed consequence: CLAUDE.md's "the best scheduler is still a
    pointwise one" is OVERTURNED at unsaturated load, not merely scoped**, and the standing
    answer is rewritten to say so.
  - **`POINTWISE-STILL-BEST`** — `POINTWISE-FASTER` at ≥ 2 of 3. The clause survives B8's
    scoping and is restored to the unsaturated rungs too.
  - **`BEST-ARM-STILL-NOT-ESTABLISHED`** — anything else. B8's reading stands unchanged, and
    the post-hoc −12.96 % is recorded as **not reproduced under a registered bar**.
- **F3 — the shape of the margin across the ladder** (20 → 40 → 80 clients), descriptive, with
  each rung's reactive queue share printed beside it. `MARGIN-GROWS-WITH-LOAD` if the graph
  arm's advantage is monotone increasing, else `MARGIN-NOT-MONOTONE`. **No consequence is
  attached to F3** — three rungs cannot establish a trend, and it is recorded so a successor
  knows where to look rather than to support a claim.
- **F4 — both arms vs reactive at each rung**, with saturation printed. Context, not a bar.

### Registered expectation

**F1: `GRAPH-ARM-FASTER` at 40 clients. UNCERTAIN at 80. `POINTWISE-FASTER` at 20.**
**F2: `BEST-ARM-STILL-NOT-ESTABLISHED`.**

The 40-client prediction is close to a formality — that rung is already measured and the
post-hoc number is −12.96 % on 15/16; **it is registered anyway so the other two are read on the
same footing**, and if it fails, something is wrong with this lineage rather than with the
arm. The 20-client prediction is the substantive one: at 6 servers with 20 clients **every
learned arm loses to reactive Knative**, `peeronly` lost to `mpoff_516` wherever the two have
been compared at power, and a task has only ~3.55 candidates — the regime where the repair this
arm carries does nothing (`bipartite_aggr_v1` E2). If the graph arm is going to lose anywhere
unsaturated, it is there.

`BEST-ARM-STILL-NOT-ESTABLISHED` follows from expecting a split. **It is a prediction that this
lineage does NOT overturn the standing answer**, recorded so it can be scored wrong — as
`bipartite_edge_v1`'s D3 was.

### What this will not license, whatever it reads

**It is not a GNN-vs-MLP claim.** `mpoff_516` is the MP-OFF twin of a GNN, not the MLP, and it
is trained on a **different corpus (516 vs 1,670)** — B1 measured the corpus as the largest
lever in this programme (`CORPUS-DOES-NOT-HELP`, 6/6, more data made every arm slower live).
Any F1 result is therefore **confounded with corpus size** and must be quoted with both arms'
training caches, exactly as B4 and B8 are. A corpus-matched successor is the thing that would
make this a model-class claim, and it is not this lineage.

---

## Record (newest first)

### 2026-09-18 — REGISTERED

Bars committed before either missing arm-set was served. Two arm-sets to run: `mpoff_516` at
R0 (server gate tasks 1108–1171) and `gnnedge0` at 80 clients (client gate tasks 976–1039);
40 clients is already complete on both arms and is **not** re-run. Both blocks need
`PO_TASK_OFFSET` — the gate tables are past `MaxArraySize = 1001`.
