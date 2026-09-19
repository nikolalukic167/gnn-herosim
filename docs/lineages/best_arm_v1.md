# best_arm_v1 — is the best scheduler this programme has still a pointwise one?

**Status:** `CLOSED` (2026-09-18) — **`BEST-ARM-STILL-NOT-ESTABLISHED`.** Closed on a live gate
across the unsaturated client ladder (rule 6). Registered 2026-09-18; every bar is a module
constant in `scripts_cosim/best_arm_v1_read.py`, committed before either missing arm-set was
served.

**Outcome. The clause survives — and the ladder is SPLIT, which is not the same as the graph
arm failing.** The repaired bipartite arm (`gnnedge0`) beats the best pointwise arm
(`mpoff_516`) at two of the three unsaturated rungs and loses badly at the third. The signed
rule requires **two wins AND no losses** to overturn CLAUDE.md's *"the best scheduler is still a
pointwise one"*, so the clause stands.

| rung | F1 | median | p | ahead | reactive queue |
|---|---|---|---|---|---|
| **20 clients** (= the R0 cells) | **`POINTWISE-FASTER`** | **+15.71 %** | 0.0019 | 3/16 | 66 %, unsaturated |
| **40 clients** | **`GRAPH-ARM-FASTER`** | **−14.76 %** | 0.0004 | **16/16** | 77 %, unsaturated |
| **80 clients** | **`GRAPH-ARM-FASTER`** | −11.32 % | 0.0386 | 12/16 | 80 %, unsaturated |

**F2 `BEST-ARM-STILL-NOT-ESTABLISHED`. F3 `MARGIN-NOT-MONOTONE`** (descriptive only).

**The post-hoc number DID reproduce.** `bipartite_aggr_v1` recorded −12.96 %/−14.76 % at 40
clients as post-hoc and explicitly not evidence. Under a registered bar it reads **−14.76 % on
16/16, p = 0.0004** — the strongest single contrast in this comparison. What is not established
is a claim about the **ladder**, not that measurement.

**Four clauses, none optional.**

1. **At 20 clients BOTH arms lose to reactive Knative catastrophically** — `gnnedge0` +94.99 %
   (0/32) and `mpoff_516` +69.34 % (0/16). The pointwise "win" there is a **ranking among
   losers**, on a rung where nothing the programme has built is deployable. At the two rungs
   where learned arms actually beat reactive, the graph arm wins. **That is suggestive and it is
   NOT used to reinterpret F2** — the bar was signed before the data and the verdict is read as
   signed.
2. **It is not a GNN-vs-MLP claim, and it is confounded with corpus.** `mpoff_516` trains on
   516 datasets and `gnnedge0` on 1,670, and B1 measured corpus as the largest lever in this
   programme (`CORPUS-DOES-NOT-HELP` 6/6 — more data made every arm *slower* live). Quote both
   arms' training caches or do not quote the comparison. A corpus-matched successor is what
   would make this a model-class result.
3. **`mpoff_516` is the MP-OFF twin of a GNN, not the MLP.** The pointwise baseline this clause
   names has never been the tabular MLP.
4. **20 clients is the R0 server cell** (6 servers, 20 clients), read from the server gate's
   own results — the same cells, not a parallel measurement. The reader asserts it.

**Registered expectation, scored: RIGHT on the composite and on two of three rungs.** F2
`BEST-ARM-STILL-NOT-ESTABLISHED` was predicted, as was `POINTWISE-FASTER` at 20 clients and
`GRAPH-ARM-FASTER` at 40; 80 clients was registered UNCERTAIN and came in for the graph arm.

**What this changes in the standing answer: nothing, deliberately.** It was registered as a
prediction that this programme's own three bipartite lineages do **not** overturn the pointwise
clause, and that is what it found. The clause stays, now tested rather than merely unchallenged.

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
    > **Correction, 2026-09-18, after the read.** That last clause is wrong as drafted and is
    > left here verbatim because a registration is annotated, never rewritten. This verdict
    > covers a *split* ladder as well as a barren one, and on the actual result the post-hoc
    > margin **did** reproduce (−14.76 %, 16/16, p = 0.0004 at 40 clients). What
    > `BEST-ARM-STILL-NOT-ESTABLISHED` denies is a claim about the **ladder**, not the per-rung
    > measurement. The reader was fixed to say so; the verdict itself is unaffected.
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

### 2026-09-18 — CLOSED on the live gate: the ladder is split

**Execution.** Two missing arm-sets, 128 live arms, all COMPLETED: `mpoff_516` at R0 (it had
never been served there) and `gnnedge0` at 80 clients (served only at 40 and the two server
rungs). 40 clients was already complete on both arms and was **not** re-run.

**Two defects caught before they reached the record.**

1. **A collision, and it was the loader being right.** Serving `mpoff_516` at R0 for all 16
   seeds duplicated `partial_state_v3` P3's seeds 1, 2, 4, 5 on the same cells — B4 avoided the
   same overlap at R3 by running only the other 12. `peer_only_v1_gate_read.tables()` refused
   rather than letting one copy silently win. The 16 overlapping runs were verified
   **bit-identical, max |delta| = 0.000000** (an unplanned re-serve confirmation at R0), the
   duplicates were **archived, not deleted**, and re-reading afterwards reproduced all three
   rungs exactly.
2. **F2's fall-through printed a false sentence.** It said *"the post-hoc −12.96 % is NOT
   reproduced under a registered bar"* for every non-overturning outcome — but on this result
   the graph arm wins at 40 clients by −14.76 % on 16/16. The verdict was right and the
   explanation was wrong, and it is exactly the sort of sentence that gets copied into a node
   verbatim. The branch now distinguishes a **split** ladder (per-rung wins are real) from one
   with no graph win at any rung, which is the only case that may say the margin did not
   reproduce.

### 2026-09-18 — REGISTERED

Bars committed before either missing arm-set was served. Both blocks needed `PO_TASK_OFFSET` —
the gate tables are past `MaxArraySize = 1001`.
