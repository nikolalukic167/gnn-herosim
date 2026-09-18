# bipartite_aggr_v1 — is the bipartite penalty a `sum` over a candidate set that grows with the cluster?

**Status:** `REGISTERED` (2026-09-18) — bars signed before any arm is trained. Successor to
`bipartite_edge_v1`, which closed **`BIPARTITE-WORKS-BUT-NOT-BY-EDGE-CONDITIONING`** and left
exactly one cheap suspect standing.

**This lineage makes a prediction, and that is its whole value.** Every other attempt on this
mechanism has been a screen: change something, see whether the number moves. This one names a
pattern that must appear and several that would falsify it, before any datum exists.

## The claim

`bipartite_edge_v1` established that replacing the bipartite `GIN` with a `BipartiteEdgeConv`
is worth **−20.4 %** at 80 servers (16/16) and **−28.7 %** at 40 clients, and that
**edge-conditioning is not the reason** — the zeroed-attribute control matches the treatment
(+0.44 %, 8/16 at 40 clients), and the conv *alone* carries −24.7 %/−15.7 % on 15/16. Two
differences between the conv and the `GIN` remain: **aggregation** (`sum` → `mean`) and MLP
shape. This tests the first, as a single flag on the otherwise unchanged module.

**Why aggregation, specifically.** A task aggregates over its **candidate platforms**, and that
set is not a fixed size — it is the platforms its client can reach, which grows with the
cluster. `cluster_scale_v1` S0.d measured it on 12 minted cells:

| rung | servers | candidates/task | vs corpus max of 5 |
|---|---|---|---|
| R0 | 6 | **3.55** | 0.71× (in support) |
| R3 | 80 | **47.92** | 9.58× (out of support) |

**A 13.5× ratio.** A `sum` over that set scales with it; a `mean` does not. And the bipartite
penalty `peer_only_v1` measured was **largest at 80 servers (−18.94 %, 16/16) and ABSENT at 6
(p = 0.61)** — a null that has sat unexplained through C3, C4, C5, C6 and D1–D4.

**So the mechanism, if it is the aggregator, predicts its own exception.** That is the test.

**E0, the precondition, is met by citation rather than re-measurement** (one fact, one home):
the candidate counts above come from `cluster_scale_v1` S0.d on rungs defined identically to
this gate's — same deterministic generator, same 6/80 server counts, same topology seeds. It is
recorded as a **cross-lineage citation**, and E1/E2's pattern would corroborate it
independently. Had those counts been equal, E1/E2 could not have separated this hypothesis from
any other story about cluster scale, and the lineage would not be worth running.

## The arm

One new arm. `gnnedgesum` is `gnnedge0` with **one flag changed**: `mp_bipartite_aggr="sum"`.
Attributes stay zeroed on both, because `bipartite_edge_v1` showed they do not matter and
carrying them would reintroduce a second difference for no gain.

| arm | bipartite stage | aggregation | sees `edge_attr` |
|---|---|---|---|
| `gnn` (exists) | `GIN` | `sum` | no |
| `gnnedge0` (exists, 16 ckpts) | `BipartiteEdgeConv` | **`mean`** | no (zeroed) |
| **`gnnedgesum`** (new) | `BipartiteEdgeConv` | **`sum`** | no (zeroed) |

`gnnedgesum` vs `gnnedge0` is therefore a **single-flag contrast** — same module, same
parameter count, same depth, same MLP shape, same corpus, split, schedule, learning rate and 16
seeds. If aggregation is the mechanism, this pair is where it lives.

**The flag is weight-invisible.** `aggr` changes no parameter, so a `sum` checkpoint and a
`mean` checkpoint are byte-compatible and load into each other in silence — the same hazard as
`mp_bipartite_edge_attr_zero`, and handled the same way: sidecar key, serving whitelist,
`prefix_serving`, both gate scripts, pinned on every arm. It is a **string**, so it is read
outside `checkpoint_mp_config`'s boolean block; `bool("sum")` and `bool("mean")` are both
`True`, and a key coerced there would serve every arm as the same one while looking correctly
whitelisted.

**The mechanism is verified at the module level before any training**
(`tests/test_bipartite_edge_arm.py`): holding weights fixed, the `sum` arm's task embeddings
grow faster with candidate count than the `mean` arm's. If that had not held, the flag would
not be testing what it claims to.

## Registered bars — signed 2026-09-18, before any arm is trained

Module constants in `scripts_cosim/bipartite_aggr_v1_read.py`. Bars reused **unchanged** from
`bipartite_edge_v1`'s D bars and `peer_only_v1`'s C bars, so all three lineages' numbers are
comparable rather than merely similar-looking. Unit: the **checkpoint**. Verdicts read through
`read_pair_pct`, orientation-neutral, negative median = the `sum` arm faster.

| bar | constant | value |
|---|---|---|
| separation | `E_SEPARATE_PCT` | 5.0 % |
| significance | `E_ALPHA` | 0.05 |
| unit count | `E_MIN_SEEDS` | 16 checkpoints |
| rungs | `E_RUNGS` | `("R3", "R0")` — 80 and **6** servers |

**Both ends of the ladder, and that is the point.** A one-rung read cannot distinguish *"sum is
worse"* from *"sum is worse **because** the candidate set is large"*, and only the second is a
mechanism.

- **E1** — `gnnedgesum` vs `gnnedge0` at **R3** (47.92 candidates/task).
  `SUM-COSTS` / `SUM-HELPS` / `AGGREGATION-NOT-SEPARATED`.
- **E2** — the same contrast at **R0** (3.55 candidates/task). Same three verdicts.
- **E3, the prediction.** All four E1 × E2 patterns named in advance:

| E1 (80 srv) | E2 (6 srv) | verdict |
|---|---|---|
| `SUM-COSTS` | `NOT-SEPARATED` | **`SUM-COSTS-ONLY-WHERE-CANDIDATES-ARE-MANY`** — predicted |
| `SUM-COSTS` | `SUM-COSTS` | `SUM-COSTS-EVERYWHERE` — a worse aggregator, but **not** by scaling; the 6-server null stays unexplained |
| `NOT-SEPARATED` | `SUM-COSTS` | `SUM-COSTS-ONLY-WHERE-CANDIDATES-ARE-FEW` — opposite of the prediction, no mechanism on offer accounts for it |
| anything else | | `AGGREGATION-IS-NOT-THE-MECHANISM` — the conv's remaining difference is its MLP shape |

### Registered expectation

**E1 `SUM-COSTS`. E2 `AGGREGATION-NOT-SEPARATED`. E3 `SUM-COSTS-ONLY-WHERE-CANDIDATES-ARE-MANY`.**

Stated as a **positive prediction**, not hedged, so it can be scored wrong — as this
programme's last three registered expectations were (`peer_only_v1` C1's magnitude, and
`bipartite_edge_v1`'s D3). The argument for it is the arithmetic: a 13.5× candidate ratio, a
`sum` that scales with it, a penalty that is largest where the ratio is largest and absent
where it is smallest. The argument against it is that **every** mechanism this programme has
proposed for this penalty has failed, including one (C3's over-smoothing) that had a measured
geometric effect behind it.

**What a falsification buys.** `SUM-COSTS-EVERYWHERE` would still be useful — it would make
`mean` a recommended default rather than an accident — but it would leave the 6-server null
unexplained and the mechanism open. `AGGREGATION-IS-NOT-THE-MECHANISM` closes the aggregation
hypothesis and leaves **MLP shape** as the last structural difference between the two convs,
which is a much weaker lever and would probably end this line of enquiry.

**What this lineage will NOT license, whatever it reads.** No GNN-vs-MLP claim: both arms are
message-passing arms and the pointwise comparison is not in scope. No claim about
edge-conditioning, which `bipartite_edge_v1` settled. And one corpus, one physics, two rungs.

---

## Record (newest first)

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
