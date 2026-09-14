# drainable_serving_config_v1 — is the learned arms' loss at ρ ≈ 0.16 a batching artifact?

**Status:** `REGISTERED` — signed off 2026-09-14, **before any arm has been run**.

**Parent:** [`drainable_regime_v1`](drainable_regime_v1.md), whose S1 live gate closed
POINTWISE-BETTER / REACTIVE-WINS at x4000 with every control holding.

## The question

S1's decomposition shows both learned arms paying **12.37–12.40 s** of peer-group batch wait that
the reactive arms do not pay — 15–17 % of their own latency. That is a **serving configuration**,
not a model property: `GNN_BATCH_BY_PEER_GROUP=1` makes the scheduler hold a task until its peer
group co-arrives, and at 0.46 arrivals/s a 10-task group takes ~21.7 s to assemble.

So: **is the loss an artifact of how the arms were served rather than of what they learned?**

**The arithmetic says no, and that is the prediction this sweep tests.** Removing *all* of the
batch wait leaves `gnn` at 83.69 − 12.37 = **71.32 s** against `knative_network`'s **25.95 s** —
still 2.75× worse. If the sweep contradicts that, the S1 verdict is the thing that needs revising.

## Design

Everything is held fixed at the S1 gate's settings — trace `drainable_f4000_n50000.json`, cell
`cell_s7901`, T1b `lr2e3` checkpoints, 16 seeds per arm, `HEROSIM_PEER_EXCHANGE=1`, server-only
replicas, `node_disk_v2`, no `--seed`, CPU. **Uncapped only**: `GNN_PREFIX_PLATFORM_CAP=1`
deadlocks 3/16 seeds at this rung (drainable_regime_v1 Amendment 2) and is not a serving default.

The only thing that varies is how the scheduler forms a batch. Four configurations, each the
34-arm slate (`knative_network`, `knative_network_batch`, `gnn`×16, `mpoff`×16):

| id | `scheduler.batch_size` | `scheduler.batch_timeout` | `*_BATCH_BY_PEER_GROUP` | what it is |
|---|---|---|---|---|
| **A `nobatch`** | 1 | 0.02 s | 0 | place per arrival; no wait, no group, one-step decode |
| **B `pg8`** | 10 | 8 s | 1 | peer group, window cut 10× — partial groups, less wait |
| **C `tw8`** | 10 | 8 s | 0 | plain time window, peer-blind batching at the same width as B |
| **D `pg80`** | 10 | 80 s | 1 | **already measured** — the S1 gate, quoted for reference |

B vs D separates *waiting longer* from *grouping better*. C vs B separates *batching at all* from
*batching by peer group*. A removes both.

## Bars (committed before any number is read)

Sign convention is the live one: "% vs X" = 100·(X − arm)/X, positive means the arm is faster.
All tests are over the registered 16 seeds.

- **C1 — the attribution primary.** In configuration **A** (`nobatch`, zero batch wait by
  construction), `gnn` vs `knative_network`: **BATCHING-EXPLAINS** if `gnn` is within 25 % of
  Knative or faster; **BATCHING-DOES-NOT-EXPLAIN** if `gnn` remains more than 100 % slower.
  Between the two: **PARTIAL**. The S1 arithmetic predicts BATCHING-DOES-NOT-EXPLAIN.
- **C2 — does any configuration rescue the graph arm against its twin?** `gnn` vs `mpoff`,
  paired Wilcoxon, per configuration. **GNN-NEEDED** at p < 0.05 with ≥ 12/16 seeds;
  **POINTWISE-BETTER** if the reverse; **TIE** otherwise. Reported for A, B and C.
- **C3 — does any configuration beat reactive Knative?** `gnn` and `mpoff` each vs the faster
  Knative arm, per configuration. **LEARNED-WINS** / **REACTIVE-WINS** / **TIE** on the same test.
- **C4 — assembly control, per configuration, VOID for that configuration if it fails.** The arms'
  own counters must show the batching the configuration names: **A** mean batch size ≤ 1.05;
  **B** and **C** mean batch size ≥ 4; **B** additionally ≥ 60 % of its in-batch peer pairs
  retained relative to D (`prefix_pairs_in_batch`). A configuration that did not batch the way it
  claims is not read — this is the bar that caught three confounded reads in the parent lineage.
- **C5 — regime control, inherited and unchanged.** `knative_network`'s cool-down share < 5 % and
  peer share ≥ 20 % of `total_rtt`. Identical across configurations (the cap and the batch window
  are scheduler settings; the reactive per-arrival arm is the same run), so it is read once.

## What each outcome means

**BATCHING-DOES-NOT-EXPLAIN with POINTWISE-BETTER or TIE everywhere** confirms the parent's
verdict and removes the last serving-configuration objection to it: the learned arms lose at a
defensible load however they are batched.

**BATCHING-EXPLAINS**, or **GNN-NEEDED plus LEARNED-WINS in any configuration**, would be the
first measurement in this program where a graph arm beats both its pointwise twin and reactive
Knative — and it would mean `drainable_regime_v1`'s S1 verdict is scoped to one serving
configuration and must be re-stated. Both outcomes are recorded; neither is preferred.

**A note on what this cannot show.** Every configuration serves the same T1b checkpoints, which
were fitted on states from a 940× overload. A null here is a statement about *serving*, not about
whether a model trained in this regime could win. That question is closed separately by
`peer_affinity_warm_v1` W0.b on the supervised route.
