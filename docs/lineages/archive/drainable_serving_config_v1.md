# drainable_serving_config_v1 — is the learned arms' loss at ρ ≈ 0.16 a batching artifact?

**Status:** `CLOSED` 2026-09-14 — **BATCHING-DOES-NOT-EXPLAIN; no configuration rescues the
graph arm.** With zero batch wait the graph arm is **−1731.86 %** against reactive Knative, 8×
worse than the 80 s window it was suspected of being handicapped by: peer-group batching is most
of what keeps the learned arms within an order of magnitude of a reactive baseline, not a tax they
pay. **C3 is REACTIVE-WINS in all four readable configurations, 0/16 seeds each.** One finding
revises the parent: `gnn` vs `mpoff` runs −499.81 % → TIE → TIE → −16.32 % as the window widens,
so `drainable_regime_v1`'s POINTWISE-BETTER is specific to its 80 s window and is a **TIE** at the
windows that serve both arms best. Registered with every bar committed before any arm ran.

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

### 2026-09-14 — Amendment 1: B fails C4; add E `pg16` and F `pg24` (bars unchanged)

First-arm assembly check, `gnn_s1` of each configuration, read against C4 before letting the
sweep run to conclusion:

| config | mean batch | incomplete | `prefix_pairs_in_batch` | `prefix_peers_outside_batch` | batch wait |
|---|---|---|---|---|---|
| A `nobatch` | 1.00 | 0.0 % | 0 | 177,788 | 0.00 s |
| **B `pg8`** | **3.91** | 63.4 % | **43,398** | 90,992 | 4.56 s |
| C `tw8` | 4.59 | 0.0 % | 34,173 | 109,442 | 7.98 s |
| D `pg80` (S1) | 8.41 | 3.0 % | 83,788 | 10,212 | 12.37 s |

**A passes C4** (mean batch ≤ 1.05, and zero in-batch peer pairs, which is what "no grouping"
means). **C passes C4** (mean batch ≥ 4).

**B fails C4 on both of its thresholds** — mean batch **3.91** against the bar's 4, and
**43,398** in-batch peer pairs, **51.8 %** of D's 83,788 against the bar's 60 %. **B is VOID and
is not read.** It failed narrowly and its seed-1 `total_rtt` is the most favourable of the three,
which is precisely why the bar is not being moved: C4's thresholds were committed before any arm
ran and they stay where they are. A 8 s window at 0.46 arrivals/s recovers about half a peer
group, and "about half a group" is not the configuration the sweep set out to test.

**Added, with C1–C5 unchanged and no threshold touched:**

| id | `scheduler.batch_size` | `scheduler.batch_timeout` | `*_BATCH_BY_PEER_GROUP` | poll |
|---|---|---|---|---|
| **E `pg16`** | 10 | 16 s | 1 | 0.8 s |
| **F `pg24`** | 10 | 24 s | 1 | 1.2 s |

A 10-task peer group takes ~21.7 s to co-arrive at this rate, so E sits just below that span and
F just above it. Between them they bracket the region where a peer group first assembles without
paying D's 80 s window, and at least one should clear C4. If neither does, the sweep reports that
the peer-group serving mode has no configuration at this rate that both assembles groups and
avoids a long wait — which is itself the answer to the question the sweep asks.

**Disclosure.** Seed-1 `total_rtt` for A, B and C was visible in this assembly check before E and
F were registered. C1, C2, C3 and C5 are untouched by that: they were committed in the original
registration and no threshold in them has moved. C4's thresholds are likewise unchanged. E and F
are added because B fell below C4, not because of any arm's latency.

### 2026-09-14 — READ: **BATCHING-DOES-NOT-EXPLAIN; no configuration rescues the graph arm**

Jobs 765655 (A), 765656 (B), 765725 (E), 765726 (F); D is the parent's S1 gate. Uncapped,
16 seeds per arm, `cell_s7901` + `drainable_f4000_n50000.json` throughout. C5 holds (peer 21.19 %,
cool-down 0.07 %). Attachment `drainable_serving_config_v1/read.json`.

| config | window | mean batch | pair retention | C4 | C2 `gnn` vs `mpoff` | C3 `gnn` vs reactive | C3 `mpoff` vs reactive |
|---|---|---|---|---|---|---|---|
| **A `nobatch`** | — | 1.00 | 0 % | pass | **−499.81 %**, p = 9.2e-05, 1/16 → POINTWISE-BETTER | **−1731.86 %**, 0/16 | −205.41 %, 0/16 |
| B `pg8` | 8 s | 3.91 | 52.3 % | **FAIL** | VOID | VOID | VOID |
| **E `pg16`** | 16 s | 5.97 | 84.7 % | pass | **−4.15 %**, p = 0.86, 10/16 → **TIE** | **−106.02 %**, 0/16 | −97.81 %, 0/16 |
| **F `pg24`** | 24 s | 7.35 | 96.1 % | pass | −3.13 %, p = 0.60, 6/16 → **TIE** | −141.80 %, 0/16 | −134.45 %, 0/16 |
| **D `pg80`** | 80 s | 8.16 | 99.9 % | pass | −16.32 %, p = 0.0010, 3/16 → POINTWISE-BETTER | −222.56 %, 0/16 | −177.30 %, 0/16 |

**C1: BATCHING-DOES-NOT-EXPLAIN**, and not marginally. With zero batch wait the graph arm is
**−1731.86 %** against reactive Knative — **8× worse than the 80 s window it was accused of being
handicapped by.** Peer-group batching is not a tax the learned arms pay; it is most of what keeps
them within an order of magnitude of a reactive baseline.

**C3: REACTIVE-WINS in every readable configuration, 0/16 seeds, every time.** The best of them,
E, still leaves both learned arms roughly 2× slower than `knative_network`.

#### The decomposition, medians over 16 seeds

| config | arm | latency, s | queue, s | batch wait, s | `totalPeerRendezvousWait`, s | `scaleEventCount` |
|---|---|---|---|---|---|---|
| (any) | `knative_network` | 25.95 | 17.76 | 0.00 | 1.316e5 | 1,036 |
| A | `mpoff` | 79.24 | 70.74 | 0.00 | 1.213e5 | 1,767 |
| A | **`gnn`** | **475.29** | **467.01** | 0.00 | 1.200e5 | **39,910** |
| E | `mpoff` | 51.32 | 37.41 | 7.22 | 6.306e4 | 6,382 |
| E | **`gnn`** | **53.45** | 39.59 | 7.22 | 6.369e4 | 6,091 |
| F | `mpoff` | 60.83 | 45.72 | 8.67 | 4.350e4 | 7,072 |
| F | `gnn` | 62.74 | 47.71 | 8.66 | 4.427e4 | 6,994 |
| D | `mpoff` | 71.95 | 53.40 | 12.40 | 2.684e4 | 9,190 |
| D | `gnn` | 83.69 | 65.19 | 12.37 | 2.646e4 | 8,878 |

**The window trades rendezvous against queue, and the trade has an interior optimum.** Widening
it from 16 s to 80 s cuts peer rendezvous wait 2.4× (6.4e4 → 2.6e4) and costs 5.2 s more batch
wait and 26 s more queue. E wins on net; D is past the optimum. The environment's own mechanism is
real and purchasable — it is just never worth what the queue charges for it.

**The graph arm cannot decode singletons.** At A it issues **39,910** scale events against the
pointwise twin's 1,767 — a 22× gap — and carries 467 s of queue against 70.7 s. Given a peer group
it is indistinguishable from its twin; given one task at a time it thrashes the replica lifecycle.
This is what the parent lineage's confounded read was actually measuring when it reported
"autoscaler churn 1,036 → 1,774 → 20,400": a singleton-decoding graph arm.

#### The finding that revises the parent

**`gnn` vs `mpoff` is a function of the batch window, and POINTWISE-BETTER is not the general
answer.** Across the four readable configurations the contrast runs **−499.81 % → TIE (−4.15 %) →
TIE (−3.13 %) → −16.32 %** as the window widens from none to 80 s. At the two windows that best
serve *both* arms it is a **TIE**; the parent's S1 headline of POINTWISE-BETTER at −15.94 % is
specific to an 80 s window, 3.7× the ~21.7 s a peer group needs, and is recorded there as such.

What does not move under any configuration is C3. **No batching configuration produces a graph arm
that beats reactive Knative**, and none produces one that beats its own pointwise twin.

**Status: `CLOSED` 2026-09-14 — BATCHING-DOES-NOT-EXPLAIN; no configuration rescues the graph arm.**
