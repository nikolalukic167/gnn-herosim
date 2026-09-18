# bipartite_edge_v1 — can a bipartite graph work in this environment, if its messages can see the edge?

**Status:** `REGISTERED` (2026-09-18) — bars signed before any arm is trained. Successor to
`peer_only_v1`'s **C7**, opened as a new lineage because it needs a new conv module, a
weight-visible flag, a sidecar key, a serving-whitelist entry and train/serve parity coverage
— none of which is an amendment to a closed node.

**The question, and why it is still open.** `peer_only_v1` measured a real, rung-specific live
cost to the bipartite stage: `peeronly` (PeerConv, bipartite GIN removed) beats the full `gnn`
arm by **−18.94 %** at 80 servers (16/16, p = 0.0004) and **−18.66 %** at 40 clients (13/16,
p = 0.0262), and the penalty is **absent at 6 servers** (p = 0.61) and at 80 clients (p = 0.28).
It closed with that penalty having **no established mechanism**: C3 found geometric compression
but measured it in the regime where the penalty is absent, C5 was not measurable on the corpus,
C6 was not runnable, and **C4 tested the residual repair end to end and it did not transfer**.

It also closed with the contrast **CONFOUNDED**, by code read rather than by measurement:

| | residual | sees `edge_attr` |
|---|---|---|
| `PeerConv` (task↔task) | **yes** — `return x + self.update_mlp(...)` | **yes** — `message()` concatenates it |
| bipartite `GIN` (task↔platform) | **no** — `x = h`, the output *replaces* the encoding | **no** — `GIN` has no `edge_dim` |

Two differences, one contrast. **C4 closed the residual column** (`RESIDUAL-DOES-NOT-TRANSFER`:
−9.60 % vs `gnn` at R3, p = 0.4326, does not clear, and still 13.02 % behind `peeronly`; at 80
clients it loses to reactive outright). This lineage is the other column, and it is the **only
remaining lever with evidence behind it**.

**What the bipartite stage is blind to, exactly.** `data.edge_attr` is five columns per
(task, platform) pair — `[exec_time, latency, is_warm, energy, comm_time]`
(`src/notebooks/prepare_graphs_cache.py`, the `# Edge attributes:` comment). That is the entire
physics of the pairing: how long this task runs on this platform, how far away it is, whether
it is warm. `torch_geometric.nn.models.GIN` takes `(x, edge_index)` and is constructed with no
`edge_dim`, so today those five columns reach the `EdgeScorer` and **nothing else**. The
bipartite message-passing stage averages platform embeddings without knowing which pairing is
fast, near, or warm — while `PeerConv`, the stage that does not cost anything, is conditioned
on its edge attribute throughout. **That asymmetry is the hypothesis.**

**Nothing in `peer_only_v1` licenses "a bipartite graph does not work in this environment."**
What it licenses is "a bipartite GIN that is neither residual nor edge-aware costs 18.94 % live
at 80 servers." This lineage is what would be needed to say more than that, in either direction.

---

## The arms

Two new arms, both trained on the **same 1,670-dataset corpus, split artifact, objective,
schedule, learning rate and 16 seeds** as `peer_only_v1`'s `gnn` — so `gnn`, `gnnres`,
`gnnedge` and `gnnedge0` are four points differing in the bipartite stage and nothing else.

| arm | bipartite stage | sees `edge_attr` | residual |
|---|---|---|---|
| `gnn` (exists, 16 ckpts) | `GIN` | no | no |
| `gnnres` (exists, 16 ckpts) | `GIN` | no | yes |
| **`gnnedge`** (new) | `BipartiteEdgeConv` ×3 | **yes** | no |
| **`gnnedge0`** (new) | `BipartiteEdgeConv` ×3 | **no — zeroed** | no |

`gnnedge0` is the **architecture control**, and it is the reason this is a measurement rather
than a demonstration. `GIN → BipartiteEdgeConv` changes the aggregation (`sum` → `mean`) and
the MLP shape as well as edge-awareness, so `gnnedge` vs `gnn` alone cannot separate "the
attributes help" from "this conv is better." `gnnedge0` is the *same module, same parameter
count, same depth*, with `edge_attr` replaced by zeros — so `gnnedge` vs `gnnedge0` isolates
edge-conditioning exactly. `tests/test_bipartite_edge_arm.py` pins that the two are
bit-identical on a graph whose attributes are already zero.

**The control is weight-invisible.** Zeroing a tensor leaves no parameter behind, so the two
arms' checkpoints are byte-compatible and load into each other without complaint. The sidecar
key `mp_bipartite_edge_attr_zero` is the *only* thing that distinguishes them, and it is on the
serving whitelist in `src/executesimulation.py` and wired into `prefix_serving.py`. This is the
exact defect class `docs/lessons.md` records under sidecar keys needing a serving whitelist;
omitting it would have served the control as the treatment and made the two arms one arm.
`mp_bipartite_edge_conv` *is* weight-visible (`bip_convs.*`), so a strict load across that
boundary fails on its own.

**Deliberately not residual.** `BipartiteEdgeConv.forward` returns
`update_mlp(cat[x, out])`, not `x + update_mlp(...)`, which keeps `mp_residual` an orthogonal
flag. {GIN, EdgeConv} × {replace, residual} stays a clean 2×2 instead of two entangled arms. If
D1 fires positive, the residual cell of that square is the follow-up — not part of this
registration.

---

## Registered bars — signed 2026-09-18, before any arm is trained

Every constant below is a module constant in `scripts_cosim/bipartite_edge_v1_read.py`,
committed before the first datum exists. The statistical unit is the **checkpoint**
(`collapse_to_seed`, median over that checkpoint's cells), never the (cell, seed) pair —
`peer_only_v1` B3 halved a headline by getting that wrong. Verdicts are read with
`read_pair_pct`, which is orientation-neutral, because a named reader reused with its arguments
swapped inverts its verdict string while leaving its number correct
(`docs/gates/gate-tools.md`, 2026-09-18).

**Rungs.** 80 servers (R3) and 40 clients — **the two rungs where the penalty was actually
measured**, and deliberately not 6 servers or 80 clients, where it is absent and there is
nothing to repair. Both are gated live. A negative at a rung where the effect does not exist
would be uninformative, and this programme has already spent C3 on exactly that error.

| bar | constant | value |
|---|---|---|
| separation | `D_SEPARATE_PCT` | 5.0 % |
| significance | `D_ALPHA` | 0.05 |
| unit count | `D_MIN_SEEDS` | 16 checkpoints |
| rungs | `D_RUNGS` / `D_CLIENTS` | `("R3",)` / `(40,)` |

### D1 — does edge-conditioning close the bipartite penalty? (`gnnedge` vs `gnn`)

- **`EDGE-CONDITIONING-HELPS`** — `gnnedge` faster by ≥ 5 % with p < 0.05 at **either** rung.
- **`EDGE-CONDITIONING-COSTS`** — slower by ≥ 5 % with p < 0.05 at either rung.
- **`NOT-SEPARATED`** — otherwise.

### D2 — is it the attributes, or the conv? (`gnnedge` vs `gnnedge0`)

**This is the load-bearing bar.** D1 without D2 is confounded with the conv swap.

- **`ATTRIBUTES-ARE-THE-LEVER`** — `gnnedge` faster by ≥ 5 %, p < 0.05.
- **`CONV-IS-THE-LEVER`** — not separated from its own zeroed control, while D1 reads
  `EDGE-CONDITIONING-HELPS`. Signed consequence: the gain is **not** edge-conditioning, it is
  `mean`-aggregation or MLP shape, and it must be reported that way.
- **`ATTRIBUTES-COST`** — `gnnedge` slower than its zeroed control by ≥ 5 %, p < 0.05.

### D3 — does either new arm reach `peeronly`? (`gnnedge` vs `peeronly`)

The standing question is not "is `gnnedge` better than `gnn`" but "can a bipartite stage stop
costing anything." Signed in advance so a within-family improvement cannot be quoted as one.

- **`BIPARTITE-REPAIRED`** — `gnnedge` not separated from `peeronly` (|median| < 5 % or
  p ≥ 0.05) **and** D1 reads `EDGE-CONDITIONING-HELPS`.
- **`PENALTY-SURVIVES`** — `peeronly` still faster by ≥ 5 %, p < 0.05.

### D4 — the live floor (`gnnedge` vs reactive Knative)

Reported at both rungs with the **saturation** classification beside it (a rung is saturated if
reactive's queue is ≥ 90 % of elapsed). At 80 servers every arm beats reactive, so a win there
is a ranking among arms, not a result; at 40 clients the rung is unsaturated and `gnn` **loses**
to reactive by +8.44 %, which is the number D4 is really asking about.

### Registered expectation, recorded so it can be scored

**D1: UNCERTAIN. D2: UNCERTAIN. D3: NEGATIVE — `PENALTY-SURVIVES`.**

The mechanistic argument for D1 is the best this programme has had for a bipartite repair: the
blindness is specific, named, and in the one stage that costs latency, while the stage that
costs nothing has the attribute all along. Against it, everything this lineage's parent
measured — C4's residual repair was also the textbook fix for a measured defect and did not
transfer; the offline/live sign has reversed more than once here; and D3 asks the new arm to
close an 18.94 % gap, which no architecture change in the programme has ever done.

Recorded as **NEGATIVE on D3** rather than hedged, so it can be scored wrong. `peer_only_v1`'s
C1 magnitude prediction was recorded and was wrong, and that is the point of recording it.

**Kill condition, signed in advance:** if D1 and D2 both read `NOT-SEPARATED`, this lineage
CLOSES as `EDGE-CONDITIONING-DOES-NOT-TRANSFER` and the standing answer gains the sentence that
**both** halves of the `peer_only_v1` confound have now been tested and neither explains the
penalty. There is no third half. That would make "the bipartite penalty has no mechanism" a
measured statement rather than an open question, and the next move would have to come from
outside this family.

---

## Record (newest first)

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
