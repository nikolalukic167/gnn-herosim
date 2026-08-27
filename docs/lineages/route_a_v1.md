# route_a_v1 — FALSIFIED

> **Status:** `FALSIFIED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-25 → 2026-08-25

**Outcome.** **NO-GO.** DAG + distance is genuinely pairwise and still pointwise-optimal. Breaking separability is **necessary but not sufficient** — you also need contention, which is what route B tests.

**Related:** [route_b_v1](route_b_v1.md) · [graph_structure_physics](graph_structure_physics.md)

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [route_a_v1 — **NO-GO, now properly tested. Breaking separability is NECESSARY BUT NOT SUFFICIENT** (2026-08-25)](#route-a-v1-no-go-now-properly-tested-breaking-separability-is-necessary-but-not-sufficient-2026-08-25)
- [route_a_v1 — scaling probe: **NO-GO**, and the reason is a defect in the term, not a verdict on route A (2026-08-25)](#route-a-v1-scaling-probe-no-go-and-the-reason-is-a-defect-in-the-term-not-a-verdict-on-route-a-2026-08-25)
- [route_a_v1 — the five blockers, cleared (2026-08-25)](#route-a-v1-the-five-blockers-cleared-2026-08-25)

---

### route_a_v1 — the five blockers, cleared (2026-08-25)

**The hypothesis.** `program_verdict_v1` closed the supervised route by theorem: with
per-task costs separable and placements freely chosen, the componentwise minimiser is
optimal under **any** monotone aggregation, so no objective, target or scoring rule can
create structure — and co-location coupling cannot supply it either, because co-residents
are exchangeable within type and every symmetric function of a multiset is a function of
counts. Route A is the one structure that defeats both barriers at once: **a child's input
read priced by the network distance from its parent's node**, which is a pairwise term over
two *jointly decided* placements, between tasks playing *different structural roles*.

The trap that governs the whole design: if the parent is already placed when the child is
decided, "distance from the parent's node" is just another edge feature and a pointwise
model recovers optimality. Non-separability requires parent and child in the **same
jointly-decided set** — which co-sim already gives, since a `placement_plan` fixes every
task's platform before the episode runs.

**This entry is the groundwork, not the experiment.** No corpus has been generated and no
gate has been run. What it records is that the simulator can now express the hypothesis at
all — it could not before, in five separate ways, none of which could ever fire while every
application in every corpus is a single-node dag:

| # | Blocker | State |
|---|---|---|
| 1 | `workflow_process` dispatched a **linearization**, so `A → {B,C,D}` ran as a depth-3 chain — siblings never overlapped and were never co-decidable | fixed, 5 tests |
| 2 | `DeterminedScheduler._collect_task_batch` blocked without a timeout ⇒ **deadlock** on any DAG (`batch_timeout` was configured and never read) | fixed |
| 3 | Placement enumerator walked `dag.keys()` while the simulator assigns ids by `static_order()` ⇒ silent `forced_placements` mis-assignment | fixed |
| 4 | **No server↔server distance exists anywhere** — `network_map` and backbone routes are client↔server only, so a parent and child on two servers have no distance and no path | `build_server_mesh`, opt-in |
| 5 | `prepare_graphs_cache` hardcoded dnn1/dnn2 candidates ⇒ any other task type gets **zero** candidates, label `-1`, contract-5.5 failure for the whole cache | fixed, filter only |

**The new physics.** `Platform._dependency_transfer_time` charges each *remote* parent's
`stateSize[...]["output"]` over this node's bandwidth plus the parent→child network latency.
It reads **all** parents, closing the `dependencies[-1]` FIXME that silently drops every
parent but one on a fan-in. Gated on `HEROSIM_DATA_LOCALITY=1` and inert without
dependencies. Missing reachability **raises** rather than charging 0.0 — a free bad
placement is the signal inverted.

`HEROSIM_STATE_SIZE_BYTES` scales the input `stateSize` in memory. It is not a convenience:
`data/nofs-ids/task-types.json` is shared by *every* corpus and is never copied per dataset,
so editing the welded 153,600 B would rewrite the physics of every existing collection. It
is also the lever — the coupled term scales with `stateSize` while queue work does not, so
unlike link bandwidth (where both scale as `1/bandwidth` and the ratio is invariant) **the
ratio moves**. At the welded value the dependency read is ~1.2% of the queue term.

**Zero-diff is the load-bearing claim here**, since all of this touches shared physics: a
30k-event GNN episode on `cell01` reproduces `total_rtt = 1375056.421447831` bit-identically
before and after, *including with `HEROSIM_DATA_LOCALITY=1` set* — the term cannot fire
without dependencies, and no corpus has any. 385 tests pass. New: `tests/test_dag_dispatch.py`
(5), `tests/test_data_locality_cost.py` (14); the DAG-specific dispatch tests are verified to
fail against the old implementation.

**Not done, and required before any claim** — the pilot itself: DAG workload templates +
grid preset (§9 route A), the `stateSize` scaling probe with its **go/no-go** (if no
plausible `stateSize` makes additive-argmin regret non-zero on spread plans, stop — that is
the cheap intended failure point), pre-registration, the n≥200 corpus, the k-integer and
parent-node-identity repair controls in `separability_diagnostic.py`, and set-valued labels
(makespan optima tie 2–34 deep, and `audit_label_provenance` asserts a unique minimum).

**Status: SUPERSEDED by the two probe entries below — route_a_v1 is CLOSED (NO-GO).** This
row documents the *machinery*, which stands and is reusable: DAG dispatch, fan-in, the
server mesh, the path-bandwidth transfer term, and the makespan channel are all
prerequisites for §9 route B, which is where the closing entry points next. Nothing here
is evidence for or against route A; the verdict is below.

---

### route_a_v1 — scaling probe: **NO-GO**, and the reason is a defect in the term, not a verdict on route A (2026-08-25)

The pre-registered go/no-go before spending an n≥200 corpus
(`scripts_cosim/score_route_a_scaling_probe.py`, thresholds fixed before any arm ran):
proceed only if spread-plan **additive-argmin regret > 5%** *and* it **rises with
`stateSize`**. Artifacts: `simulation_data/route_a_scaling_probe_final_{rtt,makespan}.json`.

**4 arms × 6 datasets (23 with sweeps), `stateSize` spanning 100,000× — 8 KB to 800 MB
transfer payloads:**

| `stateSize` | transfer payload | best RTT | spread-plan regret (rtt) | (makespan) |
|---:|---:|---:|---:|---:|
| 153,600 | 8 KB | 9.75 s | **0.000%** | **0.000%** |
| 15,360,000 | 800 KB | 10.34 s | **0.000%** | **0.000%** |
| 153,600,000 | 8 MB | 15.18 s | **0.000%** | **0.000%** |
| 15,360,000,000 | 800 MB | **950 s** | **0.000%** | **0.000%** |

Zero in every arm, every dataset, both objectives — `nonzero_frac = 0.00` throughout. Not a
marginal miss.

**Everything the probe needed was verified present**, which is what makes the diagnosis
below trustworthy rather than a shrug:
- the DAG is real — retained `task_times` show both children dispatching at the parent's
  completion (0.204) and the join dispatching at `max(0.891, 0.505)`;
- the mesh is real — 380 server↔server edges, **190 distinct latencies**, 4.8× spread;
- the term is material — at the top arm it *dominates*, taking RTT from ~10 s to ~950 s.

**So why exactly zero? The implemented term's magnitude-carrying half is separable by
construction.** `_dependency_transfer_time` charges

```
payload / bandwidth(CHILD's node)   +   latency(parent, child)
```

The first term is indexed by the **child alone** — a per-task cost, exactly the shape a
pointwise model already fits. Only `latency(parent, child)` is pairwise, and latency does
**not** scale with `stateSize`: it stays bounded at 0.031–0.149 s while the separable half
grows to hundreds of seconds. Raising `stateSize` therefore drove the *additive* term and
left the coupled term pinned. This is the same class of error as the `input`/`output` field
mismatch found earlier in the same probe (the lever initially scaled `input` while the
transfer reads the parent's `output`) — one level deeper.

**What this does and does not establish.**
- It does **not** show route A's hypothesis is false. The hypothesis — a child's cost
  depending on *where its parent went* — was never actually exercised at magnitude.
- It does show the term **as implemented cannot express that hypothesis**, and that
  `stateSize` is the wrong lever for it.

**What a real test needs:** the payload must divide by a **path** bandwidth between parent
and child (minimum link bandwidth along the route), not the child's local NIC — so that
distance carries magnitude rather than only a small additive latency. The machinery exists
(`NetworkFabric.route_links`, per-link bandwidth), but it requires the backbone enabled for
server↔server routes, which `build_core_backbone` now emits and `ROUTE_A_PILOT_V1_GRID`
does not yet turn on.

**Status of that first probe: NO-GO on that term, superseded below.** It was re-run against
a path-bandwidth term, as it said to.

---

### route_a_v1 — **NO-GO, now properly tested. Breaking separability is NECESSARY BUT NOT SUFFICIENT** (2026-08-25)

The probe above was rejected as a test of route A because the term's magnitude was
child-indexed. Both defects were fixed and it was re-run twice.

**Fix 1 — the payload is now pairwise.** `_payload_transfer_time` does store-and-forward
over the parent→child route (`n_hops × payload / bottleneck_bandwidth`), the same model the
ingress path already uses, instead of dividing by the child's own NIC. The preset now
requires a backbone, which yields **server↔server hop counts of 2–8** — so distance carries
magnitude, with a 4× spread between the nearest and farthest server pair.

**Fix 2 — `HEROSIM_OUTPUT_SIZE_BYTES`**, scaling the transfer payload *alone*.
`HEROSIM_STATE_SIZE_BYTES` moves `input` too, and `input` is a per-task storage read: at the
extreme arm it drove the episode to ~1000 s while the pairwise variation was ~4.6 s, burying
the coupled term **200:1**. Same error as the first probe, one level up.

**The properly isolated measurement** — baseline input, transfer payload 8/80/800 MB:

| payload | pairwise cost per remote parent | episode RTT | spread-plan regret (rtt / makespan) |
|---:|---:|---:|---:|
| 8 MB | 0.02–0.06 s | 10–100 s | 0.000% / 0.000% |
| 80 MB | 0.15–0.61 s | — | 0.000% / 0.000% |
| **800 MB** | **1.53–6.10 s** (4× by hop count) | 58–135 s | **0.000% / 0.000%** |

At the top arm the coupled term is roughly **10–30% of episode cost and varies 4× with the
parent/child pair** — and the componentwise minimiser is *still* exactly optimal, on every
dataset, both objectives. The scorer was also corrected mid-probe: it had been *dropping*
datasets where the componentwise plan is infeasible (the strongest coupling signal there
is), and now scores them by masked greedy decode, as a real pointwise scheduler would.

**The finding, and it is sharper than "route A failed":**

> **Non-separability is necessary but not sufficient.** `f_child(p_child, p_parent)` is
> genuinely pairwise here — the composition theorem's *hypothesis* is violated — and its
> *conclusion* still holds empirically. Breaking separability does not make the
> componentwise minimiser suboptimal. For that, the tasks' individually-best placements
> must **conflict**, so that one has to yield. Dependency + distance creates coupling
> without creating competition: every task can take its own favourite, and does.

That is why the five earlier co-location mechanisms and this one fail for the *same*
underlying reason, and it identifies what the program has never actually tried:
**contention for a scarce resource** — hard capacity, anti-affinity, exclusive GPUs — i.e.
§9's route B, which attacks the theorem's *free choice* hypothesis rather than its
separability hypothesis. The one hint already on record points the same way: the M3 pilot's
only non-collision-shaped escape (17.25% regret) came from the **distinct-node matching
constraint**, which is a feasibility restriction, not a physics term.

**Route B's stated caveat still applies and must be handled**: a grouped-argmax pointwise
decoder cannot represent a matching at all, so "GNN beats MLP under constraints" would be a
*decoder* result unless compared against a constraint-aware sequential pointwise decoder.

**Status: CLOSED — NO-GO.** Route A is tested and does not clear its pre-registered bar.
Do not soften the 5% threshold, and do not retry it with a larger payload: the term was
taken to 30% of episode cost with 4× pairwise variation and produced exactly zero. The
groundwork (DAG dispatch, fan-in, server mesh, path transfer, makespan channel) stands and
is reusable — route B needs all of it.

**Retro-audit against the mid-episode scale-down defect (2026-08-25, during route_b_v1
step (c)).** The defect that fix `42627d8` closes — `KEEP_ALIVE=30s` silently evicting a
task's FORCED replica before dispatch, truncating the sweep — was fixed 8 hours *after*
this NO-GO closed (`17792db` at 02:37 vs `42627d8` at 10:46), on the same physics
(locality on, DAG dispatch) that produces the truncation. This was a real, not
hypothetical, risk to the verdict above, and `score_route_a_scaling_probe.py` never
checked `placement_metadata.json`'s `sweep_complete` — only whether the file existed
(`n_missing_sweeps`) — so the original NO-GO could not have caught it either way. Checked
directly: regenerated 6 datasets from `ROUTE_A_PILOT_V1_GRID` at the 800 MB point, once
under `KEEP_ALIVE=30s` (route A's original condition) and once under the fix. **4 of 6
datasets were truncated under the original condition** (one losing 59% of its sweep,
33/56 rows) — confirming the defect was live during route A's own probe, not merely
theoretically possible. **Both corpora score identically: 0.000% spread-plan regret, 0
nonzero, NO-GO on both.** The truncation did not change the verdict on this rerun. This
is reassuring but is a 6-dataset spot-check, not a re-run of route A's original n≥200
scale corpus (which no longer exists on disk — gitignored, never persisted) — **the
NO-GO stands, now with the defect risk checked rather than merely disclosed, but the
original probe's own datasets were never re-verified because they no longer exist.**

---
