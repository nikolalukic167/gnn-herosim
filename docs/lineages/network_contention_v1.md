# network_contention_v1 — SUPERSEDED

> **Status:** `SUPERSEDED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-18 → 2026-08-18

**Outcome.** Shared per-node ingress bandwidth: **the physics works and is opt-in**, but the corpus lever is replica concentration, not bandwidth. One of the four mechanisms in the throughline.

**Entry points:** `src/placement/scheduling_cost.py` (`ingress_transfer_time`, `ingress_wait`), `scripts_cosim/test_network_contention.py`, `datalab/netc_v1_cosim.sbatch`, grids `netc_{scarce,funnel,hotspot}_v1`

**Datasets:** `netc_pilot_*` (local, n=12-16)

**Related:** [throughline](throughline.md) · [link_contention_v1](link_contention_v1.md)

## Standing (from the index table)

Shared per-node ingress bandwidth, opt-in via `--ingress-bandwidth-mbps`. Physics works; the corpus lever is replica concentration, not bandwidth alone. **Outcomes below.**

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [network_contention_v1 — outcomes (2026-08-18)](#network-contention-v1-outcomes-2026-08-18)

---

### network_contention_v1 — outcomes (2026-08-18)

**The physics works and is opt-in.** Each server node gets one shared ingress pipe;
propagation latency stays an un-serialized timeout (additive), while input transmission
(`stateSize / bandwidth`) is served through the pipe, so concurrent inbound transfers
queue. Unset ⇒ no pipe, no transmission term, `node_disk_v2` physics. The transfer formula
lives once in `scheduling_cost.py` and is imported by both the simulation and the ECT
mirror. 12 tests in `test_network_contention.py`.

Verified on a matched A/B (identical workload and topology, only the flag differs): mean
RTT delta by extra co-located tasks was **3.20 → 4.63 → 6.50 transfers** — monotone, i.e.
plans that co-locate pay real *waiting*. Contrast `node_contention_v3`, whose
`nodeContentionTime` was 0.0 everywhere.

**⛔ BLOCKER FIXED — the workload draw was unseeded.** `generate_workload_templates` drew
task source nodes from the *global* `random`, with no `random.seed()` anywhere in the
generator. Two runs of the same grid with the same seeds produced different workloads and
RTTs. **No existing corpus is reproducible from its recorded seed, and matched A/B arms
were impossible** — `node_contention_v3`'s "metrics disagreed and were underpowered" pilot
was built this way. Now a local `random.Random(--workload-seed)`, default 42. Infrastructure
was always deterministic; `placements.jsonl` row *order* still varies (parallel completion
order) — compare sweeps as sets, never with `diff`.

**Bandwidth alone does not create a joint decision.** M4 moves monotonically with dose but
M1 stays at exactly 0%:

| arm | additive R² | collision gain (plat/node) | additive-argmin regret mean/max | argmin optimal | M1 coupled |
|---|---|---|---|---|---|
| baseline | 0.9667 | 2.62 / 2.76 pp | 3.01% / 15.5% | 67% | **0%** |
| 1.5 MB/s | 0.9596 | 3.32 / 3.46 pp | 3.94% / 19.6% | 58% | **0%** |
| 0.5 MB/s | 0.9478 | 4.38 / 4.57 pp | 2.42% / 14.7% | 33% | **0%** |

**Why: spreading was free.** A no-simulation pre-check (min premium `θ*` admitting a
task→distinct-node matching, by Hall's condition) found `θ* = 0` in **92%** of datasets —
each task's single favourite node was already distinct (3.83 of 4 tasks).

**`FALSIFIED` — scarcity-by-count and topology funnelling.** Both *reduced* the overlap
they were meant to create, because shrinking candidate sets makes them more **disjoint**
when tasks are anchored to different clients. Mean pairwise candidate-node overlap
(of 4 tasks) and union:

| grid | cand/task | pairwise overlap | union nodes | θ*=0 |
|---|---|---|---|---|
| shallow_v1 | 4.56 | 0.93 | 13.58 | 92% |
| `netc_scarce_v1` (sparser links) | 3.23 | **0.36** | 11.00 | 92% |
| `netc_funnel_v1` (degree_skewed_core) | 2.18 | **0.14** | 8.00 | **100%** |

**`ACTIVE` — replica concentration is the lever.** The blocker was
`generate_infrastructure.py`'s `replica_server_pct = max(server_pct, 0.6)`, which spread
replicas over ≥60% of servers whatever the grid asked. Now overridable via
`preinit.replica_server_percentage` / `--replica-server-percentage`. Dense links + few
replica hosts gives overlap 2.29, union **2.56 nodes for 4 tasks** (pigeonhole), θ*=0 in
**0%** of datasets:

| replica_server_pct | additive R² | collision gain | additive-argmin regret mean/max | argmin optimal | sweep size | best RTT |
|---|---|---|---|---|---|---|
| 0.6 (floor) | 0.9478 | 4.4 pp | 2.42% / 14.7% | 33% | 456 | 0.76 s |
| 0.45 | 0.9210 | 5.6 pp | 1.40% / 9.6% | 62% | 130 | 1.44 s |
| 0.30 | 0.8280 | 12.2 pp | 2.13% / 13.9% | 69% | 92 | 3.44 s |
| **0.15 + 0.5 MB/s** | **0.5645** | **31.7 pp** | **21.51% / 93.0%** | **25%** | 38 | 10.8 s |

Ingress bandwidth roughly **doubles** the effect at 0.15 (collision gain 16.5 → 31.7 pp,
argmin optimal 44% → 25%), so the two levers compose.

**⚠ METHODOLOGICAL — M1 is the wrong gate, and the pre-registered gate would have rejected
the one configuration that works.** At `hotspot + 0.5 MB/s` the M1 coupled fraction is
**0%** while a pointwise fit picks a suboptimal plan **75% of the time at 21.5% mean
regret**. M1's "marginal greedy" scores each per-task option as `min RTT over all joint
plans with task t there` — it has **oracle access to the joint sweep**, so it is not a
pointwise model and trivially recovers the optimum once the candidate set is small.
**The statistic matched to `PointwiseEdgeMLP`'s expressive power is M4's additive-fit
argmin** (`additive_choice_regret_rel` / `additive_choice_is_optimal`), since the additive
fit is literally what that model class can express. Gate on that, not on
`--gate-coupled-fraction`.

**`FALSIFIED` — the hotspot (0.15) configuration is degenerate. Do not build on it.**
Two checks settled it, both on existing local data.

*Cliff, not a regime.* Filling in the gap between 0.30 and 0.15 (host-node counts in
brackets) shows the intermediate points sit in one flat band and 0.15 jumps
discontinuously:

| replica_server_pct | hosts | sweep | additive R² | coll gain | argmin regret mean/max | argmin optimal |
|---|---|---|---|---|---|---|
| 0.6 (floor) | 14 | 456 | 0.9478 | 4.4 pp | 2.42% / 14.7% | 33% |
| 0.45 | 9 | 130 | 0.9210 | 5.6 pp | 1.40% / 9.6% | 62% |
| 0.30 | 7 | 92 | 0.8280 | 12.2 pp | 2.13% / 13.9% | 69% |
| 0.25 | 7 | 76 | 0.9049 | 6.1 pp | 3.36% / 19.6% | 69% |
| 0.20 | 6 | 80 | 0.8883 | 7.8 pp | 4.26% / 33.5% | 62% |
| **0.15** | **2** | **38** | **0.5645** | **31.7 pp** | **21.51% / 93.0%** | **25%** |

*The interaction is one integer, again.* Adding a **single** per-plan node-occupancy-excess
column to the additive fit repairs **9/12 (75%)** of its wrong picks at 0.15 and cuts mean
regret **21.51% → 4.15%**. The failures concentrate on **2 distinct node identities — which
are the only two hosts that exist**. With 4 tasks over 2 nodes the "joint decision" is just
*how many tasks land on node 21 vs node 22*: a scalar a pointwise MLP learns from one extra
feature, not graph structure. This is the same trap already recorded above ("the whole
interaction is one integer"), reproduced at larger magnitude. By contrast that column
repairs only 33-40% of failures at 0.20/0.25, where failures spread over 4 node identities.

*My own sweep was confounded.* `netc_hotspot_v1` changed **two** things at once —
`replica_server_percentage` 0.6→0.15 **and `per_client` 1→0**, which deletes client-local
replicas entirely. The intermediate arms varied only the percentage, so they never fell
below 6 hosts. The cliff coincides with removing client-local replicas, not with crossing a
percentage. "0.15" is therefore not even a clean descriptor of the regime.

*No realism anchor.* A literature check found no reported "fraction of nodes hosting
replicas" metric in serverless-edge placement work, so 0.15 has no external justification —
it is where the arithmetic moved, which is exactly the reasoning `netc_scarce_v1` was
careful to avoid.

**Retroactive scope of the M1 finding: nil for automated gating, real for reasoning.**
`--gate-coupled-fraction` appears in no committed script, sbatch or log — **no corpus was
ever accepted or rejected by it**. But the recommendation to make it primary is recorded in
both `LINEAGES.md` and `memory/memory.md`, and shallow_v1's since-retracted "31.0% coupled"
was the stated justification for the whole shallow lever. Correct the recommendation:
**gate on M4's `additive_choice_regret_rel` / `additive_choice_is_optimal`**, and always
report how much of the regret a single collision-count column repairs — without that
control, a degenerate one-integer corpus looks like a GNN opportunity.

**Nothing has been trained; no ablation has been run; nothing submitted to datalab.**
