# dag_fabric_contention_v1 — FALSIFIED

> **Status:** `FALSIFIED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-09-08 → 2026-09-08

**Outcome.** Paper screen, **NO-GO before any physics was built.** Asked whether serving the
parent→child *output* payload (route_b's 800 MB lever, today charged additively by
`_payload_transfer_time` with no pipe held) store-and-forward over the contended link fabric
would make link waiting a material, **non-count-shaped** share of the coupled tasks' RTT *at
the optimum*. Computed on the stored `arm_s` sweep (204 datasets, α = 2.0 feasible set): the
manipulation bar would pass at ≤ 100 MB/s (median link-wait share of children's RTT 0.17 /
0.45 at 100 / 25 MB/s), but the α = 2.0 optimum carries **zero** link wait in 98–100% of
datasets at every hosting-node count, including the 17 squeezed to 2 nodes — avoidance is
free, the `link_contention_v1` / θ* = 0 shape. At 8 tasks (two instances) the joint optimum
cannot avoid *some* wait in ~50% of datasets, but that wait sits on the root's / sink's own
**access** link (a remote-children / remote-parents count); the genuinely non-count **core**
sharing binds at the optimum in 10–12% of datasets, under the 0.25 firing bar before any
control. The name is granted only to record the measurement; **no corpus, no code.**

**Related:** [route_c_link_transfer_v1](route_c_link_transfer_v1.md) ·
[link_contention_v1](link_contention_v1.md) · [route_b_v1](route_b_v1.md) ·
[route_b_env_pivot_v1](route_b_env_pivot_v1.md)

**Entry points:** `scripts_cosim/dag_fabric_ceiling_probe.py` (all three stages; no
simulation) → `simulation_data/dag_fabric_ceiling_probe.json`.

## Record

- [dag_fabric_contention_v1 — PAPER SCREEN, NO-GO (2026-09-08)](#dag-fabric-contention-v1-paper-screen-no-go-2026-09-08)

---

### dag_fabric_contention_v1 — PAPER SCREEN, NO-GO (2026-09-08)

**Why this was the candidate.** The program's throughline is that in this simulator coupling
is either count-shaped or negligible: every node-indexed mechanism is a symmetric function of
a node's co-resident multiset and therefore a function of per-type counts, which the
`hetdem`/`krank` pointwise blocks express (`route_b_env_pivot_v1`, extended pooled closure
0.892). The one mechanism that ever escaped the count control is **link-indexed** contention
(`link_contention_v1`: link-repair medians 0.000) and its only defect was magnitude
(0.08–0.35% regret). `route_c_link_transfer_v1` tried to buy magnitude with the *input*
payload and hit a structural ceiling (one client, a diamond ⇒ ≤ 2 concurrent transfers ⇒
link-wait share of RTT ≤ 4–10%). The untried lever was the **output** payload: at 800 MB it
is already the corpus's dominant coupled term (a child costs 4.78 s with its parent local vs
9.54 s remote, `route_b_v1` 2026-09-07), and `src/placement/infrastructure.py:929-968`
computes its hop count and bottleneck from `fabric.hops()` **without ever requesting
`fabric.pipe()`** — so today it is additive by construction. The seam is one generator.

**Registered GO rule (plan of 2026-09-08, before computing).** Bandwidth-free ceiling
`wait/(wait+transfer)` for k concurrent equal transfers on one capacity-1 link is
((k−1)/2)/(1+(k−1)/2): k = 2 → 0.33, 3 → 0.50, 4 → 0.60. GO iff the median ceiling on the
α = 2.0 feasible set ≥ 0.25 at 4 tasks or ≥ 0.40 at 8. The plan did not say which ceiling
— `any` link or `core`-only — so both are reported; the rationale (survive the count control)
makes `core` the binding one, since two remote children always share the parent's own access
link, which is a parent-indexed count.

**Stage 1 — ceiling** (`arm_s`, 204 datasets, hosting nodes {2: 17, 3: 78, 4: 84, 5: 25},
fabric = 12-router ring, attach_degree 1):

| statistic (median over datasets) | `any` link | `core` only |
|---|---|---|
| fraction of α = 2.0 plans with k ≥ 2 | 0.703 | 0.375 |
| median plan ceiling, 4 tasks | **0.333** | **0.000** |
| optimum's k (histogram over 204) | 0: 84 · 1: 97 · 2: 23 | 0: 103 · 1: 92 · **2: 9** |
| median ceiling, 8 tasks (300 sampled instance pairs / ds) | 0.500 | 0.333 |
| fraction of 8-task pairs sharing a core link | — | 0.933 |

GO rule: `any` passes at both sizes; `core` fails at both (0.000 < 0.25; 0.333 < 0.40).

**Stage 2 — synthetic waits on the stored sweep.** rtt′(plan) = rtt + Σ_phases Σ_links
T_hop·k(k−1)/2, T_hop = 800 MB / bw, simultaneous starts (an upper bound); children's RTT
from `task_times`.

| bw (MB/s) | manipulation share, median | share ≥ 0.10 (frac ds) | α=2.0 optimum moved | new optimum has **any** wait | has **core** wait | additive-argmin regret > 5% | … after one per-parent count column |
|---|---|---|---|---|---|---|---|
| 1000 | 0.020 | 0.00 | 0.029 | 0.083 | 0.025 | 0.76 | 0.38 |
| 100 | 0.168 | 0.84 | 0.088 | **0.025** | **0.000** | 0.90 | 0.48 |
| 25 | 0.446 | 0.88 | 0.113 | **0.000** | **0.000** | 0.93 | 0.54 |

By hosting-node count at 100 MB/s, the new optimum carries any wait in 0.0% (2 nodes),
2.6% (3), 1.2% (4), 8.0% (5) of datasets and core wait in **0.0% at every count**. The
mechanism would pass the manipulation check and destroy the additive fit of the *sweep*
(R² 0.70 → 0.30 → 0.13) while leaving the *optimum* untouched: the label never sees it.
This is the `docs/lessons.md` 2026-08-18 warning verbatim — "the ratio is measured on
optimal plans, which select against contention" — now on the output payload.

**Stage 3 — joint 8-task optimum** (60 datasets, two instances, α = 4.0 equal-tightness cap,
exhaustive over feasible × feasible, ≤ 60k pairs):

| bw | joint optimum has any wait | has core wait | zero-wait joint plans (median frac) | avoidance premium > 5% (frac ds) | wait share at optimum, median |
|---|---|---|---|---|---|
| 100 | 0.50 | **0.117** | 0.001 | 0.48 | 0.037 |
| 25 | 0.47 | **0.100** | 0.001 | 0.47 | 0.000 |

At 8 tasks avoidance is no longer free in about half the datasets (the premium is bimodal:
0 or > 5%), but what binds is the access link into the sink / out of the root — a count of
remote parents / children per node that one column per node repairs (stage 2's last column
shows the 4-task version of that repair). Core sharing binds at the optimum in 1 dataset in
8–10.

**Reading.** NO-GO on the binding (`core`) measure at both sizes; the `any` pass is the
count-shaped part. Two structural reasons, neither a tuning miss: (i) on a 12-router ring
with 6 servers, disjoint core paths are plentiful and the optimum takes them; (ii) funnelling
the backbone (small `n_core`, tree) to force core sharing turns the shared term into
"number of remote DAG edges" — a sum of pairwise parent→child indicators, i.e. the
hop+coupling block `route_b_v1` §9d already measured closing 0.997 at 8 tasks. Not built;
not proposed for tuning. `docs/hard-stops.md` carries the entry.

**What would reopen it.** A measured optimum-level core-link binding fraction ≥ 0.25 on some
*other* substrate (more instances per batch than the enumerable sweep allows, or transfers
whose concurrency is not capped by a diamond's fan-out) — which is the concurrency wall
`route_c_link_transfer_v1` already hit.

**Side findings filed elsewhere.** The H2 separable control's non-additivity is the
parent-locality branch, not sibling co-residency (`route_b_env_pivot_v1`, 2026-09-08). The
root task's duration on `arm_s` is exactly pointwise (R²_own = 1.000), confirming the
`route_b_v1` 2026-09-07 attribution of the 39 s term to warmup writes
(`scripts_cosim/task_duration_decomposition.py`).
