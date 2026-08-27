# route_b_env_pivot_v1 — ladder feasibility findings (2026-08-27)

**Status when written: no rung is readable.** H0 and H1 are VOID-INFEASIBLE under the
registered fallback; H2 as configured is VOID-GENERATION. None of this is a verdict on the
pivot hypothesis — the ladder is being defeated by configuration and decoder artifacts
before its bars are ever consulted.

> **Superseded in part the same day.** AMENDMENT 2 (signed off 2026-08-27) replaced the
> decoder behind `greedy_stuck`; **H0 and H1 now have clean counters** at α=2.0 and α=3.0
> and are no longer VOID-INFEASIBLE. H2 is unchanged and still VOID-GENERATION. **No S-bar
> has been read on any rung.** §1's and §4's counter tables remain correct as the
> *pre-amendment* forward-only numbers, which every rung's artifact still carries as
> `legacy_forward_only`.

This document records what was measured, so the next session does not re-derive it. It is
findings only: **no threshold, grid, or reading rule is changed here.** Every remedy in §6
needs its own registered amendment.

> **§9 is a second pass over this document's own findings (2026-08-27, later session).**
> Three of them were arm- or α-scoped and read differently once extended: §3's causal
> claim and §4.1's per-rung classification are **superseded** there. Read §9 before
> citing §3 or §4.1.

Companion documents: `docs/lineages/route_b_env_pivot_v1/screen-preregistration.md` (the registration),
`docs/lineages/route_b_env_pivot_v1/screen-amendment-1.md` (the S0 control definition),
`simulation_data/route_b_pivot_h0_reading.json` (H0's amended reading).

---

## 1. The registered fallback, and why it keeps firing

`docs/lineages/route_b_env_pivot_v1/screen-preregistration.md` §3: a rung is read at the tightest α on its ladder with
**clean counters** (`no_feasible_rows == 0` *and* `greedy_stuck == 0`); if none is clean,
the rung is VOID-INFEASIBLE.

Measured on the registered ladders (cap_mode as registered per rung):

| rung | α=1.5 | α=2.0 | α=3.0 |
|---|---|---|---|
| **H0** (alpha_max) | nofeas 204 | stuck 95 | stuck 87 |
| **H1** (alpha_mean) | nofeas 204 | nofeas 70, stuck 100 | stuck 83 |

No α on either ladder is clean. Both rungs are VOID-INFEASIBLE. H0's outcome was already
recorded; H1 now reads the same way.

## 2. There is no α that is both clean and binding — it is a cliff

The obvious question is whether some unregistered α would be clean. It would — but only by
switching the constraint off entirely. Fine sweep on H1 (204 datasets, `alpha_mean`):

| α | nofeas | stuck | binds on | mean feasible rows |
|---|---|---|---|---|
| 3.0 | 0 | 83 | 204/204 | 19.4 |
| 3.5 | 0 | 81 | 204/204 | 29.9 |
| 3.9 | 0 | 80 | 204/204 | 31.0 |
| **4.0** | 0 | **0** | **0/204** | 40.0 |

At α=4.0 the row is byte-identical to the unconstrained anchor (same mean feasible rows,
same R_exact). **"Clean counters" and "the constraint binds" are mutually exclusive on this
grid.** Relaxing α is therefore not a route to a readable rung, and must not be attempted
as one.

Verified on both cap modes and both rungs: α=4.0 and α=6.0 are clean-and-non-binding for
H0 and H1 alike, under `alpha_max` and `alpha_mean`.

## 3. On H0/H1, `greedy_stuck` is a configuration artifact

> **SUPERSEDED 2026-08-27 by §9. Read that first.** This section's causal claim was
> measured at **α=3.0** and is false at H1's **registered primary α=2.0**, where 71 of the
> 102 zero-confinement datasets are stuck. The correlation below is real; the *cause* is
> not confinement. §9 shows `greedy_stuck` is decoder myopia on **every** rung, H0/H1
> included — backtracking rescues 365/365 there, not just H2's 93.

`greedy_stuck` is entirely explained by **single-node confinement** — task types whose
candidate replicas all sit on one node. H1 at α=3.0:

```
(stuck, #tasks confined to a single node) -> count
  (False, 0) 102     (False, 2) 19     (True, 2) 83
```

No dataset with zero confined tasks is ever stuck. And confinement is deterministic per
replica-config arm, identically in H0 and H1:

```
(sweep_rows, confined_tasks):  (16, 0) -> 102     (64, 2) -> 102
```

### 3.1 Root cause: the allocator is first-come-first-served

`src/generate_infrastructure.py:625-660` walks `replicas_config` in dict order and marks
each chosen platform in `assigned_platforms`; without `replica_overlap` a later task type
may not reuse it. Early types take the platforms; later types get whatever single node
still has room.

The confined set is **always exactly `('rf', 'cnn')`** — the last two in iteration order —
102 times in every corpus examined.

### 3.2 More hosting nodes does NOT fix it (measured, and it was my first hypothesis)

`replica_server_percentage` maps through `k = max(1, int(server_node_counts * pct))`, so
0.5–0.7 → 2 nodes, 0.75–0.9 → 3, 1.0 → 4. Two full 204-dataset probes were generated:

| setting | hosting nodes | confined-task hist | α=3.0 stuck |
|---|---|---|---|
| 0.5 (registered) | 2 | `{0:102, 2:102}` | 83 |
| 0.75 (probe) | 3 | `{0:102, 2:102}` | 88 |
| 1.0 (probe) | 4 | `{0:102, 2:102}` | 92 |

**Identical confinement at 2, 3 and 4 hosting nodes**, and stuck counts drift slightly
*up*. Direct inspection at 4 hosts: `dnn1` spans node0–node3 while `rf`/`cnn` sit only on
node1 — despite both being eligible for `rpiCpu`, which is free on node0 and node3. It is
not platform scarcity and not platform-type eligibility. It is allocation order, and extra
nodes merely give the early types more to take.

> **Do not conclude "more hosting nodes = more dispersal" from this knob.** It changes how
> many nodes are *eligible* to host, not how the allocator distributes across them.

A methodological note worth keeping: the first probe used `--max-datasets 24` and read
"clean". That was an artifact — 24 datasets never reach the 64-row arm, which is the only
arm that strands. Probes of this grid must cover both arms or they are misleading.

## 4. `replica_overlap` (H2's registered lever) dissolves confinement

H2 (`route_b_pivot_h2`, `replica_overlap: True`) was generated. On the datasets that
completed, **confinement goes to zero**: the confined-task histogram is `{0: 102}`, against
`{0: 102, 2: 102}` everywhere else. All four types share one platform set spanning both
nodes:

```
dnn1/dnn2/rf/cnn -> [(node0,104),(node0,105),(node1,108),(node1,109)]
```

This is also the contested-indivisible-resource shape the pivot was designed to create:
one slot, many claimants.

Counters on the 102 completed datasets (`alpha_mean`):

| α | nofeas | stuck | binds on | mean feasible rows |
|---|---|---|---|---|
| 1.5 | 102 | 0 | 102/102 | 0.0 |
| 2.0 | 0 | 66 | 102/102 | 6.4 |
| 2.4 | 0 | 27 | 92/102 | 15.5 |
| 3.0 | 0 | 2 | 10/102 | 23.6 |
| 3.2 | 0 | **0** | **0/102** | 24.0 |

Stuck falls sharply (83→2 at α=3.0) but the same cliff remains: stuck hits zero exactly
where binding hits zero.

### 4.1 On H2 the remaining `greedy_stuck` is provably DECODER MYOPIA

This is the substantive difference, and it was tested rather than assumed. "A feasible plan
exists" is necessary but not sufficient, so both rescue paths were measured over the
**identical candidate sets and capacities**:

| α | stuck | feasible plan exists | rescued by a different task order | rescued by backtracking |
|---|---|---|---|---|
| 2.0 | 66 | 66/66 (100%) | 63/66 (95%) | **66/66 (100%)** |
| 2.4 | 27 | 27/27 (100%) | 27/27 (100%) | **27/27 (100%)** |

Inspected example at α=2.4: every task reaches both nodes with 4 candidates each, 20 of 24
plans are feasible, and the forward pass still strands.

So the classification changes by rung:

- **H0/H1** — `greedy_stuck` is a *configuration artifact* (two task types pinned to one
  node).
- **H2** — the environment is healthy; `greedy_masked_plan`'s single non-backtracking
  forward pass (`scripts_cosim/score_route_b_contention.py:432-468`) fails to find plans
  that demonstrably exist.

> **SUPERSEDED 2026-08-27 by §9.** The classification does **not** change by rung. The
> H2-only measurement above was one arm; extended to H0/H1 it gives the same 100%. The
> split above is an artifact of where the probe was pointed.

The registered fallback voids a rung on a dirty counter *regardless of cause*, so H2 would
still read VOID — now for a reason that has nothing to do with the environment under test.

## 5. H2 is VOID-GENERATION: overlap and the sparse arm are incompatible

H2 generated **102 SUCCESS / 102 SKIPPED**. §3's generation-integrity rule requires 204/204
SUCCESS, so the rung is VOID-GENERATION as configured.

Cause is pigeonhole against the no-replica-reuse mask, proven by direct enumeration:

```
SKIPPED ds_00000: 4 tasks, 2 candidates each, only 2 DISTINCT platforms -> 0 valid plans
SUCCESS ds_00051: 4 tasks, 4 candidates each,      4 distinct platforms -> 24 plans (= 4!)
```

Four tasks need four distinct platforms. In the `per_server=1` arm, `replica_overlap`
collapses every type onto the same 2 platforms, so **no unique assignment exists at all**.
Overlap fixes confinement and simultaneously destroys the sparser arm. The 24-row sweeps in
the surviving arm are exactly 4! — the signature of full overlap on 4 platforms.

### 5.1 Bug found: skip reasons are mislabelled under `replica_overlap`

The skipped datasets carry:

```json
{"reason": "too_many_combinations", "skip_threshold": 2000000}
```

but their pre-uniqueness product is **16**, not over two million.
`src/executecosimulation.py:1893-1899` assumes an empty combination list can only mean the
over-limit skip, because "the zero-candidate infeasible case returned earlier". Under
`replica_overlap` that assumption breaks: **uniqueness exhaustion also yields empty**, and
is then misattributed to the threshold.

Consequence: anyone reading these `skip_reason.json` files goes hunting for a
`MAX_PLACEMENT_COMBINATIONS_SKIP` problem that does not exist. This is a gate-tool
correctness defect and belongs in the LINEAGES GATE TOOLS table; it is *not* specific to
this lineage. Not fixed here.

## 6. Options, none taken

Each changes registered semantics and needs its own signed-off amendment. Listed with what
the measurements say about them, not ranked.

1. **Backtracking decoder.** Now strongly evidenced: 100% rescue at both α values tested,
   over identical candidate sets. Makes `greedy_stuck` measure the environment instead of
   the decoder. Changes a registered statistic, so it needs the same
   bug-fix-vs-moving-the-bar treatment the scorer fixes received (report both numbers, log
   the deviation, move no threshold).
2. **Amend the fallback** to distinguish decoder-stuck from environment-infeasible, leaving
   the decoder alone. Cheaper; arguably the more honest description of what the counter
   currently conflates.
3. **Raise `per_server`** so overlap still leaves ≥4 distinct platforms, making H2
   generable at 204/204. Changes the rung's scarcity, which is part of what it tests.
4. **Drop the sparse arm** from H2 only. Breaks the 204 = 2×2×3×17 shape the registration
   fixes for cell-for-cell comparability.
5. **Fix the allocator** (round-robin / least-loaded instead of FCFS, or per-seed shuffle of
   `replicas_config`). Addresses the H0/H1 root cause directly, but changes the
   infrastructure of every existing corpus — a new lineage, not an amendment.

Explicitly ruled out by measurement: **relaxing α** (§2, the cliff) and **raising
`replica_server_percentage`** (§3.2, no effect on confinement).

## 7. Artifacts

Corpora (all gitignored):

| path | what |
|---|---|
| `simulation_data/gnn_datasets_route_b_pivot_h0` / `_ctrl` | H0 main + control, 204/204 |
| `simulation_data/gnn_datasets_route_b_pivot_h1` | H1 main, 204/204 |
| `simulation_data/gnn_datasets_route_b_pivot_h1_ctrl` | **empty directory — H1's paired separable control was never generated** (see §9.4) |
| `simulation_data/gnn_datasets_dag4_route_b_pivot_h2` | H2, **102/204** — VOID-GENERATION |
| `simulation_data/probefull_rsp_075` | probe, 3 hosting nodes, 204/204 |
| `simulation_data/probefull_rsp_10` | probe, 4 hosting nodes, 204/204 |

The two `probefull_rsp_*` corpora exist only to support §3.2 and are not registered rungs;
they may be deleted once this document is accepted.

Reports: `simulation_data/route_b_pivot_h0_rtt.json`,
`route_b_pivot_h0_ctrl_rtt.json`, `route_b_pivot_h0_reading.json` (all committed).

## 8. What is NOT claimed here

- No statement about whether the pivot environment contains pointwise-irreducible
  structure. S1–S4 have not been read on any rung.
- No threshold, bar, grid, α ladder, or reading rule is modified by this document.
- H2's §4 numbers are over the 102 datasets that generated. They are diagnostic only —
  the rung is VOID-GENERATION and its bars must not be read from this half-corpus.

---

## 9. Second pass (2026-08-27, later session) — three of the above were arm-scoped

Same discipline, applied to this document's own findings: every number below states the
arm it was measured on. **No threshold, bar, grid, α ladder or reading rule is changed
here either.** Scripts are in the session scratchpad (`arm_contingency.py`,
`confine_by_alpha.py`, `rescue_h0_h1.py`), all read-only over the existing corpora.

### 9.1 `greedy_stuck` is decoder myopia on EVERY rung, not just H2 — supersedes §3, §4.1

§4.1's 100% backtracking rescue was measured on H2 only, and within H2 only on the 102
datasets that generated. Extended to the arm it was never measured on — over the identical
candidate sets, demands and caps the production greedy saw, same option ordering, the only
difference being that a dead end backtracks:

| rung | α | stuck | feasible plan exists | **backtracking rescues** | stuck by sweep-row arm |
|---|---|---|---|---|---|
| H0 | 2.0 | 95 | 95/95 | **95/95 (100%)** | `{64: 95}` |
| H0 | 3.0 | 87 | 87/87 | **87/87 (100%)** | `{64: 87}` |
| H1 | 2.0 | 100 | 100/100 | **100/100 (100%)** | `{16: 71, 64: 29}` |
| H1 | 3.0 | 83 | 83/83 | **83/83 (100%)** | `{64: 83}` |

**365/365 on H0+H1**, on top of H2's 93/93 — **458/458 across the ladder.** Every dataset
the greedy stranded had a feasible plan sitting in its own enumerated sweep.

So `greedy_stuck` has never measured the environment on any rung. It is a property of
`greedy_masked_plan`'s single non-backtracking forward pass, and it is **logically
redundant** with `no_feasible_rows`: the two search the same masked space, so a complete
search succeeds exactly when a feasible row exists. §3's confinement story survives only as
a *correlation* — see 9.2 for where it breaks.

### 9.2 §3's causal claim holds at α=3.0 and fails at the registered primary α=2.0

§3's histogram was quoted at H1 α=3.0. Reproduced at every registered α:

| rung | α | (stuck, confined_tasks) | stuck with ZERO confined tasks |
|---|---|---|---|
| H0 | 2.0 | `{(F,0):102, (F,2):7, (T,2):95}` | 0 |
| H0 | 3.0 | `{(F,0):102, (F,2):15, (T,2):87}` | 0 |
| H1 | 3.0 | `{(F,0):102, (F,2):19, (T,2):83}` | 0 |
| **H1** | **2.0** | `{(F,0):31, (F,2):73, (T,0):71, (T,2):29}` | **71** |

"No dataset with zero confined tasks is ever stuck" is true on three of the four cells and
false on the fourth — which is H1's own **registered primary α**. Under H1's levers
(`demand_spread` + `cap_mode: alpha_mean`) the caps stop tracking the sweep's own max, so
the 16-row arm's caps genuinely bind and strand a decoder that has no confinement at all.
Same defect class as everything else in this file, on the α axis rather than the arm axis.

### 9.3 `no_feasible_rows` is confounded with the arm, and had no breakdown reporting it

The greedy-denominator fix (GATE TOOLS, 2026-08-27) gave `greedy_stuck` an arm breakdown.
The **stricter** censor — the one that removes a dataset from `r_exact` and every LS/repair
statistic, not just from `r_greedy` — had none. Keyed on the unconstrained sweep size:

| rung | α | `no_feasible_rows` by arm | `n_exact_scored` by arm | `n_greedy_scored` by arm |
|---|---|---|---|---|
| H0 | 2.0 | `{}` | `{16:102, 64:102}` | `{16:102, 64:7}` |
| H0 | 3.0 | `{}` | `{16:102, 64:102}` | `{16:102, 64:15}` |
| **H1** | **2.0** | **`{64: 70}`** | **`{16:102, 64:32}`** | **`{16:31, 64:3}`** |
| H1 | 3.0 | `{}` | `{16:102, 64:102}` | `{16:102, 64:19}` |

At H1's primary α the 64-row arm is 69% censored from `r_exact` and the 16-row arm 0%; the
greedy denominator there is three datasets from one arm and 31 from the other. Now reported
in every rung's artifact as `censoring_by_arm` (additive; 0 pre-existing values moved).

### 9.4 H1's paired separable control was never generated

`simulation_data/gnn_datasets_route_b_pivot_h1_ctrl` exists as an **empty directory**.
SCREEN §3 requires a paired separable control per rung and §4's S0 reads
`r_exact.frac_gt_1pct ≤ 0.02` on it. Moot while H1 is VOID-INFEASIBLE — a control cannot
rescue a rung that cannot be read — but it must be generated before H1 is ever read, and
nothing in the record said it was missing.

### 9.5 What this changes about §6

Nothing is taken; the options still need their own signed-off amendment. But the evidence
under them has moved:

- **Option 1 (backtracking decoder)** is no longer an H2-specific remedy. It removes the
  blocker on **every** rung. Under it — and with the registered fallback untouched —
  "clean counters" reduces to `no_feasible_rows == 0`, which makes **H0 readable at its
  registered primary α=2.0** (nofeas 0, binds 204/204) and **H1 readable at α=3.0** (nofeas
  0, binds 204/204) by the fallback exactly as written. No threshold moves.
- **Option 2 (amend the fallback to distinguish decoder-stuck from environment-infeasible)**
  reaches the same rungs by relabelling rather than by fixing the decoder, and leaves
  `r_greedy` — a registered statistic — still produced by a decoder that demonstrably
  fails to find plans that exist. §4.1's argument for it as "the more honest description"
  is weaker now that the counter is known to describe the tool on every rung.

**Option 1 was recommended, signed off by the user the same day, and landed** —
`docs/lineages/route_b_env_pivot_v1/screen-amendment-2.md`, whose §8 records the discharged
obligations. H0 and H1 now have clean counters at α=2.0 and α=3.0 respectively; §6 options
3, 4 and 5 remain open for H2, which is untouched and still VOID-GENERATION. **No S-bar has
been read on any rung.**
