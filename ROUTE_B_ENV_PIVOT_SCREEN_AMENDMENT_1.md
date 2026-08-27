# route_b_env_pivot_v1 — AMENDMENT 1: the S0 separable-control definition

**Written 2026-08-27, BEFORE any H1 control corpus exists or is scored.** Amends
`ROUTE_B_ENV_PIVOT_SCREEN.md` @ `019bdcb` (signed off; LINEAGES entry `6b7b915`).
Nothing in §4 executes until this document is signed off and a LINEAGES entry records it
at its commit SHA.

**Scope of this amendment: the S0 control DEFINITION only.** No bar value changes. No
rung, α ladder, seed block, grid, or reading rule changes. S1, S2, S3, S4 are untouched.

## 1. Why the registered control definition is defective

§3 of the registration defines each rung's paired separable control as "same grid and
seeds, `HEROSIM_DATA_LOCALITY`/`HEROSIM_OUTPUT_SIZE_BYTES` unset", and S0 requires its
`r_exact.frac_gt_1pct ≤ 0.02`.

**That ablation does not produce separable physics, and cannot.** Measured 2026-08-27 on
the already-generated `gnn_datasets_route_b_pivot_h0_ctrl` (204 datasets):

`src/placement/infrastructure.py:1244-1252` computes

```python
local_dependencies = all(
    dependency.storage["output"] in self.node.storage.items
    for dependency in task.dependencies
)
```

and `:1265-1280` then selects the child's `input_storage`: local `flashCard` when
`local_dependencies` holds, otherwise remote `someRemote`. From
`data/nofs-ids/storage-types.json`, on the welded 153,600 B input:

| tier | latency | throughput | read cost |
|---|---|---|---|
| `flashCard` (local) | 0.00012 s | 235 MB/s | 0.000743 s |
| `someRemote` (remote) | 0.015 s | 108 MB/s | 0.016356 s |

**Δ = 0.0156 s per task, charged as a function of where the task's PARENTS ran.** That is
precisely a non-additive, pairwise parent→child term — the thing the control exists to
exclude. It is **all-or-nothing** (`all(...)`): one remote parent prices identically to
all-remote.

This term is **always on**. It is *not* `_dependency_transfer_time` (route_a's data
locality), which correctly returns `0.0` when `HEROSIM_DATA_LOCALITY != 1`
(`infrastructure.py:981`). It is the storage tier immediately above it, whose own comment
concedes the remote arm "charges a constant `someRemote` latency, blind to where the
parent actually ran". Unsetting the two Arm-S env vars removes only the *pivot-era*
coupling terms and leaves this one untouched.

### Evidence

- Fitting an exact per-`(task, placement)` additive model to the full 16-row sweep leaves a
  near-constant **~0.040 s** max residual. Adding a single per-task "all parents local"
  indicator cuts it **4–8×**: ds_00114 0.03993→0.00953, ds_00038 0.04001→0.00468,
  ds_00111 0.03993→0.00995, ds_00145 0.03997→0.00764, ds_00017 0.03997→0.00403,
  ds_00005 0.03997→0.01087.
- **The DAG root is perfectly additive.** Task 0 (`diamond4`, no parents) has *zero*
  duration spread across all 16 plans; tasks 1, 2, 3 (which have parents) all vary. That is
  the signature of a parent-mediated term and of nothing else.
- Direct per-task read: task 1 on platform `(20,105)` takes 0.325449 s when its parent ran
  on node0 and 0.356083 s when on node1 — Δ 0.030633 s ≈ 2 × 0.0156.

### Falsified alternative (recorded, per §6 of the registration)

The `node_disk_v2` cold-pull-serialization hypothesis (`HANDOVER_route_b_env_pivot.md`
§2.3) is **FALSIFIED**. The predicate "≥2 queue-empty replicas on one node" holds for
31/204 = 15.2%, coincidentally close to the reported 15.5% — but the contingency table
against genuine (tie-corrected) coupling is `cold≥2 & genuine = 0 | cold≥2 & clean = 31 |
cold<2 & genuine = 21 | cold<2 & clean = 152`. Zero correlation. Corroborated by
`optimal_result.json`: `averagePullTime = 0.0`, `averageColdStartTime = 0.0`.

## 2. Why this is an amendment and not a bar move

The registered bar `frac_gt_1pct ≤ 0.02` **does not change**. What changes is which corpus
is the control — i.e. the definition of the thing being measured, which the registration
got wrong on a matter of fact about the simulator.

The distinction matters and is auditable: the current definition is **unpassable by
construction** on any DAG corpus whose constrained feasible-set spread is narrower than
0.0156 s. It is not measuring the pivot's physics; it is measuring an unrelated
always-on storage branch. Under §5 a rung whose control fails is VOID, so leaving the
definition as-is would VOID the entire ladder on a defect that has nothing to do with the
pivot hypothesis — a null result about the wrong thing.

**Registration-integrity constraints honoured here:**

- Written and signed off **before** any H1 control corpus is generated or scored. It is not
  post-hoc: no H1 control number exists.
- Bar value untouched; `mean_tied` remains the fair reading (per `Cell.tie_band`'s
  docstring and §4).
- The H0 control's numbers under the OLD definition are recorded, not discarded (§5 below).
- Every deviation is logged in the LINEAGES outcome entry with its evidence, per §6 of the
  registration.

## 3. The amended control definition

A rung's paired separable control is generated with the **same grid, same seeds, same
`replica_configs`** as its main corpus, with:

1. `HEROSIM_DATA_LOCALITY` unset and `HEROSIM_OUTPUT_SIZE_BYTES` unset (unchanged), **and**
2. **`HEROSIM_STORAGE_NEUTRAL=1`** (new, defined in §4): the local and remote storage tiers
   are made cost-identical for the read path, so `local_dependencies` cannot change a
   task's cost.

Everything else about the control is unchanged, including `HEROSIM_COSIM_KEEP_ALIVE`,
`HEROSIM_RETAIN_TASK_TIMES`, the skip cap, and `node_disk_v2`.

**Predicted consequence, registered now as a falsifiable claim:** under this definition the
control's `r_exact` band should read `optimistic == 0` on essentially every dataset, and
`mean_tied.frac_gt_1pct` should fall at or below the 0.02 bar. **If it does not, the
control is still not separable and the rung is VOID-S0** — this amendment does not license
reading past a control that still fires. Stated as a prediction so it can fail.

## 4. Build item A1 — `HEROSIM_STORAGE_NEUTRAL` (VOID if skipped)

Opt-in, default-off, byte-identity verified — the same contract every other pivot lever
carries.

- **Where:** `src/executecosimulation.py`, a new `apply_storage_neutral_override(sim_inputs)`
  called immediately after `apply_state_size_override(sim_inputs)` at `:340`, where
  `sim_inputs["storage_types"]` has just been loaded. In-memory mutation only —
  `data/nofs-ids/storage-types.json` is shared by every corpus and is never edited, exactly
  the reasoning `apply_state_size_override`'s docstring gives at `:390-407`.
- **What:** when `HEROSIM_STORAGE_NEUTRAL=1`, pin **both** tiers' `throughput.read` to a
  common value (`HEROSIM_STORAGE_NEUTRAL_READ_MBPS`, default 100.0) and both `latency.read`
  to `flashCard`'s. Write path, capacity, `remote` flag, iops and energy are **untouched** —
  only the read cost the `local_dependencies` branch prices. Fail loudly if either tier is
  missing or the throughput is non-positive.

  **Why both tiers, clamped down, rather than raising remote to local's 235 MB/s:**
  `infrastructure.py:1288-1294` prices the remote arm at
  `min(throughput.read, node.network.bandwidth)` and the local arm at `throughput.read`
  *unclamped*. Every node in `data/nofs-ids/infrastructure.json` is **100 Mbps**, so simply
  equalising the two `throughput.read` fields still left **0.00084 s** of parent-locality
  coupling — an 18.7x reduction from 0.01572 s, but not zero (measured 2026-08-27 during
  A1 implementation). Pinning both at or below the fabric bandwidth makes the `min()` a
  no-op and the two arms bit-identical. A "negligible" residual is not acceptable in a gate
  that decides VOID-S0.
- **Not** a change to the branch itself. `local_dependencies` still computes, still selects
  `someRemote`; the two arms simply cost the same. This keeps the blast radius inside the
  control corpus and leaves every frozen corpus's physics untouched.

**Tests (in `tests/`, with verified teeth — must fail before the change):**

- `test_parent_locality_gap_exists_without_the_override` — pins the defect itself at
  0.015722 s, so the amendment's premise fails loudly if it ever goes stale.
- `test_storage_neutral_zeroes_parent_locality_cost` — local and remote read costs are
  **exactly** equal (not `approx`). Fails before.
- `test_storage_neutral_survives_the_node_bandwidth_clamp` — gap is exactly 0.0 at 50, 100,
  235 and 1000 Mbps. Regression for the near-miss described above.
- `test_storage_neutral_is_inert_when_unset` — byte-identical `sim_inputs` with the var
  absent, `=0`, `=""` and `=false`.
- `test_storage_neutral_leaves_write_path_untouched` — `throughput.write`/`latency.write`,
  `remote`, `capacity` and `iops` unchanged, so warmth/disk semantics are not perturbed.
- `test_storage_neutral_read_mbps_is_overridable` / `..._rejects_nonpositive`.
- `test_storage_neutral_fails_loudly_on_missing_tier` — no silent skip.
- `test_storage_neutral_is_wired_into_the_sim_input_loader` — the lever must actually be
  called from `load_simulation_inputs`; one that exists but is never invoked is how a
  control corpus silently keeps its coupling.

**Byte-identity gate:** regenerate one H0 *main* dataset with the var unset and confirm the
sweep is byte-identical to the committed one. The lever must not touch Arm S.

## 5. What happens to H0

H0's reading is **unchanged: VOID-INFEASIBLE** — α=1.5 is 204/204 `no_feasible_rows`,
α=2.0 and α=3.0 both have `greedy_stuck > 0`, so no α on its registered ladder has clean
counters and §3's fallback is terminal for the rung. That verdict does not depend on the
control at all.

H0's control numbers under the **old** definition are recorded in the outcome entry as
measured, not deleted: at α=2.0, over all 204 (denominator-corrected),
`registered` 0.0784 / `mean_tied` 0.0392 / `optimistic` 0.0196 / `pessimistic` 0.0833; the
pre-fix published 0.1553 was inflated by the two scorer defects
(`decode_regret` id tie-break; `r_exact` censored by `greedy_stuck`).

**H0 is not re-generated under the amended definition.** It is VOID on feasibility
regardless, so a storage-neutral H0 control would cost a full generation pass to change a
verdict that is already settled. If the ladder later needs an H0-shaped calibration point,
that is a new decision.

## 6. Scope exclusions (carried forward, plus one)

Unchanged from §6 of the registration: no physics changes to Arm S, no training, no caches,
no checkpoints, no `task-types.json` demand-table edits, no new decode modes, thresholds
immutable.

**Added:** `HEROSIM_STORAGE_NEUTRAL` is a **control-arm-only** lever. It must never be set
on a main (Arm S) corpus. Setting it on a main corpus would remove a real coupling term
from the very physics the screen is testing, which would fabricate a pass. Any rung whose
main corpus was generated with it set is VOID-GENERATION.

## 7. Sign-off

Amends `ROUTE_B_ENV_PIVOT_SCREEN.md` @ `019bdcb`. Requires: user sign-off, a LINEAGES
registration entry at this document's commit SHA, and build item A1 landed with its tests
green, **before** any H1 control corpus is generated.
