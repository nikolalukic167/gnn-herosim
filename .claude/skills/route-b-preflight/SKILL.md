---
name: route-b-preflight
description: Preflight and arm-coverage discipline for the route_b_env_pivot_v1 screen — the 204-dataset 2×2×3×17 corpus where every defect so far has been an assumption that held on one arm and broke on a neighbouring one. Load before trusting, quoting, or acting on any number from a route_b rung; before editing score_route_b_contention.py, verify_route_b_scorer_agreement.py, route_b_coefficient_transfer.py or the co-sim skip/combination path; and before proposing any change to a registered threshold, bar, grid, α ladder or reading rule. Not a description of the screen (see docs/lineages/route_b_env_pivot_v1.md) — this is what has already gone wrong.
---

# route_b preflight

## The rule: never validate on one arm

Every defect found in this lineage is **one failure class**. Code and assumptions get
written against whichever cell was in front of someone, then silently break on a
neighbouring arm. A list of known bugs only prevents the bugs you already know; this rule
prevents the next one.

**Before trusting any number, enumerate the arms it was computed over and check the
assumption on each — especially the arm that did not exist when the code was written.**

The corpus is **2 conn-probs × 2 replica-configs × 3 queue-dists × 17 seeds = 204**. The
arms that keep breaking things:

| arm | why it breaks code | how to spot it |
|---|---|---|
| `replica_overlap` (H2) | several task types share one platform set — uniqueness can now be exhausted, confinement goes to zero | `n_rows` = 24 = 4!, `confined_tasks` = 0 |
| `demand_scale != 1.0` (H1+) | every per-instance demand, cap and repair changes; anything defaulting to 1.0 is inert on H0 and wrong on H1 | `application.demand_scale` present in `workload.json` |
| the **64-row** arm | the only arm that strands the greedy on H0/H1 — a probe that never reaches it reads clean | `n_rows == 64` (vs 16) |
| `per_server = 1` | four tasks need four distinct platforms; overlap collapses it to two and the arm generates **zero** plans | 102 SKIPPED against 102 SUCCESS |

**The α axis counts as an arm too.** A claim measured at α=3.0 can be false at the rung's
registered primary α=2.0 — that is exactly what happened to the confinement story.

## The four steps

Run all four before reading any number. Steps 1–2 are cheap; do not skip them because the
change "looks inert".

**1. Full suite.** Use the full form — a stray local `.venv` hijacks `pipenv run` and
surfaces as a misleading `ModuleNotFoundError`.

```bash
PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=/root/projects/my-herosim \
  pipenv run python3 -m pytest tests/ -q
```

The route_b surface is `tests/test_route_b_env_pivot_{b1,w2,w3,w4,fixtures,storage_neutral,cap_mode_verifier,demand_scale_verifier}.py`,
`test_route_b_exact_denominator.py`, `test_route_b_positive_controls.py`,
`test_route_b_repair_fixtures.py`, `test_route_b_skip_reason_attribution.py`.

**2. Independent recomputation on the rung.** This is a separate implementation with its
own solver and no scorer imports. It must agree to **1e-9 on every (dataset, α) cell**,
never a sample — disagreement is an **S0 VOID**.

```bash
# the scorer report must carry per-dataset rows
pipenv run python3 scripts_cosim/score_route_b_contention.py --corpus <corpus> \
  --alphas 1.5,2.0,3.0 --cap-mode <alpha_max|alpha_mean> --include-per-dataset --out <report>.json
pipenv run python3 scripts_cosim/verify_route_b_scorer_agreement.py \
  --corpus <corpus> --report <report>.json --check-repairs
```

`--check-blocks` / `--check-krank` / `--check-decoder` are **separate passes** — one at a
time, and `--check-decoder` needs its own `--report`.

**3. Print the per-arm histogram of whatever you are about to summarize** — `n_feasible`,
`greedy_stuck`, `no_feasible_rows`, `confined_tasks`, `demand_scale` — and confirm the
denominator is not confounded with an arm.

Read `censoring_by_arm` from the report: it keys both censors and both surviving
denominators on `n_rows` (the **unconstrained** sweep size, the replica-config arm's
signature, which survives censoring). Read `n_exact_scored` / `n_greedy_scored` — they are
reported for exactly this reason. **Do not infer a denominator.**

A denominator concentrated in one arm makes the statistic "over that arm only", whatever
its name.

**4. Cross-check against `legacy_greedy_censored`**, which reproduces the pre-fix numbers
from the **same run**, so a deviation is audited against one artifact rather than a commit
message.

## What has already gone wrong

Five instances, one class. Each is real, each cost a session.

1. **`r_exact` tie band.** A decoder tie-break (`tuple(sorted(plan.items()))` — platform
   id, unrelated to cost) was read as physics. On the H0 separable control, 12 of 16 firing
   datasets had the true optimum inside the argmin tie set. The module docstring asserted
   the opposite of the truth. → Read `r_exact_band`; `mean_tied` is the fair reading,
   `optimistic` an upper bound only, never a verdict.
2. **`greedy_stuck` denominator.** The censoring variable was perfectly confounded with the
   replica-config arm (`{(9,False):102, (16,True):101, (16,False):1}`), so one entire cell
   was dropped from every `r_exact` statistic and the published number was really "over the
   9-feasible-row arm only". → Step 3.
3. **Verifier `demand_scale`.** The verifier and the scorer disagreed on a definition.
   **Inert on H0** (all scales 1.0), **live on H1**. Since 1e-9 agreement is an S0 VOID
   gate, H1 would have voided on a defect in the checking tool. Two call sites escaped a
   mechanical grep and were found by an **AST arity audit**. → Grep both spellings, then
   verify structurally.
4. **Skip-reason attribution.** `too_many_combinations` was asserted for any empty
   combination list, on the assumption that the zero-candidate case returned earlier.
   `replica_overlap` breaks it: uniqueness exhaustion also yields empty. 102 H2 datasets
   recorded `skip_threshold: 2000000` against a real product of 16.
5. **`no_feasible_rows` had no arm breakdown** — the stricter censor, one counter over from
   #2, and confounded too: at H1's registered primary α all 70 censored datasets sat in the
   64-row arm and none in the 16-row arm.

And the methodological instance, which no code review would have caught:

6. **`--max-datasets 24` read "clean".** 24 datasets never reach the 64-row arm, which is
   the only arm that strands. **A probe of this grid must cover both arms or it is
   misleading.**

## Registered semantics are not yours to move

- **Move no threshold, bar, grid, α ladder or reading rule without a signed-off amendment
  recorded in `LINEAGES.md` at its commit SHA.** Draft the amendment, state the
  measurement, get sign-off. Do not silently pick an option.
- **A bug fix that changes a registered statistic gets the bug-fix-vs-moving-the-bar
  treatment:** report both numbers (a `legacy_*` block **from the same run**), log the
  deviation, **move no threshold**.
- **A fix must be byte-identity-verified where it claims to be inert.** Diff every
  pre-existing key against the frozen artifact; do not assert inertness.
- **Do not read S1–S4 on a VOID rung**, and do not read a rung's bars from a half-corpus.
- **Ruled out by measurement — do not re-propose:** relaxing α (it is a cliff — clean
  counters and "the constraint binds" are mutually exclusive on this grid; α=4.0 is
  byte-identical to the unconstrained anchor) and raising `replica_server_percentage`
  (confinement is identical at 2, 3 and 4 hosting nodes). See `docs/hard-stops.md`.

## Where the record lives

`docs/lineages/route_b_env_pivot_v1.md` — the node. Its attachments:
`screen-preregistration.md` (registration), `screen-amendment-1.md` (S0 control
definition), `screen-amendment-2.md` (**draft**, the decoder), `ladder-findings.md`
(**§9 supersedes §3 and §4.1**). Gate-tool corrections go in `docs/gates/gate-tools.md`,
never in a lineage narrative. **Session handovers are ephemeral — scratchpad, never the
repo.**
