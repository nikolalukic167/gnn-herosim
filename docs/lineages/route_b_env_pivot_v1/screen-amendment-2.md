# route_b_env_pivot_v1 — AMENDMENT 2 (DRAFT): the decoder behind `greedy_stuck`

> **DRAFT — NOT SIGNED OFF. NOTHING IN THIS DOCUMENT EXECUTES.**
> Written 2026-08-27 as the proposal `ladder-findings.md` §6 asks for. It is recorded here
> so the recommendation and its measurement live in the research record rather than in a
> chat message. It becomes live only when the user signs it off **and** a LINEAGES entry
> records it at this document's commit SHA. Until then the registered semantics are
> exactly those of `screen-preregistration.md` @ `019bdcb` as amended by
> `screen-amendment-1.md` @ `3719aad`.

**Proposed scope: the DECODER that produces `r_greedy` and `greedy_stuck`, only.** No bar
value changes. No rung, α ladder, seed block, grid, corpus or reading rule changes. S0's
counter-cleanliness condition is quoted verbatim and left as written. S1, S2, S3, S4 are
untouched.

## 1. The measurement

`greedy_masked_plan` (`scripts_cosim/score_route_b_contention.py:441-478`) is a single
non-backtracking forward pass. Over the **identical** candidate sets, demands, caps and
option ordering the production decoder saw, changing only that a dead end backtracks
instead of returning `None`:

| rung | α | stuck | feasible plan exists in the sweep | **backtracking rescues** | stuck by arm |
|---|---|---|---|---|---|
| H0 | 2.0 | 95 | 95/95 | **95/95 (100%)** | `{64: 95}` |
| H0 | 3.0 | 87 | 87/87 | **87/87 (100%)** | `{64: 87}` |
| H1 | 2.0 | 100 | 100/100 | **100/100 (100%)** | `{16: 71, 64: 29}` |
| H1 | 3.0 | 83 | 83/83 | **83/83 (100%)** | `{64: 83}` |
| H2 | 2.0 | 66 | 66/66 | **66/66 (100%)** | half-corpus (§4.1) |
| H2 | 2.4 | 27 | 27/27 | **27/27 (100%)** | half-corpus (§4.1) |

**458/458.** Every dataset the greedy stranded had a feasible plan sitting in its own
enumerated sweep. `greedy_stuck` is therefore a property of the decoder, not of the
environment, on **every** rung — H0 and H1 included, where `ladder-findings.md` §3
attributed it to configuration. §9.2 shows why that attribution looked right: confinement
correlates with stuck at α=3.0 and breaks at H1's registered primary α=2.0, where 71 of
102 **zero-confinement** datasets are stuck.

The two counters are not independent. A complete backtracking search over the masked
option sets succeeds exactly when a feasible sweep row exists, so under this amendment
`greedy_stuck` becomes **logically implied** by `no_feasible_rows`.

## 2. What is proposed

Replace the decoder's dead-end behaviour with a complete search over the same masked
space. **Same** option ordering (ascending `(marginal, placement)`), **same** task order
(the fixed topological order), **same** replica-reuse mask, **same** capacity test, **same**
tie-breaks. The first complete assignment in that deterministic order is the plan — so on
every dataset the greedy already completed, the returned plan is unchanged by construction.

`greedy_stuck` is retained as a counter and is expected to read 0 wherever
`no_feasible_rows` is 0. It is not deleted: a nonzero value under this decoder would mean
the mask and the sweep disagree, which is a fail-loud condition worth keeping.

## 3. What it does NOT propose

- **No threshold, bar, grid, α ladder, seed block or reading rule moves.** In particular
  §3's registered fallback keeps its exact wording — "`no_feasible_rows > 0` **or**
  `greedy_stuck > 0`". The clause simply stops firing on the second disjunct, because the
  condition it tests stops being true, not because it was rewritten.
- **No corpus is regenerated.** This is a scorer change; every corpus is untouched.
- **No claim about the pivot hypothesis.** S1–S4 remain unread on every rung.

## 4. Consequence, stated in advance so it cannot be read as a post-hoc gain

Under the fallback **exactly as registered**, with `greedy_stuck` no longer firing:

| rung | α read at | `no_feasible_rows` | binds on | outcome |
|---|---|---|---|---|
| H0 | **2.0** (its registered primary) | 0 | 204/204 | becomes READABLE |
| H1 | **3.0** (tightest clean α on its registered ladder) | 0 | 204/204 | becomes READABLE |
| H2 | — | — | — | still VOID-GENERATION (102/204; unaffected by this amendment) |

H1 does **not** become readable at its primary α=2.0: `no_feasible_rows = 70` there, all
70 in the 64-row arm (§9.3), so the registered fallback moves it to α=3.0 on its own terms.

This is a real gain in readability, and it must be signed off *knowing* that — an amendment
that unblocks two rungs is exactly the kind that needs a decision rather than a patch note.
The argument for it is not that it unblocks them; it is that the counter blocking them was
measuring the tool.

## 5. Reporting obligations if signed off

Same treatment the scorer fixes received (GATE TOOLS, 2026-08-27) — the change alters a
**registered statistic** (`r_greedy`, `greedy_stuck`), so:

- **Report both numbers.** Every rung's artifact carries a `legacy_forward_only` block
  reproducing the pre-amendment `r_greedy` / `greedy_stuck` **from the same run**, exactly
  as `legacy_greedy_censored` does, so a deviation is audited against one artifact rather
  than a commit message.
- **Log the deviation** per rung per α in the lineage node: old `greedy_stuck`, new
  `greedy_stuck`, old `n_greedy_scored`, new `n_greedy_scored`, and the `r_greedy` band on
  both denominators.
- **Move no threshold.** S1–S4 read on their registered values, unchanged.
- **Byte-identity gate:** on every dataset the forward-only decoder already completed, the
  new decoder must return the **same plan** and the same `r_greedy_pct` — verified across
  all three rungs before any bar is read. The change may only ever *add* completions.
- **Independent verifier:** `verify_route_b_scorer_agreement.py --check-decoder` must be
  extended to the backtracking decoder in its own recomputation (no scorer import), and
  must agree to 1e-9 on every (dataset, α) cell. Disagreement is an S0 VOID.

## 6. The alternative, and why it is not recommended

`ladder-findings.md` §6 option 2 — amend the fallback to distinguish decoder-stuck from
environment-infeasible — reaches the same two rungs by relabelling. It is rejected here
because it changes a **reading rule** (the thing the registration is most protective of)
in order to avoid fixing a tool that is measurably wrong, and it leaves `r_greedy` — a
registered statistic that S-bars do not read but the record does — produced by a decoder
that fails to find plans that provably exist in 458 of 458 cases. Option 1 changes a tool;
option 2 changes the rules the tool is judged by. On this evidence the tool is what is
broken.

Options 3, 4 and 5 remain open and are unaffected: they address H2's generation shortfall
and H0/H1's allocator, which this amendment does not touch. H2 stays VOID-GENERATION
either way.

## 7. Sign-off

Amends `screen-preregistration.md` @ `019bdcb`, as already amended by
`screen-amendment-1.md` @ `3719aad`. Requires: user sign-off, a LINEAGES registration
entry at this document's commit SHA, and §5's byte-identity gate plus the extended
independent verifier green — **before** any rung is re-read.

**Status: DRAFT. Not signed off. No rung has been re-read.**
