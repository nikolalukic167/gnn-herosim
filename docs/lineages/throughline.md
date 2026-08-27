# throughline — SYNTHESIS

> **Status:** `SYNTHESIS` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-18 → 2026-08-18

**Outcome.** Cross-lineage synthesis: four mechanisms, one collapse. **In this simulator, coupling is either count-shaped or negligible** — and the negligible half is demonstrated, not assumed.

**Related:** [graph_structure_physics](graph_structure_physics.md) · [network_contention_v1](network_contention_v1.md) · [link_contention_v1](link_contention_v1.md) · [shallow_longexec_v1](shallow_longexec_v1.md)

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [The throughline — four mechanisms, one collapse (2026-08-18)](#the-throughline-four-mechanisms-one-collapse-2026-08-18)

---

### The throughline — four mechanisms, one collapse (2026-08-18)

**`shallow_longexec_v1` is UNBLOCKED and is the fourth confirmation.** The config gap its
note described is closed — `sample_loader.ensure_workload_params` synthesizes the missing
`workload_nofs-{cnn,rf}`, and the grid now generates cleanly (cnn 3.086 s on rpiCpu vs rf
0.004 s, a 730× exec-time contrast). Gated locally at n=16 per arm:

| arm | additive R² | argmin regret mean/max | argmin optimal | **one-integer repair** | +col optimal |
|---|---|---|---|---|---|
| `shallow_longexec_v1` | 0.9330 | 3.05% / 37.0% | 62% | **100%** (n=6) | 81% |
| ...+ 0.5 MB/s ingress | 0.9290 | 0.99% / 4.8% | 56% | **86%** (n=7) | 81% |

Every dataset where the pointwise fit picked a suboptimal plan was repaired by one scalar,
and optimal-recovery rises 62%→81% / 56%→81% from that single column. *(Honest wrinkle: the
augmented fit's mean regret is higher (5.24%/4.57%) because the extra column shifts the
argmin on some datasets that were already optimal — the degeneracy claim rests on the
repair of actual failures and the optimal-recovery jump, not on mean regret.)*

**So five structurally different attempts to inject coupling all end the same way:**

| # | mechanism | how it failed |
|---|---|---|
| 0 | `added_in_batch` (base physics) | +1.10 pp — "a column you hand an MLP" |
| 1 | execution-slot pool (`node_contention_v3`) | no interaction at all — `nodeContentionTime` ≡ 0.0 |
| 2 | deep queues (`contention_v4_v5`) | interaction diluted by depth; R² moved the wrong way |
| 3 | node-ingress bandwidth (`network_contention_v1`) | R² scales continuously, but at its only high-regret setting one integer repairs 75% |
| 4 | long-exec task types (`shallow_longexec_v1`) | one integer repairs 86-100% |
| 5 | per-link capacity (`link_contention_v1`) | **first mechanism to escape the one-integer control** — isolated on spread plans the node column repairs 0% and link scalars only 11-22% — but the effect is 0.08-0.10% regret against a 5% gate, and bandwidth is a null lever (wait/transfer 0.0100 → 0.0088 across a 3× cost change) |

**Whenever coupling in this simulator shows up with teeth, it is capturable by one
count-like feature.** That now looks like a property of this class of scheduling problem
rather than of any single experiment — and it is the reusable finding, together with the
diagnostic that detects it before any GPU-hours are spent.

**`link_contention_v1` sharpens that statement rather than breaking it.** It is the first
mechanism to produce coupling a scalar count cannot capture — the escape was real, and the
reason it worked is now understood: its contended object (a link) has more identities than
the destination-node count, so two tasks that share *no* node can still contend. But the
resulting effect has no teeth (0.08-0.10% regret), so the observed rule survives in its
stronger form:

> **In this simulator, coupling is either count-shaped or negligible.** Five mechanisms, two
> failure modes, and the second one is now demonstrated rather than assumed — the isolation
> control shows base physics is additive to R² = 1.00000 exactly once collisions are removed,
> so there is no reservoir of non-count coupling waiting to be uncovered by a better lever.

The practical corollary for anyone extending this: **check the additive/interaction scaling of
a proposed lever before building it.** Deep queues failed because the additive term grew with
the lever and the interaction term did not; link bandwidth failed because both grow together
and the ratio is invariant. A lever only helps if it moves interaction *faster* than additive
— that is a two-line calculation, and it would have predicted both outcomes.

⚠ **The scaling test rules levers out; it does not rank the survivors.** Tested against the
hub↔mesh sweep above, `wait / transfer` got the ordering backwards — the configuration with
the lowest ratio had the highest regret, because the ratio is measured on optimal plans, which
select *against* contention, while regret is a property of the spread across the whole sweep.
A bandwidth-invariant ratio is decisive evidence to stop; a favourable ratio is not evidence
to proceed.

**The corrected gate is wired into the tool, not just written down here.**
`separability_diagnostic.py` gained `--gate-additive-argmin-regret` (primary) which
**argparse-errors unless `--gate-one-integer-repair` is supplied**, plus a first-class
`one_integer_repair_frac` in the M4 block and a `!! DEGENERATE` banner at ≥0.5. Verified:
both failure modes fire and the tool exits 1. `--gate-coupled-fraction`'s help text now
carries the deprecation and the reason.

**2026-08-18 additions from `link_contention_v1`,** both covered by
`scripts_cosim/test_link_repair_control.py` (11 tests) so the gate is not itself untested —
the `--gate-coupled-fraction` episode was a gate nobody had exercised:

- **`--gate-link-repair`** — repair columns for the busiest link load (`k1`), the top-2 loads
  (`k2`), and total link-sharing excess (`excess`). Needed because the existing node-collision
  control is *structurally blind* to link contention and would have returned a false PASS on
  any per-link mechanism.
- **`--spread-plans-only`** — the isolation control described above. Reusable for any future
  mechanism: it answers "is there coupling here that is not the collision term?", which the
  headline gate cannot, because the collision term dominates every corpus in this repo.
