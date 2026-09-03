# reliability_matched_v1 — REGISTERED

> **Status:** `REGISTERED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md)
> &nbsp;·&nbsp; **Registered:** 2026-09-03, **before any gate episode was run**
>
> ⚠ **Drafted by Claude, not counter-signed by the user at registration time.** The user
> commissioned the experiment ("do it") in response to a recommendation; the *content*
> below deliberately makes no new statistical decisions — every threshold, statistic,
> α and reading rule is inherited **verbatim** from the already-signed
> [Phase 1 registration](objective_pivot_v1/phase1-registration-draft.md) (2026-08-28).
> The two places where a genuinely new choice had to be made are listed under
> §Judgment calls and flagged for review. Committed before data so the
> pre-registration property is objective (git timestamp), not a claim.

## The question

`objective_pivot_v1` Phase 1 established the GNN's reliability edge: across seeded
training draws, its severe-collapse burden is stochastically smaller than the MLP's
(rank-sum p = 0.00143 at +50%). That gate compared a GNN and an MLP **trained on
different corpora**.

On 2026-09-03 a pilot measured that the *latency* half of the same comparison was
entirely a corpus effect: an MLP trained on the GNN's own corpus ties it on both backbone
fabrics ([link_mp_v1](link_mp_v1.md), 2026-09-03 entry). The reliability claim is now the
program's last unconfounded edge — and it rests on the same unmatched pairing.

**The claim under test, worded now so it cannot drift:** *"Across seeded training draws
on the 20 backbone gate cells, with both model classes trained on the same corpus, the
MLP's per-draw severe-collapse burden (cells at `total_rtt` ≥ +50% vs same-cell Knative)
is stochastically greater than the GNN's."*

That is Phase 1's claim with one word changed — "same corpus" — and it is the whole
experiment.

## Design

| | |
|---|---|
| **GNN group (FROZEN)** | the 16 `lgon` arms of [link_mp_v1](link_mp_v1.md), trained on `graphs_cache_link_mp_v1_core_v1_dim14`, gated on these same 20 cells at pin `8aef27a`. Read from the existing summary, never re-selected. Recorded burden: **0 collapses in 16 × 20 cells at every threshold.** |
| **MLP group (NEW)** | 16 seeds of `experiments/link_mp_v1_mlp.yaml` — the deployed MLP recipe on **the same cache** — via `run_experiment.py --seed 1..16`. |
| **Cells** | the 20 BACKBONE cells `score_link_mp_v1.py` names: `drawgate/backbone` and `promo175/backbone` (configs from `a1_backbone_bw1p5`, workloads `150-100` and `175-100`), `bbrob/bb_core8_bw1p5` and `bbrob/bb_core4_bw0p5` (own configs, workload `150-100`). Full traces, not the 30k slice. |
| **Knative baseline** | re-run in the same job, same cells, same trace — the collapse rule is defined against the *same-cell* Knative and a baseline from another run is not that. |
| **Decode** | `argmax` for every arm. Nothing that produces a reported number samples. |

### Venue control — because the GNN group is frozen at another commit

The GNN arms were gated at pin `8aef27a`; the MLP arms will run at HEAD. Phase 1 made
code identity binding, so this gate carries its own control rather than assuming:
**4 GNN arms (`lgon` seeds 1, 5, 9, 13, chosen by fixed stride before any data) are
re-gated at HEAD on all 20 cells.** If any of the four shows a collapse at any threshold
where the frozen table records zero, **the frozen group is VOID and so is this gate** —
the comparison would be measuring the tree, not the corpus.

## Statistics — inherited verbatim from the signed Phase 1 registration

- **Primary**: one-sided Mann-Whitney rank-sum (midranks, fixed-seed 200,000-permutation)
  on the per-draw collapse-**count** vectors at **+50%**, MLP n=16 vs GNN n=16, α = 0.05,
  H1: MLP burden greater.
- **Sensitivity (must also clear for a PASS)**: the same statistic at **+100%**.
- **Descriptive only, no verdict role**: the **+30%** row, the clean/unclean dichotomy at
  all thresholds, per-draw worst-cell magnitudes.
- **Permutation seed** `20260903`, fixed here so the p-value is reproducible bit-for-bit.

## Outcomes, all four written before the data

1. **PASS** (primary and sensitivity both clear at α = 0.05): the reliability claim
   survives corpus matching. It becomes the program's one measured, unconfounded
   model-class result, and the paper's reliability section stands with "same corpus"
   added to its wording.
2. **TIE-AT-ZERO** (both groups all-zero at +50%): the rank-sum is undefined-by-tie /
   p = 1.0. **This is a real answer, not a failure**: reliability, like latency, was the
   corpus — a corpus whose backbone binds produces collapse-free schedulers of *both*
   classes. Pre-stated limitation: at n = 16 a floor effect cannot be distinguished from
   "both classes are genuinely robust"; the honest sentence is "no reliability difference
   is detectable once the corpus is matched, because neither class collapses here."
3. **FAIL** (MLP burden not stochastically greater, with non-zero counts present): the
   reliability claim does not survive corpus matching and must be withdrawn from the
   paper as a model-class claim.
4. **VOID**: the venue control fires; or any arm is missing cells; or a checkpoint lacks
   its contract declaration; or the scorer deviates from this file.

## Judgment calls — the two places this is not purely inherited

1. **The cell set is link_mp_v1's 20, not Phase 1's 30.** Reason: the 30 Phase 1 cells
   include flat-fabric blocks, and the GNN group's frozen burden table exists only for
   these 20. Using the cells the frozen group was actually measured on is the
   conservative choice; using Phase 1's would require re-gating the GNN group entirely.
2. **The GNN group is frozen rather than re-trained.** Reason: 16 fresh GNN trainings on
   this corpus are GPU-hours that would reproduce an existing, pinned, zero-collapse
   table. The venue control above is the price paid for that reuse, and it can VOID the
   gate.

Both are recorded here so that neither can be presented later as having been obvious.

## Named selection hazard

This gate exists **because** the latency pilot came out as a tie. That is a hypothesis
formed from data. What bounds it: the claim wording, thresholds, statistic, α, cell set
and all four outcomes are fixed in this file before any MLP arm is trained, and the GNN
comparison group is frozen material that no choice here can move.

## Entry points

- Registration: this file.
- MLP arms: `experiments/link_mp_v1_mlp.yaml`, seeds 1–16.
- Gate: `scripts_cosim/datalab/reliability_matched_v1_gate.sbatch` (array, one task per
  block).
- Scorer: `scripts_cosim/important/score_reliability_matched_v1.py` — a sibling of
  `score_objective_pivot_phase1.py`, not an edit; its thresholds are this registration.

## Record

*(Outcome to be written here after the gate runs. Nothing has been read yet.)*
