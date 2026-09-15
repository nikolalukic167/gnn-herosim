# Closed-loop policy gradient does not improve the served scheduler

> **Status:** `CLOSED — MEASURED-NEGATIVE` · **Lineage:** [objective_pivot_v1](../objective_pivot_v1.md) · **Dates:** 2026-09-01 → 2026-09-03
>
> Paper-grade synthesis of `objective_pivot_v1` Phase 3. The dated operational record —
> every increment, job ID and intermediate table — lives in the lineage node; this
> document is the argument, written once, for citation.

## Claim

Training the served GNN scheduler by policy gradient against the live simulator's own
total-RTT metric **does not improve it**. Measured on a held-out network fabric across
120 independent training runs: paired median **−0.85%**, mean −0.85%, **53/120** runs
better than the frozen supervised checkpoint, one-sided Wilcoxon **p = 0.928**.

The result is **adequately powered**, which is what makes it an answer rather than an
absence of one. At the observed across-run standard deviation of 4.89 pp, the registered
3% minimum detectable effect requires n ≥ 84; 120 were run. A 3% improvement would have
been visible. The point estimate is negative.

**This closes the last of the three routes** named in `CLAUDE.md`: better labels (closed
by `program_verdict_v1`), a richer environment (`route_b_env_pivot_v1`, PARKED), and
training against the live objective (here).

## What was tested, precisely

The supervised path was closed by `program_verdict_v1` on the grounds that the co-sim
target is pointwise-separable — the MLP is the correctly specified model class for it, so
no volume of labelled data changes the outcome. Phase 3's premise was that the *objective*,
not the data, was the constraint: replace the label with the live metric and the ceiling
should move.

**The estimator.** REINFORCE with a self-critical baseline. For each cell the current
policy plays one greedy (argmax) episode and *S* temperature-sampled ones; the advantage is
`A = (RTT_greedy − RTT_sampled) / RTT_greedy`, so `A > 0` means sampling beat the policy's
own greedy behaviour. Argmax is deterministic given (weights, cell, trace), so one baseline
episode serves all *S* samples — the loop costs `1+S` episodes per cell per step, not `2S`.

Advantages are deliberately **not** mean-centred. The usual within-batch centring assumes
an arbitrary baseline; here the baseline is the policy's own greedy episode, so `A = 0`
already carries meaning. Centring would make a step in which *every* episode beat greedy
teach nothing.

**Two passes, because the graph does not fit.** An episode is ~30,000 placement decisions
across ~25,000 forward passes. Pass 1 runs the episode as a subprocess of the ordinary
serving path under `no_grad`, sampling actions and reservoir-sampling *k* decode batches
(inputs plus chosen indices) to disk. Pass 2 replays those *k* with gradients as
`(N/k) · Σ log π(a|s)`. Algorithm R gives every batch inclusion probability exactly `k/N`,
which is what makes the rescale unbiased — measured over 3,000 trials rather than asserted.

**Both model classes ran the identical loop.** `MLPBatchScheduler → XGBoostBatchScheduler
→ GNNScheduler`, so the sampled decode, the trajectory recorder and the replay hook were
inherited rather than reimplemented; only the replay *payload* differs (a PyG `Data` for
the GNN, the serving matrix and its row spans for the MLP). Each arm was tuned on the same
grid, the same budget and the same cells, and each ran the pilot at its own dev-selected
optimum — the registration asks for the same loop, objective, budget and seeds, not the
same learning rate.

**Correspondence, checked continuously.** Pass 2 must reproduce pass 1's log-probs on the
same weights, or the gradient points at a distribution the simulator never sampled from and
nothing downstream notices. Measured on the real checkpoint: maximum per-decision error
**1.1e-16** — float64 machine epsilon. The trainer re-checks this every step and aborts on
drift.

## The cell split

| | cells | role |
|---|---|---|
| train | `bbrob_bb_core4_bw0p5` cell01, cell02, cell04 | the loop's episodes |
| dev | `bbrob_bb_core4_bw0p5` cell03, cell05 | hyperparameter selection only; never reported as a gate |
| gate | `bbrob_bb_core8_bw1p5`, all 5 cells | the verdict — unseen fabric *and* unseen cells |

The gate fabric differs from the training fabric in both core count and link bandwidth.
Nothing in it was touched by training or by selection.

## Result

### Primary — CL-GNN vs Frozen-GNN, 120 fresh training seeds

| | |
|---|---|
| paired mean | **−0.849%** |
| paired median | **−0.846%** |
| seeds better than frozen | **53 / 120** |
| one-sided Wilcoxon (200k seeded sign-flip) | **p = 0.9283** |
| across-run sd | **4.89 pp** |
| n required for the registered 3% MDE | **84** |

**MEASURED-NEGATIVE. The registered kill criterion fires. P1 freezes.**

### The seed lottery, which is the methodological centre of this result

An earlier gate at n = 16 returned median **+0.27%** (p = 0.372). The 120 fresh seeds
return **−0.85%**. The two samples come from the same configuration, the same code and the
same gate; the difference is entirely the draw.

That +0.27% was never reported as a result — it was recorded as NOT ESTABLISHED — but the
episode is the reason the extension excluded those 16 seeds from the primary analysis. The
decision to run more seeds *followed a look at them*; pooling them back in would have
inflated Type I error by exactly what that look was worth, and in the event would have
dragged a negative result toward zero on the strength of a draw. The pooled n = 136 figure
(median −0.425%) is reported as secondary and cannot change the verdict.

### Secondary, pre-registered — the trainability asymmetry

| arm | across-run sd | n |
|---|---|---|
| CL-GNN | **4.89 pp** | 120 |
| CL-MLP | **0.0325 pp** | 16 |

A ratio of **150×**. The same loop, objective, budget, grid and seeds moves the graph model
by percent-scale amounts and the pointwise model by hundredths of a percent. Every MLP
configuration that actually moved its weights made it *worse*, monotonically in the learning
rate; the selected rung was the one that barely trains it.

**This is a claim about trainability, not latency.** "Can be moved" is not "is improved",
and the primary settles that it is not improved — at the selected configuration the GNN
moves the wrong way about as often as the right way. The asymmetry was registered as a
secondary claim *before* the run precisely so it could not become a consolation headline
afterwards.

### What is unaffected

On the same held-out fabric, the frozen supervised checkpoints stand where they always did:

| arm | mean total RTT | vs Knative |
|---|---|---|
| Frozen-GNN (`gnn-linkmp-lgon-s8`) | 4,945,398.72 | **−44.8%** |
| Frozen-MLP (`fc_siv1_dim22_tempfix`) | 6,340,642.33 | −29.2% |
| Knative | 8,953,094.24 | — |

The GNN's supervised latency edge over the reactive baseline is large, transfers to an
unseen fabric, and is untouched by this result. So is the Phase 1 reliability claim. What
is now falsified is the attempt to *widen* them by changing the objective.

### Serving-path reproducibility

The three deterministic arms reproduce the earlier gate exactly — `frozen_gnn`
4,945,398.718675, `frozen_mlp` 6,340,642.331262, `knative` 8,953,094.238518, all
Δ = 0.000000000. Nothing in the serving path moved between the two gates, so the comparison
across them is sound. This check cost ten episodes and is the only thing that could have
distinguished a real change from a drifting instrument.

## Three defects in our own instruments

All three were found by this phase, all are mine, and all are recorded because each would
have produced a plausible-looking artefact rather than an error.

**1. The verdict script applied the kill criterion to a secondary arm.** `analyze_gate.py`
printed "P1 freezes as measured-negative" for the CL-MLP comparison because it applied the
registered logic to whatever arm it was pointed at. The registration defines the kill on the
paired CL-GNN difference and nothing else. Fixed: the kill now requires an explicit
`--primary`. Left alone, a script's default would have entered the permanent record as a
program-closing verdict nobody signed.

**2. The tuning stage was selecting on noise.** Amendment D3 specified **one seed per
configuration**. At the across-run sd later measured (4.9–5.8 pp), that cannot distinguish a
configuration from a draw. `lr = 1e-4` won its grid on a single +10.6% run that turned out to
sit near the top of a distribution centred on zero, and the +8.5% it showed on held-out dev
cells was the same draw evaluated twice, not independent confirmation. Every downstream run
then spent its budget at a learning rate chosen by noise. **A dev-cell evaluation does not
rescue this** — it controls for memorisation, not for the training draw, and the two failure
modes are indistinguishable in a single run.

It was not repaired mid-flight. Re-selecting a learning rate after seeing a gate is
selection with knowledge of the outcome, which is what the kill criterion exists to forbid;
the under-powered selection is carried as a stated limitation of this result instead. A
properly powered re-tune would be a new registration.

**3. The registration specified an n its own instrument could not execute.** Amendment E
registered n = 120 with an exact Wilcoxon, against an analyzer that raised above n = 22
because 2ⁿ enumeration is infeasible. The guard was correct and fired correctly — **at the
analysis step, after all 615 gate episodes had been computed.** The artefacts survived, so
only the analysis was re-run, but the general failure is that a registration fixes a number,
a test and an alpha, and nobody runs the analysis path on synthetic data at the registered n
first. Fixed by sampling the same null rather than enumerating it: a seeded 200,000-draw
sign-flip permutation, cross-checked against the tie-corrected normal approximation, which
fails the run if the two disagree by more than 0.01 — so no one picks whichever reads
better. Verified against scipy at n = 30/60/120 and against the exact path at the n = 22
boundary, and **written and tested before any per-seed number of the gate had been looked
at**.

## Scope and what would reopen this

The result is about **this configuration**: REINFORCE with a self-critical baseline, at
`lr = 1e-4, T = 0.1`, 20 steps, warm-started from `gnn-linkmp-lgon-s8`, on this corpus and
this trace. It is not a claim that no closed-loop method could ever help.

What it *does* establish is that the obvious form of the idea, given a fair tuning budget
and adequate power, does not — and that the instability at the selected configuration is
large enough (sd 4.89 pp) that a 3% effect would have to fight through it. Reviving this
needs a different configuration with its own **powered** tuning stage, registered before its
data exists. Another n at `lr = 1e-4` would not be a new experiment.

## Cost

~1,000 CPU-hours: one shakedown, twelve tuning runs across two model classes, thirty-two
pilot runs, one hundred and twenty powered runs, and two gates totalling 790 greedy
episodes. Against the `program_verdict_v1` anchor of ~500 CPU-h pilot / ~5K CPU-h gate
scale: over the pilot line, inside the gate allowance, and named rather than absorbed.

## Artefacts

| | |
|---|---|
| trainer | `scripts_cosim/closed_loop/train_closed_loop.py` |
| episode runner / adapters | `scripts_cosim/closed_loop/{episode,adapters}.py` |
| evaluation and verdict | `scripts_cosim/closed_loop/{evaluate_policy,analyze_gate}.py` |
| guards | `scripts_cosim/test_closed_loop_{gradient,episode}.py` |
| cluster jobs | 733169 (probe) · 733519 (shakedown) · 733566/733631 (tuning) · 733648 (pilot) · 734064 (gate n=16) · 734154 (n=120) · 734369 (gate n=120) |
| registration | Phase 3 + Amendments A–E, `docs/lineages/objective_pivot_v1.md` |
