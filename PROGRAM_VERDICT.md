# Program Verdict — plain-language summary

*Written 2026-08-24. This is the executive summary of the `program_verdict_v1`
investigation. The authoritative technical record — every claim with its citation,
job ID, and pre-registered threshold — is the `program_verdict_v1` entry in
[LINEAGES.md](LINEAGES.md). If this file and that entry ever disagree, the entry wins.*

## The original idea didn't work, and we now know for certain

The goal was for the GNN to be smarter at placing tasks than the simple pointwise
model (the MLP). It can't be — not because anything was done wrong, but because the
practice problems it trains on don't have the answer in them: the co-simulation
target is pointwise-separable, so a simple model is the correctly specified one.

This has now been checked every way there is to check:

- five structurally different physics mechanisms, all ending count-shaped or negligible;
- the live states the schedulers actually visit, including every recorded moment of
  MLP collapse (4,400 swept states, median additive-choice regret 0.000);
- the least-additive third of the training data (the warmth stratum), which goes to
  additive R² = 1.00000 exactly once collisions are removed;
- and, as of 2026-08-24, out-of-sample checks (held-out R² = 1.0 to machine precision)
  that an overfit could not fake.

**Stop looking.** Anyone reopening this needs a measurement, and the LINEAGES entry
says exactly which one would count.

## We found something better instead

**The GNN never breaks.** Across 120 scheduler runs on the backbone gates, the MLP
collapsed 14 times (queues run away, the cluster idles, latency goes 2–8× worse);
the GNN collapsed zero times. And the GNN beat Knative — the industry baseline — on
all 30 cells of every condition tested.

Precision matters here: the GNN is not "reliable but slow." It is **reliable and
better than the standard**; it is just not the fastest on a good day — the MLP packs
harder and wins the typical cell, until packing is exactly what kills it.

That is a real, publishable result: *the GNN is the only scheduler that beats Knative
everywhere, and the only learned scheduler that never falls over.*

## But right now that result is a lucky spot, not a proper test

The reliability win was noticed in data collected for something else (the link-physics
A/B), and "winning" was defined after seeing the numbers. That is fine for finding
things, not for publishing them. The next work is redoing it properly: rules decided
first, then run.

## A labelling mistake, caught before the paper

The traces' `rps` field is **per client node**. `workload-150-100` does not run at
150 events/s — it runs at ~3,000 events/s steady state (20 clients × 150). Every
measurement is still correct; the sentences *about* them ("this is realistic load at
150 rps") were off by 20×. Paper text quoting an operating point must state the
per-client convention and the system rate.

## What happens next, in order

1. **Try to break our own result (1–2 days).** Give the MLP one extra piece of
   information — a candidate-relative queue feature — retrain, re-gate. If it stops
   collapsing, the finding is smaller than we thought (feature engineering, not
   architecture), and we want to know that before anything else is built on it.
2. **Redo the reliability test properly (2–3 days, CPU).** Win condition, collapse
   detector, and thresholds registered in advance; MLP arm included, not just Knative;
   fresh test cells minted outside the old A/B design.
3. **One last big experiment (a few days).** Add realistic in-horizon traffic to the
   labelling oracle and test whether it creates structure a simple model can't express.
   It got ~6× more expensive than first estimated (the calibration slice sat in the
   trace's ramp-up), so the cost check is a blocking step: calibrate on 3 snapshots
   first, confirm combos run in-process, and register the horizon length before
   queueing anything.
4. **A half-day on the one untested physics idea.** Residency-length node contention
   (cold starts hold ~1,500× longer than exec). Expected outcome: a sixth confirmation
   of the count-shaped rule — but it converts the weakest ruling in the set into a
   measured one, for two orders of magnitude less than the closed-loop alternative.
   Both controls pre-registered, including the one that could actually surprise.

Only if step 3 (or step 4) finds non-count signal does the expensive path — a
closed-loop training objective — get considered.
