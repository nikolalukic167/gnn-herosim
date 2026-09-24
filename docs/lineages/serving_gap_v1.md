# serving_gap_v1 — CLOSED (NO-GO)

> **Status:** `CLOSED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-09-12 → 2026-09-12

**Outcome.** **NO-GO on the registered kill bar, same day.** S0 passed; **neither H1 nor H2 fired, and
both were refuted in the direction OPPOSITE to the one registered**, significantly. The graph-aware arm is
not herded and not load-blind: on every corpus it spreads its choices *more* evenly than its pointwise twin
and reacts to queue depth *3-7x more strongly*. No serving constraint follows, so stage 2 is not
authorised and the lineage closes. The refutations are themselves a measured fact and are recorded below,
because they invert the intuition the whole search rested on.

**The question.** `peer_affinity_v1` measured, on three corpora and 16 paired seeds each, that
`gnn` − `mpoff` reads **+5.14 / +6.53 / +9.61 pp offline** and **+2.37 / −4.42 / −10.67 % live**,
monotone in peer-graph density and perfectly anti-correlated between venues. The live peer term
inverts with it: `mpoff` achieves *lower* `totalPeerExchangeTime` than `gnn` on the live stream in
every condition measured, while achieving *higher* peer cost than `gnn` on the offline batches from
the same generator. **Why does message passing get worse at its own objective when the same
checkpoint is served on a stream?** This lineage asks only that, and only with diagnostics on
existing checkpoints plus serving-side constraints. It proposes no new physics, no new corpus and
no training.

**What this lineage must NOT become.** Three neighbours are closed and the obvious framings of this
question walk straight into them:

- **"Label the batch against a loaded system instead of a captured snapshot."** CLOSED by
  [cosim_deepdive_v1](cosim_deepdive_v1.md): 4,400 **live-visited** states, including all 14 MLP
  collapse trajectories, are equally additive (median additive R² 0.99999, median additive-choice
  regret 0.000). A single-batch target labelled on a loaded state is still pointwise-expressible, so
  relabelling cannot carry the streaming behaviour. *(This was proposed in session on 2026-09-12 and
  is recorded here as refused, not as an option.)*
- **"Train the served policy closed-loop against the live simulator."** CLOSED by
  `objective_pivot_v1` Phase 3: n = 120, paired mean −0.849 %, one-sided Wilcoxon p = 0.928,
  adequately powered (3 % MDE needs n ≥ 84). Reviving it needs a different configuration with its
  own powered tuning stage, registered in advance. **This lineage trains nothing**, so it does not
  touch that stop.
- **"Cite offline regret as evidence about live placement."** Already a standing stop
  (`docs/hard-stops.md`). This lineage's premise is that stop's mechanism, and every statistic below
  is measured on the live stream, never inferred from an offline read.

**Related:** [peer_affinity_v1](peer_affinity_v1.md) · [cosim_deepdive_v1](cosim_deepdive_v1.md) ·
[objective_pivot_v1](objective_pivot_v1.md) · `docs/gates/gate-tools.md` (2026-09-11 rows) ·
`docs/lessons.md` (the spreading-axis rule, 2026-09-11)

**Entry points (to be built under this registration):**
`scripts_cosim/serving_gap_trace_read.py` — reads the decode traces
(`GNN_PREFIX_TRACE_PATH`, which already pickles `{sim_time, task_ids, combo, graph, diag}` per
batch) and emits the H1/H2 statistics below · the live gate is the existing
`scripts_cosim/datalab/peer_affinity_v1_stage3_live_gate.sbatch` with the serving knobs already in
`src/policy/gnn/prefix_serving.py`.

## Record

- [Stage 1 — NO-GO, both hypotheses refuted backwards (2026-09-12)](#stage-1-no-go-both-hypotheses-refuted-backwards-2026-09-12)
- [Registration (2026-09-12)](#registration-2026-09-12)

---

### Stage 1 — NO-GO, both hypotheses refuted backwards (2026-09-12)

**S0 (control) PASSED.** The harness recomputed the registered offline `gnn` − `mpoff` contrast from the
stored per-checkpoint reports, restricted to each rung's frozen held-out block: **+5.142 / +6.532 /
+9.606 pp** against the registered **+5.14 / +6.53 / +9.61**, signs matching on all three corpora
(`simulation_data/serving_gap_v1_s0.json`). The diagnostics below are therefore reading the artifacts the
registered read was computed from.

**Substrate.** 96 live production runs (3 corpora x 2 arms x 16 seeds, job 759571) writing slim decode
traces for H1, and 96 more (job 759842) writing graph-carrying traces subsampled to ~500 decision points
for H2 (job 760166 scores them). **Uncapped**, stated in the sbatch: the platform cap is itself an
anti-concentration constraint and would mask the pathology under test. Two default-off serving flags were
added for this and change nothing otherwise: `GNN_PREFIX_TRACE_SLIM`, `GNN_PREFIX_TRACE_EVERY`.

**H1 — herding. DOES NOT FIRE; refuted backwards.** Registered direction: `gnn` more herded (higher modal
repeat rate, lower node entropy). Measured, paired over 16 seeds:

| corpus | modal repeat rate, gnn − mpoff | p | node entropy, gnn − mpoff | p |
|---|---|---|---|---|
| x200 p2 | +0.001 | 0.50 | −0.017 | 0.50 |
| x800 p2 | −0.067 | 0.19 | +0.016 | 0.90 |
| x800 p3 | **−0.083** | **0.009** | **+0.062** | **0.018** |

Zero of three fire. On the corpus where the live gap is worst the *pointwise twin* is the herded arm, and
significantly so. Median normalised node entropy over 45,375 batches, x800 p3: `gnn` 0.835, `mpoff` 0.786.

**H2 — queue insensitivity. DOES NOT FIRE; refuted backwards, and by a wide margin.** Registered direction:
`gnn` less sensitive. Measured (mean |Δ score| per candidate, normalised by the step's score spread, when
every platform is told it is one standard deviation busier):

| corpus | `gnn` | `mpoff` | difference | p |
|---|---|---|---|---|
| x200 p2 | 0.582 | 0.169 | **+0.252** | 6.1e-05 |
| x800 p2 | 0.424 | 0.113 | **+0.241** | 9.2e-05 |
| x800 p3 | 0.770 | 0.111 | **+0.637** | 3.1e-05 |

The graph-aware arm is **3 to 7 times more** responsive to load than its twin, on every corpus, at
p < 1e-4. It is the opposite of blind.

**What the two refutations say together.** Every registered explanation assumed the graph arm is too rigid
— stuck on a favourite node, deaf to load. The measurements say it is the *most* responsive arm in the
program: it spreads more, and its scores move several times further when load changes. Put beside
`peer_affinity_v1`'s finding that it carries **more** live peer-exchange time than its twin, the coherent
reading is over-responsiveness — an arm that chases queue signal and scatters the peer groups it was
trained to keep together, while a sluggish pointwise scorer keeps them. **That reading is post hoc and is
recorded as a hypothesis, not a result.** Testing it needs its own registration with its own bars; it is
explicitly NOT folded into this one, and neither is the within-batch group-splitting statistic that the
same traces would support.

**Closed.** No constraint follows from a refuted hypothesis, so stage 2 is not authorised. Artifacts:
`simulation_data/serving_gap_v1_{s0,h1,h2}.json`, traces under
`simulation_data/serving_gap_v1/` on datalab.

---

### Registration (2026-09-12)

**Why this is the candidate.** The platform cap is the existence proof that the streaming gap is at
least partly a *serving* defect with a serving fix: one constraint the supervised label never
needed, `GNN_PREFIX_PLATFORM_CAP=1`, moved `gnn` from −1.5 % to +21.9 % against reactive Knative
(16/16 seeds, p = 3.1e-05) and gave `mpoff` nothing (−0.80 %, p = 0.56) — i.e. it corrected a
pathology specific to how message passing places, and it was found by reading a decode trace rather
than by retraining. If a second such pathology is measurable, it is worth the same cheap search. If
none is, the honest outcome is that the streaming gap is not a serving defect and `peer_affinity_v1`
stands as written.

**Substrate.** The three existing corpora and their existing checkpoints, unchanged:
`x200 p2` (T1b), `x800 p2`, `x800 p3`, `gnn` and `mpoff` at lr2e3, 16 seeds each. The three
matched production traces already built and md5-verified on both venues. No new generation, no new
training, no new physics.

**S0 — the control that must pass before any treated statistic is read.** On the *offline* held-out
datasets, with the decoder driven from the same trace-reading code, `gnn` must reproduce its
registered offline advantage over `mpoff` on each corpus to within the tie tolerance already used
(`peer_affinity_t1_read.py`). If the diagnostic harness cannot reproduce the offline sign, it is
measuring its own bug and nothing below may be read. (This is the lesson of
`route_b_env_pivot_v1`'s void rungs: a control that cannot fire is not a pass.)

**H1 — herding.** Claim: MP's per-step scores are more correlated *across consecutive batches*, so
the arm re-selects the same node while `mpoff` spreads. Statistics, per arm and per corpus, over
all 45,375 live batches of one seed and then over 16 seeds:

| statistic | definition |
|---|---|
| node choice entropy | Shannon entropy of the chosen-node distribution, normalised by log(n candidate nodes) |
| modal repeat rate | fraction of consecutive batch pairs whose modal chosen node is the same |
| choice autocorrelation | lag-1 autocorrelation of the per-batch node-histogram vector (cosine) |

**H1 fires** when, paired by seed, `gnn`'s modal repeat rate exceeds `mpoff`'s by ≥ 0.05 absolute
**and** its node-choice entropy is lower, both with exact Wilcoxon p < 0.05 over 16 seeds, on at
least two of the three corpora. Direction is registered: `gnn` more herded, `mpoff` less.

**H2 — state blindness.** Claim: MP's peer block crowds out the queue features, so its score
responds less to how loaded a candidate already is. Statistic: on 500 sampled live graphs per arm,
perturb every candidate platform's queue-depth feature by +1 standard deviation of its live
distribution and record the mean absolute change in the per-candidate score, normalised by the
score spread at that step (a finite-difference sensitivity). **H2 fires** when `gnn`'s normalised
queue sensitivity is below `mpoff`'s, paired by seed, exact Wilcoxon p < 0.05, on at least two
corpora.

**Kill bar (registered).** If neither H1 nor H2 fires, the lineage is **NO-GO** and closes: the
streaming gap is then not explained by herding or by queue insensitivity, no serving constraint
follows from it, and the entry goes to `docs/hard-stops.md` as "the offline/live reversal in
`peer_affinity_v1` is not a diagnosable serving defect". No further stage is authorised by this
registration.

**On GO — stage 2, one constraint per fired hypothesis, each behind a default-off env flag.**
H1 → an anti-herding constraint (a per-node budget over a sliding window of batches, the streaming
analogue of the platform cap). H2 → a queue-feature rescale at serve time. Each is gated exactly
as the platform cap was: 34 arms on the matched production trace, capped and uncapped controls,
16 seeds, and it counts only if (a) `gnn` improves against its own uncapped arm with exact
Wilcoxon p < 0.05 over 16 seeds and (b) the live `gnn` − `mpoff` contrast moves toward zero by
≥ 5 pp on the corpus where it is most negative (`x800 p3`, −10.67 %). A constraint that improves
`gnn` while leaving the contrast unmoved is recorded as a second serving fix, not as progress on
the question.

**What a GO would and would not establish.** It would establish that a measurable serving pathology
explains part of the offline/live reversal, and it would produce a deployable constraint. It would
**not** establish that message passing helps: that needs `gnn` to beat both `mpoff` and reactive
Knative in one measurement, which no reading of this program has yet produced, and the standing
answer in `peer_affinity_v1` holds until one does.

**Cost.** Stage 1 is trace reading and finite differences on existing artifacts: ~1 day, no GPU,
one 3k-event trace per arm per corpus plus one full-trace run per arm per corpus for the
45,375-batch statistics (6 runs, ~20 min each on CPU-amd). Stage 2 on GO is two gates of 66 arms,
~3 h each.

| stage | what | status |
|---|---|---|
| S0 | diagnostic harness reproduces the offline sign | registered |
| 1 | H1 herding · H2 queue sensitivity, 16 seeds × 3 corpora | registered |
| 2 | one serving constraint per fired hypothesis, gated live | authorised only on GO |
