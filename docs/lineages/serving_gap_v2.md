# serving_gap_v2 — CLOSED (NO-GO)

> **Status:** `CLOSED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-09-12 → 2026-09-12

**Outcome.** **NO-GO on the registered kill bar.** H3 fires on **one** corpus of the three and the bar
required two, so the lineage closes and H4 is not read. What the measurement does show, and what is worth
carrying: the co-location shortfall is **monotone in peer-graph density and orders exactly like the live
performance gap**, but only clears significance on the densest corpus. The bar was fixed in advance and is
not moved after seeing that.

**The question.** `serving_gap_v1` closed NO-GO by refuting both its hypotheses **backwards**: the
graph-aware arm is not herded and not load-blind — it spreads its node choices *more* than its pointwise
twin (x800 p3: repeat rate −0.083, p = 0.009) and is **3–7× more** responsive to queue depth on every
corpus (p < 1e-4). Together with `peer_affinity_v1`'s finding that the same arm carries *more* live
peer-exchange time than its twin, one reading fits all three facts: **over-responsiveness** — the graph arm
chases the queue signal and breaks up the peer groups it was trained to keep together, while a sluggish
pointwise scorer keeps them. That reading is **post hoc** and is why this is a separate registration rather
than an extra statistic inside the closed one.

**Related:** [serving_gap_v1](serving_gap_v1.md) · [peer_affinity_v1](peer_affinity_v1.md) ·
[cosim_deepdive_v1](cosim_deepdive_v1.md) · [objective_pivot_v1](objective_pivot_v1.md)

**What this must NOT become** (unchanged from v1 and restated so it cannot drift): no relabelling on loaded
states (closed by `cosim_deepdive_v1`, 4,400 live-visited states, additive R² 0.99999), no closed-loop
training (closed by `objective_pivot_v1` Phase 3, n = 120, p = 0.928), no offline number quoted as evidence
about live placement.

**Entry points:** `scripts_cosim/serving_gap_trace_read.py` (new stages `h3`, `h4`) over the existing
`simulation_data/serving_gap_v1/traces_h1/` (every batch's plan, 96 runs) and the per-seed sensitivities
already in `simulation_data/serving_gap_v1/h2/`.

## Record

- [H3 — NO-GO on one corpus of two required (2026-09-12)](#h3-no-go-on-one-corpus-of-two-required-2026-09-12)
- [Registration (2026-09-12)](#registration-2026-09-12)

---

### H3 — NO-GO on one corpus of two required (2026-09-12)

96 live production runs (3 corpora x 2 arms x 16 seeds, the `serving_gap_v1` traces; job 760475 computes
the statistics). Co-location rate = the fraction of a batch's peer pairs whose two tasks landed on the same
node, averaged over all 45,375 batches of a run, paired by seed.

| corpus | `gnn` | `mpoff` | gnn − mpoff | p | fires | live gap (`peer_affinity_v1`) |
|---|---|---|---|---|---|---|
| x200 p2 | 0.362 | 0.355 | +0.002 | 0.78 | no | +2.37 % |
| x800 p2 | 0.359 | 0.369 | −0.024 | 0.13 | no | −4.42 % |
| x800 p3 | **0.334** | **0.416** | **−0.093** | **0.006** | **yes** | **−10.67 %** |

**Verdict: H3-DOES-NOT-FIRE** (1 of the 2 required corpora). Per the registered kill bar the lineage closes
and **H4 is not computed**: it was registered as meaningful only if H3 fired, and reading it now would be
choosing a statistic after seeing the one that failed.

**The two facts worth carrying.**

1. **The shortfall orders exactly like the live gap.** Across the three corpora the co-location difference
   goes +0.002 / −0.024 / −0.093 while the live contrast goes +2.37 / −4.42 / −10.67 %. Same ordering, same
   sign change between the first and second rung. On the densest corpus the graph arm co-locates 20 %
   fewer peer pairs than its twin *relative* (0.334 against 0.416) and uses more distinct nodes per batch
   (+0.033, p = 0.025). That is consistent with the scattering reading and does not establish it.
2. **Both arms co-locate only about a third of their peer pairs live** (0.33–0.42 on every corpus and every
   arm). Whatever separates the arms is a small difference on top of a regime where most pairs are already
   split — which is itself a fact about serving this environment on a stream, not about either model.

**Closed.** No stage 2. Artifact: `simulation_data/serving_gap_v2_h3.json` (per-seed values included).

---

### Registration (2026-09-12)

**H3 — the graph arm splits peer groups more.** The live cost is charged per **node pair** and is zero
within a node, so the quantity that decides the peer term is how much of a batch lands together. Statistics
per served batch, averaged over all 45,375 batches of a run, paired by seed over 16 seeds per corpus:

| statistic | definition |
|---|---|
| distinct nodes per batch | number of distinct node ids in the decoded plan, divided by batch size |
| co-location rate | fraction of the batch's *peer pairs* whose two tasks landed on the same node |
| largest same-node share | size of the biggest same-node group in the plan, divided by batch size |

**H3 FIRES** when `gnn`'s co-location rate is **below** `mpoff`'s, exact Wilcoxon p < 0.05, on ≥ 2 of the
3 corpora. Direction registered in advance: `gnn` co-locates **less** live. (Note this is the opposite of
what it does offline, which is the whole puzzle.)

**H4 — the split is explained by responsiveness.** If over-responsiveness is the mechanism, then across the
32 arm-seeds of a corpus, the seeds that react more to queue depth should co-locate less. Statistic: the
Spearman rank correlation, within each corpus and pooling both arms, between the per-seed queue sensitivity
already measured by `serving_gap_v1` H2 and the per-seed co-location rate from H3.

**H4 FIRES** when that correlation is negative with p < 0.05 on ≥ 2 of 3 corpora **and** remains negative
within the `gnn` arm alone (16 seeds) on at least one corpus — the within-arm check is what stops a
two-point between-arm difference masquerading as a trend.

**Kill bar (registered).** If H3 does not fire, the "scattering" reading is wrong and the lineage closes
NO-GO — the graph arm's live peer cost would then be higher for some reason other than splitting groups,
and this registration authorises no further stage. If H3 fires but H4 does not, the split is real but
unexplained by responsiveness: that is recorded as a measured fact, the lineage still closes, and any
mechanism claim needs its own registration.

**On H3 + H4 — stage 2, authorised by this registration only if both fire.** One serving constraint behind
a default-off flag: a per-batch co-location floor (the decoder may not place a peer pair on different nodes
while a cap-feasible same-node candidate exists), gated exactly as the platform cap was — 34 arms on the
matched production trace, capped and uncapped controls, 16 seeds — and it counts only if (a) `gnn` improves
against its own unconstrained arm at p < 0.05 over 16 seeds and (b) the live `gnn` − `mpoff` contrast moves
toward zero by ≥ 5 pp on `x800 p3`.

**What a GO would and would not establish.** It would explain the offline/live reversal mechanically and
yield a deployable constraint. It would **not** establish that message passing helps: that still needs one
measurement where `gnn` beats both `mpoff` and reactive Knative, which no reading in this program has
produced.

**Cost.** H3 and H4 are arithmetic over traces already on disk: minutes, no cluster, no GPU. Stage 2 on GO
is two gates of 66 arms, ~3 h each.

| stage | what | status |
|---|---|---|
| H3 | group splitting, 16 seeds × 3 corpora | registered |
| H4 | splitting vs responsiveness, Spearman | registered |
| 2 | co-location floor, gated live | authorised only if both fire |
