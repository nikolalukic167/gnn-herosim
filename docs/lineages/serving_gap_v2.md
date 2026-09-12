# serving_gap_v2 — REGISTERED

> **Status:** `REGISTERED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-09-12 → (open)

**Outcome.** Pending. Registered 2026-09-12, before any statistic of it exists, on artifacts that already
exist (the 192 live production traces of `serving_gap_v1`). Nothing here trains, relabels or changes physics.

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

- [Registration (2026-09-12)](#registration-2026-09-12)

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
