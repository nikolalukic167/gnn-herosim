# wide_choice_s0_v1 — does a wider per-task choice set push CD far from the per-batch optimum?

**Status:** `FALSIFIED` (2026-09-27) — **S0-FAIL, NO-GO.** Registered 2026-09-26; read 2026-09-26.
Widening the per-task choice set does not push CD far from the optimum: CD's regret *falls* as width
grows, and live replica pools cap the width at ~2.3–2.9 candidates/task regardless. No live gate is
licensed on this lever; do not re-propose a wide-choice environment without new evidence that live
pools can be widened past that cap.
- **W1 `CD-MIDDLE`:** CD's median regret against the exact optimum is 13.9 % (tier A, 138 groups) and
  12.0 % (tier B, 60).
- **W2 `LEARNED-AT-OR-BELOW-CD`:** −1.8 pp paired.
- **W3 `COMPLEMENTARY`:** 33 %.
- **CD's regret does not grow with slate width.** The width the live replica pools allow is ~2.3–2.9
  candidates per task, not 3–4. The live offline-to-live gate this S0 would have licensed is not
  registered.

**An offline screen (S0).** It *orders* work; it does not close anything (rule 6).
- A pass leads to a registered live gate on a wide-choice environment, with training on a wide-choice
  corpus.
- A fail is recorded here, and the live gate is still owed if this direction is ever pursued.

**Why.** The learned burst-seat arms tie CD live ([`exchange_seconds_v1`](exchange_seconds_v1.md)), and
CD seeded by them beats CD by only −3.7 % ([`seeded_cd_xs1_v1`](seeded_cd_xs1_v1.md)). A ~20 % learned
win needs an environment where CD is far from optimal.
- CD's median offline regret against the per-batch optimum is 8.2 % on the jb2 held-out groups
  (`cd_gap_v1` D0) and 6.1 % on the `backlog_corpus_v1` test groups.
- Both corpora offer only about 2.1 candidates per task: `make_warm_corpus.choose_candidates` caps each
  group at 20,000 plans. Live serving offers 6–10.
- Re-analysing D0 by quintile: CD's median regret is 1.2 % at 1.2–1.5 candidates per task, and 14–20 %
  at 2.1–3 candidates or 972–18k plans.
- On the batches where CD's regret is ≥ 20 %, the one-pass `xs1load` does better on 82 %.
- Scratch analysis is in the session job directory; its method is the engine replay below.

## Design

- **Groups:** the `backlog_corpus_v1` held-out snapshot shards (`joint_burst_v2` shards of cells
  cc40s9223 and cc40s9224, windows w0 and w1). Same traces, cell configs, physics env and synthetic-backlog
  mechanism as `backlog_corpus_v1_corpus.sbatch`. Three things change:
  - **Wider slate:** `--target-combos` 100,000 (tier A, main) and 250,000 (tier B, reported), with
    `--min-choice-fraction 0.8`. Tier A holds ~3.1 candidates per task and tier B ~3.5 where the live
    pools allow; the achieved counts are recorded per dataset.
  - **Light-to-moderate load:** synthetic backlog rungs {0, 3} s only, busy probability 0.5. The
    analysis found backlog makes CD relatively better, so the heavy rungs 8 and 20 are dropped.
  - **Size:** 50 groups per shard (tier A) and 15 per shard (tier B), about 200 + 60 groups. Every one
    carries the full exhaustive `placements.jsonl` sweep. The optimum is **exact**: every group is
    brute-forced, with no bound.
- **Arms**, each replayed through the real engine on the dataset's own slate (the
  `joint_burst_v1_offline_rule_regret` path that reproduced D0's 8.21 %):
  - **CD** (`peer_greedy_network_cd`) and the 1-pass greedy.
  - **`xs1load` s1–4 served live** through the engine (`gnn_gnn`, the `peer_affinity_live_serve_check`
    path, which matched the offline decode on 59/60 at P1). Three modes each:
    - one-pass;
    - `GNN_PREFIX_SELF_REFINE=3`;
    - `GNN_CD_REFINE=apply`, i.e. GNN-seeded CD.
  - Served with `HEROSIM_INFLIGHT_CAPTURE=service_end_v1`, as gated.
  - **Disclosed:** `xs1load` was trained on ~2.1-candidate slates, so ~3+ candidates is out of
    distribution. That is part of what W2 measures.
- **Regret** = `100 × (rtt / sweep optimum − 1)`. A replay below the optimum fails loudly, since it
  would mean a plan outside the slate.
- **Code:**
  - Replay: `scripts_cosim/wide_choice_s0_v1_replay.py`.
  - Read: `scripts_cosim/wide_choice_s0_v1_read.py`.
  - Job: `scripts_cosim/datalab/wide_choice_s0_v1.sbatch`.

## Bars (signed 2026-09-26, before any datum)

| read | quantity (tier A) | fires as |
|---|---|---|
| **W1** | CD's median regret | `CD-WEAK` ≥ 20 % / `CD-MIDDLE` 10–20 % / `CD-STRONG` < 10 % |
| **W2** | paired per group: `xs1load` one-pass regret (median over 4 seeds) minus CD regret, median over groups | `LEARNED-AT-OR-BELOW-CD` (median ≤ 0 pp) / `LEARNED-ABOVE-CD` |
| **W3** | share of groups where the better of {CD, `xs1load` one-pass} is ≥ 10 % faster than CD | `COMPLEMENTARY` ≥ 25 % / `PARTIAL` 10–25 % / `REDUNDANT` < 10 % |

- **Reported, not bars:**
  - W2 for self-refine and GNN-seeded CD;
  - tier B's W1–W3;
  - the mean regret and the share ≥ 20 %;
  - W1 by candidates-per-task and by plan-count quantile;
  - the 1-pass greedy.
- **S0 passes** when W1 = `CD-WEAK` **and** W2 = `LEARNED-AT-OR-BELOW-CD`. That licenses a registered
  wide-choice corpus, training and live gate.
- **Groups are not independent:** 2 cells × 2 windows × snapshots in time. The reads are distributional
  and quote no p-values.
- **Expectations:**
  - W1: `CD-WEAK` 35 %, `CD-MIDDLE` 45 %, `CD-STRONG` 20 %.
  - W2 at or below CD: 55 %.
  - W3 `COMPLEMENTARY` 50 %.

## Record (newest first)

- 2026-09-27 — **Closed, `FALSIFIED` / `S0-FAIL`.** No new evidence since the 2026-09-26 read; recorded
  here so the index and this head agree. Re-opening needs a signed amendment showing live pools can be
  widened past ~3 candidates/task, not a re-run of this design.

- 2026-09-26 — **S0 read** (datalab; generation array 809098 plus read 809099, at 3e75638 on
  `wide-choice-s0-v1`).
  - **Attachment:** [`wide_choice_s0_v1_read.json`](wide_choice_s0_v1/wide_choice_s0_v1_read.json).
  - **Groups, fewer and narrower than designed:**
    - Tier A kept 138 of 479 snapshots; tier B kept 60 of 167. Every rejection is the
      `--min-choice-fraction 0.8` rule: fewer than 80 % of a group's tasks can keep ≥ 2 candidates.
    - The cause is the live replica pools. Most snapshots hold 2–3 replicas per task type (pool
      sizes (3, 3) 41×, (2, 3) 38× in tier A), so the target caps are rarely binding.
    - Achieved width: tier A has a median of 2.3 candidates per task (q10–q90 1.9–3.2) and 2,124 plans;
      tier B has 2.9 (2.0–3.7) and 22,950 plans. Only 10 + 22 groups exceed 3.25.
    - The regret study's "6–10 live candidates per task" does not hold for these states: pools are the
      binding limit.
    - Every learned replay formed exactly 1 batch, and no replay beat the exhaustive optimum.
  - **The landscape is a needle.** The median share of plans within 5 % of the optimum is 0.23 %
    (tier A) and 0.19 % (tier B).
  - **Median regret over the exact optimum, tier A** (tier B in brackets):

    | Arm | Median regret | Share of groups ≥ 20 % |
    |---|---|---|
    | CD | 13.9 % (12.0) | 46 % |
    | 1-pass greedy | 43.5 % (44.1) | — |
    | `xs1load` one-pass (median of 4 seeds) | 14.1 % (13.9) | — |
    | self-refine | 12.0 % (12.4) | — |
    | GNN-seeded CD | 12.7 % (10.6) | — |

    CD's mean regret is 90.0 % (49.1): a heavy tail.
  - **W1 `CD-MIDDLE`** (13.9 %).
  - **W2 `LEARNED-AT-OR-BELOW-CD`:** one-pass minus CD is −1.8 pp (tier B −1.9). The learned arm is better
    on 52 % of groups and worse on 41 %.
    - Self-refine: −2.5 pp. GNN-seeded CD: 0.0 pp at the median, mean −50 pp. It removes CD's tail.
  - **W3 `COMPLEMENTARY`:** in 33 % (32 %) of groups, the better of {CD, one-pass} is ≥ 10 % faster than CD.
  - **Regret by width, the decisive read.** CD's regret *falls* as the slate widens; the learned arm's
    *rises*:

    | Candidates per task | CD median (tier A) | One-pass median (tier A) |
    |---|---|---|
    | < 2 (n = 23) | 22.6 % | 7.7 % |
    | 2–2.75 (n = 74) | 17.7 % | 16.0 % |
    | 2.75–3.25 (n = 31) | 13.5 % | 14.9 % |
    | > 3.25 (n = 10) | 7.3 % | 34.7 % |

    - Tier B shows the same: CD 30.0 / 12.7 / 7.6 / 12.3 %, one-pass 0.0 / 13.3 / 13.7 / 18.9 %.
    - This refutes the premise that widening the slate makes CD weak. The earlier quintile correlation
      (1.2 % → 14–20 %) confounded width with other group properties.
    - The learned arm, trained on ~2.1-candidate slates, degrades out of distribution at > 3 candidates.
  - **Reading:** in this environment the slate is capped by the live replica pools, and within that
    range CD is not systematically weaker on wider choices.
    - A 20 % learned edge would need a physics change that grows the pools (more replicas per type), and
      even then this S0 shows CD improving with width.
    - CD's weakness is a heavy tail of needle groups: 46 % of groups ≥ 20 % regret, mean 90 %. The
      learned arms and GNN-seeded CD already trim that tail (mean −31 to −50 pp).
    - The actionable thread is the tail, not the width.
    - The live gate this S0 would have ordered is not registered.

- 2026-09-26 — Registered.
