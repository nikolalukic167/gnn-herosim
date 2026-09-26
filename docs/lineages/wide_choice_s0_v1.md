# wide_choice_s0_v1 — does a wider per-task choice set push CD far from the per-batch optimum?

**Status:** `ACTIVE` (2026-09-26). Registered; not yet run. Every bar below was signed before any datum.

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

- 2026-09-26 — Registered.
