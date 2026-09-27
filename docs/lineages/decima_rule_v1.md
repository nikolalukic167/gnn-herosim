# decima_rule_v1 — Decima's strongest hand baseline (tuned weighted fair) as a burst-seat bar

**Status:** `ACTIVE`. Registered 2026-09-26; gated 2026-09-26.
- **D2 `SELFPREDICT-REMAINS-BAR`:** Decima's tuned weighted-fair rule (α = +1) is +46.4 % slower than
  self-predict (0/12) and +65.7 % slower than CD (0/11).
- Against it, the learned arm reads −41.4 % (11/11) and GNN-seeded CD −42.9 %. That is a Decima-style
  "beats the heuristic" number against a weak bar, never a headline.

**Why:** the user asked to replicate Decima's best hand rule and gate it against the learned arms and CD.
Decima (Mao et al., SIGCOMM 2019, §7) reports its learned scheduler ≥ 21 % ahead of hand-tuned
heuristics. Its strongest heuristic in the Spark batched and continuous-arrival settings is the **tuned
weighted-fair scheduler**. Free executors are split across jobs in proportion to `W_j^α`, where `W_j` is
the job's remaining work and α is grid-searched; the reported best is α = −1. α = 0 is plain fair and
α = 1 is naive weighted fair.

## Design

- **Arm `decima_wfair_network`** (`src/policy/decima_wfair/scheduler.py`, CD's serving stack and peer-group
  batching). The mapping:
  - job = peer group;
  - executor = platform;
  - each batch's job gets `k_t = round(s_J · |P_t|)` platforms of its candidate pool per task type, namely
    the earliest-finishing ones;
  - within them, each task goes to its earliest-finishing platform, with batch-mates' service charged and
    exchange off.
  - α is set by `HEROSIM_DECIMA_ALPHA`.
- **What does not transfer, disclosed:**
  - Decima's lever is *ordering* (which job gets the next free executor). HeROsim queues are FIFO and
    placement is committed at arrival, so only per-job parallelism is expressible.
  - The rule is locality-blind by construction, as Decima's baselines are, so it prices no peer
    exchange: the term that wins in this environment.
- **Tuning (phase `decimatune`):** α ∈ {−2, −1, −0.5, 0, 0.5, 1} plus CD, on tuning topologies 9101–9104 ×
  4 windows. None of them is in the study. The α with the lowest median elapsed relative to CD is chosen by
  `scripts_cosim/decima_tune_read.py` and recorded here before the study runs.
- **Study (phase `decima`):** the rule at the chosen α on the `fresh_topo_burst_v1` topologies × 4 windows.
  CD, self-predict, `xs1load_selfref` and `xs1load_cdapply` are the existing runs.

## Bars (signed 2026-09-26, before any datum)

| read | contrast | fires as |
|---|---|---|
| **D2** | rule vs self-predict | `DECIMA-IS-STRONGER-BAR` (median < 0, p < 0.05) / `SELFPREDICT-REMAINS-BAR` otherwise |
| D1 | rule vs CD | reported |
| D_*_vs_decima | `xs1load_selfref` vs rule; `xs1load_cdapply` vs rule | reported: the Decima-style "learned vs best heuristic" number |

- **Expectations:** D2 `SELFPREDICT-REMAINS-BAR` 90 %. The exchange-off peer-greedy rule ties Knative
  (`peer_greedy_live_v1`), and a local smoke on tuning cell 9101 w1 read the rule at about 2× CD's elapsed.
  D1 is CD-faster by a wide margin; D_*_vs_decima is learned-faster by a wide margin.
- **Standing risk:** if D2 fires as expected, D_*_vs_decima is a large number against a weak bar. It must never be
  quoted as a headline over "the best hand rule". The burst-seat bars remain self-predict and CD.

## Entry points

- Rule: `src/policy/decima_wfair/scheduler.py`. Tests: `tests/test_decima_wfair.py`.
- Gate: `scripts_cosim/fresh_topo_burst_v1_gate.py` phases `decimatune` and `decima`, via
  `backlog_corpus_v1_gate.sbatch` with `WT=` pointing at the `decima-rule-v1` worktree.

## Record (newest first)

- 2026-09-26 — **Study gate read** (job 809132, phase `decima`, 48 of 48 runs, α = +1, at 5ead8a1).
  - **Attachment:** [`decima_read.json`](decima_rule_v1/decima_read.json).
  - **D1:** rule vs CD +65.73 %, p = 0.001, 0/11 (9466 dropped: CD w3). Per topology from +54.6 % (9435)
    to +222.9 % (9461).
  - **D2 `SELFPREDICT-REMAINS-BAR`:** rule vs self-predict +46.44 %, p = 0.0005, 0/12.
  - **Pooled means, rule vs CD:** elapsed 12.71 vs 7.63 s; queue 6.86 vs 2.91 s; exchange 5.62 vs
    4.18 s per task.
    - The rule pays in both terms: it spreads groups exchange-blind, so exchange rises, and the queue
      rises with it because exchange occupies the platform.
  - **Learned vs the rule:**
    - `xs1load_selfref` −41.37 % (11/11; 9423 dropped);
    - `xs1load_cdapply` −42.94 % (10/10; 9423 and 9466 dropped).
  - **Reading:**
    - Decima's best hand baseline does not transfer to a seat whose cost is dominated by peer exchange,
      because it is locality-blind by construction.
    - The ~41 % margin is the kind Decima reports (≥ 21 % over tuned heuristics), and like Decima's it is
      measured against a baseline that does not see the dominant cost.
    - The honest burst-seat bars stay self-predict and CD.

- 2026-09-26 — **Tuning read** (job 809092, phase `decimatune`, 34 min, at 06f66f8): **α = +1 chosen**.
  - **Only 2 of the 4 tuning topologies read.** 9102 and 9103 time out at 1800 s in every window for
    *every* arm, CD included (40 of 112 runs, 8 of them CD), so they carry no ordering information.
  - **Median elapsed vs CD over 9101 and 9104:**

    | α | vs CD |
    |---|---|
    | −2 | +90.6 % |
    | −1 (Decima's reported best) | +90.7 % |
    | −0.5 | +90.6 % |
    | 0 | +86.5 % |
    | +0.5 | +85.8 % |
    | **+1 (chosen)** | **+84.9 %** (9101 +99.8 %, 9104 +70.1 %) |

  - α moves the rule by at most 6 pp. Spreading a group wider (larger α) helps slightly.
  - Disclosed: 2 topologies is a thin tuning set. The choice matters little given the 6 pp range.
  - The study runs at `HEROSIM_DECIMA_ALPHA=1.0`.

- 2026-09-26 — Registered (f40dc43 plus this node). Tests 8/8. Local smoke (not a result): 9101 w1, first
  3,000 arrivals.
  - Rule 10.78 s at α = −1 and −2, 10.21 s at α = +1; CD 5.31 s.
  - The median batch sees about 0.3 other active jobs, so the share rarely binds.
