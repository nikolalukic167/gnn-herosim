# decima_rule_v1 — Decima's strongest hand baseline (tuned weighted fair) as a burst-seat bar

**Status:** `ACTIVE` (2026-09-26). Registered, not yet gated. Every bar below was signed before any
datum of the rule existed.

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

- 2026-09-26 — Registered (f40dc43 plus this node). Tests 8/8. Local smoke (not a result): 9101 w1, first
  3,000 arrivals.
  - Rule 10.78 s at α = −1 and −2, 10.21 s at α = +1; CD 5.31 s.
  - The median batch sees about 0.3 other active jobs, so the share rarely binds.
