# cost_to_go_v1 — does a learned post-batch value beat exact batch-myopic search?

**Status:** `REGISTERED` (2026-10-10). Every bar below was signed before any data.

**Why.** Within the declared top-5 slate, exact search on CD's surrogate S (cd_exactS) dominates learned batch scorers
offline: hit 79.4 % against the GNN's 64.8 %, regret 6.1 % against 21.7 % of the summed optimum (`accel_replica_v1`,
2026-10-10). S is batch-myopic: it prices only this batch's tasks. The no-split give-back, where co-located large
groups starved later small groups, is direct evidence that the batch optimum is the wrong objective online. A value of
the post-batch state is the one lever exact batch search cannot reach. It is the classical case for approximate DP and
rollout (SCIENTIST, 2026-10-10).

## Design (fixed now)

- **Environment:** the accel seat. R1.1 + T-fix, fastest_compatible replicas, 160c × 24s, moderate and heavy rungs.
  Replay **G** (rp/replay-g e787127a) for any state with in-flight pulls.
- **Policy:** argmin over the slate of S(plan) + λ·V(post-plan state), by the cd_exactS enumeration (fallback as
  cd_exactS).
- **Target:** a short-horizon rollout ADVANTAGE, never a raw return (objective_pivot_v1 found raw horizon returns
  chaotic).
  - Q_H(state, plan) = this batch's latency + the summed latency of the tasks arriving in the next H seconds of the
    cell's own trace, with a fixed deterministic continuation policy. The continuation is cd_exactS, or 1-pass CD if
    cost forces it; that choice is disclosed and checked on 100 states.
  - A = Q_H(plan) − Q_H(argmin-S plan), within the same state.
  - H ∈ {5, 15, 30} s: the shortest H passing S1 is fixed before any training.
- **Data:** accel fidelity snapshots, steady plus an opening split built at G. Train topologies are for training;
  held-out topologies are for selection. Per state: the argmin-S plan, the next 4 plans by S, and the GNN plan.
- **Hard stops, offline, before any model is trained:**
  - **S0 headroom:** on held-out states, choose the best-by-Q_H of the 6 plans vs the argmin-S plan. STOP if the summed
    improvement is below 3 % of Σ Q_H.
  - **S1 stability:** perturb in-flight remaining times by ×(1 ± ε), ε ∈ {0.001, 0.01}. STOP if the median Spearman of
    the plan advantages is below 0.9, or the best plan changes on more than 10 % of states.
- **Models, smallest first:**
  - (a) a hand V: post-batch backlog seconds on the touched replicas, × a fitted λ. A rule with the same information.
  - (b) a pointwise MLP on post-plan per-replica load features.
  - (c) the GNN encoder plus its MP-OFF twin, only if (b) leaves at least half of S0's headroom.
  - Selection is on validation advantage ranking. Offline results order the work only.
- **Gate:** cd_exactS + V vs cd_exactS, paired, on 24 fresh topologies × 2 rungs × 4 windows.
  - WIN: median ≤ −5 % with Holm p < .05 at both rungs (Holm over the arms gated).
  - Descriptive: plain CD, cd_expand, cd_pull, and each V variant.
  - λ, H and the arm set are fixed before test data.
- **Claims:** a learned V that beats the hand V is a learned contribution. A GNN-V that beats its MP-OFF twin is the
  only route to an MP claim.

## Record (newest first)

- 2026-10-10 21:15 — **Registered** (coordinator, on the user's "do both in parallel"). S6 builds the rollout tool and
  runs S0 and S1. S5 builds the opening split at G if S0 says the opening matters.
