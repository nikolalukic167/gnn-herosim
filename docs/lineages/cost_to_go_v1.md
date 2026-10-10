# cost_to_go_v1 — does a learned post-batch value beat exact batch-myopic search?

**Status:** `ACTIVE` (2026-10-11). Registered 2026-10-10; every bar below was signed before any data.

**Now (2026-10-11).**
- **S0 PASSES pooled at every H, but thinly:** the headroom is 4.74 / 3.25 / 3.35 % at H 5/15/30. It sits in a tail
  (the per-state median is 0).
- At short H it is mostly S's batch-surrogate error. The post-batch window part reaches the bar only at H30 (3.17 %).
- **Disclosed ceiling:** the one-step oracle headroom is below the gate's −5 % WIN bar, so expect DIRECTION-ONLY at best.
- In the split 9–10-task groups the lever is search, not V.
- S1 is running.

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

- 2026-10-11 05:50 — **Hand V offline: the rank signal is real at short H, and it recovers about half the H=5
  headroom on held-out topologies** (S7; rp/cost-to-go-handv cc305957; feature pass 856141, 2,386 rollouts, 0 errors;
  every rollout bit-identical to its S0 row; offline, orders the work only).
  - **Split, fixed before the read:** FIT 16251/53/55/57, EVAL 16252/54/56/58. That is 4 + 4 topologies, so treat it
    as a direction.
  - **On EVAL:**

    | H | λ (fit) | pooled Spearman dV vs dQ | hit rate, λ=0 → best λ | Q_H saved vs cd_exactS |
    |---|---|---|---|---|
    | 5 | 0.1 | +0.64 | 73.1 → 74.3 % | 72.3 s of 127.6 s headroom (56.6 %) |
    | 15 | 0.0316 | +0.57 | 67.4 → 67.4 % | 72.3 of 379.4 s (19.1 %) |
    | 30 | 0.0562 | +0.48 | 63.4 → 60.0 % | −19.5 of 1,082 s (−1.8 %, overfit) |

  - The saving comes from a few large-advantage states. The small-λ (tie-breaker) regime is λ=0 at every H and saves 0.
  - **Features:** queue drain is nonzero in 88 of 346 states.
  - **In-flight remaining read 0 in every state: an accessor artefact, corrected 06:10 (S7).**
    - Cause: `inflight_remaining_seconds` (`src/placement/live_audit.py:135-141`) reads `platform.inflight_service_end`,
      which only live `_serve_task` sets. Replay ghosts never set it.
    - The snapshots do hold in-flight work: 344 of 346 states have ghosts at t0, mostly input_io. The state total has
      median 208 s, against a queue-drain median of 0 s.
    - So view (b) is untested, not null. The same accessor would also read 0 for `pg_inflight` in any replay.
    - Next: recompute (b) offline from the snapshot ghosts, as sum (b1) and max-over-ghosts (b2).
  - The cold term is not broken out, and the pooled platform-load term is not yet in V.
- 2026-10-11 03:40 — **S0: PASS pooled at every H, marginal; S1 launched; ruling** (S6; 855965 at a0048522; 346
  non-split single-decision held-out states; 7,158 rollouts, 0 errors. Split read 856004 at bd92e30c; S1 856121).
  - **Headroom** = best of the 6 plans vs cd_exactS, summed share of Σ Q_H(cd_exactS); the bar is 3 %:

    | | H5 | H15 | H30 |
    |---|---|---|---|
    | all | 4.74 % | 3.25 % | 3.35 % |
    | without the GNN slot | 4.43 % | 3.14 % | 3.24 % |
    | heavy (169) | 1.61 % | 1.88 % | 3.50 % |
    | moderate (177) | 6.88 % | 4.39 % | 3.22 % |
    | self-check residual pairs dropped (robustness, not the registered bar) | 4.74 % | 2.81 % | 3.39 % |

    - Per-state median 0.000 % at every H; p90 1.8 / 2.9 / 7.8 %.
    - cd_exactS's own plan is best in 239 / 226 / 202 states. The GNN plan wins 14 / 14 / 17.
    - Four self-check residual states are among the winners.
  - **Decomposition of the headroom:**
    - batch part 2.85 / 0.92 / 0.18 %. This is S's own surrogate error: within the same plan set, the best label beats
      cd_exactS's by 26 % of the summed batch label.
    - post-batch window part 1.89 / 2.33 / 3.17 %. This is the part only a value of the post-batch state can capture.
  - **Horizon edge:** pairs beyond the cut have median 0.0 % (mean 0.6 % at H5), far under the 30 % flag.
    cd_exactS fell back to expansion in 170 / 264 / 313 rollouts.
  - **Exclusions:** heavy 37 split views and 17 multi-decision states; moderate 30 and 8. A first run (855959) was
    cancelled for mixed code (a96b8f27 / a0048522) and kept unused.
  - **Split read (38 whole 9–10-task groups, reported apart):**
    - raw headroom 13.5 / 6.6 / 6.8 %;
    - almost all of it is search: above the 100k cap, the exact lowest-S slate plan beats cd_exactS's expansion
      fallback by a median 18.4 % on batch latency;
    - against the better of cd_exactS and argmin-S, the cost-to-go part is 0.07 / 0.05 / 0.39 %. There the lever is
      search, not V.
  - **Ruling (coordinator):**
    - S0 passes as registered (pooled, every H), so the design proceeds: S1, then hand V, then MLP, then GNN only if
      the MLP leaves ≥ half the headroom.
    - **Disclosed before any model:** the one-step oracle headroom (3.2–3.4 % at H15/30) is below the gate's −5 % WIN
      bar. A perfect V captures at most the oracle, so a WIN would need live compounding beyond it. The expected best
      outcome is DIRECTION-ONLY.
    - Heavy has headroom only at H30.
    - Per rule 6 the registered live gate still runs, with whatever V arms survive selection, and at minimum the hand V.
    - The search finding (the cap fallback in large groups) is recorded as an input for cd_exactS's successor, not as
      part of this lineage.
- 2026-10-11 02:30 — **Gate inputs built and sealed** (S4; 856084 at 9e5ee729, s160_gate_inputs.sbatch).
  - **Topologies:** 16369–16392, 24 of 24 built, 0 failures. Spares 16393–16399 are unbuilt.
  - **Range check:** a scan of datalab and every rp/* branch; the ids are disjoint from all prior pools, validation, gate
    and training ids.
  - **No admission screen**, matching the accel gates. A topology may be replaced only on a build failure, never on a
    reference's behaviour.
  - **Setup:** accel base cfg (fastest_compatible); moderate m 11.6139, heavy m 27.622665025860204; windows g0–g3
    byte-identical to the accel gate's.
  - **Selection file:** `/share/nikola.lukic/cost_to_go_sel/selected.json`, md5 b48e72cc4bf71fd1201ac14fd5360aef.
  - Nothing has run on these topologies.
- 2026-10-11 01:45 — **Literature context and design notes, before S0 is read** (SCIENTIST; [V] = abstract checked on
  Semantic Scholar, [M] = from memory).
  - **Exact myopic solver plus a learned V has deployed precedent in dispatch:**
    - Xu et al. KDD 2018 [V]: KM matching on reward + γV;
    - NeurADP, AAAI 2020 [V]: ILP plus a neural V, up to 16 %;
    - Simão/Powell 2009 [V]: assignment LP with ADP values;
    - Ulmer et al. 2019 [V]: offline V plus online rollout beats either alone.
  - None found for cloud placement; treat it as a gap, not a prior.
  - CEVD 2021 [V]: a decomposed per-agent V costs up to 9.8 % when agents compete. That supports a joint V over the
    batch.
  - **No paper shows a GNN V beating an MLP V.** The matched MLP rung stays required.
  - **Design notes:**
    - (a) The target is already a paired advantage on one shared future, the standard variance fix.
    - (b) Once argmin S + λV is served, states drift away from cd_exactS continuations. If a learned rung passes
      offline, plan one re-collection pass under the new policy before the gate.
    - (c) Pick λ on validation environments disjoint from the gate topologies. A small λ (V as a near-tie breaker) is
      the safe regime.
- 2026-10-10 21:15 — **Registered** (coordinator, on the user's "do both in parallel"). S6 builds the rollout tool and
  runs S0 and S1. S5 builds the opening split at G if S0 says the opening matters.
