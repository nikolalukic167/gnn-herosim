# selfpredict_bar_v1 — how big is the new hand-rule bar, against Knative and the CD greedy?

**Status:** `CLOSED` (2026-09-24) — **BAR=SELFPREDICT**. Registered 2026-09-24; every bar below was
signed before its data. Triggered by `lookahead_mp_v1` closing with no Knative, random or CD arm.

**Outcome (2026-09-24).** `peer_greedy_selfpredict_network` — the peer-greedy rule plus a price for
each unarrived partner at the node the rule would give it now — **is the programme's best placement
policy at these rungs.** Six arms fresh at one commit (`639e9b1`, 192/192 runs), 16 + 16
`unsaturated_edge_v1` environments, paired `total_rtt`:
**vs reactive Knative −19.42 % (C40) / −21.44 % (C80), 16/16 each**; vs the old rule −6.19 % (15/16) /
−8.53 % (16/16) (`REPLICATES`); vs the CD greedy −7.78 % (12/16, p = 0.077, `NOT-SEPARATED`) /
−8.90 % (16/16); vs Knative-ECT −33.1 / −37.1 %; vs random −27.3 / −27.5 %. The sanity line reproduces
`peer_greedy_live_v1` exactly (rule vs Knative −12.96 / −16.01 %).

Caveats a reader must not quote without: 6 servers, 0.460393 arrivals/s, these 32 environments. The
simulation is deterministic, so B2 re-running P0b to the digit confirms the code, not sampling
variability. The CD greedy is a batched seat: the self-predict rule wins by skipping its 7.1 s
wait while paying ~3.4 s more queue and ~2.5 s more rendezvous per task, so B3 compares served
policies, not scoring functions — and at C40 it does not separate on the sign test. Every learned
arm in the record loses to the old rule live, so none is within reach of this bar **per arrival**; in the burst
seat uncapped `gnnedge0` beats this rule −7.25 % (thin, not an MP win; [`selfpredict_burst_v1`](selfpredict_burst_v1.md)).

## Bars (signed 2026-09-24, before any data)

Six arms FRESH in one OUT_DIR on the 16 + 16 `unsaturated_edge_v1` environments (6 servers,
0.460393 arrivals/s), paired per (rung, env), primary metric `total_rtt`, chain 5 % / p < 0.05 /
n = 16 (`scripts_cosim/selfpredict_bar_v1_read.py`):

- **B1 vs reactive** (`knative_network`, `peer_greedy_live_v1`'s baseline):
  `SELFPREDICT-BEATS-REACTIVE` / `REACTIVE-FASTER` / `NOT-SEPARATED`.
- **B2 vs the rule** (`peer_greedy_network`): `REPLICATES` only if ≤ −5 %, p < 0.05 on BOTH rungs.
- **B3 vs CD greedy** (`peer_greedy_network_cd`, batched seat): `SELFPREDICT-BEATS-CD` /
  `CD-FASTER` / `NOT-SEPARATED`.
- Disclosed: vs `knative_network_ect` (what `rollout_imitation_v1` called "reactive") and vs random;
  sanity — the rule vs reactive against `peer_greedy_live_v1`'s −12.96 % / −16.01 %.
- **Verdict:** `NOT-REPLICATED` if B2 fails (the bar stays `peer_greedy_network`);
  `BAR=SELFPREDICT` if B2 replicates and B3 is never `CD-FASTER`;
  `BAR=CD-BATCHED/SELFPREDICT-PER-ARRIVAL` if B2 replicates and B3 is `CD-FASTER` on a rung.

Disclosed before data: CD runs the batched peer-group seat (16 s window) and so pays a scheduler
wait the per-arrival arms do not; B3 is a comparison of served policies, not of scoring functions.

## Entry points

- Gate: `scripts_cosim/datalab/selfpredict_bar_v1_gate.sbatch` (192 tasks, 4 blocks of ≤ 48).
- Reader: `scripts_cosim/selfpredict_bar_v1_read.py` → `simulation_data/selfpredict_bar_v1/read.json`.
- Policy: `PeerGreedySelfPredictNetworkScheduler` in `src/policy/peer_greedy_network/scheduler.py`.

## Record (newest first)

- 2026-09-24 — **CLOSED `BAR=SELFPREDICT`.** Jobs 804467 / 804521 / 804570 / 804645, 192/192
  summaries on `639e9b1`. Transcript: `selfpredict_bar_v1/read_2026-09-24.txt` (+ `.json`).
  B1 vs `knative_network` **−19.42 % / −21.44 %** (16/16, 16/16) → `SELFPREDICT-BEATS-REACTIVE`;
  B2 vs the rule −6.19 % (15/16) / −8.53 % (16/16) → `SELFPREDICT-BEATS-RULE` on both rungs
  (`REPLICATES`); B3 vs CD −7.78 % (12/16, p = 0.0768) → `NOT-SEPARATED` / −8.90 % (16/16) →
  `SELFPREDICT-BEATS-CD`; never `CD-FASTER` → `BAR=SELFPREDICT`. Per task vs Knative: queue −1.65 /
  −2.36 s, exchange −1.93 / −1.82 s, rendezvous +0.27 / +0.43 s. Median elapsed s/task C40 / C80:
  self-predict 14.73 / 14.79, rule 15.59 / 15.90, CD 15.03 / 16.03, Knative 17.87 / 18.27,
  ECT 19.62 / 22.61, random 18.71 / 20.75.

- 2026-09-24 — **Registered.** Reader tested on a mixed-code fixture (mechanics only) and on an empty
  directory (fails loud). Its sanity line exposed that the rollout gate's "reactive" arm was
  `knative_network_ect`, not the `knative_network` baseline of `peer_greedy_live_v1`, so this gate
  runs both and gates B1 on `knative_network` (correction filed in `docs/gates/gate-tools.md`).
