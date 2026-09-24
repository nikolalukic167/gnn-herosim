# selfpredict_bar_v1 — how big is the new hand-rule bar, against Knative and the CD greedy?

**Status:** `REGISTERED` (2026-09-24) — live gate signed before any data. Triggered by
`lookahead_mp_v1` closing `HAND-COORDINATION-RECOVERS` with no Knative, random or CD arm in its gate.

**Standing answer (2026-09-24): not run.** `lookahead_mp_v1` found that
`peer_greedy_selfpredict_network` — the peer-greedy rule plus a price for each unarrived partner at
the node the rule would give it now — beats `peer_greedy_network` −6.19 % / −8.53 % live. That made
it the programme's best hand rule, but its size against reactive Knative and against the CD greedy
(the batched ceiling every learned arm has lost to) was never measured. This gate measures both,
fresh, paired, at one commit, and replicates the P0b margin over the rule.

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

- 2026-09-24 — **Registered.** Reader tested on a mixed-code fixture (mechanics only) and on an empty
  directory (fails loud). Its sanity line exposed that the rollout gate's "reactive" arm was
  `knative_network_ect`, not the `knative_network` baseline of `peer_greedy_live_v1`, so this gate
  runs both and gates B1 on `knative_network` (correction filed in `docs/gates/gate-tools.md`).
