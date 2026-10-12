# Oracle loss diagnostic (post-close, `ib0_confirm_v1`)

Measured by S7 on branch `rp/surrogate-gap` (`scripts_cosim/oracle_loss_diag.py`, head cfdec05d). Reads `oracle_diag/` of `/share/nikola.lukic/ib0_confirm/oracle_8cc890fa` (12 cells,
read-only) and, for the slate question, the brute-force tables of the uncapped accel corpus. No simulation. Outputs: `/share/nikola.lukic/surrogate_gap/oracle_loss_v1/`.
The node's verdict (FAMILY-CLOSES, moderate -2.82 %, heavy -1.96 % against `cd_exactS_ib0`) is not reopened.

**Question.** Where does the hand bar lose to the live oracle, and is the remaining room reachable by a scorer over the declared slate or only outside it?

## Pool 1: inside the slate (the live -2.8 % / -2.0 %)

Both oracle modes (`_pg_oracle_apply`, `_pg_oracle_search`) search only the declared top-5 slate product plus the couples of the hand plan, so every oracle plan is inside the slate by construction;
slate membership cannot be read from the diag. Gain = hand-plan label minus oracle-plan label per batch, summed.

| rung | oracle batches | improved | gain s | % of hand label s | within-cap (batches / improved / s) | over-cap (batches / improved / s) |
|---|---|---|---|---|---|---|
| moderate | 73,279 | 9,718 | 45,148 | 7.9 | 59,908 / 6,359 / 26,527 | 13,371 / 3,359 / 18,621 |
| heavy | 73,281 | 14,512 | 82,168 | 15.8 | 56,700 / 9,487 / 48,632 | 16,581 / 5,025 / 33,536 |

(Counts match `read_864107.log`; over-cap gains are replay-scored local-search lower bounds.) The batch-level gain is larger than the live latency gap, which is a paired mean over whole traces.

- **By batch size** the room is spread, not concentrated: gain as a share of hand label seconds, moderate n = 1..10: 1.4, 8.7, 8.2, 12.3, 5.4, 8.0, 10.6, 8.6, 2.6, 6.7 %; heavy: 6.5, 22.7, 20.2, 15.6, 18.1, 13.7, 11.6, 25.9, 5.4, 9.6 %. Full counts per size in `diag.txt`.
- **Scorer-reachable, but deep in the S rank.** Among improved within-cap batches the oracle plan is never the S-rank-1 plan of the slate table; it is in the S top 4 in 4,930 of 6,359 (78 %) moderate and
  7,195 of 9,487 (76 %) heavy, top 8 in 87 % / 86 %, top 32 in 95 % / 95 %. Weighted by gain seconds, S top 4 reaches only 28 % (moderate) / 21 % (heavy) of the within-cap gain, top 32 about 66 %.
- **Cold (proxy only).** Batches whose oracle plan uses a platform no applied plan of the cell had used before: 403 moderate / 500 heavy (0.5-0.7 % of batches) carry 25 % / 17 % of the gain seconds.
  The sim's real cold charge (previous task type differs) is not in the diag.

## Pool 2: outside the slate

On the 519 states of `rp/surrogate-gap` frontier_v1 (held-out 373), the full-table optimum is NOT in the declared top-5 product in 17 of 182 held-out moderate states (9 %) and 24 of 191 heavy (13 %)
(all sources 32 / 274 and 30 / 245). There the best slate plan is a median 92 % (moderate) / 19 % (heavy) worse than the optimum (p25 29 % / 10 %); almost all exceed 1 %. Multi-node optimum in 7 of the 17 and
3 of the 24; the optimum charges a real cold start in 16 / 182 and 2 / 191, of which 3 and 1 lie outside the slate. Only a wider (uncapped) slate reaches this room. Its **live size is unmeasured**:
corpus states are not the 12 live cells and a per-state regret is not a latency.

Not measured here: the labels' cold-start mix on the live cells, and an oracle over a wider slate.
