# Evaluation-budget frontier of the hand surrogate (anchor for `decision_cost_s0_v1`)

Measured by S7 on branch `rp/surrogate-gap` (code `scripts_cosim/surrogate_gap_screen.py`, head cfdec05d; job 864012, 6 m 34 s on 32 CPUs). Accel physics, full candidate
sets, tables of at most 1e5 plans, both rungs, batch sizes n = 2..6. Nothing trains, no live run.

**Question.** Regret against the brute-force optimum as a function of k exact evaluations: (a) argmin of the ib0 surrogate S (k = 0); (b) ib0 S plus the k best S plans replayed;
(c) ICM plus alpha-expansion from argmin-S with a k-evaluation budget; (d) the GNN plan score, top-1 and top-k re-ranked by exact replay. A random arm is the analytic expected best of k.

**Exact evaluation is a lookup.** A fidelity replay of a plan reproduces its stored co-sim label bit for bit (max |replay - label| / label = 0.0 on 30 states), so the stored full-product
table is the oracle. Wall per exact replay: median 0.05 s steady (p90 0.22 s), 5.6-7.7 s for the first call on a fresh state. S over a 77k-plan table: 8.5 s; GNN scoring about 0.8 ms per plan.

## Pooled n = 2..6, mean regret % of the optimum (median regret is 0 for every non-random arm at every k)

| sample | rung | arm | k=0 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|
| all sources (519 states) | heavy | ib0 S + k | 4.06 | 1.48 | 0.34 | 0.01 | 0 | 0 |
| | | GNN re-rank | 9.76 | 3.39 | 2.45 | 0.37 | 0.06 | 0.02 |
| | | ICM + alpha-expansion | 4.06 | 3.66 | 3.35 | 2.91 | 2.03 | 1.75 |
| | moderate | ib0 S + k | 1.39 | 0.54 | 0.03 | 0.01 | 0 | 0 |
| | | GNN re-rank | 27.1 | 21.7 | 12.5 | 2.80 | 1.97 | 1.86 |
| held-out only (373 states: 191 heavy, 182 moderate) | heavy | ib0 S + k | 4.32 | 1.64 | 0.34 | 0 | 0 | 0 |
| | | GNN re-rank | 12.05 | 4.00 | 2.86 | 0.38 | 0.06 | 0.03 |
| | | ICM + alpha-expansion | 4.32 | 3.81 | 3.46 | 3.16 | 2.33 | 1.97 |
| | moderate | ib0 S + k | 1.73 | 0.75 | 0.04 | 0.01 | 0 | 0 |
| | | GNN re-rank | 25.84 | 22.63 | 16.90 | 3.11 | 2.25 | 2.24 |
| | | ICM + alpha-expansion | 1.73 | 1.66 | 1.46 | 1.29 | 1.19 | 0.88 |

Share of states within 1 % at k = 4 (held-out, ib0 S + k): 96 % heavy, 99 % moderate; at k = 8: 100 % / 99 %. GNN re-rank: 93 % / 93 % at k = 4, 97 % / 97 % at k = 8.
The budgeted alpha-expansion stalls at a local optimum (held-out: median 16-17 evaluations to convergence, mean regret there 1.94 % heavy, 0.66-0.86 % moderate).
Label shape (held-out): multi-node optimum 16 % heavy, 37 % moderate; top-2 gap median 9.7 % / 9.4 %; optimal ties median 1.
Per-n cells: `/share/nikola.lukic/surrogate_gap/frontier_v1/read.txt` (all sources) and `/share/nikola.lukic/surrogate_gap/frontier_v1_heldout/read.txt` (held-out), rows in `rows.jsonl` beside each.

## Disclosed limits

- **n = 4 hook errors bias the sample.** 27 of 546 selected states errored ("the decision hook never ran": the batch was not decided whole), all at n = 4: 25 held-out (17 heavy, 8 moderate)
  and 2 train. They were counted, not rerun. They are the co-location-heavy states: among held-out n = 4 states the errored ones have a multi-node optimum in 100 % (heavy) / 88 %
  (moderate) against 21 % / 46 % for the scored ones, and a top-2 gap median of 1.0 % / 2.6 % against 7.5 % / 8.6 %; plan counts are alike (1155 vs 1000, 652 vs 400).
  The n = 4 cells, and therefore the pooled cells, are optimistic.
- State counts per cell are 60 except n = 4 and n = 6 (all sources: heavy n = 4 42, n = 6 23; moderate n = 4 51, n = 6 43). Held-out n = 5 and n = 6 are thin (heavy n = 6: 4 states, moderate n = 6: 7).
- The GNN arm (checkpoint `accel-replica-v1-gnn-eng-seed1.pt`) is in-sample on the 146 train states of the all-source read; the held-out read has no such overlap. GNN scores are taken at the PG decision
  with the rest of the replay skipped (queued tasks are not pinned by the GNN policy).
- Batches of 7 or more tasks (over the plan cap) are not in the screen. Regret is of the batch label, not live latency.
