# live_headroom_v1 — how far is CD from the best single-batch plan, on the live states CD itself produces at ×2 / ×3 / ×5?

**Status:** `REGISTERED` (2026-10-01). The bar and the expectations below were fixed before any sweep of these states was read.

**Outcome (2026-10-01, offline, descriptive): `MIDDLE`.** On 3,935 live batches (mean 3.8 tasks, 138 of 144 capture units; the six
that hung dropped, mostly ×5) the CD cost model's plan is **5.6 / 5.0 / 5.6 %** above the single-batch optimum at ×2 / ×3 / ×5 —
above the 5 % `CLOSED` line at two rungs, far below the 10 % `OPEN` line. The 10-task-trained engineered-context arms
(`xs1load`, `xs1mpoff`, self-refine 3) are already at **2.1–2.5 %**, below that CD replica at every rung; the raw arms are at
10.5–11.1 % (`rawgnn`) and 16–17 % (`rawmlp`). So the one-batch headroom beyond CD is about 5 % and the learned
engineered arms already take more than half of it, yet they lose to CD live. Whatever separates them live is **not**
single-batch plan quality. It is between batches (the queue a plan leaves behind), which this measure excludes by
construction. Offline, replica not live CD, development topologies; no latency claim.

**Question.** The environment review (`raw_plan_v2` follow-up, 2026-10-01) judged the environment *likely closed* for a
message-passing win: the cost is a pairwise sum plus a count-form queue, which a hand rule or a pointwise scorer expresses.
It also found the one unmeasured quantity that decides whether any learned model can beat CD here: the gap between CD's
plan and the best plan **on live states at the graded load**. The only number on record is offline and unloaded
(`cd_gap_v1`: CD median regret 8.2 %, optimal on 688 of 2,516 groups). If CD is within 5 % of the optimum at every rung, no
learned scorer can win by more than that, and the program's remaining lever is a different environment.

**Design.**
1. Capture the batches and standing queues that the CD greedy (`peer_greedy_network_cd`) produces on the 12 development
   topologies at the grounded ×2 / ×3 / ×5 rungs (windows g0–g3; `datalab/live_headroom_v1_capture.sbatch`; a unit that hangs
   is killed at 20 minutes and dropped).
2. Sweep every captured batch exhaustively with the live state **as captured** (no synthetic backlog; whole peer groups of
   2+ tasks; `make_warm_corpus.py --variable-group-min 2`), so the optimum is the true single-batch optimum of a state CD
   actually met.
3. On the same graphs score, by the full sweep: the CD cost model's coordinate-descent plan (`small_batch_rules_eval.py`), the
   independent argmin, and the existing 10-task-trained checkpoints (`xs1load`, `xs1mpoff`, `rawgnn`, `rawmlp`, 4 seeds,
   self-refine 0 and 3).

**Statistic and call (signed now).** Per rung: mean regret in seconds and as % of the optimum batch RTT, over batches.
**CLOSED** if the CD replica is within 5 % of the optimum at every rung; **OPEN** if it exceeds 10 % at any rung; otherwise
**MIDDLE**. The same table is read for each learned arm as a "does it reach the optimum" row (descriptive).

**Expectations (judgement, not measurement):** CLOSED 30 %, MIDDLE 45 %, OPEN 25 %. A learned arm within 2 % of the optimum
at every rung is unlikely (15 %).

**What this does not measure — read before quoting.**
- The optimum is **one batch's** best plan on a captured state. It ignores what the plan leaves in the queues for later
  batches, so it can understate the headroom of a policy that plans across batches. It is a lower bound on joint headroom.
- The CD plan is the offline replica of CD's cost terms, not the live CD arm; its validity as CD is inferred, not shown.
- States are CD-induced and from 12 development topologies; one capture per unit; offline. Under AGENTS.md rule 6 this orders
  the work and closes nothing; a live gate is still needed for any claim about latency.
- Units that hang are dropped, which may drop exactly the most loaded states.

**Entry points.** `datalab/live_headroom_v1_{capture,corpus,cache,eval}.sbatch`, `scripts_cosim/live_headroom_read.py`,
`small_batch_rules_eval.py`, `queue_gap_diag_eval.py --cache --split`.

## Record

- 2026-10-01 — **Read (`live_headroom_v1/live_headroom_read.json`): `MIDDLE`.** Capture 138 / 144 units (hung and dropped:
  9423 ×5 g0, g2; 9466 g1 at ×2, ×3, ×5 and ×5 g2); corpus 4,001 datasets, 66 quarantined (`peer_norm 0`), 3,935 graphs. Regret
  against the full single-batch sweep, % of the optimum batch RTT (45.2 / 47.6 / 48.6 s at ×2 / ×3 / ×5; 1,371 / 1,355 / 1,207 batches):

  | | ×2 | ×3 | ×5 |
  |---|---|---|---|
  | CD replica | 5.6 | 5.0 | 5.6 |
  | independent argmin | 20.5 | 18.7 | 22.2 |
  | `xs1load` (refine 3) | 2.4 | 2.1 | 2.5 |
  | `xs1mpoff` (refine 3) | 2.4 | 2.2 | 2.5 |
  | `rawgnn` (refine 3) | 10.5 | 10.7 | 11.1 |
  | `rawmlp` (refine 3) | 17.3 | 16.0 | 16.4 |

  - **Reading:** within a batch CD is about 5 % from optimal, the pointwise engineered arm (`xs1mpoff`) matches the GNN, and
    both beat the CD replica offline. This agrees with the environment review (pairwise-sum cost, pointwise-expressible) and
    with check A of `small_batch_v1`. It does **not** show that nothing can beat CD live: CD wins live while losing offline,
    so the live gap sits outside what a single batch's regret sees.
  - **Limits:** one-batch optimum (a lower bound on joint headroom); CD replica, not the live CD arm; development topologies;
    hung units dropped; a single capture per unit.
- 2026-10-01 — registered; capture submitted (job 823293, 12 topologies × 3 rungs × 4 windows).
