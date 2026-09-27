# replica_guard_v1 — does a keep-warm serving guard remove the learned arm's 9434 w0 collapse and let its ×1 w0 lead over CD separate?

**Status:** `ACTIVE`. Registered 2026-09-27; gated 2026-09-27. **K1 `NOT-FIXED`, K2 `TIES`.**
- The keep-warm guard halves the 9434 collapse but does not remove it.
- It adds a new failure: a starved-client spin on 9119 draw d3, every seed.
- Against CD it reads −11.25 %, p = 0.24, 8/11, the same as its unguarded twin (−10.91 %).

**Parent:** [`burst_ladder_v1`](burst_ladder_v1.md). Its Amendment 1 read `xs1load_selfref` against CD at
×1 w0 over 4 perturbed draws at −11.9 %, faster on 9/12, p = 0.15 (`DISSOLVES` by its bar).
- On the 9 topologies where it leads, the lead is a tight −9.5 to −16.4 %.
- 9434 collapses on every draw and seed (+274 %). 9456 (+8 %) and 9461 (+13 %) trail.

## The 9434 diagnosis (2026-09-26/27, exploratory, disclosed)

Scratch reruns on datalab (jobs 809147, 809165, 809166), on a throwaway branch that is not pushed.
- **The reruns reproduce the gate exactly:** CD 8.701 s; `xs1load_selfref` s1 51.588 s, s2 19.557 s.
- **The excess is queue, only in w0, episodic and late** (about 58–82k s and 94–102k s).
  - From about 58k s, `dnn1` has two live replicas (node1:203 `xavierCpu`, node5:228 `pynqFpga`), so the
    median `dnn1` task has **one** candidate.
  - node1:203's queue reaches about 800 s. It drains once a new replica appears at about 80k s.
  - At the same moments CD spreads `dnn1` over 5–6 platforms.
- **Cause:**
  - Before the peak, the model sends `dnn1` to the fast devices. The rpi replicas it stops using idle past
    `keep_alive` (30 s), and the autoscaler removes them: it scales down whenever ceil(concurrency / 100) is
    below the count.
  - It scales back up only at about 200 tasks in flight.
  - CD-seeded `xs1load` (seed 2) collapses the same way, so CD's refine is equally blind to expiry.
- **Counterfactual** (`keep_alive` effectively infinite, every arm):
  - CD 2.80 s; `xs1load_selfref` s1–s4 2.83 / 2.93 / 2.87 / 2.89 s.
  - The collapse vanishes.
  - Replica churn is about two thirds of every arm's w0 latency on this cell.
- **Classification:** a closed-loop weakness of the learned policy, not a serving bug. The features and the
  decode are correct, and nothing in the per-batch label rewards keeping a replica alive.

## Design

- **Guard** (`GNN_REPLICA_KEEPWARM=1`; `GNNScheduler._keep_replicas_warm`, `src/policy/gnn/scheduler.py`).
  Serving only, default off, applied after decode and self-refine.
  - For each task type in the batch with at most **4** replicas: every replica that is idle (empty queue, no
    current task), not already targeted by the batch, and within **5 s** of `keep_alive` expiry receives one
    task of that type.
  - The task chosen is the one whose chosen platform has the longest queue, then the lowest index.
  - The guard is score-blind by design. Counters: `keepwarm_armed`, `keepwarm_moves`,
    `keepwarm_batches_fired`, `keepwarm_expiring_seen`.
  - **The parameters (4 replicas, 5 s) are fixed a priori, not tuned on any cell.** 4 is below the 5–6 platforms
    CD keeps warm on 9434. 5 s is five reconcile intervals.
- **Arms**, all at this lineage's commit:
  - `xs1load_selfrefkw`: guard plus 3 self-refine passes, the exchange_seconds_v1 checkpoints s1–4.
  - Its twin `xs1load_selfref`: the same code with the guard off.
  - CD.
- **Cells:**
  - ×1 w0, burst_ladder_v1 Amendment 1's perturbed draws d1–d4, 12 study topologies: CD, guard and twin (s1–4).
  - 9434 w0 as every gate ran it: guard and twin, s1–4.
  - w1–w3 unperturbed: guard and twin, s1–2.
  - 584 runs, one job (`backlog_corpus_v1_gate.sbatch` phase `guard`).
- The driver checks per run that the guard is armed only on the guarded arm, and that self-refine ran.

## Bars (signed 2026-09-27, before any datum)

- **Statistic:** per topology, the median elapsed over all of an arm's runs (draws × seeds), then
  100 · (arm / ref − 1), then an exact two-sided Wilcoxon over topologies (`scripts_cosim/replica_guard_v1_read.py`).

| read | contrast | fires as |
|---|---|---|
| **K1** | guard vs twin on 9434 (w0 and draws, per seed) | `COLLAPSE-FIXED` if every guard seed is within +25 % of CD's same-draw median and every twin seed is not; `PARTIAL` if the guard median improves ≥ 50 %; else `NOT-FIXED` |
| **K2** | guard vs CD, ×1 draws, 12 topologies | `BEATS-CD` (median ≤ −5 %, p < 0.05) / `BEATS-CD (direction only)` (median < 0, p < 0.05) / `TIES` (p ≥ 0.05) / `CD-FASTER` |
| **K3** | guard vs twin on the draws (11 topologies other than 9434), and on w1, w2, w3 | reported: the guard's cost where nothing collapses; expected within ±2 % |
| reported | twin vs CD on the draws (4 seeds); guard moves per 1,000 tasks | |

- **Expectations:**
  - K1 `COLLAPSE-FIXED` 55 %, `PARTIAL` 30 %.
  - K2 `BEATS-CD` 45 %, `BEATS-CD (direction only)` 25 %, `TIES` 30 %.
  - K3 within ±2 %.
- **Standing risks:**
  - 9456 and 9461 also trail CD at ×1 with no replica collapse. The guard does not address them, so K2 can
    still read `TIES`.
  - A guard move can raise exchange for the moved task.
  - This remains the w0 window only; 4 draws, 4 seeds, 12 topologies.
  - A K2 pass is a live gate on the learned arm plus a serving guard. It must be quoted as such, never as a
    bare model win.

## Entry points

- Guard: `src/policy/gnn/scheduler.py` (`_keepwarm_params`, `_keep_replicas_warm`). Tests:
  `tests/test_replica_keepwarm.py`.
- Gate: `scripts_cosim/fresh_topo_burst_v1_gate.py` phase `guard`. Reader: `scripts_cosim/replica_guard_v1_read.py`.

## Record (newest first)

- 2026-09-27 — **Gate read** (jobs 809178, 809299, 809311, phase `guard`, all at ec7fa71).
  - **Runs:** 576 of 584 summaries.
  - **Attachment:** [`guard_read.json`](replica_guard_v1/guard_read.json).
  - **Infrastructure, disclosed:**
    - Job 809178's tail lost 89 runs to "memory cap or crash" (rc 120 after the simulation completed) and
      6 to zero-byte summaries. Both are the `/home`-quota signature.
    - The cause was the guard's own spin logs: 4 runs on 9119 d3 wrote 4.7 GB each before the 2700 s
      timeout.
    - Those 95 runs were rerun unchanged, at `PAR=24` (809299) and `PAR=8` (809311). The spin logs were
      truncated.
  - **New guard failure, 9119 draw d3, all 4 seeds, reproduced on the rerun with no contention:**
    - Simulated time freezes at 32,291 s in the starved-client spin ("No compatible hardware available for
      dnn2 on nodes with connectivity to client_node8", retried forever), until the 2700 s timeout.
    - The twin runs the same cell in about 150 s (7.5–8.2 s elapsed).
    - Plausible mechanism, not verified: replicas the guard keeps alive hold the hardware that a `dnn2`
      replica reachable from client_node8 needs.
    - The drop rule removes 9119 from K2 and K3_draws. **That hides a guard failure, so it is stated here:**
      counted as a loss, the guard is worse than the twin on 9119.
  - **K1 `NOT-FIXED`** (9434):
    - Over the 4 draws × 4 seeds, the median is guard 10.69 s, twin 20.74 s, CD 9.05 s. The median improves
      48.5 %, just under the 50 % `PARTIAL` bar. The excess over CD is cut by 86 %.
    - Per draw (s1–s4, s), with CD's value:

      | draw | CD | guard | twin |
      |---|---|---|---|
      | d1 | 8.12 | 8.0 / 24.7 / 10.7 / 8.7 | 33.0 / 8.5 / 11.9 / 8.7 |
      | d2 | 9.12 | 25.6 / 19.5 / 24.3 / 21.3 | 30.3 / 20.7 / 25.8 / 20.7 |
      | d3 | 9.25 | 9.0 / 24.6 / 8.9 / 8.8 | 34.7 / 22.9 / 9.2 / 22.3 |
      | d4 | 8.99 | 8.1 / 22.9 / 10.6 / 8.6 | 50.5 / 8.7 / 12.0 / 8.8 |

    - On unperturbed w0 the guard is not better: 42.5 / 19.5 / 31.9 / 37.4 against the twin's
      51.6 / 19.6 / 35.2 / 20.6.
    - Seeds still collapse to about 20–25 s, and draw d2 collapses on every seed of both arms.
  - **K2 `TIES`:** guard vs CD −11.25 %, p = 0.24, 8/11 (9119 dropped).
    - Per topology (%): 9414 −14.2, 9420 −7.2, 9423 −12.0, 9434 +18.0, 9435 −13.7, 9444 −11.3, 9446 −8.3,
      9456 +4.0, 9461 +13.4, 9466 −17.3, 9469 −11.4.
    - Reported: the twin vs CD −10.91 %, p = 0.23, 9/12 (9434 +129 %, 9456 +8 %, 9461 +13 %).
  - **K3 (reported):**
    - Guard vs twin on the draws, the other topologies: −0.63 %, p = 0.19, 7/10.
    - Unperturbed w1 −1.49 % (10/12, p = 0.021), w2 −1.28 % (11/12, p = 0.034), w3 −0.98 % (10/11,
      p = 0.042).
    - Outside 9434 the guard is neutral to slightly helpful, except 9461 (w2 +7.9 %).
    - Guard moves: median 7.5 per 1,000 tasks (max 9.9).
  - **Reading:**
    - Keeping replicas warm attacks the right mechanism on 9434, but it is not the only cause there.
      Collapses of about 20 s survive on some seeds and on every seed of d2.
    - It does not move the statistic: K2 ties exactly as the twin does, because 9456 and 9461 trail CD
      with no replica collapse and 9434 remains +18 %.
    - Its interaction with the replica allocator creates a starved-client spin on another topology.
    - A serving guard is not the lever for a quotable ×1 win. The remaining candidates are training-side:
      replica idle age as a feature, or training on loaded bursty states. Neither is registered.

- 2026-09-27 — Registered.
