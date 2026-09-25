# cd_gap_v1 — why does the burst-seat GNN lose to the CD greedy, and can imitating CD or changing load close it?

**Status:** `REGISTERED` (2026-09-25). Every bar below was signed before its data.

**Parents:** [`fresh_topo_burst_v1`](fresh_topo_burst_v1.md) (on 11 fresh topologies `gnnedge0` is
+11.4 % behind CD, 0/11), [`joint_burst_v2`](joint_burst_v2.md) (the checkpoints and the corpus; label =
the group sweep optimum, which offline strictly dominates CD).

## Why

The one learned arm that reaches the self-predict rule in the burst seat still loses to the CD greedy
by ~11–12 % everywhere. **The decomposition of the fresh gate (medians over runs, per task) puts the
whole gap in queue, not co-location:**

| arm | elapsed | queue | exchange |
|---|---|---|---|
| CD greedy | 7.62 s | 2.89 s | 4.19 s |
| `gnnedge0` | 8.64 s | 4.13 s | 4.23 s |
| self-predict | 8.54 s | 4.22 s | 4.14 s |
| MP-OFF | 8.97 s | 4.45 s | 4.35 s |

About 10 % of partner links fall outside the served batch (`prefix_peers_outside_batch` 18,050 vs
79,928 in-batch pairs per `gnnedge0` run). The record's leads for the gap are (a) the model does not
reach its label (v1 `gnnedge0` offline regret 27.5 s vs CD 11.2 s on 48 groups; never measured for
v2), (b) it is blind to partners outside its batch, (c) the offline→live gap: training states are
the 1-pass rule's, labels are group-local. None of them has been isolated.

## Design

Nothing here is a new model until A. Probes run on existing checkpoints and code, ordered cheapest-first.

- **D0, fit vs gap (offline, orders the work).** On the 480 `joint_burst_v2` held-out groups (cells
  9222–9224, never trained on): per group, regret % = 100 × (rtt − sweep optimum) / optimum, for CD and
  the 1-pass greedy (`joint_burst_v1_offline_rule_regret.py`, the engine replay) and for the 13 jb2
  `gnnedge0` and 13 jb2 MP-OFF checkpoints (`eval_route_b_stage2_arm.py`, alpha key `inf`, the decode
  `peer_affinity_live_serve_check.py` proved identical to serving). Learned arm per group = median over
  checkpoints. Unit for direction = cell × window; groups are described, not treated as independent.
- **D1, out-of-batch information (live).** `peer_greedy_network_cd` with `HEROSIM_PG_BATCH_BLIND=1`:
  identical except a partner outside the current batch is never priced, exactly what the served GNN
  sees. Paired against CD on the `fresh_topo_burst_v1` study (11 topologies × 4 windows), statistic
  as that lineage's (per topology median over windows, exact Wilcoxon over topologies).
- **A, CD imitation (trained, live-gated).** jb2 `gnnedge0` recipe, same corpus and split, label =
  CD's plan on each snapshot (via the engine replay) instead of the sweep optimum. 4 seeds. Offline:
  regret vs the sweep optimum and plan agreement with CD on held-out. Live: on the fresh study, vs CD
  and vs jb2 `gnnedge0`. Its ceiling is a tie with CD; it is a diagnostic of the offline→live gap
  (does a model that reproduces CD offline reproduce it live?), not a route to beat CD.
- **B, burst load ladder (live).** The fresh study at a lighter admissible rate (arrival factor
  ×2 slower than the current burst workloads) with every policy time constant scaled by the same
  factor (hard-stop rule, `drainable_regime_v1`). Arms: CD, self-predict, jb2 `gnnedge0` × 13, jb2
  MP-OFF × 13. A topology whose reactive screen fails at that rate is dropped by name.

## Bars (signed 2026-09-25, before any D0/D1/A/B datum)

| read | contrast | fires as |
|---|---|---|
| **D0** | `gnnedge0` vs CD, offline regret % on held-out | `FIT-GAP` if `gnnedge0`'s median regret ≥ CD's (it does not reach CD even offline); `OFFLINE-LIVE-GAP` if it is below CD's by ≥ 1 pp median and in every cell × window; else `MIXED` |
| D0b | `gnnedge0` vs MP-OFF, offline | reported (does MP's live direction exist offline?) |
| **D1** | blind-CD vs CD, live | `OUT-OF-BATCH-MATTERS` (≤ −5 % for CD, p < 0.05) / `(direction only)` (p < 0.05, \|median\| < 5 %) / `NOT-SEPARATED` |
| **A1** | CD-imitator vs CD, offline | `IMITATES-CD` if its median regret is within 1 pp of CD's; else `CANNOT-FIT-CD` |
| **A2** | CD-imitator vs CD, live (fresh study) | as D1's labels; with A1 = `IMITATES-CD`, a live loss ≥ 5 % is `OFFLINE-LIVE-GAP-CONFIRMED` |
| A3 | CD-imitator vs jb2 `gnnedge0`, live | reported |
| **B1** | `gnnedge0` vs CD at the lighter rate | as D1's labels |
| B2 | `gnnedge0` vs MP-OFF, `gnnedge0` vs self-predict at the lighter rate | reported, with the rate named |

**Expectations, written before data.** D0: `FIT-GAP` 60 % (v1's 27.5 vs 11.2 s), `MIXED` 25 %,
`OFFLINE-LIVE-GAP` 15 %. D1: `NOT-SEPARATED` or direction only 75 % (only ~10 % of partner links are
outside and CD's lead is queue, not exchange). A1: `IMITATES-CD` 45 %. A2 given A1: CD faster ≥ 5 %
40 %. B1: CD still faster 85 %.

**Rule 6.** D0 is offline and only orders the work; it closes nothing. D1, A and B are live.

## Entry points

- D0: `scripts_cosim/joint_burst_v1_offline_rule_regret.py`, `scripts_cosim/eval_route_b_stage2_arm.py`.
- D1/A/B live: `scripts_cosim/fresh_topo_burst_v1_gate.py` (its study selection and failure rule).

## Record (newest first)

- 2026-09-25 — **D4 gate WITHDRAWN (premise false), and amendment D5 signed before any refine run.**
  - **Why D4 is withdrawn:** the smoke test on 9119 w1 (both arms) shows the live candidate set is
    already *smaller* than the corpus slate. There are ~16 candidate slots per 10-task batch, about
    1.6 per task, against 2.1 in training (CD: 87,731 slots over 5,335 slated batches), and the slate
    rule removes 0.5 % of them. The slated arms reproduce their unslated elapsed within 0.01 s (CD
    8.793 vs 8.801, `gnnedge0` s1 9.603 vs 9.610). The candidate set is not the gap; the gate would
    compare near-identical arms. The two smoke summaries are kept in the scratch D4 directory, not read
    as a gate.
  - **D5 design.** Knob `GNN_CD_REFINE` (098dcd6). After the GNN decodes a batch, CD's own refine
    passes (`_pg_batch_pass(refine=True)`, bound, not copied, up to 3) run from the GNN's plan on the
    same live state, with CD's books seeded from that plan. Every move is counted: node change or
    platform only, and whether it un-stacks a batch-mate pile.
    - `shadow` serves the GNN's plan and only counts: what CD would change on the GNN's own states,
      on-policy.
    - `apply` serves the refinement: a GNN-seeded CD arm.

  | read | contrast | fires as |
  |---|---|---|
  | D5a (descriptive) | `gnnedge0_cdshadow`, seeds 1–3, fresh study | share of batches and tasks CD would change, and the move types. Shadow must reproduce `gnnedge0`'s `total_rtt` to the digit |
  | **D5b-1** | `gnnedge0_cdapply` × 13 vs CD | `SEED-REACHES-CD` (not separated, or seeded faster) / `CD-FASTER` (≥ 5 %, p < 0.05) / direction only |
  | D5b-2 | `gnnedge0_cdapply` vs `gnnedge0` (same seed) | reported: what CD's moves are worth on the GNN's plan |

  **Expectations:** D5b-1 `SEED-REACHES-CD` 60 % (`gnn_seeded_cd_v1` in the dynamic seat: seeded
  within +1.7 % of the hand start). D5a: CD changes ≥ 30 % of the GNN's batches 70 %.

- 2026-09-25 — **Amendment D4 (candidate set), signed before any slate run.** Code trace
  (`make_warm_corpus.choose_candidates`): each training/eval group offers a **seeded random
  per-type subset** of the live replicas (≤ 20,000 plans, about 2.1 candidates per task), chosen
  without regard to quality. The offline eval decodes only over it, and so does D0's CD replay. Live,
  both the GNN and CD choose among every reachable replica (6–10 per task by the corpus docstring).
  D0's offline tie was therefore measured under the slate; the live gap is measured without it.
  Knob `GNN_SERVE_CORPUS_SLATE=1` (cc905d7): each batch is decided over the slate the corpus rule
  would draw for it — the same function and arguments, seeded per batch — via a view of the replica
  table; enqueueing and the physics see every replica. A batch the rule rejects is decided
  unrestricted and counted. Knob-off identity at cc905d7: `gnnedge0` s1 and CD on 9119 w1 match
  the fresh gate to the digit.
  Arms on the fresh study (11 topologies + 9466, dropped by the fresh rule): `cd_slate`,
  `gnnedge0_slate` × 13.

  | read | contrast | fires as |
  |---|---|---|
  | **D4-1 (primary)** | `gnnedge0_slate` vs `cd_slate` | `CANDIDATE-SET-IS-THE-GAP` (not separated, or `gnnedge0` faster) / `PARTIAL` (CD faster, direction only) / `GAP-PERSISTS-UNDER-SLATE` (CD faster ≥ 5 %, p < 0.05) |
  | D4-2 | `gnnedge0_slate` vs `gnnedge0` (same seed) | reported |
  | D4-3 | `cd_slate` vs CD | reported |

  **Expectations:** D4-1 `GAP-PERSISTS` 50 %, `PARTIAL` 30 %, `CANDIDATE-SET-IS-THE-GAP` 20 %.
  **Consequence:** `CANDIDATE-SET-IS-THE-GAP` means the fix is the corpus, i.e. training on the
  full live candidate set. `GAP-PERSISTS` puts the cause in live states and dynamics (the
  standing-load replay is a synthetic busy period, not real queued tasks), which A and an on-policy
  corpus test.

- 2026-09-25 — **D1 read: `OUT-OF-BATCH-MATTERS (direction only)`.** Blind-CD vs CD +3.41 %
  (median over 11 topologies, 0/11 faster, p = 0.001; per topology +2.0 to +6.7 %). Queue 2.91 →
  3.09 s; exchange unchanged (4.18 → 4.21 s). 9466 dropped (its CD w3 cell timed out in the fresh gate).
  Blind runs are at 5cc38d9 and 805bc79, which have the same simulator code; CD cells are the fresh
  gate's at 1ae90af, reproduced to the digit at 5cc38d9 with the knob off (9119 w1). About 3.4 of
  `gnnedge0`'s ~11.4 % gap to CD is information it structurally lacks. [Read](cd_gap_v1/d1_read.json).
- 2026-09-25 — **D3 read: `NO-EXTERNALITY`.** Paired vs CD over 480 held-out groups (medians):
  - label: sum −4.76 s (lower on 100 %), backlog **−3.55 s** (higher on only 6.9 %);
  - `gnnedge0`: sum 0.0 s, backlog −1.53 s (higher on 28.8 %);
  - 1-pass: sum +5.50 s, backlog +0.04 s.

  Every plan was matched to a sweep row. CD's `total_rtt` matched several tied rows on 78 groups and
  the 1-pass greedy's on 75; there the metrics are the mean over the tied rows.
  Offline, on the 1-pass rule's states, the label and the GNN leave *less* busy time behind than CD.
  The objective is not what loses the queue live. [Read](cd_gap_v1/d3_externality_summary.json).
- 2026-09-25 — **D2 gate WITHDRAWN before its data**, signed with the reason. The witness cell shows
  the probe cannot answer its question: the flag re-assigns 2 % of tasks because a node rarely holds
  two valid replicas of one type, so "spread vs no spread" would compare two nearly identical arms.
  The stacking measurements stay recorded in the D2 amendment below. The mechanism they point to
  (CD splits co-located batch-mates across siblings about twice as often) is what CD's refine passes
  do, not a serving tie-break.

- 2026-09-25 — **Amendment D3 (externality), signed before D3 is computed.** The witness run showed
  the sibling-spread flag almost never fires (1,039 of 50,000 tasks moved; stacking 0.80 → 0.78;
  one cell, −1.3 %). Replicas are per task type, and a node rarely holds two valid replicas of one
  type at decode time. Every same-node spread both arms make is between same-type siblings, over
  the same 17 platforms. CD splits 38–41 % of same-node pairs onto siblings, `gnnedge0` 20–25 %,
  the 1-pass greedy 24 %. CD's queue lead comes from its refine passes (the 1-pass greedy has the
  same seconds-based information and queues 4.5 s).
  **Hypothesis:** the label, and a GNN that imitates it, minimises the group's own summed elapsed
  while leaving more busy time on its platforms for the groups that follow. The offline score never
  charges that externality; live, it is the queue.
  **D3 (offline, orders the work):** on the 480 held-out groups, from each plan's sweep row
  (`task_times`), compute sum, makespan and **backlog** (Σ over used platforms of last done − first
  dispatched) for the label, CD, the 1-pass greedy and `gnnedge0` (median over 13 checkpoints).
  `EXTERNALITY` iff the label's backlog exceeds CD's (paired median > 0, and higher on > 50 % of
  groups) while its sum is lower. Otherwise `NO-EXTERNALITY`. Reported the same way for `gnnedge0`.
  Expectation: `EXTERNALITY` 55 %.

- 2026-09-25 — **Amendment D2 (sibling stacking), signed before any spread run.** What prompted it
  (data already read, disclosed in full):
  - **D0 read, registered statistic: `FIT-GAP`** (median regret: `gnnedge0` 11.7 %, CD 8.2 %,
    1-pass 28.5 %, MP-OFF 11.4 %). The per-group paired median of `gnnedge0` minus CD is **0.0 pp**
    (they tie on most groups) and the mean regret is equal (54.6 vs 54.5 %). MP vs MP-OFF offline:
    paired median 0.0 pp.
  - **What the served model sees** (code trace): one per-platform queue count and the running
    task's remaining time, frozen at batch start (`partial_state_edges.py`: the encoding is computed
    once). The in-batch load signal is per node and per type (`reduced_features.partial_state_columns`
    cols 0–3). Nothing distinguishes sibling platforms. A xavier node has 8 identical `xavierCpu`
    platforms; each platform is one FIFO queue. CD charges in-batch commitments per platform, in
    seconds.
  - **Stacking, measured** (share of same-node batch pairs that share one platform):
    - Offline, 480 held-out groups: sweep optimum 0.62, `gnnedge0` 0.68, MP-OFF 0.68.
    - Live, two witness cells (9119 w1, 9444 w2): `gnnedge0` 0.80 / 0.75, MP-OFF 0.81, CD 0.62 / 0.59,
      1-pass 0.76, self-predict 0.84.
    - A task stacked behind a batch-mate queues ~7–8 s, against ~1 s for the first task on a platform.
    - Co-location is equal: same-node pairs per batch are ~10 for `gnnedge0` and ~11 for CD.

  **D2 design.** Serving flag `GNN_PREFIX_SIBLING_SPREAD=1`: keep every decoded node, and re-pick the
  platform among that node's valid same-type replicas. The key, in order: fewest batch-mates already on
  it, then shortest queue, then the decoder's own pick. Node choice and all prefix features are unchanged.
  Arms `gnnedge0_spread` and `mpoff_spread` × 13 seeds on the fresh study, paired with that study's
  cells. Code is identical with the flag off, witnessed on 9119 w1 s1.

  | read | contrast | fires as |
  |---|---|---|
  | **D2-1 (primary)** | `gnnedge0_spread` vs `gnnedge0` (same seed) | `SPREAD-HELPS` (≤ −5 %, p < 0.05) / `(direction only)` / `NOT-SEPARATED` / `SPREAD-HURTS` |
  | **D2-2** | `gnnedge0_spread` vs CD | `GNN-BEATS-CD` / `CD-FASTER` / `(direction only)` / `NOT-SEPARATED` |
  | D2-3 | `gnnedge0_spread` vs `mpoff_spread` (same seed) | as fresh_topo F1 |
  | D2-4 | `gnnedge0_spread` vs self-predict | as fresh_topo F2 |

  **Expectations:** D2-1 `SPREAD-HELPS` 55 %, direction only 30 %. D2-2 CD still faster 60 %, not
  separated 30 %, GNN faster 10 %.

  **Consequence, signed:** if D2-1 fires, the offline→live gap has a named serving cause: the model
  cannot see sibling platforms within a batch. The deployable fix is then a representation change
  (a per-platform in-batch commitment feature), not the hand spread; the hand spread is the probe.

- 2026-09-25 — **Registered.**
