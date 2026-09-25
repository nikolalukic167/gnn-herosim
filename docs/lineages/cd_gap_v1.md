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

- 2026-09-25 — **A2/A3 read (live): the label does not matter.** Datalab job 806624 at 934c324
  (`src/` identical to the witnessed 00dae37), 192/192 runs, no failures.
  - **A2:** CD-imitator vs CD **+13.58 %**, 0/11, p = 0.001 (`CD-FASTER`; 9466 dropped). A1 fired
    `CANNOT-FIT-CD`, so `OFFLINE-LIVE-GAP-CONFIRMED` does not apply.
  - **A3:** CD-imitator vs jb2 `gnnedge0` (seeds 1–4 paired) +0.58 %, p = 0.52, 5/12
    (`NOT-SEPARATED`). Queue 3.97 vs 4.13 s.
  - **Reading:** trained on the sweep optimum or on CD's own plans, the same architecture serves the
    same arm, offline (A1) and live. The bottleneck is what the representation can express, not the
    label. [Read](cd_gap_v1/a2_read.json).

- 2026-09-25 — **A1 read (offline): `CANNOT-FIT-CD`.** The CD labels cover 2,516 groups: every CD plan
  matched a sweep row, 527 are any-of-k ties, and CD is at the sweep optimum on 688. Four
  CD-imitator seeds, the jb2 `gnnedge0` recipe with the train label overridden and selection on
  true-sweep val regret, give on 480 held-out groups:

  | | median regret | mean regret |
  |---|---|---|
  | CD-imitator (median over seeds) | 11.1 % | 47.5 % |
  | CD | 8.2 % | 54.5 % |
  | `gnnedge0` | 11.7 % | 54.6 % |

  - **Exact agreement with CD's plan:** 17.8 % (per seed 17.3–18.1 %).
  - **Paired medians:** 0.0 pp for both learned arms against CD.
  - **Reading:** trained directly on CD's decisions, the architecture does not reproduce them. It
    lands where the sweep-trained model does, which agrees with D6: the representation cannot express
    what CD's score uses. [Read](cd_gap_v1/a1_read.json).
  - **Training facts** (datalab jobs 806568 test, 806570 train, 806571 eval):
    - `tests/test_trainer_determinism.py` 17/17 passed, no skips.
    - The override replaced the train label on all 1,629 train graphs (364 any-of-k) in every seed.
    - Early stop on true-sweep val regret selected epochs 14 / 75 / 47 / 91 (seeds 1–4).
    - **Caveat:** checkpoints are selected for sweep regret, not CD agreement, so the 17.8 % is
      agreement at regret-selected epochs. The `-final` checkpoints are kept but unread.
  - The A2 live gate (4 seeds × the fresh study) is running (datalab job 806624, worktree at
    934c324; `src/` is identical to 00dae37).
- 2026-09-25 — **D6 read: `SCORE-EXPLAINS`.**
  - **D6-1:** self-revision on the model's own score vs `gnnedge0` −1.42 %, p = 0.092, 9/12
    (`NOT-SEPARATED`).
  - **D6-2:** vs CD +8.98 %, 0/11 (`CD-FASTER`).
  - **D6-3:** from the same GNN seed, self-revision recovers a **median 8.9 %** (10 topologies) of
    the gain CD's refine passes get.
  - **Reading:** decode order and the lack of revision are not the gap; the learned score is.
    Revising with it barely moves anything useful, while revising with CD's seconds-based score
    closes the whole gap and more (D5). [Read](cd_gap_v1/d6_read.json).
- 2026-09-25 — **D5 read: the GNN seed under CD refinement beats CD (direction only).** Datalab,
  00dae37, clean; venue witness to the digit (CD 440055.157, `gnnedge0` s1 480523.270). 9423 (9 cdapply
  timeouts) and 9466 are dropped by the failure rule, leaving 10 topologies.

  | read | median | p | faster |
  |---|---|---|---|
  | **D5b-1** GNN-seeded CD vs CD | **−3.83 %** | 0.002 | **10/10** |
  | D5b-2 GNN-seeded CD vs `gnnedge0` | −14.78 % | 0.002 | 10/10 |
  | D5b-3 GNN-seeded vs MP-OFF-seeded CD (same seed) | −0.88 % | 0.037 | 8/10 |
  | disclosed: MP-OFF-seeded CD vs CD | −2.56 % | 0.002 | 10/10 |

  - **Queue:** GNN-seeded CD 2.74 s vs CD 2.99 s.
  - **D5b-1 label:** `SEED-REACHES-CD` fires as "seeded faster", **direction only** (|median| < 5 %).
  - **Reading:** under identical refinement, the learned seed beats the hand seed on every
    topology. About two-thirds of that is not message passing (the MP-OFF seed gets −2.6 %); MP adds
    a small, consistent further edge. This is a learned-seed win, not an MP win.
  - **D5a (shadow, 144 runs, every one identical to unshadowed `gnnedge0` to the digit):** on the
    GNN's own live states CD would change 61.1 % of batches and 16.7 % of tasks (per run 12.2–22.0 %).
    92.7 % of the moves change the node, and 52.9 % un-stack a batch-mate pile.
    [Read](cd_gap_v1/d5_read.json).

- 2026-09-25 — **Venue change for B, D5 and D6 (procedure only; no gate datum read).** The local
  runs are discarded unread, and each phase reruns in full on datalab CPU nodes at **00dae37**. The
  reasons:
  - B holds slots for 30 min on its spinning cells (54/1,440 done in ~1 h).
  - Two local B runs crashed from an editing race (the live tree was mid-edit; `NameError` in the
    scheduler), and several recorded a dirty code state.
  - Spin logs filled the shared root disk; a log watchdog now truncates them.

  Datalab reproduces these arms to the digit (fresh_topo parity, 4/4). A venue witness at 00dae37
  (CD and `gnnedge0` s1 on 9119 w1 against the local totals) must match before any phase runs. The
  D5 smoke cell stays disclosed above. The driver gained `--no-scope`: no per-run cgroup on SLURM,
  where the job allocation caps memory, with the same 1,800 s timeout.

- 2026-09-25 — **Amendment D6 (decode order vs score), signed before any D6 run.**
  - **What D5's smoke says:** CD's refine passes, started from the GNN's own plan, move mostly nodes.
    The GNN's masked_topo decode is one pass in id order and never revisits a task, whereas CD's edge
    is revision.
  - **Two causes to separate:** no revision (decode order), or the score (seconds of drain and
    committed service vs the model's counts).
  - **Design.** Knob `GNN_PREFIX_SELF_REFINE=3` (68b6ecd): after the decode, up to 3 passes of
    coordinate descent on the **model's own score**. Each task is re-scored with every other
    batch-mate committed where the plan puts it and moved to the model's argmax, with the decoder's
    tie rule; unmasked decode only (fail loud otherwise). Arm `gnnedge0_selfref` × 13 on the fresh
    study.

  | read | contrast | fires as |
  |---|---|---|
  | **D6-1** | `gnnedge0_selfref` vs `gnnedge0` (same seed) | `SELF-REVISION-HELPS` (≤ −5 %, p < 0.05) / direction only / `NOT-SEPARATED` / `SELF-REVISION-HURTS` |
  | D6-2 | `gnnedge0_selfref` vs CD | reported |
  | **D6-3** | share of D5's CD-refine gain recovered by self-revision, per topology: median of (gnnedge0 − selfref) / (gnnedge0 − cdapply), elapsed | `REVISION-EXPLAINS` ≥ 50 % / `SCORE-EXPLAINS` < 25 % / `MIXED` |

  **Expectations:** D6-1 helps or direction only 50 %. D6-3 `SCORE-EXPLAINS` 50 %, `MIXED` 30 %,
  `REVISION-EXPLAINS` 20 %.

- 2026-09-25 — **Amendment D5-MP, signed after the D5 smoke cell and before the D5 gate.**
  - **Disclosed smoke (9119 w1, seed 1, 4883c27):**
    - shadow reproduces `gnnedge0` `total_rtt` to the digit (480523.270);
    - CD would change 57.8 % of the GNN's batches, moving 15.6 % of tasks; 94 % of the moves change
      the node and 57 % un-stack a pile;
    - `gnnedge0_cdapply` elapsed 8.364 against CD 8.801 and `gnnedge0` 9.610 (one cell: a direction
      at most).
  - **Why this amendment:** `gnnedge0_cdapply` vs CD differs only in the seed: the GNN's decoded plan
    versus the 1-pass greedy's, under identical refinement. A seed win is a learned-seed win, not a
    message-passing win (`gnn_seeded_cd_mp_ablation_v1`). So the gate adds `mpoff_cdapply` × 13.
  - **Added read D5b-3:** `gnnedge0_cdapply` vs `mpoff_cdapply` (same seed), labelled as
    `fresh_topo_burst_v1` F1. Its expectation: not separated 55 %.

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
