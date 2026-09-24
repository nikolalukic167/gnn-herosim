# lookahead_mp_v1 — can message passing earn its keep on partners that have not arrived?

**Status:** `ACTIVE` (2026-09-24) — P0 (live headroom gate) READ: **`GO-P1`**. P1–P4 registered in
outline, to be signed in full before P1 starts; a hand-coordination control (P0b) is proposed first.
Triggered by `joint_burst_v2`'s K4 tie and the 2026-09-24 reads recorded below.

**Standing answer (2026-09-24): there IS live headroom in pricing partners that have not
arrived, and the obvious hand guess does not reach it.** The question is the one CLAUDE.md asks —
does a graph-aware scheduler beat its pointwise twin — narrowed to the one mechanism that is
graph-specific and was never testable here: pricing unarrived partners. P0 (live, 16 + 16
environments, all 96 runs on `e62945b`): an oracle that knows where each unarrived partner ran
beats the rule **−6.56 % (C40, 15/16, p=0.0005) and −8.18 % (C80, 15/16, p=0.0005)** — `HEADROOM`
on both rungs, `GO-P1`. The two-step hand guess recovers **2 % / 7 %** of that (−0.11 % / −0.48 %,
not separated): 43 % of unarrived partners have no placed partner to guess from. **Not yet a GNN
result:** the oracle's gain may be *coordination* (every group converging on the same nodes) as
much as *prediction*, and a hand convention could capture coordination without learning — the
`mlp_t1x` pattern. P0b tests exactly that before P1's representation work is spent.

## Why this lineage exists — the measured facts it rests on (2026-09-24)

- **K4 is a competent-vs-competent tie, not a broken baseline.** Re-read of the
  `joint_burst_v2` gate summaries (`results/jb_v2/gate/`, 206 env×seed pairs): `jb2_mpoff`
  co-locates as well as `gnnedge0` — exchange 3.152 vs 3.067 s/task, both −40 % against the
  random floor (5.259) and below the hand rule (3.798). `gnnedge0` is −4.50 % faster (189/206),
  under the 5 % bar, and the residual sits in **queue (−6.45 %), not exchange (−2.83 %)**.
- **The served GNN cannot see a partner outside its batch.** `attach_live_prefix_block`
  (`src/policy/gnn/prefix_serving.py:447-460`) keeps in-batch peer pairs only and counts the
  rest (`prefix_peers_outside_batch`, median 13,916 per jb2 run vs 81,988 in-batch pairs). A
  batch-size-1 GNN sees no partner at all; message passing has never had an unarrived partner
  to propagate through.
- **The physics prices every partner, arrived or not.** Each task pays exchange to every
  partner at its input stage; an unplaced partner is waited for and then charged where it lands
  (`Platform._peer_rendezvous_events`, `_peer_exchange_time`). The rule prices unarrived
  partners at 0 (`pg_partners_unknown`).
- **The trace has the structure** (`drainable_f4000_n50000`): groups of 10, within-group density
  0.40 (never a clique, ~3.6 partners/task), arrival span median 13.8 s; 77 % of tasks have ≥ 1
  unarrived partner when they arrive; 45 % of unarrived partners already have a placed partner
  the arriving task does not see directly (pure two-step information).

## Plan

- **P0 — live headroom (this registration).** Per arrival, rule stack, the 16 + 16
  `unsaturated_edge_v1` environments (6 servers, 0.460393 arrivals/s):
  `peer_greedy_network` (rule, bar), `peer_greedy_lookahead_network` (an unarrived partner's
  node = the node of its bytes-heaviest already-known partner, a two-step hand estimate),
  `peer_greedy_oracle_network` (the node the partner actually ran on in the paired rule run).
  Both new arms price the predicted entries with the rule's own `_pg_exchange_seconds`, weight 1.0.
- **P1 — representation.** Serving and cache builder carry out-of-batch partners as peer nodes
  (placed: pinned; unarrived: unplaced), with a mandatory cache-vs-serve parity test.
- **P2 — corpus/label.** Per-arrival decisions from unburst live states, labelled by the
  realised-cost optimum given the placed prefix (brute force over ≤ 9 unarrived members);
  argmax-CE as a disclosed secondary (separates the two hypotheses in the
  `rollout_imitation_v1` 2026-09-21 addendum).
- **P3 — three arms, one representation and corpus.** `gnnedge0` (MP on), its MP-OFF twin, and
  `mlp_2hop` (pointwise, given the P0 two-step column). `experiments/` configs, W&B,
  determinism test first.
- **P4 — live gate, per arrival.** Primary: the GNN beats BOTH its twin AND `mlp_2hop` by ≥ 5 %,
  sign p < 0.05, paired n = 16 per rung. Registered mechanism check: the edge must sit in
  exchange and grow with unarrived partners; an edge in queue is not lookahead.

## Bars — P0 (signed 2026-09-24, before any P0 data)

Paired per (rung, env), primary metric `total_rtt`, chain 5 % / p < 0.05 / n = 16
(`scripts_cosim/lookahead_mp_v1_p0_read.py`):

- **H1 oracle vs rule:** `HEADROOM` (median ≤ −5 %, p < 0.05) / `ORACLE-SLOWER` (≥ +5 %,
  p < 0.05) / `NO-HEADROOM-FOR-A-5%-GAP`.
- **P0 verdict:** `GO-P1` if H1 reads `HEADROOM` on at least one rung, else `STOP` — and the stop
  is a live measurement (CLAUDE.md rule 6).
- **H2 two-step vs rule:** same thresholds; `recovered` = median Δ(two-step − rule) / median
  Δ(oracle − rule). `HAND-RECOVERS` if ≥ 0.80 on a `HEADROOM` rung — the room left for message
  passing over a pointwise two-step column is then ≤ 20 % of the headroom.
- Mechanism (reported, not gated): where the oracle's gain sits (queue / exchange / rendezvous).

Disclosed before data: (a) the oracle replays the RULE's trajectory; its own decisions shift
where partners land, so it is a trajectory-conditional bound, not a strict one; (b) weight 1.0
prices this task's own charge only — the partner's symmetric charge stays unpriced, as in the
rule; (c) in both new arms `pg_joined_partner` counts joining a known OR predicted partner node.
Odds stated at registration: ~15–25 % that P4 opens a gap surviving both controls.

## Entry points

- Policies: `src/policy/peer_greedy_network/scheduler.py` — `PeerGreedyLookaheadNetworkScheduler`,
  `PeerGreedyOracleNetworkScheduler` (registered in `src/placement/simulation.py`).
- Gate: `scripts_cosim/datalab/lookahead_mp_v1_p0.sbatch` (phase 1: rule + two-step, 64 tasks;
  phase 2: oracle, 32 tasks, `afterok` on phase 1).
- Reader: `scripts_cosim/lookahead_mp_v1_p0_read.py` → `simulation_data/lookahead_mp_v1/p0_read.json`.

## Record (newest first)

- 2026-09-24 — **P0 READ: `GO-P1` — live headroom on both rungs; the two-step hand guess does not
  reach it.** Jobs 804235 / 804285 (phase 1, rule + two-step) and 804330 (phase 2, oracle), 96/96
  summaries, every one on code `e62945b`. Transcript: `lookahead_mp_v1/p0_read_2026-09-24.txt`
  (+ `.json`). Paired per env, `total_rtt`:
  H1 oracle vs rule **−6.56 % C40 (15/16, p=0.0005), −8.18 % C80 (15/16, p=0.0005) → `HEADROOM`**;
  H2 two-step vs rule −0.11 % (9/16, p=0.80) / −0.48 % (8/16, p=1.0) → `NOT-SEPARATED`,
  recovered **0.02 / 0.07** (no `HAND-RECOVERS`); oracle vs two-step −6.92 % (16/16) / −7.58 % (16/16).
  Mechanism (oracle − rule, per task): queue −0.49 / −0.72 s, exchange −0.59 / −0.74 s,
  rendezvous +0.11 / +0.09 s — the gain is co-location that also shortens queues, not less waiting.
  Counters (C40 medians): partner joins rule 20,633 → two-step 26,199 → oracle 36,668; the
  two-step arm prices 50,477 of 88,895 unarrived references and is blind on 38,380 (43 %).
  **Reading, stated before any follow-up:** the registered bar is met, so the lineage continues
  (rule 6: the stop would have been live; so is the go). But the oracle knows the partner's
  realised node, which mixes *prediction* with *coordination* — each group converging on the
  nodes its members will use. A deployable HAND convention could capture coordination without
  any model, and would make P4's `mlp_2hop` control too weak. **Proposed P0b (cheap, live,
  before P1):** a hand arm that predicts an unarrived partner's node as the rule's own current
  argmin for that partner (its type, its client, the current queues) — "place it where the rule
  would place it now", no learning, no message passing. If P0b recovers ≥ 80 % of the oracle's
  headroom, lookahead is hand-buildable here and P1 does not start; if not, P1 starts with P0b
  as a fourth, stronger control in P4.

- 2026-09-24 — **Registered; P0 built and smoke-tested.** The two arms override only
  `_pg_peer_nodes`, so the rule's scoring is reused unchanged. Smoke run on the committed tree
  (3,000-event prefix, `cc40s9001` w0, warm-up transient — a mechanics check, not evidence):
  all three arms complete; of 5,318 unarrived-partner references the oracle prices 5,318
  (0 blind), the two-step arm 3,020 (2,298 blind). Reader validated on a known fixture
  (HEADROOM / GO-P1 / HAND-RECOVERS fire as registered; a missing arm fails loud).
