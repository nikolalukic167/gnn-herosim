# r1_attribution_v1 — train on R1 + WF1 and run the attribution battery

**Status:** `REGISTERED` (no runs). Depends on: `physics_audit_v1` (R1, including I11 pass),
`workload_fix_v1` (WF1), `load_recalibration_v1` (rungs). Plan: [`reference_physics_programme.md`](reference_physics_programme.md).
Revision 2026-10-08 (before any run): trains on WF1 instead of today's workload; candidate pruning declared.

## Questions
- Q1 (performance): does any learned arm, trained on R1 + WF1, beat CD there?
- Q2 (attribution): where learned arms differ, is it message passing, relational features, or set context?
- Q3 (hybrid): does a learned seed improve CD, and is any gain from message passing?

## Labels (fixed now)
Brute-force optima from co-simulation under R1 + WF1 (valid only after I11 passes), same capture protocol and batch
sizes as the original corpus (batches of ≤ 10 tasks). One corpus, shared by every learned arm; size matched to the
original (~5,000 batches) and recorded. Train/validation split by topology; the 19 test topologies are excluded.

**Candidate pruning (declared before labels exist).** Eager scale-out can raise candidates per task beyond what brute
force can enumerate. If a captured batch has more than 5 candidates for any task:
- keep each task's top 5 candidates ranked by that task's exact standalone cost (cost model shared by CD and the
  simulator), and enumerate the pruned plan space;
- if a batch still exceeds 100,000 plans, split it into sub-batches of ≤ 4 tasks.
Record the fraction of pruned batches and of sub-batched batches.

**Pruning ablation (part of this node).** On 300 captured batches small enough to enumerate unpruned, report the
optimality gap of pruned vs unpruned labels; and at serving time, evaluate the best learned arm with and without the
pruning rule on 6 test topologies. If the label gap exceeds 2 % (median), the node is paused and the rule amended
before training.

## Arms
| Arm | What it isolates |
|---|---|
| CD, locality-first, one-pass greedy, self-predict | classical reference (Knative as context) |
| GNN-eng | engineered relational features + message passing (retrained `so1load` recipe) |
| Twin-eng | as GNN-eng, message passing off |
| MLP-same | same inputs, no graph |
| GNN-raw / Twin-raw | no hand-built relational features, MP on / off |
| GNN-eng-physMP | GNN-eng with physics columns live inside message passing (`NEAR_RTT_MP_BIPARTITE_EDGE_ATTR_ZERO=0`) |
| Set-transformer | same inputs, all-pairs attention over the batch, no graph edges |
| CD←GNN, CD←Twin, CD←random | CD started from each plan (random start is the control for the "worse basin" reading) |

**Fair-comparison budget (after Errica et al., ICLR'20):** every learned arm gets the same hyperparameter search
budget (6-configuration grid, same epoch cap, same early stopping, 3 training seeds). The configuration per arm is
chosen on validation topologies only, and the "best learned arm" for the primary family is declared from validation
before any test read.

## Cells
19 test topologies × 3 recalibrated rungs × seeds as in `transfer_physics_v1`.

## Primary family (Holm, 3 tests)
Best learned arm vs CD at each rung.

## Secondary families (Holm within each)
- S1 attribution: GNN-eng vs Twin-eng; GNN-raw vs Twin-raw; GNN-eng vs MLP-same; GNN-eng vs Set-transformer;
  GNN-eng-physMP vs GNN-eng (5 contrasts × 3 rungs).
- S2 hybrid: CD←GNN vs CD; CD←GNN vs CD←Twin; CD←random vs CD (3 × 3).

## Predictions
1. Q1: no learned arm wins against CD at any rung.
2. S1: GNN-raw beats Twin-raw (old 8–24 % pattern, smaller on R1); GNN-eng ties Twin-eng.
3. S1: GNN-eng beats MLP-same at moderate and heavy rungs, by less than on old physics.
4. S1: Set-transformer ties GNN-eng.
5. S1: physics-live MP within ±3 % of GNN-eng.
6. S2: CD←GNN does not beat CD; CD←random ≤ CD←GNN.
7. Decision time: CD ≥ 10× faster per task than any learned arm.

## Outcomes
- Q1 win anywhere → registered result; checked again in `call_graph_pairing_v1`.
- Q1 no win → stopping rule applies; S1 and S2 form the paper's attribution section.
- If prediction 2 fails, "message passing substitutes for feature engineering" is old-physics only and labelled so.

## Cost
Corpus capture and co-simulation under R1 + WF1 (largest item); about 126 trainings (7 trained models × 3 seeds ×
6 configs); evaluation ≈ 15 arms (5 classical incl. Knative, 7 learned, 3 seeded-CD) × 19 × 3 rungs × seeds.
