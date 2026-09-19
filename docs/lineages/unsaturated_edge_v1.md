# unsaturated_edge_v1 — the −20.8 % headline, at the power it was never read at

**Status:** `REGISTERED` (2026-09-19) — bars, reader, 14 tests and this expectation committed
before any learned arm runs. The saturation screen (reactive only) is running.

**Question.** The programme's standing headline is `gnnedge0` (the bipartite, mean-aggregating,
attribute-zeroed arm) beating reactive Knative by **−20.8 % (15/16)** at **6 servers / 40
clients**, unsaturated (`bipartite_edge_v1` → `best_arm_v1`, cells `cc40s9001/9002/9003/9005`).
It was read on **four cells, all on arrival window w0**. `unsaturated_scale_v2` then showed,
on the 80-server rung, that a 4-environment read on this apparatus keeps every **direction** and
inflates every **magnitude 4–10×**, because w0 is the burstiest of the four windows the trace
offers (`gnnedge0` −6.40 % on w0 against −1.06 / −0.65 / −0.18 % on w1–w3). That caution
applies to the headline verbatim. **Does the headline hold on 16 environments?**

**Design — v2's, at the 6-server client rungs, nothing else changed.** An environment is a
(topology, arrival window) pair. Per rung: 12 candidate topologies (9001/9002/9003/9005 +
9101–9108 minted here; 9004 excluded, it hangs on every policy) × 4 windows (w0 =
`drainable_f4000_n50000`, the rung's own workload; w1–w3 minted by v2, rate-matched to w0 at
0.4604 arrivals/s). The screen runs reactive alone on all 96 candidate environments; the study
takes the **4 lowest-numbered topologies admissible (queue share ≤ 0.80) on all 4 windows**, per
rung — a rule no arm can steer.

- **Primary rung, 40 clients** — where the headline lives. Arms: `gnnedge0` (be1670), its
  corpus-matched MP-OFF twin `mpoff` (1670), `peeronly` (1670), `gnn` (1670, the GIN-sum
  bipartite arm). 16 checkpoints each.
- **Secondary rung, 80 clients** — `gnnedge0` alone (w0 read −13.5 % there), to read the shape
  of the margin across load.
- **Baselines** on every study environment: `knative_network` (the reference, from the screen),
  `knative_network_ect`, `random_network`. Random is a registered comparison (E4), not a
  decoration: v2 measured it +8.4 % behind reactive with a bimodal tail to +782 %.

**Bars — v2's, unchanged** (`scripts_cosim/unsaturated_edge_v1_read.py`, tests alongside):
unit = checkpoint; statistic = median over the 16 environments of the checkpoint's relative %
against the reference **on the same environment**; two-sided signed-rank against zero;
|median| ≥ 5 %, p < 0.05, n ≥ 16. E5 design bar: worst-arm sd ≤ 10 pp, read before E7.

| read | what | fires as |
|---|---|---|
| E1 | each arm vs reactive, per rung | `ARM-BEATS-REACTIVE-AT-THE-EDGE-RUNG` / `REACTIVE-FASTER…` / `NOT-SEPARATED` |
| E2 | `gnnedge0` vs `mpoff`, paired on environment and checkpoint (C40) | `GRAPH-FASTER…` / `POINTWISE-FASTER…` |
| E3 | `gnn` vs `gnnedge0` (C40) | `SUM-COSTS…` / `SUM-HELPS…` / `SUM-NOT-SEPARATED…` |
| E4 | each arm vs `random_network`, per rung | `ARM-BEATS-RANDOM…` / `RANDOM-FASTER…` / `RANDOM-NOT-SEPARATED…` |
| E5 | design power, per rung | `POWER-DELIVERED` / `POWER-NOT-DELIVERED` |
| E6 | sign per window, per arm (descriptive, qualifies E1) | `SIGN-CONSISTENT…` / `SIGN-FLIPS…` |
| E7 | the composite on `gnnedge0` | `HEADLINE-HOLDS-AT-POWER-ON-BOTH-RUNGS` / `…ON-THE-PRIMARY-RUNG` / `HEADLINE-SHRINKS-BELOW-THE-BAR-AT-POWER` |

E7's BELOW-BAR is `interpretable` only if E5 delivered at C40 — an under-powered negative is
never called a tie. A positive stands either way.

**Registered expectation (signed 2026-09-19, before any learned arm).** *Roughly even odds
that E1 fires for `gnnedge0` at C40.* The mechanism-based estimate: on w0 the arm halves
reactive's ~22 s queue and pays a **6.8 s batch-assembly tax** reactive never pays, net −20 %.
On w1–w3 reactive's queue is smaller (v2's share 0.53–0.66 → 0.41–0.48; the first screen cells
here read 0.55 / 0.47 / 0.47), so the same halving saves ~5–7 s against the same 6.8 s tax:
net −5 to +5 % per window. Pooled median therefore lands in **−3 to −10 %**, p < 0.05 likely,
|median| ≥ 5 % uncertain; E6 spread > 10 pp. E4 fires at both rungs. E2 fires (the twin pays
the same tax, so the model-class contrast is not tax-limited; best_arm_v1's −14.76 % will
shrink). E3 `SUM-NOT-SEPARATED` (bipartite_aggr_v1's prediction at 3.55 candidates/task).
C80 `gnnedge0`: `NOT-SEPARATED`.

**Consequences, signed in advance.**
- `HEADLINE-HOLDS…` (either form): the standing answer's −20.8 % is **replaced** by the
  16-environment number, quoted with E6; that is the paper's headline.
- `…BELOW-THE-BAR`, interpretable: the −20.8 % is **retired as a w0 number**; CLAUDE.md says
  so; and the batch window is registered as the next lineage — every learned arm's margin is
  tax-limited (E1 median ≈ queue saved − 6.8 s) and the window is the one untested lever that
  needs no retraining. `drainable_serving_config_v1` swept it only at the 20-client rung,
  where every arm loses by ≥ 70 %.
- `…BELOW-THE-BAR`, not interpretable: redo the design with more topologies, never more tasks.

**Cost.** 96 reactive arms (~40 s each) + 1,344 study arms (~165 s each) ≈ 2.5 h at the
48-task `MaxSubmit` ceiling. Scripts: `scripts_cosim/datalab/unsaturated_edge_v1_{mint,screen,study}.sbatch`;
read: `scripts_cosim/unsaturated_edge_v1_gate_read.py {m0,study}`.

**Datasets.** No new corpus; the 1,670-dataset checkpoints of `peer_only_v1` /
`bipartite_edge_v1` are served as-is. Cells `cc{40,80}s{9101..9108}` minted here
(`simulation_data/unsaturated_edge_v1/cells.json`); results under
`simulation_data/peer_affinity_live_gate/results/ue_v1_screen/` and `ue_v1/`.

## Record (newest first)

- 2026-09-19 — Registered; mint (job 791283) complete; screen (791284, +1 block) running.
