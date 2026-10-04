# local_features_v1 — with only per-candidate features, does the GNN beat CD, CD+ext, the same-input MLP and the engineered MLP?

**Status:** `REGISTERED` (2026-10-04). Arms, topologies, statistic and verdict were fixed before any training run;
Amendments 1–2 (below) narrowed the verdict and added completeness and robustness reads before any gate summary was read.

**Amendment 1 (2026-10-04, before any lf1 gate run; only offline val regret and the three smoke runs had been seen).**
The success claim is scoped to the same-input comparison: primary family `lf1gnn` vs {CD, `cdextr`, `lf1mlp`} × {×2, ×3,
×5}, Holm across 9; **GNN-BEATS-MLP-AND-CD** iff all three are CONFIRMED (median ≤ −5 %, Holm p < 0.05) at the same ≥ 2
rungs and no reference is Holm-confirmed faster anywhere; else **NO-WIN**. `sb1mpoff` (the engineered pointwise scorer,
which sees hand-computed relational columns this arm withholds from both models) moves to the descriptive reads and is
still reported in this node. Reason: the question is whether the GNN learns the relational structure a same-input MLP
cannot, and whether that suffices to beat the search rules. The original 12-test family is below, struck by this amendment.

**Amendment 2 (2026-10-04, before any lf1 gate summary was read; after an outside design review).** (a) Every run that timed
out at 2700 s (lf1 arms, and the reused CD / `cdextr` runs) is rerun once at 8100 s (`datalab/local_features_v1_retry.sbatch`;
the references rerun into `local_features_v1/ref_retry`, `small_batch_confirm_v1`'s directory untouched). A run that hangs
again is dropped by name and listed. The verdict reads **INCOMPLETE** if a primary test has fewer than 19 topologies or a
primary run is missing without that second failure record. Revised the same day after the failure list (names and
timeouts only, 1–2 % of runs; no contrast computed) showed three CD timeouts that a rerun could not have been planned for.
(b) Reported beside the verdict, never replacing it: the registered 12-test verdict (with `sb1mpoff`); every `lf1gnn`
seed alone vs CD, `cdextr` and the median-over-seeds MLP; a seed-pairing-free contrast vs the MLP (per window, median
over seeds against median over seeds), since seed i of one arm has no natural partner in seed i of another. (c) If the
verdict is a win, the claim waits on an MLP learning-rate check: `lf1mlp` retrained at lr 5e-4 and 5e-3 (8 seeds each)
and gated on the same cells; it stands only if `lf1gnn` also beats the best of the three MLP rates by the same bar.
Disclosures: Amendment 1 was made knowing the offline val regrets and `sb1mpoff`'s live results on these 19 topologies
(`small_batch_confirm_v1`), so dropping it from the family is a forking path and both verdicts are reported; this is
the topologies' third confirmatory use (`raw_plan_v2` Phase C, `small_batch_confirm_v1`, here); a GNN–MLP win bundles
the convs with the committed-load sum channel, and `lf1gnn` vs `lf1twin` says which; the learned arms serve with 3-pass
self-refinement.

**Question.** With the engineered context, the GNN ties its MP-OFF twin (`small_batch_confirm_v1`): the context's
relational columns (exchange to placed partners, lookahead mass, batch-mates' committed service and occupancy) are a
hand-written message pass, so the pointwise model already has what MP would compute. With no engineered features at all
the GNN beats the MLP by 14–33 % but loses to CD by 20–60 % (`raw_plan_v2`), mostly on queue. This lineage takes the middle:
both models get only what describes **the candidate itself**, and the relational part must come from the graph.

**Why the raw-plan architecture.** In the engineered arms the GNN encode runs once per batch, before any commitment
(`make_partial_state_score_fn` caches it); the committed plan reaches only the scorer, through the engineered columns.
With those columns gone, the GNN could not see the plan at all. The raw-plan path (`plan_raw`) re-encodes per committed
set, with the committed flag and per-platform count on every bipartite edge, peer edges (log bytes) and same-node edges.

**The local block** (`plan_raw_local`, env `GNN_PLAN_RAW_LOCAL`, sidecar key; `src/policy/gnn/plan_raw.py`): five static
columns per (task, candidate) edge, independent of the committed set — log1p backlog seconds of the replica, log1p the
task's own service seconds there, the node's standing load / capacity, headroom after this task, node rank (size-free).

**Arms** (small-batch corpus, cache and split of `small_batch_v1`; 100 epochs, best validation checkpoint; 8 seeds each):
- `lf1gnn`: `raw_plan_v2` `rawS` (raw plan, committed-load sum channel, two bipartite convs) + local block;
- `lf1twin`: the same with every conv skipped (keeps the sum channel) — descriptive;
- `lf1mlp`: no message passing and no sum channel (the same-input pointwise control: a partner is invisible unless it
  lands on one of the task's own candidates) + local block.
- References from `small_batch_confirm_v1`'s gate (same topologies and inputs): CD, `cdextr` (1 run), `sb1mpoff`
  (engineered pointwise scorer, 8 seeds), `sb1load` and `xs1load` (descriptive).

**Topologies.** `small_batch_confirm_v1`'s 19 (raw_plan_v2's admitted pool, 9483–9568). No lf1 model or this proposal
has seen any of them; results of other arms on them were read, and only in aggregate (disclosed).

**Statistic.** Per topology, the median paired % over (window, seed); median over 19 topologies, exact Wilcoxon.

**Primary family as registered (superseded by Amendment 1)** (Holm across 12): `lf1gnn` vs {CD, `cdextr`, `lf1mlp`, `sb1mpoff`} × {×2, ×3, ×5}.
- **GNN-BEATS-ALL**: the same ≥ 2 rungs at which all four are CONFIRMED (median ≤ −5 %, Holm p < 0.05), none faster anywhere.
- **GNN-BEATS-SAME-INPUT**: CD, `cdextr` and `lf1mlp` CONFIRMED at the same ≥ 2 rungs; `sb1mpoff` not beaten.
- **NO-WIN** otherwise. (`scripts_cosim/local_features_v1_read.py`.)

**Pre-gate checks.** Unit tests (`tests/test_plan_raw.py`: the block is per-candidate and plan-independent, the twin
stays blind to a partner elsewhere, the GNN sees it); trainer determinism; per-arm sidecar checks in the training job;
smoke (`SMOKE=1`): cached-graph train/serve parity for seed 1 of each arm, one live run per arm, `lf1gnn`'s live
decisions dumped and replayed bitwise through the trainer's model. An offline val read orders nothing here; the live
gate runs regardless.

**Expectations (judgement).** `lf1gnn` beats `lf1mlp` ≥ 2 rungs: 75 %. Beats CD and `cdextr` too: 35 %. GNN-BEATS-ALL: 10 %.

**Risks.** Learned relational aggregation was far weaker than the hand version in `raw_plan_v2` (offline val regret ~19 s
vs 6.6 s), though those arms also lacked the queue columns added here. The MLP is weaker by construction: the claim is
scoped to "the GNN learns the relational structure the same-input MLP cannot", never "graphs beat hand-coding".

**Entry points.** `datalab/local_features_v1_train.sbatch`, `datalab/local_features_v1_gate.sbatch`,
`experiments/local_features_v1_{lf1gnn,lf1twin,lf1mlp}.yaml`, `fresh_topo_burst_v1_gate.py lf1conf`.

## Record

- 2026-10-04 — **Smoke (job 826416).** Cached-graph train/serve parity bitwise for seed 1 of all three arms (237 scored
  steps each); one live run per arm completed. The replay of `lf1gnn`'s 200 dumped live calls failed at 4.8e-7 under the
  job's 8 BLAS threads and passed **bitwise** when re-run with one thread, as the live runs score (login node,
  `OMP_NUM_THREADS=1`); the sbatch now pins it. The gate parts were launched by hand after this check. First training
  run (826314) died with the home quota at its limit (all 24 tasks at one second, 17:59:43); rerun 826415 after
  compressing old result files.
- 2026-10-04 — Registered.
