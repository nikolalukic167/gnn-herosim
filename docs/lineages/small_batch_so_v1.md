# small_batch_so_v1 — retrained on single-origin groups, does the engineered GNN beat CD again?

**Status:** `CLOSED` (2026-10-07) — **NO-WIN** (Amendment 1: **NO-WIN**). Registered 2026-10-06; the corpus recipe,
arms, topologies, statistic and verdict were fixed before any capture ran, Amendment 1 before its arms trained.

**Outcome.** Retrained on single-origin groups, the engineered GNN `so1load` still does not beat CD on the single-origin
cells: **+13.1 / +7.3 %** at ×2 / ×3 (CD Holm-faster, 0/19 and 3/19) and **−3.5 %** at ×5 (not separated); it ties its
MP-OFF twin (+0.8 / −0.4 / −2.0 %) and matches the scattered-origin checkpoint served zero-shot (+1.2 / +1.7 / +2.1 %), so
retraining recovers nothing. It still beats Knative −40 / −54 / −68 % and self-predict −26 to −30 %, and ties
locality-first and the one-pass greedy. Amendment 1, the raw-plan pair: `so1lfgnn` beats its same-input MLP −10.2 / −7.2 %
(×2 / ×3, CONFIRMED; ×5 −2.5 %, not separated) and loses to CD at every rung, +19.4 / +17.7 / +13.4 % (Holm). **The
grounded-workload win over CD (`small_batch_confirm_v1`, −10 / −35 / −17 %) depends on scattered request origins.**
Caveats: the 19 topologies were read many times before, the zero-shot result was seen before registration, 1,003 of 5,535
corpus datasets (single-node groups) were quarantined by the recipe's rule, and 8 of 24 training cells were lost to capture
saturation.

**Why.** [`client_local_v1`](client_local_v1.md) found that the grounded mint scatters a peer group's tasks over clients
(97 % of multi-task groups), while a trace request has one caller. With every group given one origin, the rules run 26–82 %
faster, and `sb1load` served zero-shot (trained on scattered origins) no longer beats CD: +12.2 % at ×2 (CD faster, 0/19),
+3.8 % at ×3 and −7.0 % at ×5 (neither separated), while still beating Knative −41 / −54 / −65 % and tying its MP-OFF twin.
CD is a search and adapts to the workload; the learned arm was trained on the other one. This lineage retrains it on the
single-origin workload and re-runs the comparison.

**Corpus** (`small_batch_v1`'s recipe verbatim; only the workloads change). The 24 training topologies `cc40s9201–9224`
and grounded windows g0–g3 at ×1 (`backlog_corpus_v1/gates/inputs/grounded/wl`), rewritten by
`scripts_cosim/client_local_v1_single_origin.py` into `simulation_data/small_batch_so_v1/wl`. Capture, corpus and cache are
`small_batch_v1_{capture,corpus,cache}.sbatch` with `CORPUS=small_batch_so_v1` and `WLDIR` set (defaults unchanged, so
`small_batch_v1` reruns as before). Server-only candidates, as before. Split: `experiments/small_batch_so_v1_split.json`,
same rule (held-out cells 9222–9224 = test, val = cells 9207–9209). The capture now runs with the starved-type eviction
and rendezvous release (`client_local_v1`, verified replay-identical); `small_batch_v1`'s capture lost 40 of 96 runs to
the hang it fixes.

**Arms** (4 seeds each, 100 epochs, best validation checkpoint; configs `small_batch_v1_{sb1load,sb1mpoff}.yaml` with the
cache and split swapped): `so1load` (`sb1load`'s recipe) and `so1mpoff` (its MP-OFF twin). References on the same cells:
CD, Knative, locality-first, one-pass greedy, self-predict (`client_local_v1/gate_so_server`) and the zero-shot `sb1load` /
`sb1mpoff` runs there (seeds 1–4).

**Cells.** `small_batch_confirm_v1`'s 19 topologies × g0–g3 × {×2, ×3, ×5} with single-origin workloads, server-only
(`client_local_v1/inputs_so_server`). Disclosure: these topologies have been read several times before, and the
zero-shot result above was seen on these exact cells before this registration.

**Amendment 1 (2026-10-06, before any training of these arms).** Adds the raw-plan pair, where message passing beats
the same-input MLP (`local_features_v1`; zero-shot on the single-origin cells still −11 / −14 / −22 %, while CD is faster
+21 / +18 / +16 %): `so1lfgnn` and `so1lfmlp`, `local_features_v1_{lf1gnn,lf1mlp}.yaml` with this lineage's cache and split
(`experiments/small_batch_so_v1_{so1lfgnn,so1lfmlp}.yaml`), 4 seeds, same cells and statistic. Own family, Holm across 6:
`so1lfgnn` vs {CD, `so1lfmlp`} × rungs; **LF-BEATS-MLP-AND-CD** iff the same ≥ 2 rungs have both CONFIRMED and no
reference is Holm-confirmed faster anywhere, else **NO-WIN** (INCOMPLETE as below). The primary verdict is unchanged.

**Statistic.** Per topology, the median paired % over (window, seed); learned arms pair on the seed, rules on their single
run. Median over the 19 topologies, exact two-sided Wilcoxon over topologies.

**Primary family** (Holm across 6): `so1load` vs {CD, `so1mpoff`} × {×2, ×3, ×5}. CONFIRMED = median ≤ −5 % and Holm
p < 0.05. Verdict, first that holds:
- **BEATS-CD** — vs CD CONFIRMED at ≥ 2 rungs and CD Holm-confirmed faster at none (a message-passing win additionally
  needs vs `so1mpoff` CONFIRMED at the same rungs, reported beside it);
- **NO-WIN** — anything else;
- **INCOMPLETE** — a primary test short of 19 topologies, or a primary run missing without a failure record from one
  rerun at 3× the timeout.

Descriptive: `so1load` vs zero-shot `sb1load` (does retraining help), vs Knative, locality-first, one-pass greedy,
self-predict.

**Entry points.** Workload rewrite `scripts_cosim/client_local_v1_single_origin.py`; corpus sbatches above; training
`small_batch_v1_train.sbatch` with `CORPUS=small_batch_so_v1 ARM_PREFIX=so1 SPLIT_SHA=<split sha>` (configs
`experiments/small_batch_so_v1_{so1load,so1mpoff}.yaml`); gate phase `so1` of `fresh_topo_burst_v1_gate.py` on
`client_local_v1/inputs_so_server`.

## Record

- 2026-10-07 — **Read; CLOSED NO-WIN (Amendment 1 NO-WIN).** Training 833205 (`so1load`/`so1mpoff`, ~1 h each) and 833337
  (`so1lfgnn`/`so1lfmlp`, 1.3–7.2 h); gates 833214 and 833338, 1,824 runs per pair, 0 failed, 0 unexplained missing.
  Primary, median paired % over 19 topologies, Holm across 6: vs CD +13.10 (0/19, REF-FASTER) / +7.30 (3/19, REF-FASTER) /
  −3.51 % (11/19, p_holm 0.34); vs `so1mpoff` +0.77 / −0.38 / −2.03 % (all NOT-SEPARATED). Descriptive vs the zero-shot
  `sb1load`: +1.2 / +1.7 / +2.1 %. Amendment 1 (Holm across 6): `so1lfgnn` vs CD +19.37 / +17.67 / +13.43 % (REF-FASTER at
  all three); vs `so1lfmlp` −10.18 (18/19) / −7.24 (16/19) / −2.50 % (10/19, NOT-SEPARATED). Retraining vs zero-shot:
  `so1lfgnn` −1.2 / −0.4 / +0.8 %, `so1lfmlp` −2.1 / −5.7 / −13.9 % (the MLP gains most). `so1lfgnn` vs `so1load`
  +6.6 / +9.2 / +14.7 %. Reader `scripts_cosim/small_batch_so_v1_read.py`, output `simulation_data/small_batch_so_v1/so1_read.json`
  on datalab.
- 2026-10-06 — **Corpus and cache (jobs 833174, 833201).** 4,851 train + 684 held-out complete sweeps; **1,003 of 5,535
  datasets quarantined** for peer_norm 0 (every candidate on one node), against 35 in `small_batch_v1`: with one origin,
  many groups reach a single node, and the recipe excludes those. Cache 4,532 graphs, candidate check 0 offenders, backlog
  > 0 on 48.9 % of candidate replicas. Split `experiments/small_batch_so_v1_split.json` (sha `30ab0719…`): train 3,115,
  val 841 (cells 9207–9209), test 576 (cells 9223–9224). `small_batch_v1`: 3,305 / 981 / 719.
- 2026-10-06 — **Capture (job 832835): 64 of 96 runs finished, 16 of 24 cells.** The 8 cells whose four windows all hit
  the 4 h limit (9205, 9210, 9212, 9213, 9216, 9218, 9219, 9222) were not frozen: a stack dump of 9205 g0 showed the
  clock advancing about 100 simulated seconds in 3 h at 32,600 of 108,600 s, inside the batch collector with an 11 GB
  queue — the one-pass greedy capture policy saturates on them. `small_batch_v1`'s capture lost 40 of 96 runs to the
  starved-client hang; this one loses 32 to saturation, and the corpus step skips unfinished captures by design. Held-out
  test cells are 9223 and 9224 only (9222 lost); validation cells 9207–9209 are complete. Corpus job 833174 launched.
- 2026-10-06 — Registered. The three corpus sbatches take `CORPUS` / `WLDIR` (and the capture `HEROSIM_RAW_DIR`).
