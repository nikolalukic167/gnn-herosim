# small_batch_so_v1 — retrained on single-origin groups, does the engineered GNN beat CD again?

**Status:** `REGISTERED` (2026-10-06). The corpus recipe, arms, topologies, statistic and verdict below were fixed before
any capture ran.

**Outcome.** None yet.

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

**Entry points.** Workload rewrite `scripts_cosim/client_local_v1_single_origin.py`; corpus sbatches above; training and
gate to be added before the corpus is read.

## Record

- 2026-10-06 — **Capture (job 832835): 64 of 96 runs finished, 16 of 24 cells.** The 8 cells whose four windows all hit
  the 4 h limit (9205, 9210, 9212, 9213, 9216, 9218, 9219, 9222) were not frozen: a stack dump of 9205 g0 showed the
  clock advancing about 100 simulated seconds in 3 h at 32,600 of 108,600 s, inside the batch collector with an 11 GB
  queue — the one-pass greedy capture policy saturates on them. `small_batch_v1`'s capture lost 40 of 96 runs to the
  starved-client hang; this one loses 32 to saturation, and the corpus step skips unfinished captures by design. Held-out
  test cells are 9223 and 9224 only (9222 lost); validation cells 9207–9209 are complete. Corpus job 833174 launched.
- 2026-10-06 — Registered. The three corpus sbatches take `CORPUS` / `WLDIR` (and the capture `HEROSIM_RAW_DIR`).
