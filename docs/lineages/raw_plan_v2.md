# raw_plan_v2 — can a raw-plan GNN beat CD, its no-convolution twin and the MLP?

**Status:** `REGISTERED` (2026-10-01). Every read, the selection rule and the success bar below were fixed before
any data.

**Question.** [`raw_plan_v1`](raw_plan_v1.md) found that, on the raw plan, the GNN beats its MP-OFF twin at every rung
(−14 / −24 / −12 %) but loses to CD by +24 to +53 %. Most of the deficit is queue time: at ×2 the queue is 25.3 s,
against 13.4 s for `xs1load` and 20.6 s for CD, while exchange is close to CD's.

**Hypothesis (code read, not shown).** Missing load information *may* explain the queue gap. Two facts:
1. The bipartite conv mean-aggregates over each platform's candidate set (`BipartiteEdgeConv`). A mean may hide total
   committed work.
2. The v1 recipe sets `NEAR_RTT_MP_BIPARTITE_EDGE_ATTR_ZERO=1`, so the conv never sees exec time or the other
   physics columns.

v2 tests each change separately, then confirms on unseen topologies.

**The committed-load channel** (`plan_raw_sum`, env `GNN_PLAN_RAW_SUM`, sidecar `plan_raw_sum`):
- `platform_emb += scatter_add(load_mlp([task_emb ‖ edge_attr]), platform)`, over the committed task→platform rows only;
- `edge_attr` is the unzeroed physics `[exec, latency, warm, energy, comm]`;
- nothing is precomputed. It is a learned **sum** where the conv uses a mean.

It is weight-visible (`load_mlp.*`) and refuses without `plan_raw`.

**Arms** (`backlog_corpus_v1` cache and split, v1's recipe otherwise, 4 seeds, 3 self-refine passes). They form a 2×2
over the two changes:

| Arm | Conv sees edge physics | Committed-load channel | Convs |
|---|---|---|---|
| v1 `rawgnn` (exists) | no | no | yes |
| `rawE` | yes | no | yes |
| `rawS` | no | yes | yes |
| `rawES` | yes | yes | yes |
| `rawStwin` | n/a | yes | no |

- `rawStwin` is the twin of `rawS` and `rawES`. Zeroing acts only inside the convs.
- v1 `rawmlp` is the twin of `rawE`.
- The channel is itself message passing (a learned sum of task messages), so `rawStwin` is a **no-convolution
  twin**, not an MLP. Its contrast asks whether the convolutions help beyond the load sum.
- v1 `rawmlp` stays the plain pointwise reference and is kept visible in every read.

**Matched training budget.** Every learned arm uses its best-validation-regret checkpoint within the first
100 epochs.
- v1's checkpoints satisfy this. `rawmlp` last improved at epochs 93 / 44 / 92 / 30; `rawgnn` at epochs ≤ 99, with
  none after 100.
- v2 trains with `epochs: 100`, `min-epochs: 100` (no early stop).
- ψ arms (`xs1load`, `xs1mpoff`) appear only descriptively.

**Train/serve parity, before training.** For a fixed partial plan, and across a decode and its self-refine passes,
the trainer closure and the serving path must give identical raw-plan edge attributes and logits. A live dump of
(committed set, logits) from one servesmoke run must match its offline replay.

**Phase D — development** (the 12 `fresh_topo_burst_v1` study topologies; they informed this proposal).
- Cells: ×2 / ×3 / ×5 grounded, as in `raw_plan_v1`. Phase `rp2dev`.
- **Selection rule:** carry forward the convolution arm (`rawE`, `rawS` or `rawES`) with the lowest
  median-over-topologies paired % vs `cdextr`, pooled over the three rungs, together with its twin.
- Descriptive reads, labelled development:
  - each single change vs v1 `rawgnn`;
  - `rawES` vs `rawS` / `rawE`;
  - each arm vs CD / `cdextr` / its twin / `rawmlp`.

**Phase C — confirmation** (unseen topologies; the only phase that can claim a win).
- **Cells.** Mint ids 9473–9520 with the same generator (`cluster_scale_v1_cells.py`, base `cs6s9001.json`, 40
  clients, seed = id), in a separate inputs tree so the development cells are never rebuilt. The range must be checked as unused and disjoint from the
  training corpus and the 12 development topologies. Build the grounded ×2 / ×3 / ×5 ladder as in `peak_load_v1`.
- **Admission** (the rule that selected the development topologies):
  - reactive queue share ≤ 0.80 and the batched rule finishing, in all 4 **burst** windows w0–w3 at base load (×1),
    i.e. `fresh_topo_burst_v1_gate.py screen` + `fresh_topo_burst_v1_read.py select` unchanged (clarified
    2026-10-01, before any screen: the development topologies were admitted on w0–w3, not on the grounded g0–g3);
  - a failed, hung or missing run is not a pass;
  - the overloaded rungs are not screened.
- **Study.** The study is the 12 lowest admitted ids. If fewer than 12 are admitted, the result is `DESIGN-SHORT`
  and 48 more ids are added under the same rule.
- **Arms.** The selected arm, its twin, v1 `rawmlp`, CD, `cdextr` and reactive. Learned arms run 4 seeds, rules 1.
- **Reuse.** Phase C topologies are spent once read. Any later amendment moves them into development and confirms on
  newly minted ids (9569+; 9521–9568 went to Phase C's own extension, Amendment A1).

**Statistic.** Per topology, take the median paired % over (window, seed). Then take the median over topologies and
run an exact two-sided Wilcoxon over topologies. n = 12 topologies, never the runs.

**Primary family (Phase C).** {selected vs CD, vs `cdextr`, vs its twin, vs `rawmlp`} × {×2, ×3, ×5} = 12 tests,
Holm across all 12.

**Success** — `RAW-GNN-WIN`. Both of these must hold:
- there exist **the same ≥ 2 rungs** at which **all four** comparisons are `CONFIRMED` (median ≤ −5 % and Holm
  p < 0.05);
- at no rung is any reference Holm-confirmed faster.

Anything less is reported by label.

**Fallbacks** if Phase C fails. Each is a separate registered amendment, confirmed on new ids. In order:
1. train on the model's own decoded and refined prefixes;
2. four conv layers;
3. a one-batch-ahead queue-cost label.

**Parents:** [`raw_plan_v1`](raw_plan_v1.md), [`peak_controls_v1`](peak_controls_v1.md),
[`fresh_topo_burst_v1`](fresh_topo_burst_v1.md).

## Record

### 2026-10-01 — Phase C screen `DESIGN-SHORT` (0 / 48); Amendment A1: extend to 9521–9568

Amendment A1 was signed before any extension cell was minted or screened.

- **Screen of 9473–9520** (job 821825; 384 runs, 600 s timeout, 60 parallel, no scope): **0 of 48 admitted**.
  - 14 topologies pass reactive queue share ≤ 0.80 on all four burst windows: 9475, 9483–9488, 9491, 9492, 9499,
    9502, 9506, 9518 and 9520. Every one of them also has at least one reactive or batched run that timed out.
  - All 133 failed runs are timeouts. The hung runs stall at the first event, or at event 10,001 with simulated time
    far advanced; this is the starved-client spin, not slowness. Finished runs take 44–125 s.
  - Selection: `confirm/selected_before_9521.json`.
- **Witness** (job 821923, `raw_plan_v2_screen_witness.sbatch`). The same code, venue and timeout reran the screen on
  four topologies `fresh_topo_burst_v1` admitted (9434, 9435, 9444, 9446). All 32 runs finished in 53–84 s, every
  share is ≤ 0.64, and all 32 are **identical to the digit** to the original screen. The hangs are a property of the
  new cells, not of the code or the cluster. For comparison, the earlier extension 9425–9472 admitted 8 of 48.
- **Amendment A1**, as the registration prescribes:
  - the pool extends with 48 newly minted ids, **9521–9568**: same generator, same unused check, same rule;
  - the study is the 12 lowest admitted ids of the combined pool 9473–9568;
  - if fewer than 12 are admitted, the result is a final `DESIGN-SHORT` and is recorded as such;
  - the ids reserved for post-read amendments move to 9569+.

### 2026-10-01 — parity passed; training and Phase C screen launched

- **Smoke.** All four 2-epoch smoke trainings (jobs 821802–821805) passed the sidecar and weight checks, including
  `plan_raw_sum`, the `load_mlp.*` and `bip_convs.*` presence per arm, and the scorer width of 135.
- **Train/serve parity** (job 821864, `raw_plan_v2_servesmoke.sbatch`). Each smoke checkpoint was served live on
  9420 g0x20 with the dump instrument on; all four live runs finished. `raw_plan_v2_parity.py` then checked:
  - **cache:** the serving model and the trainer model (built from the experiment YAML, not the sidecar) scored
    20 cached graphs over a one-pass decode plus two self-refine passes. 600 steps per arm were bitwise identical.
  - **replay:** 150 live-dumped encodes per arm, re-scored by the trainer model, were bitwise identical.
  - It passed on all four arms.
- **Training.** Jobs 821878 (rawE), 821879 (rawS), 821880 (rawES) and 821881 (rawStwin): 4 seeds, 100 epochs, no
  early stop.
- **Development gates.** Phases `rawE` / `rawS` / `rawES` / `rawStwin` (jobs 821894–821897) are queued
  `afterok` on their arm's training.
- **Phase C.** Job 821825 minted 9473–9520 and verified all 48 cells. The re-mint of 9409 and 9470 was
  byte-identical, and no file used the range. It is screening 384 runs (reactive + batched, burst w0–w3, ×1).

### 2026-10-01 — registered

Registered from the `raw_plan_v1` read, before any v2 code ran on data.
