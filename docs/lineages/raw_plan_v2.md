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

### 2026-10-01 — queue-gap diagnostic (offline, descriptive; changes no registered read)

`scripts_cosim/queue_gap_diag_eval.py` / `queue_gap_diag_read.py` (job 822198; read `raw_plan_v2/queue_gap_read.json`).
The served decoder ran on all 1,186 VAL graphs of the `backlog_corpus_v1` split for the v1 raw GNN (`rawgnn`), v1 raw MLP
(`rawmlp`), the engineered-context GNN (`psignn` = `xs1load`) and its MP-OFF twin (`psimlp`), 4 seeds each, self-refine
0 and 3. Seed 1 of `rawgnn` reproduces the trainer's val regret (19.109 s against 19.101 s). The checkpoints were
selected on this split, so these are comparisons across arms, not held-out estimates.

- **Val regret (s), refine 0 → 3:** rawgnn 19.2 → 18.3; rawmlp 31.0 → 30.6; psignn 6.57 → 3.99; psimlp 6.96 → 3.87.
  Self-refine removes 39 % of the engineered GNN's regret and 5 % of the raw GNN's.
- **Not the standing queue.** The raw GNN's gap to psignn is present in every backlog quartile, including the emptiest
  (Q1, mean candidate queue 0.12 tasks: 15.4 s vs 6.2 s). On choice tasks it picks candidates with a *shorter* standing
  queue than the label does (−0.07 tasks, against psignn −0.03).
- **Not stacking, and not how often it co-locates.** Tasks on the busiest platform: 3.83 (label 3.70), psignn 3.92.
  Peer pairs on one node: 35.6 % (label 37.8 %), psignn 38.8 %.
- **The CD greedy's own cost terms, decoded plan minus label plan, per graph (s),** refine 0:

  | | service | exchange | backlog | in-batch wait | total | val regret |
  |---|---|---|---|---|---|---|
  | rawgnn | −0.01 | **+7.68** | +1.23 | **+9.07** | +17.97 | 19.2 |
  | rawmlp | −0.05 | +15.52 | −0.93 | +18.03 | +32.57 | 31.0 |
  | psignn | −0.01 | −0.05 | −0.52 | +4.40 | +3.81 | 6.6 |
  | psimlp | 0.00 | −0.18 | −0.48 | +4.06 | +3.40 | 7.0 |

  The totals track the regrets, so these terms explain them. The raw GNN loses on **exchange seconds** and
  **in-batch wait**. The two are not independent: a task's charge to its platform includes its exchange, so extra
  exchange also lengthens the wait of later batch-mates. Service is unaffected, so platform-type choice is fine.
- **Reading.** What the raw graph lacks is the cost of *where* a non-co-located partner sits. The cached tensors
  hold no inter-node route: `platform_features` are type, queue length, remaining times and concurrency, `edge_attr`
  is task-to-platform, and the node-to-node transfer seconds, `route_hops_bneck` and the per-candidate backlog seconds
  live only in `partial_state_ctx`, which only ψ reads. The raw GNN gets the *number* of co-located pairs about right
  and puts the rest on the wrong remote nodes. This is an inference from the cost split and the cache layout. It has
  not been tested by giving the raw graph the missing information.

### 2026-10-01 — information-injection test (offline, descriptive; changes no registered read)

An independent reviewer (a fresh subagent given only the draft and the repository) read the diagnostic above and
found the inference "the raw graph lacks route information" **partly unsupported**: the exchange excess falls roughly
linearly with the co-location rate across arms (rawmlp 30.6 % / +15.5 s, rawgnn 35.6 % / +7.7 s, psignn 38.8 % / −0.05 s),
so some of it may be use of information the model has, not information it lacks. It also noted that the corpus's
validation states are near-empty (mean candidate queue 0.12–3.1 tasks by quartile), while the live deficit sits at
25 s of queue per task, so **the diagnostic cannot see the live standing-backlog regime**, and that the live record
(`raw_plan_v1`: exchange 3.60 s per task against CD 3.24, queue 25.3 against 20.6) was never reconciled with it. Both
points stand. "Not the standing queue" above holds for the validation states only.

The reviewer proposed a no-training test. `queue_gap_inject_eval.py` / `queue_gap_inject_read.py` (jobs 822301, 822308)
subtract hand-computed terms, in seconds from the cached context, from the model's own logits at decode: the peer
transfer to committed partners (`exch`), the in-batch service + exchange already on the candidate's platform (`load`),
the expected transfer to partners not yet placed, averaged over their candidate nodes (`mass`, the lookahead column),
and the standing backlog seconds on the platform (`back`). The weights are tuned on the even-indexed half of VAL and
read on the odd-indexed half, per checkpoint, 4 seeds. Stage 1 grids (`exch`, `load`); stage 2 fixes them at the stage-1
optimum, 0.3 and 0.1 (the same on all four seeds), and grids (`mass`, `back`). The zero setting reproduces the
diagnostic's regrets. The injection is additive and post hoc, so it understates what a trained channel could use.

Held-out half of VAL, mean regret (s), 4 seeds:

| | none | + exch | + exch, load | + mass | + mass, back |
|---|---|---|---|---|---|
| rawgnn | 19.45 | 15.81 | 15.07 | 11.88 | **10.02** |
| rawmlp | 31.15 | 18.60 | 16.73 | 11.30 | **10.78** |
| psignn (reference) | 6.65 | 6.65 | 6.65 | 7.12 | 7.13 |

(`+ mass` and `+ mass, back` are on top of `exch` 0.3 and `load` 0.1. psignn takes no injection, as expected: it already
reads these terms.)

- **The missing information is real and large.** The four terms remove 9.4 s of the raw GNN's 12.8 s gap to the
  engineered GNN (73 %), and 20.4 s of the raw MLP's 24.5 s. The largest single term for the GNN is the **lookahead over
  unplaced partners**, −3.2 s, more than committed-partner exchange, −3.6 s.
- **It does not remove it all.** The raw GNN stays 3.4 s above the engineered GNN. The cause of that remainder is not
  isolated.
- **The message-passing edge goes with the information.** Raw GNN beats raw MLP by 11.7 s with no injection and by 0.8 s
  with all four terms. The raw GNN's offline advantage over its twin is the recovery of this same information.
  Once the facts are supplied, a pointwise model matches it, as with the engineered context.
- **Consequence for a route channel limited to committed partners:** it addresses 3.6–4.4 s of the 12.8 s; the lookahead
  term is the larger piece and the channel as drafted lacks it.

### 2026-10-01 — Phase C study selected (Amendment A1): 12 of 19 admitted in 9473–9568

- **Screen of 9473–9520** (job 821825; 384 runs, 600 s timeout, 60 parallel, no scope). **7 of 48 admitted**: 9483,
  9484, 9485, 9487, 9491, 9502 and 9506. That is `DESIGN-SHORT`, since 12 are needed
  (`confirm/selected_before_9521.json`).
  - The job's closing check printed "0 admitted": it counted `topologies`, which `select` omits on a short verdict.
    The fix counts `qualified`.
  - An earlier version of this entry repeated "0 of 48". It also claimed every topology passing the reactive
    test had a hung run. **Both were wrong**; the verdict (short of 12) was right.
  - All 133 failed runs are timeouts. They stall at the first event, or at event 10,001 with simulated time far
    advanced: the starved-client spin. Finished runs take 44–125 s.
- **Witness** (job 821923, `raw_plan_v2_screen_witness.sbatch`). It was run because of the misread "0". The same
  code, venue and timeout reran the screen on four topologies `fresh_topo_burst_v1` admitted (9434, 9435, 9444,
  9446). All 32 runs finished in 53–84 s, with every share ≤ 0.64. All 32 are **identical to the digit** to the
  original screen: this code and venue reproduce the development pool's admission.
- **Amendment A1**, signed before any extension cell was minted, as the registration prescribes:
  - the pool extends with 48 newly minted ids, **9521–9568**: same generator, same unused check, same rule;
  - the study is the 12 lowest admitted ids of the combined pool;
  - fewer than 12 would be final `DESIGN-SHORT`;
  - ids for post-read amendments move to 9569+.
- **Extension screen** (job 821929):
  - the generator witness was byte-identical again, the range was unused, and all 48 cells were verified;
  - 12 of the 48 new cells were admitted: 9525, 9529, 9533, 9538, 9540, 9548, 9550, 9551, 9557, 9565, 9566, 9568;
  - combined, **19 of 96 were admitted, verdict `DESIGN-READY`**;
  - **Phase C study: 9483, 9484, 9485, 9487, 9491, 9502, 9506, 9525, 9529, 9533, 9538, 9540**
    (`confirm/selected.json`).

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
