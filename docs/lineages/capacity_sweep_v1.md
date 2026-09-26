# capacity_sweep_v1 — does the learned burst-seat arm sustain more load than CD, and is its x1 lead placement or replica churn?

**Status:** `ACTIVE` (2026-09-27). Registered, not yet gated. Every bar below was signed before any datum
of the new rungs or the keep-alive control existed.

**Parent:** [`burst_ladder_v1`](burst_ladder_v1.md). With 4 perturbed draws at w0 ×1.5, `xs1load_selfref` beats
CD −28.1 % (p = 0.009, 11/12), `cd_inflight` −29.8 %, self-predict −51 % and Knative −79 %
(`OVERLOAD-LEAD-SURVIVES`). ×1.5 is inadmissible (reactive queue share 0.88–0.96 > 0.80), so the overload lead
is not quotable as latency. At ×1 the lead is −11.9 % (9/12, p = 0.15, `DISSOLVES`), and 9434 collapses on
every draw.

**Two hypotheses this lineage tests:**
- **Capacity.** Peer exchange occupies the platform: exec is about 0.07 s per task against about 4 s of exchange.
  The learned arm pays 5–11 % less exchange per task than CD (3.87 / 3.94 / 3.75 vs 4.06 / 4.31 / 4.20 s at
  ×1 / ×1.5 / ×2), so it should have more effective capacity, which queueing amplifies.
  - A capacity knee is horizon-free and quotable under the program's admissibility rule, where the overload
    latency gap is not.
- **Churn.** The 9434 diagnosis (2026-09-26, scratch reruns) found that idle-replica expiry (`keep_alive` 30 s)
  followed by slow scale-up is about 2/3 of every arm's w0 latency there. With `keep_alive` removed, CD goes
  8.7 → 2.8 s and the learned arm's collapse vanishes (within +1 to +5 % of CD).
  - The ×1 lead elsewhere may therefore partly be replica dynamics, not placement.

## Design

- **Rungs:** w0 at intensity ×1.0, ×1.1, ×1.2, ×1.3 and ×1.5, built like the ladder
  (`cd_gap_v1_build_b.py`, factor 1/intensity on timestamps and policy time constants).
  - Each rung runs under the same 4 perturbed draws (`burst_ladder_jitter.py`, seeds 1–4), identical across arms.
  - ×1.0 and ×1.5 are `burst_ladder_v1` Amendment 1's runs; ×1.1–×1.3 are new (phase `capacity` of
    `fresh_topo_burst_v1_gate.py`).
- **Arms:**
  - on every rung: CD, Knative (`reactive`), `xs1load_selfref` seed 1;
  - at ×1.0 also self-predict, added here;
  - `cd_inflight` and self-predict at ×1.5 come from Amendment 1.
  - One learned seed: disclosed; Amendment 1's ×1 draws also ran seed 1 only.
- **Stability record:** every new run stores its mean queue per quarter of tasks in dispatch order
  (`queue_drift`), computed from the raw per-task results before they are deleted.
- **C2 control:** the ×1 draws served with `HEROSIM_KEEP_ALIVE=1e9` (idle replicas never expire) for CD, Knative,
  self-predict and `xs1load_selfref` s1.
  - The new flag defaults to unset, which keeps the default path bit-identical, and it is provenance-whitelisted.
  - The driver fails a run whose recorded flag disagrees with its window.
- **Scale:** 720 new runs, one job. Code: branch `capacity-v1`.

## Bars (signed 2026-09-27, before any datum)

| read | what | fires as |
|---|---|---|
| **C1** | per topology, each arm's **knee**: the highest rung at which it, and every lower rung, has median-over-draws queue share (queue / elapsed) ≤ 0.80 | `CAPACITY-GAIN` if the learned knee is above CD's on ≥ 8 of the 12 topologies and the exact two-sided sign test over non-tied topologies has p < 0.05; else `NO-CAPACITY-GAIN` |
| **C2** | ×1, keep_alive removed: learned vs CD, per topology the median over draws of the paired %, Wilcoxon over topologies | `PLACEMENT-LEAD` (median ≤ −5 %, p < 0.05) / `PLACEMENT-LEAD (direction only)` (median < 0, p < 0.05) / `CHURN-LEAD` (otherwise) |

- **Reported, not bars:**
  - the interpolated capacity index (where the share crosses 0.80) and its learned / CD ratio;
  - per rung, the learned arm's paired contrasts vs CD, `cd_inflight`, self-predict and Knative, wherever both
    were run;
  - per rung, the topologies where Knative's share is ≤ 0.80 (admissible);
  - the median last-quarter / first-quarter queue drift per arm;
  - C2's same-draw contrast with churn on, and each arm's churn share of elapsed.
- The reader is `scripts_cosim/capacity_sweep_v1_read.py`.
- **Expectations:**
  - C1: `CAPACITY-GAIN` 35 %, `NO-CAPACITY-GAIN` 65 %. The rungs are 10 % apart and the hypothesised gain is
    5–10 %, so many topologies may tie.
  - C2: `PLACEMENT-LEAD` 20 %, direction only 40 %, `CHURN-LEAD` 40 %.
- **Standing risks:**
  - A 10 % rung spacing can hide a 5 % capacity gain. The capacity index is the finer read, but it is reported,
    not a bar.
  - w0 is non-stationary, so queue drift is confounded with where the burst peak falls. It is reported, not a bar.
  - The learned arm is one seed; its seed-dependent collapses (9434) enter C1 as low knees.
  - Removing keep_alive changes the environment for every arm. C2 separates placement from churn; it is not a
    proposal to change the physics.
- **Rule 6:** these are live runs. A pass on C1 makes a capacity claim quotable. The learned arm's latency
  headline stays the admissible-rung contrast.

## Entry points

- Flag: `src/executesimulation.py` (`_resolve_keep_alive`). Driver: `scripts_cosim/fresh_topo_burst_v1_gate.py`
  (`capacity`, `KA_WINDOWS`, `queue_drift`). Inputs: `scripts_cosim/datalab/backlog_corpus_v1_gate.sbatch`.
- Tests: `tests/test_capacity_sweep_v1.py`.

## Record (newest first)

- 2026-09-27 — Registered.
