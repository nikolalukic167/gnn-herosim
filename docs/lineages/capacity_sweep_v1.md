# capacity_sweep_v1 — does the learned burst-seat arm sustain more load than CD, and is its x1 lead placement or replica churn?

**Status:** `ACTIVE`. Registered 2026-09-27; gated 2026-09-27.
- **C1 `NO-CAPACITY-GAIN`:** the learned knee is above CD's on 5 of 11 topologies, below on 1, tied on 5
  (sign p = 0.22). The median interpolated capacity ratio (learned / CD) is 1.05.
- **C2 `PLACEMENT-LEAD` (direction only):** at ×1 with replica expiry removed, learned vs CD −8.43 %
  (p = 0.042, 10/12).

The ×1 lead is placement, not churn. The 9434 collapse vanishes (+1.2 %), but 9461 turns +16.6 %. A
capacity gain is not shown at 10 % rung spacing.

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

- 2026-09-27 — **Gate read** (jobs 809177 and rerun 809232, at 19a86df; 720 runs, 719 summaries).
  - **Attachment:** [`capacity_read.json`](capacity_sweep_v1/capacity_read.json). No dirty-code runs.
  - **Runs:**
    - 6 runs first crashed with rc 120 (memory cap on a 62-way node) and completed on the 16-cpu rerun;
      their failed records are kept in `gates/capacity/failed_attempt1/`.
    - **`cc40s9119__w0x13d4__cd_s0` hangs:** the `create_first_replica` retry loop at simulated
      t = 24,839 s (no compatible `dnn2` hardware reachable from `client_node8`) wrote a 2 GB log.
      - It timed out at 1800 s and was cancelled 20 min into the 5400 s rerun; recorded failed.
      - CD therefore has no ×1.3 draw 4 on 9119, and 9119 is dropped from C1 (CD knee unknown).
  - **C1 `NO-CAPACITY-GAIN`:** over 11 topologies the learned knee is higher on 5 (9423, 9435, 9461, 9466,
    9469), lower on 1 (9434: learned 0.0, i.e. inadmissible already at ×1, against CD 1.1), and tied on 5.
    Sign p = 0.22.
    - Knees (learned / CD):

      | Topology | Learned | CD |
      |---|---|---|
      | 9414 | 1.1 | 1.1 |
      | 9420 | 1.5 | 1.5 |
      | 9423 | 1.3 | 1.2 |
      | 9434 | 0.0 | 1.1 |
      | 9435 | 1.3 | 1.1 |
      | 9444 | 1.3 | 1.3 |
      | 9446 | 1.5 | 1.5 |
      | 9456 | 1.5 | 1.5 |
      | 9461 | 1.5 | 1.1 |
      | 9466 | 1.3 | 1.2 |
      | 9469 | 1.5 | 1.3 |

      Knative's knee is 1.0–1.3 except 9456 (1.5).
    - **Capacity index ratio** (learned / CD, reported): median 1.05, range 0.87 (9434) to 1.32 (9461).
      4 topologies are censored at 1.5 for both arms.
    - **Reading:** where the knees differ the learned arm is mostly higher, but it is not separated. The
      rung spacing and the ×1.5 ceiling censor it, and the 9434 collapse costs one topology.
  - **Per-rung paired contrasts, learned vs CD:**

    | Rung | Median | p | Faster on | Admissible topologies |
    |---|---|---|---|---|
    | ×1.0 | −12.2 % | 0.064 | 9/12 | 12 |
    | **×1.1** | **−13.7 %** | **0.042** | **10/12** | **11** |
    | ×1.2 | −12.8 % | 0.003 | 10/12 | 4 |
    | ×1.3 | −23.0 % | 0.002 | 10/11 | 2 |
    | ×1.5 | −29.5 % | 0.0005 | 12/12 | 1 |

    - ×1.1 per topology (%): 9119 −20.2, 9414 −14.2, 9420 −9.8, 9423 −12.1, 9434 +46.7, 9435 −37.7,
      9444 −11.0, 9446 −14.4, 9456 +3.9, 9461 −31.3, 9466 −13.3, 9469 −19.2.
    - ×1.0 also reads vs self-predict −21.8 % (11/12, p = 0.034) and vs Knative −52.8 %.
    - Median queue drift, last / first quarter: learned 1.31 / 1.70 / 2.30 against CD 2.01 / 2.31 / 2.86
      at ×1.1 / ×1.2 / ×1.3. Reported: w0 is non-stationary.
  - **C2 `PLACEMENT-LEAD` (direction only):** replica expiry removed, ×1, 4 draws: learned vs CD −8.43 %,
    p = 0.042, 10/12.
    - Per topology (%): 9119 −9.2, 9414 −2.7, 9420 −10.5, 9423 −6.2, 9434 +1.2, 9435 −9.2, 9444 −7.7,
      9446 −14.1, 9456 −9.1, 9461 +16.6, 9466 −4.3, 9469 −14.4.
    - Same draws with churn: −12.2 % (p = 0.064), 9434 +290 %.
    - With expiry removed, learned vs self-predict −47.8 % and vs Knative −67.2 % (12/12 each).
    - Churn share of elapsed (median, % removed by no expiry): learned 48, CD 50, self-predict 25,
      Knative 14.
    - **Reading:**
      - The lead survives without replica churn at about 2/3 of its size. It is placement, not churn.
      - Churn costs the learned arm and CD equally (~50 %), so it is not what separates them.
      - What churn does do is turn the learned arm's node concentration into the 9434 collapse.
      - 9461 flips to +16.6 % without churn: there the learned arm's advantage was avoiding CD's churn-driven
        collapse (see `burst_ladder_v1` Amendment 1).
  - **What is quotable now:**
    - At the first admissible burst rung (×1.1, 11/12 admissible), `xs1load_selfref` s1 beats CD −13.7 %
      (p = 0.042, 10/12). That is one learned seed, disclosed.
    - Its ×1 lead is a placement lead (C2).
    - A capacity gain is not established.

- 2026-09-27 — Registered.
