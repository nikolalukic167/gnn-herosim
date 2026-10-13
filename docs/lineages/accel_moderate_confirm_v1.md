# accel_moderate_confirm_v1 — does the accel moderate-rung GNN lead over CD hold on fresh topologies?

**Status:** `CLOSED` (2026-10-10) — **DIRECTION-ONLY**. Registered 2026-10-10; every bar below was signed before its data.

**Outcome.** On 24 fresh topologies the moderate lead shrank and was not confirmed.
- **gnn_eng:** −4.38 % vs CD (16/24 faster, p .046, Holm .091).
- **physmp:** −3.84 % (16/24, Holm .091).
- **CD←GNN** beat CD on 24/24 topologies (−3.76 %); **self-predict** −1.77 % (both descriptive).
- The accel gate's −5.45 / −6.05 % was partly selection (winner's curse). The direction holds; the magnitude is about
  −4 %.
- **Diagnostics run alongside** (recorded in `accel_replica_v1`): the GNN's deficit sits in batches the declared slate
  splits into blind sub-groups.

**Why.** `accel_replica_v1` closed NOT-SEPARATED. At the moderate rung gnn_eng read −5.45 % (Holm .065) and physmp
−6.05 % (Holm .081) against CD, on 12 topologies. Heavy ran against the GNN (+2.7 %). This node tests the moderate
direction alone, on topologies no session has seen. The rung was chosen after seeing that read; fresh topologies are
what make the test honest.

## Design (fixed now)

- **Environment:** identical to the `accel_replica_v1` gate.
  - Physics: R1.1 + the T-fix, replay F, `replica_placement_rule = fastest_compatible`.
  - Size: 160c × 24s, p = 0.6, top-5 slate.
  - Workload: the same 4 WF1 windows.
  - **Moderate rung only:** m = 11.6139.
- **Topologies:** 24 freshly minted ids, **16321–16344**. These are disjoint from the accel corpus pool (16201–16300),
  the accel gate (16301–16312) and its spares (16313–16320). Disjointness from every other range is verified before
  minting (S5). An overlap shifts the ids by a dated amendment before any run.
- **Arms:**
  - gnn_eng and gnn_eng_physmp, using the exact checkpoints `accel_replica_v1` selected (no retraining);
  - CD, the bar;
  - self-predict and CD←GNN, both descriptive.
  - Seeds 1 and 2 for every arm.
- **Code:** 9e5ee729 for all arms, learned and classical, so nothing is paired across commits.
- **Statistic:** paired % vs CD per topology (median over windows and seeds), then the median over the 24 topologies,
  with an exact two-sided Wilcoxon.
- **Primary family:** {gnn_eng vs CD, physmp vs CD} at moderate, Holm over 2.
- **CONFIRMED** (per arm): median ≤ −5 % and Holm p < 0.05. **CD-FASTER:** CD Holm-confirmed faster. Anything else is
  DIRECTION-ONLY (median < 0) or NOT-SEPARATED.
- **Sealed:** SEALED=1. No result is read before every arm completes. Stuck cells are killed by progress rate, never
  rerun, excluded and counted.
- **Scope of any claim:** a CONFIRMED result is a learned-scorer win over CD at moderate load with accelerator replicas
  only. It is not a message-passing claim, since no MP-off twin exists, and it says nothing about heavy load.

## Record (newest first)

- 2026-10-10 15:10 — **Gate read: DIRECTION-ONLY, closed** (S4; one read, job 855818; output
  `accel_confirm/read_9e5ee729.{json,txt}`; classical 192/192 and learned 576/576, 0 failed, 24 topologies; the
  sensitivity pass is identical).

  | Arm vs CD (moderate) | Median | Topologies faster | p | Holm |
  |---|---|---|---|---|
  | gnn_eng | −4.38 % | 16/24 | .046 | .091 |
  | physmp | −3.84 % | 16/24 | .079 | .091 |
  | self-predict (descriptive) | −1.77 % | 18/24 | .0011 | — |
  | CD←GNN (descriptive) | −3.76 % | 24/24 | <.0001 | — |

  - **Decision cost per task:** GNN 6.6–7.1 ms, CD 0.12 ms. No cell has deferrals above 5× CD's.
  - **Reader fix, disclosed:** the reader was committed at f736c9f1 before any data. A test-isolation fix, 967e7858 (the
    rung override scoped to `read()`), came after the arrays finished and before any read, with no semantic change.
- 2026-10-10 13:05 — **Reader committed before data** (S4). rp/accel-confirm f736c9f1:
  `accel_moderate_confirm_v1_read.py` and its test. Moderate only, Holm over 2, with the labels as registered.
- 2026-10-10 12:45 — **Launched, sealed** (S4).
  - Inputs: rp/accel-confirm 8f6f45f0, data only (`accel_moderate_confirm_v1_selected.json`). All 24 cfgs carry
    fastest_compatible, and the moderate windows are byte-identical to the accel gate's (factor 1/11.6139).
  - Arrays at 9e5ee729: classical 855775 (cd and selfpredict at s0, 192 cells) and learned 855776 (ra_gnn_eng, physmp
    and cdapply at s1 and s2, 576 cells). Output `accel_confirm/{classical,learned}_9e5ee729`.
  - The builder also made heavy inputs (m 27.6227) for these ids. They are never run and stay on disk.
- 2026-10-10 12:20 — **Id range verified** (S5). 16321–16344 is disjoint from every range in use:
  - scale_160 16001–16020 and 16101–16200;
  - accel 16201–16320;
  - r1a and workload_fix 9xxx;
  - REGISTRY has no 16xxx, and no 1632x–1634x cell exists on datalab.
  The next free pool is 16345+. S4 may mint.
- 2026-10-10 11:45 — **Registered** (coordinator, on the user's go). S4 mints and runs the gate; S5 verifies the id
  range first.
