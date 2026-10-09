# accel_replica_v1 — put the per-server replicas on the accelerators

**Status:** `REGISTERED` (2026-10-10). There is no data yet. Every bar below was signed before its data.

**Why.** Under R1.1, `per_server = 1` places each type's server replica on the **first** compatible platform in the
node's list order (`src/generate_infrastructure.py:745–765`).
- On the 160 × 24 calibration topology 9905, every type sits on rpiCpu and xavierCpu. dnn1 also gets pynqFpga.
- **The 7 xavierGpu and 7 xavierDla host no replica at t = 0.** They are reachable only through KPA scale-out.
- CD's lock wait is 90 % on xavierCpu, and half of it is cnn, which runs 7× faster on xavierGpu (`scale_160_v1`,
  2026-10-10).
- The environment is therefore less heterogeneous than its hardware, which may also hide placement structure a graph
  model could use.

## Design (fixed now)

- **Physics change (the only one).** A generator flag `replicas.placement_rule`:
  - `first_compatible` is today's rule and the default, so R1.1 is unchanged.
  - `fastest_compatible` places each type's `per_server` replica on the compatible platform with the lowest
    exec time for that type, from the same exec table the simulator uses. Ties break deterministically by platform id.
    Everything else is R1.1 + the T-fix + replay F.
- **Size:** 160c × 24s, p = 0.6, top-5 slate, as in `scale_160_v1`, so the headroom compares directly with its 69 %.
- **Rungs:** re-derived by the `scale_160_v1` calibration protocol on the reserved seeds 9905–9908 (CD-steered,
  effective share moderate 0.20–0.30 and heavy 0.40–0.50, the same guards, at most 8 steps).

## Worth-it screen (all must pass before any corpus build)

- **W1, the change takes effect.** Under CD at the calibrated heavy rung, at least 20 % of tasks execute on an
  accelerator (xavierGpu, xavierDla or pynqFpga), and at least one task type uses each of xavierGpu and xavierDla.
- **W2, headroom.** S5's probe (`scale_probe`) on 4 fresh seeds (9909–9912), at the calibrated rungs: at least 3 of 4
  seeds have more than 50 % of batches above 1 % joint (non-pointwise) regret. The pooled share is also reported against
  `scale_160_v1`'s 69 %. This is descriptive and does not gate.
- **W3, replay.** The I11 replay at the calibrated heavy rung, 40 states: median ≤ 1 %, p95 ≤ 1 %, max ≤ 5 %. The
  `scale_160_v1` PASS-WITH-CAUSE ruling applies to sub-second batches with an absolute miss under 0.1 s.
- **W4, no hangs.** 0 captures hang, and every CD guard passes at both chosen rungs.
- **Any failure is a NO-GO.** It is recorded, and the calibration finals (CD, self-predict, batched greedy and
  Knative, run live at the calibrated rungs) stand as this lineage's live read. No corpus is built.

## If GO

The pipeline is identical to `scale_160_v1`, under its own IDs and stem `accel-replica-v1`:
- **Corpus:** replay F, about 50 train topologies and 8 held-out, exact top-5 labels.
- **Arms:** CD vs gnn_eng and gnn_eng_physmp. Each gets 6 configs × 3 seeds, selection on validation only, and the
  100-epoch cap. The MP-off twin is trained only if a GNN wins.
- **Gate:** 12 fresh test topologies × 2 rungs × 4 windows × seeds 1, 2.
- **Primary family:** Holm over 4. A win is median ≤ −5 % with Holm p < .05 at both rungs.
- **Secondary, descriptive:** the same arms against their `scale_160_v1` counterparts (first_compatible).
- **Scheduling:** the corpus build queues **behind** `scale_160_v1`'s build. The combined datalab cap of 44 still holds.

## Record (newest first)

- 2026-10-10 — Registered on the user's request ("send this study now … check if it's worth and then do if yes").
  Owner: S7.
