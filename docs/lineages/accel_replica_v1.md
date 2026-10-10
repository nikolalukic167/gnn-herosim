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

- **W1, the change takes effect.** With CD at the calibrated heavy rung, the accelerator share under
  `fastest_compatible` is at least the share under `first_compatible` + 15 points, and xavierGpu hosts at least one
  type. Amended twice on 2026-10-10 (see the Record): the DLA clause was dropped, and the bar was tightened from an
  absolute 20 %.
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

- 2026-10-10 — **W3 PASS; the replay applies the rule** (S7, `rp/accel-calib` c8b56d8e; I11 855277, check 855296).
  - **I11** at heavy ×27.6227, 9905, 40 states, CD capture: median 0.000 %, p95 0.098 %, max 0.123 %, 40 of 40
    replayed.
  - **Rule in replay:** the same 40 states were replayed under both rules.
    - 10 states create a replica inside the horizon.
    - In 7 of them, the end-of-replay platform mix differs; each new replica takes the faster available platform.
    - The replayed batch latencies are identical under the two rules, because placements are forced. So I11 alone
      cannot detect the rule, and this platform comparison is the evidence.
  - Still to verify before labels: the corpus-labelling path (`make_warm_corpus`) shows the rule line.
- 2026-10-10 — **Rungs calibrated; W1 PASS** (S7, `rp/accel-calib` bb26e296; seeds 9905–9908 × g0/g1; 8/8 guards).
  - **Rungs (fastest_compatible):** moderate ×11.6139 (CD effective share 0.220), heavy ×27.6227 (0.468). Against
    first_compatible's ×16.4245 / ×23.2278.
  - **Finals, mean latency moderate / heavy:**
    - CD 3.13 / 2.51 s
    - Self-predict 3.15 / 2.49 s (a tie with CD)
    - Batched greedy 3.67 / 2.82 s
    - Knative collapses (0.99 effective share).
  - **The congestion moves to the GPU.** CD's heavy lock wait is 47.8 % xavierGpu and 43.9 % xavierCpu, against 90 %
    xavierCpu before.
  - **W1** (job 855273, `accel_s7/w1/w1.json`): CD accelerator share at each rule's own heavy rung.
    - first_compatible: 5.2 % (cell range 1.8–8.3 %).
    - fastest_compatible: 34.0 % (31.2–45.0 %).
    - The difference is **+28.8 points**, against a bar of 15. xavierGpu hosts cnn and rf. **PASS.**
    - Equal load (×23.2278): +31.0 points, so this is the rule's effect, not the load's.
  - Descriptive: under both rules, CD's mean latency is higher at moderate than at heavy (first_compatible: 2.79 vs
    2.69 s). The cause has not been checked.
- 2026-10-10 — **Step 1 done; coordinator's preinit claim RETRACTED** (S7, `rp/accel-replica` 00a08d1a; 16 tests pass).
  - **Correction, measured:** live runs do **not** preinit at t = 0. A 4,000-event CD run on 9903 creates no initial
    replicas, and its first scale events are reachability 'up' events at t = 0.158 s.
    - `simulation.py:395–418` is the fallback branch of `precreate_replicas`, which needs
      `preinitialize_platforms` and a `replica_plan`. `executesimulation.py:200` builds neither.
    - Live replicas are created on demand. Under R1 (kpa) every arm uses `src/policy/gnn/autoscaler.py`
      (`create_first_replica`, `create_replica`).
    - My 2026-10-10 amendment read the code path without running it.
  - **Code:** one helper, `src/placement/replica_rule.py`, at the generator and precreate fallback, in
    `gnn/autoscaler.py`, and in `knative_network/autoscaler.py`. A non-default rule refuses any autoscaler without
    `supports_replica_rule`.
  - **Default-rule bit-identity vs R1.1** (4,000 events, same cell):
    - IDENTICAL for reactive, self-predict, batched greedy and CD.
    - random_network is nondeterministic at a single commit (two base runs differ), so it is not checkable. This is
      disclosed and predates the change.
    - The learned family is **required before the gate** (one ra_* run, default rule, identical to R1.1).
  - **Lever acts live** (CD, uncalibrated ×46.46, descriptive): accelerator share is 16.8 % under first_compatible and
    50.2 % under fastest_compatible. cnn moves to xavierGpu (681 of 1,044 tasks). Mean elapsed is 80.0 s against
    59.3 s.
  - **W1 tightened** (coordinator; the default already reaches 16.8 % at an uncalibrated rung, so 20 % is not a
    test). W1 is now:
    - accelerator share under fastest_compatible ≥ the share under first_compatible **+ 15 points**, both measured with
      CD at the calibrated heavy rung;
    - and xavierGpu hosts at least one type.
    - This is a tightening, disclosed after seeing the uncalibrated descriptive mix.
  - Pre-existing, disclosed: the precreate fallback crashes under `replica_overlap` (double `initialized.succeed()`).
    It is never reached on overlap cells.
- 2026-10-10 — **Amendment before any data (coordinator, on S7's step-1 audit).**
  - **Where the platform is chosen.** It is chosen in three places, and the rule must apply in all three:
    - (i) the generator (`generate_infrastructure.py:745–765`), which feeds co-sim capture and replay through
      `replica_placements`;
    - (ii) **live t = 0 preinit** (`src/placement/simulation.py:395–418`), which re-derives the replicas from the
      same `replicas` config by the same first-suitable rule over `node.platforms`. It does not read
      `infrastructure.json`. So live does create t = 0 replicas; S7's "live doesn't place replicas" covered only the
      network-generation path;
    - (iii) autoscaler scale-out (`create_first_replica` / `scale_up`, `sorted(available_hardware)`, alphabetical,
      e.g. `knative_network/autoscaler.py:210`).
  - **Ruling:** option 1. One shared helper orders compatible platforms by `preinit.replica_placement_rule`
    (`first_compatible` reproduces today's order at each site; `fastest_compatible` uses exec time, ties by name). It
    is used at (i), at (ii), and at (iii) in every autoscaler the gate arms use.
    - The flag lives in `preinit`, not `replicas`, because a string key in `replicas` would be read as a task type
      (`executecosimulation.py:1328`).
    - Required before calibration:
      - default-rule bit-identity of one live run per arm family against R1.1;
      - a test that capture-precreated and live-preinit replica sets are equal under both rules;
      - the run JSON's t = 0 platform mix under each rule.
  - **W1 amended.** Under `fastest_compatible` with overlap on, the exec table sends dnn1 to pynqFpga, dnn2 to
    xavierCpu, and rf and cnn to xavierGpu. xavierDla hosts no type, so "DLA used" cannot hold. W1 is now: at least
    20 % of CD's tasks run on an accelerator, **and** xavierGpu hosts at least one type. DLA use is descriptive.
- 2026-10-10 — Registered on the user's request ("send this study now … check if it's worth and then do if yes").
  Owner: S7.
