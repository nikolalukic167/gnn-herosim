# burst_ladder_v1 — is the learned arm's lead over CD in bursty w0 real, and does it grow with intensity?

**Status:** `ACTIVE`. Registered 2026-09-26; the S0 gate was read 2026-09-26.
- **×1 `NO-LEAD`:** A −11.1 %, p = 0.064, 10/12. The in-flight blind spot is **not** the cause:
  `cd_inflight` recovers 5 % of the lead.
- **×1.5 `CD-DEFECT`** (share 0.67; B −12.2 %, p = 0.11) and **×2 `NO-LEAD`** (A −37 %, p = 0.13;
  B −30 %, p = 0.042) are both **inadmissible**: reactive saturates on 11/12 and 12/12 topologies.
- The 20 % criterion is not met.

**What this is:** a live S0 screen that orders work toward a ~20 % learned-arm win. It is live, so rule 6 is
met for what it reads. It does not by itself close the program question. A lineage that trains on its
answer still ends with its own registered live gate.

**Parents and motive:**
- [`seeded_cd_xs1_v1`](seeded_cd_xs1_v1.md) and [`exchange_seconds_v1`](exchange_seconds_v1.md): on the
  mean of windows the learned arm ties CD. Re-reading the existing gate summaries per window (the
  2026-09-26 environment study, no new runs):
  - in **w0, the burstiest window**, `xs1load_selfref` beats CD **−11.1 % median over topologies** (faster
    on 10/12), with a range from −61 % (9461) to +220 % (9434);
  - in w1–w3 it ties (−0.6 to −1.5 %).
- **The competing explanation, a CD defect:** CD's drain term (`platform_queue_drain_seconds`) walks only
  the queue and never charges the task a platform is serving now. That task's remaining service includes
  its input + peer-exchange stage (~4 s), so under a burst a busy platform reads ~idle and CD stacks onto
  it (9461 w0: node5's queue 19.5 s). The learned arms are served `service_end_v1` and see that backlog.
  If CD fixed this way recovers the lead, the lead is a defect in the bar, not something learned.

## Design

- **The ladder:** window w0 (`burst_drainable_f4000_n50000`) at arrival intensity **×1, ×1.5, ×2**,
  labelled `w0x10`, `w0x15` and `w0x20`.
  - Every timestamp and every policy time constant (`batch_timeout`, `keep_alive`, the reconcile interval)
    is multiplied by 1, 2/3 or 1/2, with `cd_gap_v1_build_b.py` and `HEROSIM_POLICY_TIME_SCALE`: the
    `cd_gap_v1` B′ protocol run in the other direction. Physics (execution, transfer, cold start) is
    unscaled, so only the load changes.
  - Inputs are built once into `inputs/ladder/x{10,15,20}`; the existing inputs are never rewritten.
- **Topologies:** the 12 `fresh_topo_burst_v1` study topologies.
- **Arms**, 432 runs in one datalab job (phase `ladder`, timeout 2700 s):
  - `cd`: the registered CD greedy.
  - **`cd_inflight`**: CD with `HEROSIM_PG_INFLIGHT=1`, which adds `live_audit.inflight_remaining_seconds`
    (the service end the platform recorded) to the drain of every candidate. The default is off and
    byte-identical (`tests/test_pg_inflight.py`). Checked per run: the flag is served and
    `pg_inflight_charged > 0`; no other arm charges it.
  - `selfpredict`: the per-arrival bar, reported.
  - `reactive`: Knative; admissibility disclosure only.
  - `xs1load_selfref` s1–4: the learned arm, as gated in `exchange_seconds_v1`.
  - `xs1load_cdapply` s1–4: the hybrid from `seeded_cd_xs1_v1`, reported.
- **Witness:** every ×1 run whose arm also ran at w0 in the reference gates (`ref_fresh_1ae90af`, `xs1`,
  `xs1cd`) must equal it in `total_rtt` to the digit. A mismatch is disclosed, and the ×1 rung is then
  not "the existing w0".

## Statistic and bars (signed 2026-09-26, before any datum)

- **Statistic:** per rung, one window: per environment the median over seeds of the paired %, exact
  two-sided Wilcoxon over topologies, a topology missing any run of either arm dropped by name. The
  reader is `scripts_cosim/burst_ladder_v1_read.py`.

| read | contrast |
|---|---|
| A | `xs1load_selfref` vs `cd`: the learned lead |
| **B** | `xs1load_selfref` vs `cd_inflight`: the lead over the repaired bar (blocking) |
| I | `cd_inflight` vs `cd`: what the defect is worth |
| share | per topology, (cd − cd_inflight) / (cd − learned) in elapsed, over topologies where the learned arm leads; the median |

- **Label per rung, decided in this order:**
  1. `NO-LEAD`: A is not faster at p < 0.05.
  2. `CD-DEFECT`: share ≥ 0.80, or B is not faster at p < 0.05.
  3. `LEARNED-LEAD-REAL`: B is faster at p < 0.05. It reads "direction only" when |B| < 5 %.
- **Growth, reported:** "the lead grows" means B's median falls strictly from ×1 to ×1.5 to ×2.
- **What counts toward the 20 % path:** `LEARNED-LEAD-REAL` at some rung with B ≤ −10 %, and that rung
  admissible for at least 8 topologies.
- **Admissibility, disclosed per rung and not filtered:** reactive queue share ≤ 0.80
  (`unsaturated_scale_v1`); a missing reactive run is unknown, not a pass.
- **Expectations:**
  - ×1: `CD-DEFECT` 40 %, `LEARNED-LEAD-REAL` 45 % (direction only in most of those), `NO-LEAD` 15 %.
  - Lead grows with intensity: 35 %.
  - Some rung reaches B ≤ −10 %: 20 %.
- **Standing risks:**
  - **Seed-dependent collapse of the learned arm in w0** (9434 w0 +220 % in the `xs1` gate). A collapse is
    kept, not dropped. The seeds' median absorbs one bad seed per environment, and the collapse is
    disclosed per topology.
  - At ×1.5 and ×2, overload may push runs past the timeout. The drop rule applies and can fall below the
    8-topology floor (`DESIGN-SHORT`).
  - `inflight_remaining_seconds` is unknown during a cold start or a peer rendezvous, and is charged as 0
    there. That is disclosed; it under-repairs CD rather than over-repairs it.
  - One window shape: the ladder scales w0, it does not sample new bursty traces.

## Entry points

- CD knob: `src/policy/peer_greedy_network/scheduler.py` (`PG_INFLIGHT_ENV`, `_pg_choose`). Test:
  `tests/test_pg_inflight.py`.
- Gate: `scripts_cosim/fresh_topo_burst_v1_gate.py` phase `ladder`, run through
  `scripts_cosim/datalab/backlog_corpus_v1_gate.sbatch` with `PHASE=ladder`, `WT=<burst_ladder worktree>`
  and `TIMEOUT=2700`.
- Reader: `scripts_cosim/burst_ladder_v1_read.py`.

## Record (newest first)

- 2026-09-26 — **S0 read** (datalab job 809097 at b2ec7f9, 20 min; 432/432 runs, no failures).
  - **Witness:** 120/120 ×1 runs equal their existing w0 runs in `total_rtt` to the digit (CD,
    self-predict and reactive from `ref_fresh_1ae90af`, the learned arms from `xs1`/`xs1cd`).
  - **Attachment:** [`ladder_read.json`](burst_ladder_v1/ladder_read.json).

  | rung | A learned vs CD | B learned vs `cd_inflight` | I `cd_inflight` vs CD | share | admissible | label |
  |---|---|---|---|---|---|---|
  | ×1 | −11.07 %, p 0.064, 10/12 | −6.63 %, p 0.13, 10/12 | −0.56 %, p 0.064 | 0.05 | 12/12 | `NO-LEAD` |
  | ×1.5 | −29.93 %, p 0.021, 9/12 | −12.20 %, p 0.11, 10/12 | −10.63 %, p 0.11 | 0.67 | 1/12 | `CD-DEFECT` |
  | ×2 | −36.95 %, p 0.13, 10/12 | −30.00 %, p 0.042, 9/12 | −6.48 %, p 0.15 | 0.37 | 0/12 | `NO-LEAD` |

  - **At ×1 (the admissible regime), the CD-defect explanation is refuted.**
    - The in-flight term moves CD −0.56 %, and recovers 5 % of the lead (median over the 10 leading
      topologies).
    - The lead is large and broad (per topology: −13.5, −14.1, −3.7, −11.2, −17.7, −11.5, −8.8, −61.4,
      −5.1, −10.9 %).
    - It does not reach significance because of two topologies: **9434 +220 %**, the registered w0
      collapse, reproduced to the digit, and 9456 +6.5 %.
    - Pooled means: learned 7.24 s vs CD 8.52 s; queue 3.47 vs 4.38 s; exchange 3.87 vs 4.06 s per task.
  - **×1.5 and ×2 are overload, not a regime to quote.**
    - Reactive queue share is 0.88–0.96 at ×1.5 and 0.96–0.98 at ×2. Every arm's queue is 13–106 s per
      task.
    - The gaps are large (A −30 % and −37 %; per topology up to −85 %). They are a race to collapse, with
      the learned arm itself collapsing on 9434 (+29 %, +42 %) and 9461 at ×2 (+356 %).
    - B's median grows with intensity (−6.6 → −12.2 → −30.0 %), but only across inadmissible rungs.
  - **Reported:**
    - The hybrid (`xs1load_cdapply`) vs CD is −3.5 % at ×1 (p = 0.042) and not separated at ×1.5 and ×2.
      Under overload, CD's refine pulls the learned plan back.
    - The learned arm vs self-predict: −20.3 % at ×1 (11/12), −49.9 % at ×1.5, −33.4 % at ×2.
  - **Reading:**
    - In w0 at the real load, the pure learned arm leads CD by about 11 % on 10 of 12 topologies, for
      reasons other than CD's in-flight blind spot.
    - What stops it from being a result is **reliability**: seed- and topology-dependent collapses
      (9434) that one bad cell makes decisive in an 12-topology Wilcoxon.
    - Raising intensity by 1.5× already saturates the reactive reference, so the ladder has no admissible
      upper rung. An intensity between ×1 and ×1.5 (e.g. ×1.1, ×1.25) is where growth could be measured
      admissibly.

- 2026-09-26 — Registered.
