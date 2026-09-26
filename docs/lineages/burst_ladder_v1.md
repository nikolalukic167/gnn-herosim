# burst_ladder_v1 — is the learned arm's lead over CD in bursty w0 real, and does it grow with intensity?

**Status:** `ACTIVE` (2026-09-26). Registered; the S0 gate is running. Every bar below was signed before
any datum of this ladder existed.

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

- 2026-09-26 — Registered.
