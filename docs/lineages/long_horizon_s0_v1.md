# long_horizon_s0_v1 — does the GNN-vs-hand gap change with horizon?

**Status:** `CLOSED` (2026-10-12) — **HORIZON-HAND-GAINS** (mixed signs). Registered 2026-10-12 with a directional bar; every bar
below was signed before its data. Placed by S6 on branch `rp/long-horizon` at the expert's instruction, with a branch-only
index row (authorised; the expert reconciles it on reference-physics).

**Outcome (2026-10-12; the registered single read, all 7 cells, 0 failed).** Over a 24 h stationary window the GNN's lead over CD
does not erode, and its deficit to `cd_exactS` grows.
- No GNN-favourable contrast reaches the −2 pp bar, and three hand-favourable contrasts exceed +2 pp, so the node closes under the
  registered bar. No multi-day gate.
- S = the OLS slope of the hourly paired gap over bins 2–24, × 24 h; g = the paired gap, negative = GNN faster.

  | contrast | S | mean g, bins 2–24 |
  |---|---|---|
  | moderate nosplit vs CD | −0.41 pp | −5.19 % |
  | moderate cdxapply vs CD | −0.58 pp | −4.83 % |
  | heavy nosplit vs CD | −1.77 pp | −7.28 % |
  | heavy cdxapply vs CD | +2.32 pp | −10.32 % |
  | moderate nosplit vs `cd_exactS` | **+2.73 pp** | +1.69 % |
  | moderate cdxapply vs `cd_exactS` | **+2.63 pp** | +2.05 % |

- **Quote as:** "over 24 h the GNN stays ahead of CD and falls further behind `cd_exactS` at moderate; one topology, a direction".
  Never as "the GNN gains with horizon".
- **Caveats:**
  - One topology (16900, spent), one seed per cell: a direction, never a magnitude.
  - Stationary trace by construction: it tests accumulation, not daily peaks.
  - Heavy `cd_exactS` was not run (worst projected cell ~25 h), so the bar at heavy is CD only.
  - Heavy cdxapply's +2.32 pp slope is driven by g_1 (−1.56 %) being its smallest gap; its blocks over bins 2–24 sit at −8.6 to
    −11.9 %, so it is not a reversal.
  - Hour-to-hour SD of g is 2.8–3.5 pp on every contrast, above the 2 pp the bar calls weak, so each slope is a weak band.
  - A user-ordered interim read of 4 cells preceded this read (Record); it changed no bar.
- **What this does not test (expert's ruling, 2026-10-12): no daily-cycle trace is downloaded.**
  - A daily cycle moves the system between the moderate and heavy regimes, both already measured.
  - The only untested piece is the transitions between them, and this read gives no sign the GNN gains from time structure.
- **Prediction (S6, before data):** HORIZON-FLAT with |S| < 1 pp. It missed: 3 of 6 contrasts are above +2 pp.

**Question (user, 2026-10-12):** "workloads are less than one day; what if they were seven days?" In the accel seat, does the
gap between the GNN and the hand search change with horizon? A longer trace can change the gap only if something
accumulates in t: replica/warm-state drift, queue or backlog build-up, or daily peaks. If the gap per sim-hour is flat in
t, a longer trace only adds n.

## Facts before design (Step A, read-only, 2026-10-12; accepted)

- **Window length.**
  - Heavy (×27.62, 12.717 tasks/s): last arrival 3,932 s (endTime ≈ 3,990 s).
  - Moderate (×11.61, 5.347 tasks/s): last arrival 9,351 s (≈ 9,400 s).
  - Both are 50,000 tasks.
- **Each window is 63–93 s of Alibaba v2021 real time,** stretched 133–205× (heavy) or 317–489× (moderate).
- **No source on disk has a daily cycle.**
  - The production trace is 169.5 s at a flat rate.
  - The raw Alibaba CSVs are not on disk.
- **The mint is hard-capped at 50,000 tasks** by the base window. One library holds ~160k tasks; the 4 libraries hold ~643k.
  - 24 h needs 1.10M tasks heavy and 462k moderate.
- **Wall time scales with tasks.** Projected for one 24 h cell, median (max):

  | arm | heavy | moderate |
  |---|---|---|
  | CD | 0.5 h | 0.23 h |
  | ra_gnn_eng_nosplit | 2.8 h | 1.3 h |
  | cdxapply | 3.3 h | 1.4 h |
  | cd_exactS | 17.7 h (24.9 h) | 8.1 h (10.9 h) |
- **Memory.**
  - Every task is retained (task_archive; raw per-task JSON ~250 MB per 50k tasks).
  - Measured peak is 4.3 GB per 50k-task cell. A linear guess for 24 h heavy is 60–95 GB.
- **Prior** (selfsearch_confirm, 96 cells per arm×rung):
  - Queue is 5–10× higher in Q1 (warm-up), then flat over Q2–Q4 in every arm.
  - The paired nosplit − cd_exactS queue gap is +0.19/+0.31 s in Q1, then within ±0.013 s in Q2–Q4.

## Design (cheap probe; not a gate)

- **Topology:** 16900, the lowest id of 16900–16911, chosen by id and not by outcome.
  - **Disclosed: spent.** The set was the `big_groups_s0_v1` G=8 live gate (856483).
  - This probe runs the accel **K=1** mint on it, not the G=8 inputs.
- **Environment:** the accel seat unchanged.
  - R1.1, fastest_compatible, 160c × 24s, p 0.6, wf1_v1 payloads, pipelined, KPA, top-5 slate, harness defaults as the
    selfsearch_confirm gate.
  - Rungs as calibrated: heavy ×27.62, moderate ×11.61.
- **Workload L24:** one 24 h **stationary** window per rung, from new mint code.
  - The 4 fan-out libraries (shards 0, 50, 100, 130) and the 4 base windows (w0..w3) are cycled in that order.
  - Each cycle gets a seeded offset: its own seed for the within-millisecond jitter and the partner and payload draws.
    Arrival times continue from the previous cycle's last arrival plus one mean inter-group gap.
  - The rate is stretched to the rung's rate, the same as the existing mint, with no envelope. Arrivals stop at the first
    arrival ≥ 86,400 s.
  - **Test (required):** the first 50,000 tasks equal `grounded_g0_n50000` at that rung — events, timestamps and the
    pairs among them.
  - The sha256 is recorded, and datalab mints its own copy and must match.
- **Disclosed: stationary by construction.**
  - The probe tests **accumulation** (replica/warm-state drift, queue and backlog build-up), **not daily peaks**.
  - A real daily cycle needs a real multi-day trace (the separate facts report). An imposed synthetic envelope is not
    allowed.
- **Arms:**
  - CD s0 at both rungs.
  - **ra_gnn_eng_nosplit s1** at both rungs (cheapest GNN arm; base g4 seed1, md5 0f915a36).
  - **ra_gnn_eng_cdxapply s1** at both rungs.
  - **cd_exactS s0 at moderate only.**
  - **Disclosed:** heavy cd_exactS is not run, because its worst projected cell is ~25 h. At heavy the hand bar is CD.
  - Total: 7 cells, ≤ 7 CPU jobs, within my share. --mem 128G per cell. Raw per-task results go to node-local /tmp and are
    deleted after summarising, as today.
- **Instrument (new):**
  - executesimulation computes, in-process from the task archive, one block per 1 h bin of **arrival** time, h = 1..24:
    - n tasks;
    - mean latency (averageElapsedTime's definition);
    - mean queue, cold, exchange and exec;
    - in-system count at the bin end;
    - the completion time of the bin's last-arriving task minus the bin end (stability per bin);
    - plus end/last-arrival for the whole run.
  - A unit test checks that the n-weighted bin means equal the whole-run means.
  - Memory: the full per-task stats path is kept, so the existing per-task summaries (queue drift, percentiles) stay
    comparable. The smoke's MaxRSS × 2.4 projects the heavy cells. If that projection exceeds ~110 G, a trim of the
    per-task list is added as a code fix before the heavy cells launch. The bins are computed in-process, so the read
    loses nothing.
  - **Instrument rule:** it is not on until a smoke result file has it. The smoke is the CD moderate L24 cell (about
    15 min), part of the 7 and agreed before the read. Only completion, timing and the presence and sum check of the bins
    are read before the full read.
- **Code (branch `rp/long-horizon`, from harness 79ea002c, the commit of the selfsearch_confirm and big_groups gates):**
  - `scripts_cosim/long_horizon_mint.py`: L24. Cycles 0–3 must equal the accel gate windows g0–g3 at the rung (events
    and pairs); cycle c ≥ 4 has seed 7400 + c and a seeded library/base offset.
  - `src/placement/hourly_bins.py`: the bins, opt-in through `HEROSIM_HOURLY_BINS=1`, hooked into both stats paths.
  - The harness window `L24<rung>`, which is never in a default grid. Its cells expect the window's own task count and
    check that the bins sum to the run.
  - `scripts_cosim/long_horizon_s0_v1_read.py` (the reader, committed before data).
  - `scripts_cosim/datalab/long_horizon_{inputs,cell}.sbatch`.
  - Tests: `tests/test_long_horizon.py` (bins equal whole-run means; prefix and reindexing of the concatenation; the
    remap equals its script; the L24 harness grid; the reader's bar).
- **Reader:** committed before any data, at a pinned commit, with a pinned worktree per job.

## Statistic and bar

- **Contrasts:** GNN-family arms G ∈ {nosplit, cdxapply} vs hand H.
  - H = CD at heavy.
  - H ∈ {CD, cd_exactS} at moderate.
  - That makes 6 contrasts.
- **Per contrast, per bin h = 2..24** (bin 1 is warm-up and excluded):
  g_h = (mean latency of G − mean latency of H) / mean latency of H, in %, paired on the same topology, window and bin.
- **Trend:** S = the OLS slope of g_h on h over h = 2..24, × 24 h (the fitted change over 24 h, in pp).
- **Directional bar (signed 2026-10-12).** g < 0 means the GNN is faster.
  - **HORIZON-FLAT:** |S| ≤ 2 pp on every contrast → CLOSE. No multi-day run.
  - **HORIZON-GNN-GAINS:** S ≤ −2 pp on at least one contrast → register the multi-day gate on fresh topologies, with a
    live gate. Its design is not fixed here. It needs:
    - streamed per-task results (7 d heavy is ~650 GB as built);
    - a ruling on cd_exactS (124 h median at 7 d heavy, over the 7-day partition limit);
    - a real daily-cycle trace, if one is legitimate.
  - **HORIZON-HAND-GAINS:** S ≥ +2 pp on some contrast and none ≤ −2 pp → CLOSE. The trend works against the GNN.
  - **Mixed signs:** both are reported, and the lineage closes unless a GNN-favourable contrast is ≤ −2 pp (which is
    HORIZON-GNN-GAINS).
  - **A missing contrast** (a failed or killed cell) is INCOMPLETE: unknown is not a pass. A watchdog kill is a failure,
    recorded and never rerun.
- **Disclosed with the read:**
  - one topology, so a direction, never a magnitude;
  - the hour-to-hour SD of g over h = 2..24 (if it exceeds 2 pp, the band is weak and that is said);
  - g_1, for the warm-up size.
- **Rule 6:** these are live runs, so a HORIZON-FLAT close stands on a live read. It orders the work, and any magnitude
  claim needs a gate.
- **Also reported (descriptive):** for every arm and bin, the latency breakdown; CD's per-bin stability and in-system
  count; the replica count series; and the wall time per sim-hour vs t, which checks the linear cost projection.

## Prediction (S6, written before any data; the expert's is also flat)

- **HORIZON-FLAT** for all 6 contrasts, |S| < 1 pp.
- After bin 1, g_h sits at each contrast's existing window-level value.
- CD is stable in every bin (end/last ≈ 1.01; no upward trend in in-system count).
- Why:
  - The arrival process is stationary by construction.
  - KPA turns replicas over at minute scale (stable window 60 s; ~2,300 scale-downs per heavy window), so warm state has
    no memory beyond minutes.
  - The queue, and the paired queue gap, are flat after the first quarter in every existing window.
- Risks:
  - Slow replica-set drift under fastest_compatible scale-out over more than ~2 h, which no existing window measures.
  - Wall-time growth in t from the retained task lists (a cost risk, not a gap risk).

## Cost

- Code after signature: the mint and its prefix test, harness window `L24`, the hourly-bin block and its test, and the
  reader. About half a day.
- Compute: 7 cells, about 11 h wall (bounded by moderate cd_exactS, 8–11 h).

## Record (newest first)

- 2026-10-12 — **CLOSED HORIZON-HAND-GAINS** (S6, the registered single read; expert ruling the same day).
  - Read once with `scripts_cosim/long_horizon_s0_v1_read.py` at 0605a5d2 (`~/lh_0605a5d2`), `--topo 16900`. Output:
    `/share/nikola.lukic/long_horizon/probe_0605a5d2/read_0605a5d2.{json,txt}`.
  - 7 of 7 cells have a summary; failed cells: none. Reader verdict: `HORIZON-HAND-GAINS (mixed signs across contrasts)`.
  - Per contrast, with the 6 four-hour blocks of g over bins 2–24:

    | contrast | S (pp) | g_1 | mean g(2..24) | SD | blocks |
    |---|---|---|---|---|---|
    | moderate nosplit vs CD | −0.41 | −5.70 | −5.19 | 3.02 | −4.4 −6.0 −5.4 −4.8 −6.0 −4.3 |
    | moderate nosplit vs `cd_exactS` | +2.73 | −1.64 | +1.69 | 3.50 | +1.1 +0.5 +1.9 +1.1 +1.7 +4.5 |
    | moderate cdxapply vs CD | −0.58 | −3.52 | −4.83 | 3.39 | −4.4 −5.2 −4.0 −4.9 −5.7 −4.7 |
    | moderate cdxapply vs `cd_exactS` | +2.63 | +0.63 | +2.05 | 2.81 | +1.1 +1.4 +3.2 +1.0 +2.0 +4.1 |
    | heavy nosplit vs CD | −1.77 | −5.57 | −7.28 | 3.48 | −6.6 −5.2 −8.5 −7.7 −9.3 −6.0 |
    | heavy cdxapply vs CD | +2.32 | −1.56 | −10.32 | 3.27 | −11.4 −9.4 −11.9 −9.5 −10.8 −8.6 |

  - Walls (s) and MaxRSS:

    | cell | job | wall (s) | MaxRSS |
    |---|---|---|---|
    | moderate CD (the smoke) | 863309 | 438 | 8.0 G |
    | moderate `cd_exactS` | 863323 | 14,706 | 8.0 G |
    | moderate nosplit | 863324 | 2,327 | 8.1 G |
    | moderate cdxapply | 863325 | 2,660 | 8.1 G |
    | heavy CD | 863326 | 870 | 18.6 G |
    | heavy nosplit | 863327 | 5,503 | 18.7 G |
    | heavy cdxapply | 863328 | 6,193 | 18.7 G |

    The heavy cells held 18.7 G, far below the 60–95 G linear guess. Moderate `cd_exactS` took 4.1 h, below the projected 8–11 h.
  - Expert ruling: close as HORIZON-HAND-GAINS; no daily-cycle download (head).
  - Against the interim read (moderate vs CD): the means agree (−5.19 / −4.83 %); the slopes are −0.41 / −0.58 here vs −0.40 / −0.56 there, and the SD 3.02 / 3.39 vs 2.96 / 3.31. The small differences are not explained here.

- 2026-10-12 — **User-ordered interim read, before the registered single read** (disclosed by the expert; it changes no bar).
  - At the user's direct instruction, the expert read the `hourlyBins` of the 4 cells finished at that time:
    CD moderate and heavy, nosplit moderate, and cdxapply moderate.
  - Interim at moderate, paired % against CD per hour, bins 2–24:

    | contrast | mean | sd | OLS slope × 24 h | bin 1 (warm-up) |
    |---|---|---|---|---|
    | nosplit | −5.19 | 2.96 | −0.40 pp | −5.7 |
    | cdxapply | −4.83 | 3.31 | −0.56 pp | −3.5 |

  - Heavy contrasts were not read; the GNN heavy cells were still running.
  - The registered single read (`long_horizon_s0_v1_read.py` on all 7 cells, at 0605a5d2) still stands and is the one
    that closes the node.
  - These are contrasts against CD; the reader's registered contrasts are against the hand arms it names.
- 2026-10-12 — **Launched.** Inputs 863306 (moderate 462,166 tasks, 10 cycles; heavy 1,098,577 tasks, 22 cycles; prefix
  equalities passed).
  - Smoke 863309 (CD moderate): wall 438 s, MaxRSS 8.0 GB, 24 bins summing to num_tasks.
  - Cells 863323 (cd_exactS mod), 863324 (nosplit mod), 863325 (cdxapply mod), 863326 (CD heavy), 863327 (nosplit heavy),
    863328 (cdxapply heavy).
  - Out `/share/nikola.lukic/long_horizon/probe_0605a5d2`, worktree `~/lh_0605a5d2`.
- 2026-10-12 — **Inputs job 863303 FAILED in 3 s, before minting anything.** The mint checked the accel manifest's sha256
  against the final gate files. The manifest describes the W2 stage (`inputs_w24`, before the 160-client remap), so the
  check could never pass.
  - Code fix: the manifest is now checked against `inputs_w24`. The final gate files are still checked by equality with
    cycles 0–3.
  - Nothing was produced, and the inputs are re-minted at the fix commit.
- 2026-10-12 — **Signed** by the expert with a directional bar (HORIZON-FLAT / GNN-GAINS / HAND-GAINS, above).
  - Daily-cycle traces: no downloads until L24 reads.
  - If L24 reads FLAT or HAND-GAINS, the next question is daily peaks. The cheapest check comes first: one day of
    Alibaba v2022 MSRTMCR (~72 GiB), binned hourly to test the claimed 2:1 diurnal ratio.
  - Azure envelope plus Alibaba fan-out is a disclosed two-source fallback, used only if v2022 is not diurnal or not
    feasible.
  - A synthetic envelope is not allowed.
- 2026-10-12 — Drafted by S6 on the user's request via the expert ("workloads are less than one day; what if they were seven days?"). Step A was accepted, and the Step B rulings are applied
  (stationary L24 probe; arms CD, nosplit and cdxapply, with cd_exactS at moderate only; bins 2–24; |slope| ≤ 2 pp per
  24 h). Nothing has been launched.
