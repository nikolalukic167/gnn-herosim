# peak_load_v1 — does the learned burst-seat arm beat every strategy at published peak and surge loads?

**Status:** `CLOSED` (2026-09-28) — **`PEAK-LOAD-WIN`** vs CD and every industry-style baseline at ×1.5–×5,
on the old w0 window and on the Alibaba-grounded windows; **ties the self-predict rule** on the grounded
windows. Parts (a) and (b) were fixed before any of their runs (commit 35593af). **The ×2 / ×3 / ×5 ladder
(Amendment 1, "peak_load_v2") was added after (b) was read** — its rungs were chosen from the literature
below, not from its data, but the decision to climb was post hoc. Disclose it as such.

**Outcome** (`xs1load_selfref` vs each strategy; median over topologies of the per-topology median paired %;
exact Wilcoxon over 12 topologies; ✓ = `CONFIRMED`, ≤ −5 % and p < 0.05):

| Load (source) | Workload | vs CD | self-predict | one-pass | Decima | Knative | random | Knative queue share |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ×1 (`grounded_workload_v1`) | grounded | +0.4 % tie | −0.9 % tie | −7.1 ✓ | −20.5 ✓ | −16.2 ✓ | −39.8 ✓ | ≤ 0.80 (filtered) |
| ×1.5 peak hour (a) | old w0 | **−24.6 ✓** (p .007, 11/12) | −50.3 ✓ | −52.4 ✓ | −63.2 ✓ | −78.8 ✓ | −95.4 ✓ | 0.94 |
| ×1.5 peak hour (b) | grounded | **−5.3 ✓** (p .002, 11/12) | −4.6 tie | −12.1 ✓ | −26.4 ✓ | −26.2 ✓ | −53.6 ✓ | 0.70 |
| ×2 peak/trough (Am. 1) | grounded | **−12.5 ✓** (p .002, 11/12) | −9.6 tie (p .37) | −18.4 ✓ | −34.1 ✓ | −37.1 ✓ | −75.0 ✓ | 0.79 |
| ×3 minute surge (Am. 1) | grounded | **−20.4 ✓** (p .027, 10/12) | −14.5 tie (p .23) | −29.2 ✓ | −50.2 ✓ | −71.3 ✓ | −91.1 ✓ | 0.97 |
| ×5 to saturation (Am. 1) | grounded | **−9.2 ✓** (p .002, 11/12) | −2.3 tie (p .30) | −12.8 ✓ | −22.9 ✓ | −30.5 ✓ | −68.6 ✓ | 0.98 |

- The gap to CD rises to a peak at ×3 and falls at ×5: the saturation knee. Quote the curve, never ×3 alone.
- **Not quotable as a message-passing win.** At ×1.5 grounded the plain decode of the same weights ties CD
  (−2.6 %, p = 0.13), `gnnedge0` and its MP-OFF twin both trail CD (+2.9 / +4.5 %), and MP vs its twin is
  −1.95 % (direction only). Self-refine carries much of the margin. The ladder did not run the twins.
- **Admissibility.** The program's filter (reactive queue share ≤ 0.80) is **not applied** here: peak hour
  is the regime under study, and the share is reported per rung instead. Only ×1.5 grounded (0.70) and ×2
  (0.79) sit under the line; ×3 and ×5 are overload. (a) at 0.94 is the regime `burst_ladder_v1` already
  called inadmissible, re-read with every rule.
- ×3 is the least stable rung: 9434 and 9461 go the other way (+19.9 / +19.7 %).
- **No learned run failed** in any part. Rule timeouts (2,700 s) are recorded and dropped by name.

**Parents:** [`grounded_workload_v1`](grounded_workload_v1.md) (the grounded windows; ×1 tie),
[`burst_ladder_v1`](burst_ladder_v1.md) (the w0 ×1.5 draws and the ladder protocol),
[`exchange_seconds_v1`](exchange_seconds_v1.md) (the `xs1load` checkpoints),
[`decima_rule_v1`](decima_rule_v1.md) (Decima α = +1).

## Load sources

Every rung is a multiple of the study's ×1 rate (0.4604 tasks/s on 6 servers), applied with the
`cd_gap_v1_build_b.py` ladder protocol: timestamps and policy time constants × 1/k.

| Rung | Source | Kind |
|---|---|---|
| ×1.5 | Shahrad et al., ATC'20, Fig. 4: hourly invocations relative to peak; peak/mean ≈ 1.6 | figure read-off |
| ×2 | Shahrad Fig. 4 / §3.3: "a constant baseline of roughly 50% of the invocations" → peak ≈ 2× trough | stated |
| ×3 | Luo et al., SoCC'22 ("The Power of Prediction"), Fig. 15(a): ~250 k → ~720 k within ~1 min | figure read-off |
| ×5 | Gan et al., ASPLOS'19 (DeathStarBench): load swept to saturation (Figs. 9, 12, 13); real diurnal trace spans ~8× (Fig. 21) | method + read-off |

Also on record: Shahrad §3.6, 40 % of apps have inter-arrival CV > 1. Not usable for load: Luo SoCC'21 (no
peak or burst numbers) and UNSW-NB15 (no arrival rates).

## Design

- **Arms.** Rules, one run per cell: `random`, `reactive` (Knative), `batched` (one-pass greedy),
  `selfpredict`, `cd`, `decima` (α = +1); (a) adds `cd_inflight`. Learned, seeds 1–4: `xs1load_selfref`
  (primary); (b) adds `xs1load`, `gnnedge0`, `mpoff`. All checkpoints unchanged.
- **(a)** phase `x15fill`: the w0 ×1.5 draws d1–d4 of `burst_ladder_v1` Amendment 1 completed with random,
  one-pass and Decima (144 runs). CD, `cd_inflight`, self-predict, reactive and the GNN are the ladderjit
  runs. Witness: CD and GNN seed 1 on 9119 / 9420 `w0x15d1` rerun at this commit must equal ladderjit to
  the digit.
- **(b)** phase `groundedx15`: the grounded windows g0–g3 at ×1.5 (1,056 runs).
- **Amendment 1** phase `groundedladder` (commit 01282d1): g0–g3 at ×2 / ×3 / ×5, CD + `xs1load_selfref`
  first, then the other five rules (1,440 runs).
- **Topologies:** the 12 `fresh_topo_burst_v1` study topologies.
- **Statistic** (`scripts_cosim/peak_load_v1_read.py`, `--rung` for Amendment 1): per topology the median
  over (window, seed) of the paired % arm vs reference on the same window; exact two-sided Wilcoxon over
  topologies; a pair with a missing run is dropped by name; < 8 topologies → `DESIGN-SHORT`. Labels
  `CONFIRMED` / `DIRECTION-ONLY` / `REF-FASTER` / `NOT-SEPARATED` as `grounded_workload_v1`.

## Record

### 2026-09-28 — Amendment 1 read (job 812291)

1,425/1,440 summaries, 0 empty; 15 timeouts, none learned: 9466 random / reactive / self-predict on
g1 at every rung and g3 at ×2 (and self-predict g0, g2 at ×2); CD 9423 g2 ×2 (as at ×1.5).
Primary per topology:
- ×2: −66.8 (9434) … +6.3 (9461); 11/12 faster.
- ×3: −65.1 (9119) … +19.9 (9434); 9461 +19.7; 10/12.
- ×5: −14.7 … +2.8 (9456); 11/12.

Rules vs reactive at ×3: CD −41.6 %, self-predict −54.3 %, one-pass −41.6 %, Decima −22.6 %, random
+184.9 %. The self-predict contrast loses 9466's worst cells to timeouts, which favours the rule.
Reads: `peak_load_v1/peak_load_v2_{x20,x30,x50}_read.json`.

### 2026-09-28 — (b) read (jobs 812180 / 812219)

1,048/1,056; 8 rule timeouts (7 on 9466, CD 9423 g2). 21 runs lost to a /home storage incident
(19:10–19:15 CEST, 0-byte files) were moved aside to `incident_2026-09-28/` and rerun. Primary −5.26 %,
p = 0.0024, 11/12. Per topology −11.8 (9435) … +1.2 (9456). Read: `peak_load_v1/peak_load_b_read.json`.

### 2026-09-28 — (a) read

Witness passed (4/4 equal to the digit). Primary −24.61 %, p = 0.0068, 11/12; `cd_inflight` −24.9 %. An
earlier −28.1 % in `burst_ladder_v1` used medians over the rules' own draws; −24.6 % is the pairing above.
Read: `peak_load_v1/peak_load_a_read.json`.
