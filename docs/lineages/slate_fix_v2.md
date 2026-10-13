# slate_fix_v2 — does the deploy transient hide or inflate the GNN's heavy deficit, and does the no-split MP direction hold on 4 seeds?

**Status:** `ACTIVE` (2026-10-12; registered the same day, signed by the expert as drafted at 24672635, plus one descriptive line; drafted by S6). As of 2026-10-13: H 48/48 and L12 288/288 are complete, with cells at two pins (6a5c922c and f4f0453f, identity-tied; see the record). L34 864486 is running at f4f0453f. No cell has been read.

**Parent.** `slate_fix_v1` (CLOSED E-FASTER / UNCAPPED-DATA-NOT-THE-LEVER). Its statistic, averageElapsedTime over all 50k tasks, includes the cluster's own scale-from-zero cold start. SCIENTIST estimates that is about 40 % of heavy mean latency. Its per-task results were deleted, so no steady-state number exists for it. Its no-split MP contrast (new gnn vs new twin) was −2.57 % heavy (12/12, Holm .002) and −1.55 % moderate on two seeds.

**Questions.**
1. **Transient:** on the same cells, is the GNN's deficit to `cd_exactS_ib0` larger or smaller once the opening is cut (tasks dispatched at or after 360 s)?
2. **MP:** does new gnn_eng beat its separately trained MP-OFF twin in no-split mode on 4 training seeds? **This is a model-class result, not a hand-win claim.** Both arms lost to `cd_exactS_ib0` in slate_fix_v1, and nothing here can change that.

## Design

**Seat and code.** The accel seat as slate_fix_v1: 160 × 24, p 0.6, `fastest_compatible`, WF1 windows g0–g1, moderate ×11.6139 and heavy ×27.622665025860204, pipelined, `HEROSIM_REPLICA_RELEASE=1`, KPA. Topologies 19420–19431, slate_fix_v1's inputs `/share/nikola.lukic/slate_fix/inputs_12/inputs`.
- Every cell runs at **6a5c922c** (reference-physics). It carries the summarizer of c81f0e88: `steady_state.after_360s`, `after_p5_dispatch`, `decile_mean`, and `<cell>.tasks.npy`.
- Datalab worktree `~/gnn-herosim-wt/sf2_6a5c922c`. Harness phase `r1a`, `SELECT=scripts_cosim/slate_fix_v1_selected.json`, LIMIT 8,100. Watchdog kills and failed cells are counted, never rerun.

**Arms** (a cell is arm × topology × window × rung; 48 per arm and seed):

| arm | submission | checkpoints | cells |
|---|---|---|---|
| `cd_exactS_ib0` (the bar) | H (`R1A_SLATE` unset) | seed 0 | 48 |
| new `ra_gnn_eng_selfsearch` | L12 (`R1A_SLATE=declared_pruning_v1_uncapped`, `R1A_STEM=slate-fix-v1`) | seeds 1–2 (slate_fix_v1's) | 96 |
| new `ra_gnn_eng_nosplit` | L12 | seeds 1–2 | 96 |
| new `ra_twin_eng_nosplit` | L12 | seeds 1–2 | 96 |
| new `ra_gnn_eng_nosplit`, `ra_twin_eng_nosplit` | L34 (as L12) | seeds 3–4 (new) | 192 |

- Total **528 cells**.
- Seeds 3–4 are trained with slate_fix_v1's configs (`experiments/slate_fix_v1_{gnn,twin}_eng.yaml`, the g4 recipe on the uncapped cache, 100 epochs, W&B `gnn-slate-fix-v1`). Run as `slate_fix_v1_train.sbatch` with `SEEDS="3 4"`. Checkpoint names: `slate-fix-v1-{gnn,twin}-eng-seed{3,4}`. The sidecheck and md5 are logged.

**Identities (before and inside the gate).**
- **Cross-commit, before the gate** (16490 g0, both rungs, at 6a5c922c, 8 cells): new `ra_gnn_eng_selfsearch`, `ra_gnn_eng_nosplit` and `ra_twin_eng_nosplit` s1 against slate_fix_v1's smoke A (4d90119c), and `cd_exactS` against `identity_B2_d192081f` / `gate_79ea002c`. Every shared simulated key must be bit-equal (timing and provenance excluded, as before). If any differs, the gate does not launch, and the cause is fixed in code.
- **In the read:** every rerun cell (the bar and seeds 1–2) must reproduce its slate_fix_v1 cell's averageElapsedTime bit-equal. A mismatch is reported per cell. Those cells, and their pairs, leave the steady-state families, and the read says so. Seeds 3–4 have no earlier cell.

**Statistic.** As slate_fix_v1 (`r1_attribution_v1_read` helpers): per topology the median over (window, seed) of the paired %, then the median over 12 topologies, the sign count and the exact two-sided Wilcoxon. Steady-state latency is `steady_state.after_360s.mean`, the mean of (done − dispatched) over tasks dispatched at or after 360 s, the corpus MIN_TIME.

**Registered families (Holm within each).**
- **PRIMARY, transient (Holm over 4):** new self-search and new no-split vs `cd_exactS_ib0`, × 2 rungs, on after_360s latency (seeds 1–2).
  - Beside each: the whole-run % (this gate's cells; equal to slate_fix_v1's when the identity holds).
  - Also beside each: **Δ = steady % − whole %** in pp, per topology then the median, with its Wilcoxon p (not in the Holm family).
- **SECONDARY, robustness (Holm over 4):** the same contrasts on `after_p5_dispatch` latency.
- **MP (Holm over 4):** new gnn_eng no-split vs new twin_eng no-split, seeds 1–4 (seed s against seed s), × 2 rungs × {whole-run, after_360s}.
- **Descriptive, untested:**
  - the MP contrast on seeds 1–2 and seeds 3–4 separately (the replication check);
  - deciles of each arm's latency in dispatch order (median over cells);
  - per-arm served-slate counters (as slate_fix_v1's M4 table);
  - self-search vs no-split on after_360s;
  - the queue proxy;
  - **the bar's own transient share** (added at signature): per rung, the median over `cd_exactS_ib0` cells of the whole-run mean, the after-360 s mean and 1 − after-360 / whole. It is the first steady-state number on the bar itself.

**Labels (fixed now).**
- Transient, per contrast and rung:
  - **HIDES** (cutting the opening makes the deficit larger): Δ ≥ +1 pp with Wilcoxon p < .05;
  - **INFLATES** (the opening makes it look larger): Δ ≤ −1 pp with p < .05;
  - otherwise **NEUTRAL**.
- Primary on steady state: E-FASTER / WIN / DIRECTION-ONLY / NOT-SEPARATED with slate_fix_v1's rules (bar −2 %).
- **MP, per metric:**
  - **CONFIRMED** needs median ≤ −2 % (slate_fix_v1's bar) **and** Holm p < .05 at **both** rungs on 4 seeds;
  - **TWIN-FASTER** if the twin is faster with Holm p < .05 at either rung;
  - otherwise **DIRECTION-ONLY** (median < 0 at both rungs) or **NOT-SEPARATED**.
  - CONFIRMED is a model-class statement (message passing helps this scorer in this decode). It is never "the GNN beats a hand rule".

**Predictions, before any data.**
- **S6, transient:** self-search **HIDES** at heavy. Its Q2–Q4 queue was +45 % vs the bar, so the deficit lives in the steady phase, and a smaller denominator makes the % larger: heavy after_360s +7 to +12 % (whole +5.3 %).
- **S6, transient, no-split:** NEUTRAL to HIDES at heavy, +7 to +11 % (whole +7.4 %).
- **S6, transient, moderate:** both arms NEUTRAL, within ±1 pp of whole-run.
- **S6, primary on steady state:** E-FASTER for both arms at heavy, as whole-run. No WIN.
- **S6, MP on 4 seeds:** heavy −1.5 to −3 %, moderate −1 to −2 %; CONFIRMED on at most one metric. P(CONFIRMED on whole-run) ≈ 0.3; the moderate rung is the likely miss. Seeds 3–4 point the same way as 1–2.

**Not claimed:** any hand win; any seat other than the accel seat; a message-passing result in self-search mode; magnitudes from two seeds.

**Cost and order** (≤ 10 queue entries; at most 8 running):

| step | entries | resources | wall |
|---|---|---|---|
| training seeds 3–4 (gnn_eng, twin_eng; started 864264) | 2 | CPU-amd 18 CPU, 150 G | ~2 h |
| cross-commit identity (8 cells) | 1 | 64 CPU, PAR 8 | ~25 min |
| H: `cd_exactS_ib0` (48) | 1 | 64 CPU, PAR 24 | ~1–1.5 h |
| L12: 3 learned kinds × seeds 1–2 (288) | 3 shards | 64 CPU, PAR 24 | ~1.5 h |
| L34: 2 no-split kinds × seeds 3–4 (192), after training and sidechecks | 2 shards | 64 CPU, PAR 24 | ~30 min |
| reader | 1 | CPU-amd 2 CPU | minutes |

- About 10 entries in all; ≈ 4–5 h end to end.
- Order: training (running) ∥ cross-commit identity → H and L12 → L34 when training is done → single read.
- The reader `scripts_cosim/slate_fix_v2_read.py` is committed with tests before any gate cell.

## Record (newest first)

- 2026-10-13 — **Second-pin identity PASSED; H and L12 complete; L34 launched.**
  - Identity at f4f0453f: 864369 and 864370 are SIM-IDENTICAL to the 6a5c922c cells on every shared key (369 and 418). No reference key is missing. `cd_exactS` adds 9 scheduler counters (`obs_*`, `pg_cold_tail_calls`), all 0.
  - Relaunch of the missing registered cells into the same OUT dirs, `R1A_WINDOWS=g0,g1`: H 864446 ran 48 tasks, the 36 missing ones plus 12 skipped; L12 864447 ran 3 × 96. Every job COMPLETED. H holds 48/48 summaries and L12 288/288, with 0 failed cells in each. `.tasks.npy` is now written per cell for the f4f0453f cells.
  - Training 864264 COMPLETED with 0 failures (cache check PASS; the per-seed sidecheck passed for all four). W&B `gnn-slate-fix-v1`: gnn s3 sbl869no, s4 an3uwur1; twin s3 ftghmr6s, s4 wss6hqbw. md5 of the checkpoint / contract:
    - gnn s3 eed9b179 / 4880973a; gnn s4 cd67edba / 1a7c9f88;
    - twin s3 4fa561f8 / 43907c72; twin s4 797cdfda / 0d987922.
    - The copies in `inputs_12/inputs/models` match.
  - L34: 864486 (`--array=0-1`, 192 cells) at f4f0453f, `v2_L34_f4f0453f`.
- 2026-10-12 — **Harness fault and a launch fault; missing registered cells to be relaunched at a second pin** (accepted by the expert).
  - *Harness fault (not a rerun).* All 3 shards of L12 864293 FAILED at 11–40 min with 126 summaries landed. With the cell `name` shadowed, the 24 concurrent cells of a shard wrote one `HEROSIM_PG_BATCH_ALPHA.tasks.npy.partial`. One cell's `os.replace` removed another's (`FileNotFoundError` in `write_task_times`), and the `pool.map` exception ended the shard. H 864289 was exposed to the same race and was cancelled at 50 min with 23 summaries.
  - *Launch fault (S6).* Both launch lines left `R1A_WINDOWS` unset, so the harness ran its default windows g0–g3: 96 H cells and 576 L12 cells instead of 48 and 288. The g2/g3 cells are not registered. Their 34 (H) and 135 (L12) files were moved, unread, to `/share/nikola.lukic/slate_fix/v2_unregistered_g2g3_6a5c922c/{H,L12}`. They are kept, never read and never deleted. The reader would have refused them in any case (`check_grid` raises on any window outside g0–g1).
  - *Kept at 6a5c922c:* the registered cells that landed, H 12/48 and L12 67/288 (all moderate). Their summaries are valid, because the summary path is bound before the loop.
  - *Second pin:* `~/gnn-herosim-wt/sf2b_f4f0453f` at reference-physics f4f0453f, which carries the fix e3d5e069 and `tests/test_gate_name_not_shadowed.py`. A 2-cell identity on 16490 g0 heavy (864369 `ra_gnn_eng_nosplit` s1, 864370 `cd_exactS` s0, `v2_identity_f4f0453f`) must match the 6a5c922c cells on every shared key before the relaunch. The relaunch covers only the missing registered cells (H 36, L12 221) into the same OUT dirs, then L34, all at f4f0453f with `R1A_WINDOWS=g0,g1`.
  - *Reader* (dc84d3c0): accepts exactly the pins 6a5c922c and f4f0453f, refuses any other commit and prints the cells per pin.
- 2026-10-12 — **Disclosure: the per-cell `.tasks.npy` files from this gate are not usable.** At 6a5c922c, the alpha check loop in `run_one` (`scripts_cosim/fresh_topo_burst_v1_gate.py:1076`, from plan_rerank 6ad51eaf) rebinds the cell `name`. So line 1209 writes every cell's task times to one env-var-named file, and concurrent cells overwrite each other. The identity dir shows a single `HEROSIM_PG_BATCH_ALPHA.tasks.npy`. Summaries, including `steady_state.*` and `decile_mean`, use the path bound at line 749 and are unaffected. So are the primary, secondary, MP and descriptive families. `slate_fix_v2_read.py` reads summaries only. The `.npy` files from the identity, H, L12 and L34 are never read and support no later cut. The `task_times_rows` count in each summary describes rows written to the shared file, not a per-cell file. The fix (loop variable renamed, `tests/test_gate_name_not_shadowed.py`) is on rp/warm-surge e3d5e069 and goes into any future pin. v2 is not relaunched, by the expert's ruling (found in S5's warm_surge work).
- 2026-10-12 — **Cross-commit identity PASSED; gate launched.** Jobs 864268 (learned, s1) and 864269 (`cd_exactS`) at 6a5c922c, 8/8 cells on 16490 g0, both rungs, `/share/nikola.lukic/slate_fix/v2_identity_6a5c922c`. Every cell is SIM-IDENTICAL on all shared keys (385–434 per cell; timing and provenance excluded) with `smoke_A_4d90119c` or `identity_B2_d192081f`. No reference key is missing. Each cell has 9 new keys, all instrumentation: `steady_state.*` (7), `task_times_rows` and `env.HEROSIM_WARMTH_PHYSICS`. Submitted at the pin: H 864289 (`cd_exactS_ib0` s0, 48 cells, `v2_H_6a5c922c`) and L12 864293 (`--array=0-2`, 288 cells, `v2_L12_6a5c922c`). A first L12 submission, 864290, left R1A_SPLIT unset. All 3 shards stopped at the pre-cell checkpoint sidecheck (no `r1_attribution_v1_split.json` in inputs_12); zero cells ran. It was resubmitted with `R1A_SPLIT=accel_replica_v1_split.json`, as slate_fix_v1 used. L34 waits for training 864264 and its sidechecks.
- 2026-10-12 — **Signed** by the expert as drafted, adding the bar's transient-share line. Reader `scripts_cosim/slate_fix_v2_read.py` and `tests/test_slate_fix_v2.py` are committed before any gate cell. The expert writes the LINEAGES.md row (REGISTERED → ACTIVE at launch).
- 2026-10-12 — Drafted (S6) on the expert's request. Seeds 3–4 training submitted at 6a5c922c (job 864264, `--array=0-1`, `SEEDS="3 4"`) before signature, by the expert's order.
