# gnn_overcap_v1 — does the GNN's own plan beat the hand expansion on batches too big to enumerate?

**Status:** `CLOSED` (2026-10-12) — **HAND-FALLBACK-FASTER** (LOSS as registered). Registered 2026-10-11 by S7 on the coordinator's ruling; every bar below was fixed before any live data. The reader
(`scripts_cosim/gnn_overcap_gate_read.py`, tests in `tests/test_gnn_overcap_gate_read.py`; by-batch readout `scripts_cosim/gnn_overcap_batches_read.py`) was committed before any cell ran. Gate pin 9d19aebf (amendment below).

**Outcome.** At G = 8 (160 clients × 24 servers, 12 fresh topologies 19206–19217, 4 windows, 384/384 cells, 0 failed), serving the GNN's own self-search plan on over-cap batches (H = `ra_gnn_eng_ovg`; the within-cap path is `cd_exactS`'s)
is **not faster than `cd_exactS`**: **+0.24 %** at moderate (5/12 topologies ahead, Holm p = 0.30, a tie) and **+2.47 %** at heavy (1/12, Holm p = 0.0049, H slower). `WIN False, LOSS True, E_FASTER_RUNGS = both`.
The expansion fallback `cd_exactS` runs on over-cap batches beats the GNN's plan there. H ties its MP-OFF twin (−0.29 % / −0.45 %, 8/12 each, Holm p = 0.47), so there is **no message-passing effect**. H beats `cd` (−9.06 % / −4.79 %); `cd_exactS` beats `cd` by −8.84 % / −7.41 %.
The deficit scales with the over-cap share served (28.0 % of batches at moderate, 49.6 % at heavy; 0 fallbacks, served == predicted in every H cell): +0.2 % at 28 %, +2.5 % at 50 %. The secondary seat (current group size, g0, 5–7 % of batches over cap) ties: +0.46 % / +0.64 % (Holm p = 0.94).
The prediction written before data (H beats `cd_exactS` by about 1–2 % at heavy, under 1 % at moderate, P(WIN) about 0.25) **missed in sign at heavy**.

**Slate caveat.** H serves the declared top-5 slate (`declared_pruning_v1`) everywhere, within the cap and over it, and the checkpoint was trained on that slate. The declared slate sets cold = 0 on initialised platforms where the simulator charges the table cold cost, and its recall of the true best plan falls with batch size; this result is about
this checkpoint on that slate and says nothing about a model trained on the uncapped slate (`slate_fix_v1`, `rp/corpus-uncap`).

**By-batch readout (descriptive, g0 cells, buckets from H's over-cap log, applied to both arms; no 9+ batches exist at G = 8).** Gap in pp of `cd_exactS`'s mean latency, median over topologies:
- Heavy (+3.35 pp in all): over-cap batches +1.92 pp (62 % of the gap), within-cap batches +1.90 pp (38 %). Batches of 5–8 tasks carry +1.81 pp, of ≤ 4 tasks +1.51 pp.
- Moderate (+0.69 pp in all): over-cap batches **−1.45 pp** (H ahead; 5–8 tasks −1.45), within-cap batches +2.16 pp; batches of ≤ 4 tasks +2.42 pp, of 5–8 −1.87 pp.
- So at moderate the GNN plan is faster than the expansion on over-cap batches and the loss sits on small within-cap batches; at heavy it is slower on both. The within-cap loss (the exact search is the same in both arms) is probably a trajectory effect of H's earlier over-cap choices (not tested here).
- Stacking and cold starts, over-cap batches (H vs `cd_exactS`): stacked 99.5 % vs 90.4 % (heavy), 99.8 % vs 90.5 % (moderate); cold starts per batch 0.124 vs 0.160 and 0.233 vs 0.277. H co-locates more and cold-starts less on those batches yet is not faster.
- Disclosed: per-topology signs are mixed (heavy over-cap bucket ahead in 4/12 topologies, moderate in 9/12); the arms follow different trajectories, so a bucket's gap is not a per-decision effect.

Outputs on datalab `/share/nikola.lukic/gnn_overcap/`: `read_G8.{txt,json}`, `read_G0.{txt,json}`, `batches_g0_{heavy,moderate}.json`, `batches_read.txt`, per-cell summaries in `gate_G8`, `gate_G0`, over-cap logs in `gate_log`. The raw per-task results were deleted after the read.

**Why.** `selfsearch_gap_v1` (raw kept, 4 confirm-pool topologies, heavy g0, 1000 MB/s) split the loss of GNN self-search against `cd_exactS` by batch size.
- Self-search **wins** on batches of 9+ tasks (−0.39 pp opening, −0.80 pp rest, of `cd_exactS`'s latency, median over topologies); on those batches 72.5 % of
  `cd_exactS`'s searches are the expansion fallback, because 5^n plans exceed the 100,000-plan cap.
- It **loses** on batches of 8 or fewer (≤ 4: +0.96 / +1.18 pp, 5–8: +0.62 / +2.08 pp), where the exact search over S enumerates the whole top-5 slate.
- So the two have complementary strengths: exact S where it is exact, the learned plan where S's search is only a heuristic. 4 topologies, one window, descriptive.
- At G = 8 the over-cap population is large (28 % moderate, 49 % heavy of `cd_exactS`'s batches, `big_groups_s0_v1`), where the fallback enumerated about 4.4 billion plans.

**Distinct from `prune_g8_s0_v1` (S5), same lane.** `prune_g8` keeps `cd_exactS`'s machinery on over-cap batches: the GNN (or the cost order) only **orders** each task's slate, the cut slate
(top-k, largest k within 100,000 plans) is searched **exactly on S**, and the pass plan is kept unless the pruned optimum is strictly cheaper on S. The learned model's role is a pruning order, and the
decision is the argmin of the hand score. `gnn_overcap_v1` applies the GNN's **own plan** (its self-search argmax of the model's plan score) on over-cap batches, with no S, no pass and no expansion.
Different mechanism, different claim; neither result transfers to the other. Both run at G = 8 on disjoint topologies (18700–18711 there).

## Design (fixed now)

- **Environment (primary seat):** the `big_groups_s0_v1` G = 8 gate, unchanged except the topologies: 160 clients × 24 servers, p = 0.6, WF1 (W2 payloads + W3 access classes + W4 mix/repair, 1 s window),
  single origin spread over the 160 clients, `fastest_compatible`, pipelined transfers, `HEROSIM_REPLICA_RELEASE=1`, KPA, windows g0–g3, moderate and heavy.
  Inputs: `scripts_cosim/datalab/big_groups_inputs.sbatch` with `X1=/share/nikola.lukic/big_groups/x1_G8`, `M_MODERATE=11.6139`, `M_HEAVY=27.622665025860204`.
  **Check:** the inputs job's per-rung rate and group report must match the G = 8 gate's (6,250 groups of 8 per window; moderate 5.3470, heavy 12.7173 tasks/s) within 2 %.
- **Secondary seat (descriptive, no claim):** the same 12 topologies at the current group size (`X1=/share/nikola.lukic/big_groups/x1_G0`, the library's own groups), window g0 only, arms H, E, C, both rungs
  (72 cells). Over-cap there is 5–7 % of batches (`prune_g8_s0_v1`), so the expected effect is small; the seat is reported so the G = 8 result is not read as a statement about the current seat.
- **Topologies:** 12 fresh ids, **19206–19217**.
  - **Scan (S7, 2026-10-11).** `git grep` of 5-digit tokens 190xx–199xx over every local and remote branch (docs, `LINEAGES.md`, `scripts_cosim`, `experiments`, `tests`): only 19073, 19100, 19123, 19169, 19284, 19300–19306,
    19381, 19386, 19490, 19516, 19602, 19617, 19623, 19676, 19703, 19752, 19884, 19999 occur. Dataset directory names under `/share/nikola.lukic` (`ds_*`, which begin with a 5-digit prefix): the 190xx–199xx prefixes in use are
    multiples of 20 and 19000–19004, 19200–19204, 19400–19404, 19600–19604, 19800–19804, so 19205 and below, and 19206–19219, are free (`ds_172xx` is what forced the earlier move away from 172xx).
    A `/share` file-name scan for 192xx matches only those dataset prefixes (and 6-digit ids that contain 192xx inside a longer number); a content grep of selection/manifest JSON for 192xx found none.
  - **Disjoint from:** 19300–19305 (S4), 18700–18711 (S5, `prune_g8`), 19100–19123 (reserved for `plan_rerank`), 17007–17018 (`link_bound_s0_v1` gate), 16900–16911 (`big_groups_s0_v1` gate), 17100–17107, 16493/16503/16504/16507 (`selfsearch_gap_v1`)
    and the earlier pools listed in those nodes. The scan is token-based, not a proof; the prep job fails loudly if any `cc40s1920x`/`cc40s1921x` file or directory already exists under `/share/nikola.lukic`.
- **Arms** (cell = arm × topology × window × rung; 12 × 4 × 2 = 96 cells per arm, 384 in all):

  | arm | kind | seed | what it is |
  |---|---|---|---|
  | **H** | `ra_gnn_eng_ovg` | 1 | the no-split `gnn_eng` self-search plan seeds the CD refine; a batch within the 100,000-plan cap is refined and searched exactly on S as `cd_exactS` does; an over-cap batch is served the self-search plan **unchanged** (no passes, no expansion, no exact search) |
  | **T** | `ra_twin_eng_ovg` | 1 | the same with the MP-OFF twin's self-search plan on over-cap batches (`accel-replica-v1-twin-eng-seed1`, `disable_message_passing`) |
  | **E** | `cd_exactS` | 0 | exact search over S on the top-5 slate, expansion fallback above 100,000 plans |
  | **C** | `cd` | 0 | CD |

  - **Checkpoints:** `accel-replica-v1-gnn-eng-seed1.pt` (`/share/nikola.lukic/accel_prod/gate_inputs/models`, the one `big_groups_s0_v1` served as `ra_gnn_eng_selfsearch`) and the twin in
    `/share/nikola.lukic/selfsearch/twin_inputs_8c8ebc7f/models`; the inputs' `models/` directory symlinks both sets (as `prune_g8`'s). The harness sidecar check runs at every cell start; if T fails it, H vs T is "not run" and no
    message-passing claim is made. The two sidecars differ only in `cache_dir`, `disable_message_passing`, `mp_bipartite_edge_attr_zero` and `mp_bipartite_edge_conv` (checked 2026-10-11); corpus-build equality is **unverified**.
  - **Mechanism and code:** `rp/gnn-overcap` (off `rp/prune-g8` 8301f267). `HEROSIM_PG_CD_OVERCAP=gnn` makes `GnnCdRefiner.refine` test the batch's slate product (task slates of min(candidates, 5), a held-out task counting like any other, since the passes re-choose it)
    against the cap before the passes run, and return the seed when it is above. The predicate `pg_overcap_pred` is a dry counter that runs with the flag on or off.
  - **Learned arms use `R1A_SEEDS=1`; the rule arms seed 0.** The statement is about this checkpoint.
- **Identity, before the gate (one job, smoke cells, all counted and read):**
  - `cd_exactS` on 16900, g0, moderate with the new code reproduces the recorded `big_groups_s0_v1` G = 8 cell to the digit (mean latency and counters), so the flag-off path is unchanged.
  - **H with the over-cap path off** (`ra_gnn_eng_ovo`, the same shell with the expansion fallback): `pg_overcap_pred == pg_exact_fallbacks` exactly, so the predicate counts what the exact search calls over the cap.
    This cell is an identity cell and never enters the gate.
  - **H with the dry counterfactual** (`ra_gnn_eng_ovd`) equals H (`ra_gnn_eng_ovg`) on the same cell to the digit (mean latency and every non-dry counter): the dry S comparison is inert. The harness checks served == predicted over-cap batches and fallbacks 0 in every H and T cell.
- **Pre-gate descriptive comparison (item 6, cheap, no claim):** on the smoke cell the dry counterfactual runs, for every over-cap batch, the refine the fallback would have run (passes, expansion, passes) on the side, restores every counter, and
  records S of H's plan and of that plan. Reported: share of over-cap batches where H's plan is cheaper / equal / dearer on S, mean and median S gain, and the batch sizes. **Disclosed:** this is on the smoke cell's own over-cap batches
  (on-policy for H), not on S5's or `big_groups`' recorded G = 8 states: a GNN continuation of the replay was not built and S5 recorded 7 errored GNN replays, so no replayed RTT is attempted. The live gate runs regardless (rule 6).
- **Live gate** (`r1a` phase of `fresh_topo_burst_v1_gate.py` at the pinned commit; `r1a_gate.sbatch`; `LIMIT=20000`, `CELL_MEM=8G`, array of 12 tasks at 16 CPUs, PAR 8, `KEEP_RAW_DIR` and `OVERCAP_LOG_DIR` on `/share`):
  - Watchdog: a cell projected beyond 1.5 × LIMIT after 10 minutes is killed. A killed or failed cell is counted and reported, never rerun.
  - Raw per-task results are kept (needed by the by-batch readout) and deleted after the read unless cited.
- **Statistic:** per topology, the median over its 4 windows of the paired % of mean latency; the median over the 12 topologies; the sign count; the exact two-sided Wilcoxon. Negative = the first arm is faster.
- **Registered families (Holm within each family):**
  - **PRIMARY:** H vs E at moderate and heavy (Holm over 2).
  - **SECONDARY:** H vs T (message-passing attribution) and H vs C at both rungs (Holm over 4).
  - **Descriptive, not tested:** T vs E, T vs C, E vs C; the secondary seat.
  - **Readout (not tested):** gap by over-cap / within-cap and by batch size (≤ 4, 5–8, 9+) and phase, stacking and cold starts per batch (`gnn_overcap_batches_read.py`, g0 cells, buckets taken from H's over-cap log and applied to both arms); the over-cap share served by the GNN (batches and tasks).
- **Reading rules (fixed now):**
  - **WIN:** H's median is ≤ −2 % against E at both rungs **and** both Holm-adjusted p < 0.05 (`WIN`). The flag `E_FASTER_RUNGS` lists any rung where H's median is above 0; with a WIN it must be empty.
  - **LOSS:** H slower than E with Holm p < 0.05 at either rung (`LOSS`).
  - **Otherwise** the result is "not a win at the registered bar", reported with both medians and p-values (a direction faster at both rungs that misses −2 % or significance is said so, not rounded up).
  - Attribution: H vs T not significantly negative at either rung means "not message passing" (the plan score, not the graph, is doing the work).
  - **If E fails to beat C at either rung,** the environment is not the one `big_groups_s0_v1` measured and the gate is reported as NOT COMPARABLE.
- **Prediction, written before data:** H beats `cd_exactS` by about 1–2 % at heavy and by less than 1 % at moderate. A rough bound from `selfsearch_gap_v1`: self-search was about 3 % faster than `cd_exactS` on its over-cap tasks (−0.6 pp over about 21 % of tasks at the current group size),
  and over-cap batches hold about half the tasks at heavy G = 8 and under a third at moderate. The most likely outcome is the right direction at both rungs short of the −2 % bar (probability of a WIN about 0.25), H vs T a tie, H vs C large and negative.
  The risk is the checkpoint: it was trained at the current group size and `big_groups_s0_v1` found it 5.52 % slower than `cd_exactS` at heavy G = 8 across all batches; H uses it only where S cannot be enumerated, but the over-cap batches at G = 8 are larger than any it trained on.
- **Not claimed:** other seeds, other checkpoints, other group sizes (the secondary seat is descriptive), a retrained arm, the paper.
- **Jobs (≤ 6 counting the three test submissions already used):** tests × 3 (spent while writing the code), inputs + smoke/identity (1), live gate array (1), reader (1). The gate array is one submission of 12 tasks.
- **Cost, estimated:** `cd_exactS` G = 8 cells took 6.0–6.8 ks (max 8.0 ks); H and T skip the expansion fallback, so they should cost nearer the self-search cells (0.8–0.9 ks) plus the exact search on the within-cap half. About 1 Mcore-s,
  a few hours on 12 shards.

## Record (newest first)

- 2026-10-12 — **CLOSED, HAND-FALLBACK-FASTER (S7; coordinator's ruling).** Read once at pin 9d19aebf, when gate array 862627 (G8, 384 cells) and 862649 (g0 secondary, 72 cells) had finished, 0 failed, nothing killed. Numbers in the head. The three slowest cells (topology 19217, 2–2.7 h) were checked for stall and left to finish; walls were below 1.5 × the heavy median (7079 s). Prediction missed in sign at heavy. By-batch readout run once on the g0 cells (head), then the raw cells deleted.

- 2026-10-12 — **Amendment before any gate data (S7; pin accepted by the coordinator).** The live gate pins **9d19aebf**, not the signed d555e01a. Between them: the reader was made self-contained, a test expectation and the prep's base cfg were fixed (reader, tests, prep script only), and two code fixes came out of the smoke cells of prep 860573, which failed the harness's own checks before any gate cell was kept: (1) run provenance did not list `HEROSIM_PG_CD_OVERCAP` / `_DRY`, so the check read the env as unset (`src/executesimulation.py`); (2) the over-cap predicate counted a forced-placement task as 1, but the refine passes re-choose it, so it disagreed with the exact search on 5 of ~3150 batches (6 with the path off); it now counts slate sizes alone, and the predicate is also counted in the cd_exactS shell (it read 0 there). The scoring path and H's serving path are unchanged. **Prep submissions: 9** (3 test runs, 860531, 860542, 860549, 860573, 861711) against the limit of 6; every one before 861711 failed on my code or setup. Smoke 861711 passed: cd_exactS on 16900 g0 moderate reproduces the recorded G8 cell to the digit; rate +0.00 % vs the G8 gate; ovd == ovg in mean latency and every non-dry counter; with the path off predicted == exact fallbacks (3408, cd_exactS 3165); H served 3147 over-cap batches (25176 tasks), 0 fallbacks. **Seen before the gate, descriptive, one cell (19206 g0 heavy):** H 2.165 vs cd_exactS 2.069 (+4.6 %); on the 3147 over-cap batches H's plan is cheaper than the expansion plan on S in 193 (6 %), equal in 1189 (38 %), dearer in 1765 (56 %); median S gain -0.36 s, mean -10.9 s. The prediction above was written before this and is not revised.
- 2026-10-11 — **Registered** (S7), after the coordinator's ruling on `selfsearch_gap_v1` (self-search wins on batches of 9+, where `cd_exactS` falls back to the expansion 72.5 % of the time).
