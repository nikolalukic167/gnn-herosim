# prune_g8_s0_v1 — does a learned slate-pruning order beat hand expansion once half the batches are over the cap?

**Status:** `CLOSED` (2026-10-12) — **WIN-FOR-PRUNING / ORDER-NOT-LEARNED / NOT-MP.** Drafted by S5, ruled by the coordinator. Every bar below was fixed before any live data;
the reader (`scripts_cosim/prune_g8_gate_read.py`, tests in `tests/test_prune_g8_gate_read.py`) was committed before any cell ran.

> ⚠ **Bar moved (2026-10-12).** Contrasts here are to `cd_exactS`; the hand bar is now `cd_exactS_ib0` (`ib0_confirm_v1`, −4.73 / −2.27 % vs `cd_exactS`), and they were not re-measured against it.

**Outcome.** Pruning an over-cap batch to its top-k and enumerating that exactly beats `cd_exactS`'s expansion fallback at G=8, 12 fresh topologies × g0–g3, seed 1:
**−1.43 % moderate (11/12, Holm p 0.0049) and −2.13 % heavy (9/12, Holm p 0.0068).** The registered primary fired, so the label is WIN FOR LEARNED SEARCH ORDER by the
reading rules, and both attribution clauses fail: GNN order against cost order −0.26 % / +0.32 % (Holm p 1.0) and against the MP-OFF twin order −0.36 % / −0.05 % (Holm p 1.0).
The plain reading: **top-k pruning, by any order, of the over-cap batches beats the hand expansion fallback by 1.4–2.1 % at G=8. It is a hand-search improvement, not a GNN
result.** The cost order alone (arm B, no model) beats `cd_exactS` by −1.89 % / −2.11 %, so *`cd_exactS` plus a top-k prune on over-cap batches* is a new hand-bar candidate.
**My prediction, a tie with `cd_exactS`, missed.** `cd_exactS` beats `cd` by −9.55 % / −8.15 % (12/12), so the environment is the one `big_groups_s0_v1` measured.
**Slate caveat (resolved 2026-10-12):** the heavy deficit vs `cd_exactS` is a candidate-set handicap (`slate_bar_v1`: matched-slate, the GNN self-search wins −4.85 / −1.94 %); not an MP win.

**Why.** `cdxprune` cuts each task's top-5 slate to its top-k in over-cap batches and enumerates the cut slate exactly on S.
It was run at the current group size (`accel_replica_v1`, 12 topologies, g0, seed 1; harness fix e1cbffa8) and tied live.
- **Few batches were affected:** only 5.2 % (moderate) and 7.1 % (heavy) of batches were over the cap.
- **On S, the GNN order wins and the cost order loses:**
  - GNN-order pruning beat the expansion plan by +1.81 / +1.00 s per over-cap batch (3,530 better against 1,357 worse at moderate);
  - cost-order pruning lost to it by −0.76 / −0.45 s per over-cap batch (2,291 better against 2,876 worse);
  - the GNN top-k set differed from the cost top-k set in 92–94 % of over-cap batches.
- **Live, all of that vanished:**
  - GNN order against `cd_exactS`: −0.09 % moderate, +0.45 % heavy, neither significant;
  - cost order against `cd_exactS`: −0.30 % moderate, +0.35 % heavy, neither significant;
  - GNN order against cost order: −0.22 % moderate, −0.10 % heavy, neither significant.
- **At G=8 the over-cap share is 28 % (moderate) and 49 % (heavy) of `cd_exactS`'s batches** (`big_groups_s0_v1`), and its
  fallback enumerated about 4.4 billion plans. The record itself said "judge GNN search at G8". The per-batch S gain
  that did not reach the live latency at 5–7 % of batches now applies to 5–7 times as many batches. This is the one setting
  where a learned order has a large over-cap population to act on.
- **The bar is `cd_exactS`,** not CD: `cd_exactS` already beats CD by −9.15 % (moderate) and −7.71 % (heavy) at G=8
  (`big_groups_s0_v1`, 12/12 at both rungs), and the GNN's own self-search is slower than it at heavy (+5.52 %).

**Prior, disclosed: expect a TIE.**
- G4 tied on 12 topologies with a per-batch S gain of the sign we hope for.
- The S surrogate is nearly right (argmin-S misses the label optimum in 21 % of tables, tail-only), and `cd_exactS`'s expansion
  fallback recovers most of what enumeration loses.
- Cost-order pruning is worse than the expansion plan on S, so any win must come from the order being learned,
  not from enumerating a smaller space exactly.
- Two outcomes are reportable as results: a tie (the pruning order does not matter once the fallback is good) and a loss.

## Design (fixed now)

- **Environment:** the `big_groups_s0_v1` G=8 live gate, unchanged except for the topologies. 160 clients × 24 servers, p = 0.6,
  WF1 (W2 payloads + W3 access classes + W4 mix/repair, 1 s window), single origin spread over the 160 clients,
  `fastest_compatible` replica rule, pipelined transfers, `HEROSIM_REPLICA_RELEASE=1`, KPA, windows g0–g3, moderate and heavy rungs.
  - **Inputs:** `scripts_cosim/datalab/big_groups_inputs.sbatch`, with X1 = `/share/nikola.lukic/big_groups/x1_G8`, `M_MODERATE=11.6139`,
    `M_HEAVY=27.622665025860204` and `REPLICA_RULE=fastest_compatible`. These are the multipliers in the G=8 inputs' manifests.
  - **Check:** the per-rung rate and group report of the inputs job must match the G=8 gate's (6,250 groups of 8 per window;
    moderate 5.3470, heavy 12.7173 tasks/s) within 2 %.
- **Topologies:** 12 fresh ids, **18700–18711**.
  - **Scan method (S5).** Narrow token scan (job 858301, `/share/nikola.lukic/prune_g8/scan3.out`) of `/share`, `simulation_data`, `scripts_cosim` and `docs`
    for `cc40s`/seed/topology tokens and `ds_` names in ranges 1730x, 1870x, 1910x, 1930x: no hits (`names=[] seeds=[]`). The job was cancelled at 2 h 13 min with
    1950x and 1970x unscanned (progress rate; not rerun), and the ids are chosen from the 1870x range only. Cheap cross-checks: `git grep` over every local and
    remote branch (the only 187xx hits are unrelated counters in `gate_stats_summary.json`), `simulation_data/REGISTRY.json` and `LINEAGES.md`/`docs` (no hit),
    and the `/share` gate directory names (no hit). Earlier broader scans 858286/858287 were too noisy to use.
  - They are disjoint from 16900–16911 (`big_groups_s0_v1` gate), 17007–17018 (`link_bound_s0_v1` gate, S7), 16700–16707, 17100–17107
    and every pool the earlier scans listed.
- **Arms** (cell = arm × topology × window × rung; 12 × 4 × 2 = 96 cells per arm, 480 in all):

  | arm | kind | seed | what it is |
  |---|---|---|---|
  | **A** | `ra_gnn_eng_cdxpra` | 1 | cdxprune, GNN order: the no-split `gnn_eng` plan seeds the CD refine, and over-cap batches enumerate the GNN top-k exactly on S |
  | **B** | `ra_gnn_eng_cdxprb` | 1 | the same shell and seed plan, cost order (the first k of the standalone-cost order) |
  | **T** | `ra_twin_eng_cdxpra` | 1 | the same, with the order taken from the MP-OFF twin (`GNN_DISABLE_MESSAGE_PASSING=1`) |
  | **E** | `cd_exactS` | 0 | exact search over S on the top-5 slate, expansion fallback above 100k plans |
  | **C** | `cd` | 0 | CD |

  - **Checkpoints:** `accel-replica-v1-gnn-eng-seed1.pt` (the checkpoint `big_groups_s0_v1` served as `ra_gnn_eng_selfsearch`, in
    `/share/nikola.lukic/accel_prod/gate_inputs/models`). The twin is `accel-replica-v1-twin-eng-seed1.pt` with
    `disable_message_passing: true` and the same `psv5_inf` cache recipe, in `/share/nikola.lukic/selfsearch/twin_inputs_8c8ebc7f/models`.
    Both sidecars are checked by the harness's sidecar check; the inputs' `models/` directory carries the two sets.
  - **Mechanism:** over-cap batches only. `k` is the largest value with `Π min(len(slate_i), k) ≤ 100,000`. The pass plan is kept unless the pruned optimum
    is strictly cheaper on S. The expansion fallback runs on the side (dry, counters restored) to count differs / better / worse and the S gain. Code:
    `rp/cd-exacts` 8ddd1b8b with the harness fix e1cbffa8, merged into the `big_groups` line at 6d64bf9f.
  - **Learned arms use `R1A_SEEDS=1`.** The rule arms use seed 0.
- **Offline S screen** (no new simulations), on the `big_groups_s0_v1` G=8 states (`/share/nikola.lukic/big_groups/states`, 56 per rung, top-5 slate,
  the same snapshots and the `cd_exactS` decision replay):
  - per over-cap state: the S gain of the pruned optimum against the expansion plan, `S(expansion) − S(pruned optimum)`, by order (GNN, cost);
  - per rung: the over-cap share of states, and the share of over-cap states where the gain is positive, zero and negative.
  - **GO bar:** the median S gain over over-cap states is **> 0** at both rungs, **and** the GNN order's median is greater than the cost order's, at both rungs.
  - **Order is a tool, not a verdict:** the screen orders the work; the live gate runs whatever it says (rule 6).
  - **Declared fallback, before any data:** the cost order needs only the rule scheduler, so it is computed from the replay hook. The GNN order needs the
    checkpoint inside the replay. If a GNN continuation of the replay cannot be built and verified (one run of the dry 4-state check, no
    errors, the GNN order's counters equal the live gate's definitions) within one working session, the GNN-order screen leg is **not run**. It is read
    instead from the gate's own prune books (S gain per over-cap batch by order, in the same cells), and the node says so. The screen's GO bar is then
    evaluated on the gate's books, not as a pre-gate screen.
- **Live gate** (`r1a` phase of `fresh_topo_burst_v1_gate.py` at the pinned commit; `r1a_gate.sbatch`; `LIMIT=30000`; array of 24 tasks at 16 CPUs, PAR 8–12):
  - **Smoke first:** one heavy cell per learned arm (A, B, T) on the first topology. Only completion, timing and the instrument counters are read before the
    full read. The cell time sets the shard count and is recorded as an amendment (it cannot change the arms, the topologies or the reader).
  - **Identity check, before the gate:** the merged code's `cd` seed-0 cell on topology 16900, window g0, moderate must reproduce the recorded
    `big_groups_s0_v1` G=8 gate cell to the digit (mean latency and cell counters), because the arms C and E are the same code path as that gate.
  - **Watchdog:** the progress watchdog kills a cell projected beyond 1.5 × LIMIT after 10 minutes. A killed or failed cell is counted and
    reported, never rerun. A cell the harness rejects is a code fix after a rejection, not a rerun of a read, and is disclosed.
  - **Raw results are kept** (`KEEP_RAW_DIR` on `/share`), so a harness-check defect can be re-derived instead of rerun.
- **Statistic:** per topology, the median over its 4 windows of the paired % of mean latency; the median over the 12 topologies; the sign count
  (wins = topologies where the first arm is faster); the exact two-sided Wilcoxon. Negative = the first arm is faster. One seed per learned arm,
  so seed variance of the checkpoint is not measured, and the node says "this checkpoint".
- **Registered families** (Holm within each family; the reader prints the adjusted p):
  - **PRIMARY:** A vs E at moderate and heavy (Holm over 2).
  - **SECONDARY:** A vs B, A vs T and A vs C at both rungs (Holm over 6).
  - **Descriptive, not tested:** B, T, E each against C, B vs E, T vs E, T vs B.
  - **Twin (arm T):** the sidecar of `accel-replica-v1-twin-eng-seed1` differs from `gnn_eng` only in `cache_dir`, `disable_message_passing` and `mp_bipartite_edge_attr_zero`/`mp_bipartite_edge_conv`. Corpus-build equality is **unverified**. The harness sidecar check runs at every cell start; if T fails it, A vs T is recorded "not run" and no message-passing claim is made.
  - **Prediction, written before data: a tie with `cd_exactS`.**
- **Reading rules (fixed now):**
  - **WIN FOR LEARNED SEARCH ORDER:** A is faster than E at both rungs with Holm p < 0.05 (`PRIMARY_HOLM_SIG`). Attribution:
    A vs B significantly negative at both rungs means "learned order"; A vs T not significantly negative at either rung means "not message passing".
    A win is a statement about this checkpoint on this environment. A retrained arm is its own registered successor.
  - **TIE:** neither a Holm-significant win nor loss for A against E at either rung. The record then states: pruning to the top-k by the GNN order does not
    beat the hand expansion fallback even at G=8.
  - **LOSS:** A slower than E with Holm p < 0.05 at either rung. The decision-time cost (reported per task from the decision timing) is reported next to it.
  - **If E fails to beat C at either rung,** the environment is not the one `big_groups_s0_v1` measured and the gate is reported as NOT COMPARABLE, not as a result.
- **Not claimed:** other seeds, other group sizes, a retrained arm, the paper (not touched by this lineage).
- **Owners:** S5 runs everything above (node, reader, topology scan, inputs, screen, gate), subject to the queue limit below.
- **Cost, estimated:**
  - inputs about 40 s to 1 h; the screen about 1–2 h for the cost order and the GNN leg only if it builds;
  - the live gate is the dominant cost: the `cd_exactS` G=8 cells took 6.0–6.8 ks (max 8.0 ks), the self-search cells 0.8–0.9 ks, and a pruned-arm cell is expected at
    two to three times a `cd_exactS` cell. About 4 Mcore-s in all, so about 5–6 h on 24 shards.
  - **Queue:** at most 44 jobs counting S7's gate 858137 (4 tasks at the time of writing), so at most 40 of mine at any time.

## Record (newest first)

- 2026-10-12 — **Live gate read once (array 862599, 24 shards, 480 cells); WIN-FOR-PRUNING / ORDER-NOT-LEARNED / NOT-MP.** Read with `prune_g8_gate_read.py`, counts first; output `/share/nikola.lukic/prune_g8/read.json`.
  - **Counts:** 479 of 480 summaries. One cell failed: `ra_twin_eng_cdxpra` 18710 g2 moderate, killed by the progress watchdog (projected 47,464 s at 615 s, over 1.5 × 30,000 s); counted, never rerun. All other arm × rung cells 48/48, so T moderate is 47/48.
  - **Books:** for A, B and T `pg_exact_prune_batches` equals `pg_exact_fallbacks` (91.2k / 91.6k / 90.8k moderate, 153.9k / 154.5k / 152.7k heavy). Over-cap share of `cd_exactS` batches 29.5 % moderate, 51.2 % heavy. The pruned optimum differs from the expansion plan in 59.5 / 61.2 / 59.4 % (moderate) and 51.9 / 53.3 / 52.1 % (heavy) of over-cap batches (A / B / T); better : worse for A is 37,498 : 13,283 moderate and 51,885 : 21,829 heavy, B 33,021 : 20,791 and 48,498 : 29,849. `pg_batches` reads 0 in the prune arms, so their own over-cap share was not computed.
  - **Primary (Holm over 2), A vs E:** −1.43 % moderate (11/12, p 0.0024, Holm 0.0049); −2.13 % heavy (9/12, p 0.0068, Holm 0.0068). `PRIMARY_BOTH_RUNGS_FASTER` and `PRIMARY_HOLM_SIG` both true.
  - **Secondary (Holm over 6):** A vs B −0.26 % / +0.32 % (6/12, 5/12; Holm 1.0); A vs T −0.36 % / −0.05 % (7/12, 6/12; Holm 1.0); A vs C −10.29 % / −9.75 % (12/12 each; Holm 0.0029).
  - **Descriptive:** B vs C −9.83 / −10.11 %; T vs C −10.20 / −10.04 %; E vs C −9.55 / −8.15 % (12/12 each, so the E-beats-C condition holds); B vs E −1.89 % (11/12, p 0.0010) / −2.11 % (10/12, p 0.0122); T vs E −1.59 % (10/12, p 0.052) / −1.32 % (10/12, p 0.0093); T vs B +0.68 / +1.03 % (p 0.11 / 0.57).
  - **Decision time (descriptive, per-task mean of `decisionTiming`, moderate / heavy):** A 0.259 / 0.340 s, B 0.261 / 0.341 s, T 0.262 / 0.340 s, E 0.119 / 0.104 s, C 0.0001 / 0.0002 s. The prune arms cost about 2.5–3.3 × `cd_exactS` per task, and B equals A, so the cost is the exact enumeration of the cut slate, not the model.
  - **Disclosed:** the inputs job 858973 failed (`replica_placement_rule None, want 'fastest_compatible'`: the default base config lacks the rule); inputs were re-minted as 858999 with the G=8 config as `BASECFG`. The smoke ran arm A alone (859018, `R1A_ARMS` truncated at commas by `sbatch --export`); the other arms were smoked in 860021. Corpus-build equality of the twin is still unverified. The 2026-10-11 offline screen was NO-GO and the live gate ran regardless (rule 6); that screen's S ordering (cost order better than GNN order) is not what the live gate shows either, so S gain did not predict live latency here.
  - **Not claimed:** other seeds, other group sizes, a retrained arm, a GNN-specific or message-passing effect.

- 2026-10-11 — **Offline S screen, run before this node was signed; verdict NO-GO.** The GO bar was mine, written into the draft before any row was read, but it was not registered before data.
  Branch `rp/prune-g8-screen` (40a74d4b), 112 G=8 states (56 per rung). Over-cap share: heavy 41/55 = 74.5 %, moderate 24/50 = 48.0 %. Median S gain against the expansion plan is
  exactly 0 at both rungs for both orders. GNN order is worse than cost order on S (mean +0.91 / +1.16 s against +1.59 / +2.05 s, heavy / moderate), the reverse of G4. GNN top-k set
  differs from the cost top-k in 93 / 96 %. **Disclosed:** 7 GNN-order replays errored ("queued tasks did not run on their recorded replicas") and are dropped from both orders.
  Per rule 6 the live gate runs regardless.

- 2026-10-11 — **Drafted** (S5). Merge of `rp/cd-exacts` (cdxprune, S′, the input-route S′) into the `big_groups` line is 6d64bf9f on `rp/prune-g8`;
  the two copies of `_exact_score_fn` (cdxprune's and the self-search line's) were reconciled and `_self_search` kept; 52 tests pass in the merged tree.
  The MP-OFF twin exists for the accel seat: `accel-replica-v1-twin-eng-seed1/2` (also trained at G4: `accel-replica-v1-twin-eng-g4-seed1/2`), so arm T is included.
