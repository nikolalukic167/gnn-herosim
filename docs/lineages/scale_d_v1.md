# scale_d_v1 — does the llf_cd_confirm_v1 win survive 64 and 128 servers under option D?

**Status:** `REGISTERED` (2026-10-25, DRAFT v2 with the expert's rulings — not signed; nothing is minted or run). Drafted by S6 at the expert's request; SCIENTIST checks it before the expert merges. Every bar below is written before any scale-A or scale-B topology exists.
- **Question.** The static e40 link-feature GNN beat the CD bar and the parameter-matched wide MLP at 24 servers (`llf_cd_confirm_v1`, 4/4 CONFIRMED). Does that hold with the cluster 2.7× larger (scale A, 64 servers, primary) and 5.3× larger (scale B, 128 servers, descriptive), with clients per server and arrivals per server held at the 24-server values? Nothing is trained: the parent's `-final.pt` checkpoints (seeds 1–3) are served unchanged.
- **Outcome.** Not run.
- **What this is not.** Not a message-passing claim (no twin). Not a Knative claim (reactive is not run at these scales: under D it link-saturates, `llf_cd_confirm_v1` head). Not comparable to the earlier 64- and 128-server sets (`size64`, `size128`, 30129–30140 and 30241–30252): those used 160 clients, uncontended or TCP-off physics and the FL checkpoints. Never put a row of either in one table with a row here.

## Design

**Physics and recipe.** Option D as the parent: TCP on, fair share, default cellular uplink, cap200 payloads, `declared_pruning_v1_uncapped`, no split, k100, p = 0.6, single origin `x1_uncut`, `fastest_compatible`. Mint at pin `224dd02e` (`big_groups_inputs.sbatch` md5 `db71ec73…`, `uncut_groups_group_inputs.py` md5 `9efb1da0…`, then the cap200 clip, `cap200_build.sbatch` md5 `e05fbf3d1647037564bd9fe6e90532aa`), the recipe of `llf_mint.sh` with the two numbers below changed. Windows g0 + g1, both rungs.

**What "same ratio" fixes, and what it cannot.**

| | 24 (parent) | scale A | scale B |
|---|---|---|---|
| servers | 24 | 64 | 128 |
| clients | 160 | **427** | **853** |
| clients per server | 6.667 | 6.672 | 6.664 |
| M moderate | 11.6139 | **30.9704** (= 11.6139 × 64/24) | **61.9408** |
| M heavy | 27.622665025860204 | **73.66044006896054** | **147.32088013792108** |

- **Clients.** 160 × NS/24 = 426.67 and 853.33, rounded to 427 and 853. The rounding is 0.08 % and 0.04 %.
- **M is the arrival-rate multiplier only.** The x1 window's timestamps are divided by M (`big_groups_inputs.sbatch`, header and the `--rung` line); the client count only remaps the single origin over the clients (`scale_probe_remap_clients.py`). So arrivals per server match at M × NS/24, and clients per server match through NC. The earlier 64- and 128-server sets scaled M by NS/24 but kept 160 or 320 clients (2.5 clients per server), which is why they are a different design.
- **The windows have 50,000 tasks at every size, so a window shortens by 24/NS.** Measured last arrival (bar, g0): moderate 9,354 s at 24 servers; at 64 servers (old design, same M) 3,508 s; at 128 servers 1,754 s. Heavy 3,933 s, 1,475 s, 737 s. So the first 360 s (the scale-from-zero opening, about 40 % of the heavy mean latency at 24 servers: `live_statistic_includes_deploy_transient`) is 24 % of a 1,475 s window and 49 % of a 737 s window. **The whole-run statistic therefore weights the opening more at larger sizes.** The registered primary is the whole run, as in the parent, and the after-360-s row is reported beside every test as descriptive. A constant-duration design (tasks ∝ NS) would cost 2.7× / 5.3× the cell time and the builder takes a fixed `n50000` window; not proposed.
- **A bonus of this ratio.** The served checkpoints use `topology_feature_contract = src_index_v0` (sidecar). Its task feature is the source node's generation index divided by the total node count (`src/placement/topology_features.py:146`). With clients per server held, servers are 24/184 = 0.130, 64/491 = 0.130 and 128/981 = 0.130 of the nodes at every size, so that feature has the same distribution. With 160 clients at 64 servers it would not (0.286).

**Topologies.**
- **Scale A: 18 fresh ids. Scale B: 6 fresh ids** (expert's ruling: topologies are the n, so A takes 18 topologies × seeds 1–3, not 12 × 6). 24 ids in one run, minted as two roots.
- **Ids: PROVISIONAL until S5 confirms.** `knative_stable_v1` also needs 24 new ids ≥ 30351, so S5 allocates both in **one rescan**: `knative_stable_v1` gets the lowest free run of 24, and scale_d the **next** free run of 24 (18 for A, 6 for B). The rule is the parent's (TE3 literal rule, token range 19400–39999: git tokens on all remote refs, share and simdata names, provenance seeds). S6's earlier scan (30352–30382 free of tokens, 30351 the parent's lower bound) is superseded by that joint allocation and gives no ids here. The id list's sha256, the input md5s and the config checks (clients, servers, p, topology seed = id) are filed before any cell.
- **Mint check.** Each config asserts clients = 427 / 853, servers = 64 / 128, p = 0.6, topology seed = id, replica rule `fastest_compatible`. Mints at 24 servers take under a minute (jobs 906412–906423); I expect minutes here, with the `--time` raised to 4 h as a safety margin.
- **Admission (expert's ruling, decided before any data): the reactive admission is dropped at both scales; reactive is not run.** S7 showed reactive is link-saturated under D at 24 servers (queue share about 1e−4 while transfers take about 2,000 s), so its queue-share test carries no information, and its cells (median 8,444 s, up to 17,605 s at 24 servers) would put 12–30 h on the critical path. **Rule:** a topology is admitted only if its bar cells (`cd_ib0_inflightfull_k100` s0, g0 and g1 × both rungs, 4 cells) all have `queue_share` ≤ 0.80 **and** the bar completes the trace (no watchdog stall, no wall kill). Unknown is not a pass; only `queue_share` and completion are read. A killed cell is rerun once with a longer wall; a failing topology is replaced by the next free id and every replacement is reported. **The admission cells are the bar's primary cells, run first as a bar-only wave** (72 cells at A, 24 at B) and kept as the comparison cells, not rerun. (S6's reading of "in every admitted window": per topology, all four windows-and-rungs, as the parent admitted a topology on all its cells. The selection reads the bar's own outcome, which removes topologies where the bar runs away, so it works against a gnn-vs-bar win, not for it.)

**Arms** (the parent's, served as filed).

| Arm | Seeds | Checkpoint | Pin | Role |
|---|---|---|---|---|
| static gnn `ra_gnn_eng_combfl_linkstatic_e40_nosplit_k100` | 1–3 | `uncut-linkstatic-gnn-eng-e40-seed{1,2,3}-final.pt` | `02668ecf` | treatment |
| wide MLP `ra_mlp_wide_combfl_linkstatic_e40_nosplit_k100` | 1–3 | `…mlp-wide-e40-seed{1,2,3}-final.pt` | `02668ecf` | primary control |
| same-input MLP `ra_mlp_same_combfl_linkstatic_e40_nosplit_k100` | 1–3 | `…mlp-same-e40-seed{1,2,3}-final.pt` | `02668ecf` | descriptive |
| bar `cd_ib0_inflightfull_k100` | 0 | — | `0ec7cdcb` | primary control |
| HEFT `heft_network_batch` | 0 | — | `0ec7cdcb` | descriptive; charge 0.003035 ms moderate / 0.002807 ms heavy (`decision_cost_w.json`, expert's ruling) |

- **Checkpoints.** `xs/llf_conf_models_final.md5` (md5 `a126f46b5f30dd88ba43795d91c20ab3`) for seeds 1–3, the fixed set (expert's ruling). Seeds 4–6 (`xs/llf_conf_models_final_s456.md5`, sha256 `ae613e98…9b47`) are not served here.
- **Decision charge.** The parent's registered per-(arm, rung) values, unchanged at every size (the table in `llf_cd_confirm_v1`, "Decision charge"; bar 0.0116 moderate / 0.01098 heavy ms; the wide MLP up to 3× the gnn on groups over 50 tasks). **HEFT:** `HEROSIM_DECISION_COST_MS` 0.003035 (moderate) / 0.002807 (heavy), the node-B w in `decision_cost_w.json`.
  - **Does the cost grow with size?** The charge is a per-task constant, not a measurement, and it is not rescaled. The measured per-task decision time of the gnn (cell `decisionTiming`, `per_task_median_s`): 4.8 ms at 24 servers (this static checkpoint, option D, `fresh` g0 moderate s1 of `llf_cd_conf`), 4.3 ms at 64 and 6.4 ms at 128 (the FL checkpoint `ra_gnn_eng_combfl_nosplit_k100`, uncontended, g0 moderate s1); the median call is 23 ms, 23 ms and 32 ms. So the per-task cost does not grow at 64 and grows about a third at 128 for the FL gnn, a different checkpoint without the v8 link block; the static v8 checkpoint adds link tables that are O(V²·L) per batch (below), so this is **unmeasured for the arms served here**. Rule: every cell reports `decisionTiming`; the read quotes the measured per-task mean beside the charge for each (arm, size, rung). **A measured/charged ratio above 1.5 for the gnn that is not matched for the wide MLP is disclosed beside the gnn rows as charge-limited** (it favours the gnn). No re-charge.

**Cells.**
- Scale A: 18 topologies × 2 windows × 2 rungs = 72 per arm-seed; learned 9 arm-seeds → **648**; bar 72; HEFT 72; **792**.
- Scale B: 6 × 2 × 2 = 24 per arm-seed; **264**.
- No reactive cells. Admission uses the bar cells above (bar-only wave first).

## Pre-checks (before any wave; SCIENTIST reviews the output)

1. **Size-dependent features in the served static gnn (S6's audit, read from the code at `reference-physics` `d5b0acc9`).** One feature value depends on the node count, and it is matched by the client ratio above: `topology_features.py:146` (`src_index_v0`, source index / node count; sidecar `topology_feature_contract`). The queue features are `legacy_v0` with `queue_norm_mode = scheduler_adaptive`, the divisor being a p90 over the platforms (`src/executesimulation.py:503–522`), not a node count. The static link block (`src/placement/link_features.py:195–248`, `static_edge_columns`) sums over peers (group size, not server count), averages over each peer's candidate list (a mean, per `bipartite_aggr`'s rule) and takes a max over peers; it contains no server-count constant. `src/policy/gnn/link_static.py` has none. **Compute, not a feature:** `batched_tf.py:213–245` (`_link_tables`) allocates `[V, V, H]` and `[V, V, L]` arrays per batch, with V = candidate nodes and L = links on their routes, so the batched path grows about V²·L (about 2.7² = 7× at 64 and 28× at 128 against 24 servers, before L grows too). Only the smoke can say whether that is minutes or hours; it is not a correctness question.
2. **Serving lines.** From one cell log per arm per size, grep only the serving lines (`[PREFIX SERVING]`, the contract source, `[QUEUE FEATURES]`, the slate rule): `platform_cap=0`, no candidate cap or truncation, the same contract lines as at 24 servers (the serving check recorded in `uncut_groups_s0_v1` for the FL gnn at 64 servers). The count of server nodes in the infrastructure of one cell per size (a count, not a result).
3. **One-cell smoke at each size, before the waves** (g0 moderate, first topology of the size): bar, HEFT, gnn s1, wide MLP s1, MLP s1, with a wall of 24 h per cell at each size. Report wall seconds, `decisionTiming` mean and p95 per task, peak memory (`MaxRSS` from `sacct`), and that the cell completes (50,000 tasks, `endTime` finite, no `requestFailures` beyond the bar's). **Identity first**, at each serving pin, as the standing rule: the bar s0 on `cc40s30176` g0 moderate must reproduce its filed values (average elapsed 7.311068081724966, total_rtt 329884.30946025526), and one static gnn s1 cell of the parent (`cc40s30327` g0 moderate, pin `02668ecf`) must reproduce its filed summary, before any scale cell.
4. **Stop rules.** A smoke cell that does not complete within its wall stops that size; the expert decides whether to raise the wall or drop the size. A size is not run with a different recipe to make it finish.

## Primary family (scale A, 64 servers)

Four tests, **Holm over the 4**, read without runaways with the all-cells guard, labels and statistic exactly as `llf_cd_confirm_v1` ("Statistic", "Labels"; rulings file `live_link_features_v1/confirm_rulings.json` md5 `b609acef…`):
1. gnn vs bar, moderate; 2. gnn vs bar, heavy; 3. gnn vs wide MLP, moderate; 4. gnn vs wide MLP, heavy.

- A learned cell is the mean over seeds 1–3; the runaway arm median pools the three; a pair is dropped when either cell is flagged (runaway = more than 10× its own arm's median in the rung).
- **Power.** The exact two-sided Wilcoxon with n topologies all one sign has p = 2/2ⁿ: n = 12 → .0005, n = 9 → .0039, n = 8 → .0078, **n = 7 → .0156, above the smallest Holm threshold .0125.** If runaways leave a test with fewer than 8 topologies, it cannot be CONFIRMED however large the effect. In the parent the heavy tests kept 20 of 24 and 15 of 24 topologies; at 18 topologies the same drop rate would leave about 15 and 11. **Rule (adopted, expert):** a test with n_eff < 8 after the runaway drop is labelled `UNDERPOWERED` (reported with its median and count), not `NOT-FASTER` or `DIRECTION`.
- **Descriptive beside each test:** the all-cells row, per-seed medians, the after-360-s row, gnn vs same-input MLP, gnn vs HEFT, the runaway list per arm, the measured/charged decision-time ratios, and the wall seconds per arm.
- **Not run, not quoted:** reactive.

## Scale B (128 servers)

Descriptive, no label, no Holm: the same six rows, per-topology medians and counts, with n = 6 shown as counts only (the exact Wilcoxon floor at n = 6 is .03125, so no row can pass a threshold and none is tested). Direction only.

## Reader

- `llf_confirm_read.py`, mode `cd`, gains `--scale` / `--ids` parameters; frozen by blob and filed with the rulings before the first cell is read; the read runs once, A first. It checks the id-list sha256, each cell's pin per group, each learned cell's `env.GNN_MODEL_PATH` basename and charge string against the tables above, and **`servers` and `clients` in every cell's summary against the table above (a cell at the wrong size fails loud)**. The default mode and the 24-server `--mode cd` invocation stay byte-for-byte (a regression test on the parent's cells, as in the parent).
- Cells live in a root separate from the 24-server cells and from each other (`scale_d_v1/A/…`, `…/B/…`); no cell is pooled across sizes or with the parent.

## Timing and queue (S6's estimate; the smoke replaces these)

Measured at 24 servers under D (`llf_cd_conf`): gnn median 359 s (moderate) and 379 s (heavy) per cell, wide MLP 390 / 464 s, MLP 310 / 340 s, bar 49 / 65 s (max 2,685 s), reactive 8,444 s (max 17,605 s; not run here). The only larger-size learned cells on file are the uncontended FL gnn cells: median 423 s at 64 servers and 622 s at 128 (`size64`, `size128`, `cells_learned`, 24 cells each), a different checkpoint and physics, so only an order of magnitude against 359–379 s here. I use 1.5× (64) and 2.5× (128) over the 24-server D values, an assumption until the smoke runs.

| step | scale A (64) | scale B (128) |
|---|---|---|
| mint (3 jobs per root, `bg-in` → `llf-grp` → `cap200`) | under 30 min | under 30 min |
| smoke (5 cells, 1 job, 24 h wall per cell) | 1–2 h | 1–2 h |
| admission: bar-only wave (72 / 24 cells, 4 shards; also the comparison cells) | 20–60 min (24-server bar median 65 s, max 2,685 s) | about 20–40 min |
| learned wave: 12 shards (3 seeds × g0/g1 × moderate/heavy) of 54 cells (3 arms × 18), 16 cpus, 8 cells at a time | 1–2.5 h per shard, 3–5 h wall at 4 nodes with 2 shards each | 4 shards of 18 cells, about 1–2 h |
| HEFT (72 / 24 cells, 2 shards each) | under 1 h | under 1 h |

- **Critical path (no reactive).** Mint → smoke → bar admission → (any replacement mints and bar cells) → learned wave: about 6–9 h at scale A, 4–6 h at scale B, each if the smoke cells finish near the 1.5× / 2.5× assumption. The smoke is what can break it (the `_link_tables` size growth in the pre-checks).
- **Queue.** Scale A: 4 bar + 12 learned + 2 HEFT = 18 jobs; scale B: 2 + 4 + 1 = 7; smoke 2; mint 6. The waves are sequential per scale, so at most about 20 jobs at once; on top of the other sessions' jobs: **count before every submit; the cap of 44 is shared with `knative_stable_v1` and HEFT.** Spread over about 4 nodes, at most 2 jobs per node; nodes 21 and 15 are excluded for the w calibration only, not for cells.

## Prior (drafter's; SCIENTIST owns it)

Per test, scale A: gnn vs wide MLP moderate ≈ 0.5; gnn vs bar moderate ≈ 0.5; each heavy test ≈ 0.1 (runaways and the n < 8 rule). The 24-server effect sizes were −21 % and −9 % at moderate; the 64-server FL set (a different design) had the FL gnn ahead of its MLP by 5 % on request mean.

## Rulings (expert, 2026-10-25, before any data)

1. **Power over seeds.** Topologies are the n: scale A is 18 topologies × seeds 1–3 (648 learned cells), scale B 6 × seeds 1–3 (descriptive).
2. **Admission** on g0 + g1 only, and by the bar's own `queue_share` ≤ 0.80 plus completion; reactive is not run (see Design).
3. **HEFT charge** per rung: 0.003035 ms moderate, 0.002807 ms heavy.
4. **`UNDERPOWERED`** adopted for n_eff < 8.
5. **Charge-limited flag** adopted at a measured/charged ratio above 1.5 for the gnn that the wide MLP does not match.
6. **Ids** provisional until S5's single rescan allocates `knative_stable_v1` (24) and scale_d (the next 24) together.

## Record

- 2026-10-25 — **Draft** (S6). Not signed. Sources read: `llf_cd_confirm_v1` (design, charges, reader), `big_groups_inputs.sbatch`, `size64_mint.sh` / `s128c320_mint.sh` (the old M convention), the cell summaries of `llf_cd_conf`, `size64` and `size128`, and the code paths named under Pre-checks.
- 2026-10-25 — **Draft v2** (S6): the expert's rulings applied (18 topologies at scale A, bar-based admission without reactive, HEFT charges, `UNDERPOWERED`, 1.5× flag, joint id allocation). Cell counts: A 792 (648 learned + 72 bar + 72 HEFT), B 264.
