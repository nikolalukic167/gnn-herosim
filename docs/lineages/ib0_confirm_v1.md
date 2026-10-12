# ib0_confirm_v1 — does the hand `ib0` surrogate beat `cd_exactS` on fresh topologies, and does a GNN seed add anything on top of it?

**Status:** `REGISTERED` (2026-10-12, signed by the coordinator at 62e51229 with the amendment below). Owner S4. Parent: `plan_rerank_v1`
(`SUPERVISED-CEILING-AT-BAR`; the Phase 0 numbers below are quoted from its Record, read once, 12 oracle cells).

## Why

Phase 0 gave two facts on topologies 19300–19305, g0, accel seat:
- the myopic label oracle, served as far as it can enumerate, is −0.92 % (moderate) / −4.17 % (heavy) vs `cd_exactS`, and covered 61 % / 56 % of
  tasks only;
- the hand surrogate `cd_exactS_ib0` (`HEROSIM_PG_BATCH_ALPHA=0`: S with only the in-batch stacking term zeroed, the platform's backlog kept) is
  −4.60 % (6/6) / −3.18 % (6/6) vs `cd_exactS` at g0, −3.72 % / −2.34 % at g0+g1, with no replay at all. It was a descriptive arm, not a registered
  contrast, and it was read on the same six topologies that framed the question.

This node confirms the `ib0` lead on fresh topologies as a registered contrast, and gives the GNN one fair shot on top of it. It also adds the
full-coverage oracle that Phase 0 could not be (the ceiling of any batch-label learner against `ib0`).

## Seat and cells

The accel seat, unchanged: 160 clients × 24 servers, p 0.6, `fastest_compatible`, WF1 windows **g0–g3**, rungs moderate (×11.6139) and heavy
(×27.622665025860204), pipelined transfers, `HEROSIM_REPLICA_RELEASE=1`, KPA. 12 fresh topologies **19405–19416** (`FRESH` below).

- **Disjointness check (2026-10-12):** `git grep` over every local and remote branch (`docs`, `LINEAGES.md`, `scripts_cosim`, `REGISTRY.json`) and the
  `/share/nikola.lukic` and `~/gnn-herosim/simulation_data` names. `19400–19411` first: the only hit is `rp/gnn-overcap`'s
  `gnn_overcap_v1.md:31`, which lists **19400–19404** (with 19000–19004, 19200–19204, 19600–19604, 19800–19804 and the multiples of 20) among used
  topologies. So 19400–19404 are excluded and the range moves to 19405–19416 (no multiple of 20 in it); 19412–19416, checked by the same `git grep` over every branch: no hit.
  Every range the coordinator named is disjoint from it (19206–19217, 19300–19315, 18700–18711, 17007–17018, 16900–16911, 16301…),
  as are 19100–19123 (kept for any later gate) and Phase 0's 19300–19305.
- Cells per arm: 12 × 2 rungs × 4 windows = **96**. Windows of one topology are not independent; the unit is the topology.

## Arms

Classical, all `peer_greedy_network_cd` family unless noted; the surrogate is `HEROSIM_PG_BATCH_ALPHA` (batch α), λ = 1 − α.
- `cd_exactS_ib0` (**ib0**): exact search over S with batch α = 0 (λ frozen at 0, set before data; not tuned on these topologies).
- `cd_exactS`: the registered S (α = 1). The reference.
- `CD`: the coordinate-descent greedy.
- `cdxapply_ib0`: the GNN seed (accel `gnn-eng` g4, seed 1; no-split declared slate) refined by the CD expansion, **scored on the ib0 surrogate**.
- `cdx_ib0`: the same expansion, same pass budget, same ib0 surrogate, **seeded by the hand plan** (the CD greedy's plan) instead of the GNN. The hand-seed control.
- Descriptive, in no family, g0–g1 only:
  - `cd_exactS_a05`: batch α = 0.5 (λ = 0.5), to show 0 is not a knife-edge;
  - `cd_exactS_a0`: whole queue term to 0 (Phase 0's descriptive arm), to replicate its heavy giveback on fresh topologies;
  - `cdxapply_ib0_twin`: the MP-OFF twin of the seed. **The checkpoint exists with sidecars** (`accel-replica-v1-twin-eng-g4-seed1.pt` + `.contract.json`
    and `-final`, seeds 1 and 2, in datalab `models/`); it is trained on the same corpus and recipe as `gnn-eng-g4`, so it can serve as the twin.
    Sidecar content is checked at smoke, not assumed.

**Code needed, none touching physics, none changing a flag-off run.** (a) The alpha validation in `_pg_init` (`peer_greedy_network/scheduler.py:376`)
allows only exact-search `peer_greedy_network_cd`; it is relaxed to also allow `gnn_cd_refine` and plain `peer_greedy_network_cd` without the exact
search, whose refine already calls the alpha-weighted `_pg_choose` / `_pg_plan_cost` (`GnnCdRefiner` binds both). (b) Harness kinds `cdxapply_ib0` /
`cdx_ib0` (+ twin) set the env as the existing `_cdxapply` / `cd_expand` do plus the alpha. (c) Tests: the alpha is refused where undefined; flag-off
identity of `cd_expand` and `cdxapply` on the same 2 cells (byte-identical results). `cdxapply_ib0` is therefore **more than an env flag**, by (a) and its
test; it is small and is the only way to have the arm. If you rule that arm out, the secondary family drops to nothing, so I keep it.

## Bars (sealed; reader committed before data)

Statistic: per topology the median over its 4 windows of the paired % of mean latency; median over the 12 topologies; sign count; exact Wilcoxon
(n = 12: p floor 0.0005). Negative = first arm faster. Unit: topology.

- **Primary family (Holm over 2 rungs): ib0 vs `cd_exactS`.**
  **WIN:** median ≤ −2 % and Holm p < .05 at **both** rungs, and ib0 faster on ≥ 9 of 12 topologies at each rung (as Phase 0's sign condition, scaled).
  A rung where `cd_exactS` is Holm-faster is a loss. Otherwise `TIES` or `LOSES`, named by the sign.
- **Secondary family (Holm over rungs and the two contrasts, four tests): the GNN seed.**
  - `cdxapply_ib0` vs `cdx_ib0` — credits the GNN **seed** (same search, same surrogate, same budget).
  - `cdxapply_ib0` vs `ib0` — does the GNN-seeded non-exact search beat the exact search on the same surrogate.
  - GNN seed WIN: `cdxapply_ib0` faster than `cdx_ib0` with median ≤ −2 % and Holm p < .05 at both rungs. The MP claim needs the twin to lose to the GNN
    seed on top of that, and the twin is descriptive, so **no MP claim is available from this node**.
- **Descriptive, no bars:** a05 vs ib0 (is α = 0 on a flat floor), a0 vs ib0 (queue giveback at heavy), twin vs `cdxapply_ib0`, ib0 vs CD, buckets
  (queue / cold / exchange pp of latency, as Phase 0), per-rung stacking, first-quarter share of the gap (raw files kept this time, `HEROSIM_RAW_DIR` on `/share`).
- Quote the checkpoint with every GNN number: `gnn-eng-g4` seed 1, name both corpora (1670 / 516 in the older record; this one is the accel corpus).

## Full-coverage oracle (decisive for the supervised family)

Phase 0's oracle covered 56–61 % of tasks. This stage closes that hole against **ib0** (not `cd_exactS`):
- covered batches (≤ 625 plans): the exact label (fidelity replay, whole group), as Phase 0, argmin applied only when strictly cheaper than ib0's plan;
- **over-cap batches: α-expansion local search on true co-sim RTT, seeded from ib0's plan**, replay-scored, with a per-batch replay budget (1,500, set before
  data); a lower bound on the label headroom, never an upper one;
- unplaced-partner batches pass through ib0 unchanged, counted (Phase 0: 1.3–3 % of tasks);
- cells: **6 of the 12 topologies (19405–19410), g0, both rungs = 12 cells**, ≥ 60 replay workers on **every** cell (moderate too), one job per heavy cell.
  Contrast: oracle vs ib0 on the same cells, per rung.
- **Coverage rule:** replay-labelled tasks (covered + searched) must be ≥ 95 % of tasks at each rung; else the reading is `INCONCLUSIVE`.
- **Pre-registered rule:** if the full-coverage oracle is **< 3 % faster than ib0 at moderate** (median over the 6 topologies of the paired %), the
  supervised batch-label family for the accel seat closes entirely (any scorer trained on batch labels is bounded by the oracle). At ≥ 3 % the family stays open
  and the oracle number is the headroom a Phase 1 learner would be sized to; heavy is reported alongside and cannot rescue moderate.
- Code needed: the over-cap local search in `plan_oracle` / `_pg_oracle_apply` (the current hook falls through above the cap), the ib0 slate ranking in the
  oracle arm, tests (flag-off identity as before, search-budget accounting, books close).

## Predictions, written before any data

- **ib0 vs `cd_exactS`:** −3 to −5 % at moderate, −2 to −3 % at heavy, both rungs a WIN at 12 topologies. Phase 0's 6-topology figures (−4.6 / −3.2 %) are
  expected to shrink on fresh topologies (selection on the framing set), not to vanish.
- **GNN seed:** a tie. `cdxapply_ib0` vs `cdx_ib0` within ±1.5 % at both rungs, `cdxapply_ib0` vs ib0 +1 to +4 % slower (a non-exact search vs the exact one, same surrogate).
  The twin ties the GNN seed. No MP claim.
- **a05 vs ib0:** within ±1 %; a0 vs ib0 tied at moderate and slower at heavy (Phase 0: +1.8 % vs the oracle, queue giveback at heavy).
- **Full-coverage oracle vs ib0:** 1–2.5 % faster at moderate, 2–4 % at heavy: short of the 3 % moderate line, so the supervised family closes. Basis: ib0
  already collects 4.6 % of the oracle's gain at moderate (the oracle was 0.9 % vs `cd_exactS`), so little batch-label headroom is left over it; the over-cap
  local search is a lower bound and can only add what it finds.

## Cost and queue

Measured on the Phase 0 classical cells (wall, `wallclock_s`, 4 per node, median of 12): `cd_exactS` 2,150 s (moderate) / 2,284 s (heavy), ib0 2,398 / 2,329 s,
a0 2,504 / 2,396 s, CD 93 / 70 s. `cdxapply`-type cells are **not measured here** (estimated 15–30 min plus GNN inference; one smoke fixes it before launch).

| Arm | Cells | Cell-hours (est.) |
|---|---|---|
| ib0, `cd_exactS` | 2 × 96 | 2 × 96 × 0.63 ≈ 121 |
| CD | 96 | ≈ 3 |
| `cdxapply_ib0`, `cdx_ib0` | 2 × 96 | ≈ 2 × 96 × 0.4 ≈ 77 |
| a05, a0, twin (g0–g1) | 3 × 48 | ≈ 2 × 48 × 0.63 + 48 × 0.4 ≈ 80 |
| **Classical total** | 480 + 144 = 624 | **≈ 280** |
| Full-coverage oracle | 12 | heavy ≈ 4–6 h × 6, moderate ≈ 2–3 h × 6 at 60 workers (Phase 0: 2.4 h for the pair at 60); over-cap search adds an unmeasured amount, bounded by the 1,500-replay budget |

- Classical at 8 jobs × 64 CPUs × 4 cells per node (32 concurrent): **≈ 9 h wall**, plus queue wait. **g0–g1 only (48 cells per primary arm): ≈ 5 h.**
  My recommendation: **g0–g3** for the primary and secondary families (the unit is the topology, and four windows tighten each topology's median); descriptive arms stay at g0–g1.
- Order: classical arms first (primary arms before descriptive); the oracle stage after the classical read-in, 12 cells in 6–12 jobs.
  The oracle's own code lands and is smoked (flag-off identity, 2 cells) while the classical arms run.
- **Watchdog:** progress-rate projection with `LIMIT` 36,000 s and the 1.5× kill, as Phase 0. Any heavy replay cell is given ≥ 60 workers at launch
  (the Phase 0 lesson); a kill at that setting is a real failure, counted, not rerun. A failed launch (resources) is not a rerun, and only while no output has been read.
- **Queue:** ≤ 44 in total (S5's 24-shard array and S7's array are running; I will count at submit and shard to what is free). This lineage gets at most 8 jobs;
  the classical stage 8 shards, the oracle stage replaces them. Pinned worktree per job, raw files on `/share` (`HEROSIM_RAW_DIR`).

## Pre-launch checks (all before any cell counts)

1. Smoke: one heavy and one moderate cell per new kind (`cdxapply_ib0`, `cdx_ib0`, twin) on a topology outside `FRESH`, instrument counters only; labelled a smoke,
   not read.
2. Flag-off identity: `cd_expand` and `cdxapply` on 2 cells byte-identical to the parent commit; ib0 reproduces Phase 0's 19300 g0 cell (it is the same code).
3. Reader committed first, with the arms, expected windows (`EXPECTED_WINDOWS`), coverage books and the bars above; its tests green.

## Record

- 2026-10-12 — **Smokes (topology 19300, g0, moderate + heavy; smokes, not read) and classical launch.** Pin `ib0_d192081f` (d192081f).
  Flag-off identity: `cd_expand` and `ra_gnn_eng_cdxapply` at c7ceec53 vs d192081f identical on all 4 cells (summary fields and search counters).
  Hand-seed identity: `ra_gnn_eng_cdxhib0` = `cdx_ib0` on both rungs (moderate total_rtt 111039.48456016854, endTime 9355.114119122141; heavy 97014.7066934928,
  3998.2349206443687; expansion batches/sweeps/moves/evals equal: 12099/14665/4271/448708 and 11997/14528/4371/587578). Per-task raw files were not kept, so
  the identity is on the summary and its counters. Measured wall: `cdx_ib0` ~80 s, `ra_gnn_eng_cdxib0` ~290 s per cell (the 0.4 h estimate was high).
  Launch: main arms g0-g3 in 5 shards (862757-862761), descriptive arms g0-g1 in 2 shards (862762, 862763), inputs from 862754, out
  `/share/nikola.lukic/ib0_confirm/gate_d192081f`, LIMIT 36000.

- 2026-10-12 — **Signed (62e51229), three notes added before any code runs on data (coordinator).**
  1. The oracle stage runs whatever the classical read shows, an ib0 NO-WIN included. Its closing rule is against ib0 as served. If ib0 does not confirm,
     the oracle vs `cd_exactS` on the same cells is also reported, descriptive.
  2. **`cdx_ib0`'s hand seed, named exactly:** the plan the CD greedy's first pass builds on the batch (`_pg_batch_pass(refine=False)` in task-id order, the plan
     `_cd_native_plan` starts from), on the ib0 surrogate. `cdx_ib0` is the plain CD scheduler with `HEROSIM_PG_CD_EXPANSION=1` and batch α = 0 (the `cd_expand`
     flow: that first pass, `HEROSIM_PG_CD_PASSES` = 3 refine passes, one expansion phase, the passes again). `cdxapply_ib0` is the GNN scheduler's seed refined
     by `GnnCdRefiner` (3 passes, one expansion phase with the same sweep and mover limits, the passes again). The budget is identical by construction: the same
     env limits are read in `_pg_cd_expansion()` by both, and the counters `pg_expand_batches`, the expansion sweeps and `pg_cd_passes` are exported per cell and compared.
     **Identity smoke before launch:** `GNN_CD_SEED=hand` replaces the GNN's plan by that hand seed inside the refiner (kind `_cdxhib0`); on one cell
     (a topology outside the 12) `<gnn>_cdxhib0` must equal `cdx_ib0` bit for bit (total RTT, task results, search counters). If it does not, the difference is reported
     and `cdx_ib0` is redefined as the `_cdxhib0` path itself, with the reason in the node, before any gate cell.
  3. The alpha-validation relaxation (`gnn_cd_refine` and plain CD without the exact search) gets a flag-off identity test for `gnn_cd_refine` and for plain CD
     (alphas at 1 reproduce the unrelaxed result byte for byte). A worktree is pinned at the launch commit.
  - Cell windows as recommended: g0–g3 for the primary and secondary families, g0–g1 for the descriptive arms.

- 2026-10-12 — Drafted from the coordinator's brief and SCIENTIST's review (λ frozen at 0 with an α = 0.5 control, a hand-seed control `cdx_ib0`, the MP-OFF twin,
  the full-coverage oracle with a pre-registered closing rule). Not signed.
