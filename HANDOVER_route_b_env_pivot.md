# Handover — route_b_env_pivot_v1: ladder interrupted mid-H0; debug list + past-LINEAGES audit

**Date:** 2026-08-27
**Branch:** `feat/network-contention-v1` at `90fb6e0`, **not pushed** (17 commits ahead of origin).
**Registration:** `ROUTE_B_ENV_PIVOT_SCREEN.md` @ `019bdcb`, signed off; LINEAGES entry @ `6b7b915`.
**Session end state:** two background executors were killed by session teardown mid-ladder.
Uncommitted: `simulation_data/route_b_pivot_h0_rtt.json`, `route_b_pivot_h0_ctrl_rtt.json`,
`route_b_pivot_h0_reading.json` (PARTIAL — no transfer/verify ran for H0). Everything else
committed, tree otherwise clean.

## 1. Where the ladder stands

- Presets conform to the registered seed blocks (`90fb6e0`; H0 3001–3017 … H3 3301–3317).
  The wrong-seed (901–917) H0 corpus was deleted and regenerated BEFORE any reading — the
  deviation_log inside `route_b_pivot_h0_reading.json` records this; no reading is
  contaminated.
- Generated clean under the corrected env (`HEROSIM_COSIM_KEEP_ALIVE=1000000`,
  `HEROSIM_RETAIN_TASK_TIMES=1`, + Arm-S vars on mains, skip cap 2,000,000):
  `gnn_datasets_route_b_pivot_h0` (204/204, seeds 3001–3017),
  `gnn_datasets_route_b_pivot_h0_ctrl` (204/204),
  `gnn_datasets_route_b_pivot_h1` (204/204, seeds 3101–3117; **no ctrl yet**).
- H0 + ctrl were scored (α 1.5/2.0/3.0 + anchor). **Not done for H0:** transfer
  (`--add-linkrank --extended-blocks`), the three verifier passes, S2–S4. H1 not scored.
- H2/H3 corpora not generated.

## 2. Three anomalies in the H0 partial reading — debug BEFORE reading any bar

From `route_b_pivot_h0_reading.json` (`counters_by_alpha_*`):

1. **α=1.5 is 204/204 `no_feasible_rows` on BOTH main and control.** Whole rung
   infeasible at the tight end. Expected under per_server=1 × 4 servers? Verify against
   `cap_node` arithmetic before shrugging.
2. **`greedy_stuck` ≈ 50% at every constrained α** (main: 95/204 @2.0, 87/204 @3.0;
   ctrl similar). Under the registered feasibility fallback (dirty counters ⇒ read at the
   tightest clean α; none clean ⇒ VOID-INFEASIBLE), **H0 as configured likely reads
   VOID-INFEASIBLE** — which the registration permits (record, continue to H1). But first
   confirm the stuck datasets genuinely have no greedy-feasible completion vs a
   greedy-order artifact, and check how `n_scored`/denominators treat stuck datasets
   (firing fractions are over scored-only — is that the registered denominator?).
3. **The "separable" control has nonzero constrained R_exact** (frac>1% = 0.155 @2.0,
   0.067 @3.0; anchor clean at 0.0). For truly separable physics R_exact ≡ 0 under ANY
   feasibility restriction (scorer docstring; positive control 1). So either
   (a) `node_disk_v2` warmth couples co-located tasks through disk/image state and the
   squeeze (4 servers, per_server=1) makes that coupling material — i.e. the control was
   never fully separable, stage 1's B0 just never squeezed hard enough to see it — or
   (b) a scorer/cap artifact. If (a): the S0 control bar (frac_gt_1pct ≤ 0.02) may be
   unpassable for ANY squeezed rung and the registration's control definition needs an
   amendment (logged per §6), e.g. warmth-neutral controls. Decide with data: score the
   ctrl at looser server counts, or diff a firing ctrl dataset's sweep against additivity
   directly. **Do not weaken the bar silently.**

## 3. Bugs surfaced this session (fixed) — and the past-LINEAGES audit each one owes

Run these audits before trusting the past rows; record findings in LINEAGES (amend in
place where a recorded claim is touched, per house style).

1. **Import-time reseed clobber** (`prepare_graphs_cache.py` module-level seed(42);
   fixed `309f199`; voided the first A1 sweep — recorded in the NO-GO row).
   **Audit:** did any OTHER trainer/harness historically import `prepare_graphs_cache`
   (or any module with module-level seeding) at run time? If yes, past multi-seed GNN
   studies (`gnn_draw_study_v1` 8-draw claim "2/8 collapse"; the p5b GNN arms; any
   wandb-sweep) may have had clobbered draws.
   `grep -rn "import prepare_graphs_cache\|from src.notebooks.prepare_graphs_cache" --include=*.py . archive/` and
   `grep -rn "^random.seed\|^np.random.seed\|^torch.manual_seed" src/ scripts_cosim/` (module scope only).
   Identical wandb curves across seeds are the tell.
2. **Keep-alive reclaim trap bites squeezed 4-task configs** (previously believed
   8-task-only; H0's first generation had 29/204 deterministically stuck with
   `Invalid forced placement … not in replicas`).
   **Audit:** any frozen collection generated WITHOUT the long keep-alive whose datasets
   show failed placements / partial sweeps: check every REGISTRY collection for
   `sweep_complete: false`, nonzero failed-placement counts in metadata, and
   `placement_errors.log` presence:
   `find simulation_data -name placement_errors.log | head` ;
   `grep -l '"sweep_complete": false' simulation_data/*/ds_*/best.json 2>/dev/null | head`.
   The stage-1 pilot spot-checked clean; the contention/warmth-era collections have not
   been checked against this signature.
3. **Sandbox non-deterministic worker truncation during sweep enumeration**
   (worker_failed counts vary run-to-run; pre-existing, machine-level).
   **Audit:** for past locally-generated corpora, verify `placements.jsonl` row counts
   match the expected sweep size recorded in metadata (the validator's structural pass
   plus a row-count-vs-`total_combinations` check). Anything short = silent truncation.
4. **`compute_caps` alpha_mean dedup bug** (fixed `4614370`) — new `cap_mode` path only;
   byte-identity of the default `alpha_max` path against frozen artifacts was verified.
   **Audit:** none needed for the past (alpha_mean never existed before), but confirm the
   fix's test pins the alpha_max path too.
5. **Three live-path disjoint-platform assumptions** (fixed in `dcb8e12`) — only
   reachable under `replica_overlap: true` (new); byte-identity with overlap off
   verified. No past impact expected; keep as a note.
6. **Preset seed inheritance** (`**BASE_GRID` spreads silently inherit `"seeds"`;
   caught pre-reading, fixed `90fb6e0`).
   **Audit:** list every preset built by spreading another and check which inherit seeds,
   then whether any past LINEAGES claim describes those corpora as seed-independent when
   they share a block. (The 8-task grid shares 901–917 with the pilot BY DESIGN —
   "byte-identical infra per index" — that one is correct as recorded.)

## 4. Machinery landed this session (all committed, tests green at time of interrupt)

- Extended pointwise competitor: `hetdem` (8 cols) + `futureint` (4 cols) opt-in t1
  blocks; arms `t1hd`, `t1x` (+ per-dataset tie band for S2, `eb6e93a`); demand-weighted
  krank; `--extended-blocks` on the transfer; `--cap-mode` across scorer/transfer/verifier
  (`cfa2d0a`, `bb95a47`); B1 ablation arms (`f518cfa`); independent 1e-9 verification for
  all of it; fixtures with verified teeth. 303+ tests passing pre-interrupt.
- Context numbers (stated in the registration): on the old pilot, `t1x` closes 27/35
  firing at median 1.0; extended pooled closure 0.892 (up from frozen 0.648). H0 firing
  @2.0 = 0.220 vs pilot 0.172 — but under the dirty counters of §2, so not a reading.
- Generator: `demand_spread`, `cap_mode`, `replica_overlap` grid keys (all default-off,
  byte-identity verified); presets `route_b_pivot_h0..h3`.

## 5. Next actions, in order

1. Resolve §2 anomaly (3) — the control non-separability — since it gates every rung's
   S0. Then (2)/(1).
2. Decide H0's reading per the registered fallback (likely VOID-INFEASIBLE; record it,
   continue).
3. Generate `h1_ctrl`; score H1 (`--cap-mode alpha_mean`, α 1.5/2.0/3.0); transfer +
   3 verifier passes; read S0–S4. Then H2, H3 (H3: skip cap 16,777,216, α 3/4/6,
   primary 4.0, local ~few CPU-h).
4. Commit rung artifacts ("route_b_env_pivot: ladder screen artifacts"); write the
   LINEAGES outcome entry (PIVOT-CANDIDATE at first PASS / FAIL-BY-EXHAUSTION /
   VOIDs listed) + the §3 audits' findings; update this handover.
5. Run the past-LINEAGES audits of §3 (cheap, read-only; a subagent job).

## 6. Restore prompt

```
[CONTEXT RESTORE] feat/network-contention-v1 @ 90fb6e0, NOT pushed. route_b stage 2
closed NO-GO-PREPROBE (LINEAGES 2026-08-26); user chose the env pivot; screen
registration ROUTE_B_ENV_PIVOT_SCREEN.md signed off @ 019bdcb (LINEAGES entry 6b7b915).
Ladder interrupted mid-H0: H0+ctrl+H1 corpora generated clean (fresh seeds 3001-3017 /
3101-3117, corrected KEEP_ALIVE env), H0 scored but NOT transferred/verified; three
uncommitted JSONs in simulation_data (h0_rtt, h0_ctrl_rtt, h0_reading — partial).
Debug first: (1) separable control shows nonzero constrained R_exact (0.155 frac>1%
@2.0) — impossible for truly separable physics; suspect node_disk_v2 warmth coupling
under the squeeze, may force a registered amendment of the S0 control definition;
(2) greedy_stuck ~50% at all constrained alphas => H0 likely VOID-INFEASIBLE per the
registered fallback; (3) alpha=1.5 wholly infeasible. Then run the ladder H1->H3 per
HANDOVER_route_b_env_pivot.md §5 and the past-LINEAGES audits in §3. plan with fable,
execute with sonnet.
```
