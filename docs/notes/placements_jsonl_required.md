# `placements/placements.jsonl` — REQUIRED (never optional)

**Last Updated:** 2026-06-27  
**Applies to:** all co-sim regen, transfer, repair, recache, and training pipelines

> **Policy:** `placements/placements.jsonl` is **mandatory disk**, not an optional sweep artifact. **`--repair` + recache is NOT a substitute** for keeping it.

---

## What the file is

Each line: `{placement_plan, rtt}` for **every** brute-force placement combo scored in co-sim.

| Consumer | Why JSONL is required |
|----------|----------------------|
| `prepare_graphs_cache*.py` | Builds **RTT hash chunks** (`rtt_chunk_*.pkl`) — maps `(parent_dataset_id, placement_combo) → RTT` for near-RTT / regret training |
| `train_near_rtt*.py` | Counterfactual RTT lookup for non-optimal edges/combos |
| `audit_optimal_labels.py` | Validates `optimal_result.json` against the sweep |
| Seq cache neg-augment graphs | Second-best combo from JSONL when ≥2 combos exist |

**Without JSONL:** graphs + optimal CE labels can still be built from `optimal_result.json` (per-task queue snapshots on the **optimal trajectory only**). **Counterfactual RTT supervision is gone** — training degrades to optimal-only placement classification.

---

## What does NOT replace JSONL

| Step | Provides | Does NOT provide |
|------|----------|------------------|
| `best.json` | Optimal RTT + pointer to optimal export | Full placement universe |
| `optimal_result.json` | One placement replay + per-task scheduling snapshots | Alternative placement RTTs |
| `refresh_optimal_full_stats.py --repair --force` | SSC + disk snapshot + per-task queues on **optimal** replay | `placements.jsonl` or RTT hash rows |
| `refresh_optimal_full_stats.py --rewrite-ssc` | SSC from stored stats | JSONL or RTT hashes |

---

## Fast-path mistake (warmth_v2, 2026-06)

Regen used `GNN_CAPTURE_DATASET_STATE=0` + post-hoc `--repair` and treated the brute-force sweep as **optional disk cost**.

| Failure mode | Cause |
|--------------|--------|
| `--resume` skips on `best.json` only | Never checks `placements/placements.jsonl` → stale dirs skip full BF |
| `.bf_scratch` deleted on success | JSONL only in scratch until copy; if copy skipped/crashed, data lost |
| Repair on 824 ds | 832 graphs in cache but only **~486** parents in RTT hash (warmth ~135 + sparse 351) |

**Sparse regen (351/351):** full BF + JSONL kept — correct pattern.  
**Warmth:** ~346 dirs with `optimal_result` but **no** public JSONL — resume-skipped or scratch never copied.

---

## Required practices

1. **Always persist** `placements/placements.jsonl` from `.bf_scratch/placements.jsonl` before deleting scratch.
2. **`--resume` must require** both `best.json` **and** non-empty `placements/placements.jsonl` (or re-run BF).
3. **Transfers/rsync** must include `placements/` (not just `optimal_result.json`).
4. **Never delete** `.bf_scratch` until `placements/placements.jsonl` exists and is non-empty.
5. **Recache sanity:** `rtt_parent_dataset_ids.txt` count should match datasets with JSONL; investigate large gaps before training.

---

## Recovery

- **Scratch still present:** copy `ds_*/.bf_scratch/placements.jsonl` → `ds_*/placements/placements.jsonl` (datalab had 8 warmth gaps in this state).
- **Scratch gone:** re-run brute-force for that `ds_*` (do not rely on repair alone).

---

## Code / script anchors

- Writer: `generate_gnn_datasets_fast.py` (BF → `.bf_scratch` → `placements/`)
- Reader: `prepare_graphs_cache_seq.py`, `prepare_graphs_cache.py` (`build_rtt_hash_table_chunks`)
- Policy comments: `refresh_optimal_full_stats.py`, `run_warmth_parallel_regen.sh`, datalab `run_warmth_regen_shard.sh`
- Non-unique backfill (cartesian re-enum on existing ds): `generate_non_unique_placements_fast.py --datasets-dir` (datalab `warmth_non_unique_*.sbatch`)

---

## Non-unique backfill (2026-06-15)

When an older corpus was generated **without** `--allow-non-unique-replicas`, JSONL exists but omits colliding placements. **Do not** treat repair/recache as a substitute — run cartesian backfill to rewrite/extend `placements/placements.jsonl`:

| Corpus | Status |
|--------|--------|
| warmth_v2 | **498 jsonl** (492S+9SK) |
| sparse_warmth_v2 | **351 jsonl** |
| contention_v2 | **900/900 jsonl** (had `--allow-non-unique-replicas` at regen; finisher closed last 189) |

After backfill: rsync `placements/` to mitrix before recache/training.
