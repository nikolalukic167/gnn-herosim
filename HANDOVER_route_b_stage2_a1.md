# Handover — route_b stage 2, arm A1 (genuine T2 GNN)

**Date:** 2026-08-26
**Branch:** `feat/network-contention-v1`
**Base commit at start of work:** `9c7a789` (B2: dim63crk layout)
**State:** implementation complete and verified; **nothing committed** — all changes are
in the working tree.

---

> ## STATUS UPDATE — 2026-08-26, follow-up session
>
> Everything §5 listed as blocking-before-runs is now done and committed; the sections
> below are kept as written for the record. What changed:
>
> - **Committed** in grouped commits: `26cb6f1` (B2+ DAG cache path + dim63crk
>   extraction), `28fbe35` (this handover's A1 work), `523aeaf` (docs/domain model),
>   `0ac184c` (B6). Working tree clean.
> - **B6 CLOSED.** Shared split artifact `experiments/route_b_stage2_split_v1.json`
>   (142/31/31 over 204 parents, seed 42, sha256 `0171ef14…`), producer
>   `scripts_cosim/make_split_artifact.py`, consumers in both trainers
>   (`NEAR_RTT_SPLIT_ARTIFACT` / `--split-artifact`), fail-loud coverage + bypass
>   guards, `{path, sha256}` stamped in sidecar/meta — verified identical across an A1
>   and an MLP smoke. 18 tests in `tests/test_split_artifact.py`; full suite 249 pass.
> - **Real DAG cache built** — locally in 7.65 s, not on datalab (step 2's datalab
>   estimate was wrong by ~3 orders of magnitude; timing metadata in every cache says
>   so): `graphs_cache_route_b_pilot_s_dag`, 204 graphs, all alpha rungs feasible,
>   dim14/`legacy_v0`/`partial_state_v1`. Built with explicit
>   `--platform-feature-dim 14` — the CLI default is 16 and the trainer only warns.
> - **`experiments/route_b_stage2_a1.yaml` repointed** to the real cache and pinned to
>   the artifact. Frozen decoder acceptance re-run: still 408 cells.
>
> **Remaining before the registered draws:** write `experiments/route_b_stage2_a{2,3}.yaml`
> (MLP arms first, per registration), then multi-seed runs, then the LINEAGES outcome
> row. LINEAGES.md now carries a stage-2 build-queue progress row dated 2026-08-26.
> §6's honest risk is unchanged.
>
> ## STATUS UPDATE — 2026-08-26, §9 pre-probe
>
> The §9 pre-probe ran on the pilot-204 corpus (registered deviation, 4 draws/arm) →
> **NO-GO-PREPROBE** per §9: A1 train-split median regret 28.45% ≥ A2's 19.34%. Along
> the way, a seed-clobber bug in `prepare_graphs_cache.py` (import-time
> `torch.manual_seed(42)` was overwriting every A1 draw's actual seed) was found and
> fixed, with regression tests added. Full record — deviations, build items, the void
> first A1 sweep, results, σ calibration, and context observations — is in the
> `route_b_v1 — stage 2 §9 pre-probe: NO-GO-PREPROBE (2026-08-26)` row of LINEAGES.md.
> The next decision (environment pivot per CLAUDE.md option 2, vs closing route B's GNN
> argument) is the user's and needs its own registration either way.

---

## 1. What this session did, in one paragraph

Before this session the "GNN" arm of the route_b stage-2 comparison was structurally
**identical to A3**: message passing saw only the bipartite task↔platform graph (no
workload-DAG edges), and `masked_topo` decode read a single static logit vector computed
before any placement was committed. An A1-vs-A2/A3 comparison would therefore not have
measured what §2/§3 registered. This session added the two missing capabilities behind
explicit, default-off flags, so A1 is now genuinely T2: it sees DAG structure, and it
re-scores each task against the prefix already committed.

**Nothing about GNN-vs-MLP performance was measured.** Only a 2-epoch smoke run on 12
graphs. The comparison is now *askable*, not *answered*.

---

## 2. What changed (all uncommitted)

### New files
| File | Purpose |
|---|---|
| `src/policy/gnn/partial_state_edges.py` | The one seam: `candidate_edge_rows`, `refresh_partial_state_edge_attr`, `make_partial_state_score_fn` |
| `tests/test_gnn_t2_prefix_parity.py` | T1/T2 column parity + train/decode prefix parity (6 tests) |
| `experiments/route_b_stage2_a1.yaml` | Arm A1's config |

### Modified files
| File | Change |
|---|---|
| `src/policy/gnn/gnn_model.py` | `mp_dag_edges` / `task_type_onehot_dim` / `partial_state_edge_dim`; `forward` split into `_encode`+`_score` |
| `src/policy/gnn/seq_decode.py` | `run_decode_with_timing` forwards `score_fn`; **bug fix**: `dict(chosen)` copy |
| `src/notebooks/train_near_rtt.py` | A1 flags, startup guards, device whitelist, `loss_tied_teacher_forced_ce`, sidecar, checkpoint-metric routing |
| `src/executesimulation.py` | Stage-2 T2 serving refusal; `checkpoint_mp_config` whitelist extended |
| `scripts_cosim/verify_cache_live_feature_parity.py` | DAG block → hard failure (live parity undefined) |
| `scripts_cosim/test_train_serve_mp_parity.py` | +10 DAG/prefix tests |
| `tests/test_masked_topo_decoder.py` | +10 `score_fn` tests |
| `tests/test_trainer_determinism.py` | +1 A1 teacher-forced determinism case |

> Note: `src/notebooks/prepare_graphs_cache.py`, `src/policy/tabular/reduced_features.py`
> and `src/policy/tabular/train_mlp_dim22_from_batch.py` were **already modified before
> this session** (B2 work in progress). This session did not touch them.

### Design decisions worth not relitigating
- **DAG edges are undirected.** Parent→child only would leave the root task with a
  bit-identical no-DAG embedding, and §4's prefix-oracle curve puts nearly all decoder
  myopia in the first two of four steps — exactly where a directed-only variant is blind.
- **The 4-way task-type one-hot is mandatory with DAG edges** (constructor raises). On
  diamond4 `cnn` and `rf` share a `task_features` encoding, and undirected mixing over a
  3-layer GIN would make them interchangeable. It is also a *fairness repair* — the T1 MLP
  already sees task type via the krank block.
- **`edge_attr` was NOT widened.** Its 5-column width is load-bearing for the
  dim22/dim25cr/dim63crk extractors that build the A2/A3 baselines. The 38 prefix columns
  ride in a separate `partial_state_edge_attr`.
- **Prefix enters at the EdgeScorer only**, so the GIN is prefix-independent and runs once
  per graph. Recorded in the sidecar as `prefix_conditioning_scope: edge_scorer_only`.
  The prefix-into-node-features variant is a §3 sensitivity row, not this arm.

---

## 3. Verification actually run (not claimed — run)

| Check | Result |
|---|---|
| Frozen `masked_topo` acceptance (`--check-decoder`) | **408 cells pass**, before and after |
| Full suite `tests/` + `test_queue_features.py` | **212 passed** |
| Targeted plan suite (8 files) | **100 passed** |
| A1 trainer smoke (2 epochs, 12 graphs) | CE 5.2343 → 5.1477; `count_regret_masked_topo=2` |
| Serving refusal on a real A1 checkpoint | raises `ValueError` |
| T1/T2 column parity | **bit-identical** (keyed on `logit_idx`) |
| Flood ratio, measured on `route_b_smoke_s/ds_00000` | **0.200x** (8 DAG vs 40 bipartite) |

---

## 4. Two bugs found during implementation

1. **`score_fn` received the live `chosen` dict by reference.** A callback that mutated it
   corrupted decoder state mid-plan. Fixed to `dict(chosen)`. Pinned by
   `test_score_fn_cannot_mutate_decoder_state`.
2. **The serving refusal silently did not fire.** `checkpoint_mp_config` builds from an
   explicit key whitelist, so `partial_state_edge_features` was written correctly by the
   trainer and **dropped at load time** — a real T2 checkpoint loaded successfully. Fixed
   and verified against an actual checkpoint. Saved to memory as
   `herosim-sidecar-keys-need-serving-whitelist` — it generalizes to every future sidecar
   field.

---

## 5. What is left to do

### BLOCKER — do this before running any registered draws
**[RESOLVED 2026-08-26 — see STATUS UPDATE above; commit `0ac184c`.]**

**B6: the shared split artifact.** `split_by_parent_three_way` is MLP-only; the GNN still
uses `split_ids_by_canonical_parent(random_state=42)` at `train_near_rtt.py`. Until one
split artifact is loaded by both arms, a "draw" varies the split as well as the
initialisation, and §3's paired test is confounded. The A1 sidecar currently records:

```json
"split_artifact": {"mode": "split_ids_by_canonical_parent", "random_state": 42}
```

Swap in `{"path": ..., "sha256": ...}` when B6 lands.

### Then, in order
2. **[DONE 2026-08-26 — built locally, not on datalab; see STATUS UPDATE.]**
   **Build the real DAG cache.** `prepare_graphs_cache.py --dag-partial-state` over
   `simulation_data/gnn_datasets_dag4_route_b_pilot_v1_arm_s` (204 datasets). Only a
   12-graph smoke cache (`graphs_cache_route_b_smoke_s_dag`) exists today. Likely a
   datalab job — read `PARITY.md` and the `datalab-pitfalls` skill first, and never write
   `pipenv run python3` in an `.sbatch`.
3. **Run the arms, multiple seeds each.** A1 vs A2/A3. Multi-seed is not optional: memory
   `herosim-mlp-reliability-is-a-draw-not-a-feature` records collapse swinging 0→26/30 on
   the seed alone, and `herosim-gnn-never-collapses-is-falsified` records 2 of 8 GNN draws
   collapsing. A single run per arm tells you nothing.
4. **Write the `LINEAGES.md` row.** Per CLAUDE.md the lineage is not done until it has an
   outcome.

### Also worth doing
- **[DONE]** Commit this work (currently uncommitted on `feat/network-contention-v1`).
- **[DONE — repointed to `graphs_cache_route_b_pilot_s_dag` + the B6 artifact.]**
  Consider whether `experiments/route_b_stage2_a1.yaml` should point at the full cache
  rather than the smoke cache before the real runs (`cache_dir` is currently
  `graphs_cache_route_b_smoke_s_dag`, and `epochs: 40`).

---

## 6. The honest risk, stated up front

Route A closed because the problem stayed **pointwise-optimal** — DAG + distance was
genuinely pairwise, so a simple model could match the graph-aware one
(`herosim-route-a-blockers-cleared`). Route B added contention to break separability. But
this project also measured that the **concurrency lever is exhausted**: even 8-task /
2-client caps the contention ceiling below 10% (median 7.0%, max 9.99%) —
`herosim-link-contention-charges-input-ingress`.

So a plausible outcome of step 3 is **"still not enough contention for the GNN to win."**
That would not be a defect in what this session built. It would be a finding about the
simulator, and it points at CLAUDE.md's option 2 — make the environment more dynamic —
rather than at more model work. Worth knowing before spending the GPU hours.

---

## 7. Useful commands

```bash
# Frozen decoder acceptance (must stay at 408 cells)
PIPENV_IGNORE_VIRTUALENVS=1 PYTHONPATH=. pipenv run python3 \
  scripts_cosim/verify_route_b_scorer_agreement.py --check-decoder \
  --corpus simulation_data/gnn_datasets_dag4_route_b_pilot_v1_arm_s \
  --report simulation_data/route_b_pilot_v1_arm_s_rtt.json

# The plan's verification suite
PIPENV_IGNORE_VIRTUALENVS=1 PYTHONPATH=. OMP_NUM_THREADS=1 pipenv run python3 -m pytest \
  scripts_cosim/test_train_serve_mp_parity.py tests/test_masked_topo_decoder.py \
  tests/test_gnn_t2_prefix_parity.py tests/test_partial_state_features.py \
  tests/test_route_b_positive_controls.py tests/test_trainer_determinism.py -q

# A1 config dry-run (shows resolved env + argv)
PIPENV_IGNORE_VIRTUALENVS=1 PYTHONPATH=. pipenv run python run_experiment.py \
  experiments/route_b_stage2_a1.yaml --dry-run
```

The full implementation plan, with rationale and the risk register, is at
`/root/.claude/plans/i-have-all-the-snazzy-token.md`.
