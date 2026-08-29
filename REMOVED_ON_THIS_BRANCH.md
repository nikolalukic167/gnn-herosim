# What this branch removed, and where it still lives

`artifact/edge-cloud-gnn` is scoped to **one question**: does a graph-aware scheduler place
serverless tasks better than a pointwise one, and does network contention explain when?

Everything below was removed to keep that thread followable. **Nothing is lost** — all of it
is intact on **`feat/network-contention-v1`**, branched at `6afb77e`, which remains the full
research tree. Recover any of it with:

```bash
git checkout feat/network-contention-v1 -- <path>
```

## Removed

| what | why it went | count / size |
|---|---|---|
| `archive/` | Retired pre-GNN HeROsim stack. The live tree was already verified closed against it (no imports; only provenance references in comments). | 680 MB |
| Parked-lineage nodes under `docs/lineages/` | `route_a_v1`, `route_b_v1`, `route_b_env_pivot_v1`, `topology_transfer_v1`, `shallow_v1`, `shallow_longexec_v1`, `siv1_full_corpus`, `serving_speed_v1`, `mp_parity`, `m3_batch_makespan_v1`, `graph_structure_physics`, `cosim_deepdive_v1`, `cache_live_divergence_audit`, `p5b_candidate_relative`, `throughline` — all closed, parked or superseded, none load-bearing for the current claim. | 15 nodes |
| `simulation_data/route_b*`, `route_c*`, `regime_b*` result JSONs | Outputs of parked screens. | ~6 MB tracked |
| `scripts_cosim/*route_b*`, `*route_c*` + their tests | Tooling for those screens. | ~36 files |
| `.claude/skills/route-b-preflight` | Preflight discipline for a parked screen. | 1 skill |
| Policies: `gnn_hetero`, `herocache`, `herocache_network`, `herocache_network_batch`, `determined`, `evaluator`, `offload_network`, `roundrobin_network`, `knative_network_ect`, `knative_network_ect_pull` | Not arms in this study. Registry went 22 → 5 strategies; `valid_policies` 15 → 6. XGBoost entries dropped too, though the classes stay in `tabular/` since they share that package with the MLP. | 10 packages |
| `test_hetero_static_guard_mode_name_absent` | A guard whose only purpose was to stop the registered decoder being wired into the `gnn_hetero` copy. That copy is gone, so the hazard is gone; the test's own message said it needed updating if the directory disappeared. | 1 test |

## Kept, and why

**Lineage nodes** — `objective_pivot_v1` (the live program and the Phase 1 PASS),
`program_verdict_v1` (why the supervised latency route closed), `gnn_draw_study_v1` and
`p5b_draw_study` (the frozen GNN and MLP comparison arms — the Phase 1 statistic is
meaningless without them), `trainer_determinism_v1` (the determinism gate), plus
`network_contention_v1`, `link_contention_v1` and `route_c_link_transfer_v1` — the last three
because they are the **network story**, including the measurement showing link waiting cannot
be made material at co-simulation batch sizes.

**Policies** — `gnn` (subject), `tabular` (MLP baseline), `knative_network` and
`knative_network_batch` (reactive baselines used in every gate), `random` (≈20-line template
for adding one), `knative` (base classes), `state_capture.py`.

## Verification

Full suite after every removal step: **221 passed, 0 failed**. The count fell from 389 only
because the removed `route_b`/`route_c` test files went with their tooling. The simulator
imports and its registry resolves to exactly the five intended strategies.
