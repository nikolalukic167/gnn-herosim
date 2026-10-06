# g32_exact_val_selection_v1 — exact validation for matched 32-task checkpoints

**Status:** `CLOSED` (2026-09-23) — **NO-GO for bipartite attribution**. Registered 2026-09-23; validation selection and all live bars were frozen before outcomes.

**Outcome.** Exact simulator validation selected final checkpoints for all three trained architectures and repaired the parent gate's censored checkpoint choice. On eight new paired live topologies, the full bipartite GNN still loses at the median to peer-only and MP-OFF, both directly and in equal-size portfolios. It beats matched hand directly, but so do both learned controls; this is a learned-proposal win, not evidence that bipartite message passing helps. The held-out training test topology 9619 was not read. This closes the exact-validation-selected 16-plan sampled-slate recipe, not bipartite GNNs generally.

**Record** (newest first):

- [2026-09-23 — exact-validation-selected sealed live gate: NO-GO](#2026-09-23--exact-validation-selected-sealed-live-gate-no-go)

**Question.** Does the trained edge-aware bipartite GNN beat separately trained peer-only and MP-OFF controls when checkpoint selection uses exact simulator RTT for each decoded validation plan? The [parent sampled-slate gate](g32_sampled_slate_v1.md) served epoch-zero weights because its capped 16-plan validation sidecar penalized most final-epoch decodes as unmapped. Its negative live verdict stands for those served weights.

**Validation-only selection.** The exact candidate evaluator at `/tmp/g32_exact_val_candidates.py` (SHA-256 `da244533e3eab08a336d7ed033cc30a9742f3c6b106691c332123bdcaecdffe4`) simulated the epoch-zero and final-epoch decoded plans for each architecture on all **16** groups of the previously assigned validation topology **9612**. The epoch-zero simulation matches cached slate row 0 in every group. The shared rule is to select the checkpoint with the lower *sum of exact simulator RTT* over these groups, before reading the held-out test topology **9619** or any fresh live outcome. The per-group validation read at `/tmp/g32-exact-val-candidates.json` and its hash are captured in the [gate freeze](g32_exact_val_selection_v1/gate_freeze_2026-09-23.json).

| Architecture | Epoch-zero RTT sum | Final-epoch RTT sum | Selected |
|---|---:|---:|---|
| Full bipartite GNN | 2,204.862315 s | 970.803304 s | Final |
| Peer-only GNN | 2,204.862315 s | 1,400.261207 s | Final |
| MP-OFF | 2,204.862315 s | 977.403900 s | Final |

Exact validation selects **final weights for all three arms**. The full model's final sum is close to MP-OFF's; its median paired group gain over MP-OFF is **0%**, positive on only **1/16** groups. This is a checkpoint-selection repair and a reason to run the live gate, not an offline architecture win. Validation groups are not independent topology replicates.

**Fresh live gate, registered before outcomes.** Coverage-only fixed-replica preflight selected the first eight eligible new topology seeds: **9713, 9714, 9715, 9719, 9720, 9721, 9723, 9725**. No task outcome informed that choice. Run seven arms on each topology: direct full, peer-only, MP-OFF and matched hand; and three equal-size deterministic selectors that each add one of the newly trained proposals to the same frozen old MP-OFF and hand proposals. Serve final checkpoints with their sidecars and architecture flags from the freeze. All arms use the same connected fixed-replica, peer-enabled workload of **160 groups and 5,120 tasks**, six coordinate-descent passes, exchange scale **4**, and CPU serving. Topology is the paired independent unit.

The primary metric is mean elapsed seconds per task within topology. Gain is `(control − full) / control × 100`; ties are not positive. A bipartite-specific **PASS** requires the full direct arm to exceed *each* of peer-only and MP-OFF direct arms by **at least 2% median** and be positive on **at least 7/8** topologies against each, **and** the full portfolio to meet the same two bars against *each* corresponding peer-only and MP-OFF portfolio. Report direct hand separately. A failed bar is **NO-GO** for this exact-validation-selected recipe. Do not change the bar after seeing outcomes.

Audit all **56** runs for complete task and group counts, matching fixed-replica manifests within topology, nonzero peer-exchange time, checkpoint and sidecar contracts, code and input hashes, full result parse, and inference time. An invalid audit is repaired and rerun under the frozen protocol. The seven-arm runner passed an opened development smoke on topology **9303**; that topology is excluded from the fresh gate. The [freeze manifest](g32_exact_val_selection_v1/gate_freeze_2026-09-23.json) records sizes and SHA-256 hashes for the eight configs, workload, three final checkpoint/sidecar pairs, frozen old MP-OFF pair, validation evaluator/output, training configs and split, generator, direct/portfolio/hand launchers, orchestrator, both Python runners, and both auditors. Manifest SHA-256: `900947e13425b6ddaeca967b0ff710347b68ccb63129a0811217cca941ae3d40`.

### 2026-09-23 — exact-validation-selected sealed live gate: NO-GO

All seven arms completed on each of the eight fresh topologies: **56/56** runs, each with 160 groups and 5,120 tasks. All per-topology [audits and raw-result hashes](g32_exact_val_selection_v1/raw_archive_manifest_2026-09-23.json) pass the registered checks, including fixed-replica manifest equality and nonzero peer-exchange time. The paired independent unit is topology, not task or group. Gains below use `(control − full) / control × 100` on mean elapsed seconds per task; a positive count excludes ties.

| Full GNN compared with | Median gain | Positive topologies | Registered bar |
|---|---:|---:|---|
| Peer-only, direct | −0.640% | 3/8 | FAIL |
| MP-OFF, direct | −0.362% | 3/8 | FAIL |
| Peer-only, equal-size portfolio | −0.307% | 2/8 | FAIL |
| MP-OFF, equal-size portfolio | −0.096% | 4/8 | FAIL |

Each comparison misses the pre-registered **≥2% median and ≥7/8 positive** bar. The full model's direct gain over matched hand is **+4.872% median, 8/8**; peer-only gains **+8.197%, 8/8**, and MP-OFF gains **+6.085%, 8/8** over the same hand control. The trained policies can improve over hand in this fixed-replica seat, but this experiment finds no bipartite-specific gain after exact validation selects final weights. The small full-versus-MP-OFF median gap is not a positive architecture result; the signed threshold and consistency bar both fail.

The [sealed raw archive](g32_exact_val_selection_v1/sealed_live_gate_2026-09-23.tar.gz) contains all **56 gzipped raw results** and **16 audit JSONs**, plus the member-hash manifest. Every archive member was re-read and SHA-256 verified against that manifest. Archive SHA-256: `861f0f2c455f862241570b8e1c4208d07daf1580ad5fdd24813fc5045a39d885`; manifest SHA-256: `c86a9b55e5c222fa082122bfec0f588669a63842eb677a1c4d7c2399ed462486`. The manifest also preserves every topology's seconds-per-task values and paired gains. The test topology **9619** remained sealed throughout selection and this gate; no test result is claimed. A new decision target or proposal mechanism needs a separate lineage and fresh controls.
