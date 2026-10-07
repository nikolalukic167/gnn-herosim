# gnn_proposal_selector_v1 — GNN proposal value in fixed-replica 32-task search

> Produced on local-record-2026-10-06 at afea04bd; code not merged; old physics.

**Status:** `CLOSED` (2026-09-23) — **PASS** on the corrected, pre-registered eight-topology live gate. The first registration remains invalid because five topologies failed fixed-replica coverage before simulation.

**Outcome.** Adding the old bipartite GNN checkpoint's plan to a deterministic MP-OFF-plus-hand selector cuts live mean task elapsed time by **3.926% median** versus the same selector without that plan, with a positive paired gain on **8/8** fresh topologies. The three-proposal selector also beats the best single arm on each topology by **11.048% median**. All three signed bars pass. This establishes the checkpoint's **complementary proposal value in this fixed-replica hybrid**, not a message-passing-specific gain or stand-alone GNN superiority: the earlier [fixed-seat ablation](gnn_seeded_cd_mp_ablation_v1.md) found its MP-OFF twin better as a single seed. Post-hoc same-checkpoint lesions find only **0.114%** median gain for full message passing over removal of the bipartite platform-edge stage (5/8 positive); removing all message passing still beats the no-GNN selector by **2.970%** median (8/8 positive). Both checkpoints came from ten-task training and were served on 32-task groups; no new model was trained here.

**Parents:** [gnn_seeded_cd_mp_ablation_v1](gnn_seeded_cd_mp_ablation_v1.md), [gnn_seeded_cd_fixed_v1](gnn_seeded_cd_fixed_v1.md).

**Record** (newest first):

- [Post-hoc message-passing lesion diagnostic](#2026-09-23--post-hoc-message-passing-lesion-diagnostic)
- [PASS — corrected sealed live gate](#2026-09-23--pass-on-corrected-sealed-live-gate)
- [Corrected sealed live-gate registration](#2026-09-23--corrected-sealed-live-gate-registration)
- [Initial registration invalid at coverage preflight](#2026-09-23--registered-gate-invalid-at-coverage-preflight)
- [Initial sealed live-gate registration](#2026-09-23--sealed-live-gate-registration)

## 2026-09-23 — sealed live-gate registration

The eight previously unopened topology seeds are **9401–9408**, with no outcome-based replacement. Each receives the same 160-group, 5,120-task connected degree-4 weighted-bridge half-payload workload, 32 tasks per group, peer exchange enabled, and the fixed-replica manifest rule from the parent. All arms use six coordinate-descent passes at exchange scale 4 on CPU. The old matched `joint_burst_v2` GNN and MP-OFF checkpoints are served out of their ten-task training distribution. No new training or GPU allocation is part of this test.

Five arms run on each topology: GNN-only, MP-OFF-only, hand-only, a **three-proposal selector** choosing among GNN, MP-OFF, and hand, and a **two-proposal selector** choosing among MP-OFF and hand. The selectors use the same fixed, deterministic physical base-plus-queue score with coefficient **1** and no adaptation or fitting on gate results. Proposal ties use the frozen runner's deterministic arm-name tie-break. The 9303 and 9307 live pilots, where adding the GNN proposal improved selector elapsed time by **4.30%** and **2.48%**, respectively, are development evidence only; neither seed is in this gate.

Frozen input and runner SHA-256 hashes:

| Artifact | Path | SHA-256 |
|---|---|---|
| Weighted-bridge workload | `/tmp/fixed-g32-weighted_bridge-5120.json` | `d3439e73d098ab83d4ffa3bad0022d3d36b5ed8e8d7ba436d4bffdaa8fc04663` |
| GNN checkpoint | `/tmp/jb2-gnnedge0-seed1.pt` | `d811cbc20d03a81529adee187691f8f2fa6f1dbdd185688e6ddc4a9ee2d41d2f` |
| MP-OFF checkpoint | `/tmp/jb2-mpoff-seed1.pt` | `826108ba634f2fb993b5bf23064c475fe575023ef2ad3c1232ade2ac43783658` |
| Selector runner | `/tmp/fixed_g32_selector_runner.py` | `5fd39a52de7bcce0a7ad9f1a4cef4f4fa0ca45faa92ac3c3561c3dec96dc9930` |
| Selector shell | `/tmp/run_g32_selector.sh` | `5579bdab44824e6a068e4cb3fb2addf0c5ca5d2ec16bb24c5238f17302c23398` |

The primary metric is **mean elapsed seconds per task**, compared within each topology. Define paired percentage gain as `(control − three-proposal selector) / control × 100`. A **PASS** requires all three bars: at least **2% median** gain over the two-proposal selector, a positive gain on at least **7/8** topologies against that selector, and at least **5% median** gain over the best of the three single arms *within each topology*. A tie is not a positive gain. Failure of any bar is **NO-GO** for promoting this selector recipe. No threshold is recalibrated after seeing a sealed result.

The gate audit must verify all **40** runs complete 5,120 tasks and 160 full groups, compare the five fixed-replica manifests within each topology, confirm peer exchange is enabled and produces a measured nonzero peer metric, and record the effective code, workload, checkpoint, and runner hashes plus inference time. Any missing or mismatched audit field makes the affected comparison invalid until repaired and rerun under the frozen protocol; it is not counted as a pass.

## 2026-09-23 — registered gate invalid at coverage preflight

The frozen 9401–9408 gate was attempted. Five seeds (9403, 9404, 9406, 9407, 9408) raised `fixed-replica pool does not cover every source` before producing live results. Only 9401, 9402, and 9405 completed all five arms. The required 40-run, eight-topology audit therefore cannot be performed, and the registered gate is **invalid**, neither PASS nor NO-GO. These failed seeds are not replaced within this registration.

The three complete topology pairs are exploratory evidence, not a reduced three-topology gate:

| Topology | Three-proposal gain over two-proposal selector | Three-proposal gain over best single arm |
|---|---:|---:|
| 9401 | 1.025% | 8.006% |
| 9402 | 1.964% | 5.729% |
| 9405 | 3.650% | 8.447% |

The next gate must first preflight fixed-replica source coverage using topology alone, then register a fresh seed set and run its full live protocol. This is a feasibility repair, not a recalibration of the selector or its pass bars from the three observed outcomes.

## 2026-09-23 — corrected sealed live-gate registration

A coverage-only preflight tested topology seeds **9501–9540** in ascending order using the original fixed-replica rule. It stopped after building each replica manifest and checking coverage, before any task execution or outcome read. The first eight seeds that passed are **9502, 9503, 9504, 9507, 9509, 9511, 9517, 9518**. This rule selected feasible topologies without choosing on performance. The new gate uses these eight seeds, phase `sealed_gate_v2`, and the same five arms, workload, immutable runners and checkpoints, primary metric, three PASS bars, and 40-run audit requirements registered above. The failed 9401–9408 registration remains invalid and is not pooled with this gate.

Frozen config SHA-256 hashes for the corrected gate:

| Topology seed | Config SHA-256 |
|---:|---|
| 9502 | `78ce6b432baad892c04f44daf73d409c81ae5344a1ee8e5575c2d00f6dbdafaf` |
| 9503 | `0e2350d6a8087deddd290a8d0a12208f1f8ea9082f26810580224e6d22c42277` |
| 9504 | `27df4453e191ba2271b4b32374b1aa8558c8c025c8827fd19db14f70ee4831c8` |
| 9507 | `b468f86760b74443593d5a6ae63ce4b51b35b6e5e5ea6ac81e348605ef8dbfd1` |
| 9509 | `50f07859110e931176d5ae742b28356aacf31a1c246084769e73eabc5dab8b96` |
| 9511 | `21466bddc481e609100de6f662f1acb29e9a81043928cc3c9797b9ce0a19e359` |
| 9517 | `400c1133a9b20c36881341d0745134cea9478a58d0b2883333677875dbe83620` |
| 9518 | `f54e832205876b86f3df4d387ad891c8181086dae6b2b7a614b34a0ca9d4ac23` |

### 2026-09-23 — PASS on corrected sealed live gate

All **40** live runs completed the prescribed **5,120 tasks** per arm, for **204,800** task completions total. Both selectors traced **160 full 32-task groups** on each topology. The five arms shared an identical fixed-replica manifest within every topology; peer exchange was enabled and produced nonzero measured exchange time. Workload, checkpoint, selector runner, shell, config and effective-code hashes matched the frozen inputs. The full per-arm results, trace hashes and audit fields are in [sealed_gate_v2_read.json](gnn_proposal_selector_v1/sealed_gate_v2_read.json). The exact [replay bundle](gnn_proposal_selector_v1/sealed_gate_v2_replay.tar.gz) contains the runner scripts, configs, workload, both `.pt` checkpoints and their `.contract.json` sidecars (SHA-256 `2217815d816a3092183048ac83b66e0358c3ac03629170c51120e0f866e61480`). The eight topology seeds, selected solely by coverage preflight, are the independent paired units.

| Topology | Gain over no-GNN selector | Gain over best single arm |
|---:|---:|---:|
| 9502 | 4.410% | 33.067% |
| 9503 | 4.289% | 11.276% |
| 9504 | 3.563% | 8.858% |
| 9507 | 2.726% | 10.819% |
| 9509 | 1.049% | 7.512% |
| 9511 | 5.653% | 15.385% |
| 9517 | 0.651% | 2.426% |
| 9518 | 7.176% | 14.035% |
| **Median** | **3.926%** | **11.048%** |

The registered bars were ≥2% median over the no-GNN selector, positive on ≥7/8 topology pairs, and ≥5% median over the best single arm. Observed values were **3.926%**, **8/8**, and **11.048%**, respectively: **PASS**. Across 1,280 groups, the three-proposal selector chose GNN **506** times, hand **465**, and MP-OFF **309**. Median measured inference time was **2.178 ms/task** for three proposals versus **1.153 ms/task** for two; the latency gain remains after this extra serving work.

The result answers whether this frozen GNN checkpoint contributes a useful *candidate plan* under a fixed deterministic selector and fixed replicas. It does not isolate the contribution of message passing from the checkpoint's other learned features, establish superiority of the GNN-only arm, or transfer to dynamic autoscaling and other workloads. A message-passing claim requires a matched architecture ablation inside this same selector seat.

### 2026-09-23 — post-hoc message-passing lesion diagnostic

After the sealed portfolio result was opened, the same GNN checkpoint was loaded with its normal sidecar and served with two inference-time lesions. `peeronly` disables the bipartite platform-edge stage; `alloff` disables all message passing. These are **post-hoc exploratory** comparisons on the eight already-open gate topologies, with no matched retraining or new sealed test. The per-topology measurements and runner hashes are in [posthoc_mp_lesions.json](gnn_proposal_selector_v1/posthoc_mp_lesions.json); the exact lesion runners are preserved in [posthoc_mp_lesion_runners.tar.gz](gnn_proposal_selector_v1/posthoc_mp_lesion_runners.tar.gz).

| Paired comparison | Median elapsed-time gain | Positive topologies |
|---|---:|---:|
| Full GNN proposal vs `peeronly` proposal | 0.114% | 5/8 |
| Full GNN proposal vs `alloff` proposal | 0.661% | 5/8 |
| `alloff` proposal vs no-GNN selector | 2.970% | 8/8 |

The full checkpoint's edge over its lesioned versions is small and inconsistent across topologies; even with all message passing disabled, its proposal adds value to the selector. Thus the registered **portfolio PASS remains intact**, but these diagnostics do not support attributing that gain to message passing. Proposal diversity or the retained encoder weights may explain much of it. Because lesioning a trained checkpoint changes its served computation without matched retraining, this is a diagnostic rather than a causal architecture comparison.
