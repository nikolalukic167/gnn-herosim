# l2d_size_transfer_v1 — inference-only transfer from 6×6 job shops

**Status:** `CLOSED` (2026-09-23) — **ZERO-SHOT TRANSFER FAIL**. Registered before the seed-301 target gate; all four signed GNN-versus-control contrasts miss both bars.

**Outcome.** On fresh 10×10 and 15×15 job shops, the converged 6×6 GNN checkpoint has negative median paired gain against both MP-OFF and the typed receptive-field MLP. The registered ≥1% median and ≥7/8 positive-seed bars fail in all four contrasts. FDD/MWKR also beats every learned arm's mean. This closes inference-only transfer of these 6×6 checkpoints, while the parent's positive within-size 6×6 result stands.

**Question.** Does the message-passing advantage measured after converged 6×6 L2D training persist when those same policies schedule larger, unseen job shops? The [parent study](literature_reeval_v1.md) found a GNN advantage over MP-OFF and a typed receptive-field MLP on fresh 6×6 instances at learning rate 5e-4 and 30,000 updates. Its 10×10 and 15×15 training results do not answer inference-only size transfer.

## Frozen gate

- **Source:** The existing eight training seeds (0–7) per `gnn`, `mpoff`, and `mlp_t1` arm from 6×6, `lr=5e-4`, 30,000-update runs. Use each run's `best_val.pth`, selected solely on source-size validation, with its `.contract.json` sidecar. No new training or target-size checkpoint selection. The exact source path pattern and contract checks are in [`transfer_eval.py`](../../scripts_reeval/l2d/transfer_eval.py).
- **Targets:** Two independent, fresh sets of 100 instances: 10×10 and 15×15, each generated with NumPy seed 301 by the upstream L2D instance generator at commit `7b2efbb`. Freeze as `simulation_data/literature_reeval_v1/l2d/test_10x10_seed301.npy` and `test_15x15_seed301.npy`. Every arm and training seed sees the same 100 instances at each size. Generation is specified in [`prepare_transfer_test.py`](../../scripts_reeval/l2d/prepare_transfer_test.py). Record each dataset SHA-256 in the result before evaluation.
- **Evaluation:** CPU greedy decoding through the upstream L2D environment, with the policy weights loaded at the target size. Record all 100 makespans and their mean for each arm, seed, and size. Run FDD/MWKR on the same instances as an external hand-rule reference; report it separately from the architecture gate. Record checkpoint and sidecar SHA-256 values and the upstream commit in the result. A missing or mismatched sidecar invalidates that run.
- **Statistic:** For each target size, pair arms by training seed. For control `c` and seed `s`, gain is `(mean_makespan[c,s] − mean_makespan[gnn,s]) / mean_makespan[c,s]`; positive favors the GNN. The independent unit for the gate is the **training seed** (eight paired values), not the 100 shared test instances. Report per-seed means, each paired gain, medians, and positive-seed counts for both controls at both sizes.
- **PASS:** At **both** target sizes, against **both** `mpoff` and `mlp_t1`, median paired GNN gain is at least **1%** and gain is positive on at least **7/8** seeds. Otherwise this specific size-transfer claim fails. Do not fold FDD/MWKR into this architecture criterion; show its makespan and whether the learned arms beat it.

The earlier opened smokes used seed 300 (two 10×10 and ten 15×15 instances). They check execution only and contribute no observation to this gate. The [gate freeze](l2d_size_transfer_v1/gate_freeze_2026-09-23.json) (SHA-256 `e9a0512c385e7fa5c50814160dbdc11424df3f8cbad471f4c3a13f74b79739de`) records 57 artifacts, including all 24 checkpoint/sidecar pairs, the two seed-301 datasets, code, and protocol. The upstream checkout was clean at the registered commit.

## Record

- [2026-09-23 — sealed target gate: zero-shot transfer fails](#2026-09-23--sealed-target-gate-zero-shot-transfer-fails)
- **2026-09-23 — registered.** The source, targets, paired unit, statistic, and PASS bar above were fixed before the seed-301 target evaluation. The parent study and seed-300 smoke informed feasibility, not the target-size result.
- **2026-09-23 — source artifact recovery, before target evaluation.** The converged 6×6 seed-7 `best_val.pth` and contract sidecar were absent locally for all three arms. Their datalab copies were verified, downloaded, and checked byte-for-byte by SHA-256 against the remote files (six matching files). `/tmp/l2d_transfer_union` links all 24 source run directories: local seeds 0–6 and the recovered seed 7 for each arm. This restores the registered eight-seed source set; no seed-301 target outcomes were opened.
- **2026-09-23 — artifact freeze, before target evaluation.** The linked manifest fixes both seed-301 datasets and all source checkpoints and sidecars by SHA-256. The eight-seed source set is complete. No target outcome was read during registration.

## 2026-09-23 — sealed target gate: zero-shot transfer fails

The [audited gate read](l2d_size_transfer_v1/gate_read_2026-09-23.json) reports **48 model evaluations** (three arms × eight training seeds × two target sizes), each scheduling the same 100 fresh seed-301 instances at its target size in the upstream environment. Audit passed all 57 frozen artifact hashes, checkpoint contracts, per-instance makespans, and the clean upstream commit. Training seed is the paired independent unit; the 100 instances are shared within each target split.

| Target | GNN vs MP-OFF median gain; positive seeds | GNN vs typed MLP median gain; positive seeds | FDD/MWKR mean |
|---|---:|---:|---:|
| 10×10 | −0.577%; 4/8 | −0.327%; 3/8 | 949.07 |
| 15×15 | −2.200%; 3/8 | −0.104%; 4/8 | 1457.59 |

All four contrasts fail the registered ≥1% median and ≥7/8 positive-seed bars. Per-seed arm means and paired gains are in the gate read. FDD/MWKR's mean is below every learned-arm mean at both sizes; it is an external reference, not an architecture control. The [sealed archive](l2d_size_transfer_v1/sealed_gate_2026-09-23.tar.gz) has SHA-256 `ce11d7e9965c4ced9456bd8b358afb1989723c06c3677b57980607a44b87280d`; its [manifest](l2d_size_transfer_v1/archive_manifest_2026-09-23.json) has SHA-256 `b6796461ef8a21b1dd1e1fac919c23015942fb2423aca1702bf6b3e1af197056` and 63 verified members.

**Scope.** This rejects zero-shot 6×6→larger transfer for these converged checkpoints and this fixed feature contract. There was no training-data-budget sweep or target-size adaptation. The raw lower-bound feature is scaled by a fixed `/1000`, so its distribution may shift with job-shop size; that mechanism was not isolated. The result does not overturn the parent's 6×6 within-size message-passing advantage or decide what size-specific training would achieve.
