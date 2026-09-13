# Case Study: Sequential Cost Alignment in Cold Bursts (Regime B)

> **Not the global evaluation.** Primary paper story remains RQ3: offline GNN looks good;
> sealed live holdouts favor MLP (13/20) / Knative (6/20) / GNN (1/20).
> This section isolates FilterStore serialization on a constructed intelligence stub.
> It is **not** the global deployment model (that remains 873/v5.5, argmax, dim14).
> Multi-cell (2026-08-13): 31.66s holds without arrival jitter; jitter 0.5s piles (125s).

---

## Two-regime contract

| | Regime A / live gates (RQ1–RQ3, RQ8) | Regime B case study (RQ9) |
|---|---|---|
| Workload | `workload-100-100` (~201k) · `workload-125-225` (~562k) | N=12 cold `dnn1` burst |
| Physics | `node_disk_v2` (typical) | `platform_reuse_v1` only |
| Cell | standard5 / all7 / sealed ER holdout | `oracle_split_v1` |
| Primary metric | `total_rtt` | `regime_b_primary_score_s` (max burst elapsed) |
| Deploy model | 873/v5.5 GNN or MLP-bc | distill ckpt is an **instrument**, not a ship candidate |
| Decode | argmax | `seq_reforward_pull` required |

Do not mix metrics or claim the distill checkpoint is the Regime A scheduler.

---

## Setup (constructed cell)

- Stub: `simulation_data/regime_b_cold_burst_v1/live_stub_oracle_split_v1`
- Physics: `platform_reuse_v1` (each cold replica serializes through node FilterStore)
- N=12 simultaneous cold tasks; scarce attractor on node0; union seeds open an oracle-split action space
- Teacher: live DES `knative_network_ect_pull` (marginal FilterStore wait)
- Student: same GIN family, dim24 features, soft KL + hard CE on 6000 cold sequential frames
- Decode: `GNN_DECODE_MODE=seq_reforward_pull` (cross-batch `pulls_committed` ledger)

---

## Systems alignment ladder (not “architecture is a trap”)

Co-sim CE with **N=4 static** `placements.jsonl` labels stays in a **4-to-12-deep pile (94–125s)** because the label is a joint 4-tuple RTT, not sequential FilterStore depth `(committed+1)×T_pull`.

| Rung | What was aligned | Live primary | Verdict |
|---|---|---:|---|
| Features alone (dim24 pull-obs / CACHE 5.6) | observation | 125.28s | Fail |
| Ledger alone (`seq_reforward_pull` on CE weights) | decode | 125.28s (hurt vs some CE argmax ~94s) | Fail |
| Distillation without ledger (argmax) | teacher pick, no serve ledger | 125.28s | Fail |
| Hard-CE distill (α=0) | hard label only | 125.28s | Fail |
| Warm/busy v1 harvest distill | poisoned teacher (busy⇒warm⇒zero pull) | 375.87s | Catastrophic |
| **Aligned** (dim24 + cold soft-distill + ledger) | observation + teacher + decode | **31.66s** | **Succeeds** (oracle 31.65s) |

Single-stub distill (12 frames) was PARTIAL at 62.63s and is superseded by the 6000-frame cold corpus.

**Evidence:** `simulation_data/normal_sim_sweeps/regime_b_phase3_ect_pull_distill_eval_multiseed_cold/summary.json`  
**Ckpt:** `models/near-rtt-v2-regime-b-oracle-split-v1-ect-pull-distill-multiseed.pt` (`f98f43f5…`)

---

## Defensible claim (paste into the case-study close)

> On a FilterStore-serialized cold burst (N=12), a GIN trained with static N=4 co-simulation CE labels remains trapped in a pull pile (94–125s). A live greedy teacher scoring marginal FilterStore wait matches the parallel oracle (31.65s). Distilling that teacher with mixed hard CE and soft ECT-Boltzmann on cold-only sequential frames—while providing the student the identical `pulls_committed` ledger at serve time—recovers 31.66s. Removing the ledger, the soft target term, or poisoning the trajectories via busy-as-warm initialization reverts the model to the pile. This demonstrates that sequential storage costs cannot be resolved by static batch labels, but require strict alignment across train-time features, teacher cost topology, and decision-time state.

---

## Limitations / future work (pre-empt reviewers)

We did **not** reinvent serverless ML or falsify set-equivariant architectures.

1. **Set Transformers / HGT not tested.** Open whether inductive bias could learn sequential depth from sequential labels *without* a manual ledger. Phase 2 only rejected `soft_combo_conc` on N=4 static JSONL — that is not a Set Transformer experiment.
2. **Feature incompatibility.** Distill encoder is dim24 (plat≥16). Global anchor is dim14. Wrong layout is a contract mismatch, not a policy upgrade.
3. **Decode overhead.** `seq_reforward_pull` re-forwards the GNN once per task. Required for 31.66s (argmax of the same ckpt is still 125s). On 125–225-scale traces, even plain `seq_reforward` was ~88–97 min decode vs ~17–30s argmax.
4. **Initialization poisoning.** Harvest is cold-only. Busy-as-warm under `platform_reuse_v1` collapsed to 375s. True busy-without-warm coverage is unsolved.
5. **Jitter is not free.** Multi-cell eval (2026-08-13): zero-jitter / latency-shift / second burst stay at 31.66s; **arrival jitter 0.5s → 125s pile; jitter 2.0s → 62.63s**. The frozen-cell 31.66s is not a jitter-robust result.
6. **Not shipping.** Deploy path remains 873/v5.5 + argmax unless a transfer `/compare` says otherwise.

### Multi-cell grid (same physics, not one JSON)

Artifact: `simulation_data/normal_sim_sweeps/regime_b_multicell_20260813/summary.json`

| Cell | N | Oracle | ect_pull | Distill + ledger |
|---|---:|---:|---:|---:|
| Frozen `oracle_split_v1` | 12 | 31.65 | 31.65 | **31.66** |
| seed7, jitter 0 | 12 | 31.65 | 31.65 | **31.66** |
| seed11, jitter 0.5s | 12 | 32.08 | 31.65 | **125.24** (pile) |
| seed42, jitter 2.0s | 12 | 33.07 | 31.65 | **62.63** (partial) |
| seed7, high latency overlay | 12 | 31.66 | 31.66 | **31.66** |
| Dual burst t=0 + t=40s | 24 | 31.65 | 31.65 | **31.66** |

Second burst at t=40s is cheap after the first pull warms sandboxes — it does **not** re-prove FilterStore split. Jitter cells are the actual stress.

---

## What this is not

- Not a replacement for RQ3 sealed holdout.
- Not a claim that GNNs beat MLP/Knative on hub / warmth / 125–225.
- Not proof that “static SL is fundamentally inadequate for serverless.”
- Not a reason to reopen hub9 decode, Set Transformer, or N=4 CE retrains.
