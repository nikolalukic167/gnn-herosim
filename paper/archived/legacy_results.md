# Legacy Results (Not For Publication)

> These results are archived because they involve older model generations, broken feature caches,
> or post-hoc interventions that were rejected for deployment. Do not cite in the paper.

---

## 150k–450k Task Benchmarks (dg-26 / 13-dim)

**Why archived:** `desert-galaxy-26` was trained on ~3,705 datasets with a 13-dim feature vector where `shared_fate_signal` was zeroed (bug present) and `is_warm` edge attributes were incorrect (always 0). Results are not comparable to the 14-dim CE-only model and do not reflect the corrected architecture.

**Workload:** 150 clients × 150 servers, ~450,729 tasks, seed 101

| Policy | Total RTT (M s) | vs Knative |
|---|---|---|
| GNN seqblend (dg-26) | 20.36 | −14.3% |
| GNN seqblend+1 (dg-26) | 21.91 | −7.8% |
| Knative | 23.76 | baseline |
| Fair HRC (kn-autoscale) | 24.94 | +4.9% |
| GNN argmax (dg-26) | 27.88 | +17.3% |
| Legacy HRC | 166.87 | +602% (autoscaler bug, fixed separately) |

Note: "seqblend advantage with dg-26 does not directly transfer to CE-only argmax" — explicitly documented in codebase memory. The seqblend override compensates for a broken per-task argmax (dg-26), not a feature of the current model.

---

## Seqblend Decode Analysis (dg-26 / 13-dim)

**Why archived:** Seqblend is a post-hoc inference-time queue override applied to the dg-26 model to partially compensate for logit hot-spotting. It is NOT part of the 14-dim CE-only deployment. The CE-only model does not require seqblend because it has no hot-spotting problem to compensate for.

Seqblend+1 override stats on 444,582 tasks:
- p1 override rate: 32.3%
- Classic-would-override: 55.7%
- On p1 override: gnn_q mean 358 → final 352 (saved ~6)

This data shows the dg-26 model was picking platforms with queue ~358 instead of minimum-queue platforms, requiring a post-hoc override. CE-only's qvm p95=408 is a baseline diagnostic — not a sign of the same failure mode at lower severity.

---

## LQB (Logit Queue Blend) Full Data

**Why archived:** LQB was a post-hoc inference-time fix for the ranking model's hot-spotting. It was **rejected for deployment** because it recovered `default` but hurt 7-config sum (+3.7% vs CE-only).

**Probe: dim14-ranking model, `default` config**

| λ | default RTT | qvm p95 | Notes |
|---|---|---|---|
| 0 (baseline argmax) | 11.62M | 2857 | catastrophic |
| T=3 (temperature, not LQB) | 10.74M | — | insufficient |
| λ=0.8 | 4.64M | 227 | hot-spotting returns |
| λ=1.0 | 4.30M | — | best λ for default |
| λ=1.5 | 4.26M | 122 | best default recovery |

**LQB λ=1.5 full 7-config sum:** 18.55M vs CE 17.89M (+3.7%) — 2/7 wins only. Rejected.

**Mechanism (for internal reference):**
`Score = Logit − λ × log1p(queue)`
At training range (queue 0–6): penalty ≈ λ × 1.9
At live range (queue ~2857): penalty ≈ λ × 7.96
With λ=1.5: max live penalty ≈ 11.9 vs ranking margin cap 8.0.

This bridges the training/live queue gap but only works for `default`. The ranking model's logit sharpening is too severe for a scalar λ to repair across all topology variants.

---

## Historical Multi-Model Comparisons (Confounded)

**Why archived:** These comparisons occurred during cache repair phases where multiple bugs were fixed simultaneously. The architectural changes and cache fixes cannot be separated, making clean attribution impossible.

### silvery-sun-4 vs dg-26 (same 7 configs)
- dg-26 wins 7/7; avg +163% RTT for silvery-sun-4
- silvery-sun-4 was near-RTT v1, not v2; pre-dates SSC repair; pre-dates is_warm fix

### clean-1230 argmax collapse
- Offline strong (val top5 0.054s) but live `default` qvm p95 = 4217 (catastrophic)
- Identified the hot-spotting problem, led to CE-only pivot
- Not a publishable result by itself — it's the failure case that motivated the finding

### Co-sim batch oracle audit (177 snapshots, good-plasma-43)
- HRC: 2.01s avg regret; Knative: 5.67s; GNN (good-plasma-43): 5.69s
- good-plasma-43 was a seq-filtered training model on a different data pipeline
- Not comparable to dim14-CE architecture

---

## Track B Gate Results (Rejected)

**Why archived:** Post-fix 3×3 gate complete. r030 **rejected** — not competitive with CE-only.

| Checkpoint | 3cfg sum | vs CE anchor | Status |
|---|---|---|---|
| r030 | 8.18M | **+1.9%** | **Rejected** |
| CE-only (anchor) | 8.03M | — | Deploy anchor |

Sweep: `dim14_3model_3cfg_queuefix_20260609/`. r002 recipe was queued but superseded by warmth v2 retrain priority.

---

## Dim14 Ranking vs Dim13 Historical Lineage

Not publishable — too many confounding factors between generations:
- dim13 used broken `initialized_snapshot` (shared_fate_signal always 0)
- dim13 used broken `is_warm` edges (always 0)
- dim14 rebuilt cache from scratch with all fixes applied
- Any dim13 vs dim14 comparison is a hardware+software composite, not a clean ablation
