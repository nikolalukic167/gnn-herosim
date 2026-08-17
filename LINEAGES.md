# LINEAGES.md — what is current

**Read this before starting work.** It is the map of which experiment lineages are live
and which are retired. `simulation_data/REGISTRY.json` does the same for datasets; this
does it for code.

Statuses:

| Status | Meaning |
|---|---|
| `ACTIVE` | Current work. Change this code. |
| `SUPERSEDED` | Replaced by a later lineage. Result still stands; don't build on it. |
| `FALSIFIED` | Hypothesis was disproven. **Do not revive without new evidence.** |
| `PAPER` | Frozen because the paper cites it. Change only with a paper edit. |

Retired code lives in [`archive/`](archive/README.md) — moved with `git mv`, so
`git log --follow` still works. Nothing was deleted. Restore point: tag
`pre-cleanup-2026-08`.

---

## ACTIVE

| Lineage | Entry points | Datasets | Notes |
|---|---|---|---|
| **siv1_full_corpus** | `scripts_cosim/datalab/full_corpus_siv1_{recache,gnn_train,mlp_train}.sbatch` → `run_full_corpus_siv1_*.sh` | whole `legacy_v0_node_disk_v2_4task` group | Trains on the full corpus under `scale_invariant_v1`. GNN: `src/notebooks/train_near_rtt.py`. MLP: `src/policy/tabular/train_mlp_dim22_from_batch.py`. Recache: `src/notebooks/prepare_graphs_cache.py`. **Outcome 2026-08-17 — see `mp_parity` below.** |
| **mp_parity** | `scripts_cosim/test_train_serve_mp_parity.py`, `experiments/full_corpus_siv1_gnn_mp_residual{,_node_edges}.yaml`, `datalab/mp_arm_gnn_train.sbatch` | full corpus siv1 | Train/serve message-passing parity, and what to do about it. Outcomes below. |
| **contention_v4_v5** | `scripts_cosim/datalab/contention_v4_deepq_cosim.sbatch`, `contention_v5_quick_test.sbatch` | `contention_v4_pilot`, `contention_v5_quick_test` | Deep queues + coupling optimisation — the current attempt at giving the GNN real graph structure to exploit. |
| **contention_v2_v3** | `important/run_contention_v{2,3}_train_and_live_gate_nohup.sh`, `important/compare_contention_v2_live_gate.py` | `contention_v2{,_verify}`, `contention_v3` | Baseline contention series the v4/v5 work is measured against. Trainers: `train_near_rtt_v2_contention_v{2,3}_dim14_ce_only.py`, `train_mlp_contention_v{2,3}_dim22_batchcache.py`. |
| **sealed_holdout** | `important/run_contention_v2_873_sealed_holdout{,_rebaseline}.sh`, `compare_sealed_live_holdout.py`, `datalab/sealed_holdout_gpu.sbatch` | `contention_v2` | The honest generalisation gate. |
| **coupled_trio** | `important/run_contention_v2_873_coupled_trio.sh`, `chain_coupled_trio_then_rebaseline.sh` | `contention_v2` | See memory note: ECT is not a ceiling. |
| **encoder_ablation** | `important/run_gnn_encoder_ablation.sh`, `compare_encoder_ablation.py` | contention series | Is the graph encoder doing work, or is it the features? |
| **seed_variance** | `scripts_cosim/run_gnn_seed_variance_siv1.sh` | contention_v2 | Uses `train_near_rtt_v2_contention_v2_dim14_ce_only.py`. |
| **queue_feature_contract** | `src/placement/queue_features.py`, `scripts_cosim/test_queue_features.py`, `verify_cache_live_feature_parity.py` | all | `legacy_v0` vs `scale_invariant_v1`. See CLAUDE.md. |
| **dataset_metadata** | `scripts_cosim/{extract_dataset_metadata,validate_dataset_collection,compute_compatibility_matrix}.py` | all | Produces `REGISTRY.json`, `METADATA.json`, `COMPATIBILITY_MATRIX.json`. |

Shared core (not a lineage — everything depends on it): `src/placement/`, `src/policy/{gnn,tabular,knative*,determined,evaluator}/`, `src/executecosimulation.py`, `src/executesimulation.py`, `scripts_cosim/generate_gnn_datasets_fast.py`, `src/notebooks/non_unique_lib/`.

### mp_parity — outcomes (2026-08-17)

**Root cause.** `train_near_rtt.py` fitted `self.gin(x, data.edge_index)` (bipartite only)
while the serving copy in `src/policy/gnn/gnn_model.py` concatenated every same-node
platform↔platform edge — ~26:1 more edges than bipartite on the full-corpus cache. The
served model ran message passing on a graph its weights had never seen. Fixed by making
same-node edges opt-in, and structurally by deleting the second copy of the model: the
trainer now imports the one definition.

**Baseline gate** (`normal_sim_sweeps/gnn_mp_parity_gate_20260816`, deployed checkpoint
with parity fix, 3 configs × 5 seeds):

| config | GNN/Kn | MLP/Kn | GNN cell wins | p99 winner |
|---|---|---|---|---|
| sparse_p25 | 1.14x | 0.83x | 0/5 | mlp |
| sparse_p25_skew | 0.84x | 2.27x | 3/5 | **gnn** (71.0s vs MLP 498.5s) |
| sparse_p35 | 1.02x | 0.77x | 0/5 | mlp |

Pre-registered PRIMARY (GNN > MLP on total_rtt in ≥2 of 3 configs) = 1/3 **FAILED**.
TAIL (same on p99) = 1/3 **FAILED**. The parity fix removes the 12.4x catastrophe but the
fixed baseline still loses to MLP on the two large-RTT configs. It does reproduce the
pre-registered *collision cliff* on `sparse_p25_skew`, where the MLP is catastrophically
unreliable (2.27x Knative) and the GNN is not — a bounded claim, not a general win.

**`FALSIFIED` — same-node edges.** Arm B (`full_corpus_siv1_gnn_mp_residual_node_edges`)
trained *with* candidate-restricted same-node edges (0.37x bipartite, present on 80% of
graphs, recorded in the checkpoint sidecar) and was worse than Arm A on every metric:
val acc 62.6% vs 65.6%, test greedy regret 0.4944s vs 0.2621s. Co-location coupling is
not the signal the GNN was missing. Do not re-try this without new evidence.

**`ACTIVE` — the GIN residual.** Arm A (`full_corpus_siv1_gnn_mp_residual`) more than
halves offline greedy regret vs the deployed baseline: **0.5682s → 0.2621s (−54%)**,
top-5 regret 0.0346 → 0.0239. Learned `mp_gate` = 1.08, i.e. the model leans on message
passing slightly *more* once it augments rather than replaces the per-node encoding.
Live re-gate running at `normal_sim_sweeps/mp_residual_gate_20260817`.

**Two reproducibility traps found, both still open.**
1. `run_provenance` records neither the git commit nor `OMP_NUM_THREADS`. The
   2026-08-16 ablation figure of 0.88x Knative on `sparse_p35/s42` is **not reproducible**
   — the current gate gives 1.04x for the same cell/seed/model/config. That arm ran with
   `GNN_DROP_NODE_EDGES=1`, a variable implemented nowhere in the tree today, so the code
   that produced it no longer exists. Treat the gate as the baseline of record.
2. `logit_tied_rate ≈ 0.54` — the scoring head's top-2 margin is under 0.1 on half of all
   live decisions. A model that indifferent is sensitive to FP reduction order, which is
   why thread count matters. If the residual does not move this, the next lever is the
   ranking loss or edge features, **not** the encoder.

---

## RETIRED

| Lineage | Status | Archive | Files | Outcome |
|---|---|---|---|---|
| **pre_gnn_herosim** | `PAPER` | `archive/pre_gnn_herosim/` | 145 | The original HeROsim proactive-autoscaling paper: XGBoost/GPR demand prediction, Bayesian optimisation over infrastructure, LHS sampling, `scenario-*.sh`, and the HRO/HRC/proactive-Knative policies. Superseded by the GNN co-simulation work; kept because the paper cites it. |
| **regime_b** | `FALSIFIED` | `archive/regime_b/` | 38 | Cold-burst regime with `platform_reuse_v1` physics. Phases 0–3.1 closed the gap 125→31 only by distilling `ect_pull`, and `ect_pull` itself lands at Knative level on the coupled trio — so it was never a ceiling to chase. CLAUDE.md already marks Regime B outdated. |
| **soft_combo** | `FALSIFIED` | `archive/soft_combo/` | 6 | Joint combination scoring (`soft_combo`, `soft_combo_conc`) gave no gain over CE on `oracle_split_v1` (commit `d6f1999`). The **loss functions stay live** in `non_unique_lib/soft_combo_loss.py` — `train_near_rtt.py` still imports them; only the experiment wrappers are archived. |
| **warmth_sparse** | `SUPERSEDED` | `archive/warmth_sparse/` | 110 | The warmth/sparse/skew-merged/hub9 series, plus its regen, repair and health-monitor tooling. Superseded by the contention series, which produces the contention the GNN actually needs. Largest single lineage. |
| **model_sweeps** | `SUPERSEDED` | `archive/model_sweeps/` | 38 | One-off sweeps named after their wandb run (`woven_totem`, `silvery_sun`, `ethereal_lake`, `worthy_bush`, `ssc_trash`, `clean_1230`, `mitrix`) plus `atomic21`, `dim14_1060`, `mega_matrix`, `reviewer_triangle`. **The reason the naming convention changed:** a filename encoding a run name tells you nothing about the hypothesis. |
| **topology_sweeps** | `SUPERSEDED` | `archive/topology_sweeps/` | 32 | `tiered_hub` and `bipartite_coordination` topology experiments. |
| **strategic_merge** | `SUPERSEDED` | `archive/strategic_merge/` | 10 | Merged-corpus training strategy, replaced by the siv1 full-corpus approach. |
| **decode_ablations** | `SUPERSEDED` | `archive/decode_ablations/` | 6 | `seqblend`, `seq_reforward`, pull-decode ablations. The decode paths themselves remain in `src/policy/gnn/seq_decode.py`; only the sweep scripts are archived. |
| **hetero_training** | `SUPERSEDED` | `archive/hetero_training/` | 4 | Heterogeneous-graph GNN training. The **`gnn_hetero` policy stays live** (`executesimulation.py` dispatches it); only the training/caching scripts are archived — nothing invoked them. |
| **exact_rtt** | `SUPERSEDED` | `archive/exact_rtt/` | 2 | Exact-RTT regression objective, replaced by near-RTT + CE. |
| **live_finetune** | `SUPERSEDED` | `archive/live_finetune/` | 3 | Live-trajectory finetuning experiment. |
| **old_scripts** | `SUPERSEDED` | `archive/old_scripts_{idk_big,old_bash}/` | 11 | Pre-`generate_gnn_datasets_fast.py` bash pipeline and one-off state-discrepancy analyses. |

---

## Conventions

**A lineage is not finished until it has a row in this table with an outcome.** A sweep
whose result was never written down will be re-run by someone in three months.

**Do not fork a training script per experiment.** That habit produced 40 near-identical
`train_near_rtt_v2_*.py` wrappers that differed only in cache dir and wandb name. New
experiments get a config, not a copy.

**Do not import from `archive/`.** The live tree is verified closed against it. Re-run
the gate after any move:

```bash
# no live file may reference an archive-only filename
python3 - <<'PY'
import subprocess, pathlib, re
all_f = subprocess.check_output(["git","ls-files"], text=True).split()
live  = [f for f in all_f if not f.startswith("archive/")]
arch  = {pathlib.Path(f).name for f in all_f if f.startswith("archive/")
         and f.endswith((".py",".sh",".sbatch"))}
arch -= {pathlib.Path(f).name for f in live}
bad = []
for f in live:
    if not f.endswith((".py",".sh",".sbatch")): continue
    for ln, line in enumerate(pathlib.Path(f).read_text(errors="ignore").split("\n"), 1):
        if "archive/" in line: continue
        for n in arch:
            if re.search(r'(?<![\w.-])'+re.escape(n)+r'(?![\w])', line):
                bad.append(f"{f}:{ln} -> {n}")
print("\n".join(bad) or "CLEAN"); print("broken:", len(bad))
PY
```

Known-benign hits: six comments in `src/placement/{orchestrator,simulation}.py` that
name `executeinitial.py` while explaining why a guard exists. They are prose about a
code path that no longer exists, not references to it.
