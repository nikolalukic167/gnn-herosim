# LINEAGES.md — the index of what is known

**Read this before starting work.** It is the one entry point to this repo's research
record: every lineage, every gate-tool correction, every transferable rule, and every
closed question. `simulation_data/REGISTRY.json` does the same for datasets.

This file is an **index only**. It carries a status and a one-line outcome per lineage;
the full record lives in the node file each row links to. Nothing here is a summary you
may cite — cite the node.

## The graph

| | |
|---|---|
| [`docs/lineages/`](docs/lineages/) | **One node per lineage.** Standing, entry points, datasets, and the full dated record. Indexed below. |
| [`docs/lessons.md`](docs/lessons.md) | **The transferable rules** — 350 of them, newest first. What generalises past any one lineage. |
| [`docs/hard-stops.md`](docs/hard-stops.md) | **Falsified — do not revive without new evidence.** Each entry names the measurement that closed it. |
| [`docs/gates/gate-tools.md`](docs/gates/gate-tools.md) | **Corrections to the gates themselves.** Deliberately not filed under a lineage. |
| [`docs/notes/`](docs/notes/) | Design notes on physics and features that outlive any lineage (warmth model, storage contention, separability, corpus regen). |
| [`docs/adr/`](docs/adr/) | Decision records for choices with two live answers (warmth physics, queue contracts, the mandatory placement sweep). |
| [`CONTEXT.md`](CONTEXT.md) · [`PARITY.md`](PARITY.md) · [`CO_SIMULATION_GUIDE.md`](CO_SIMULATION_GUIDE.md) | Vocabulary · cross-venue comparability · the co-sim pipeline. |

## Statuses

| Status | Meaning |
|---|---|
| `ACTIVE` | Current work. Change this code. |
| `REGISTERED` | Pre-registered, not yet run. The registration is signed off and dated; the result is not in. |
| `CLOSED` | The question was answered. Result stands, no further work — re-opening needs a new question, not a re-run. |
| `SUPERSEDED` | Replaced by a later lineage. Result still stands; don't build on it. |
| `FAILED` / `FALSIFIED` | The gate failed, or the hypothesis was disproven. **Do not revive without new evidence.** |
| `SYNTHESIS` | Not an experiment — a cross-lineage reading of several. |
| `PAPER` | Frozen because the paper cites it. Change only with a paper edit. |

## Open — the current program

| Lineage | Status | Outcome |
|---|---|---|
| [**route_b_env_pivot_v1**](docs/lineages/route_b_env_pivot_v1.md) | `ACTIVE` | Current work. Screen registered 2026-08-27. `greedy_stuck` was **decoder myopia on every rung** (458/458 rescued), not a configuration artifact; **AMENDMENT 2 signed off 2026-08-27** replaced the decoder and **H0/H1 counters are now clean** (α=2.0, α=3.0), H2 still VOID-GENERATION. Both controls then generated under AMENDMENT 1 and **both PASS S0**. Bars read 2026-08-27: **H0 VOID-TIE-INDETERMINATE on S1**, **H1 FAILS S1**. **S2 (the kill bar) is uncomputable on this grid** — `t1x` needs ≥82 sweep rows, the scarcity squeeze gives 16 and 64, refused 204/204. S3/S4 were blocked only by a **transfer-tool defect** (one arm's saturation refusal aborted the whole run); fixed, so **S4 PASSES both rungs** (`hop+coupling` closes 0.0000, full stratum) and **S3 lands on the bar** (H0 0.500000, H1 0.500835 vs `≤ 0.5`). Ladder **not** exhausted (H2 VOID-GENERATION, H3 never generated). S2 was then measured to be a **grid** problem, not a competitor problem — a 204/204 wide-arm probe fits `t1x` on 41/41 firing datasets with **zero saturation in both arms**, contention binding *harder* (0.91) and the squeeze untouched (2 hosting nodes everywhere) — and **H3 generates 0/204 as registered** (8 tasks, uniqueness-exhausted on a pool of 2 or 4). **AMENDMENT 3** moves the H2/H3 grid only and leaves H0/H1 alone. The pair it proposes was probed at 4 tasks (204/204) and **passes all four bars** (S1 0.2843 reg / 0.3137 pess, S2 58/58 fitted 0 saturated, S3/S4 0.0000) — so signing it likely yields the first **PIVOT-CANDIDATE**. Probe, not a rung: S0's control is ungenerated and the rung gets fresh seeds. **AMENDMENT 3 SIGNED OFF 2026-08-28** — H2/H3 move to `per_server` 4/5, H2 on fresh seeds 3401–3417, H3's skip threshold re-derived to 3e10. **The amended H2 was then generated and is VOID: its paired separable control FAILS S0** — `optimistic` **0.4853** vs a `≤ 0.02` bar at the primary α=2.0 (band 0.4853–0.8333, max 59.5%). Both corpora 204/204 clean, `{1680: 102, 3024: 102}`, squeeze intact `{2: 204}`. Not the tie artifact, not arm-confounded (per-arm 0.6765/0.6471, zero censoring), **not the scorer** (independent verifier 1e-9 on all 612 cells). **S1–S4 not read and must not be.** The failure appears only where the *cap* excludes the componentwise plan — unconstrained the band is exactly 0.0000, and where overlap's uniqueness alone excludes it (87/204) the surrogate still recovers the optimum every time. **Tightness is not the discriminator**: H1's control passes at `cw_infeas` 0.963 while H2's fails at 0.877. `node_caps` is plan-independent, so the evidence points at non-additive cost in the control arm, **mechanism unresolved**. **H3 generates** (40,320 rows, pool-8 arm, 0 skips) where as registered it gave 0/204 — the amendment's other half confirmed. Now open and bigger than one rung: **is S0 readable on any `replica_overlap` rung at all?** H2/H3 are the only two, and H3 inherits H2's grid. The earlier two open questions closed without action: the S0 rule is moot, S3's knife-edge stays. |
| [**route_b_v1**](docs/lineages/route_b_v1.md) | `ACTIVE` | **Stage 1 PASS** — contention + coupling produces the non-pointwise structure five mechanisms and route A could not. **Stage 2 NO-GO-PREPROBE** (2026-08-26): a GNN cannot beat pointwise-plus-prefix on this environment even at memorization. Forked to `route_b_env_pivot_v1`. |
| [**route_c_link_transfer_v1**](docs/lineages/route_c_link_transfer_v1.md) | `REGISTERED` | Screen registered 2026-08-26 before generation; **name is reserved and is only claimed if the screen passes.** Asks whether an environment where link waiting is a material share of RTT resists a fairly-armed pointwise competitor. |
| [**siv1_full_corpus**](docs/lineages/siv1_full_corpus.md) | `ACTIVE` | First real live-gate FAILED, then **SUPERSEDED the same day** — it measured an uncommitted code diff, not the model. The synced-code re-gate (job 709163) wins **5/5 on `workload-150-100` and `workload-175-100`**, 2W/1T/2L on `workload-125-225`. Corrected-cache retrain (job 709234) is ungated. |
| [**trainer_determinism_v1**](docs/lineages/trainer_determinism_v1.md) | `ACTIVE` | The seed fix reached **1 of 4 trainers**. Three defect classes now — newest: `prepare_graphs_cache` seeded 42 at module import and clobbered every GNN draw's seed. `tests/test_trainer_determinism.py` covers every trainer; run it before training anything you intend to gate. |

## Closed — answered, do not re-run

| Lineage | Status | Outcome |
|---|---|---|
| [**route_a_v1**](docs/lineages/route_a_v1.md) | `FALSIFIED` | **NO-GO.** DAG + distance is genuinely pairwise and still pointwise-optimal. Breaking separability is **necessary but not sufficient** — you also need contention, which is what route B tests. |
| [**program_verdict_v1**](docs/lineages/program_verdict_v1.md) | `CLOSED` | Terminal answer to the D3 fork: **the supervised co-sim path to "GNN > MLP on latency" is closed by measurement.** The reliability/regime win exists on the 30-cell record but is exploratory. P1 (closed-loop objective) is the only remaining path to the latency claim. |
| [**cosim_deepdive_v1**](docs/lineages/cosim_deepdive_v1.md) | `CLOSED` | Does the target's additivity come from the synthetic t=0 snapshot regime? **No — live-visited states are equally additive** (4,400 swept live states, median additive R² 0.99999). The GNN's dispersal edge is a closed-loop property no single-batch regret target can express. |
| [**graph_structure_physics**](docs/lineages/graph_structure_physics.md) | `CLOSED` | **The co-sim target is pointwise-separable, so a pointwise MLP is the correctly specified model class and the GNN cannot beat it by training.** Additive R² 0.988 → 1.00000 across collections. Deep queues as a coupling lever: FALSIFIED — the lever runs backwards. The finding is closed; its **M4 diagnostic stays live** and is the standing separability gate every later lineage is measured with. |
| [**throughline**](docs/lineages/throughline.md) | `SYNTHESIS` | Cross-lineage synthesis: four mechanisms, one collapse. **In this simulator, coupling is either count-shaped or negligible** — and the negligible half is demonstrated, not assumed. |
| [**p5b_draw_study**](docs/lineages/p5b_draw_study.md) | `CLOSED` | **Q1 = LOTTERY, Q2 = DRAW-DOMINATED.** The MLP trainer never seeded torch, so every MLP checkpoint before 2026-08-24 is an unreproducible draw. Collapse counts swing 0→26 on the seed alone. **Retires "the MLP collapses 7/30"** and every inference built on it. |
| [**p5b_candidate_relative**](docs/lineages/p5b_candidate_relative.md) | `CLOSED` | **INDETERMINATE — and the indeterminacy is the result.** Kills the mechanism sentence "a pointwise scorer collapses because it cannot condition on its peers": one arm has exactly that conditioning, uses it, and stops collapsing. Resolved by `p5b_draw_study` — it was the training draw. |
| [**gnn_draw_study_v1**](docs/lineages/gnn_draw_study_v1.md) | `CLOSED` | **INDETERMINATE**, and **"the GNN never collapses" is FALSIFIED** — 2 of 8 seeded draws collapse. Direction is clear, the test is under-powered. |
| [**m3_batch_makespan_v1**](docs/lineages/m3_batch_makespan_v1.md) | `CLOSED` | Below both registered thresholds, then **closed permanently** by the argmax-flip diagnostic: when per-branch costs are separable and component choices are free, min-max and min-sum share the same argmin, so re-scoring under makespan was never an independent escape at any fan-out width. |
| [**cache_live_divergence_audit**](docs/lineages/cache_live_divergence_audit.md) | `CLOSED` | Platform reordering: **18/18 collections, BENIGN** (no recache, no asterisk). Dims 9-11 temporal estimate: **8/18, REAL**. The parity verifier now compares by platform identity instead of position. |
| [**serving_speed_v1**](docs/lineages/serving_speed_v1.md) | `CLOSED` | The episode cost was `Data.to()`, not the device — 174× per-call win from moving tensors only. CPU serving is slower than cuda; the cpu default exists for parity. |

## Falsified / failed — the mechanisms that did not work

| Lineage | Status | Outcome |
|---|---|---|
| [**topology_transfer_v1**](docs/lineages/topology_transfer_v1.md) | `FAILED` | Changed the win condition from per-plan accuracy to inductive generalization across topology sizes — and **all four arms FAILED**, including `gnn_topo`, the only arm that ever had backbone topology in the graph. ⚠ Never live-gated; checkpoints were not persisted until the 2026-08-21 unblock. |
| [**mp_parity**](docs/lineages/mp_parity.md) | `FALSIFIED` | Both arms falsified, gate FAILED on both pre-registered criteria. The residual pays only where interaction exists and costs RTT where the target is additive — which is `graph_structure_physics`' finding arriving from the model side. |
| [**link_contention_v1**](docs/lineages/link_contention_v1.md) | `FALSIFIED` | Per-link capacity over a multi-hop backbone. Gate FAILED on both criteria. **The link controls do not repair (median 0.000)** — genuinely new, and the first escape from the one-integer control — but node-collision coupling still dominates and the magnitude is 14× below the gate. |
| [**network_contention_v1**](docs/lineages/network_contention_v1.md) | `SUPERSEDED` | Shared per-node ingress bandwidth: **the physics works and is opt-in**, but the corpus lever is replica concentration, not bandwidth. One of the four mechanisms in the throughline. |
| [**shallow_longexec_v1**](docs/lineages/shallow_longexec_v1.md) | `FALSIFIED` | The last physics attempt before the paper pivot, and the **fourth independent confirmation** of the throughline: one-integer repair 100%. |
| [**shallow_v1**](docs/lineages/shallow_v1.md) | `SUPERSEDED` | Shallow queues lower the pointwise ceiling but **MISS the coupling gate**. Contains a **retraction**: the coupled(>1%) = 31.0% figure does not reproduce at full corpus size. Read the retraction before citing any number in this node. |
| **contention_v4_v5** | `FALSIFIED` | Deep queues + coupling optimisation — **moved the corpus the wrong way** (additive R² 0.988 → 0.9997). |

## Standing apparatus — baselines, gates and tooling

| Lineage | Status | Outcome |
|---|---|---|
| **contention_v2_v3** | `ACTIVE` | Baseline contention series that v4/v5 is measured against. |
| **sealed_holdout** | `ACTIVE` | The honest generalisation gate. |
| **coupled_trio** | `ACTIVE` | The three-cell coupled comparison. **ECT is not a ceiling and not a distillation teacher** — 0.98–1.13× Knative; see [`docs/hard-stops.md`](docs/hard-stops.md). |
| **encoder_ablation** | `ACTIVE` | Is the graph encoder doing work, or is it the features? |
| **seed_variance** | `ACTIVE` | Seed spread on the contention_v2 GNN. |
| **queue_feature_contract** | `ACTIVE` | `legacy_v0` vs `scale_invariant_v1`. See `docs/adr/0002-two-queue-feature-contracts.md`. |
| **dataset_metadata** | `ACTIVE` | Produces `REGISTRY.json`, `METADATA.json`, `COMPATIBILITY_MATRIX.json`. |

These carry no separate node file. `contention_v4_v5` is listed here too — it is
`FALSIFIED` and has no node, but its entry points are still the only record of it.

| Lineage | Entry points | Datasets | Notes |
|---|---|---|---|
| **contention_v4_v5** | `scripts_cosim/datalab/contention_v4_deepq_cosim.sbatch`, `contention_v5_quick_test.sbatch` | `contention_v4_pilot`, `contention_v5_quick_test` | Deep queues + coupling optimisation — the attempt at giving the GNN real graph structure to exploit. **`FALSIFIED` 2026-08-17: moved the corpus the wrong way** (additive R² 0.988 → 0.9997). See `graph_structure_physics`. |
| **contention_v2_v3** | `important/run_contention_v{2,3}_train_and_live_gate_nohup.sh`, `important/compare_contention_v2_live_gate.py` | `contention_v2{,_verify}`, `contention_v3` | Baseline contention series the v4/v5 work is measured against. Trainers: `train_near_rtt_v2_contention_v{2,3}_dim14_ce_only.py`, `train_mlp_contention_v{2,3}_dim22_batchcache.py`. |
| **sealed_holdout** | `important/run_contention_v2_873_sealed_holdout{,_rebaseline}.sh`, `compare_sealed_live_holdout.py`, `datalab/sealed_holdout_gpu.sbatch` | `contention_v2` | The honest generalisation gate. |
| **coupled_trio** | `important/run_contention_v2_873_coupled_trio.sh`, `chain_coupled_trio_then_rebaseline.sh` | `contention_v2` | See memory note: ECT is not a ceiling. |
| **encoder_ablation** | `important/run_gnn_encoder_ablation.sh`, `compare_encoder_ablation.py` | contention series | Is the graph encoder doing work, or is it the features? |
| **seed_variance** | `scripts_cosim/run_gnn_seed_variance_siv1.sh` | contention_v2 | Uses `train_near_rtt_v2_contention_v2_dim14_ce_only.py`. |
| **queue_feature_contract** | `src/placement/queue_features.py`, `scripts_cosim/test_queue_features.py`, `verify_cache_live_feature_parity.py` | all | `legacy_v0` vs `scale_invariant_v1`. See CLAUDE.md. |
| **dataset_metadata** | `scripts_cosim/{extract_dataset_metadata,validate_dataset_collection,compute_compatibility_matrix}.py` | all | Produces `REGISTRY.json`, `METADATA.json`, `COMPATIBILITY_MATRIX.json`. |

Shared core (not a lineage — everything depends on it): `src/placement/`,
`src/policy/{gnn,tabular,knative*,determined,evaluator}/`, `src/executecosimulation.py`,
`src/executesimulation.py`, `scripts_cosim/generate_gnn_datasets_fast.py`,
`src/notebooks/non_unique_lib/`.

## Retired code

Retired code lives in [`archive/`](archive/README.md) — moved with `git mv`, so
`git log --follow` still works. Nothing was deleted. Restore point: tag
`pre-cleanup-2026-08`.

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

## Conventions

**A lineage is not finished until it has a row in this index and a node under
[`docs/lineages/`](docs/lineages/) with an outcome.** A sweep whose result was never
written down will be re-run by someone in three months.

**A gate tool's own correctness is a recorded fact, not folklore.** When a gate turns out to
have been measuring the wrong thing, it goes in
[`docs/gates/gate-tools.md`](docs/gates/gate-tools.md) — not inside whichever lineage
happened to trip over it. Two of this repo's near-misses came from a tool-level fact being
buried in a lineage narrative.

**A rule that outlives its lineage goes in [`docs/lessons.md`](docs/lessons.md).** The node
records what one investigation found; `lessons.md` records what generalises past it. A
falsified direction also gets a line in [`docs/hard-stops.md`](docs/hard-stops.md), with the
measurement that closed it.

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

**Session handovers are ephemeral and are not committed.** A handover is one session's
note to the next; it is not a record. Anything in it that is still true a week later
belongs in a lineage node, `docs/lessons.md`, or `docs/gates/gate-tools.md` — put it
there instead. Three committed `HANDOVER*.md` files were retired on 2026-08-27 for
drifting out of agreement with this index while claiming the same facts. If you need to
hand off, write the file outside the repo (the session scratchpad) or leave it untracked.

**One fact, one home.** Before adding a paragraph, find the file that already owns that
fact and edit it. This index existed at 4,995 lines because five files each narrated the
same experiments and drifted apart; the duplication cost more than the writing saved.
