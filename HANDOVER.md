# 🚀 Session Handover (2026-08-23)

**Status:** All work committed and pushed on `feat/network-contention-v1`; datalab synced and
idle apart from one probe (job 710432, see §4). This session's headline: **the GNN's win under
network contention is regime-level and robust on every axis tested** — and the two results
that most undermined the GNN's story (the "training-draw lottery" and the corrected-cache
"0/15 FAIL") were **both confounded by a serving bug**, now found, measured and fixed.

> Read first: `LINEAGES.md` → search **"The backbone win is REGIME-LEVEL"**
> (`link_contention_v1`, 2026-08-23). Then the two superseding notes it points back to.

## 0. The one-paragraph story

The plan was to harden the recorded backbone win (deployed GNN, 5/5 vs Knative at −24.0%)
against training-draw variation, because the 2026-08-22 variance control said a single
checkpoint's verdict is a claim about one draw. Setting that gate up surfaced a **serving
confound** first: `load_gnn_model` silently defaulted an undeclared platform-feature layout to
`atomic21`, and the deployed checkpoint declares `dim22` in its sidecar while `prefixctl` and
`tempfix` declare none. So **every deployed-checkpoint gate served `dim22` and both
alternate-draw gates served `atomic21`** — worth **up to 40.8% of live `total_rtt`**, 102× the
noise floor, with `dim22` uniformly better. Re-running everything at a fixed layout: all three
draws win ≥4/5 vs Knative under a backbone, the corrected-cache checkpoint (`tempfix`) turns
out to be the **best artifact on disk** rather than a failure, and the lottery shrinks to a
real-but-modest draw effect. Extending to a second trace and two more backbone configs
reproduced it all to within ~2pp. Along the way: the backbone rng coupling was fixed at the
root, and a **53rd pipenv call site** was found that meant every datalab gate ever had run in
the rogue venv despite the 2026-08-21 "closed" claim.

## 1. What is now established (safe to build on)

**The GNN win, at a fixed `dim22` layout, vs Knative, `node_disk_v2`:**

| axis varied | result |
|---|---|
| 3 training draws (deployed, prefixctl, tempfix) × backbone | **all ≥4/5**; 5/5 for two of three |
| 2 traces (150-100, 175-100) × {backbone, no-backbone} | **all 5/5** for deployed and tempfix |
| 3 backbone configs (`n_core` 4/8, `bw` 1.5/0.5) | **30/30 cells**, mean margins within 2pp |

- **Binding bandwidth roughly doubles the GNN's edge.** Every draw improves markedly when the
  backbone is added: deployed −9.4→−24.0%, prefixctl +2.3→−8.6%, tempfix −14.0→−34.1%.
- **`tempfix` is the promotion candidate, not a failure.** 5/5 on both traces in both
  conditions, beating the deployed checkpoint on mean margin in all four, losing on only 2 of
  20 individual cells. Its 2026-08-22 "0/15 FAIL" was the layout, not the cache — **the
  corrected cache is better**, which reverses the "mismatch was an accidental regularizer"
  reading entirely.
- **The lottery is real but much smaller than recorded.** `prefixctl` is genuinely the weak
  draw (cell04: +24.5% no-backbone, +35.4% backbone) and that survives the layout fix. The
  qualitative rule stands — a single-checkpoint gate is a claim about one draw — but the
  2026-08-22 magnitudes were draw **plus** layout.
- **Venue/code is inert, confirmed again.** The deployed arm re-run on datalab at a clean
  committed tree reproduces its 2026-08-21 local numbers to ≤0.2pp on all five cells.

## 2. Bugs found and fixed this session (all with tests)

1. **Inference-layout guess** (`load_gnn_model`) — `task_dim=3/platform_dim=14` is valid under
   both `atomic21` and `dim22`, which give the same columns different meanings, so nothing
   raised. Now fails loud when neither the sidecar nor the env declares one.
   `tests/test_inference_layout_contract.py`. **GNN-specific** — `mlp_scheduler` reads its own
   checkpoint, so all MLP results are unaffected.
2. **The trainer recorded the layout from the shell** (`train_near_rtt.py`), defaulting to
   `null` whenever an sbatch forgot to export it — the origin of (1). Now taken from the
   cache's `metadata.json`, which already records it, with a hard error on disagreement. Both
   ends of the loop are now closed.
3. **Backbone rng coupling, fixed at the root** — `network.backbone.rng_stream`.
   `independent_v1` derives jitter from the seed alone; `legacy_v0` stays the default in
   `build_core_backbone` so every existing corpus still regenerates byte-identically (that
   bit-reproducibility was the stated reason for deferring this on 2026-08-21). New corpora
   default to `independent_v1` via `generate_gnn_datasets_fast.py`.
4. **Parity classifier couldn't see backbone repair edges** — under a backbone a repair edge's
   latency is a route sum, not `base_latency`, so real repair edges were reported
   "unexplained" and waived. Now checked against each edge's own recorded route (stricter, not
   a relaxation). Parity now reports `repair=34/174` where it used to say `0/174`.
5. **The 53rd pipenv call site.** `run_simulation.py` and
   `important/run_normal_sim_config_sweep.py` built `["pipenv","run","python",...]` as argv
   **lists**, invisible to the 2026-08-21 grep sweep over the shell spelling. Since
   `run_simulation.py` is what launches the simulator, the shell `HEROSIM_PY` guard pinned only
   the wrapper: **every datalab gate through it ran in the rogue venv** (torch 2.12.0+cu130)
   while its sbatch banner said otherwise. Job 710366 is the first datalab gate whose result
   JSON shows the declared `gnn` env. `PARITY.md` and `CLAUDE.md` corrected — both asserted
   this was closed.
6. Smaller: fail-loud hardening across 6 gate/validation scripts; a decode-stats probe that
   could discard a valid placement; `network_graph.py` treating a genuine 0.0-latency access
   link as unset (verified to move no served logits).

## 3. Reproducibility / provenance state

- Working tree clean, everything pushed; datalab at the same commit with a clean tree.
- `topology_transfer_v1`'s 10 checkpoints + sidecars rsynced local, **md5-verified 20/20**.
  Its 5 eval JSONs and the two previously-untracked sbatch are now in git.
- New backbone cells are **regenerable, not committed** — `simulation_data/**` is gitignored,
  and `important/make_backbone_gate_cells.py` reproduces them deterministically from a seed.
- **Read `run_provenance.python_env` from the result JSON** to know which interpreter served.
  An sbatch banner describes the process that printed it — that is how bug (5) hid.
- Gate results carry `run_provenance.code` (commit / dirty / diff hash), so cross-venue
  disagreements are a lookup.

## 4. In flight / open

- **Job 710432** (`interpreter_delta_probe.sbatch`) — was still RUNNING at handover. Re-runs
  one drawgate cell under the fixed interpreter and diffs `total_rtt` against the same cell
  produced under the rogue venv. `PARITY.md` measured the library axis at exactly 0.0 on
  *logits* but never end-to-end on a 301k-task trace. Expected inside the 0.1–0.4% noise
  floor; check `logs/interp-delta-710432.out`, which prints its own verdict line. Only matters
  for comparing pre-fix (150-100 drawgate) against post-fix (175-100, bbrob) numbers — each
  gate is internally consistent regardless.
- **Promote `tempfix`?** Wants `workload-125-225` (where the deployed checkpoint is weakest,
  2W/1T/2L) and `workload-200-200` first. `workload-200-200` is **not on datalab** (224MB,
  local only). Reuse `tempfix_promotion_gate.sbatch` with `WORKLOAD` changed.
- **Re-serve any GNN checkpoint whose sidecar says `inference_feature_layout: null`** before
  citing its past results — `prefixctl`, `tempfix`, and anything else trained before fix (2).
- **`topology_transfer_v1`** still has no live cells minted at 60/80 servers; unchanged.
  `make_backbone_gate_cells.py` is a usable template for minting them.
- **Root cause still open:** the additive, queue-dominated co-sim target
  (`logit_tied_rate ≈ 0.54`). Untouched.
- `main` is far behind `feat/network-contention-v1`, which is acting as trunk. Not merged —
  flagging rather than deciding.

## 5. Environment gotchas

```bash
PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=/root/projects/my-herosim \
  pipenv run python3 scripts_cosim/<script>.py ...
# datalab: eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
# in an .sbatch: export HEROSIM_PY=python3 right after activation
# pin OMP/MKL/OPENBLAS/TORCH_NUM_THREADS=4 for ML runs
```

- **Never `[[ -d X ]] || cp -r`** to stage shared files in a SLURM array — not atomic, killed
  3 tasks of job 710315. Use `mkdir -p` + `rsync -a`.
- Running `pipenv run` from a directory outside the repo creates a **new empty venv** and fails
  with confusing `ModuleNotFoundError`. Always invoke from the repo root.
- Grep **both** pipenv spellings when auditing the env leak: the shell form and `"pipenv"` in
  Python argv lists.

## 6. Restore prompt for next session

```
[CONTEXT RESTORE] feat/network-contention-v1 is pushed and synced to datalab; only job 710432
(interpreter delta probe) may still be running. This session found that load_gnn_model silently
defaulted an undeclared platform-feature layout to atomic21, so every deployed-checkpoint gate
served dim22 while the prefixctl and tempfix gates served atomic21 -- worth up to 40.8% of live
total_rtt. That confounded BOTH the "training-draw lottery" and the corrected-cache "0/15 FAIL".
Re-gated at a fixed dim22 layout: the GNN beats Knative 5/5 under a binding-bandwidth backbone
across 3 training draws, 2 traces and 3 backbone configs (30/30 cells), and tempfix -- the
corrected-cache checkpoint -- is now the best artifact on disk, beating the deployed one on
every trace/condition. Fixed at both ends (trainer now reads the layout from the cache; serving
refuses to guess). Also fixed the backbone rng coupling at the root and found a 53rd pipenv call
site meaning every prior datalab gate ran in the rogue venv. Read LINEAGES.md "The backbone win
is REGIME-LEVEL" first, then HANDOVER.md §4 -- top item is gating tempfix on workload-125-225
and workload-200-200 to decide promotion.
```
