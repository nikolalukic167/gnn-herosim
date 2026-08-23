# 🚀 Session Handover (2026-08-23, later session)

**Status:** All work committed and pushed on `feat/network-contention-v1` (`6129e9e`); datalab
synced at the same commit, clean tree both sides, **nothing in flight**. This session added the
**MLP verification baseline** to all three network-contention gates — and the result narrows
the claim the previous session's headline supported.

> Read first: `LINEAGES.md` → search **"The MLP baseline says the GNN's edge is RELIABILITY"**
> (`link_contention_v1`, 2026-08-23). It sits directly under the "backbone win is REGIME-LEVEL"
> subsection, which it qualifies rather than overturns.

## 0. The one-paragraph story

The three gates that establish the network-contention win (`drawgate`, `promo175`, `bbrob`)
compared GNN draws against **Knative only**, which cannot separate *"the graph model wins"*
from *"any learned model wins here"*. The MLP was run as a fourth arm on all 30 cells. On the
summary line the GNN looks dominant — **the MLP's mean margin vs Knative is positive (worse
than Knative) in 5 of 6 conditions, while every GNN arm is negative in all 6**. But that is
driven entirely by a tail: **7 of 30 cells collapse** (up to +509.8%), and on the other 23 the
MLP usually beats *both* GNN arms. `tempfix` wins the head-to-head on only **17/30** cells,
`deployed` on **13/30**. The supported claim is therefore *"the GNN is the only arm that beats
Knative on every cell of every condition tested"* — a **reliability** argument — and **not**
"GNN > MLP on latency", which these 30 cells actively contradict.

## 1. Next session: exactly two tasks

### Task A — run the `tempfix` MLP as a second MLP arm

The MLP arm above used `batch_edge_mlp_full_corpus_siv1_dim22_batchcache.pt` (the standard siv1
baseline). A **corrected-cache MLP exists and was deliberately not run**:
`models/tabular/batch_edge_mlp_full_corpus_siv1_dim22_batchcache_tempfix.pt` (trained
2026-08-22, **datalab-only** — not on this machine; `models/` is gitignored, so rsync it and
md5 both sides if you want it locally).

The question it answers: **does the corrected cache fix the collapse, or does the collapse
survive retraining?** If `tempfix`-MLP collapses on the same cells, the failure is the
pointwise architecture and the reliability claim hardens into an architectural one. If it does
not, the collapse was a training-data artifact and the GNN's whole reliability advantage over
the MLP needs restating.

Mechanically this is small — the infrastructure landed this session:

- `scripts_cosim/datalab/mlp_arm_all_gates.sbatch` is the template. Add a second checkpoint
  dimension, or copy it with `MLP_MODEL` and the `_mlp` sweep-dir suffix changed (e.g.
  `${PREFIX}_${COND}_mlptempfix`). **Keep the block→(cells, workload, `PARITY_EXTRA_ARGS`)
  mapping exactly as it is** — it is matched per-gate to the Knative arm each run is scored
  against, and the bbrob blocks must keep `PARITY_EXTRA_ARGS=""`.
- `score_live_gate_matrix.py` needs **one line**: add the new arm name to `ARM_SUFFIX` pointing
  at whatever result suffix the runner writes (`mlp` → `mlp_dim22` today; the runner names the
  file from the policy, not the checkpoint, so **two MLP arms cannot share one sweep dir** —
  give the second arm its own `SWEEP_DIR`, which the suffix above already does).
- Then re-score all three gates with the extra arm and rewrite the verdict JSONs.
- 30 tasks, `CPU-amd`, ~16h wall is ample (this session's 30 finished well inside it).

**Verify the same three things this session did:** result `total_rtt` non-zero,
`run_provenance.env.INFERENCE_FEATURE_LAYOUT == "dim22"` on every arm (the scorer fails loud
otherwise), and `run_provenance.code.dirty == false` at the same commit as local.

### Task B — do the collapse cells share a structure the GNN sees and the MLP cannot?

**Start from this table, not from scratch.** `averageOccupation` from each result's `stats`,
per cell and arm (`gnn` = `tempfix`):

| gate / condition | cell | mlp | gnn | knative | mlp vs Knative |
|---|---|---:|---:|---:|---:|
| drawgate / nobackbone | cell05_p20 | **1.02** | 5.13 | 8.71 | **+509.8%** |
| drawgate / backbone | cell05_p20 | **0.44** | 2.00 | 1.90 | **+147.4%** |
| promo175 / nobackbone | cell05_p20 | **0.78** | 6.53 | 8.60 | **+365.5%** |
| promo175 / backbone | cell05_p20 | 1.63 | 1.95 | 1.91 | −35.0% (healthy!) |
| bbrob / core8_bw1.5 | cell03_p15 | **0.30** | 1.61 | 1.45 | **+79.0%** |
| bbrob / core8_bw1.5 | cell05_p20 | **0.36** | 1.28 | 1.88 | **+195.4%** |
| bbrob / core4_bw0.5 | cell03_p15 | **0.15** | 0.53 | 0.46 | **+31.4%** |
| bbrob / core4_bw0.5 | cell05_p20 | **0.17** | 0.39 | 0.53 | **+119.1%** |

Three findings already in hand, each of which saves a wrong start:

1. **The collapse is NOT a pure property of the topology — the same cell flips on trace
   alone.** `cell05` under the *same* `a1_backbone_bw1p5` backbone collapses on
   `workload-150-100` (+147.4%, occ 0.44) and is perfectly healthy on `workload-175-100`
   (−35.0%, occ 1.63). Any hypothesis of the form "cell05's graph has property P" must
   therefore explain a trace interaction too. **Frame the question as structure × load, not
   structure.**
2. **Sparsity is not monotone.** `cell03` is the *sparsest* cell (p=0.15) and collapses only
   under the two bbrob configs; `cell05` (p=0.20) collapses in 5 of 6. `cell01` (p=0.25),
   `cell02` (p=0.35) and `cell04` (p=0.50) never collapse. So "sparser ⇒ collapse" is too
   coarse — something distinguishes p=0.20 from p=0.15 here.
3. **The absolute `averageOccupation ≈ 1` detector from
   `memory/herosim-mlp-collapse-is-occupation-collapse.md` DOES NOT transfer to backbone
   runs** — a binding backbone compresses every arm's occupation to ~0.15–2.0, so Knative
   itself sits near 1. Use the **ratio to the Knative arm on the same cell**: every collapse
   here is ≤0.33× Knative, every healthy cell ≥0.41×. Worth promoting into the memory file
   once confirmed on more cells.

Suggested cheap next probes, in order (all read existing artifacts — no new sims):

- `*.decode_stats.json` sits beside every result. Compare `chosen_queue_vs_min` p95/median for
  MLP vs GNN on collapse vs healthy cells; the memory file says the tail is a *minority* of
  decisions that compounds, and this is where that shows.
- Diff the cell topologies directly (`cell_infrastructure/<cell>/infrastructure.json` under
  `a1_backbone_bw1p5`, `bb_core8_bw1p5`, `bb_core4_bw0p5`,
  `full_corpus_siv1_live_gate_20260820`): degree distribution, replica-to-node concentration,
  and how many platforms are reachable at low latency from the busiest ingress. The mechanism
  in the memory file is a packed platform that cannot drain — look for cells with few
  low-latency alternatives to the platform the MLP prefers.
- `src/placement/network_fabric.py:133` `link_wait_total` and `linkWaitTime` are computed and
  **never serialized** (LINEAGES gate-tools row, 2026-08-21). If the collapse is link-side
  rather than platform-side, surfacing that field is the direct measurement — a reporting-only
  change, no behaviour change.

**Do not** treat this as a lineage that needs a new corpus or new training. It is an analysis
of artifacts already on disk; only Task A needs the cluster.

## 2. What this session changed (all committed)

- `scripts_cosim/important/score_live_gate_matrix.py` — `ARM_SUFFIX` lookup, so `mlp` is a
  valid arm. Everything else in the scorer was already arm-generic.
- `scripts_cosim/datalab/mlp_arm_all_gates.sbatch` — **new**, 30 tasks = 3 gates × 2 conditions
  × 5 cells, one checkpoint. Jobs 710450 (smoke, array 5) + 710451 (array 0-4,6-29), all green.
- The three verdict JSONs gained an `mlp` column: **135 insertions, 0 deletions** — no GNN or
  Knative number moved, which is the integrity check that the re-score reused the identical
  result files.
- `LINEAGES.md` — new subsection stating the narrowed claim explicitly, so the stronger
  "GNN > MLP" version does not get written into the paper by someone reading only the table.
- Memory `herosim-mlp-collapse-is-occupation-collapse.md` updated with the 30-cell head-to-head.

## 3. Still-standing context from the previous session

- **The GNN win itself is unchanged**: 3 training draws × (backbone, no-backbone), 2 traces ×
  (backbone, no-backbone), 3 backbone configs — every GNN arm ≥4/5 vs Knative under binding
  bandwidth, 5/5 for all but the weak `prefixctl` draw. Scope limits unchanged: one topology
  family (20c/20s sparse); `workload-125-225` and `workload-200-200` still not run under the
  corrected `dim22` layout.
- **`tempfix` (GNN) is still the promotion candidate** — best artifact on disk, beats deployed
  on both traces in both conditions. Gating it on `workload-125-225` (where deployed is
  weakest) and `workload-200-200` remains open; `workload-200-200` is **not on datalab**
  (224 MB, local only). Reuse `tempfix_promotion_gate.sbatch` with `WORKLOAD` changed.
- **Re-serve any GNN checkpoint whose sidecar says `inference_feature_layout: null`** before
  citing its past results. `mlp_scheduler` reads its own checkpoint, so **MLP results are not
  affected by the layout confound** — that is why this session's MLP numbers needed no caveat.
- `topology_transfer_v1` still has no live cells minted at 60/80 servers.
- Root cause still open: the additive, queue-dominated co-sim target (`logit_tied_rate ≈ 0.54`).
- `main` is far behind `feat/network-contention-v1`, which is acting as trunk. Not merged —
  flagging rather than deciding.

## 4. Environment gotchas

```bash
PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=/root/projects/my-herosim \
  pipenv run python3 scripts_cosim/<script>.py ...
# datalab: eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
# in an .sbatch: export HEROSIM_PY=python3 right after activation
# pin OMP/MKL/OPENBLAS/TORCH_NUM_THREADS=4 for ML runs
```

- **Never `[[ -d X ]] || cp -r`** to stage shared files in a SLURM array — not atomic, killed
  3 tasks of job 710315. Use `mkdir -p` + `rsync -a`.
- Running `pipenv run` from outside the repo creates a **new empty venv** and fails with a
  confusing `ModuleNotFoundError`. Always invoke from the repo root.
- Grep **both** pipenv spellings when auditing the env leak: the shell form and `"pipenv"` in
  Python argv lists.
- **Read `run_provenance.python_env` from the result JSON**, not the sbatch banner — the banner
  describes the process that printed it.

## 5. Restore prompt for next session

```
[CONTEXT RESTORE] feat/network-contention-v1 is pushed and synced to datalab, nothing in flight.
Last session added the MLP baseline as a fourth arm to all 30 cells of the drawgate/promo175/bbrob
contention gates. Result narrows the claim: the MLP's mean margin vs Knative is POSITIVE in 5 of 6
conditions while every GNN arm is negative in all 6, but that is driven entirely by 7 of 30 cells
collapsing (up to +509.8%, the averageOccupation packing failure) -- on the other 23 the MLP usually
beats both GNN arms, and tempfix wins the head-to-head on only 17/30. So the supported claim is "the
GNN is the only arm that beats Knative on every cell", NOT "GNN > MLP on latency". Read LINEAGES.md
"The MLP baseline says the GNN's edge is RELIABILITY" first, then HANDOVER.md section 1, which has
exactly two tasks: (A) run the tempfix MLP as a second arm to see whether the corrected cache fixes
the collapse -- template is scripts_cosim/datalab/mlp_arm_all_gates.sbatch, needs one line in
score_live_gate_matrix.py's ARM_SUFFIX and its own SWEEP_DIR; (B) work out whether the collapse cells
share a structure the GNN sees and the MLP cannot -- start from the occupation table in section 1,
and note the key clue that cell05 collapses on workload-150-100 but is healthy on 175-100 under the
SAME backbone, so it is structure x load, not structure alone. Task B needs no cluster and no
training -- it is analysis of artifacts already on disk.
```
