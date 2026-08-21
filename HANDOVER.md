# 🚀 Session Handover (2026-08-21, late evening)

**Status:** The siv1 re-gate is **done and CONFIRMED** — the recorded FAIL is formally
superseded, not merely re-graded. All three deferred code fixes are implemented but sit on an
**unmerged branch**. A corrected-cache retrain and one local sweep are in flight. The MLP's
catastrophic tail is root-caused. `topology_transfer_v1` is unblocked.

> Read first: `LINEAGES.md` → `siv1_full_corpus` → "✅ The prediction is CONFIRMED — job
> 709163". Then the two subsections after it (MLP tail root cause; the corrected-cache
> train/serve gap), and `### topology_transfer_v1 — unblocked` further down.

## 0. The one-paragraph story

Job 709163 (synced code, 15/15 COMPLETE) closed the case the previous session opened. The GNN
arm reproduces the local working-tree numbers on **all five cells to +0.03%–0.40%**, inside its
own 0.1–0.4% run-to-run floor on four of them. `gnn/cell01` went **65.8M → 50.6M** with the
checkpoint, cells, trace, cluster and GPU partition all held fixed — only the committed code
changed. Head-to-head vs Knative: **2W/1T/2L** on `workload-125-225`, identical to the local
verdict; the checkpoint still wins 5/5 on `workload-150-100` and `workload-175-100`. The
environment measurement is now confirmed at the `total_rtt` level, not just at the logit level.
Three things came out of the follow-up work: the MLP tail is an **occupation collapse**, the
deployed checkpoint has a **31.7%-of-rows train/serve feature gap** (retrain running), and the
`topology_transfer_v1` serving port is a **three-module rename** containing a silent trap.

## 1. FIRST THING TO DO: two in-flight jobs

**(a) Datalab retrain 709235** — GNN on the corrected (post-dims-9-11-fix) cache. At handover:
epoch 40/100, ~25 s/epoch, so ~25 min left.

```bash
ssh datalab 'squeue -u nikola.lukic; cd ~/gnn-herosim && \
  ls -la models/near-rtt-v2-full-corpus-siv1-dim14-ce-only-tempfix.* && \
  tail -5 logs/fc-siv1-gnn-709235.out'
```

**When it lands, gate it — that is the whole point of the retrain and it is not yet run.** Same
15-task gate, three traces, against the deployed checkpoint's recorded numbers (§3):

```bash
ssh datalab 'cd ~/gnn-herosim && for wl in 125-225 150-100 175-100; do
  sbatch --export=ALL,\
GNN_MODEL=models/near-rtt-v2-full-corpus-siv1-dim14-ce-only-tempfix.pt,\
WORKLOAD=data/nofs-ids/traces/workload-${wl}.json,\
SWEEP_DIR=simulation_data/normal_sim_sweeps/siv1_tempfix_gate_${wl} \
    scripts_cosim/datalab/full_corpus_siv1_live_gate.sbatch; done'
```

Score with `compare_sealed_live_holdout.py --sweep-dir <sweep>`; **write the outcome into
`LINEAGES.md` either way** — a retrain that beats the deployed checkpoint changes what gets
deployed, and one that loses is the more interesting result (see §4.3).

**(b) Local `a4_wl200200`** (800k events, PAR=2). At handover **8/15** results: 5 knative done,
MLP on cells 04/05, then 5 GNN cells. ~2–2.5 h.

```bash
ls simulation_data/normal_sim_sweeps/a4_wl200200/results/*_s0_*.json | grep -v decode | wc -l  # 15 = done
pgrep -af run_rest2.sh || echo "runner gone"
```

## 2. ⚠ THE UNMERGED BRANCH — do not lose this

All three deferred fixes are implemented, tested and pushed on **`fix/deferred-gate-fixes`**
(`374accc`), developed in a git worktree specifically so the in-flight local sweep kept running
on unchanged code. **It is not merged**, because merging mid-sweep splits `a4_wl200200` across
two code versions — the exact failure this session spent its first hours undoing.

```bash
# ONCE a4_wl200200 has all 15 results and no executesimulation is running:
git merge fix/deferred-gate-fixes
PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=$PWD pipenv run python3 -m pytest \
  scripts_cosim/test_autoscaler_scaledown_determinism.py tests/ -q      # expect all green
git push && ssh datalab 'cd ~/gnn-herosim && git pull --ff-only'
```

What is on it:

1. **Scale-down tie-break → total key** in `{gnn,knative,knative_network}/autoscaler.py`.
   Removes the residual 0.05–0.1% run-to-run floor. Guard test went 3 failed → 9 passed.
   `mlp_batch` inherits the gnn autoscaler. The other seven policies deliberately untouched.
2. **`run_provenance` now records code + interpreter** — `describe_code_provenance()` stamps
   commit / branch / dirty / `diff_sha256` / `changed_files`, plus `python_env` and
   `env_fingerprint`. 7 tests. A future 708549 is a one-command diff instead of a probe matrix.
3. **`HEROSIM_PY` leak closed** at all 52 call sites / 18 scripts + 3 sbatch, plus
   `envs/herosim-lock.txt` — which `PARITY.md` and `CLAUDE.md` had both called canonical while
   it did not exist.

**After merging, the first gate run on the new code is not comparable to anything recorded
before it** (the tie-break fix moves `total_rtt` by up to 0.1%). That is fine and expected —
just do not quote a pre-merge and a post-merge number in the same row without saying so.

## 3. What is now established (safe to build on)

**Deployed checkpoint** (`near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt`), served
train-consistent features, margins vs knative (− = GNN better):

| trace | per cell (01..05) | verdict |
|---|---|---|
| workload-125-225 (**datalab 709163**) | +8.7 / −5.3 / +17.6 / −7.6 / −0.3 % | 2W/1T/2L |
| workload-125-225 (local) | +8.3 / −5.3 / +17.4 / −7.6 / −0.06 % | 2W/1T/2L |
| workload-150-100 | −0.9 / −12.9 / −14.8 / −5.3 / −12.9 % | **5/5 W** |
| workload-175-100 | −9.4 / −7.6 / −7.5 / −10.8 / −10.8 % | **5/5 W** |
| workload-150-100 + backbone 1.5MB/s | −31.8 / −24.4 / −26.7 / −8.9 / −28.0 % | **5/5 W** |

- **Cross-venue agreement: +0.03% to +0.40%.** Repo sync is sufficient; the venue is not a
  variable. Confirmed at both the logit level (exactly 0.0) and the `total_rtt` level.
- **The MLP's catastrophic tail is `averageOccupation` collapsing to ~1**, not
  `intra_batch_platform_collisions` (normal in every collapsed run) and not physics (every
  component identical). The MLP *wins by packing and loses by packing*: healthy occupation 13.6
  vs Knative 10.6 vs GNN 5.6, and its pointwise score has no queue-relative term. 8 collapse
  instances across two traces and two venues; **the GNN has never collapsed in 20 cell-runs.**
  `averageOccupation ≈ 1` is the cheap detector — no need to parse the 200 MB result JSON.
- **The deployed checkpoint's own train/serve gap is 31.7% of platform rows** (100% of graphs,
  max delta 2.775 — ~320× the `shallow_v1` figure). Its wins are real but are not the ceiling.

## 4. Open threads, prioritized

1. §1 — gate the retrain; finish `a4_wl200200`; write both into `LINEAGES.md`.
2. §2 — merge `fix/deferred-gate-fixes` once nothing is mid-run.
3. **The retrain's own risk.** A corrected-cache model is *expected* to be better, but the
   deployed one won live **despite** a 31.7% mismatch — so if the retrain loses, the honest
   reading is that some of the deployed checkpoint's edge came from the mismatch, which would
   be a finding, not a regression to fix. A matched **MLP** retrain has not been run; the MLP
   moved only ~1% under the dims 9-11 fix, so it is the fair comparison arm.
4. **`logit_tied_rate ≈ 0.54` / `confident_worse_queue_rate ≈ 0.8`** — the model still has no
   sharp ranking on half its decisions (root cause: the additive, queue-dominated co-sim
   target, `graph_structure_physics`). Untouched this session. The corrected-cache retrain is
   the first test of whether the feature bug was contributing to it.
5. **`topology_transfer_v1` partial gate** — now blocked on ~14 GPU-hours **and nothing else**.
   §a (checkpoint persistence) landed; §b (serving port) turned out to be a three-module
   rename, verified bit-exact. Not launched: large speculative spend on a `FAILED` lineage,
   and it would contend with the retrain for GPUs.
6. **A6 (`soft_combo` live retest) is dead as specified** — checked, not assumed. Both
   `oracle_split` checkpoints are sidecar-less *and* take 16 platform features against the gate
   cells' 14. Needs a retrain under a recorded contract before it is worth anything.

## 5. Traps confirmed this session (all now in the docs)

- **`load_state_dict(strict=True)` is not a compatibility check.** Ablation arms load into
  `TaskPlacementGNN` cleanly under the default `mp_residual=False` and then compute different
  logits (0.196, different argmaxes). Bit-exact under `mp_residual=True`. Architecture flags
  invisible in weight shapes must come from the contract, never from a successful load.
- **An import-closure preflight only covers eager imports.** Job 709234 passed one and died
  84 s later on a lazy in-function sklearn import (`training_contract.py:128`).
- **A micromamba env can hold two disagreeing installs.** `conda-meta` said scikit-learn 1.9.0
  (built for numpy 2); pip said 1.6.1; numpy was 1.26.4; scipy was damaged the same way.
  Repaired with `pip install --no-deps --force-reinstall scipy==1.15.3 scikit-learn==1.7.0`,
  then **proved inert** with `verify_venue_parity.py --mode logits` (0.0, 0/256 flips).
- **Recache/train runners `rm -rf` their output dir and derive the checkpoint name from the
  wandb run name.** Both are now overridable (`CACHE_DIR`, `WANDB_RUN_NAME` → `OUT_CKPT`); the
  defaults still point at the deployed checkpoint's only training data.

## 6. Environment gotchas (unchanged)

```bash
PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=/root/projects/my-herosim \
  pipenv run python3 scripts_cosim/<script>.py ...
# datalab: eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
# pin OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_NUM_THREADS=4 for ML runs
# ssh alias `datalab` works; the raw hostname with BatchMode=yes does not (key is agent-loaded)
```

## 7. Restore prompt for next session

```
[CONTEXT RESTORE] The siv1 live-gate FAIL is formally SUPERSEDED: the synced-code re-gate
(job 709163, 15/15) reproduces the local re-grading on all 5 cells to +0.03%-0.40%, so
cell01's 65.8M -> 50.6M was a code diff, not the model. GNN is 2W/1T/2L on workload-125-225
and 5/5 on 150-100 and 175-100. Three follow-ups landed: the MLP tail is an occupation
collapse to ~1 (not collisions), the deployed checkpoint has a 31.7%-of-rows train/serve
feature gap (retrain = job 709235, UNGATED), and topology_transfer_v1 is unblocked (its
serving port is a 3-module rename that is silently wrong under mp_residual=False). Read
HANDOVER.md §1 to gate the retrain and check the local a4_wl200200 finisher, and §2 for the
unmerged fix/deferred-gate-fixes branch that must land once nothing is mid-run.
```
