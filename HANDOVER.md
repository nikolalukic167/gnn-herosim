# 🚀 Session Handover (2026-08-21, evening)

**Status:** The siv1 live-gate mystery is RESOLVED and written into `LINEAGES.md`; the
formal synced-code re-gate is running on datalab; one local sweep (`workload-200-200`) is
still finishing. Two deferred code fixes are queued for when everything drains.

> Read first: `LINEAGES.md` → `siv1_full_corpus` → "Resolution (2026-08-21, later the same
> day)" — the probe matrix that closed the case — and the `link_contention_v1` full-scale
> result right below its A/B design section. Memory:
> `herosim-gnn-results-not-portable-across-envs.md` (RESOLVED) and
> `herosim-siv1-live-gate-failed.md` (RE-GRADED).
>
> An earlier HANDOVER revision (17:05 today) framed the control failure as an
> ~30-point *environment* effect ("F10/F11"). That framing is **superseded**: the probe
> matrix proved the split is the uncommitted dims 9-11 code fix, and environment
> contributes ≤0.3%. Its noise-floor correction survives: the GNN's run-to-run floor is
> 0.1–0.4%, not the 0.05% measured on knative_network.

## 0. The one-paragraph story

The recorded siv1 live-gate FAIL (GNN 0/5 vs Knative, job 708549) was measured on
datalab's committed tree, which lacked the **uncommitted** dims 9-11 live-feature fix
(`src/placement/temporal_features.py` + `feature_builder.py` hunk, in the local working
tree since 2026-08-19). A 7-run probe matrix on `gnn/cell01` proved it: committed-tree
runs cluster at 65.80–65.87M across datalab-GPU / datalab-CPU / local-CPU (0.11% spread);
working-tree runs cluster at 50.36–50.52M (0.32%); 23.3% apart, split exactly on code
version. Environment/GPU/library differences all exonerated (each ≤0.3%, nothing
cascades; checkpoint sha256 identical both sides). With train-consistent features the
checkpoint goes 2W/1T/2L on `workload-125-225`, **wins 5/5 on `workload-150-100` AND
`workload-175-100`**, and wins 5/5 under a binding-bandwidth backbone with margins that
*widen* (mean −9.4% → −24.0%). The working tree has been committed and pushed; datalab
pulled it; the formal re-gate was submitted (see §1).

## 1. FIRST THING TO DO: check the two in-flight runs

**(a) Datalab formal re-gate** (synced code, 15 array tasks = 3 policies × 5 cells, fresh
sweep dir `simulation_data/normal_sim_sweeps/full_corpus_siv1_live_gate_20260821_synced`):

```bash
ssh datalab 'squeue -u nikola.lukic; cd /home/nikola.lukic/gnn-herosim && \
  for f in simulation_data/normal_sim_sweeps/full_corpus_siv1_live_gate_20260821_synced/results/*.json; do \
    echo "$f $(grep -o "\"total_rtt\": [^,}]*" "$f" | head -1)"; done'
```

Expected if everything is right: knative bit-identical to the recorded gate
(cell01 = 46,556,946.73649), GNN near the local working-tree values
(cell01 ≈ 50.4M ± ~0.1%, i.e. 2W/1T/2L overall). Score with
`compare_sealed_live_holdout.py --sweep-dir <sweep>`. **Then update the LINEAGES siv1
resolution subsection with the re-gate outcome — it currently notes the re-run as the
remaining step.** Job id: see the sbatch submission echo in the session log /
`logs/siv1-gate-*` on datalab dated 2026-08-21 evening.

**(b) Local `workload-200-200`** (800k events, last phase of the run_rest2.sh queue,
PAR=2, takes hours):

```bash
ls simulation_data/normal_sim_sweeps/a4_wl200200/results/*_s0_*.json | grep -v decode | wc -l  # 15 = done
pgrep -af run_rest2.sh || echo "runner gone"
```

Its result extends the trace ladder but cannot change the re-grading — that rests on the
matched-code probe matrix.

## 2. AFTER both finish — the two deferred code fixes (in this order)

1. **F1 residual tie-break fix** — in all three of
   `src/policy/{gnn,knative,knative_network}/autoscaler.py`, the scale-down sort becomes
   `key=lambda couple: (len(couple[1].queue.items), couple[0].id, couple[1].id)`.
   Rationale: GATE TOOLS 2026-08-21 row;
   `scripts_cosim/test_autoscaler_scaledown_determinism.py` exists to guard it. Removes
   the residual 0.05–0.1% run-to-run noise (`mlp_batch` inherits the gnn autoscaler).
2. **`run_provenance` git stamp** — record `git describe --dirty --always` + a hash of
   `git diff` in every live result JSON (`executesimulation.py`), so a gate can never
   silently measure a code diff again. See the GATE TOOLS 2026-08-21 protocol row.

Both were deferred all session to avoid splitting in-flight sweeps across code versions.
Commit + push them, and pull on datalab, before any further gate.

## 3. What's now established (safe to build on)

- **Deployed checkpoint** (`near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt`, sha256
  `4df64b6a…`, identical local↔datalab), served train-consistent features, on
  parity-verified cells (margins vs knative, − = GNN better):

  | trace | GNN vs Knative per cell (01..05) | verdict |
  |---|---|---|
  | workload-125-225 | +8.3 / −5.3 / +17.4 / −7.6 / −0.06 % | 2W/1T/2L |
  | workload-150-100 | −0.9 / −12.9 / −14.8 / −5.3 / −12.9 % | 5/5 W |
  | workload-175-100 | −9.4 / −7.6 / −7.5 / −10.8 / −10.8 % | 5/5 W |
  | workload-150-100 + backbone 1.5MB/s | −31.8 / −24.4 / −26.7 / −8.9 / −28.0 % | 5/5 W |

- **Noise floors**: GNN local run-to-run 0.1–0.4% (F1 residual + thread wobble); datalab
  GPU run-to-run ±0.04%; GPU↔CPU within datalab ±0.03%. All margins above are ≥25× these.
- **MLP** beats Knative on more cells by raw margin but has a catastrophic tail — one cell
  per trace at +147% to +366% (cell05 on both dur=100 traces; different cells on
  125-225). The GNN has no such tail in six sweeps. This is currently the strongest
  argument for the GNN over the MLP live, ahead of any mean-margin claim.
- **Backbone at binding bandwidth is a 7–14× absolute-cost effect** at real concurrency,
  and the GNN's margin widens under it (`link_contention_v1` full-scale A/B in LINEAGES).
  The co-sim FALSIFIED verdict stands for its own statistic.

## 4. Open threads, prioritized

1. Finish §1 (re-gate + wl200200), update LINEAGES verdicts.
2. Apply §2 fixes.
3. **The real remaining GNN problem**: `logit_tied_rate ≈ 0.54` /
   `confident_worse_queue_rate ≈ 0.8` — the model has no sharp ranking on half its
   decisions (root cause: the additive, queue-dominated co-sim target — see
   `graph_structure_physics`). The live wins above came *despite* this. Candidate next
   lineage: retrain on the corrected cache (the temporal fix changed 18.7% of platform
   rows in `shallow_v1`'s cache; **the full-corpus siv1 cache was never rebuilt
   post-fix**), and/or backbone-corpus training to exploit the regime where the GNN's
   live edge is largest.
4. MLP catastrophic-tail root cause (`decode_stats` sidecars: p95 `chosen_queue_vs_min`
   ~30× normal on the collapsing cell) — real, separate, never investigated.
5. topology_transfer_v1 partial retest still blocked on ~14 GPU-hours on datalab (plan
   file: `/root/.claude/plans/read-lineages-md-fully-first-spicy-bee.md`).

## 5. Environment gotchas (unchanged)

```bash
PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=/root/projects/my-herosim \
  pipenv run python3 scripts_cosim/<script>.py ...
# datalab: eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
# pin OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_NUM_THREADS=4 for ML runs
# ssh alias `datalab` works; the raw hostname with BatchMode=yes does not (key is agent-loaded)
```

## 6. Restore prompt for next session

```
[CONTEXT RESTORE] The siv1 live-gate FAIL was re-graded 2026-08-21: it measured an
uncommitted dims 9-11 live-feature fix (temporal_features.py), not the checkpoint —
proven by a 7-run probe matrix (LINEAGES.md siv1 resolution subsection). With
train-consistent features the GNN beats Knative 5/5 on workload-150-100 and 175-100,
2W/1T/2L on 125-225, and 5/5 under a binding backbone with widening margins. The working
tree was committed+pushed, datalab pulled it, and a formal synced-code re-gate was
submitted (sweep dir full_corpus_siv1_live_gate_20260821_synced). Read HANDOVER.md §1 to
check the re-gate and the local workload-200-200 finisher, §2 for the two deferred code
fixes (F1 tie-break total key + run_provenance git stamp) to apply once nothing is
mid-run.
```
