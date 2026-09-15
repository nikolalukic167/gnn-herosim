# program_verdict_v1 — CLOSED

> **Status:** `CLOSED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-24 → 2026-08-24

**Outcome.** Terminal answer to the D3 fork: **the supervised co-sim path to "GNN > MLP on latency" is closed by measurement.** The reliability/regime win exists on the 30-cell record but is exploratory. P1 (closed-loop objective) is the only remaining path to the latency claim.

**Entry points:** this file (2026-08-24 subsection); artifacts cited in place

**Datasets:** P7 frozen reports (scratchpad)

**Related:** [cosim_deepdive_v1](cosim_deepdive_v1.md) · [p5b_candidate_relative](p5b_candidate_relative.md) · [graph_structure_physics](graph_structure_physics.md)

## Standing (from the index table)

**Closed 2026-08-24.** Terminal answer to the D3 fork: the supervised co-sim path to "GNN > MLP on latency" is closed by measurement (5 mechanisms + live-state additivity + the P7 warmth-stratum controls, which take the least-additive 31% of the cache to spread-plans R² = 1.00000 exactly). The reliability/regime win over both baselines exists on the 30-cell backbone record but is exploratory — it needs one pre-registered gate. P2 ruled out (labeller is the one-step oracle); P4 ruled out on the empirical rule (the built slot holds exec only; the residency-hold variant is unbuilt but node-indexed); P3 (in-horizon dynamics, tail-sensitive pre-registration) is the highest-upside open measurement; P1 (closed-loop objective) the only path to the latency claim. **Outcomes below.**

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [program_verdict_v1 — can the co-sim → GNN program ever work? (2026-08-24)](#program-verdict-v1-can-the-co-sim-gnn-program-ever-work-2026-08-24)

---

### program_verdict_v1 — can the co-sim → GNN program ever work? (2026-08-24)

**Closed. Read-only investigation (no new sims, no retrains) answering the generalised D3
fork: is there any path by which a GNN trained here beats MLP and Knative on a live gate.**

**Verdict, in two halves.**
1. **On the win condition the evidence already supports — regime win + reliability — the
   program has already worked, but the evidence is post-hoc, not gate-grade.** `tempfix`
   beats Knative on 30/30 backbone cells across 3 training draws, 2 traces, 3 backbone
   configs (jobs 710315/710335/710341/710366/710398), and beats the MLP's *aggregate*
   margin in 5 of 6 gate conditions because 7/30 MLP cells collapse (jobs
   710450/710451/710656/710657); 0 of 120 GNN runs ever collapsed. **Provenance caveat,
   verified 2026-08-24:** those cells were minted for the `link_contention_v1` real-trace
   A/B, the "< −0.4%" win rule appears at scoring time, and no pre-registration language
   exists anywhere in that campaign's span — each follow-up run was motivated by the
   previous result. Internally well-controlled (same-batch Knative baselines, parity-exact
   cells, measured noise floors), but a reviewer will correctly call it exploratory. The
   claim to publish — after one pre-registered confirmatory gate (below).
2. **On per-cell mean latency vs the MLP, through any single-batch supervised co-sim
   target, the program is terminally closed** — five physics mechanisms
   (`graph_structure_physics` → `link_contention_v1`) plus the live state distribution
   (`cosim_deepdive_v1`, incl. all 14 collapse trajectories) all measure the target as
   pointwise-separable. No corpus design can reopen this; only a change of objective can.

**Path verdicts** (mechanism-level; citations in the session record):
- **P1 closed-loop RL/DAgger — EXPENSIVE-BUT-VIABLE.** A rollout episode exists today
  (`executesimulation.py` + `run_simulation.py`); wall-clock **measured** from
  `logs_sim/`: GNN 754.9 s on the 301k-event `workload-150-100`, 1001.8 s on `175-100`
  (~2.9 GB RSS/worker). ~10³ episodes ≈ 5K CPU-h at gate scale (~500 on the 30k trace)
  plus a 1–2 week build; prior 0.15–0.25 of beating the MLP's healthy-cell packing margin.
  The only path whose objective matches the known live edge (closed-loop dispersal).
- **P2 live-snapshot labels — RULED OUT.** `label_live_snapshots_for_training.py` imports
  `oracle_choice_cosim` — it *is* the one-step oracle `cosim_deepdive_v1` already tested
  on exactly those states. Any non-one-step labelling is P1 or P3 by definition. Flip
  condition: exhibit a single-state labelling that is neither one-step RTT nor a
  horizon/trajectory return and encodes dispersal value — none is known.
- **P3 dynamics inside the oracle horizon — VIABLE, the highest-upside measurement.**
  The two-line scaling test does **not** kill it: node-mediated interaction under arrivals
  is occupancy-count-shaped (empirical rule, 5 mechanisms), but the *link* term is the one
  mechanism that escaped the count control, and it scales with concurrency (0.08–0.35%
  regret at 4-task vs 7–14× cost and ordering changes at rps=150). Pre-registered pilot
  (thresholds fixed 2026-08-24, before any build): extend `live_snapshot_cosim_oracle` to
  continue trace arrivals for a ~10 s horizon on backbone cells; M4 unmodified. Because
  the hypothesised mechanism is link-shaped and link effects surfaced at t=0 as a **3.3%
  tail**, a median-only criterion cannot resolve its own hypothesis — co-primaries:
  (a) median `additive_choice_regret_rel` > 0.02, **or** (b) fraction of snapshots with
  regret > 2% ≥ 15% (≥ 4× the 3.3% t=0 base rate, binomial-testable at the stated n) —
  either fires only with node-count repair < 0.5 AND link repair < 0.5 **on the affected
  stratum**. n ≥ 300 snapshots (≥ 2 backbone cells, K=4 ≈ 256 combos), so the tail holds
  ≥ 10 states under H0 and ~45 under H1. **Cost calibrated 2026-08-24, then corrected the
  same day when the trace-rate naming was measured** (see the propagation note below):
  `workload-150-100` runs ~3,000 events/s at steady state (t≈20–100 s; "rps=150" is
  *per client node*, ×20 clients) with ~20 s ramps at both ends, and the first
  calibration slice (first 10 s, 4,609 events) sat in the ramp. Measured floor: one
  `knative_network_batch` episode on that slice is 7.3 s wall of which 5.1 s is process
  startup ⇒ ~0.48 ms marginal per event ⇒ **~1.4 s per horizon-second per combo at
  steady state**: h=10 s ⇒ ~14 s/combo (~300 CPU-h at 300×256), h=5 s ⇒ ~7 s
  (~154 CPU-h), h=3 s ⇒ ~4.3 s (~92 CPU-h), h=2 s ⇒ ~2.9 s (~62 CPU-h) — before
  backbone-fabric overhead (assumed 1.5–2×). **The horizon length is therefore a
  registered design parameter, not a default**: the in-harness calibration (3 snapshots,
  h ∈ {2, 5, 10}) must (a) confirm combos genuinely run in-process — if each combo pays
  the 5.1 s startup the budget quadruples and the pilot is re-scoped before queueing —
  and (b) fix h at the largest value the stated budget affords, registered before the
  array. Honesty cost of a short horizon, stated up front: queue-runaway ignition may
  need sustained load, so a null at h ≤ 3 s is terminal only for *short-horizon*
  dynamics; terminality for the axis as a whole requires h ≥ 5 s. ~1–2 days build.
  Prior 0.2–0.35.
- **P4 held-duration node contention — RULED OUT on the empirical rule, corrected
  2026-08-24 (the first write-up of this entry overstated it).** What exists holds the
  node slot only around exec (`infrastructure.py:1213-1218` wraps
  `yield timeout(task_duration)` alone; exec ≈ 0.024 s), and `memory/memory.md:79`
  correctly recorded in 2026-08-17 both why that measured `nodeContentionTime ≡ 0.0`
  (backlogs drain in one timeout; placed tasks never overlap) and the unbuilt candidates
  that would couple: a hold across the whole residency (cold starts reach 38 s) or a warm
  lifetime. So the residency-hold variant is *not implemented here*, not already
  falsified. It stays ruled out because its contended object is still node-indexed ⇒ the
  interaction is a co-residency count ⇒ the throughline predicts one-integer degeneracy —
  an invocation of the empirical rule, not a measurement. Flip condition unchanged: a
  slot-contention config with additive-argmin regret > 5% and node-occupancy repair < 50%
  under `--spread-plans-only`.
- **P5a reliability win condition — VIABLE, needs one pre-registered gate** (see the
  provenance caveat in half 1: the 30/30 record is exploratory). The gate must
  co-register the **MLP arm**, not just Knative — the paper's claim is collapse-freedom
  *relative to the MLP*, and the 14/120 collapse count comes from the same exploratory
  campaign as the 30/30. Fresh cells must be minted by a **new** script that does not
  inherit the A/B design (`make_backbone_gate_cells.py`'s cells are the ones the
  "< −0.4%" rule was written against). Remaining: register win condition + thresholds
  (incl. the collapse detector, `chosen_queue_vs_min` p95 with the measured 9.7×
  no-overlap gap) *before* running `tempfix` + MLP + Knative on
  `workload-125-225`/`200-200` and the fresh cells, then promote.
- **P5b batch conditioning — claim must be re-worded before publication.** The deployed
  gates run an *identical* decode for GNN and MLP arms (`mlp_scheduler.py:5-7` inherits it;
  in `argmax` mode `chosen_idx = gnn_idx` unconditionally, `seq_decode.py:719-728` — the
  queue roll-forward feeds stats only). The separation is therefore *score-side
  set-conditioning* (message passing sees the candidate context; a pointwise edge score
  cannot), *not* decode-time peer conditioning. Required control before the paper: MLP +
  candidate-relative queue feature (rank/z-score), retrain + 30-cell re-gate (~1–2 days).
  If that MLP stops collapsing, the honest claim shrinks to feature engineering.
- **P5c topology transfer — stays FAILED for the supervised objective** (its own 5-seed
  gate; note that gate scored the additive target, so the FAIL is *predicted by*
  additivity). Reopening evidence: a live *reliability* gate across sizes (~14 GPU-h
  partial gate, already unblocked). Only worth it as an extension of P5a's claim.
- **P6 freeze — the recommended frame.** Publish (1) the terminal negative — single-batch
  placement targets in this simulator class are pointwise-separable, with
  `separability_diagnostic.py` + the one-integer/link repair controls as the reusable
  artifact — and (2) the reliability result (half 1). Residual reviewer risks, named: one
  topology family (20c/20s); two traces not yet re-gated under the corrected layout; the
  P5b control unrun (the largest); backbone physics authored, not trace-calibrated;
  "GNN > MLP on latency" must not be written.

- **P7 — the least-additive stratum, measured (2026-08-24): the terminal statement holds
  unqualified.** `cosim_deepdive_v1`'s census left `warmth_1060`/`sparse_warmth` (31.1%
  of the training cache, additive-argmin-optimal only 51.7%/56.1%) as the one stratum
  where the one-integer and `--spread-plans-only` controls had never run. Pre-registered
  (STRENGTHEN iff median repair ≥ 0.5 or spread-plans R² ≥ 0.999 with ~zero spread
  regret; QUALIFY iff repair < 0.5 and spread regret > 1%), then run at n=150 per
  collection: `1060_warmth` regret 2.01% → 1.15% under the node-count column (repairs 51%,
  n=67 with any regret, `!! DEGENERATE`); `sparse_warmth` 2.12% → 0.49% (repairs 68%,
  n=64, `!! DEGENERATE`). Decisively, **`--spread-plans-only` takes both collections to
  additive R² = 1.00000 exactly, 0.00% regret, 100% optimal** (sparse: 142/150 fitted,
  8 underdetermined by too few spread plans) — identical to the base-physics isolation
  result. The warmth stratum's non-additivity is entirely the collision term. Frozen
  reports: `simulation_data/separability_{warmth_1060,sparse_warmth}_n150{,_spread}.json`.
  **The spread control itself was then audited for mechanical saturation** (R² = 1.00000
  exactly is also the signature of params ≈ observations, and the control is load-bearing
  for the throughline's "no reservoir" sentence too). Pre-registered (suspect if median
  rows/params < 2; survives iff median held-out R² ≥ 0.999 on a seed-0 half split), then
  measured with `scripts_cosim/audit_spread_fit_saturation.py`: `warmth_1060` ratio
  median 97.3, held-out R² = 1.00000 on 150/150; **`mh_off` (the throughline's base
  corpus) ratio median 7.8 (min 2.0), held-out R² = 1.00000 on 48/48**; `sparse_warmth`
  ratio median 14.7, held-out 1.00000 on 137/144, with 7 failures all at ratio ≤ 2.29 /
  ≤ 16 spread rows — unresolvable at their own n, excluded from the claim's basis rather
  than counted as counter-evidence. Held-out *exactness* is the strong form: overfitting
  cannot produce exact out-of-sample predictions, so the spread-plans conclusion — and
  the "no reservoir of non-count coupling" sentence it underwrites — now rests on
  out-of-sample evidence, not in-sample R². **The claim's basis is 137/144 sparse
  datasets plus 150/150 warmth_1060 plus 48/48 mh_off — not 144/144**; the 7 excluded
  sweeps are unresolved, full stop. The one-integer repair fractions (51%/68%, with 51%
  one point over the DEGENERATE threshold) are secondary; the spread control is the
  criterion this entry leans on.

- **Trace-rate naming propagation check (2026-08-24).** Measured on
  `workload-150-100.json`: the `rps` field is **per client node** — 20 sources ×
  ~15,067 events each, steady state ~3,000 events/s over t≈20–100 s, ~20 s ramps at both
  ends (ts max 118.7 s). Every claim framed "at rps=150" therefore describes an
  operating point whose *system* arrival rate is 20× the label — most prominently
  `link_contention_v1`'s real-trace A/B ("the effect is not small at rps=150", 7–14×
  Knative cost): the measured numbers are untouched, but the concurrency they occurred
  at is ~3,000 events/s, and any external-realism judgement ("is 150 rps a realistic
  load?") made against the label is off by that factor. The same naming corrupted this
  entry's first P3 cost estimate (calibrated on the ramp; corrected above). Paper text
  quoting an "rps" operating point must state the per-client convention and the system
  rate.

**Recommended sequence (reordered 2026-08-24 — downside protection before upside):**
P5b control first (1–2 d; if a candidate-relative queue feature stops the MLP collapsing,
every subsequent P5a gate would have been wasted) → P5a pre-registered gate + re-gates
(2–3 d CPU) → P3 pilot (2–3 d) → **if P3 nulls, the residency-hold scaling test on paper
and, if it demands one, its 16-dataset pilot (~0.5 d) before any P1 spend** — cold start
at 38 s vs 0.024 s exec is a ~1,500× longer hold, so mechanism #1's ≡ 0.0 says nothing
about it, and unlike link bandwidth (where additive `hops×T` and interaction
`crossings×T` scale together, ratio-invariant) a longer hold flips overlap from *never*
to *always* — a threshold, not a ratio. The two-line prediction: interaction =
`(co-residents − slots)⁺ × residency` — a node-occupancy count times a scalar — so the
**expected** outcome is a sixth confirmation of the empirical rule: coupling *with
teeth* that the one-integer column repairs. The pilot is still worth its 0.5 day
because it converts the weakest ruling in the set (a rule invocation) into a
measurement before any P1 spend. Pre-register **both** controls before generating:
the node-count repair column *and* `--spread-plans-only` — the spread control is the
comparison that would actually surprise (residual regret on all-distinct-node plans
under residency holds would be the first non-link escape), and registering only the
count column would leave the surprising outcome ungated → P1 only if P3 (or that
pilot) finds non-count signal (P3's horizon labels are also DAgger targets).

---
