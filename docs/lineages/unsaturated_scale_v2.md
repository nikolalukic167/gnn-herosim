# unsaturated_scale_v2 — the same question as v1, at the power v1 lacked

**Status:** `REGISTERED` (2026-09-19). Bars, reader and 24 tests
(`scripts_cosim/unsaturated_scale_v2_read.py`, `scripts_cosim/test_unsaturated_scale_v2_read.py`)
committed before any arm ran. Three live steps (rule 6): **mint**, **M0** the saturation screen,
**L** the study.

**Why this exists.** `unsaturated_scale_v1` closed
`NO-LEARNED-ARM-BEATS-REACTIVE-AT-UNSATURATED-SCALE` with `gnnedge0` at **−6.46 % (p = 0.47)**
and `peeronly` at **−7.97 % (p = 0.12, 13/16 ahead, sign-test p = 0.021)**. Those are effects
the right size to matter read at a power that cannot resolve them, and v1's own post-hoc
decomposition says exactly why:

- the checkpoint statistic's sd scales as **exactly 1/√(environments)** for every arm — 36.4 →
  18.1 pp for `gnnedge0` from 1 cell to 4 — so **all** of it is (checkpoint × environment)
  interaction and **none** is a stable "this checkpoint is good" component. Four environments
  give 14–29 pp against a 5–8 pp effect;
- the simulator is **deterministic**: re-running one (environment, checkpoint) is bit-identical,
  so 50,000 tasks carries **zero** sampling noise. Task count is not a lever, and a longer trace
  cannot tighten any error bar — it only changes which load regimes are averaged;
- between the two halves of the single arrival window every gate in this programme has used,
  the arms' median against reactive **swings 19 pp and flips sign**.

So v2 changes the **replication unit** and nothing else. An **environment** is a
(topology, arrival window) pair; the study runs **4 × 4 = 16** of them, fully crossed, against
v1's 4. Same rung, same checkpoints, same bars, same 16 checkpoints per arm. Only
`n_environments` moves — which is what makes v2 a power replication of v1 rather than a new
question.

**Parents.** `unsaturated_scale_v1` (the rung, the bars, the defect), `peer_only_v1` (the cells
and the saturation bar), `bipartite_aggr_v1` (L3's mechanism), `corpus_matched_v1` (L2's matched
twin), `cluster_scale_v1` (9.58× out of candidate support at 80 servers, at any load).

**Entry points.** Bars/reader: `scripts_cosim/unsaturated_scale_v2_read.py` (+ tests).
Mint: `scripts_cosim/datalab/unsaturated_scale_v2_mint.sbatch`.
Baseline smoke: `scripts_cosim/datalab/unsaturated_scale_v2_baseline_smoke.sbatch`.
M0: `scripts_cosim/datalab/unsaturated_scale_v2_screen.sbatch` → `results/us_v2_screen/`.

## Registration — 2026-09-19

### Two measurements that changed the design before it was registered

**1. `knative_network_batch` is NOT a batching control.** v1's finding was that the entire
margin is a **6.82 s batch-assembly wait** the learned arms pay and reactive does not, so
"reactive with that wait" looked like the control that separates *the model is better* from
*batching is worse*. Smoke-tested first (job 789846): it pays **0.002 s**, and reads 23.269 s
against `knative_network`'s 23.27 s — a near-duplicate of reactive, not a batching arm. It is
carried as a second reactive variant and labelled as one, and a test pins it out of `M_ARMS`.
**The batching tax therefore still has no control**, and building one is not in this scope.

**2. Window 0 is 1.6× slower than the rest of the trace.** At a common rescale factor the four
windows deliver **0.460 / 0.747 / 0.753 / 0.757 arrivals/s** — the one window every gate in this
programme has ever used is the slow ramp-in of the production trace. Holding the *factor* fixed
would have made the window axis a **load** axis in disguise, and at 0.75/s reactive on 80
servers sits between its 0.66 queue share at 0.46/s and its 0.955 at 0.92/s. Each window
therefore carries the factor matching w0's realised rate; all four now deliver 50,000 tasks at
**0.4604/s over 30.17 h**, asserted to 0.5 %, and the only moving part is the arrival **pattern**.

**Also recorded:** the smoke test put `random_network` at **104.9 s against reactive's 23.3 s**
(queue share 0.928) — a 4.5× floor that establishes the placement problem on this rung is not
trivial, which no previous gate in this programme had shown.

### The environments

12 candidate topologies at 80 servers (`9001, 9002, 9003, 9005` from v1 plus `9101–9108`, newly
minted; **9004 is excluded — it hangs on every policy**) × 4 arrival windows (`w0` = events
[0, 50k) of the 450,729-event production trace, the one every gate has used; `w1`/`w2`/`w3` =
[100k, 150k), [200k, 250k), [300k, 350k)). Windows are real, disjoint arrival data, not a
resampling, enabled by `--skip-events` on `rescale_workload_arrivals.py` (byte-identical at
skip 0, pinned by `tests/test_rescale_window_offset.py`).

### M0 — the saturation screen (blocking; 48 reactive arms, ~40 s each)

Reactive `knative_network` on all 48 candidate environments. An environment is **admissible**
iff reactive completes 50,000 tasks and its queue share is **≤ `M_TARGET_SHARE` = 0.80** (v1's
bar, unchanged). A missing or `None` share is **inadmissible** — unknown is not a pass.

**The selection rule, signed:** the **4 lowest-numbered** topology seeds admissible on **all
four** windows. Lowest-numbered because the seeds are arbitrary generator labels, so the
ordering cannot be steered by anything any arm does; all four windows because a topology
admissible on three would make the environment set ragged, and the whole point of the
checkpoint statistic is that two checkpoints are never compared on different environments.
Fewer than 4 qualify ⇒ **`TOO-FEW-UNSATURATED-ENVIRONMENTS`**, the study does not run, and that
is the finding — it would be about the baseline, as v1's was.

### L — the study (5 learned arms × 16 checkpoints × 16 environments + 3 baselines × 16 ≈ 1,328 arms)

Arms, all existing checkpoints, **no training**: `be1670_gnnedge0`, `1670_peeronly`,
`1670_mpoff`, `516_mpoff`, `1670_gnn`. Baselines: `knative_network` (the reference, from M0),
`knative_network_batch`, `knative_network_ect`, `random_network`.

**The statistic changes, and this is the one deliberate departure from v1.** v1 took the median
of raw elapsed across cells and divided by reactive's median — a ratio of medians, which does
not pair. Reactive's own elapsed ranges 20–46 s across environments, so the pairing carries most
of the leverage: `checkpoint_stats` pairs each checkpoint against reactive **on its own
environment** and takes the median of those percentages. Arm-vs-arm pairs raw elapsed within
environment *and* checkpoint (`pair_checkpoint_stats`) rather than differencing two
already-relative numbers — whose denominator is near zero exactly when an arm ties reactive,
which is what every v1 arm did.

- **L1** each arm vs reactive: one-sample signed-rank against zero, |median| ≥ 5 %, p < 0.05,
  n = 16 checkpoints.
- **L2** `gnnedge0` vs `mpoff_1670` — model class at matched corpus.
- **L3** `gnn` vs `gnnedge0` — sum vs mean. v1 read **+26.69 % (p = 0.039)**; this is a
  replication at power of a bar that already fired.
- **L4** the composite. Fires positive if **any** registered arm clears L1.
- **L5 — the design-validation bar.** Worst-arm sd of the checkpoint statistic ≤
  **`M_DESIGN_SD_BAR` = 10 pp** (v1's decomposition projects 7–9). Signed now so that a
  disappointing sd cannot be re-described afterwards. **If L5 fails, an L4 negative is reported
  `UNINTERPRETABLE`, never as a tie** — repeating v1's mistake at a larger n would be worse, not
  better. An L4 *positive* stands either way: a bar that fires, fires.
- **L6** the window axis: does each arm keep its sign across the four windows? Descriptive by
  registration — it qualifies L1, never overrides it. v1's split-half is the reason it exists.

**Two bugs the tests caught before any datum existed**, both of which would have produced
confident wrong numbers: `read_l1` divided by zero (the statistic is already a percentage, so it
needs a one-sample test, not a two-sample one against a constant-zero arm), and arm-vs-arm was a
ratio of two relative numbers. This is what shipping bars with tests is for.

**Registered expectation (signed, and v1's was wrong).** L5 delivers (sd 7–9 pp). `peeronly`
and `gnnedge0` clear L1 as `ARM-BEATS-REACTIVE`; `gnn` reads `REACTIVE-FASTER`; L3 replicates
`SUM-COSTS`; L4 fires positive. Reasoning: both effects sat at −6.5 % and −8.0 % with 9/16 and
13/16 ahead, and halving the sd is exactly what turns those into separations **if the effects
are real**. The alternative — which v1's own bimodality supports — is that the effect is not a
location shift at all but a mixture, half the checkpoints winning 6–13 % and half losing
14–130 %, in which case L1 stays `NOT-SEPARATED` at any n and L6 shows the window driving it.
**v1's registered expectation was recorded wrong; this one may be too.**

**Cost.** Mint ~10 s; M0 48 arms; study ~1,328 arms at ~3 min in blocks of 48, ~2 h wall.

**Explicitly not in scope.** Retraining anything; a control for the 6.82 s batching tax (no such
arm exists — see above); training data at scale; the client axis; rungs other than 80 servers.
