# unsaturated_scale_v2 — the same question as v1, at the power v1 lacked

**Status:** `CLOSED` (2026-09-19) — **`EFFECTS-ARE-REAL-AND-AN-ORDER-OF-MAGNITUDE-SMALLER`.**
Closed on a live gate (rule 6): 1,328 arms, 0 failures. Registered 2026-09-19; bars, reader and
24 tests committed before any arm ran. **The registered expectation was wrong again** — and
wrong in a way neither registered alternative named.

**Outcome. Every effect v1 measured is real in SIGN and 4–10× SMALLER in SIZE.** The design
delivered far more power than it was built for (**L5 `POWER-DELIVERED`**: worst-arm sd
**4.48 pp** against the registered 10 pp bar, v1's 14–29 pp at 4 environments), and what that
power showed is not sharper separations but collapsed magnitudes with tiny p-values:

| read | v1 (4 environments, all on w0) | v2 (16 environments) |
|---|---|---|
| `gnnedge0` vs reactive | −6.46 % (p = 0.47) | **−1.62 % (p = 0.0023, 13/16)** |
| `peeronly` vs reactive | −7.97 % (p = 0.12) | **−1.23 % (p = 0.0013, 15/16)** |
| `gnn` vs reactive | +16.98 % (p = 0.044) | **+2.31 % (p = 0.0052, 3/16)** |
| `gnnedge0` vs matched twin (L2) | −6.10 % (p = 0.47) | **−2.38 % (p = 0.0004)** |
| `sum` vs `mean` (L3) | +26.69 % (p = 0.039) | **+2.70 % (p = 0.0019)** |

**So the arms genuinely are faster than reactive — by about 1.5 %.** `gnnedge0` is ahead on
13/16 checkpoints at p = 0.0023 and `peeronly` on 15/16 at p = 0.0013; these are not nulls, they
are real effects far under the programme's 5 % practical bar. Under the registered rule every
arm is **`NOT-SEPARATED`** and **L4 reads `NO-LEARNED-ARM-BEATS-REACTIVE-AT-UNSATURATED-SCALE`,
`interpretable = True`** — v1's verdict survives, but for a completely different reason. v1 could
not tell a 6 % effect from noise; v2 can tell a 1.6 % effect from noise, and it is 1.6 %.

**Where v1's magnitudes came from: the window.** L6, per-window median vs reactive —

| arm | w0 | w1 | w2 | w3 | |
|---|---|---|---|---|---|
| `gnnedge0` | **−6.40** | −1.06 | −0.65 | −0.18 | sign-consistent |
| `peeronly` | **−8.07** | −0.12 | −0.42 | **+0.87** | **sign flips** |
| `mpoff_516` | −2.21 | −0.28 | +0.28 | **+1.49** | **sign flips** |
| `gnn` | **+5.86** | +2.21 | +2.53 | +2.86 | sign-consistent |

v1's −6.46 % and −7.97 % are **w0 numbers**, reproduced here to within 0.1 pp. On the three
windows no gate in this programme had ever run, the same arms read −0.18 to +0.87 %. w0 is the
one arrival window every gate has used and it is the burstiest of the four (reactive queue share
0.53–0.66 on w0 against 0.41–0.48 on w1–w3 at an identical 0.4604 arrivals/s).

**The mechanism is unchanged and now fully quantified: the batching tax is the whole margin.**
Median batch-assembly wait, this rung: `gnnedge0` 7.125 s, `peeronly` 7.123 s, `mpoff` 7.131 s,
`gnn` 7.134 s — and **reactive 0.000 s, `knative_network_ect` 0.000 s, `random_network`
0.000 s**. Against a reactive elapsed of 19.148 s that tax is **37 % of the total**. The learned
arms win roughly that much queue back and end up 1–2 % ahead or behind depending on the window.

**Carry.** (a) These are 6-server checkpoints served 9.58× out of candidate support
(`cluster_scale_v1`); v2 closes "the existing checkpoints at 80 servers", not "a GNN trained at
scale". (b) `peeronly` and `mpoff_516` **flip sign between windows**, so neither L1 number may be
quoted without L6. (c) v2 measures the **80-server** rung only — it does not re-measure the
6-server rungs where the standing −20.8 % headline lives, but it does show that a 4-environment
read on this apparatus inflated every effect 4–10×, which is a caution that applies to those
numbers too. (d) The `sum` mechanism (`bipartite_aggr_v1`) **replicates in direction at a far
better p-value and collapses in magnitude**, +26.69 % → +2.70 %.

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
(queue share 0.928) — read at the time as a 4.5× floor establishing that the placement problem
on this rung is not trivial. **WITHDRAWN by the study below**: across all 16 environments
`random_network`'s median is **+8.4 %**, and the distribution is bimodal rather than a floor.
The 4.5× was one environment. Left here rather than edited away, because it is the same
one-cell-generalisation this lineage exists to fix, committed by the same session that was
fixing it.

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

## 2026-09-19 — L: **`EFFECTS-ARE-REAL-AND-AN-ORDER-OF-MAGNITUDE-SMALLER`**

Jobs 789909–791233, **1,328 arms, all COMPLETED, 0 failures**, every one at `num_tasks = 50000`.
Environments: `[9001, 9002, 9005, 9101] × [w0, w1, w2, w3]`, selected by the registered rule
from a 48-environment screen. Reactive median elapsed **19.148 s**, queue share 0.41–0.66.

**L5 first, because it licenses everything else.** Worst-arm sd of the checkpoint statistic
**4.48 pp** (`gnn`; the other four are 1.26–1.77 pp) against the registered **10.0 pp**. v1's
decomposition projected 7–9 pp and the design beat that. `POWER-DELIVERED`, so L4's negative is
**interpretable** — this is a measured null, not v1's absence of evidence.

**L1** — one value per checkpoint, the median over 16 environments of its relative % vs reactive
on that same environment; one-sample signed-rank; bar |median| ≥ 5 % and p < 0.05.

| arm | median | p | ahead | sd | verdict |
|---|---|---|---|---|---|
| `be1670_gnnedge0` | −1.62 % | 0.0023 | 13/16 | 1.26 | `NOT-SEPARATED` |
| `1670_peeronly` | −1.23 % | 0.0013 | 15/16 | 1.38 | `NOT-SEPARATED` |
| `516_mpoff` | −0.11 % | 0.6051 | 8/16 | 1.77 | `NOT-SEPARATED` |
| `1670_mpoff` | +1.86 % | 0.0016 | 2/16 | 1.75 | `NOT-SEPARATED` |
| `1670_gnn` | +2.31 % | 0.0052 | 3/16 | 4.48 | `NOT-SEPARATED` |

Every arm sits inside the ±5 % band at p ≤ 0.006 except `mpoff_516`, which is a genuine null
(p = 0.61, 8/16 — a coin flip). **L2** `gnnedge0` vs `mpoff_1670`: −2.38 %, p = 0.0004 —
the model-class edge at matched corpus is real and is 2.4 %. **L3** `gnn` vs `gnnedge0`:
+2.70 %, p = 0.0019 — `SUM-NOT-SEPARATED` under the bar, direction replicated. **L4**
`NO-LEARNED-ARM-BEATS-REACTIVE-AT-UNSATURATED-SCALE`, winners none, **losers none**, nothing
missing, interpretable.

**Baselines** (median elapsed over the 16 environments): reactive **19.148 s**;
`knative_network_batch` 19.153 s (**+0.0 %** — a near-exact duplicate of reactive, as the smoke
test predicted, and the reason it is not a batching control); `knative_network_ect` 20.877 s
(**+9.0 %**, the physics-aware greedy loses); `random_network` 20.094 s (**+8.4 % median**).

**A correction that belongs in the record.** The baseline smoke test put `random_network` at
104.9 s against reactive's 23.3 s on `cs80s9001/w0`, and that single cell was quoted as "a 4.5×
floor showing the placement problem is non-trivial". Across all 16 environments the median is
**+8.4 %**, and the distribution is bimodal, not a floor: +4 to +8 % on nine environments,
+29 % on three, then +46 / +147 / +285 / **+782 %** (the worst at `(9002, w0)`). Random placement
is usually nearly free at this rung and occasionally catastrophic. The one-cell number was
wrong as a characterisation and is withdrawn.

**The registered expectation, and how it failed.** It said L5 would deliver (it did, better than
projected) and that `peeronly` and `gnnedge0` would then clear L1 as wins. They did not. The
registered alternative — a mixture rather than a location shift, staying `NOT-SEPARATED` at any
n with L6 showing the window driving it — got the *outcome* right and the *mechanism* wrong:
the per-checkpoint distributions are tight (sd 1.26–1.38 pp), so it is a location shift after
all, just a ~1.5 % one. What the window drove was not the checkpoint spread but the **size of
the effect itself**, which is v1's environments all having been w0.
