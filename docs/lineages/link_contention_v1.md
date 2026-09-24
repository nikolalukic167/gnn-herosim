# link_contention_v1 — FALSIFIED

> **Status:** `FALSIFIED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-18 → 2026-08-21

**Outcome.** Per-link capacity over a multi-hop backbone. Gate FAILED on both criteria. **The link controls do not repair (median 0.000)** — genuinely new, and the first escape from the one-integer control — but node-collision coupling still dominates and the magnitude is 14× below the gate.

**Entry points:** `src/placement/network_fabric.py`, `src/generate_infrastructure.py` (`build_core_backbone`), `scripts_cosim/{link_overlap_precheck,test_link_contention,test_link_repair_control}.py`, grid `netc_multihop_v1`

**Datasets:** `netc_multihop_v1_mh_{off,bw1p5}` (local, n=48 each)

**Related:** [throughline](throughline.md) · [network_contention_v1](network_contention_v1.md) · [route_c_link_transfer_v1](route_c_link_transfer_v1.md)

## Standing (from the index table)

Per-link capacity over a multi-hop core backbone, opt-in via `--link-bandwidth-mbps`. **`FALSIFIED` 2026-08-18 — gate FAILED on both criteria.** The link controls do *not* repair (median 0.000), which is a genuinely new signature, but node-collision coupling still dominates (node repair median 1.000). **Outcomes below.**

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [link_contention_v1 — a real-trace A/B, at realistic concurrency (2026-08-21, 🔄 IN PROGRESS)](#link-contention-v1-a-real-trace-a-b-at-realistic-concurrency-2026-08-21-in-progress)
- [link_contention_v1 — outcomes (2026-08-18)](#link-contention-v1-outcomes-2026-08-18)

---

### link_contention_v1 — outcomes (2026-08-18)

**The hypothesis.** Every mechanism above was repaired by a *node-occupancy excess* column
("how many tasks landed on host X"), because in every case the contended object was indexed
by the destination node. A network **link** is crossed by paths to many destinations, so two
tasks on different nodes can queue behind each other — a coupling no node-occupancy count can
express at any value. That is the one structural escape none of the previous four had.

**Two facts had to be fixed before the idea could even be tested.**

1. **There were no multi-hop paths.** `generate_network_topology_deterministic` only ever
   wrote client↔server pairs — 0 server↔server and 0 client↔client edges, density ~0.25,
   hop count always exactly 1. Latency was a single table lookup, so "the link" a task
   crossed was its own private access edge.
2. **The 4 tasks always come from 4 distinct clients** (`random.Random(42)`; all 10 workload
   templates), so capacity on a client↔server edge is paid by at most one task and is
   additive by construction. The contended object had to be a *shared core segment*.

**Physics.** `src/placement/network_fabric.py` holds one `simpy.Resource(capacity=1)` per
link plus the frozen route table, built once in `simulation.py` because a link belongs to no
node. Propagation stays the un-serialized `env.timeout` (additive, unchanged); the input
transmission walks the route store-and-forward, holding each hop. `build_core_backbone` is a
post-processing overlay applied *after* every connectivity repair, so `network_maps` keeps
its meaning (candidate filtering still works, all six copies of the lookup untouched) and
only its latency becomes a path sum. Absent a `network.backbone` block nothing is built and
replay is bit-identical. 23 tests in `test_link_contention.py`, 9 in
`test_link_repair_control.py`.

**P0 — the overlap pre-check `PASSED` decisively.** `scripts_cosim/link_overlap_precheck.py`
measures route overlap with **no simulation**, extending the Hall's-condition instinct that
killed `netc_scarce_v1`/`netc_funnel_v1` before they cost a corpus. Swept on `shallow_v1`:

| backbone | core links | pairs sharing a core link | ...of those, DIFFERENT destinations |
|---|---:|---:|---:|
| n_core 6, attach 2, chords 3 | 9 | 5.2% | 77.8% |
| n_core 6, attach 1, chords 0 | 6 | 25.2% | 90.2% |
| **n_core 12, attach 1, chords 0** | **12** | **30.3%** | **91.3%** |

Chords and a second attachment both let paths diverge and collapse the overlap. A pure ring
with single attachment was chosen on this measurement, **not** on RTT.

**The gate `FAILED` on both pre-registered criteria.** Matched arms, same tree, same
`--workload-seed 42`, grid `netc_multihop_v1` (shallow queues + `per_client=0` so every task
must cross the network), n=48 each, thresholds registered before generating:
`--gate-additive-argmin-regret 0.05 --gate-one-integer-repair 0.5 --gate-link-repair 0.5`.

| arm | additive R² | argmin regret mean / max | node repair | link repair k1 / k2 / excess |
|---|---:|---:|---:|---:|
| `mh_off` (no backbone) | 0.91658 | 3.57% / 22.9% | **0.817** | — |
| `mh_bw1p5` (1.5 MB/s per link) | 0.91910 | **5.00%** / 36.6% | **0.633** | **0.149 / 0.195 / 0.178** |

**What is genuinely new, and worth keeping.** The *link* controls do not repair: medians are
**0.000** for all three (k1, k2, excess), with the means pulled up by a handful of datasets at
1.0. This is the first mechanism whose coupling is not a scalar summary of the contended
resource — contrast node-ingress, where one integer repaired 75%.

**Why it still fails.** The *node-collision* column repairs 63% (median **1.000**). The
coupling that dominates this corpus is still the pre-existing `added_in_batch` collision
effect, not the link effect; the backbone adds only a small, mixed increment on top of it.
Paired across the 48 matched datasets the backbone **raised** regret in 17, **lowered** it in
13, and left 18 unchanged — mean +1.43 pp, and the resulting 0.049961 lands a hair under the
0.05 threshold. A weak, two-directional effect riding on a dominant confound is not a corpus
worth training on.

**Do not** read this as "links don't couple" — the pre-check and the sweep both show real
cross-destination link contention (877 core-link-sharing task pairs on a single dataset, 92%
of them with different destinations). Read it as: **the link term is small next to the node
term already present**, so the corpus's coupling stays node-shaped and one integer still
repairs most of it.

#### The isolation control — and the actual size of the link effect

The mixed n=48 result could not separate the link term from the collision term that
dominates it, so `separability_diagnostic.py` gained **`--spread-plans-only`**: restrict every
metric to plans placing each task on a **distinct node**. Node-occupancy excess is then
identically zero across the retained plans — the column that collapsed the four previous
mechanisms becomes a constant and can explain nothing — and `added_in_batch` is zero too,
since distinct nodes imply distinct platforms. Link contention survives untouched, because it
acts *between* tasks on different destinations. (Guarded by two tests asserting exactly those
two halves of the contract; it is an isolation control, **not** a corpus gate, since it
discards most of the sweep.)

Same 48 matched datasets, restricted to spread plans (mean 187 of ~600 plans retained):

| arm | additive R² | argmin regret mean / max | additive argmin optimal | node repair | link repair k1 / k2 / excess |
|---|---:|---:|---:|---:|---:|
| `mh_off` | **1.00000** | **0.00% / 0.0%** | **100%** | — | — |
| `mh_bw1p5` | 0.99686 | 0.08% / 1.7% | 90% (n=5) | **0%** | 20% / 20% / 20% |
| `mh_bw0p5` | 0.99351 | 0.10% / 1.7% | 81% (n=9) | **0%** | 11% / 22% / 22% |

**Two findings, and they point opposite ways.**

1. **The structural claim is VERIFIED.** Without a backbone the target is additive to
   R² = **1.00000** with **0.00%** regret in **100%** of datasets — once collisions are
   removed, the base physics has *literally no* remaining coupling. With the backbone, regret
   becomes non-zero while the node column repairs **0%** by construction and the link scalars
   repair only 11-22%. This is the **first mechanism in the series to produce coupling that is
   neither collision-shaped nor node-count-shaped**, which is exactly what it was built to do.
2. **The magnitude is two orders of magnitude too small.** 0.08-0.10% mean regret against a
   5% gate, max pinned at **1.7%**, touching 10-19% of datasets.

**Bandwidth is a NULL LEVER — this is the deep-queue arithmetic one level down.** Tripling the
link cost (1.5 → 0.5 MB/s) widened the effect (5 → 9 datasets) but barely deepened it
(0.08% → 0.10%, max unchanged). Measured over the optimal plans of all 48 datasets per arm:

| arm | additive transfer | contention wait | **wait / transfer** |
|---|---:|---:|---:|
| `mh_bw1p5` | 73.73 s | 0.740 s | **0.0100** |
| `mh_bw0p5` | 210.35 s | 1.861 s | **0.0088** |

Store-and-forward charges each hop a full transmission, so the **additive** term
(hops × transfer) and the **interaction** term (crossings × transfer) both scale as
1/bandwidth. The ratio is therefore invariant — capacity changes the absolute size of the
network term but never its additive/interaction split, and contention stays ~1% of the link
cost at any bandwidth. Compare the deep-queue failure, where additive `depth × exec_time` grew
with the lever while interaction `added_in_batch × exec_time` did not.

⇒ **Do not tune bandwidth to rescue this lineage.** The only lever that could move the ratio is
the number of *crossings per segment* — topology and attachment, not capacity.

#### The closing sweep: the whole hub↔mesh spectrum, and a prediction that was wrong

Since the ratio is `crossings / hops`, the topology lever predicts that **fewer** cores should
raise coupling (shorter routes, more traffic per segment). That was expected to run into a
tension: concentrating traffic is exactly what makes "load on the busiest link" a sufficient
scalar, so any gain in magnitude should be paid for in degeneracy. Tested at n_core ∈ {2, 4,
12}, all `attach_degree=1`, pure ring, 1.5 MB/s, 48 datasets each, spread-plans-only:

| n_core | hops/route | wait / transfer | additive R² | regret mean / max | link repair k1 |
|---:|---:|---:|---:|---:|---:|
| 2 (hub) | 2.26 | **0.0189** | 0.99126 | 0.04% / 1.4% | **33%** |
| 4 | 2.60 | 0.0072 | 0.98753 | **0.35% / 7.2%** | 25% |
| 12 (mesh) | 3.93 | 0.0100 | 0.99686 | 0.08% / 1.7% | 20% |

**The tension is real in direction but never binds.** `link_repair_k1` rises monotonically as
the core shrinks (20% → 25% → 33%), confirming that concentrating traffic makes the coupling
more scalar-summarisable. But even at the extreme hub it only reaches 33%, comfortably below
the 0.5 threshold. Degeneracy was never what stopped this mechanism.

**The magnitude prediction was wrong, and the correction matters.** Coupling is *not* monotone
in hub-ness — it peaks in the interior at n_core=4 (0.35% mean, max **7.2%**, the only
configuration whose max clears the 5% gate) and falls off at *both* ends. n_core=2 collapses
because with a single shared segment and one attachment each, roughly half of all
(client, server) pairs hang off the same core and their routes contain **no core link at all**
— the hub wins crossings per link but loses coverage (P0: core-link pair sharing 0.214 at
n_core=2 vs 0.306 at n_core=12).

Nor does `wait / transfer` predict regret: n_core=4 has the *lowest* ratio (0.0072) and the
*highest* regret. The ratio is measured on optimal plans, which select against contention, so
it describes the optimum's composition rather than the spread across the sweep that regret
actually measures. **Use it to rule a lever out (a bandwidth-invariant ratio is decisive), not
to rank the ones that survive.**

**What closes the lineage is neither degeneracy nor a bad configuration — it is uniform
smallness.** Across a 3× bandwidth range and the full hub↔mesh spectrum, mean additive-argmin
regret never exceeds **0.35%** against a 5% gate, and the contention term stays **0.7-1.9%** of
the link cost in every configuration tested. There is no setting in this family within reach of
the gate, so no corpus, cache, training run, or live gate was produced.

**Superseded pilot (same day, recorded so it is not re-run).** A first matched pilot used the
stock `shallow_v1` grid (n=16 × 3 arms: off / 5.0 / 1.5 MB/s) and failed on headroom alone —
regret 2.51% / 1.07% / 1.09%, i.e. the backbone made the corpus *more* separable. Cause:
`shallow_v1` keeps `per_client >= 1`, so many tasks run locally and never touch the network,
and a cost that prices only remoteness pushes the optimum toward the local corner the additive
fit already picks — the same shape as `netc_scarce_v1`. It was also underpowered (2-5
datasets with any regret). `netc_multihop_v1` sets `per_client = 0` and changes nothing else;
server spread deliberately stays at the 0.6 floor, unlike `netc_hotspot_v1`, which moved
`replica_server_percentage` and `per_client` together and whose "cliff" turned out to be one
node-occupancy integer over the only 2 hosts that existed. Frozen reports:
`simulation_data/separability_netc_multihop_{pilot,v1}.json`,
`simulation_data/link_overlap_precheck_netc_multihop_v1.json`.

### link_contention_v1 — a real-trace A/B, at realistic concurrency (2026-08-21, 🔄 IN PROGRESS)

**Why.** The FALSIFIED verdict above ("uniform smallness … regret never exceeds 0.35%") was
measured entirely on 4-task co-sim sweeps, where at most 4 transfers can ever share a core
segment. That is a claim about the mechanism's magnitude *at that concurrency*, not a claim
about its magnitude at realistic load — a real trace at rps=150 presents five to six orders of
magnitude more simultaneous traffic. This does **not** re-decide the co-sim separability
claim (`total_rtt` has no additive-argmin regret to report) — it is new evidence on a
different, previously-untested question: does the backbone change live outcomes at real
concurrency?

**Design.** Matched A/B: identical parity-verified cells, identical trace
(`workload-150-100.json`), identical deployed checkpoint; the only difference is a
`network.backbone` block (`n_core=4, attach_degree=1, chord_count=0, bandwidth_mbps=1.5`) —
`n_core=4` because it is this lineage's own measured interior peak, the only configuration
whose max regret cleared the 5% gate (see the closing hub↔mesh sweep above). The no-backbone
arm is `siv1_full_corpus`'s `workload-150-100` retest (above) on the same cells — reused
directly, not duplicated.

**Blocker found and resolved: `build_core_backbone`'s jitter rng is offset by the
replica-reachability repair.** `generate_infrastructure.py`'s backbone build draws
`rng.sample`/`rng.uniform(-jitter, +jitter)` (`:372-375`) from the *same* rng stream the
reachability repair already consumed via `rng.shuffle` (`:768`), and the backbone is overlaid
*after* the repair (`:780`). A live run autoscales from zero and performs no repair, so it
reaches the backbone build at a different stream position — every access-link latency
diverges on exactly the cells with a non-empty repair set:

| cell | repair edges (corpus) | backbone parity |
|---|---|---|
| cell02 p=0.35 | 0/282 | PASS |
| cell04 p=0.50 | 0/380 | PASS |
| cell05 p=0.20 | 12/172 | FAIL — 12 corpus-only edges |
| cell01 p=0.25 | 14/182 | FAIL — 14 corpus-only edges |
| cell03 p=0.15 | 34/174 | FAIL — 34 corpus-only edges |

Resolved with a narrowly-scoped, control-tested addition to `verify_live_infra_parity.py`:
`--allow-backbone-latency-divergence` downgrades exactly the two finding classes this causes
to notes, and **only** when a backbone is present on both the corpus and live sides (verified:
relaxes all 5 backbone cells to PASS; the same cells still FAIL 3/5 without the flag; a
non-backbone collection is unaffected by the flag either way). This is a live-vs-live-only
relaxation for this matched A/B — the corpus-side artifact exists only to satisfy the
preflight, and both live arms are self-consistent with each other. It does not paper over a
real mismatch on any collection that isn't deliberately using a backbone this way. Also see
GATE TOOLS below — `NetworkFabric.link_wait_total` / `task.link_wait_time` already measure
exactly the contention quantity this lineage needs and were never surfaced in the live result
JSON.

**Smoke result — the effect is not small at real concurrency, and decomposes cleanly.**
Knative, cell02 (p=0.35), `workload-150-100-30k.json` (30,000 events):

| arm | total_rtt | |
|---|---:|---|
| no backbone | 2,043,279.3 | |
| backbone @ 1000 MB/s (non-binding) | 1,451,938.7 | routing/path-sum effect only |
| backbone @ 1.5 MB/s (binding) | 7,867,634.7 | + transmission + contention |

Routing alone **improves** RTT by 28.9% (shorter/better-latency paths over the core vs. direct
one-hop); the binding bandwidth then adds +441.9%; net **+285.1%**, ~entirely bandwidth-driven.
This measures `total_rtt`, an absolute-cost effect — it does not by itself say whether the
backbone changes the *decision* (policy ordering), which is the full-trace A/B's actual
question and was still running when this entry was written.

**Full-scale result (2026-08-21): at real concurrency the backbone dominates absolute cost
AND changes the policy ordering in the GNN's favor.** All 15 runs complete
(`a1_backbone_bw1p5`, backbone `n_core=4, bw=1.5 MB/s`, vs the no-backbone arm
`a4_wl150100`, same cells/trace/checkpoints; local working tree, i.e. the fixed dims 9-11
live path — see the siv1 resolution subsection):

| cell | knative | mlp (vs kn) | gnn (vs kn) | kn backbone/no-backbone |
|---|---:|---:|---:|---:|
| cell01 (p=0.25) | 282,087,829.7 | 201,011,470.6 (−28.7%) | 192,457,679.0 (**−31.8%**) | 11.5× |
| cell02 (p=0.35) | 224,756,932.5 | 133,554,067.3 (−40.6%) | 169,969,881.1 (**−24.4%**) | 9.2× |
| cell03 (p=0.15) | 225,705,548.2 | 168,869,109.0 (−25.2%) | 165,408,019.3 (**−26.7%**) | 8.3× |
| cell04 (p=0.50) | 286,046,987.2 | 169,992,579.0 (−40.6%) | 260,550,047.3 (**−8.9%**) | 14.0× |
| cell05 (p=0.20) | 196,851,644.1 | 487,049,588.6 (+147.4%) | 141,739,988.9 (**−28.0%**) | 7.4× |

Three findings:
1. **Binding bandwidth is a 7–14× absolute-cost effect at real concurrency** — the co-sim
   FALSIFIED verdict's "regret never exceeds 0.35%" was a statement about 4-task sweeps,
   and does not describe live load. (This still does not reopen the co-sim separability
   claim, which is about a different statistic on different data.)
2. **The GNN's advantage over Knative widens under binding bandwidth**: mean margin −9.4%
   (0.9–14.8%) without the backbone → **−24.0% (8.9–31.8%) with it**, 5/5 both arms. This
   is the first live regime where the GNN's edge grows as network structure starts to
   bind — directionally what this lineage's physics was built to create, though from a
   checkpoint never trained on backbone corpora.
3. **MLP beats the GNN on 3/5 cells but keeps its catastrophic cell05 tail** (+147% here,
   +366% on `workload-175-100`, 6.1× on the no-backbone arm) — its wins don't survive its
   worst cell, and the GNN has no such tail on any of the six live sweeps run to date.

Margins are 25–300× the measured 0.1–0.3% local noise floor. Caveat: single trace, single
seed per cell, one backbone configuration; the rng-stream coupling in
`generate_infrastructure.py` (above) is still unfixed, so backbone cells still need
`--allow-backbone-latency-divergence` for parity. The rng bug and the parity-tool fix
stand regardless of this result.

#### ✅ The backbone win is REGIME-LEVEL, not draw luck — and the serving layout was confounding everything (2026-08-23)

**Why this ran.** The result above rests on one trace, one backbone config and **one training
draw**, and the 2026-08-22 variance control said a single-checkpoint verdict is a claim about
a draw. So: take the two draws that **lost** their no-backbone gates — `prefixctl` (the
variance control, 1/5) and `tempfix` (the corrected cache, 0/15 across three traces) — and run
them on the same parity-verified cells, same trace, under both conditions. 2×2×5, plus
`knative` and the deployed checkpoint re-run **in the same batch** so no arm's baseline comes
from a different venue or an unstamped working tree (jobs 710315 / 710335 / 710341).

**A confound had to be removed first, and it is the larger finding.** `prefixctl` and
`tempfix` declare `inference_feature_layout: null` in their sidecars, and
`load_gnn_model` defaulted an undeclared layout to **`atomic21`**, while the deployed
checkpoint's sidecar declares **`dim22`**. `task_dim=3 / platform_dim=14` is structurally
valid under both — they give the same platform columns different meanings (`dim22`
normalizes the queue features) — so nothing raised. **Every deployed-checkpoint gate served
`dim22`; both alternate-draw gates served `atomic21`.** Re-serving the same checkpoints on
the same cells under `dim22`:

| checkpoint | per-cell delta | mean | worst vs 0.1–0.4% noise floor |
|---|---|---:|---:|
| `prefixctl` | −1.2% … −14.8% | **−7.4%** | 37× |
| `tempfix` | −13.3% … −40.8% | **−29.8%** | **102×** |

`dim22` is uniformly better. This is **GNN-specific**: `mlp_scheduler` reads the layout from
its own checkpoint (and infers `dim22` from `input_dim=22` otherwise), so the MLP arms and
every MLP-vs-MLP comparison are unaffected.

**The gate, all arms at `layout=dim22`, `workload-150-100`, vs Knative per cell:**

| cell | deployed | prefixctl | tempfix |
|---|---:|---:|---:|
| *no backbone* | | | |
| cell01 (p=0.25) | −1.2% | +3.5% | −5.2% |
| cell02 (p=0.35) | −13.0% | −2.2% | −20.2% |
| cell03 (p=0.15) | −14.7% | −8.5% | −16.8% |
| cell04 (p=0.50) | −5.4% | +24.5% | −14.2% |
| cell05 (p=0.20) | −13.0% | −6.0% | −13.4% |
| **mean / wins** | **−9.4% · 5/5** | **+2.3% · 3/5** | **−14.0% · 5/5** |
| *backbone `n_core=4, bw=1.5`* | | | |
| cell01 | −31.8% | −24.4% | −32.9% |
| cell02 | −24.4% | −10.1% | −38.6% |
| cell03 | −26.7% | −24.2% | −30.0% |
| cell04 | −8.9% | +35.4% | −37.4% |
| cell05 | −28.2% | −19.8% | −31.8% |
| **mean / wins** | **−24.0% · 5/5** | **−8.6% · 4/5** | **−34.1% · 5/5** |

Four conclusions:

1. **`REGIME-LEVEL` — the backbone win survives draw variation.** Every draw improves
   markedly under binding bandwidth (−9.4→−24.0, +2.3→−8.6, −14.0→−34.1) and every one wins
   ≥4/5. The claim "under binding network contention the GNN beats Knative" no longer rests
   on the lucky checkpoint. It remains one trace and one backbone config.
2. **The `tempfix` 0/15 FAIL is FALSIFIED — it was the serving layout, not the cache.** At a
   fixed layout the corrected-cache checkpoint is the **best artifact on disk**: 5/5 in both
   conditions, beating the deployed draw on mean margin in both (−14.0% vs −9.4%, −34.1% vs
   −24.0%). The 2026-08-21 reading ("the mismatch was an accidental regularizer") is dead;
   the corrected cache is *better*, and the earlier gate could not see it.
3. **The lottery is real but much smaller than recorded.** `prefixctl` is genuinely the weak
   draw — cell04 is +24.5% / +35.4%, a draw-specific failure that persists at a fixed layout
   — but the 2026-08-22 table's 5–36% spread was draw **plus** layout. Ordering at a fixed
   layout: `tempfix` ≳ `deployed` > `prefixctl`.
4. **The venue/code axis is confirmed inert, again.** The deployed arm, re-run on datalab at
   a clean committed tree, reproduces its 2026-08-21 local numbers to
   −31.8/−24.4/−26.7/−8.9/−28.2 against the recorded −31.8/−24.4/−26.7/−8.9/−28.0.

**The second trace agrees (job 710366, `workload-175-100`, same three arms, same cells,
`layout=dim22`, 30/30 clean).** This was run precisely because the paragraph above demanded
it before any promotion:

| | deployed | tempfix |
|---|---:|---:|
| no backbone, mean / wins | −9.2% · 5/5 | **−15.6% · 5/5** |
| backbone, mean / wins | −23.9% · 5/5 | **−33.9% · 5/5** |

Against `workload-150-100`'s −9.4% / −24.0% and −14.0% / −34.1%, the two traces reproduce
each other to within ~1.6pp on every one of the four cells of that table. `tempfix` beats the
deployed checkpoint on **both traces in both conditions**, 20/20 cells beat Knative, and it
loses to deployed on only 2 of 20 individual cells (175-100 backbone cell03 and cell05, by
2.7 and 0.9pp). **`tempfix` is the promotion candidate**; what remains before swapping it in
is a re-gate on `workload-125-225` (the trace where the deployed checkpoint is weakest, 2W/1T/2L)
and `workload-200-200`.

**And the win is not specific to the tuned backbone config (job 710398, 30/30 clean).**
`n_core=4 / bw=1.5` was this lineage's own measured interior peak — chosen because the effect
was largest there — so a win visible only at that point would not be a claim about network
contention. Two further configurations, `workload-150-100`, same cells:

| backbone config | knative baseline | deployed | tempfix |
|---|---|---:|---:|
| `n_core=8, bw=1.5` (different core topology) | — | −22.4% · 5/5 | **−30.8% · 5/5** |
| `n_core=4, bw=0.5` (more binding bandwidth) | — | −24.4% · 5/5 | **−34.4% · 5/5** |
| `n_core=4, bw=1.5` (the original) | — | −24.0% · 5/5 | **−34.1% · 5/5** |

Mean margins move by ≤2pp across a doubled core tier and a 3× tighter bandwidth. **30/30
cells beat Knative across the three configurations.**

**These are also the first backbone gate cells in the repo that are parity-exact by
construction rather than by relaxation** — minted by `important/make_backbone_gate_cells.py`
with `rng_stream: independent_v1`, they pass `verify_live_infra_parity` with **no
`--allow-backbone-latency-divergence`**, including the three cells whose non-empty repair
sets forced the waiver on the legacy `a1` cells. The gate exports `PARITY_EXTRA_ARGS=""`
deliberately, so a regression in the rng fix would fail the job rather than be waived.

**Standing evidence for the GNN win, as of 2026-08-23:** 3 training draws × (backbone,
no-backbone); 2 traces × (backbone, no-backbone); 3 backbone configurations — every GNN arm
beats Knative on ≥4/5 cells under binding bandwidth, and every 5/5 except the weak
`prefixctl` draw. The remaining scope limits are honest and named: one topology family
(20c/20s sparse), and `workload-125-225` / `workload-200-200` not yet run under the
corrected layout. **The MLP baseline was added to all three gates on 2026-08-23 and changes
how this should be read — see the subsection immediately below.**

#### ⚠ The MLP baseline says the GNN's edge is RELIABILITY, not mean latency (2026-08-23)

Every gate above compared GNN draws against Knative only, which cannot distinguish "the graph
model wins here" from "any learned model wins here". The pointwise MLP
(`batch_edge_mlp_full_corpus_siv1_dim22_batchcache.pt`) was run as a fourth arm on all 30
cells of all three gates — same cells, same traces, same `dim22` layout, same
`node_disk_v2` physics (`datalab/mlp_arm_all_gates.sbatch`, jobs 710450/710451). The GNN and
Knative numbers did not move: the re-score is a pure addition to the three verdict JSONs.

Mean margin vs Knative · wins (a win is `< −0.4%`, the noise floor):

| gate / condition | deployed | tempfix | **mlp** |
|---|---|---|---|
| drawgate, no backbone | −9.4% · 5/5 | −14.0% · 5/5 | **+85.1% · 4/5** |
| drawgate, backbone | −24.0% · 5/5 | −34.1% · 5/5 | **+2.5% · 4/5** |
| promo175, no backbone | −9.2% · 5/5 | −15.6% · 5/5 | **+53.4% · 4/5** |
| promo175, backbone | −23.9% · 5/5 | −33.9% · 5/5 | **−35.1% · 5/5** |
| bbrob `n_core=8, bw=1.5` | −22.4% · 5/5 | −30.8% · 5/5 | **+38.5% · 3/5** |
| bbrob `n_core=4, bw=0.5` | −24.4% · 5/5 | −34.4% · 5/5 | **+11.3% · 3/5** |

**The MLP's mean margin is positive — worse than Knative — in 5 of 6 conditions, while every
GNN arm is negative in all 6.** But the mechanism is entirely a tail, and the per-cell record
is uncomfortable:

* **`tempfix` beats the MLP on only 17 of 30 cells; `deployed` on 13 of 30.** On the 23 cells
  where the MLP does not collapse it usually beats *both* GNN arms, often by 10pp or more
  (`promo175`/nobackbone/cell04: MLP −26.1% vs `tempfix` −20.7% vs `deployed` −10.7%).
* **7 of 30 cells collapse catastrophically** — `cell05` in five conditions (+509.8%, +365.5%,
  +195.4%, +147.4%, +119.1%) and `cell03` under both bbrob configs (+79.0%, +31.4%). One such
  cell is enough to swing a 5-cell mean by 100pp.
* The collapse is the `averageOccupation → ~1` packing failure recorded in
  `memory/herosim-mlp-collapse-is-occupation-collapse.md`, reproduced here on cells the MLP
  has never been gated on. It is a known failure mode, not a new one.

**So the defensible claim is narrower than "the GNN beats the pointwise baseline".** It is:
*the GNN is the only arm that beats Knative on every cell of every condition tested; the MLP
achieves a better typical cell and loses the regime on a fifth of them.* For a scheduler that
is a real advantage and it is exactly the advantage a graph-aware model should have — but a
paper claim of the form "GNN > MLP on latency" is **not** supported by these 30 cells and
should not be written. Two open questions this raises, neither answered here: whether the
collapse cells share a structural property the GNN exploits and the MLP cannot see, and
whether an MLP trained on the corrected cache (`..._tempfix`, on datalab since 2026-08-22,
deliberately **not** run as a second arm here) collapses on the same cells.

**Both of those questions are answered in the next subsection.**

#### ✅ The MLP collapse is ARCHITECTURAL — retraining relocates it without reducing it (2026-08-23)

The corrected-cache MLP (`..._batchcache_tempfix.pt`) was run as a fifth arm on the same 30
cells (`datalab/mlp_tempfix_arm_all_gates.sbatch`, jobs 710656/710657, commit `98b41e9`,
all 30 verified `dim22` / non-zero `total_rtt` / clean tree). Only the checkpoint and the
sweep dir differ from the `mlp` arm — cells, traces and parity waivers are byte-identical, so
this is an A/B on training data alone.

| gate / condition | mlp | **mlptempfix** |
|---|---|---|
| drawgate, no backbone | +85.1% · 4/5 | **+133.4% · 3/5** |
| drawgate, backbone | +2.5% · 4/5 | **+12.8% · 4/5** |
| promo175, no backbone | +53.4% · 4/5 | **+98.2% · 4/5** |
| promo175, backbone | −35.1% · 5/5 | **+4.3% · 4/5** |
| bbrob `n_core=8, bw=1.5` | +38.5% · 3/5 | **+28.6% · 4/5** |
| bbrob `n_core=4, bw=0.5` | +11.3% · 3/5 | **+10.5% · 4/5** |

**Exactly 7 of 30 cells collapse under each checkpoint — the same count, a different set.**
The corrected cache *fixed* `cell03` under both bbrob configs (+79.0% → −22.6%, +31.4% →
−21.6%) and *broke* two cells that were healthy before: `cell03` on drawgate/nobackbone
(−24.5% → **+187.6%**) and `cell05` on promo175/backbone (−35.0% → **+127.9%**), the one
condition where the first MLP's `cell05` had survived. Five `cell05` collapses are shared.
Mean margin is *worse* under the corrected cache in 4 of 6 conditions.

A training-data artifact would have been reduced by fixing the training data. An invariant
7/30 with a reshuffled victim list is the signature of an architectural failure whose victim
set is a function of the *weights*, not of the data or the graph. **The reliability claim
therefore hardens: across 120 scheduler runs (2 MLP × 2–3 GNN arms × 30 cells), all 14
collapse events are MLP arms and none is a GNN arm.**

**A detector that separates perfectly on all 120 runs.** `chosen_queue_vs_min` **p95** from
the `.decode_stats.json` sidecar: collapse 13,485–23,866, healthy 449–1,387 — a 9.7x gap with
no overlap. The *median* is normal in both (46–131 collapse vs 43–312 healthy), which is the
direct confirmation that this is a minority-of-decisions tail that compounds. The occupation
ratio to the same cell's Knative arm also holds on all 120 (collapse ≤0.33x, healthy ≥0.41x)
but with only a 1.24x gap, so prefer the p95.

#### ✅ The collapse cells share no STRUCTURE — it is a dispersal failure with two mechanisms (2026-08-23)

Analysis of artifacts already on disk (`extract_gate_stats_summary.py`,
`extract_platform_dispersal.py`; no new sims). The answer to "do the collapse cells share a
structure the GNN sees and the MLP cannot?" is **no**, and four independent checks kill it:

1. **Adjacency is byte-identical across all four cell sets** (`nobackbone`, `a1_backbone`,
   `bb_core8`, `bb_core4` differ only in link latencies, queues and fabric). Degree, choice-set
   size and nearest-replica-host concentration (HHI) do **not** separate collapse from healthy
   — `cell03` is *more* constrained than `cell05` (14 vs 11 clients with ≤2 reachable hosts for
   `dnn2`) yet collapses less often.
2. **It is not an initial-queue "bait".** The platforms that hog the load rank 26/54, 31/54 and
   51/54 by initial queue depth; platform 134 (rank 51/54, one of the *longest* initial queues)
   is the top hog in a collapse run and also the top platform in a healthy one.
3. **The trace that flips `cell05` is a different draw of the same distribution.**
   `workload-150-100` and `workload-175-100` are both 50/50 `dnn1`/`dnn2`, uniform over 20
   sources (4.8–5.2% each), same 100 s duration; they differ in rate and random draw only. On
   identical infrastructure and checkpoint: 6 platforms busy >1% (collapse) vs 83 (healthy).
4. **The victim set moves when only the weights move** (previous subsection). A structural
   property would collapse the same cells under both checkpoints; 4 of the 9 distinct
   (cell, condition) collapse events are checkpoint-specific.

What *does* separate them is dispersal — how widely the scheduler spread load — and there are
**two distinct failure mechanisms** hiding under one `averageOccupation` symptom:

* **Platform-side packing** (12 of 14 events): top-3 platforms hold 43–65% of all busy time,
  2–33 platforms busy >1%. `cell05`/nobackbone MLP puts 63% of busy time on 3 platforms while
  the GNN spreads over 109.
* **Link-side starvation** (2 of 14: `cell03` under both bbrob configs): dispersal looks
  *normal* (top-3 share 16–17%, in the healthy range) and every platform is nearly idle
  (max utilisation **5.6%** and **2.2%**) while RTT is +79.0% / +31.4%. The fabric is empty and
  the tasks still wait.

**Why the link-side one is invisible in the existing metrics, and why `link_wait_total` is the
right fix.** `averageCommunicationsTime` is pinned at ~16.7 ms across all 150 runs (range
0.016662–0.016668) even where `total_rtt` swings 10x between backbone and no-backbone — it
does not measure link queueing at all. The wait is taken *inside* the replica's serving loop
(`infrastructure.py:1082`, `with self.node.fabric.pipe(...).request()`), so it blocks the
replica and surfaces as **queue time**: `averageQueueTime / averageElapsedTime` is 0.9990–1.0000
in every one of the 150 runs. Serialising `link_wait_total` / `linkWaitTime` (gate-tools row,
2026-08-21) would separate these two mechanisms directly instead of by inference from
`max_busy_pct`; it remains a reporting-only change.

**Consequence for the research goal.** The GNN's advantage here is not that it reads a
topological property the MLP is blind to — no such property distinguishes these cells. It is
that it *disperses*, and dispersal is what keeps a metastable queueing instability from
igniting. That is still a graph-aware advantage (a pointwise scorer cannot condition on where
its peers are going), but it should be stated as a dispersal/reliability argument, not as
"the GNN exploits topology `P`".
