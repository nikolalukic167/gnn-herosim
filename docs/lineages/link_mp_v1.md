# link_mp_v1 — CLOSED

> **Status:** `CLOSED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Registered:** 2026-08-30, before any arm existed &nbsp;·&nbsp; **Closed:** 2026-08-31

**Question.** [mp_ablation_v1](mp_ablation_v1.md) closed with a null whose every
descriptive line pointed at MP-OFF ≥ MP-ON, and explicitly could not distinguish
**(a)** "message passing is unhelpful here" from **(b)** "message passing runs over the
**wrong graph**" — the bipartite + same-node edge set cannot express shared-link
contention, the one mechanism the environment measurably rewards (the GNN's latency edge
is backbone-only: −25.1% vs +2.5%, `objective_pivot_v1`). This lineage builds the
link-aware graph and asks the paired question again. It is the decisive experiment that
node called for.

**Entry points:** `src/placement/network_graph.py` (contract `core_v1` — pre-existing,
built for `topology_transfer_v1`, never live-gated), `scripts_cosim/important/score_link_mp_v1.py`
(the registered scorer; its constants ARE this registration),
`scripts_cosim/test_score_link_mp_v1.py` (18 tests),
`scripts_cosim/datalab/link_mp_v1_{cosim,recache,train,gate}.sbatch`,
grid presets `link_mp_v1_*` in `generate_gnn_datasets_fast.py`.

**Related:** [mp_ablation_v1](mp_ablation_v1.md) · [objective_pivot_v1](objective_pivot_v1.md) ·
[topology_transfer_v1](topology_transfer_v1.md) · [link_contention_v1](link_contention_v1.md)

## Why this is not topology_transfer_v1 again

`topology_transfer_v1`'s `gnn_topo` arm — the only prior arm with the `core_v1` graph —
FAILED against pointwise. Three reasons that result does not answer this question:

1. **Its corpus's backbone was deliberately non-binding (1000 MB/s)**, chosen so link
   contention could not confound a topology-*structure* claim. Every link feature was
   label-irrelevant by construction; a model trained there has no gradient reason to read
   the fabric. Here bandwidth **binds** (0.5/1.5 MB/s — the live-gate values).
2. **It was never live-gated** — co-sim 4-task snapshots only, on a corpus measured to
   carry ~zero coupling at the held-out sizes (0/1,022 datasets with regret > 1%).
3. It used the ablation harness's small internal GIN, not the production
   `TaskPlacementGNN`, and its checkpoints were never persisted.

## Design

**Corpus** (generated at the pinned commit, before any training): three fabric variants
matching the three live-gate backbone families exactly —
`link_mp_v1_core4_bw0p5`, `link_mp_v1_core4_bw1p5`, `link_mp_v1_core8_bw1p5` — each
4 conn-probs × 5 replica-configs × 3 queue-dists × 10 seeds (1101–1110) = 600 target,
1,800 total. Physics: contention_v2's scarce-warm regime plus two `per_client=0` replica
rows (netc_multihop_v1's lesson: tasks that can run on their own source node make the
network irrelevant to the optimum). `node_disk_v2`, `--allow-non-unique-replicas`.
SKIPPED/FAILED counts are reported per variant in the outcome; the corpus rule is the
grid, not a hand-picked subset.

**Caches:** two, from the same corpus, same flags
(`scale_invariant_v1`, `--platform-feature-dim 14`): `graphs_cache_link_mp_v1_dim14`
(contract off) and `graphs_cache_link_mp_v1_core_v1_dim14` (`NETWORK_GRAPH_CONTRACT=core_v1`).
The recache job fails loud unless both caches are equal-sized and **every** core-cache
graph carries `net_*` attrs.

**Arms** — 16 seeds each (1–16), same wrapper, same hyperparameters, one factor each:

| family | cache | levers (train AND serve) | role |
|---|---|---|---|
| `lgon` | core_v1 | `NEAR_RTT_MP_NETWORK_ENTITIES=1`, `NETWORK_GRAPH_CONTRACT=core_v1` | treatment: MP over the link-aware graph |
| `lgctrl` | off | none | MP over the old graph — attribution control |
| `lgmpoff` | off | `GNN_DISABLE_MESSAGE_PASSING=1` | no MP — the pointwise bar mp_ablation_v1 set |

At a fixed seed the shared modules initialize bit-identically across families (the net
encoders are constructed after them), so seed-pairing is by construction, as in
mp_ablation_v1. Note `lgon` vs `lgmpoff` is a **composite** lever by design: the link
features enter only through message passing, so "MP + the graph that carries contention"
is one treatment, not two — that composite is exactly the capability under test.

**Gate:** the 20 BACKBONE cells of the objective_pivot_v1 corpus
(`drawgate/backbone`, `promo175/backbone`, `bbrob/bb_core8_bw1p5`, `bbrob/bb_core4_bw0p5`;
5 cells each; workloads 150-100/175-100; cell→workload→parity mapping byte-identical to
`mp_ablation_gate.sbatch`). The two FLAT blocks are excluded: a `core_v1` checkpoint fails
loud on a fabric-less graph by design, and the registered question lives where the latency
effect lives. 960 runs total. Frozen same-cell Knative baselines are the reference.

**Serving integration is proven, not assumed:** live serving of a `core_v1` checkpoint had
never run anywhere; on 2026-08-30 a smoke checkpoint (2 epochs, smoke corpus) served
`a1_backbone_bw1p5/cell01_p25_s9001` end-to-end locally — parity preflight PASS,
`[GNN] network_entities=True (core_v1)`, sane RTT.

## Registered endpoints and reading rules (fixed before any data)

α = 0.05 throughout. n = 16 seed-pairs. All Wilcoxon tests are EXACT (2^16 enumeration).

- **PRIMARY** (directional, **one-sided**): exact Wilcoxon signed-rank on per-seed
  differences in mean margin vs same-cell Knative over the 20 cells,
  H1 = `lgon` margin < `lgmpoff` margin. One-sided because the hypothesis is directional
  and fixed here, before any data; the opposite tail is computed and carries its own
  verdict, so harm cannot hide behind the sidedness.
- **S1 (attribution)**: same statistic, `lgon` vs `lgctrl`, one-sided (`lgon` better).
- **S2 (reliability)**: paired one-sided sign test on severe-collapse counts (≥ +50% vs
  Knative) over non-tied pairs, `lgon` vs `lgmpoff`. Ties expected and reported. S2 is
  reported alongside the verdict but does not decide it.
- **Context (not a verdict input):** `lgmpoff` vs `lgctrl` — mp_ablation_v1's question on
  this corpus.

**Verdicts** (mechanical, from `score_link_mp_v1.py`):

| condition | verdict |
|---|---|
| primary p ≤ .05 AND S1 p ≤ .05 | `LINK_MP_WINS_ATTRIBUTED` — ambiguity resolves to (b): MP was over the wrong graph |
| primary p ≤ .05, S1 p > .05 | `LINK_MP_WINS_UNATTRIBUTED` — claim only "MP helps on this corpus" |
| opposite tail p ≤ .05 | `OPPOSITE_DIRECTION` — MP hurts even over the right graph; the supervised MP question closes on both graphs |
| otherwise | `NO_DIFFERENCE_DETECTED` — failure to detect, not equivalence; ambiguity stays open |

**Registered consequences.** On `NO_DIFFERENCE_DETECTED` or `OPPOSITE_DIRECTION`: no
re-runs with tweaks; the remaining message-passing question moves to the closed-loop
phase (`objective_pivot_v1` P1), where the training objective is the first one that
actually pays for anticipating contention. On either WIN verdict: the link graph becomes
the default GNN configuration for subsequent phases, and the mp_ablation_v1 rewording of
the Phase 1 claim stays as written (that claim is about the deployed full-corpus model,
which remains fabric-blind).

**VOID conditions** (scorer-enforced, per arm, from the summary's own provenance): wrong
or dirty commit vs the pin; missing any of its 20 cells; `lgon` without
`NETWORK_GRAPH_CONTRACT=core_v1` in serving env; any control arm WITH it;
`lgmpoff` without `GNN_DISABLE_MESSAGE_PASSING=1`; the lever leaking into `lgon`/`lgctrl`;
any mismatch on layout dim22 / `scale_invariant_v1` / `node_disk_v2` / argmax / batch 4 /
timeout 0.002 / cpu device. A checkpoint sidecar that contradicts its family fails the
sbatch before any simulation runs. A VOID is a fix-and-re-run, never a FAIL.

**Power note.** mp_ablation_v1's two-sided primary landed at p = 0.05066 on a −5.63 pp
mean effect over 30 mixed cells. This design is one-sided (registered here, doubling
power at the same α), on backbone-only cells where per-seed margins have roughly twice
the dynamic range. If the effect is real and of comparable size, this design resolves it;
if it is materially smaller, `NO_DIFFERENCE_DETECTED` is the honest answer and the
consequence above applies.

## Amendment 1 — pin (to be filled at launch, before any job runs)

`PIN_COMMIT` in `score_link_mp_v1.py` and the four sbatch files names the commit every
arm must run at. It is set — and this section updated — in the commit immediately after
the registration commit, because a commit cannot contain its own hash. Until then the
scorer refuses to score (`PIN_COMMIT = None`), and that refusal has a test.

**Set 2026-08-30:** pin = `8aef27a98fd636000008468d75a52d645f999969` — the registration
commit itself. This amendment commit changes only the pin constants in
`score_link_mp_v1.py` and the four sbatch files, plus this section; none of those alter
arm behavior (the sbatch files are read from HEAD in the main checkout and only `cd` into
the pinned worktree, the mp_ablation pattern). Pinned worktree on datalab:
`~/gnn-herosim-pin-8aef27a`.

## Amendment 2 — corpus frozen at 1,675 datasets (2026-08-31, before any arm existed)

User instruction (verbatim): "can we run the full pipeline (without my intervention) with
the data that we have here?" — recorded as the authorization for this amendment.

**Corpus = the grid as generated at freeze time:** core4_bw0p5 556 / core4_bw1p5 561 /
core8_bw1p5 558 = **1,675** complete datasets (93% of the 1,800 target; MIN floor 1,500
holds). The three still-running tail shards (726506_9/_19/_29, covering grid indices
540-599 per variant) were cancelled at freeze; any dataset they left mid-write has no
`placements.jsonl` and is excluded by the cache builder's own completion signal.

**Validity note.** Zero registered arms existed at freeze (verified: no `gnn-linkmp-lg*`
checkpoints), so this changes no result and biases no comparison: all three families
still train on the identical corpus and differ by exactly one factor. **Scope caveat,
recorded up front:** the missing ~125 datasets are the tail of the grid enumeration —
the slowest, most contention-heavy combos (~20-30 min each vs seconds for the median).
The frozen corpus therefore under-represents the heaviest-contention training examples;
if the verdict is a null, this caveat is part of its honest reading (and is exactly the
`MAX_PLACEMENT_COMBINATIONS` skip-threshold failure mode, hit deliberately this time and
written down rather than silently).

**Pipeline change:** a final auto-score stage (`link_mp_v1_score.sbatch`) is appended
after the gate — end-to-end parse verification of all 960 results, summary extraction at
HEAD, then `score_link_mp_v1.py` — so the registered verdict lands with no manual step.

## Record

### Outcome — NO_DIFFERENCE_DETECTED on the primary, and the mechanism resolved by the secondaries (2026-08-31)

Chain 727300 (recache, 1,675 graphs both caches, fabric on all core graphs) → 727301
(48/48 trainings, venue-bound via `PROJECT_ROOT`) → 727302 (960/960 gate cells, zero
failures) → 727303 (auto-score: 960/960 parse end-to-end, all 48 arms provenance-verified
at pin `8aef27a`). Verdict JSON archived at `link_mp_v1/verdict.json`. Paper-grade
write-up of the corpus-fabric finding: `link_mp_v1/writeup_corpus_fabric.md`
(2026-09-01; numbers frozen to this node).

| endpoint | result |
|---|---|
| **PRIMARY** lgon vs lgmpoff (one-sided Wilcoxon) | mean diff **+0.47 pp**, p = **0.372** → **NO_DIFFERENCE_DETECTED** |
| **S1** lgon vs lgctrl (attribution) | mean diff **+4.98 pp**, p = **0.00459** — lgon significantly better |
| **Context** lgmpoff vs lgctrl | mean diff **+4.50 pp**, p = **0.00107** — no-MP significantly better than old-graph MP |
| **S2** severe collapse (≥+50%) | **all 48 arms, all 20 cells: zero collapses** (also at +30% and +100%); 16/16 ties, p = 1 |
| family means vs Knative (20 backbone cells) | lgon **−38.5%** · lgmpoff **−38.0%** · lgctrl **−33.5%** (deployed fabric-blind model: −25.1%) |

**Reading, in one sentence: message passing over the old graph is measurably harmful
(−4.5 pp, p = .001); the link-aware graph repairs exactly that harm (+5.0 pp, p = .005);
and repaired MP lands precisely at the no-MP level (+0.5 pp, n.s.).**

Three consequences:

1. **The mp_ablation_v1 ambiguity is resolved — both halves were true.** (b) The graph
   WAS wrong: bipartite + same-node message passing actively hurts, now at p = .001 on a
   corpus where the network matters. And (a) MP is still not a net win even over the
   right graph — the supervised MP question is **closed on both graphs**. Per the
   registered consequence: no re-runs with tweaks; the remaining MP question moves to the
   closed-loop phase (`objective_pivot_v1` P1). This outcome is exactly what
   `program_verdict_v1` predicts for a pointwise-separable supervised target: the
   correctly specified pointwise scorer is the ceiling, and the best any graph
   architecture can do is *reach* it — measured here as a tie to within half a point.
2. **The corpus is the biggest lever ever measured in this program.** Every family
   trained on the binding-backbone corpus beats the deployed model's −25.1% by ~8–13 pp,
   and **not one of the 48 arms collapses on any cell at any threshold** — the first
   all-clean reliability table this repo has produced. Architecture moved the needle
   ≤5 pp; matching the training fabric to the serving fabric moved it ~13 pp.
3. **The link graph still earns its keep** — as a *repair*, not a boost. If message
   passing is retained for the closed-loop phase (where the objective finally rewards
   joint reasoning), it must run over `core_v1`; running it over the old graph is now
   known to cost ~4–5 pp. If MP is dropped instead, `lgmpoff`-on-this-corpus is the
   strongest, tightest supervised baseline available (−38.0%, spread −34.3…−43.9, zero
   collapses).

**Scope caveats, as registered:** corpus frozen at 93% (Amendment 2 — the missing tail
is the heaviest-contention combos); backbone cells only; margins are vs frozen same-cell
Knative baselines. The preview's one bad lgon draw (pv seed 4, partial corpus) did not
recur in the registered 16 — worst lgon seed is −27.2% with zero collapses.


## 2026-08-31 — Promotion: `gnn-linkmp-lgon-s8` is the reference checkpoint

`models/gnn-linkmp-lgon-s8.pt` (+ `.contract.json`) replaces the fabric-blind full-corpus
siv1 checkpoint as the default GNN baseline for future comparisons.

**Selection rule, fixed before looking at file names:** family = the one whose serving
configuration is *self-describing* among the statistically tied pair (primary p = 0.372) —
that is `lgon`: its network encoders are weight-visible, so `load_gnn_model`
(`src/executesimulation.py:772-794`) detects them from the state dict, adopts
`network_graph_contract: core_v1` from the sidecar (setting `NETWORK_GRAPH_CONTRACT`
itself when unset), and fails loud on any mismatch. `lgmpoff` was rejected for promotion
*despite* its tighter spread because `GNN_DISABLE_MESSAGE_PASSING` is env-only
(`gnn_model.py:295`) and absent from the sidecar whitelist — served without the env var it
silently runs message passing over untrained GIN weights, exactly the mismatch class of
2026-08-16. Seed = the median of the 16 by 20-cell mean margin, taking the middle seed
closer to the family mean: seed 8, **−38.0% vs Knative** (family mean −38.5%). Median, not
best, on purpose — best-of-16 is a draw-lottery pick that inflates expected live quality
(`docs/lessons.md` 2026-08-24 seed entries).

Provenance: trained in the pin worktree at `8aef27a…` on the frozen 1,675-dataset
binding-backbone corpus; md5 verified identical local↔datalab
(`c1021bf941b30612b0cee72f54c8215f` / sidecar `d53724c9b8cf709753b33e31e275fad8`); local
smoke-load confirms `network_entities=True (core_v1)`, 52,801 params, env auto-set.

**Caveat:** `objective_pivot_v1` Phase 1's reliability claim (severe-collapse burden
< MLP's) was measured on the *old* deployed checkpoint and does **not** transfer to this
one automatically. This checkpoint's own reliability evidence is the 48-arm zero-collapse
table above (backbone cells, +30/50/100%).

## 2026-09-03 — Exploration pilot: the corpus-matched MLP control this lineage never trained

**Not a registered gate.** Four seeds, no threshold, no verdict; recorded because it
changes what the standing numbers may be quoted as.

**The gap it closes.** All 48 arms here were `TaskPlacementGNN` variants. The MLP the
promoted GNN is compared against on the bbrob cells (`fc_siv1_dim22_tempfix`, −29.2% vs
Knative on `bb_core8_bw1p5`) was trained fabric-blind on `full_corpus_siv1`. The corpus is
the largest lever ever measured in this repo (~13 pp on the GNN, above), so "GNN −44.8%
vs MLP −29.2%" confounds model class with training corpus.

**Arm.** `experiments/link_mp_v1_mlp.yaml` — the deployed MLP recipe (dim22, hidden 64,
lr 1e-3, 100 epochs / patience 10) on `graphs_cache_link_mp_v1_core_v1_dim14`, the cache
`gnn-linkmp-lgon-s8` trained on. Seeds 1–4 via `run_experiment.py --seed`. Evaluated
with `scripts_cosim/closed_loop/evaluate_policy.py` (the Phase 3 gate evaluator), argmax,
`workload-150-100-30k`. Jobs 734411 (train + core8) and 734414 (core4, with paired
reference arms). Checkpoints `models/tabular/batch_edge_mlp_link_mp_v1_dim22_batchcache_seed{1..4}.pt`
on datalab; eval JSONs under `simulation_data/link_mp_v1_mlp_pilot/`.

| arm | `bb_core8_bw1p5` mean (5 cells) | vs Kn | `bb_core4_bw0p5` mean (5 cells) | vs Kn |
|---|---:|---:|---:|---:|
| Frozen-GNN `gnn-linkmp-lgon-s8` | 4,945,399 | −44.8% | 11,701,136 | −49.3% |
| corpus-matched MLP s1 | 4,944,177 | −44.8% | 12,536,952 | −45.7% |
| corpus-matched MLP s2 | 4,420,203 | −50.6% | 10,969,036 | −52.4% |
| corpus-matched MLP s3 | 5,021,021 | −43.9% | 12,863,342 | −44.2% |
| corpus-matched MLP s4 | 4,931,116 | −44.9% | 12,513,525 | −45.8% |
| Frozen-MLP `fc_siv1_dim22_tempfix` (fabric-blind) | 6,340,642 | −29.2% | 16,969,263 | −26.4% |
| Knative | 8,953,094 | — | 23,067,250 | — |

**Reading.** On both fabrics the corpus-matched MLP seeds bracket the promoted GNN: three
of four within 1.5% of it on core8 and one 10.6% better; on core4 the GNN sits a few pp
ahead of the MLP median with one MLP seed 6% better. Zero collapse cells in all 40 MLP
cells. The 15–23 pp "GNN vs MLP" latency gap on these cells was the corpus, which is the
model-side twin of `program_verdict_v1`: on a pointwise-separable target the pointwise
model is correctly specified and, given the same corpus, reaches the same ceiling.

**Consequences.** (1) The bbrob standings must be quoted as "both corpus-matched model
classes ≈ −45 to −50% vs Knative"; the fabric-blind MLP is a corpus control, not a
model-class baseline. (2) The GNN's remaining measured edge is `objective_pivot_v1`
Phase 1 reliability, which was measured against the *fabric-blind* MLP and is untested
against this one — a 16-seed paired reliability gate on corpus-matched arms is the
registration this pilot licenses drafting. (3) Venue parity held to the last digit for the
paired arms re-run locally (`frozen_gnn` cell01 5,840,709.06 and `frozen_mlp` cell01
7,076,053.5 on both machines).
