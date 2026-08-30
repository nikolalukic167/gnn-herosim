# link_mp_v1 — REGISTERED

> **Status:** `REGISTERED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Registered:** 2026-08-30, before any arm was generated, trained, or gated

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

## Record

*(empty — no runs yet)*
