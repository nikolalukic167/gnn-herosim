# g32_fourtype_move_s0_v1 — four-type placement-move qualification

> Produced on local-record-2026-10-06 at afea04bd; code not merged; old physics.

**Status:** `CLOSED` (2026-09-23) — **NO-GO for move-value GNN training in this registered seat**. Registered 2026-09-23 before exact move scoring; the required fresh live manipulation gate completed.

**Question.** Does a four-task-type, lower-peer-payload fixed-replica workload create useful placement moves whose value depends on graph information beyond strong hand controls? The [16-plan sampled-slate experiment](g32_sampled_slate_v1.md) and its [exact-validation successor](g32_exact_val_selection_v1.md) failed their bipartite-message-passing bars. This experiment changes the environment and target: it scores exact local moves around existing complete proposals instead of imitating the best of a small slate. The corresponding [hard stop](../hard-stops.md) rules out repeating that sampled-slate recipe, not this move test.

**Outcome.** The bounded move pool has **13.936% median topology headroom** and the frozen simple hand proxy captures less than half of it on each of four training topologies. Yet leave-one-topology-out, non-GNN controls using engineered scalar, queue, platform-count, peer-cut and 1/2/4-hop features capture **86.26–100%** of the oracle gain with Ridge and **61.22–100%** with ExtraTrees at the same 16-evaluation budget. The registered residual-information gate therefore fails: this experiment does not establish a useful advantage for bipartite message passing beyond strong hand features. The separate two-topology live manipulation gate audits successfully; no GNN was trained or live-gated under this lineage. The bounded oracle is not a global optimum, and four topologies cannot rule out other move targets or environments.

**Record.** [S0 result and verdict](#2026-09-23--residual-control-and-final-verdict) · [capture and live manipulation audit](#2026-09-23--capture-and-live-manipulation-audit) · [registration](#2026-09-23--s0-registration-before-move-scoring).

## 2026-09-23 — S0 registration before move scoring

The deterministic workload transform is `/tmp/build_fourtype_g32.py` (SHA-256 `b8689ad3853fa565bf9cb5bef738ae07740c44f606ca395e4abdb541e0205af6`); its 5,120-task output is `/tmp/fixed-g32-fourtype-lowpayload-5120.json` (SHA-256 `deef43af25a148c8c9a9931cf1c205eab1c3960665a1c43d170c0a2d28f89046`). On the **opened development topology 9303**, all four task types are feasible. With six coordinate-descent passes, old MP-OFF has total RTT **6,511.724 s** and hand has **6,462.352 s** (hand **0.758%** better); peer exchange is nonzero and no scale events occur. Its 16 captured groups are `/tmp/fourtype-s0-9303-snapshots.jsonl` (SHA-256 `d661bb98325908e443ac907cdc72cace8e96e2d5dc5b60f2b2ca241f693e475f`). This is a feasibility smoke, not oracle headroom or evidence from fresh topologies.

**Hypothesis.** Four task types and a 0.1 peer-payload multiplier preserve pairwise coupling while scarce eligible platforms make joint task–platform decisions consequential. Use only structurally eligible, fresh training topologies for S0. A coverage-only preflight starting at seed **9801** selected the first four eligible seeds, **9801, 9803, 9804 and 9806**, before any exact move scoring; **9802** and **9805** failed source coverage. The frozen topology configs and SHA-256 hashes are:

| Seed | Config | SHA-256 |
|---:|---|---|
| 9801 | `/tmp/fixed-g32-config-9801.json` | `8149637473d1f232078c07ca7ed130c93b87f43b0c25d4c49a2409f9453a4d9e` |
| 9803 | `/tmp/fixed-g32-config-9803.json` | `c0bdeb6d2c1ef15491cf4c0d6063bcd94202a1638a00049b4c064fa95159be8f` |
| 9804 | `/tmp/fixed-g32-config-9804.json` | `2f9244ab2b8c7562d4598fe811d449e228ceb2a3c9caca6b23f70be89fdd2679` |
| 9806 | `/tmp/fixed-g32-config-9806.json` | `3555953b041c0f0ab9f1b6d15ccb51f751ba42f6888693fe3e9aed582d7ca389` |

Topology is the paired independent unit; groups within one topology are repeated observations, not independent replicates. Keep topology 9303 out of S0 and any future sealed gate.

For each group's hand CD6 and frozen old MP-OFF complete plans, enumerate every feasible **single-task reassignment** to a different eligible platform. Deduplicate complete plans, order candidates deterministically by source plan, task ID and platform ID, and retain at most **128**. Add deterministic **occupancy-preserving two-task swaps** only if the cap leaves room. Record candidate counts, ties and cap coverage. Exact-simulate every retained candidate under the same captured state and physics; the best retained result is a **bounded move-pool oracle**, never a global optimum. The baseline is the better of hand CD6 and old MP-OFF under exact evaluation. For an equal-cost search control, rank candidates with a frozen hand proxy using platform counts and peer features, then exact-simulate only its top **16** per group; count baseline evaluations consistently in both budgets. Freeze the proxy weights and tie-breaks before scoring fresh groups.

Compute each topology's total RTT across its 16 groups for baseline, bounded oracle and 16-evaluation hand search. Define oracle gain as `(baseline − bounded oracle) / baseline × 100` and hand capture as `(baseline − hand search) / (baseline − bounded oracle)` when the denominator is positive. **GO to any training only if** median topology bounded-oracle gain is at least **8%**, the 16-evaluation hand search captures **less than half** of available gain, and a residual decision signal remains after platform-count and peer **1-, 2-, and 4-hop** hand controls. The residual read must report matched or controlled move-ranking reversals and their exact RTT stakes by topology; a gain explained by these controls does not qualify as graph-specific headroom. A nonpositive denominator has no recoverable headroom and cannot count toward GO. No GPU allocation precedes all three S0 gates.

Regardless of an offline NO-GO, finish with a **fresh live manipulation gate** on at least two topology seeds outside 9303 and the S0 four: run hand CD6 and frozen old MP-OFF on the same fixed-replica 5,120-task four-type workload, audit complete task counts, identical eligible-replica manifests, nonzero peer exchange and zero scale events, and report topology-paired elapsed time and RTT. Freeze these seeds and run hashes before reading outcomes. This gate measures whether the environment manipulation survives live execution; it cannot by itself claim a bipartite win.

If S0 passes, register a separate matched full-bipartite, peer-only and MP-OFF move-learning experiment with topology-disjoint training and exact-simulator validation selection. Preserve checkpoint sidecars and Weights & Biases runs. Its fresh eight-topology live gate must compare equal-budget serving and meet at least **2% median gain and 7/8 positive topologies** for full versus each learned control. Neither S0 nor a hand win substitutes for that architecture test.

## 2026-09-23 — capture and live manipulation audit

The training-capture manifest (SHA-256 `259381de9e1dbe215137df7f5bd4ca05e8b0436f306648a2fb05fd526cf02dbe`) records both complete plans and the first 16 snapshots for every registered S0 seed. The [S0 capture archive](g32_fourtype_move_s0_v1/s0_capture_inputs_2026-09-23.tar.gz) preserves its inputs (SHA-256 `85862744f9f533cc12ab3c7ef9792250e0fed15bfa7d5afcb46638e39140361c`). All eight runs completed 5,120 tasks in 160 groups, with zero failures and scale events, nonzero peer exchange, the same fixed-replica manifest within each topology, and feasible captured plans. These are S0 inputs, not move outcomes.

The required fresh live manipulation gate was frozen on structurally eligible seeds **9807 and 9808** before their RTTs were read. Its freeze manifest (SHA-256 `fac98efce5c897fed0fe2ce0fe41aa42517a246446763629f256ab0a70f05b61`) fixes both configs, workload, runners, model, physics and arm list. All four runs passed the same 5,120-task, 160-group, fixed-replica, zero-scaling and nonzero-peer checks. MP-OFF RTT was **4,975.326 s** versus hand **5,016.483 s** on 9807 (0.820% lower), and **6,026.829 s** versus **6,045.356 s** on 9808 (0.306% lower). This shows the manipulated workload runs as intended; neither arm is a new GNN claim. The audit has SHA-256 `4d964f54421c7c56f15f501233204abd2a25e462e80d0eaae14f157b164008e7`. The [sealed raw archive](g32_fourtype_move_s0_v1/live_manipulation_gate_2026-09-23.tar.gz) preserves both manifests and was member-hash verified against the audit; archive SHA-256 `dde0bf3e3e0f6208056c5121fe81d7bf75e26859200bb64b226bd17133779ba3`.

The [exact S0 move archive](g32_fourtype_move_s0_v1/exact_oracle_s0_2026-09-23.tar.gz) preserves the scorer (SHA-256 `5871b6687153298d5f8fe11db891322f5116d535017c79c102a0a1493d7285e2`) and all 64 scored groups (result SHA-256 `6d310e1b8f3832e118ecc9e6ad64f175e8e004534381a7f71d22235a75cf91b5`; archive SHA-256 `72049c6bfb69b6e298bb32e85cd11c3804253a51fa81ee4c7b7c3dd88d49e632`). Both baseline plans and every retained move were scored under the same captured state with peer exchange enabled. The pool hit its 128-move cap in 58/64 groups; this is a bounded oracle, not a complete local optimum.

| Training topology | Baseline RTT (s) | Bounded-oracle gain | Frozen hand capture of gain |
|---:|---:|---:|---:|
| 9801 | 577.112 | 9.132% | 26.70% |
| 9803 | 648.625 | 18.850% | 10.77% |
| 9804 | 751.133 | 18.739% | 10.21% |
| 9806 | 1,109.379 | 5.906% | 49.31% |

The median topology bounded-oracle gain is **13.936%**, above the signed 8% bar, and the frozen-proxy 16-evaluation hand search captures less than half the gain on all four topologies. These pass the first two S0 bars. The preregistered residual-information result follows.

## 2026-09-23 — residual control and final verdict

The residual screen trained **non-GNN hand-feature controls** on three S0 topologies and tested on the fourth, rotating the holdout. Each control ranked the same exact-scored move pool and selected only 16 extra plans per group, matching the frozen proxy's simulation budget. The target was exact improvement over the source complete plan. Features covered task/platform types, current scalar costs and queues, platform counts and load, peer cut and network latency, and moved-task peer neighborhoods at 1, 2 and 4 hops. Ridge used standardized features and fixed regularization; ExtraTrees used fixed tree parameters. The [residual-control archive](g32_fourtype_move_s0_v1/residual_control_s0_2026-09-23.tar.gz) contains the scorer (SHA-256 `85dd1aafcfa6feb243fa30416cb9f1be843fc1b93db4766aebba5a1213b17838`) and full residual result (SHA-256 `26d3a878fd2219e40cf7a7c654b868328772dcd9a2acefb42c6a991eab1f215d`); archive SHA-256 `ea07e4c0e0e3406d959dbb6d7f92998dbdc9dd3e75be021eecd3729d6144f708`. The exact-score input hash is `6d310e1b8f3832e118ecc9e6ad64f175e8e004534381a7f71d22235a75cf91b5`.

| Held-out topology | Bounded-oracle gain | Frozen proxy capture | Ridge capture | ExtraTrees capture |
|---:|---:|---:|---:|---:|
| 9801 | 9.132% | 26.70% | 99.12% | 99.12% |
| 9803 | 18.850% | 10.77% | 100% | 100% |
| 9804 | 18.739% | 10.21% | 100% | 100% |
| 9806 | 5.906% | 49.31% | 86.26% | 61.22% |

Matched coarse move signatures still expose **47 rank reversals** between exact RTT and control predictions across the four held-out topologies. Their largest single exact RTT stake is **14.02 s** on 9806. These are control errors, not demonstrated graph-only information: the 16-evaluation controls already recover most bounded-oracle gain in every topology, and no matched full-bipartite model has shown it can predict those reversals. The signed third S0 bar therefore fails. The two-topology live manipulation gate above confirms the workload and simulator path, so this lineage closes **NO-GO without GNN training**. This verdict is specific to the four-type low-payload fixed-replica workload, the 128-capped single-reassignment/swap pool, and these information-matched controls; it does not close bipartite GNNs generally.
