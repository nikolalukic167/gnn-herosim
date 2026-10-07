# proposal_frontier_v1 — proposal quality versus CPU decision time

> Produced on local-record-2026-10-06 at afea04bd; code not merged; old physics.

**Status:** `CLOSED` (2026-09-23) — **NO-GO for unique full-GNN proposal value in this frozen quality-time recipe**. The first gate was invalid at its peer-accounting audit; the corrected gate retained the signed bars and used eight untouched topologies.

**Outcome.** The full-GNN portfolio beats hand multistart by **5.048% median** on **8/8** topologies, but improves on the time-feasible, equal-size new MP-OFF portfolio by only **0.251% median** on **5/8**. Both controls have lower median measured CPU decision time, so the registered ≥2% and ≥7/8 bars against *each* control fail. All 24 corrected live runs audit. This closes this checkpoint, selector, fixed-replica physics and workload recipe; it does not reverse the older checkpoint's complementary proposal result or rule out other learned proposals.

**Record** (newest first):

- [2026-09-23 — corrected sealed live gate: NO-GO](#2026-09-23--corrected-sealed-live-gate-no-go)
- [2026-09-23 — corrected sealed live-gate registration](#2026-09-23--corrected-sealed-live-gate-registration)
- [2026-09-23 — initial gate invalid at peer-accounting audit](#2026-09-23--initial-gate-invalid-at-peer-accounting-audit)
- [2026-09-23 — initial sealed live-gate registration](#2026-09-23--initial-sealed-live-gate-registration)

**Question.** Does adding the exact-validation-selected full GNN proposal improve the live quality versus CPU decision-time frontier of an existing MP-OFF-plus-hand selector? The [older selector gate](gnn_proposal_selector_v1.md) found complementary value for a different GNN checkpoint, but its three-proposal arm had an extra candidate. The [matched 32-task gate](g32_exact_val_selection_v1.md) found no full-GNN advantage over separately trained controls in equal-size portfolios. Neither result answers this cost-quality question.

**Protocol:** [proposal_frontier_v1.json](../../experiments/proposal_frontier_v1.json). No new training or checkpoint selection is part of this gate. Serve the exact-validation-selected final full GNN and new MP-OFF checkpoints, plus the frozen old MP-OFF checkpoint, with their contract sidecars. The opened topology **9303** is for calibration only and cannot contribute to the sealed result.

## 2026-09-23 — initial sealed live-gate registration

Coverage-only fixed-replica preflight examined seeds in ascending order from **10301–10340** and selected the first eight with source coverage: **10301, 10304, 10305, 10307, 10308, 10309, 10310, 10311**. Seeds **10302, 10303, 10306** failed coverage. No task outcome informed selection. Each selected topology receives the same connected, peer-enabled, fixed-replica workload: **160 groups of 32 tasks**, or **5,120 completed tasks per arm**. All arms use six coordinate-descent passes, peer-exchange scale **4**, CPU serving, and the frozen deterministic base-plus-queue selector score with coefficient **1** and arm-name tie-break.

| Arm | Candidate plans |
|---|---|
| Full GNN | Final full GNN; old MP-OFF; hand at scale 4 |
| New MP-OFF | Final new MP-OFF; old MP-OFF; hand at scale 4 |
| Hand multistart | Old MP-OFF; deterministic hand at scales 0.5, 1, 2, 4, 8, 16 |

The two learned arms have three candidates each. Hand multistart has seven; its measured decision time determines whether it is a time-feasible control, and its extra candidates remain visible in the interpretation.

The primary quality metric is live **mean elapsed seconds per task** within topology. The cost metric is measured **total CPU inference seconds divided by completed tasks**, reported per topology and at the median. Topology is the paired independent unit; groups and tasks are not independent replicates. Report the quality-time frontier, distinct candidate-plan count, and selector choice frequency. The latter two are diagnostics, not success bars.

For each control, paired quality gain is `(control elapsed − full-GNN elapsed) / control elapsed × 100`; ties are not positive. A control is time-feasible when its **median measured CPU cost is no greater than the full-GNN arm's median cost**. This recipe passes only if the full-GNN arm gains **at least 2% at the median** and is positive on **at least 7/8 topologies** against **each time-feasible control**. Report slower controls on the frontier without treating them as a pass-bar comparison. Passing establishes value for this served proposal portfolio, not a message-passing-specific effect.

The initial audit required all **24** live runs to have 160 complete groups and 5,120 task completions per arm, matching fixed-replica manifests within topology, enabled and nonzero measured peer exchange **in each arm**, checkpoint and sidecar contracts, code and input hashes, complete selector traces with all candidate plans, and measured inference time. Any failed audit invalidates the affected comparison until repaired and rerun under the frozen protocol. The [initial freeze](proposal_frontier_v1/gate_freeze_2026-09-23.json) preserves the input and runner hashes (SHA-256 `2bb8be12ffec45734fe86027b015b89060e83a0bef7b53c08bf8fd93d0c8e518`).

## 2026-09-23 — initial gate invalid at peer-accounting audit

The first gate stopped after three of eight topology triples. **10301** and **10304** passed the initial audit. At **10305**, the learned-selector arms measured zero peer-transfer time while hand multistart measured positive peer-transfer time. Zero is a valid outcome when a selected plan co-locates peers; a positive per-arm transfer-time requirement therefore tests the selected placement, not whether peer physics was enabled. The [initial audit copy](proposal_frontier_v1/invalid_audit_2026-09-23.py) contains that erroneous assertion, and the [initial runner copy](proposal_frontier_v1/invalid_gate_runner_2026-09-23.py) preserves the attempted gate. The incomplete eight-topology gate is **INVALID**, neither PASS nor NO-GO. A repair-audit command incidentally printed a truncated **10305 hand-multistart total RTT (3,621.56 s)** and inference-time snippet after the corrected 104xx seed range had already been selected. No partial quality or cost metric informed the corrected gate, and initial outputs were excluded from its result.

The repaired audit will assert that peer physics is enabled for the frozen workload and that each arm's measured peer accounting is finite and nonnegative. A separate discriminating placement can establish that the configured physics can produce peer transfer; the selected live plan need not produce transfer on every arm. The corrected gate retains the same three arms, workload, checkpoints, selector, metrics, and pass rule above, with a new freeze and untouched coverage-selected topologies from **10401–10440**.

## 2026-09-23 — corrected sealed live-gate registration

Coverage-only preflight examined seeds in ascending order, stopping after the eighth source-covered topology. The selected seeds are **10401, 10402, 10406, 10411, 10414, 10415, 10416, 10417**. Seeds **10403, 10404, 10405, 10407, 10408, 10409, 10410, 10412, 10413** failed coverage. No task execution or quality outcome informed this selection. The corrected audit and [repaired freeze](proposal_frontier_v1/gate_freeze_repaired_2026-09-23.json) (SHA-256 `936c0486ba1d0a0d68e716ca87d83c0a1bc21b2b183ad471cfb923689cce62df`) apply to **24** complete live runs, with **160 groups and 5,120 tasks per arm**. The initial 103xx outputs remain excluded.

## 2026-09-23 — corrected sealed live gate: NO-GO

All **24/24** corrected live runs pass the repaired audit across eight paired topologies, each with **160 complete groups and 5,120 completed tasks**. Fixed-replica manifests match within topology; the peer-enabled workload, finite nonnegative peer accounting, checkpoint sidecars, input and code hashes, complete candidate-plan traces, and measured CPU inference time pass. All runs share one effective code fingerprint. The independent unit is topology. The [gate read](proposal_frontier_v1/repaired_gate_read_2026-09-23.json) contains each arm's full per-topology data, and the [sealed archive](proposal_frontier_v1/repaired_live_gate_2026-09-23.tar.gz) contains all 24 raw results, traces, audits, inputs and runner sources. Archive SHA-256: `178b7db3ed6543d8b942de8dc7ef82f8f30e4f9b0b961b3525a0b8b832532339`; [member manifest](proposal_frontier_v1/repaired_archive_manifest_2026-09-23.json) SHA-256: `c5bb7787bc66fb4f307b075f2011134eed0a2fd5fbafddf80fff6381f459b27a`. Every frozen input and raw archive hash was rechecked. The [invalid initial archive](proposal_frontier_v1/invalid_initial_gate_2026-09-23.tar.gz) is preserved separately (SHA-256 `b3fc55047a229893329f75836f07126362f7c369014cc94d1ab12f108af6a3f8`) and contributes no result.

| Topology | GNN gain vs new MP-OFF | GNN gain vs hand multistart | CPU ms/task: GNN | New MP-OFF | Hand multistart |
|---:|---:|---:|---:|---:|---:|
| 10401 | +0.500% | +0.737% | 2.156 | 2.117 | 1.989 |
| 10402 | −0.298% | +26.984% | 2.128 | 2.103 | 2.306 |
| 10406 | +1.487% | +2.344% | 2.015 | 1.978 | 1.998 |
| 10411 | +0.641% | +6.639% | 2.157 | 2.129 | 2.182 |
| 10414 | +0.001% | +9.950% | 2.198 | 2.163 | 2.482 |
| 10415 | −0.342% | +3.456% | 2.000 | 1.991 | 1.884 |
| 10416 | +2.008% | +1.393% | 1.922 | 1.922 | 1.684 |
| 10417 | −0.358% | +12.939% | 2.042 | 2.005 | 2.040 |
| **Median** | **+0.251%** | **+5.048%** | **2.085** | **2.054** | **2.019** |

Both controls are time-feasible by the registered **median CPU cost** rule. GNN gain is positive on **5/8** topologies against the new MP-OFF portfolio and **8/8** against hand multistart. Thus the hand comparison passes its quality bars but the new MP-OFF comparison fails both the **≥2% median** and **≥7/8 positive** bars. The median elapsed times by arm are **2.876 s/task** (GNN), **2.897** (new MP-OFF) and **3.126** (hand multistart); these medians are descriptive and do not replace paired median gains.

The table's “CPU ms/task” is elapsed decision wall time while inference was served on CPU, not process CPU usage. It is measured outside simulated RTT and is not charged to the elapsed-time quality metric.

Candidate diversity and selector choices are diagnostic. Distinct plans per topology range **419–473** for GNN, **420–477** for new MP-OFF, and **915–1,057** for hand multistart. Across **1,280 groups per arm**, the GNN arm selected its new proposal **497** times, old MP-OFF **312**, and hand at scale 4 **471**; the new MP-OFF arm selected its new proposal **478**, old MP-OFF **331**, and hand at scale 4 **471**. Hand multistart selected old MP-OFF **380** times and its six hand scales **51, 3, 91, 211, 140, 404** times in ascending scale order. The three-candidate learned arms give the direct cardinality comparison; the seven-candidate hand arm is assessed through measured CPU time. A selected GNN proposal is evidence of diversity, not proof of a material outcome gain over the learned control.
