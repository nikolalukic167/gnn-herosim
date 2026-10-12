# slate_bar_v1

<!-- Header, status and the live plan: S5 (rp/slate-bar). S6 owns only the dated recall-audit section below. -->

## 2026-10-12 — recall audit (S6)

CPU only: no co-sim, no training. The expert asked for it on 2026-10-12 while long_horizon_s0_v1 ran. This section was written,
prediction included, before any row was read.

**Question.** At a fixed plan budget, how often does a candidate slate contain the candidate the hand rules actually pick?
Each batch task gets a slate of candidates, and a search over the slate can only return a plan made of slate candidates. So the
share of chosen candidates inside the slate (the recall) bounds what any slate-restricted search can reproduce of the rules.

**Whose choice.** Two rules:
- **full CD** (`peer_greedy_network_cd`, the gate arm `cd`);
- **cd_exactS** (`HEROSIM_PG_CD_EXACT=1`, with the expansion as its over-cap fallback; the gate arm `cd_exactS`).

Each pick comes from a **decision replay** on the snapshot. The kept raw results cannot supply it: every snapshot below was
captured under `peer_greedy_network_batch`, so no CD or cd_exactS run ever decided these batches.
- The replay is `rollout_replay._one` in batch-only mode (`H < 0`, no table).
- It runs with no sweep, so `_pg_allowed` is `None` and both rules choose from every reachable replica, exactly as they do live.
- It sees the snapshot's own batch, its queued tasks on their recorded replicas, and its ghosts, pulls and warmth.
- A decision hook (`slate_bar_v1_recall.hook`) reads the plan at the batch decision, before anything is enqueued, with the same
  scheduler objects the rule used.
- Determinism check: the first 20 jobs of each source are replayed twice, and their plans must be identical.

**Data.**
- **(a) The accel corpus snapshots,** `/share/nikola.lukic/workload_fix_v1/accel_replica_corpus/snapshots/cc40s<topo>_<rung>_g<w>.jsonl`.
  These hold every candidate, not only the slate.
  - **Topologies:** the split's `train` topologies only (`accel_replica_v1_split.json`). That excludes the test topology 16226
    (the 57 test ids), the held-out set 16251–16258 and 16259–16270. The `shard_*` copies are skipped.
  - **Every snapshot, not only the corpus' t ≥ 360 s ones.** The opening is where load lives
    (`herosim-accel-corpus-has-no-loaded-states`).
  - **Sampling:** stratified by (rung, batch size n). At most 60 snapshots are drawn per stratum, seeded
    (`random.Random(f"20261012:{rung}:{n}")`). Population counts are kept, so the pooled numbers are weighted back to the population.
  - **The physics environment and cell config** come from a train dataset of the same topology and rung.
- **(b) The G=8 screen states,** `/share/nikola.lukic/big_groups/states/top5`: every dataset tagged `_G8` (56 per rung). These are
  the `big_groups_s0_v1` states the prune_g8 screen replays.

**Slate rules.** Standalone cost is `_pg_standalone`: drain + cold + exec + latency, the terms `declared_slate.standalone_cost` ranks by.
Ties go to (node id, platform id). The universe is every reachable replica (`_get_valid_replicas`), which is the set the declared
pruning ranks. The hook checks that its top 5 equals `declared_slate.prune_candidates` on `_candidate_payload` and counts any mismatch.

| rule | per task | budget |
|---|---|---|
| (1) **top5** (declared) | the 5 cheapest by standalone cost | ≤ 5 |
| (2) **union** | the 3 cheapest by standalone cost, then the 2 cheapest of the remaining candidates by standalone + X | ≤ 5 |
| (3) **top6** | the 6 cheapest by standalone cost | ≤ 6, plans ≤ 6ⁿ (disclosed) |
| (4) **partner** (S6's own, named here before any read) | the 3 cheapest by standalone cost; then, taking partners in descending payload order, the cheapest not-yet-kept candidate on each partner's anchor node, until 2 are added; any free place filled in standalone order | ≤ 5 |

- **X in rule (2)** sums over the task's partners: Σ_j min over m ∈ M_j of `_pg_exchange_seconds(c, [(m, payload_j)])`.
  - M_j is a placed or queued partner's known node.
  - For an undecided partner in the batch, M_j is the set of nodes of that partner's 3 cheapest candidates.
  - "∪" is read at a fixed budget: the exchange-priced pair is drawn from outside the top 3, so the slate has 5 whenever the task has ≥ 5 candidates.
- **The anchor in rule (4)** is a placed partner's known node, or else the node of an undecided partner's cheapest candidate.
  - Rule (4) needs no exchange model; it asks whether co-location alone is the gap.
  - The rationale is that argmin-S misses are under-co-location (plan_rerank_v1).

**Amendment 1 (2026-10-12, after the smoke, before any recall number was read).**
- **Finding:** the 16-replay smoke (863470) found the two standalone orders disagreeing on the top-5 set in 28 of 98 tasks.
  - The declared function (`declared_slate.prune_candidates` on `_candidate_payload`) is what the GNN is served.
  - `_pg_standalone` is what cd_exactS ranks its own slate by.
- **Change:** rule (1) splits in two.
  - **top5** is now the declared function itself.
  - **top5cd** is cd_exactS's own top 5 by `_pg_standalone`. It is a ratio-1 rule and enters the recommendation like union and partner.
  - Rules (2)–(4) keep the `_pg_standalone` order.
- **Diagnostics:** the hook records both costs, the payload cold and drain, and the initialized flag for every couple in the
  symmetric difference, so the cause can be named.
- **Re-decision check:** the read also counts batches whose final replayed plan differs from the picks at the batch decision
  (`pick_not_final`). In 6 of 16 smoke replays the decision hook fired on a batch decision that was followed by further
  decisions. Recall is scored on the picks at the batch decision.

**Read** (`slate_bar_v1_recall.py read`), per source × rung × batch size n, and pooled per rung with population weights:
- **Task recall:** the share of the batch's undecided tasks whose chosen couple lies in the rule's slate. It is also reported on the
  tasks with more than 5 candidates, the only ones where the top 5 can miss.
- **Batch recall:** the share of batches with every undecided task inside. This is what an exact search over the slate needs to
  reproduce the rule's plan.
- **Plan-count ratio:** Π|slate_i| / Π|top5_i| over the undecided tasks, as a median and a geometric mean.
- **Also:** the standalone rank of each chosen couple, the share of batches over the 100k cap, and cd_exactS's own counters
  (exact batches, fallbacks).

**Prediction (written before data).**
- **CD task recall under top5:** 93–97 % pooled. On tasks with more than 5 candidates it falls to 80–90 %.
- **CD batch recall under top5:** above 90 % at n ≤ 3, falling to 50–70 % at G=8.
- **cd_exactS recall:** above CD's under every rule, at 96–99 % task recall. It searches the top-5 slate, and the
  refine passes after the search are the only route out.
- **top6:** +1–3 pp task recall over top5, at a plan ratio of up to 1.2ⁿ (≈ 4.3× at n = 8).
- **union:** at or above top5 by 0–3 pp at a ratio ≤ 1, with the gain concentrated at G=8.
- **partner:** within 1 pp of union.

**Recommendation rule (fixed now).** Recommend the rule with the highest pooled **batch** recall for cd_exactS at G=8 (the bar's
choice where the slate binds most), among the rules with a plan ratio ≤ 1.
- top6 is recommended only if it beats every ratio-≤-1 rule by ≥ 5 pp of batch recall at G=8, and that would be disclosed.
- If no ratio-≤-1 rule beats top5 by ≥ 2 pp, top5 stands.

### Result (2026-10-12): recommend `partner` (top-3 standalone, then 2 couples on partners' anchor nodes); the declared top 5 is the worst rule

**Run.** Sample 863469 at fee83955: 392 files, 151,776 batches, 1,080 accel jobs and 112 G8 states. The full run is
863476 at 94e2b1be (18 min, MaxRSS 5.6 GB, 2,464 replays). Rows are in `/share/nikola.lukic/slate_bar/recall_94e2b1be/`
(`rows.jsonl`, `rows.read.json`, `rows.read.txt`).

**Checks.**
- Errors: 8 rows, which is 4 accel batches × 2 rules (heavy n=3, heavy n=6 ×2, moderate n=6). Each snapshot has an
  unplaced partner with no placement time, so the replay refuses it. They are excluded, not imputed.
- Determinism: 80/80 repeated replays gave identical plans.
- No pick fell outside the reachable set (`rank0_picks` 0).
- `pick_not_final` is 0: in the 560 replays with more than one decision, the later decisions never changed a batch task's pick.
- No accel snapshot holds a 1-task batch, so batch sizes run 2–10. Pooled numbers are weighted by the 151,776-batch population.

**The declared slate misprices cold starts.** The declared top 5 and `_pg_standalone`'s top 5 disagree on 3,956 of 15,110 tasks.
- Of the 11,604 couples in those symmetric differences, 6,204 sit on an **initialized** platform. On each of them the declared
  cost is lower than CD's by **exactly the table cold start** (6,204 of 6,204).
- `declared_slate.standalone_cost` charges cold = 0 whenever `initialized` is true. `_pg_standalone` charges
  `incoming_cold_start_time`, which is the full table value on those platforms.
- The remaining 5,400 couples have equal costs on both sides: they are the couples displaced by the mispriced ones.
- That cost (`declared_pruning_v1`) is the one the GNN corpus and the GNN's served slate use.

**Pooled, population-weighted batch recall** (% of batches with every undecided task's pick inside the slate):

| source · rung · rule | top5 (declared) | top5cd | union | partner | top6 (plan ratio, geometric mean) |
|---|---|---|---|---|---|
| accel heavy · CD | 84.4 | 91.9 | 94.4 | 93.8 | 96.3 (×1.46) |
| accel heavy · cd_exactS | 87.0 | 94.6 | 95.5 | 95.2 | 97.6 (×1.46) |
| accel moderate · CD | 89.4 | 94.5 | 95.7 | 96.6 | 97.5 (×1.25) |
| accel moderate · cd_exactS | 88.4 | 96.0 | 94.7 | 96.1 | 98.2 (×1.25) |
| G8 heavy · CD | 80.4 | 82.1 | 92.9 | 89.3 | 91.1 (×2.22) |
| G8 heavy · cd_exactS | 78.6 | 87.5 | 89.3 | 91.1 | 94.6 (×2.22) |
| G8 moderate · CD | 80.4 | 91.1 | 94.6 | 96.4 | 94.6 (×1.71) |
| G8 moderate · cd_exactS | 80.4 | 91.1 | 94.6 | 94.6 | 92.9 (×1.71) |

**Pooled task recall**, cd_exactS:
- accel heavy: top5 95.6, top5cd 98.8, union 98.4, partner 98.8, top6 99.5 %;
- accel moderate: top5 96.4, top5cd 98.7, union 98.4, partner 99.1, top6 99.7 %.

All plan ratios other than top6 are 1.0.

**Per batch size** (cd_exactS batch recall, accel heavy, n = 2…10; per-size tables for both rules and rungs are in `rows.read.json`):

| rule | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| top5 | 96.7 | 91.5 | 90.0 | 86.7 | 74.1 | 71.7 | 58.3 | 65.0 | 56.7 |
| top5cd | 98.3 | 100 | 93.3 | 91.7 | 93.1 | 88.3 | 76.7 | 73.3 | 75.0 |
| union | 96.7 | 100 | 98.3 | 98.3 | 82.8 | 86.7 | 75.0 | 81.7 | 81.7 |
| partner | 98.3 | 100 | 96.7 | 93.3 | 89.7 | 88.3 | 76.7 | 76.7 | 76.7 |
| top6 | 100 | 100 | 96.7 | 96.7 | 98.3 | 91.7 | 90.0 | 81.7 | 88.3 |

- The top6 plan ratio at n = 8–10 is ×2.1–2.6.
- Above the cap (n ≥ 8, where 65–83 % of heavy batches fall back to expansion), every rule loses 15–40 pp of batch recall.
- On tasks with more than 5 candidates, top5 task recall at heavy n ≥ 8 is 74–85 %.

**Recommendation (the fixed rule): `partner`.**
- **Fixed-rule score:** partner has the highest G8 cd_exactS batch recall among the ratio-1 rules, averaged over rungs.
  - partner 92.9 %, union 92.0, top5cd 89.3, declared top5 79.5.
- **Ties:** partner beats union by **one batch of 112**, a tie. On the accel corpus they split by rung: heavy union +0.3 pp,
  moderate partner +1.4 pp.
- **top6 not chosen:** top6 (93.8 %) does not clear its +5 pp bar.
- **Composition of the gain over the declared slate (+13.4 pp at G8, cd_exactS):**
  - about **+9.8 pp is the cold-start cost fix** (declared → top5cd);
  - about **+3.6 pp is the composition**: 2 places traded from standalone ranks 4–5 to co-location candidates.
- **Practical reading.** Fix `declared_slate.standalone_cost` to charge `incoming_cold_start_time` (as `_pg_standalone` does),
  then spend 2 of the 5 places on partners' anchor nodes. The plan count is unchanged.
  - Union is the equally good alternative, but it needs an exchange model in the slate builder; partner needs none.
- **A corpus change, not a free swap.** Any change to the declared slate changes what the GNN corpus and the checkpoint's sidecar
  (`candidate_slate`) were built on, so it requires a re-labelled corpus.

**Against the prediction.**

| prediction | outcome | verdict |
|---|---|---|
| CD task recall under top5 93–97 % | 94.8 / 96.6 | **held** |
| CD task recall on > 5-candidate tasks 80–90 % | heavy 74–96 by size | **held, low end** |
| top5 batch recall > 90 % at n ≤ 3 | heavy n=3: CD 86.4 | **missed** |
| top5 batch recall 50–70 % at G=8 | 79–80 | **missed (better)** |
| cd_exactS ≥ CD under every rule | not on G8 heavy top5 or accel moderate union | **missed** |
| cd_exactS task recall 96–99 % | top5 95.6–96.4, top5cd ≈ 98.8 | **mixed** |
| top6 +1–3 pp task recall over top5 | +3.3–3.9 over declared, +0.7–1.0 over top5cd | **mixed** |
| union 0–3 pp over top5 | +8–14 pp batch (the cost defect, unforeseen) | **missed** |
| partner within 1 pp of union | G8 0.9 pp | **held** |

### Expert rulings on the recall audit (2026-10-12)

- **Declared rule for future capped builds: `declared_pruning_v2`.**
  - It is the partner rule with the corrected cold cost: the cost is the cold start the simulation applies, per ruling 1 below.
  - The existing corpus is **not** rebuilt with it. S7's uncapped extension (`accel_uncap`, 4f41b771, running) removes the
    100k cap for about 98 % of datasets (the expert's figure), which makes the slate question moot there.
  - v2 applies only to a future build that keeps a cap.
- **The live arm keeps `declared_pruning_v1`** (S5's part). It is the matched-information bar, defect included, because that is
  the slate the GNN was trained and served on.
  - One descriptive arm, if cheap: cd_exactS restricted to top5cd (CD's own standalone cost).
- **What the cap costs on big batches.** Read on accel heavy, cd_exactS, batch recall of the declared top 5 by batch size:

  | n | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
  |---|---|---|---|---|---|---|---|---|---|
  | recall % | 96.7 | 91.5 | 90.0 | 86.7 | 74.1 | 71.7 | 58.3 | 65.0 | 56.7 |

  - The plan cap binds from n = 8 (65–83 % of those heavy batches fall back to expansion).
  - Even CD's own top 5 recalls only 73–77 % of batches at n = 8–10, and top6 (×2.1–2.6 plans) recalls 82–90 %.
  - A slate-restricted search on a 9–10-task batch therefore cannot reproduce the hand rule's plan in a quarter to two fifths
    of batches, before any learning enters.
