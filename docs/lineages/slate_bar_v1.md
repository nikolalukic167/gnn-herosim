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
