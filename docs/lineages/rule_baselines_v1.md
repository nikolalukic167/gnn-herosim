# rule_baselines_v1 — does the per-candidate-features GNN beat six standard hand placement rules?

**Status:** `CLOSED` (2026-10-05) — **BEATS-NAIVE-RULES / CO-LOCATION-RULES-FASTER**. Registered the same day; arms,
topologies, statistic and labels fixed before any run.

**Outcome.** `lf1gnn` beats random (−66 / −85 / −66 %), least-loaded (−19 / −37 / −15 %) and Decima's weighted-fair
heuristic (−15 / −21 / −12 %) at every rung, Holm-CONFIRMED, 18–19 / 19 topologies. Every rule with a co-location term
is faster: locality-first +6 / +78 / +15 %, self-predict +14 / +64 / +6 %, one-pass greedy +5 / +20 % (×5 −2 %,
NOT-SEPARATED). The co-location rules win on queue as well as exchange (×3 median queue 28–47 s vs the GNN's 62 s).
The same-input MLP beats only random and least-loaded (×2 / ×3) and ties Decima's rule.

**Question.** `local_features_v1`'s GNN (`lf1gnn`) beats its same-input MLP and the reactive Knative-style scheduler
on 19 unseen topologies. The literature on learned schedulers (Decima, Placeto, L2D, EP-NCO) compares against a
family of hand rules; this lineage measures `lf1gnn` against six of them on the same cells.

**Arms** (one run per cell; rules are deterministic given the cell):
- `random` — uniform over the task's candidate replicas (`random_network`);
- `drain` — least-loaded: Knative's candidate set scored by queue drain + cold + exec + latency in seconds, per
  arrival, no exchange term (`drain_greedy_network`);
- `locality` — locality-first: the one-pass batched greedy with its exchange term weighted ×100, so co-locating with
  placed partners dominates and queue drain only breaks ties (`peer_greedy_network_batch`, `HEROSIM_PG_EXCHANGE_SCALE=100`);
- `decima` — Decima's tuned weighted-fair heuristic (its baseline, not the Decima RL agent), α = +1 as tuned in
  `decima_rule_v1` (`decima_wfair_network`);
- `batched` — one-pass greedy / earliest-finish: the full seconds cost (queue drain + cold + exec + latency + exchange to
  placed partners) over the peer-group batch in id order (`peer_greedy_network_batch`);
- `selfpredict` — the per-arrival rule plus a price for unarrived partners (`selfpredict_bar_v1`).
- `lf1gnn` and `lf1mlp` (8 seeds each) are read from `local_features_v1`'s gate on the same cells.

**Cells.** `small_batch_confirm_v1`'s 19 topologies × grounded windows g0–g3 × {×2, ×3, ×5}; 1,368 rule runs.

**Statistic.** Per topology, the median paired % `lf1gnn` vs rule over (window, seed); median over the 19 topologies,
exact two-sided Wilcoxon. Family: 6 rules × 3 rungs = 18 tests, Holm across all 18. Labels per test: CONFIRMED
(median ≤ −5 %, Holm p < 0.05), DIRECTION-ONLY, REF-FASTER, NOT-SEPARATED. **No aggregate verdict; every label is
reported**, including rules that beat the GNN. Descriptive: `lf1mlp` vs each rule. A run that times out (2700 s) is
dropped by name and listed. (`scripts_cosim/rule_baselines_v1_read.py`.)

**Expectations (judgement).** Beats `random`, `drain`, `locality` and `decima` at every rung: 80 %. `batched`: coin
flip. `selfpredict`: likely faster than the GNN.

**Disclosures.** The topologies' fourth use; the arm list was chosen after `local_features_v1`'s read, partly by
expected outcome, which is why every label is reported. `locality`'s ×100 weight is this lineage's construction.
Decision time is not charged to any arm.

**Entry points.** `datalab/rule_baselines_v1_gate.sbatch`, `fresh_topo_burst_v1_gate.py rb1`.

## Record

- 2026-10-05 — **Gate 827974 read** (1,359 / 1,368 runs; 9 timeouts dropped by name, the hangs seen in every arm).
  Holm over 18, `lf1gnn` vs rule (×2 / ×3 / ×5): random −66.0 / −85.2 / −66.4 % CONFIRMED; drain −18.8 / −37.0 / −15.3 %
  CONFIRMED; decima −15.2 / −20.5 / −11.6 % CONFIRMED; locality +6.5 / +78.4 / +14.9 % REF-FASTER; batched +5.3 / +20.5 %
  REF-FASTER, −2.0 % NOT-SEPARATED; selfpredict +13.6 / +63.9 / +5.6 % REF-FASTER. `lf1mlp` (descriptive): random
  −60 / −79 / −57 %, drain −8 / −11 / +4 %, decima −1 / +3 / +9 %, the co-location rules +19 to +181 %. Median latency /
  queue / exchange per task at ×3 (s): `lf1gnn` 65 / 62 / 3.68, locality 31 / 28 / 3.04, selfpredict 39 / 36 / 3.23,
  batched 50 / 47 / 3.47, decima 91 / 87 / 3.86, drain 111 / 107 / 4.03, random 454 / 449 / 4.53.
  Read: `rule_baselines_v1/rb1_read.json`.
- 2026-10-05 — Registered.
