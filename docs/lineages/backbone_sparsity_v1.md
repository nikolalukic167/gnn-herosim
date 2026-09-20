# backbone_sparsity_v1 — distance as the lever: a sparser client-server graph and a slower fabric

**Status:** `REGISTERED` (2026-09-20) — bars, reader, 6 tests, mint script, sbatch and the
expectation below committed before any levered arm ran. Shares its apparatus with
`burst_groups_v1` and `payload_scale_v1` (`scripts_cosim/env_lever_v1_*.py`,
`scripts_cosim/datalab/env_lever_v1*.sbatch`). **Registered as the weakest of the three levers**
and scoped accordingly.

**Parents:** [`payload_scale_v1`](payload_scale_v1.md) (the same ratio through payload),
[`peer_greedy_live_v1`](peer_greedy_live_v1.md), [`unsaturated_edge_v1`](unsaturated_edge_v1.md).

**Question.** The exchange cost is `hops × bytes / bottleneck + latency`; payload scales it
through bytes, and distance scales it through hops and bandwidth — the genuinely pairwise part.
Two one-field variants of the study cells: **p04** (client-server connection probability
0.6 → 0.4: fewer reachable servers per client, longer routes) and **bw250** (backbone
1000 → 250 Mbps: every remote exchange 4× slower). Does peer-aware placement pay under either?

**Caveats registered in advance.** (1) `p04` regenerates the topology from the same seed with a
different probability, so it is a *different* topology with the same label, and reachability
falls — more starved-client hangs are expected; the lever's own screen (L0) decides what is
readable, and **unknown is not a pass**. (2) `bw250` acts like payload ×4 on the transfer term
only (latency unchanged), so it is expected to read between k = 1 and k = 10 of `payload_scale_v1`.
(3) The learned arms' bipartite features carry link latency/bandwidth terms that move here;
`gnnedge0` zeroes those attributes, so its serving is unaffected in form, and that is stated.

**Arms** per variant at C40: reactive (screen), `random_network`, `peer_greedy_network`,
`peer_greedy_network_batch`, `be1670_gnnedge0` × 16. 352 runs per variant.

**Bars** (chain's; L0–L3 exactly as in `payload_scale_v1`, per variant).

**Registered expectation (signed 2026-09-20).** `bw250`: L2 BEATS 70 %; L1 BEATS 35 %, NOT-SEP
35 %; L3 RULE-FASTER 45 %. `p04`: L0 DESIGN-READY 60 % (more hanging cells); L2 BEATS 50 %,
NOT-SEP 35 %; L1 NOT-SEP 45 %, REACTIVE-FASTER 35 %; L3 MATCHES 40 %.

**Consequences, signed in advance.**
- Both variants read like the payload sweep (rule pays, arm does not amortise its wait): the
  two levers are one lever — the exchange-vs-queue ratio — and the paper says so once.
- `p04` L0 fails: sparsity is a reachability lever before it is a distance lever at 6 servers;
  recorded and not read.
- Anything the arm wins here that it does not win under payload is a distance-specific effect
  worth its own lineage; nothing here is quoted as one without that.

**Cost.** 2 × (48 + 304) = 704 runs (~1.2 h). Results under `results/el_v1/p04/` and
`results/el_v1/bw250/`; selections `simulation_data/env_lever_v1/selected_{p04,bw250}.json`.

**Datasets.** None; nothing trained.

## Record (newest first)

- 2026-09-20 — Registered.
