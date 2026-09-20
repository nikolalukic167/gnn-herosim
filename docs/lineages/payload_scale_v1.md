# payload_scale_v1 — the exchange-vs-queue ratio: where co-location starts to pay

**Status:** `CLOSED` (2026-09-20) — **`CO-LOCATION-PAYS-FROM-x1` (the rule) ·
`CO-LOCATION-PAYS-AT-NO-MEASURED-SCALE` (`gnnedge0`) · scales above ×1 are a load lever at this
rate.** Closed on live gates (rule 6): 352 arms at ×0.1 plus the ×1 study; ×3 and ×10 screened
and found unreadable (reactive saturated on every cell). Registered 2026-09-20; bars, reader,
6 tests and the expectation committed before any levered arm ran; Amendment 1 (×3) signed after
the ×10 screen. **The registered expectation was right at ×0.1 (L2 NOT-SEP, L1 REACTIVE-FASTER,
L3 RULE-FASTER) and the ×10 screen fell on the 40 % side it named.** Shares its apparatus with
`burst_groups_v1` and `backbone_sparsity_v1` (`scripts_cosim/env_lever_v1_*.py`,
`scripts_cosim/datalab/env_lever_v1*.sbatch`).

**Outcome.** The payload scale moves two things at once at 0.46 arrivals/s on 6 servers: the
exchange-vs-queue ratio and the load. **Upward it is a load lever first**: at ×3 (600 MB) reactive
Knative's queue share is 0.955–0.985 on every finished candidate cell but one topology (0.74), and
at ×10 (2 GB) 0.98–0.996 on all 34 that finish — no crossed design exists at either scale, so the
crossing point above 200 MB cannot be read at this rate (a matched-load design would confound
payload with rate; not this lineage). **Downward the answer is clean**: at ×0.1 (20 MB; 16
environments on 9001/9003/9005/9101, reactive median 9.27 s per task, exchange 0.66 s per task)
the immediate rule **ties** Knative (L2 −0.36 %, `NOT-SEPARATED`, 14/16 ahead by a hair), random
ties (+1.20 %), and every batching arm pays its wait unamortised: `gnnedge0` **+16.37 % (0/16)**,
the batched rule +14.79 % (0/16), both carrying 7.2 s of scheduler wait against a 9.3 s task; L3
`RULE-FASTER-THAN-GRAPH-ARM` +16.91 % (16/16). **Regime (L6):** peer-aware placement pays from
**×1** for the rule (×0.1 tie, ×1 −13.0 %, above unreadable) and at **no measured scale** for
`gnnedge0` (+16.4 % at ×0.1, +6.7 % at ×1). Per task at ×0.1 (medians): Knative 3.31 s queue /
0.66 exchange / **5.19 rendezvous** — with almost no exchange, tasks reach the rendezvous sooner
and wait longer for partners, which batching cuts to 1.3 s at the price of the 7.2 s wait.

**Carry.** The 200 MB scale of the synthetic peer augmentation is at the top of what this rate
can serve, not in the middle of a regime: the paper's claim is "peer-aware placement pays at the
200 MB scale and not at 20 MB; larger payloads saturate a 6-server cluster at this rate". The
learned arms' peer-edge attribute is out of training support at ×0.1 for half the pairs
(disclosed); it does not matter for the verdict because the wait dominates.

**Parents:** [`batch_window_edge_v1`](batch_window_edge_v1.md) (co-location is worth −1.05 s of
exchange per task at 200 MB), [`peer_greedy_live_v1`](peer_greedy_live_v1.md) (the rule arms; the
k = 1 point of this sweep), [`unsaturated_edge_v1`](unsaturated_edge_v1.md).

**Question.** Every peer effect in the record is at one payload scale: 200 MB × 10^U(−1, 1)
per pair over a 1 Gbps backbone, ~5.4 s of exchange per task. That scale is an assumption of
the synthetic peer augmentation, not a measurement. Scaling every payload by **k ∈ {0.1, 10}**
(20 MB and 2 GB scales; k = 1 is the study itself) moves the exchange-vs-queue ratio directly:
at 2 GB exchange is tens of seconds per task and co-location should dwarf concentration; at
20 MB nothing any scheduler does about peers should matter. The deliverable is the **crossing
point** — the smallest scale at which a peer-aware placement beats reactive — which turns the
assumption into a regime statement.

**The lever** (`env_lever_v1_mint.py pk0.1 / pk10`). Every `peer_exchange` payload is multiplied
by k; nothing else changes. Each scale gets its **own admissibility screen** (L0; unknown is not
a pass) and its own 4 study topologies. **Support caveat, registered in advance:** the learned
arms were trained on pairs of 20 MB–2 GB, so at k = 10 most served pairs (200 MB–20 GB) lie
outside the `PeerConv` edge attribute's training support (log1p(bytes/1e6) 5.3–9.9 vs 3.0–7.6)
and at k = 0.1 (2–200 MB) half do; `gnnedge0` zeroes only its *bipartite* attributes, so it is
served out of support on the peer edge either way and the read says so. `peeronly` / `gnn` are
not served. Knative and the rule do not care.

**Arms** per scale at C40: reactive (screen), `random_network`, `peer_greedy_network`,
`peer_greedy_network_batch`, `be1670_gnnedge0` × 16. 352 runs per scale.

**Bars** (chain's; L0–L3 as in `burst_groups_v1`, per scale) plus the regime read:

| read | what | fires as |
|---|---|---|
| L0 | the scale's screen | `DESIGN-READY` / `TOO-FEW-UNSATURATED-ENVIRONMENTS` |
| L1 | `gnnedge0` vs reactive (checkpoint) | `ARM-BEATS-REACTIVE-UNDER-THE-LEVER` / … |
| **L2** | `peer_greedy_network` vs reactive (env) | `RULE-BEATS-REACTIVE-UNDER-THE-LEVER` / … |
| L3 | `gnnedge0` vs the rule (checkpoint) | `GRAPH-ARM-FASTER-THAN-RULE` / `RULE-FASTER-THAN-GRAPH-ARM` / `GRAPH-ARM-MATCHES-RULE` |
| **L6** | over k ∈ {0.1, 1, 10}: the smallest k at which the BEATS verdict fires and keeps firing at every larger k | `CO-LOCATION-PAYS-FROM-x<k>` / `CO-LOCATION-PAYS-AT-NO-MEASURED-SCALE` — read for the rule (L2) and for `gnnedge0` (L1) |

**Registered expectation (signed 2026-09-20).** k = 10: **L2 BEATS 85 %**; L1 BEATS 50 %
(the 7 s wait is fixed while the exchange it buys grows tenfold), NOT-SEP 30 %; L3 RULE-FASTER
45 %. k = 0.1: L2 NOT-SEP 75 %; **L1 REACTIVE-FASTER 80 %** (the wait remains, the gain vanishes);
L3 RULE-FASTER 70 %. Regime: the rule pays from **x1** (55 %; from x10 35 %); `gnnedge0` from x10
(45 %) or at no measured scale (45 %). L0 at k = 10: DESIGN-READY 60 % — tens of seconds of
exchange per task saturates reactive on more cells.

**Consequences, signed in advance.**
- Rule regime from x1 and arm regime from x10: the paper's claim is scoped — "peer-aware
  placement pays from the 200 MB scale; the learned arm needs 2 GB to amortise its wait".
- Arm regime at no measured scale: the wait is never amortised within the measured range, and
  the model-class claim carries no operating point in this environment.
- L0 fails at k = 10: the scale is a load lever at 6 servers; recorded, not read.

**Cost.** 2 × (48 + 304) = 704 runs (~1.2 h). Results under `results/el_v1/pk0.1/` and
`results/el_v1/pk10/`; selections `simulation_data/env_lever_v1/selected_pk*.json`; regime read
`env_lever_v1_gate_read.py regime`.

**Datasets.** None; nothing trained.

**Amendment 1 (signed 2026-09-20, after L0 at k = 10 and before any study arm at any scale).**
At k = 10 reactive Knative is saturated on **every** candidate cell (queue share 0.98–0.996 on all
34 cells that finish; 14 hang): at 2 GB per pair the exchange alone (~50 s per task) exceeds the
6-server capacity at 0.46 arrivals/s, so the scale is a load lever before it is a ratio lever,
exactly as the registration's last consequence says. **This amendment adds k = 3 (600 MB scale)**
as an intermediate point with the same screen, the same arms and the same reads, so that the
regime read has a measured scale between 1 and 10; the k = 10 point is carried as "no design at
this load" and is excluded from the monotone check (a scale with no design is not a
measurement). The registered bars are unchanged. Cost +352 runs.

## Record (newest first)

- 2026-09-20 — **CLOSED.** ×0.1 study (jobs 793677 screen; 793853 / 793985 / 794062-class study
  blocks, 304 arms, all completed): L1 `gnnedge0` +16.37 % (0/16, p = 0.0004); L2 immediate rule
  −0.36 % NOT-SEP (14/16, p = 0.0052); batched rule +14.79 % (0/16); random +1.20 % NOT-SEP; L3
  +16.91 % RULE-FASTER (16/16). Decomposition (medians, s): Knative 0 / 3.31 / 0.66 / 5.19 / 9.27;
  rule 0 / 3.56 / 0.55 / 5.02 / 9.25; batched rule 7.18 / 1.51 / 0.54 / 1.32 / 10.61; `gnnedge0`
  7.20 / 1.63 / 0.59 / 1.30 / 10.77. Regime (`env_lever_v1_gate_read.py regime`,
  `simulation_data/env_lever_v1/read_regime.json`): rule `CO-LOCATION-PAYS-FROM-x1`, `gnnedge0`
  `CO-LOCATION-PAYS-AT-NO-MEASURED-SCALE`; ×3 and ×10 carried as no design.

- 2026-09-20 — **L0 at k = 3: `TOO-FEW-UNSATURATED-ENVIRONMENTS`** (job 793506, 34 completed / 14
  cancelled at 20 min): reactive queue share 0.955–0.985 on every finished cell except 9106
  w1–w3 (0.74). At 0.46 arrivals/s on 6 servers the exchange term alone saturates reactive from
  the 600 MB scale upward; **every payload scale above 1 is a load lever at this rate**, and a
  readable point above k = 1 needs a matched-load design (a lower arrival rate chosen to hold
  reactive's queue share, which confounds payload with rate and is not this lineage). Table in
  `simulation_data/env_lever_v1/selected_pk3.json`. The regime read composes k ∈ {0.1, 1} only.

- 2026-09-20 — **L0 at k = 10: `TOO-FEW-UNSATURATED-ENVIRONMENTS`** (job 793367, 34 completed /
  14 cancelled at 20 min; every finished cell at share ≥ 0.98). Table in
  `simulation_data/env_lever_v1/selected_pk10.json`. Amendment 1 signed (above): k = 3 added.

- 2026-09-20 — Registered.
