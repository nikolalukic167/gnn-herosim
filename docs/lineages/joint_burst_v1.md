# joint_burst_v1 — train on the decision that is served: whole peer groups, arriving together, from loaded states

**Status:** `REGISTERED` (2026-09-20) — bars, reader, 5 tests, the screen / capture / corpus /
training / gate scripts and the expectation below committed before any burst-trained checkpoint
exists. Nothing in this lineage is read from a cold-corpus checkpoint except as the J5 control.

**Parents:** [`burst_groups_v1`](burst_groups_v1.md) (under bursts the wait is gone and the cold
`gnnedge0` only ties Knative while the greedy reads −26 %), [`peer_greedy_live_v1`](peer_greedy_live_v1.md)
(the greedies that are this lineage's bars), [`peer_affinity_warm_v1`](peer_affinity_warm_v1.md)
(the snapshot → dataset machinery: `make_warm_corpus.py`, `live_snapshot_seed`, measured drain),
[`peer_only_v1`](peer_only_v1.md) (the `partial_state_v3` cache recipe with the measured-clock
drift label).

**Question.** Every checkpoint this programme has served was trained on cold generated states
(no standing load, one replica per type, queues drawn from Poisson(2)) with a label that is the
optimum over one batch from that state — and served on a loaded stream one arrival at a time
where the batch it was trained for never occurs. Under bursts the served decision **is** the
batch decision: one whole peer group, decided at once, from whatever the cluster looks like at
that moment. **Trained on exactly those states, with the brute-force optimum over the group on
the measured clock as its label, does a graph model beat the greedy that beat every cold
checkpoint — and Knative?**

**Design.**
- **Pool and screen (J0).** 28 study candidates (the 12 original seeds + 9109–9124) and 24
  training topologies (9201–9224), all 6 servers / 40 clients, burst workloads on all four
  windows. A study topology needs reactive admissible (share ≤ 0.80) **and** the batch path
  finishing on all four windows; the 4 lowest ids are the study (`select_burst_topologies`).
  Training and study seeds are disjoint by construction.
- **Capture.** The batched rule (`peer_greedy_network_batch`, the learned arms' stack with a
  rule in the decoder's seat) runs each training cell on w0 and w1 with `LIVE_AUDIT_SNAPSHOT_PATH`,
  stride 300 (every 30th group), ≤ 150 snapshots per run. A run that spins leaves no file and
  the cell is dropped; the corpus needs ≥ 16 training cells.
- **Corpus.** `make_warm_corpus.py` on the aligned snapshots (batch = group, 10 consecutive ids),
  12 datasets per (cell, window) at ≤ 20,000 plans, candidate subsampling as in W1; the cluster's
  standing load replays as `live_snapshot_seed` with the measured drain. Held-out: the study
  topologies' own captures are **never** used; a held-out set of 34 datasets comes from 3
  training cells set aside (as `peer_affinity_v1` T1b's held-out). ≥ 192 training datasets.
- **Cache and training.** The `peer_only_v1` recipe verbatim: `prepare_graphs_cache
  --dag-partial-state --platform-feature-dim 14 --label-objective rtt_drift:1
  --label-arrival-rate 0.46 --label-drain-table <A1 drain table>`, `PARTIAL_STATE_CONTRACT=partial_state_v3`,
  near-RTT sidecar, the candidate-not-in-sweep guard, a split artifact of this corpus.
  Arms: `gnnedge0` (bipartite edge conv, mean, attrs zeroed — the standing best learner) and its
  MP-OFF twin, `experiments/joint_burst_v1_{gnnedge0,mpoff}.yaml` = the `bipartite_edge_v1`
  configs with the cache and split swapped; lr 2e-3, 300 epochs, patience 60, 16 seeds each,
  W&B project `gnn-peer-affinity-v1`.
- **Gate (rule 6).** The 16 study environments under bursts; every arm served identically
  (`masked_topo`, `GNN_BATCH_BY_PEER_GROUP=1`, cell window 16 s — under bursts the wait is
  ~0): burst-trained `gnnedge0` × 16, burst-trained `mpoff` × 16, cold `be1670_gnnedge0` × 16
  (control), the batched greedy, the immediate rule, random; reactive from the screen. 16 × 51
  = 816 arms.

**Bars** (`scripts_cosim/joint_burst_v1_read.py`; |median| ≥ 5 %, p < 0.05, signed-rank;
learned unit = checkpoint n = 16; rule-vs-learned by broadcasting the rule):

| read | what | fires as |
|---|---|---|
| J0 | design (screen) and corpus size | `DESIGN-READY` / `TOO-FEW-SERVABLE-ENVIRONMENTS`; `CORPUS-READY` / `CORPUS-TOO-SMALL` |
| **J1** | burst-trained `gnnedge0` vs the batched greedy, GRAPH FIRST (ckpt) | `GRAPH-ARM-BEATS-BATCHED-GREEDY` / `BATCHED-GREEDY-FASTER-THAN-GRAPH-ARM` / `NOT-SEPARATED` |
| J2 | vs the immediate rule (ckpt) | `GRAPH-ARM-BEATS-IMMEDIATE-RULE` / … |
| **J3** | vs reactive (ckpt) | `ARM-BEATS-REACTIVE-UNDER-BURSTS` / `REACTIVE-FASTER-UNDER-BURSTS` / `NOT-SEPARATED` |
| J4 | vs burst-trained `mpoff`, GRAPH FIRST (ckpt, paired on seed) | `GRAPH-FASTER-THAN-TWIN-ON-THE-SERVED-CORPUS` / `POINTWISE-FASTER…` / `NOT-SEPARATED` |
| J5 | burst-trained vs cold `gnnedge0`, paired on env and seed | `SERVED-DISTRIBUTION-TRAINING-HELPS` / `…-HURTS` / `NOT-SEPARATED` |
| **J6** | composite | `GNN-WIN` iff J1 and J3 fire; `GNN-BEATS-GREEDY-NOT-REACTIVE` if only J1; else `NO-GNN-WIN` |

**Registered expectation (signed 2026-09-20).** J0: DESIGN-READY 70 % (16 more candidates
against a screen that lost 9 of 12), CORPUS-READY 80 %. **J5 HELPS 70 %** — the served
distribution is the thing every earlier corpus lacked. J4 GRAPH-FASTER 60 % (the cold twin
contrast is −6.8 to −9.8 % and should survive). **J3 BEATS 50 %**, NOT-SEP 35 %. **J1: GRAPH-BEATS
25 %, NOT-SEP 35 %, GREEDY-FASTER 40 %** — the greedy is a strong bar (−26 % vs Knative, near the
B5 lookahead's offline optimum), and a supervised argmin over ≤ 20,000 subsampled plans is not
guaranteed to beat a greedy on the true cost. **J6 GNN-WIN 20 %.**

**Consequences, signed in advance.**
- J6 GNN-WIN: the first learned win in the programme that is a win over a rule with its own
  information; the paper's claim is "trained on the served decision, a graph model beats a
  physics greedy under fan-out arrivals", scoped to bursts and this cluster.
- J1 NOT-SEP or GREEDY-FASTER with J5 HELPS: the served distribution was necessary and not
  sufficient; the label (a one-batch optimum) or the model is the remaining gap, and the
  rollout label (`rollout_imitation_v1`) is the next lever.
- J5 does not fire: the served distribution was not the problem; stop building corpora.
- J0 fails on the screen: bursts are unservable on this cluster beyond a handful of
  topologies; recorded, the corpus is still built for the record but not gated.

**Cost.** Screen 224 + capture 48 arms (~1.5 h with hangs); corpus ~300 sweeps (~1 h at 48
workers); cache ~30 min; training 32 runs × ~45 min (CPU, 16 threads); gate 816 arms (~1 h).
Scripts: `scripts_cosim/datalab/joint_burst_v1_{mint,screen,capture,corpus,cache,train,gate}.sbatch`;
read: `scripts_cosim/joint_burst_v1_gate_read.py`; results `results/jb_v1/`.

**Datasets.** `simulation_data/gnn_datasets_joint_burst_v1_{train,heldout}` (registered in
`simulation_data/REGISTRY.json` when built).

## Record (newest first)

- 2026-09-20 — Registered.
