# selfpredict_burst_v1 — does gnnedge0's burst-seat win survive the self-predict rule?

**Status:** `REGISTERED` (2026-09-24). Every bar below was signed before its data. Triggered by
`selfpredict_bar_v1` closing `BAR=SELFPREDICT` on the unburst rungs, where no learned arm was in the gate.

**Outcome.** Not yet run.

**Parents:** [`joint_burst_v2`](joint_burst_v2.md) (uncapped `gnnedge0` beats the 1-pass greedy in the
burst seat −11.9 %, 13/13, loses to the CD greedy +12.5 %),
[`selfpredict_bar_v1`](selfpredict_bar_v1.md) (the self-predict rule is the bar on the unburst rungs),
[`lookahead_mp_v1`](lookahead_mp_v1.md) (where the rule comes from).

## The question

`joint_burst_v2`'s win is the only learned win over a rule with the model's information in the
programme, and it was measured against the 1-pass greedy. The self-predict rule
(`peer_greedy_selfpredict_network`: the rule plus a price for each unarrived partner at the node the
rule would give it now) has never run under bursts. In a burst a whole peer group arrives at once, so
the first task of a group sees every partner as unarrived — exactly the case the self-predict term
prices and the 1-pass greedy prices at zero. **Does `gnnedge0` still beat the best hand rule in the
seat where it won?**

## Design

- **Environments:** the 16 `joint_burst_v1` study environments (`simulation_data/joint_burst_v1/selected.json`,
  C40, 6 servers, burst lever, 0.460393 arrivals/s before bursting) — the cells `joint_burst_v2` gated.
- **Arms, all fresh in one OUT_DIR at one commit (304 runs):** self-predict (per arrival), immediate rule,
  1-pass batched greedy, CD greedy (batched), `knative_network`, `random_network`, and jb2 `gnnedge0`
  UNCAPPED × the 13 checkpoints that exist (seeds 1–5, 7–10, 12, 14–16; 6/11/13 were never trained)
  × 16 environments, served exactly as `joint_burst_v2` served it.
- **Units and metric (`joint_burst_v2`'s):** `averageElapsedTime`; `gnnedge0` per checkpoint (median
  over environments of the paired %), a rule per environment, rule-vs-learned by broadcasting the rule
  over checkpoint seeds.

## Bars (signed 2026-09-24, before any data) — |median| ≥ 5 %, p < 0.05, two-sided signed-rank

| read | what | fires as |
|---|---|---|
| **S1** | `gnnedge0` vs self-predict, GRAPH FIRST, n = 13 checkpoints | `GRAPH-BEATS-SELFPREDICT` / `SELFPREDICT-FASTER` / `NOT-SEPARATED` |
| **S2** | self-predict vs CD greedy, n = 16 env | `SELFPREDICT-BEATS-CD` / `CD-FASTER` / `NOT-SEPARATED` |
| S3 | self-predict vs 1-pass batched greedy, n = 16 env | `SELFPREDICT-BEATS-1PASS` / `1PASS-FASTER` / `NOT-SEPARATED` |
| R | replication, disclosed: fresh `gnnedge0` vs 1-pass / vs CD against `joint_burst_v2`'s −11.90 % / +12.48 %, and every fresh cell against the jb2 gate's summary | reproduced / drifted (a drift is investigated before S1 is quoted) |
| — | disclosed: self-predict vs immediate rule, vs `knative_network`, vs random; CD vs 1-pass; per-task wait / queue / exchange / rendezvous | — |

**Verdict:** `GNN-BEATS-SELFPREDICT` if S1 fires graph-faster; `SELFPREDICT-AHEAD-OF-GNN` if it fires
the other way; else `GNN-TIES-SELFPREDICT`. **Burst-seat bar:** CD if S2 is `CD-FASTER`, self-predict
if S2 is `SELFPREDICT-BEATS-CD`, else CD~self-predict.

n = 13 on S1 is `joint_burst_v2`'s n, disclosed: all 13 on one side gives p = 2.4e−4, so p reads
sign-consistency. The self-predict rule runs per arrival, the learned arm and CD in the batched
peer-group seat; under bursts the batch wait is ~0 (`burst_groups_v1`) and the immediate and batched
1-pass rules differ by < 1 % in median elapsed (8.09 vs 8.04 s, `joint_burst_v2`), so S1 compares served
policies with the seat difference disclosed, not scoring functions.

**Registered expectation (2026-09-24, before data).** S1 `NOT-SEPARATED` 50 %, `GRAPH-BEATS-SELFPREDICT`
25 %, `SELFPREDICT-FASTER` 25 %; S2 `CD-FASTER` 70 % (the burst removes the wait that decided
`selfpredict_bar_v1`'s B3); S3 `SELFPREDICT-BEATS-1PASS` 55 %; R reproduces to the digit (the simulation
is deterministic and the served paths are unchanged).

**Consequences, signed in advance.**
- `GNN-BEATS-SELFPREDICT`: the learned arm beats the programme's best per-arrival rule in its own seat.
  It is still not a message-passing claim (`joint_burst_v2` K4: it ties its MP-OFF twin), and if S2 is
  `CD-FASTER` the burst seat's bar stays the CD search.
- `GNN-TIES-SELFPREDICT` or `SELFPREDICT-AHEAD-OF-GNN`: `joint_burst_v2`'s win does not clear the current
  bar in any seat; the standing answer says no learned arm beats the best hand rule anywhere.

## Entry points

- Gate: `scripts_cosim/datalab/selfpredict_burst_v1_gate.sbatch` (304 tasks, blocks of ≤ 48).
- Reader: `scripts_cosim/selfpredict_burst_v1_read.py` → `simulation_data/selfpredict_burst_v1/read.json`.

## Record (newest first)

- 2026-09-24 — **Registered.** Reader tested on a fixture built from the `joint_burst_v2` gate's own
  summaries with the immediate rule copied in as a stand-in self-predict arm (mechanics only): it
  reproduces K1 −11.90 %, K2 +12.48 %, K3 −34.67 % exactly and matches all 272 overlapping cells to 0.0 s;
  it fails loud on an empty or incomplete gate directory. The jb2 gate's reactive arm was
  `knative_network` (checked in its summaries), so K3 is against healthy reactive.
