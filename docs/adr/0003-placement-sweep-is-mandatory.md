# Every snapshot carries its full placement sweep, not just the optimum

Co-simulation brute-forces every valid placement plan for a snapshot, and we persist all of
them with their resulting `total_rtt` (`placements/placements.jsonl`) rather than only the
winner (`best.json`). The sweep is the expensive part and it is already computed; discarding
it to save disk would throw away the only artifact that supports near-RTT and RTT-hash
training, which need the *shape* of the RTT surface over plans, not a single argmin. A
snapshot without a sweep is incomplete, not merely unlabelled.

## Consequences

`--resume` must never treat `best.json` alone as a completed snapshot — doing so produces a
collection that looks finished and cannot train the models it was generated for. It also
means the separability diagnostics (additive R², collision covariates, argmin-regret) are
possible at all: every one of them fits over the full sweep, so the question "does this
corpus contain coupling a GNN could exploit?" is answerable retrospectively on data already
on disk.
