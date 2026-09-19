# Two queue-feature contracts coexist, and the broken one is kept bit-exact

Queue depth reaches the models as a normalized depth (platform dim 7) and a usage ratio
(dim 13). The original formulas — now `legacy_v0` — cap the dim-7 divisor at 100, which
never binds in training (per-snapshot p90 depth 26–70) and always binds live (chosen-platform
depth ~14.5k), so deployed models consumed queue inputs 100–300x outside their training
manifold and stopped ranking queues at all. `scale_invariant_v1` fixes this: the dim-7
divisor is uncapped so a uniformly deeper cluster yields identical features, and dim 13 is
log1p-compressed so genuine overload stays ordered. We kept `legacy_v0` unchanged rather
than migrating, because a checkpoint trained under one contract computes different logits
under the other, and correcting the formula in place would silently invalidate every
existing checkpoint instead of loudly refusing it.

## Consequences

The contract is part of a checkpoint's identity, declared in its `.contract.json` sidecar and
enforced by `require_matching_queue_feature_contract()` at inference. Every pre-2026-08-13
checkpoint (873/v5.5, regime B distill, ect_pull) must be served under `legacy_v0`
**forever** — it cannot be fixed, only retrained. New training uses `scale_invariant_v1`.

Uncapping dim 7 is a no-op on the `contention_v2` corpus (its p90 divisors are all below the
old cap), so on that corpus v1 differs from v0 through dim 13 alone.
