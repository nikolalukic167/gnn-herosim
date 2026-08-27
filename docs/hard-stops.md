# Hard stops — falsified, do not revive without new evidence

> **Status:** `REFERENCE` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../LINEAGES.md)
>
> Things measured and closed. Each entry names the measurement that closed it, so reviving
> one is a decision made against evidence rather than by forgetting. Moved verbatim from
> `memory/memory.md` §2 (that file retired 2026-08-27; see git history); the `FALSIFIED` rows in
> [LINEAGES.md](../LINEAGES.md) are the lineage-level counterpart.

**tune link bandwidth or core count to rescue `link_contention_v1`** (null lever + full spectrum swept: n_core 2/4/12 all ≤0.35% mean regret) · **cite offline regret/acc as evidence a model will place better live** (Arm A: −54% offline regret → +3.5%/+9.9% *worse* live, `logit_tied_rate` rose) · **cite the 0.88× Kn `sparse_p35`/s42 figure** (irreproducible: current code gives **1.04×** on the same cell/seed/model; the arm that produced it set `GNN_DROP_NODE_EDGES=1`, implemented **nowhere** in the tree today) · **revive same-node edges without new physics** (Arm B falsified *while trained with them*) · RQ3 · **RQ3b** · **claim the GNN's live failure is "structural in the joint decode"** (falsified: LQB λ=1.5 no change 275.7M; `argmax_uniq` drove collisions to **0.000** and got *worse* 301.3M) · **cite ECT as a ceiling or distill teacher for Regime A** (0.98–1.13× Kn, and `contention_v2` is pull-free so `ect_pull`≡`ect`) · **run `ect_pull` with `ECT_PULL_DISTILL_DIR` on workload-125-225** (561,848 frames/cell ≈ 50GB; fills disk) · cite pre-`25732cf` Regime A tables as current · mix pre/post-serialization in one table · re-eval 873/v5.5 as if it transfers to serialized FilterStore · hub9 decode · claim pull-obs/cosim-retrain/`soft_combo_conc`/hard-CE-distill closed 125→31 · blind retrain scarce-warm 450 · reopen hub9/`total_rtt` primary · HeteroData/Set-Transformer · claim warm/busy v1 init helps FilterStore distill
