# serving_speed_v1 — CLOSED

> **Status:** `CLOSED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-25 → 2026-08-25

**Outcome.** The episode cost was `Data.to()`, not the device — 174× per-call win from moving tensors only. CPU serving is slower than cuda; the cpu default exists for parity.

**Related:** [siv1_full_corpus](siv1_full_corpus.md)

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [serving_speed_v1 — the episode cost was `Data.to()`, not the device (2026-08-25)](#serving-speed-v1-the-episode-cost-was-data-to-not-the-device-2026-08-25)

---

### serving_speed_v1 — the episode cost was `Data.to()`, not the device (2026-08-25)

Groundwork for the `gnn_draw_study_v1` gate below: 240 gate episodes are worth speeding up
first, and the draws must be trained under a settled regime.

**What was measured.** One 30k-event episode, `cell01_p25_s9001` × `workload-150-100-30k`,
GNN policy, deployed checkpoint, same box:

| arm | wall |
|---|---:|
| before (whole-graph `Data.to()`, cuda) | **93 s** |
| after (tensor-only move, cuda) | **72 s** |
| after (tensor-only move, cpu) | **86 s** |

**The predicted lever was the wrong one.** `PROGRAM_VERDICT`'s profiling attributed ~26% of
the episode to `Data.to(device)` and concluded "a ~3–4× is free: run inference on CPU."
Half right. The 26% was real and is recovered (93 s → 72 s, 22.6%), but it was never a
*transfer* cost — it is PyG's `Data.to()` recursing through every stored attribute,
including the `queue_snapshot` and `task_logit_to_placement` dicts the scheduler attaches
before the move, which the forward pass never reads. Moving only tensors is **174× faster
per call** and the win is the same on CPU. Serving on CPU is *slower* than cuda here (86 s
vs 72 s); there is no 3–4×, on either device.

**Changes.** `move_graph_tensors_` in `src/policy/gnn/scheduler.py` replaces `graph.to()`
at all three decode call sites (the field list comes from `verify_venue_parity.py`'s
`GRAPH_TENSOR_FIELDS`, which had already worked this out for the fixture path).
`resolve_serving_device()` in `executesimulation.py` reads `HEROSIM_GNN_DEVICE`
(`cpu` default | `cuda` | `auto` = old behavior), and the resolved device is now stamped
into `run_provenance` — `env_fingerprint` previously recorded only `cuda_available`, which
describes the box, not what served. Not added to `STRICT_KEYS`, so existing fingerprints
stand.

**Why cpu is the default given cuda is faster.** For parity, not speed. `cuda` is the only
axis `PARITY.md` finds that moves GNN logits at all (1.9e-5), and it is visible end to end:
the same cell's `total_rtt` differs by **4.6e-6 relative** between the two devices. A cpu
default makes a local run and a datalab `CPU-amd` gate resolve to the same device. The cost
is ~19% on a GPU box and **zero on the partition every gate actually runs on**. Set
`HEROSIM_GNN_DEVICE=auto` to restore the old behavior.

**Status: ACTIVE.** 366 tests pass, `test_venue_parity` included — no fixture re-baseline.

---
