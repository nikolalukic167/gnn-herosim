"""Serving speedups that leave every feature and logit unchanged (rp/gnn-serve-fast; a post-interim change, not a registered arm).

Two independent env flags, each default off, so with both off serving is byte-unchanged:

    SERVEFAST_BUILD=1    run-invariant memoisation in the live graph build (target concurrency per platform type, the four-type
                         replica columns, shared fate, the same-node edge index, the node-disk hit per node and type) and
                         torch.inference_mode around the decode. Bit-identical by construction: the same arithmetic, reused.
    SERVEFAST_COLUMNS=1  partial_state_columns keeps its committed-set aggregates across the steps of one decode instead of
                         rescanning the committed set at every step. The float summation order of every aggregate is preserved.

Rows-only scoring at serving (scoring just the stepped task's edge rows) was measured and dropped: it fails decision identity on near-ties
(8/8 cells), so it is not wired into serving; see the uncut_groups_s0_v1 Record entry of 2026-10-15.

The names do not start with HEROSIM_ / GNN_ / NEAR_RTT_, which the r1a gate wipes from its environment.
"""
from __future__ import annotations

import os


def _on(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def build_on() -> bool:
    return _on("SERVEFAST_BUILD")


def columns_on() -> bool:
    return _on("SERVEFAST_COLUMNS")

