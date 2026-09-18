#!/usr/bin/env python3
"""The decode-stats write guard (peer_only_v1 C6, 2026-09-18).

`record_queue_feature_discrimination` is documented as always-on instrumentation and runs on
every decoded batch. `executesimulation` gated writing the decode stats on `gnn_batches > 0`,
a counter only `record_decode_batch` touches -- and the masked_topo path every live gate runs
never calls it. The probe's output was therefore discarded in every gate run to date.
"""
from __future__ import annotations

from src.policy.gnn.seq_decode import (
    GnnDecodeRunStats,
    run_decode_stats_have_content,
)


def test_none_has_no_content():
    assert run_decode_stats_have_content(None) is False


def test_a_fresh_stats_object_has_no_content():
    assert run_decode_stats_have_content(GnnDecodeRunStats()) is False


def test_feature_probe_tasks_alone_counts_as_content():
    """The regression this file exists for: probe ran, gnn_batches never incremented."""
    stats = GnnDecodeRunStats()
    stats.feature_probe_tasks = 1
    assert stats.gnn_batches == 0
    assert run_decode_stats_have_content(stats) is True


def test_gnn_batches_alone_still_counts_as_content():
    stats = GnnDecodeRunStats()
    stats.gnn_batches = 1
    assert run_decode_stats_have_content(stats) is True


def test_the_hetero_variant_tolerates_a_stats_object_without_the_probe_counter():
    from src.policy.gnn_hetero.seq_decode import (
        run_decode_stats_have_content as hetero_have_content,
    )

    class NoProbeCounter:
        gnn_batches = 0

    assert hetero_have_content(NoProbeCounter()) is False
    NoProbeCounter.gnn_batches = 3
    assert hetero_have_content(NoProbeCounter()) is True
