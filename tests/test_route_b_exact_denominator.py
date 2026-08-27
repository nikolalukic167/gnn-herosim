"""route_b — R_exact's denominator must not be censored by the GREEDY decoder.

score_corpus used to summarize every statistic over
`[not no_feasible_rows and not greedy_stuck]`. But R_exact is a perfect-decoder bound and
the greedy decoder has nothing to do with it: a greedy dead-end is not a reason to delete a
perfectly valid R_exact.

That censoring was not neutral. On the H0 separable control the (n_feasible, greedy_stuck)
histogram is exactly {(9, False): 102, (16, True): 101, (16, False): 1} — `greedy_stuck` is
PERFECTLY confounded with the replica-config arm, so one whole cell of the 2x2x3x17 design
was silently dropped and the published frac_gt_1pct was really "over the 9-feasible-row arm
only" (0.1553 vs 0.0784 over all 204).

Teeth: test_greedy_stuck_dataset_still_contributes_to_r_exact reads n == 1 before the fix.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts_cosim"))

from score_route_b_contention import (  # noqa: E402
    Dataset,
    greedy_masked_plan,
    min_marginals,
    score_corpus,
    score_dataset,
)

TASK_TYPES = {
    "rigA": {"memoryRequirements": {"rigCpu": 1.0}},
    "rigB": {"memoryRequirements": {"rigCpu": 1.0}},
}


def write_ds(corpus: Path, name: str, replica_placements: dict, rows: list) -> Path:
    ds = corpus / name
    (ds / "placements").mkdir(parents=True)
    (ds / "infrastructure.json").write_text(
        json.dumps({"replica_placements": replica_placements}))
    (ds / "workload.json").write_text(json.dumps({"events": [
        {"application": {"dag": {"rigA": []}}, "node_name": "client_node0"},
        {"application": {"dag": {"rigB": []}}, "node_name": "client_node0"},
    ]}))
    with open(ds / "placements" / "placements.jsonl", "w") as fh:
        for plan, rtt in rows:
            fh.write(json.dumps({
                "placement_plan": {str(k): list(v) for k, v in plan.items()},
                "rtt": rtt,
            }) + "\n")
    (ds / "placement_metadata.json").write_text(json.dumps({
        "num_placements": len(rows), "rows_written": len(rows),
        "worker_failed": 0, "timed_out": 0, "sweep_complete": True}))
    return ds


# Two nodes, one platform each per task type. Greedy walks tasks in ascending
# min-marginal order and never backtracks, so it can strand the second task.
REPLICAS = {
    "rigA": [
        {"node_name": "n0", "platform_id": 301, "platform_type": "rigCpu"},
        {"node_name": "n1", "platform_id": 302, "platform_type": "rigCpu"},
    ],
    "rigB": [
        {"node_name": "n0", "platform_id": 311, "platform_type": "rigCpu"},
    ],
}
A0, A1, B0 = (100, 301), (101, 302), (100, 311)


def stuck_corpus(tmp_path: Path) -> Path:
    """ds_00000 scores normally; ds_00001 strands the greedy but has a real R_exact.

    In ds_00001, rigB exists ONLY on n0. Task 0's cheapest option is also n0, and with a
    cap admitting one task per node the greedy commits it there and then finds task 1 has
    nowhere to go -> stuck. The exhaustive decode is unaffected: (A1, B0) is feasible.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    # Plain dataset: both tasks placeable, surrogate ranks the optimum first -> r_exact 0.
    write_ds(corpus, "ds_00000", REPLICAS, [
        ({0: A0, 1: B0}, 30.0),
        ({0: A1, 1: B0}, 10.0),
    ])
    # Greedy-stuck dataset: task 0 prefers n0 (10 < 12), taking rigB's only slot.
    # Feasible plans under a one-per-node cap: only (A1, B0) = 25.0 -> that IS the
    # constrained optimum, so r_exact is 0 but the row must still COUNT.
    write_ds(corpus, "ds_00001", REPLICAS, [
        ({0: A0, 1: B0}, 10.0),
        ({0: A1, 1: B0}, 25.0),
    ])
    return corpus


def report(tmp_path: Path, alpha=1.0) -> dict:
    corpus = stuck_corpus(tmp_path)
    rep = score_corpus(corpus, TASK_TYPES, "rtt", [alpha])
    return rep["per_alpha"][str(alpha)]


def test_the_rig_actually_strands_the_greedy(tmp_path):
    """Guard the premise: if ds_00001 stops being stuck, every test below is vacuous."""
    corpus = stuck_corpus(tmp_path)
    ds = Dataset(corpus / "ds_00001", TASK_TYPES, "rtt")
    caps = ds.node_caps(1.0, cap_mode="alpha_max")
    assert greedy_masked_plan(ds, min_marginals(ds.rows), caps) is None

    out = score_dataset(ds, alpha=1.0)
    assert out.get("greedy_stuck") is True
    # ...and it still has a perfectly well-defined exact statistic.
    assert "r_exact_pct" in out
    assert not out.get("no_feasible_rows")


def test_greedy_stuck_dataset_still_contributes_to_r_exact(tmp_path):
    """THE TEETH. Reads n == 1 before the denominator fix."""
    per_alpha = report(tmp_path)
    assert per_alpha["r_exact"]["n"] == 2
    assert per_alpha["n_exact_scored"] == 2


def test_greedy_stuck_dataset_is_excluded_from_r_greedy(tmp_path):
    """The greedy statistic keeps its own, legitimate censoring."""
    per_alpha = report(tmp_path)
    assert per_alpha["r_greedy"]["n"] == 1
    assert per_alpha["n_greedy_scored"] == 1
    assert per_alpha["greedy_stuck"] == 1


def test_legacy_block_reproduces_the_prefix_denominator(tmp_path):
    """The pre-fix number must be recoverable from the SAME artifact, so a deviation is
    audited against one file rather than a commit message."""
    per_alpha = report(tmp_path)
    legacy = per_alpha["legacy_greedy_censored"]
    assert legacy["n"] == 1
    assert legacy["r_exact"]["n"] == 1


def test_stuck_confound_counter_is_reported(tmp_path):
    """greedy_stuck's confound with the design is made visible in every rung's artifact."""
    per_alpha = report(tmp_path)
    assert per_alpha["greedy_stuck_by_n_feasible"] == {"1": 1}


def test_no_feasible_rows_still_censors_both(tmp_path):
    """The OTHER censor is legitimate and must survive — guard against over-correcting.

    At a cap below every single demand nothing is feasible, so both denominators empty.
    """
    corpus = stuck_corpus(tmp_path)
    rep = score_corpus(corpus, TASK_TYPES, "rtt", [0.5])
    per_alpha = rep["per_alpha"]["0.5"]
    assert per_alpha["no_feasible_rows"] == 2
    assert per_alpha["n_exact_scored"] == 0
    assert per_alpha["n_greedy_scored"] == 0
    assert per_alpha["r_exact"] == {"n": 0}


def test_band_is_summarized_over_the_exact_denominator(tmp_path):
    per_alpha = report(tmp_path)
    for member in ("registered", "optimistic", "pessimistic", "mean_tied"):
        assert per_alpha["r_exact_band"][member]["n"] == 2
