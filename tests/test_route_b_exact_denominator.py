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
    complete_masked_plan,
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


def test_the_rig_actually_strands_the_FORWARD_ONLY_greedy(tmp_path):
    """Guard the premise: ds_00001 must still strand the pre-AMENDMENT-2 decoder, or
    every legacy assertion below is vacuous."""
    corpus = stuck_corpus(tmp_path)
    ds = Dataset(corpus / "ds_00001", TASK_TYPES, "rtt")
    caps = ds.node_caps(1.0, cap_mode="alpha_max")
    assert greedy_masked_plan(ds, min_marginals(ds.rows), caps) is None

    out = score_dataset(ds, alpha=1.0)
    assert out["legacy_forward_only"]["greedy_stuck"] is True
    # ...and it still has a perfectly well-defined exact statistic.
    assert "r_exact_pct" in out
    assert not out.get("no_feasible_rows")


def test_amendment_2_rescues_the_stranded_dataset(tmp_path):
    """AMENDMENT 2: the complete masked decode finds the plan the forward-only pass
    walked past. (A, B) = (A1, B0) is feasible and enumerated; the forward pass commits
    task 0 to n0 and strands task 1."""
    corpus = stuck_corpus(tmp_path)
    ds = Dataset(corpus / "ds_00001", TASK_TYPES, "rtt")
    caps = ds.node_caps(1.0, cap_mode="alpha_max")
    assert complete_masked_plan(ds, min_marginals(ds.rows), caps) == {0: A1, 1: B0}

    out = score_dataset(ds, alpha=1.0)
    assert not out.get("greedy_stuck")
    assert out["greedy_rescued_by_completion"] is True
    assert "r_greedy_pct" in out


def test_the_completion_only_ADDS_never_moves_an_existing_plan(tmp_path):
    """AMENDMENT 2 §5's byte-identity obligation, on the dataset the forward pass already
    completed: same plan, same r_greedy_pct, from the same options in the same order."""
    corpus = stuck_corpus(tmp_path)
    ds = Dataset(corpus / "ds_00000", TASK_TYPES, "rtt")
    caps = ds.node_caps(1.0, cap_mode="alpha_max")
    marginal = min_marginals(ds.rows)
    assert complete_masked_plan(ds, marginal, caps) == greedy_masked_plan(ds, marginal, caps)

    out = score_dataset(ds, alpha=1.0)
    assert out["r_greedy_pct"] == out["legacy_forward_only"]["r_greedy_pct"]


def test_greedy_stuck_dataset_still_contributes_to_r_exact(tmp_path):
    """THE TEETH. Reads n == 1 before the denominator fix."""
    per_alpha = report(tmp_path)
    assert per_alpha["r_exact"]["n"] == 2
    assert per_alpha["n_exact_scored"] == 2


def test_the_rescued_dataset_now_contributes_to_r_greedy(tmp_path):
    """Post-AMENDMENT-2: `greedy_stuck` no longer censors anything here, because the
    condition it tests has stopped being true — not because the rule was rewritten."""
    per_alpha = report(tmp_path)
    assert per_alpha["r_greedy"]["n"] == 2
    assert per_alpha["n_greedy_scored"] == 2
    assert per_alpha["greedy_stuck"] == 0


def test_the_forward_only_numbers_come_from_the_same_run(tmp_path):
    """AMENDMENT 2's both-numbers obligation. Reading the deviation must not require a
    second run or a commit message."""
    per_alpha = report(tmp_path)
    legacy = per_alpha["legacy_forward_only"]
    assert legacy["greedy_stuck"] == 1
    assert legacy["n_greedy_scored"] == 1
    assert legacy["r_greedy"]["n"] == 1
    assert legacy["rescued_by_completion"] == 1
    # ...and the arm it censored is still legible, on the same key as the live counters.
    assert legacy["greedy_stuck_by_arm"] == {"2": 1}


def test_legacy_block_reproduces_the_prefix_denominator(tmp_path):
    """The 2026-08-27 denominator fix's audit block.

    Note the interaction AMENDMENT 2 creates: `legacy_greedy_censored` reproduces the
    denominator R_exact used to be summarized over — censored by `greedy_stuck` — and
    `greedy_stuck` now reads 0 wherever a feasible plan exists, so this block DEGENERATES
    to the r_exact block on any post-amendment run. It still earns its place: it is the
    audit trail for the earlier fix, and it goes non-degenerate again the moment
    `greedy_stuck` fires (which, post-amendment, means the mask and the sweep disagree).
    """
    per_alpha = report(tmp_path)
    legacy = per_alpha["legacy_greedy_censored"]
    assert legacy["n"] == 2
    assert legacy["r_exact"]["n"] == 2
    assert legacy["n"] == per_alpha["n_exact_scored"]


def test_stuck_confound_counter_is_reported(tmp_path):
    """greedy_stuck's confound with the design is made visible in every rung's artifact.
    Empty post-AMENDMENT-2; the forward-only block carries the confound now."""
    per_alpha = report(tmp_path)
    assert per_alpha["greedy_stuck_by_n_feasible"] == {}
    assert per_alpha["legacy_forward_only"]["greedy_stuck_by_arm"] == {"2": 1}


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


# --- the OTHER censor's arm confound (2026-08-27) --------------------------------------
#
# The fix above gave `greedy_stuck` an arm breakdown. `no_feasible_rows` — which censors
# r_exact and every LS/repair statistic, not just r_greedy — still had none, and it is
# confounded too: on H1 at its registered primary alpha=2.0 all 70 censored datasets sit
# in the 64-row arm and none in the 16-row arm.
#
# `greedy_stuck_by_n_feasible` cannot be reused for it: n_feasible is 0 by construction on
# a censored dataset, so every arm collapses into one bucket. The arm key is `n_rows`, the
# UNCONSTRAINED sweep size, which survives censoring.

CONFINED = {
    "rigA": [
        {"node_name": "n0", "platform_id": 401, "platform_type": "rigCpu"},
        {"node_name": "n0", "platform_id": 402, "platform_type": "rigCpu"},
        {"node_name": "n0", "platform_id": 403, "platform_type": "rigCpu"},
    ],
    "rigB": [
        {"node_name": "n0", "platform_id": 411, "platform_type": "rigCpu"},
    ],
}


def two_arm_corpus(tmp_path: Path) -> Path:
    """Two arms distinguishable by sweep size, with the strict censor confined to one.

    ds_00000/ds_00001 are the 2-row arm and stay feasible at alpha=1.0. ds_00002 is the
    3-row arm: every task type lives on n0, so every plan co-locates both tasks and no
    plan clears a cap of 1.0 x the max single demand -> no_feasible_rows, in that arm
    only. This is H1 alpha=2.0's shape in miniature.
    """
    corpus = stuck_corpus(tmp_path)
    write_ds(corpus, "ds_00002", CONFINED, [
        ({0: (100, 401), 1: (100, 411)}, 10.0),
        ({0: (100, 402), 1: (100, 411)}, 11.0),
        ({0: (100, 403), 1: (100, 411)}, 12.0),
    ])
    return corpus


def test_no_feasible_rows_gets_its_own_arm_breakdown(tmp_path):
    """THE TEETH. KeyError before the fix — the counter had no arm breakdown at all."""
    rep = score_corpus(two_arm_corpus(tmp_path), TASK_TYPES, "rtt", [1.0])
    by_arm = rep["per_alpha"]["1.0"]["censoring_by_arm"]

    assert by_arm["n_datasets"] == {"2": 2, "3": 1}
    # The strict censor fires in ONE arm only — the confound, stated rather than inferred.
    assert by_arm["no_feasible_rows"] == {"3": 1}
    # ...so r_exact is "over the 2-row arm only", which this line makes unmissable.
    assert by_arm["n_exact_scored"] == {"2": 2}


def test_arm_key_is_the_unconstrained_sweep_size_not_n_feasible(tmp_path):
    """n_feasible is 0 on a censored dataset, so it cannot separate the arms. Guard the
    choice of key: the censored dataset must land in its own bucket, not in a '0' one."""
    rep = score_corpus(two_arm_corpus(tmp_path), TASK_TYPES, "rtt", [1.0])
    per_alpha = rep["per_alpha"]["1.0"]
    assert "0" not in per_alpha["censoring_by_arm"]["no_feasible_rows"]
    assert per_alpha["censoring_by_arm"]["key"].startswith("n_rows")


def test_both_censors_share_one_arm_key_so_they_are_comparable(tmp_path):
    rep = score_corpus(two_arm_corpus(tmp_path), TASK_TYPES, "rtt", [1.0])
    by_arm = rep["per_alpha"]["1.0"]["censoring_by_arm"]
    assert by_arm["greedy_stuck"] == {}
    assert by_arm["no_feasible_rows"] == {"3": 1}
    # Totals must reconcile with the scalar counters they break down.
    per_alpha = rep["per_alpha"]["1.0"]
    assert sum(by_arm["greedy_stuck"].values()) == per_alpha["greedy_stuck"]
    assert sum(by_arm["no_feasible_rows"].values()) == per_alpha["no_feasible_rows"]
    assert sum(by_arm["n_exact_scored"].values()) == per_alpha["n_exact_scored"]
    assert sum(by_arm["n_greedy_scored"].values()) == per_alpha["n_greedy_scored"]
