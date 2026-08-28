"""Unit coverage for scripts_cosim/measure_route_b_additivity.py.

The tool's whole value is that it can contradict the scorer, so its own correctness has to
rest on synthetic sweeps whose additivity is known by construction, not on a corpus.

Arm coverage, this lineage's entire failure class: the fixtures below cover a purely
additive cost, a cost with an exact pairwise same-node term, and a cost with a
higher-order (all-four-co-located) term — the three shapes the H2 diagnosis had to tell
apart. A tool that only ever saw the additive one would report the other two as "additive
enough" and nobody would know.
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts_cosim"))

from measure_route_b_additivity import (  # noqa: E402
    fit,
    load_sweep,
    max_coresidency,
    measure_corpus,
)

# Two constraints on this fixture, both learned by getting it wrong:
#
# 1. The pool must be LARGER than the task count, as the real grid is (H2: 4 tasks over a
#    pool of 8 or 9). With pool == n_tasks every plan uses every slot, so co-residency is
#    constant AND an additive cost sums to the same value on every plan — zero variance,
#    undefined R^2.
# 2. There must be at least THREE nodes. With two, the same-node relation has at most two
#    blocks, the reachable partition space is smaller than the pair basis, and the pair
#    model fits *any* co-residency function exactly — including a genuine triple term. A
#    two-node fixture therefore cannot tell "pairwise" from "higher-order" at all, and a
#    test built on one would silently assert nothing. (The real H2 grid has two hosting
#    nodes; its pair model falls short of 1 because the residual there is not purely a
#    function of the co-residency partition.)
#
# 4/2/2 over three nodes => a pool of 8, 8P4 = 1680 plans — H2's own arm size.
SLOTS = [(0, 100), (0, 101), (0, 102), (0, 103),
         (1, 200), (1, 201),
         (2, 300), (2, 301)]
TASKS = [0, 1, 2, 3]
# Arbitrary but fixed per-(task, slot) costs. The t*i cross term keeps the cost from being
# invariant under permutation, which is what gives the regression something to explain.
BASE = {(t, s): 1.0 + 0.37 * t + 0.11 * i + 0.23 * t * i
        for t in TASKS for i, s in enumerate(SLOTS)}


def all_plans():
    """Every assignment of 4 tasks to 4 distinct slots — the uniqueness rule co-sim uses."""
    return [dict(zip(TASKS, perm)) for perm in itertools.permutations(SLOTS, len(TASKS))]


def write_corpus(tmp_path, cost_fn, n_datasets=2):
    corpus = tmp_path / "corpus"
    for d in range(n_datasets):
        ds = corpus / f"ds_{d:05d}" / "placements"
        ds.mkdir(parents=True)
        with open(ds / "placements.jsonl", "w") as fh:
            for plan in all_plans():
                rec = {"placement_plan": {str(t): list(s) for t, s in plan.items()},
                       "rtt": cost_fn(plan)}
                fh.write(json.dumps(rec) + "\n")
    return corpus


def additive_cost(plan):
    return sum(BASE[(t, s)] for t, s in plan.items())


def pairwise_cost(plan):
    """Additive plus an exact same-node pair penalty."""
    c = additive_cost(plan)
    for a, b in itertools.combinations(sorted(plan), 2):
        if plan[a][0] == plan[b][0]:
            c += 0.5
    return c


def higher_order_cost(plan):
    """Additive plus a TRIPLE term — fires only when tasks 0, 1 and 2 all share a node.

    Not representable in the pair basis: over the reachable indicator vectors you can never
    have exactly two of (0~1, 0~2, 1~2) true, so any linear combination that is zero on the
    three single-pair configurations is also zero on the all-three configuration.
    """
    c = additive_cost(plan)
    if plan[0][0] == plan[1][0] == plan[2][0]:
        c -= 3.0
    return c


def test_additive_cost_fits_exactly():
    """The contract: on separable physics R^2 is 1 and the residual is 0."""
    plans = all_plans()
    rtts = np.array([additive_cost(p) for p in plans])
    r2, resid, rms_pct, _ = fit(plans, rtts)
    assert r2 == pytest.approx(1.0, abs=1e-12)
    assert np.abs(resid).max() < 1e-9
    assert rms_pct < 1e-7


def test_pairwise_coupling_is_detected_and_then_explained_by_the_pair_term():
    """A same-node pair term must show up as residual, and pair_node must absorb it fully.

    This is the discriminator the H2 diagnosis needed: 'is the coupling pairwise?' has to
    be answerable, and it is only answerable if the pair model reaches 1 when the truth IS
    pairwise. On H2 it reached ~0.92-0.95, which is how we know half of it is not.
    """
    plans = all_plans()
    rtts = np.array([pairwise_cost(p) for p in plans])
    r2_add, _, _, _ = fit(plans, rtts)
    r2_pair, resid_pair, _, _ = fit(plans, rtts, "pair_node")
    assert r2_add < 0.999, "a real pairwise term must not read as additive"
    assert r2_pair == pytest.approx(1.0, abs=1e-12)
    assert np.abs(resid_pair).max() < 1e-9


def test_higher_order_coupling_is_not_absorbed_by_the_pair_term():
    """The failure mode the pair model must NOT paper over."""
    plans = all_plans()
    rtts = np.array([higher_order_cost(p) for p in plans])
    r2_add, _, _, _ = fit(plans, rtts)
    r2_pair, _, _, _ = fit(plans, rtts, "pair_node")
    assert r2_add < 0.999
    # It may improve, but it cannot close: the truth is not pairwise.
    assert r2_pair < 0.9999


def test_same_slot_term_is_vacuous_under_globally_distinct_replicas():
    """Documented in the tool's docstring; pinned so a future grid change is visible.

    No two tasks can share a slot, so pair_slot adds an always-empty column set and must
    be numerically identical to pair_node — not merely close.
    """
    plans = all_plans()
    rtts = np.array([pairwise_cost(p) for p in plans])
    r2_pn, _, rms_pn, _ = fit(plans, rtts, "pair_node")
    r2_ps, _, rms_ps, _ = fit(plans, rtts, "pair_slot")
    assert r2_ps == pytest.approx(r2_pn, abs=1e-12)
    assert rms_ps == pytest.approx(rms_pn, abs=1e-12)


def test_max_coresidency_counts_the_fullest_node():
    assert max_coresidency({0: (0, 100), 1: (0, 101), 2: (1, 200), 3: (2, 300)}) == 2
    assert max_coresidency({0: (0, 100), 1: (0, 101), 2: (0, 102), 3: (1, 200)}) == 3
    assert max_coresidency({t: (0, 100 + t) for t in TASKS}) == 4


def test_corpus_report_splits_on_the_arm_and_finds_the_coresidency_signal(tmp_path):
    """Preflight step 3: the summary must carry its own arm breakdown, never a pooled
    number alone. Here both datasets share an arm, so there is exactly one block — the
    point is that the key exists and is `n_rows`."""
    corpus = write_corpus(tmp_path, higher_order_cost)
    res = measure_corpus(corpus, want_cores=True, want_pairs=True)
    assert res["n_datasets"] == 2
    assert list(res["by_arm"]) == [str(len(all_plans()))]
    block = res["by_arm"][str(len(all_plans()))]
    assert block["r2_median"] < 0.999
    # The injected term needs three tasks on one node to fire and is negative, so the
    # densest buckets must carry a mean residual clearly further from zero than the
    # sparsest one — the shape that distinguished physics coupling from noise on H2.
    cores = res["coresidency"]
    assert abs(cores["3"]["resid_pct_mean"]) > abs(cores["2"]["resid_pct_mean"])


def test_missing_placements_jsonl_fails_loud(tmp_path):
    """CLAUDE.md rule 4, and placements.jsonl is mandatory — best.json alone must not do."""
    ds = tmp_path / "ds_00000"
    (ds / "placements").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="placements.jsonl missing"):
        load_sweep(ds)


def test_empty_corpus_fails_loud(tmp_path):
    corpus = tmp_path / "empty"
    corpus.mkdir()
    with pytest.raises(RuntimeError, match="nothing to measure"):
        measure_corpus(corpus, want_cores=False, want_pairs=False)
