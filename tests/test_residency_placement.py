import copy
import itertools
import numpy as np
import pytest
import torch
from src.placement.residency import evaluate, evict_list, features, priority, search
from src.policy.residency.model import ResidencyNet


def case():
    return {"requests": [0, 1, 2, 0, 3, 4, 5, 0], "memory_mib": [1, 2, 1, 2, 1, 2],
            "capacity_mib": [3, 3, 3], "resident": [[True, True, False, False, False, False]] * 3,
            "frequency": [1, 2, 3, 4, 5, 6], "cold_s": [[1., 2., 3.]] * 6,
            "exec_s": [[.1, .2, .3]] * 6, "stop_s": [.01, .02, .03]}


def scalar(c, plan):
    resident = np.array(c["resident"], bool)
    clocks = np.zeros(3)
    total = 0.
    for f, h in zip(c["requests"], plan):
        miss = not resident[h, f]
        evicted = evict_list(resident[h], f, c["capacity_mib"][h], c["memory_mib"], priority(c, h))
        for old in evicted:
            resident[h, old] = False
        resident[h, f] = True
        assert resident[h] @ c["memory_mib"] <= c["capacity_mib"][h]
        clocks[h] += c["exec_s"][f][h] + miss * c["cold_s"][f][h] + len(evicted) * c["stop_s"][h]
        total += clocks[h]
    return total


def test_vectorized_replay_matches_scalar_and_admits_memory():
    c = case()
    plans = np.random.default_rng(101).integers(0, 3, (150, 8))
    np.testing.assert_allclose(evaluate(c, plans), [scalar(c, p) for p in plans], atol=1e-12)


def test_impossible_function_fails():
    c = case()
    c["memory_mib"][0] = 10
    with pytest.raises(ValueError, match="infeasible"):
        evaluate(c, [0] * 8)


def test_future_windows_do_not_enter_features():
    a = case()
    b = copy.deepcopy(a)
    a["windows"] = [{"requests": [0] * 8}]
    b["windows"] = [{"requests": [5] * 8}]
    np.testing.assert_array_equal(features(a, [0, 1]), features(b, [0, 1]))


def test_initial_cache_not_mutated_by_search():
    c = case()
    original = copy.deepcopy(c)
    search(c)
    assert c == original


def test_independent_event_auditor_matches_vectorized_physics():
    from scripts_cosim.audit_residency import scalar_replay
    c = case()
    for plan in [[0] * 8, [0, 1, 2, 0, 1, 2, 0, 1]]:
        cost, cold, evictions, events, resident = scalar_replay(c, plan)
        actual = evaluate(c, plan, return_state=True)
        assert cost == pytest.approx(actual["cost"][0])
        assert cold == actual["cold"][0]
        assert evictions == actual["evictions"][0]
        assert resident == actual["resident"][0].tolist()
        assert len(events) == len(plan)


def test_gnn_is_equivariant_to_host_reordering():
    torch.manual_seed(1)
    model = ResidencyNet("gnn").eval()
    x = torch.from_numpy(features(case(), [0, 1])[None])
    with torch.no_grad():
        a = model(x)
        b = model(x[:, :, [2, 0, 1]])
    torch.testing.assert_close(a[:, :, [2, 0, 1]], b)


@pytest.mark.parametrize("arm", ["gnn", "mpoff", "hand_mlp"])
def test_all_arms_receive_full_finite_tensor(arm):
    x = torch.from_numpy(features(case(), [0, 1])[None])
    model = ResidencyNet(arm)
    assert model(x).shape == (1, 8, 3)
    assert torch.isfinite(model(x)).all()
