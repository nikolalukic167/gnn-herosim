"""gnn_selfsearch: the plan search over a slate, on a synthetic score (no model)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.policy.gnn import scheduler as S  # noqa: E402


def make(table, cands, plans=None):
    """table(plan) -> score; the fake score_fn returns, for task t, logits whose chosen entry carries an equal share of table(plan)."""
    n = len(cands)
    tl = {i: list(cands[i]) for i in range(n)}

    def score_fn(t, committed):
        out = []
        for c in tl[t]:
            plan = [committed[j] if j != t else c for j in range(n)]
            out.append(table(tuple(plan)) / n)
        return np.array(out)

    host = SimpleNamespace(_exact_score_fn=lambda *a: (score_fn, tl))
    host._self_search = S.GNNScheduler._self_search.__get__(host)
    sl = SimpleNamespace(kept=[[{"node_id": c[0], "platform_id": c[1]} for c in cs] for cs in cands],
                         plans=plans if plans is not None else int(np.prod([len(c) for c in cands])))
    tasks = [SimpleNamespace(id=i) for i in range(n)]
    return host, sl, tasks


def run(host, sl, tasks, start):
    return host._self_search(tasks, None, {}, None, sl, {i: p for i, p in enumerate(start)})


CANDS = [[(1, 1), (2, 1), (3, 1)], [(1, 1), (2, 1), (3, 1)], [(1, 1), (2, 1)]]


def test_exact_picks_the_global_best_even_when_a_single_move_cannot_reach_it(monkeypatch):
    best = ((3, 1), (3, 1), (2, 1))
    table = lambda p: 10.0 if p == best else (1.0 if p == ((1, 1), (1, 1), (1, 1)) else 0.0)
    host, sl, tasks = make(table, CANDS)
    out = run(host, sl, tasks, [(1, 1), (1, 1), (1, 1)])
    assert tuple(out.values()) == best and host.ss_exact_batches == 1 and host.ss_changed_tasks == 3


def test_ascent_runs_above_the_cap_and_climbs_one_task_at_a_time():
    table = lambda p: sum(1.0 for a, b in zip(p, ((2, 1), (3, 1), (1, 1))) if a == b)
    host, sl, tasks = make(table, CANDS, plans=10_000)
    out = run(host, sl, tasks, [(1, 1), (1, 1), (2, 1)])
    assert tuple(out.values()) == ((2, 1), (3, 1), (1, 1))
    assert host.ss_ascent_batches == 1 and getattr(host, "ss_exact_batches", 0) == 0 and host.ss_ascent_sweeps >= 2


def test_ascent_stops_after_five_sweeps():
    calls = []
    def table(p):
        calls.append(p)
        return float(sum(i for i, x in enumerate(p)))  # any change raises or keeps; bounded by sweeps
    host, sl, tasks = make(table, CANDS, plans=10_000)
    run(host, sl, tasks, [(1, 1), (1, 1), (1, 1)])
    assert host.ss_ascent_sweeps <= S.SELF_SEARCH_SWEEPS


def test_a_score_tie_keeps_the_incumbent():
    host, sl, tasks = make(lambda p: 1.0, CANDS)
    out = run(host, sl, tasks, [(2, 1), (3, 1), (1, 1)])
    assert tuple(out.values()) == ((2, 1), (3, 1), (1, 1)) and host.ss_changed_batches == 0
    host, sl, tasks = make(lambda p: 1.0, CANDS, plans=10_000)
    out = run(host, sl, tasks, [(2, 1), (3, 1), (1, 1)])
    assert tuple(out.values()) == ((2, 1), (3, 1), (1, 1))


def test_a_decode_outside_the_slate_fails_loud():
    host, sl, tasks = make(lambda p: 0.0, CANDS)
    with pytest.raises(RuntimeError, match="outside its slate"):
        run(host, sl, tasks, [(9, 9), (1, 1), (1, 1)])


def test_flag_off_by_default_and_requires_the_no_split_variant(monkeypatch):
    monkeypatch.delenv("GNN_SELF_SEARCH", raising=False)
    assert S._self_search_on() is False
    monkeypatch.setenv("GNN_SELF_SEARCH", "1")
    monkeypatch.delenv("GNN_SLATE_NO_SPLIT", raising=False)
    monkeypatch.delenv("GNN_SERVE_CANDIDATE_SLATE", raising=False)
    with pytest.raises(ValueError, match="GNN_SLATE_NO_SPLIT"):
        S._self_search_on()
    monkeypatch.setenv("GNN_SERVE_CANDIDATE_SLATE", "declared_pruning_v1")
    monkeypatch.setenv("GNN_SLATE_NO_SPLIT", "1")
    assert S._self_search_on() is True
    monkeypatch.setenv("GNN_SELF_SEARCH", "2")
    with pytest.raises(ValueError):
        S._self_search_on()


def test_counters_are_whitelisted_by_the_orchestrator():
    import inspect
    from src.placement.orchestrator import Orchestrator
    src = inspect.getsource(Orchestrator._scheduler_counters)
    for name in ("ss_batches", "ss_exact_batches", "ss_ascent_batches", "ss_changed_batches", "ss_scored", "ss_seconds"):
        assert f'"{name}"' in src
