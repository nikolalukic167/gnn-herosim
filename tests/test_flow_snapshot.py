import copy

import pytest

from src.placement import flow_snapshot as F


def _t(pid, status, rem=100.0, **kw):
    t = {"partner_id": pid, "src_node": "a", "payload": 100.0, "latency": 0.01, "hops": [["a>c", 10.0], ["c>b", 5.0]], "b": 0.0, "rest": 100.0, "scale": 1.0,
         "status": status, "startup_remaining": 0.0, "rem": rem, "rate": 0.0, "latency_remaining": 0.01}
    t.update(kw)
    return t


OK = {
    "version": 1, "model": "processor_sharing_v1", "partners": "serial", "tcp": True, "fairshare_last": 0.0, "active": [[5, 1]],
    "exchanges": [
        {"owner_tid": 5, "dst_node": "b", "stage": "input_io", "planned": False, "read_remaining": 0.0,
         "transfers": [_t(3, "done", 0.0, latency_remaining=0.0), _t(7, "active", 40.0, rate=5.0), _t(9, "pending")]},
        {"owner_tid": 6, "dst_node": "b", "stage": "input_io", "planned": False, "read_remaining": 1.5, "transfers": [_t(3, "pending")]},
    ],
}


def test_valid_block_passes():
    F.validate(OK)


@pytest.mark.parametrize("edit", [
    lambda b: b.update(version=2),
    lambda b: b.update(fairshare_last=0.5),
    lambda b: b["active"].clear(),
    lambda b: b["active"].append([5, 1]),
    lambda b: b["exchanges"][0]["transfers"][1].update(status="bogus"),
    lambda b: b["exchanges"][0]["transfers"][1].update(rem=101.0),
    lambda b: b["exchanges"][0]["transfers"][1].update(rate=0.0),
    lambda b: b["exchanges"][0]["transfers"].reverse(),
    lambda b: b["exchanges"][1]["transfers"][0].update(status="startup"),
    lambda b: b["exchanges"].append(copy.deepcopy(b["exchanges"][0])),
])
def test_broken_block_fails_loudly(edit):
    b = copy.deepcopy(OK)
    edit(b)
    with pytest.raises(ValueError):
        F.validate(b)


# ---- capture (snapshot_fidelity._capture_flows) on real FairShare flows and stub tasks ------------------------------------------------
import simpy
from types import SimpleNamespace

from src.placement import snapshot_fidelity as SF
from src.placement.network_fabric import FairShare


def _plan(peer_id, node, nbytes, hops, b=0.0, latency=0.0):
    return {"peer_id": peer_id, "peer_node": node, "payload": nbytes, "latency": latency, "hops": hops, "b": b, "rest": nbytes, "scale": 1.0, "solo": 0.0}


def test_owner_does_not_change_flow_dynamics():
    ends = {}
    for owner in (None, (1, 0)):
        env = simpy.Environment()
        fs = FairShare(env)
        done = []

        def go(n, ev):
            yield ev
            done.append((n, env.now))

        env.process(go("a", fs.start([("x>y", 10.0)], 20 * 1024 ** 2, owner=owner)))
        env.process(go("b", fs.start([("x>y", 10.0)], 10 * 1024 ** 2)))
        env.run()
        ends[owner is None] = done
    assert ends[True] == ends[False]


def test_capture_flows_projects_rem_without_touching_fairshare(monkeypatch):
    monkeypatch.setenv("HEROSIM_EXCHANGE_FAIRSHARE", "1")
    monkeypatch.delenv("HEROSIM_EXCHANGE_FAIRSHARE_PARTNERS", raising=False)
    env = simpy.Environment()
    fs = FairShare(env)
    hops = [("x>y", 10.0)]
    fs.start(hops, 40 * 1024 ** 2, owner=(11, 0))
    fs.start(hops, 40 * 1024 ** 2, owner=(12, 0))
    env.run(until=1.0)
    before = [(f.rem, f.rate) for f in fs.flows], fs._last, fs._gen
    node = SimpleNamespace(fabric=SimpleNamespace(fairshare=fs), node_name="n")
    plat = SimpleNamespace(node=node)

    def ghost(tid, plan):
        task = SimpleNamespace(id=tid, _fid={"fx": {"plan": plan, "i": 0, "phase": "bytes", "phase_end": None}, "read_end": 0.5}, io_end_time=None)
        return plat, task, {"stage": "input_io"}

    plan11 = [_plan(3, "a", 40 * 1024 ** 2, hops), _plan(4, "b", 5.0, hops)]
    owners = [ghost(11, plan11), ghost(12, [_plan(3, "a", 40 * 1024 ** 2, hops)])]
    sched = SimpleNamespace(nodes=SimpleNamespace(items=[node]))
    out = SF._capture_flows(sched, owners, 1.0)
    assert (before == ([(f.rem, f.rate) for f in fs.flows], fs._last, fs._gen))
    F.validate(out)
    assert out["active"] == [[11, 0], [12, 0]]
    t = out["exchanges"][0]["transfers"]
    assert [x["status"] for x in t] == ["active", "pending"]
    assert t[0]["rem"] == pytest.approx(40 * 1024 ** 2 - 5 * 1024 ** 2 * 1.0)
    assert out["fairshare_last"] == -1.0 and out["unplanned"] == []


def test_flow_log_is_off_by_default_and_writes_starts_when_on(monkeypatch, tmp_path):
    import json

    monkeypatch.delenv("HEROSIM_FLOW_LOG", raising=False)
    assert FairShare(simpy.Environment())._flow_log is None
    path = tmp_path / "flows.jsonl"
    monkeypatch.setenv("HEROSIM_FLOW_LOG", str(path))
    env = simpy.Environment()
    fs = FairShare(env)
    fs.start([("x>y", 10.0)], 1024 ** 2, owner=(7, 2))
    fs._flow_log.flush()
    assert json.loads(path.read_text().splitlines()[0]) == [0, 7, 2, ["x>y"], float(1024 ** 2)]
