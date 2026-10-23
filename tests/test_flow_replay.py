"""Replay of stored peer-exchange flows (flow_replay.py, option B 2026-10-23).

The identity gate, synthetic: a live run of several tasks whose serial exchanges share one narrow link is snapshotted at an instant T
(a stored-flow block built from the live objects and the live run's own trace), and replayed from that block in a fresh world. Every
task's end time, and a task that begins after T, must equal the live one (as seconds after T) to 1e-9, whether T falls in the read, a TCP
start-up, the shared bytes or a latency tail. Also: default off is inert, a missing block or a rate mismatch fails loudly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import simpy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from src.placement import flow_replay, flow_snapshot  # noqa: E402
from src.placement.infrastructure import Platform  # noqa: E402
from test_exchange_fairshare import MiB, world  # noqa: E402
from test_peer_exchange_cost import FakeTask  # noqa: E402

# (owner task id, start s, pre-exchange read s, {partner id: bytes}); partners 1..4 live on nodes p0..p3, every owner on node a
TASKS = [
    (0, 0.0, 0.0, {1: 30 * MiB, 2: 20 * MiB}),
    (5, 3.0, 2.0, {3: 25 * MiB}),
    (7, 6.0, 1.0, {2: 15 * MiB, 4: 12 * MiB}),
]


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    monkeypatch.setenv("HEROSIM_TRANSFER_MODEL", "pipelined")
    monkeypatch.delenv("HEROSIM_EXCHANGE_FAIRSHARE_PARTNERS", raising=False)
    monkeypatch.setenv("HEROSIM_EXCHANGE_FAIRSHARE", "1")


def build(monkeypatch, tcp):
    if tcp:
        monkeypatch.setenv("HEROSIM_EXCHANGE_TCP", "1")
    else:
        monkeypatch.delenv("HEROSIM_EXCHANGE_TCP", raising=False)
    env, fabric, platform, t0 = world(monkeypatch, [1.0] * 4)
    orch = platform.node.orchestrator_ref
    orch.peer_exchange = {tid: dict(peers) for tid, _s, _r, peers in TASKS}
    tasks = {0: t0}
    for tid, *_ in TASKS:
        tasks[tid] = t0 if tid == 0 else FakeTask(tid, platform.node)
        orch.task_by_id[tid] = tasks[tid]
    return env, fabric, platform, tasks


class Trace:
    """Start and end of each FairShare flow, keyed by its (unique) byte count."""

    def __init__(self):
        self.start, self.end = {}, {}


def traced(monkeypatch):
    trace = Trace()
    real = flow_replay.FairShare.start

    def start(fs, hops, nbytes, scale=1.0):
        ev = real(fs, hops, nbytes, scale)
        trace.start[nbytes] = fs.env.now
        ev.callbacks.append(lambda e, n=nbytes, fs=fs: trace.end.__setitem__(n, fs.env.now))
        return ev

    return trace, start


def live_process(env, platform, task, start, read, ends, plans):
    yield env.timeout(start)
    flows = Platform._fairshare_plan(platform, task)
    plans[task.id] = flows
    yield env.timeout(read)
    yield from Platform._fairshare_exchange(platform, task, flows)
    ends[task.id] = env.now


def run_live(monkeypatch, tcp, until=None, tasks_from=None):
    env, fabric, platform, tasks = build(monkeypatch, tcp)
    ends, plans = {}, {}
    for tid, s, r, _p in TASKS:
        env.process(live_process(env, platform, tasks[tid], s, r, ends, plans))
    env.run(until=until)
    return env, fabric, ends, plans


def snapshot(monkeypatch, tcp, T):
    """Ends, plans and the flow trace of one full live run."""
    trace, start = traced(monkeypatch)
    with monkeypatch.context() as m:
        m.setattr(flow_replay.FairShare, "start", start)
        _env, _fab, ends, plans = run_live(monkeypatch, tcp)
    return ends, plans, trace


def block_at(monkeypatch, tcp, T, ends, plans, trace):
    env, fabric, _e, _p = run_live(monkeypatch, tcp, until=T)
    fs = fabric.fairshare
    fs._advance()
    live = {f.nbytes: (f.rem, f.rate) for f in fs.flows}
    order = [f.nbytes for f in fs.flows]
    exchanges, active = [], []
    for tid, s, r, peers in TASKS:
        if T <= s:
            continue
        fl = plans[tid]
        read_remaining = max(0.0, s + r - T)
        transfers, begin = [], s + r
        for i, p in enumerate(fl):
            n = p["rest"]
            fl_start, fl_end = trace.start[n], trace.end[n]
            lat_end = fl_end + p["latency"]
            t = dict(partner_id=p["peer_id"], src_node=p["peer_node"], payload=p["payload"], latency=p["latency"],
                     hops=[list(h) for h in p["hops"]], b=p["b"], rest=p["rest"], scale=p["scale"], rate=0.0)
            if read_remaining > 0 or T < begin:
                t.update(status="pending", startup_remaining=p["b"], rem=p["rest"], latency_remaining=p["latency"])
            elif T >= lat_end:
                t.update(status="done", startup_remaining=0.0, rem=0.0, latency_remaining=0.0)
            elif T >= fl_end:
                t.update(status="latency_tail", startup_remaining=0.0, rem=0.0, latency_remaining=lat_end - T)
            elif T >= fl_start:
                rem, rate = live[n]
                t.update(status="active", startup_remaining=0.0, rem=rem, rate=rate, latency_remaining=p["latency"])
                active.append((order.index(n), tid, i))
            else:
                t.update(status="startup", startup_remaining=fl_start - T, rem=p["rest"], latency_remaining=p["latency"])
            transfers.append(t)
            begin = lat_end
        if all(t["status"] == "done" for t in transfers) and s + r <= T and transfers:
            continue
        exchanges.append(dict(owner_tid=tid, dst_node="a", stage="input_io", planned=False, read_remaining=read_remaining, transfers=transfers))
    return dict(version=1, model="processor_sharing_v1", partners="serial", tcp=tcp, fairshare_last=0.0,
                active=[[t, i] for _o, t, i in sorted(active)], exchanges=exchanges)


def replay(monkeypatch, tcp, T, block):
    env, fabric, platform, tasks = build(monkeypatch, tcp)
    events = flow_replay.start_active(fabric.fairshare, block)
    ends = {}

    def ghost(ex):
        yield from flow_replay.exchange_process(env, fabric.fairshare, ex, events)
        ends[ex["owner_tid"]] = env.now

    for ex in block["exchanges"]:
        env.process(ghost(ex))
    for tid, s, r, _p in TASKS:                                      # tasks that begin after T are ordinary live work on top of the stored flows
        if s > T:
            env.process(live_process(env, platform, tasks[tid], s - T, r, ends, {}))
    env.run()
    return ends


@pytest.mark.parametrize("tcp", [False, True])
@pytest.mark.parametrize("T", [0.7, 2.6, 4.1, 5.5, 7.9, 9.3, 12.2, 17.0, 21.4, 26.0])
def test_replay_from_stored_flows_reproduces_the_live_end_times(on, monkeypatch, tcp, T):
    ends, plans, trace = snapshot(monkeypatch, tcp, T)
    block = block_at(monkeypatch, tcp, T, ends, plans, trace)
    flow_snapshot.validate(block)
    got = replay(monkeypatch, tcp, T, block)
    live = {tid: t for tid, t in ends.items() if t > T}
    assert set(got) == set(live), (T, sorted(got), sorted(live))
    for tid, t in live.items():
        assert got[tid] == pytest.approx(t - T, rel=0, abs=1e-9), (tid, T, tcp)


def derived_instants(trace):
    """Instants inside the latency tails (0.02 s after each flow's last byte), inside each TCP start-up (0.01 s after the previous end) and at
    0.097 s before a flow's last byte: where the fixed grid cannot be trusted to land."""
    return sorted({round(e + 0.02, 6) for e in trace.end.values()} | {round(e - 0.097, 6) for e in trace.end.values()})


@pytest.mark.parametrize("tcp", [False, True])
def test_replay_is_exact_inside_latency_tails_and_just_before_flows_end(on, monkeypatch, tcp):
    ends, plans, trace = snapshot(monkeypatch, tcp, 0.0)
    for T in derived_instants(trace):
        block = block_at(monkeypatch, tcp, T, ends, plans, trace)
        flow_snapshot.validate(block)
        got = replay(monkeypatch, tcp, T, block)
        live = {tid: t for tid, t in ends.items() if t > T}
        assert set(got) == set(live), (T, sorted(got), sorted(live))
        for tid, t in live.items():
            assert got[tid] == pytest.approx(t - T, rel=0, abs=1e-9), (tid, T, tcp)


def test_the_snapshot_instants_cover_every_status(on, monkeypatch):
    seen = set()
    for tcp in (False, True):
        ends, plans, trace = snapshot(monkeypatch, tcp, 0.0)
        for T in (0.7, 2.6, 4.1, 5.5, 7.9, 9.3, 12.2, 17.0, 21.4, 26.0, *derived_instants(trace)):
            b = block_at(monkeypatch, tcp, T, ends, plans, trace)
            seen |= {t["status"] for ex in b["exchanges"] for t in ex["transfers"]}
            seen.add("read" if any(ex["read_remaining"] > 0 for ex in b["exchanges"]) else "-")
    assert {"pending", "active", "latency_tail", "done", "startup", "read"} <= seen


def test_background_flow_slows_a_new_flow_as_live_does(on, monkeypatch):
    """The point of the switch: a flow started on top of the stored ones is priced against them. Task 7 begins just after T and shares the
    narrow link with a stored flow; its replayed end equals its live end, and is later than it would be on an empty link."""
    T = 5.9
    ends, plans, trace = snapshot(monkeypatch, False, T)
    block = block_at(monkeypatch, False, T, ends, plans, trace)
    got = replay(monkeypatch, False, T, block)
    block_empty = dict(block, active=[], exchanges=[])
    alone = replay(monkeypatch, False, T, block_empty)
    assert got[7] == pytest.approx(ends[7] - T, abs=1e-9) and got[7] > alone[7] + 1e-6


def test_off_by_default_and_bad_values(monkeypatch):
    monkeypatch.delenv(flow_replay.ENV, raising=False)
    assert flow_replay.enabled() is False and flow_replay.require({}) is None
    monkeypatch.setenv(flow_replay.ENV, "0")
    assert flow_replay.enabled() is False
    monkeypatch.setenv(flow_replay.ENV, "2")
    with pytest.raises(ValueError):
        flow_replay.enabled()


def test_on_without_a_block_or_without_fairshare_raises(monkeypatch):
    monkeypatch.setenv(flow_replay.ENV, "1")
    monkeypatch.delenv("HEROSIM_EXCHANGE_FAIRSHARE", raising=False)
    with pytest.raises(RuntimeError, match="HEROSIM_EXCHANGE_FAIRSHARE=1"):
        flow_replay.require({})
    monkeypatch.setenv("HEROSIM_EXCHANGE_FAIRSHARE", "1")
    with pytest.raises(RuntimeError, match="no fidelity"):
        flow_replay.require({})


def test_a_stored_rate_that_the_replayed_links_do_not_reproduce_raises(on, monkeypatch):
    ends, plans, trace = snapshot(monkeypatch, False, 5.5)
    block = block_at(monkeypatch, False, 5.5, ends, plans, trace)
    ex = next(e for e in block["exchanges"] if any(t["status"] == "active" for t in e["transfers"]))
    next(t for t in ex["transfers"] if t["status"] == "active")["rate"] *= 1.5
    env, fabric, _p, _t = build(monkeypatch, False)
    with pytest.raises(RuntimeError, match="recomputed rate"):
        flow_replay.start_active(fabric.fairshare, block)
