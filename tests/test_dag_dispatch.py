"""Fan-out / fan-in dispatch in Orchestrator.workflow_process.

Before 2026-08-25 `workflow_process` took `TopologicalSorter(dag).static_order()` and
dispatched `ordered[current_index + 1]` — the next node in a *linearization*, whether or
not it was a child of the task that just finished. For `A -> {B, C, D}` that ran a width-3
fan-out as a depth-3 chain: siblings never overlapped in time, and were never presented to
the scheduler together, so they could never be placed jointly.

Nothing caught it because every application in every corpus is a single-node dag — the bug
had no structure to act on. It becomes load-bearing the moment a DAG workload exists, which
is what route A needs, so these tests come first.

Driven with light stand-ins rather than a full simulation: the change is entirely in which
children get dispatched and when, and a real Task drags in platforms, storage and a
scheduler loop that would obscure exactly the ordering being asserted.

Run: pipenv run python3 -m pytest tests/test_dag_dispatch.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pytest
import simpy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.placement.orchestrator import Orchestrator  # noqa: E402


class FakeTask:
    """The surface workflow_process actually touches."""

    def __init__(self, env: simpy.Environment, name: str, application):
        self.env = env
        self.type = {"name": name}
        self.application = application
        self.dependencies: List["FakeTask"] = []
        self.done = env.event()
        self.dispatched = env.event()
        self.finished = False
        self.storage = {"output": None, "input": None}
        self.dispatched_at: float | None = None

    # Mirrors Task.task_process's ordering exactly: succeed `done`, THEN set `finished`.
    # The gap between those two lines is why workflow_process cannot read `.finished`
    # straight off the event.
    def run(self, at: float):
        def _proc():
            yield self.env.timeout(at)
            self.done.succeed()
            self.finished = True
        return self.env.process(_proc())

    def __repr__(self):
        return f"FakeTask({self.type['name']})"


class FakeApplication:
    def __init__(self, dag: Dict[str, List[str]]):
        self.type = {"dag": dag, "name": "fake-app"}
        self.tasks: List[FakeTask] = []
        self.children_by_function: Dict[str, List[FakeTask]] = {}


class FakeScheduler:
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.put_order: List[str] = []
        self.tasks = self

    def put(self, task: FakeTask):
        self.put_order.append(task.type["name"])
        return self.env.timeout(0)


class Harness(Orchestrator):
    """Concrete Orchestrator that implements nothing but what the ABC demands."""

    def __init__(self, env, scheduler):
        self.env = env
        self.scheduler = scheduler

    # Abstract surface — unused by workflow_process.
    def monitor_process(self):  # pragma: no cover
        yield self.env.timeout(0)

    def __getattr__(self, item):  # pragma: no cover
        raise AttributeError(item)


def build(dag: Dict[str, List[str]]):
    env = simpy.Environment()
    app = FakeApplication(dag)
    tasks = {name: FakeTask(env, name, app) for name in dag}
    app.tasks = list(tasks.values())
    for name, parents in dag.items():
        tasks[name].dependencies = [tasks[p] for p in parents]
    children: Dict[str, List[FakeTask]] = {name: [] for name in dag}
    for name, parents in dag.items():
        for parent in parents:
            children[parent].append(tasks[name])
    app.children_by_function = children

    scheduler = FakeScheduler(env)
    orch = Harness.__new__(Harness)
    orch.env = env
    orch.scheduler = scheduler
    return env, app, tasks, scheduler, orch


def record_dispatch(env, tasks):
    """Watch each task's dispatched event so we can assert on simulation TIME, not order."""
    def watch(task):
        def _proc():
            yield task.dispatched
            task.dispatched_at = env.now
        return env.process(_proc())
    for task in tasks.values():
        watch(task)


def test_fan_out_dispatches_all_children_not_a_chain():
    """A -> {B, C, D}: all three siblings dispatch off A, at the same instant."""
    dag = {"A": [], "B": ["A"], "C": ["A"], "D": ["A"]}
    env, app, tasks, scheduler, orch = build(dag)
    record_dispatch(env, tasks)

    tasks["A"].dispatched.succeed()
    tasks["A"].run(at=1.0)
    env.process(orch.workflow_process(tasks["A"]))
    # Children run only after they are dispatched; give them a duration each.
    for name in ("B", "C", "D"):
        def arm(t=tasks[name]):
            def _proc():
                yield t.dispatched
                yield env.timeout(1.0)
                t.done.succeed()
                t.finished = True
            return env.process(_proc())
        arm()
    env.run()

    assert sorted(scheduler.put_order) == ["B", "C", "D"], scheduler.put_order
    times = {n: tasks[n].dispatched_at for n in ("B", "C", "D")}
    assert set(times.values()) == {1.0}, f"siblings must dispatch together, got {times}"


def test_fan_in_waits_for_every_parent_and_dispatches_once():
    """Diamond A -> {B, C} -> D: D waits for the LAST parent and is dispatched exactly once.

    Both B's and C's workflow_process see D as a child, so this is also the double-dispatch
    guard: Task.dispatched is a bare env.event() and a second .succeed() raises RuntimeError.
    """
    dag = {"A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"]}
    env, app, tasks, scheduler, orch = build(dag)
    record_dispatch(env, tasks)

    tasks["A"].dispatched.succeed()
    tasks["A"].run(at=1.0)
    env.process(orch.workflow_process(tasks["A"]))

    # B finishes early, C finishes late. D must wait for C.
    for name, duration in (("B", 1.0), ("C", 5.0)):
        def arm(t=tasks[name], d=duration):
            def _proc():
                yield t.dispatched
                yield env.timeout(d)
                t.done.succeed()
                t.finished = True
            return env.process(_proc())
        arm()

    def arm_d():
        yield tasks["D"].dispatched
        yield env.timeout(1.0)
        tasks["D"].done.succeed()
        tasks["D"].finished = True
    env.process(arm_d())

    env.run()

    assert scheduler.put_order.count("D") == 1, f"D dispatched {scheduler.put_order.count('D')}x"
    assert tasks["D"].dispatched_at == pytest.approx(6.0), (
        f"D must wait for the slower parent C (done at 6.0), got {tasks['D'].dispatched_at}"
    )


def test_a_child_is_never_dispatched_before_its_parent_is_marked_finished():
    """The ordering hazard the zero-delay timeout exists for.

    Task.task_process succeeds `done` and sets `.finished` on the NEXT line. Both it and
    workflow_process wake from that one event, so reading `.finished` immediately can see
    the pre-update value — and the scheduler's own readiness filter is written against
    `.finished`. A child released while its parent still reads unfinished would be rejected
    by that filter.
    """
    dag = {"A": [], "B": ["A"]}
    env, app, tasks, scheduler, orch = build(dag)

    observed: Dict[str, bool] = {}

    original_put = scheduler.put

    def spying_put(task):
        observed[task.type["name"]] = all(d.finished for d in task.dependencies)
        return original_put(task)

    scheduler.put = spying_put
    scheduler.tasks = scheduler

    tasks["A"].dispatched.succeed()
    tasks["A"].run(at=1.0)
    env.process(orch.workflow_process(tasks["A"]))
    env.run()

    assert observed.get("B") is True, (
        "B reached the scheduler while A still read unfinished; the scheduler's "
        "all(dependency.finished) filter would reject it"
    )


def test_single_node_application_dispatches_nothing_and_still_cleans_up():
    """Every corpus application is this shape; it must behave exactly as before."""
    dag = {"A": []}
    env, app, tasks, scheduler, orch = build(dag)

    class Storage:
        def __init__(self):
            self.removed: List[object] = []

        def remove_data(self, task):
            self.removed.append(task)

    storage = Storage()
    tasks["A"].storage["output"] = storage

    tasks["A"].dispatched.succeed()
    tasks["A"].run(at=1.0)
    env.process(orch.workflow_process(tasks["A"]))
    env.run()

    assert scheduler.put_order == []
    assert storage.removed == [tasks["A"]], "single-task application must free its output"


def test_storage_is_not_freed_while_a_sibling_still_needs_it():
    """A's output must survive until the whole application is done.

    The old terminal test was "last in the linearization", so the first branch to reach the
    end of that order freed EVERY task's output — including a parent whose other child had
    not read it yet.
    """
    dag = {"A": [], "B": ["A"], "C": ["A"]}
    env, app, tasks, scheduler, orch = build(dag)

    freed_at: Dict[str, float] = {}

    class Storage:
        def __init__(self, name):
            self.name = name

        def remove_data(self, task):
            freed_at[self.name] = env.now

    for name in dag:
        tasks[name].storage["output"] = Storage(name)

    tasks["A"].dispatched.succeed()
    tasks["A"].run(at=1.0)
    env.process(orch.workflow_process(tasks["A"]))

    # B is quick, C is slow. A's storage must not be freed when B's branch ends.
    for name, duration in (("B", 1.0), ("C", 10.0)):
        def arm(t=tasks[name], d=duration):
            def _proc():
                yield t.dispatched
                yield env.timeout(d)
                t.done.succeed()
                t.finished = True
            return env.process(_proc())
        arm()

    env.run()

    assert freed_at, "storage was never freed"
    assert freed_at["A"] == pytest.approx(11.0), (
        f"A's output was freed at {freed_at.get('A')} but C only finishes at 11.0"
    )


def test_gateway_waits_for_children_dispatched_after_the_roots_finish():
    """The termination fixed point.

    `gateway_process` snapshots which tasks are dispatched and waits on that snapshot. For
    a DAG that snapshot is only the roots — children are dispatched as their parents
    finish — so waiting on it alone ends the simulation while the children are still
    queued, and every one of them keeps `done_time is None`. That surfaces far away from
    its cause as "Task 1 has not completed or is missing required attributes", inside the
    warmup state capture, reported as an unexplained "System state capture FAILED".

    Asserted here on the shape rather than through a whole simulation: the wait must not
    conclude while a task that has since been dispatched is still unfinished.
    """
    dag = {"A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"]}
    env, app, tasks, scheduler, orch = build(dag)

    tasks["A"].dispatched.succeed()
    tasks["A"].run(at=1.0)
    env.process(orch.workflow_process(tasks["A"]))
    for name, duration in (("B", 1.0), ("C", 2.0), ("D", 1.0)):
        def arm(t=tasks[name], d=duration):
            def _proc():
                yield t.dispatched
                yield env.timeout(d)
                t.done.succeed()
                t.finished = True
            return env.process(_proc())
        arm()

    # Stand-in for the gateway's wait: snapshot, wait, then re-check for newcomers.
    concluded_at = {}

    def gateway_like():
        waited = set()
        while True:
            current = [t for t in tasks.values() if t.dispatched.triggered]
            pending = [t for t in current if t.type["name"] not in waited and not t.done.triggered]
            if pending:
                yield env.all_of([t.done for t in pending])
            waited.update(t.type["name"] for t in current)
            yield env.timeout(0)
            current = [t for t in tasks.values() if t.dispatched.triggered]
            if all(t.type["name"] in waited for t in current):
                break
        concluded_at["t"] = env.now

    env.process(gateway_like())
    env.run()

    unfinished = [n for n, t in tasks.items() if not t.done.triggered]
    assert not unfinished, f"simulation concluded with unfinished tasks: {unfinished}"
    # D is the last to finish: A(1) -> C(2) -> D(1) = 4.0
    assert concluded_at["t"] == pytest.approx(4.0), (
        f"gateway concluded at {concluded_at.get('t')}, before the DAG's last task"
    )
