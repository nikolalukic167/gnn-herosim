"""Independently replay checkpoint-continuation artifacts and verify their accounting."""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from src.placement.radical.dag_reservation import problem
from src.placement.radical.dag_reservation_bridge import BridgedDag
from src.placement.radical.dag_reservation_live import replay
from src.placement.radical.dag_resume import InfeasibleContinuation, resume, snapshot
from src.placement.radical.environment import pack, serialized

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"incumbent", "hand", "candidate", "reference"}
SEARCHES = ARMS - {"incumbent"}


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AssertionError(f"duplicate JSON key {key} in {path}")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(plan):
    return (np.asarray(plan["assignment"], dtype=np.int64),
            np.asarray(plan["rank"], dtype=np.int64), np.asarray(plan["release"], dtype=float))


def compare(plan, result):
    assert plan["objective"] == result["objective"], "objective mismatch"
    for key in ("starts", "ends"):
        np.testing.assert_array_equal(plan[key], result[key], err_msg=key)
    if "events" in plan and "events" in result:
        key = lambda event: (event["job"], event["operation"], event["event"])
        assert sorted(plan["events"], key=key) == sorted(result["events"], key=key)


def history(state, plan, result):
    frozen = state.started.copy()
    frozen.ravel()[list(state.committed)] = True
    for key, original in (("assignment", state.assignment), ("starts", state.starts), ("ends", state.ends)):
        np.testing.assert_array_equal(np.asarray(plan[key])[frozen], original[frozen])
    saved = {(e["job"], e["operation"]): e for e in state.events if e["event"] == "start"}
    executed = {(e["job"], e["operation"]): e for e in result["events"] if e["event"] == "start"}
    for node in zip(*np.where(frozen)):
        assert executed[node] == saved[node], "immutable event changed"


def verify_plan(engine, b, plan):
    a, rank, _ = arrays(plan)
    materialized = np.asarray(plan["starts"], dtype=float)
    independent = replay(b, a, rank, materialized)
    compare(plan, independent)
    cost, starts, ends = engine.replay(b, a, rank, materialized)
    compare(plan, {"objective": cost, "starts": starts, "ends": ends})
    return independent


def verify_live(actual, plan, expected):
    a, _, _ = arrays(plan)
    assert abs(actual["job_completion_sum"] * 1000 - expected["objective"]) < 1e-6
    seen = set()
    for task in actual["tasks"]:
        node = tuple(map(int, task["taskType"]["name"][2:].split("_")))
        assert node not in seen
        seen.add(node)
        assert task["executionNode"] == f"node{a[node]}"
        assert abs(task["doneTime"] * 1000 - expected["ends"][node]) < 1e-6
    assert seen == set(np.ndindex(a.shape))
    events = {(e["job"], e["operation"], e["event"]): e for e in expected["events"]}
    observed = set()
    for event in actual["mixed_execution"]["events"]:
        key = event["job"], event["operation"], event["event"]
        assert key not in observed and key in events
        observed.add(key)
        expected_event = events[key]
        when = expected_event["start"] if key[2] == "start" else expected_event["end"]
        assert event["host"] == f'node{expected_event["host"]}'
        assert abs(event["time"] * 1000 - when) < 1e-6
        assert abs(event["setup"] * 1000 - expected_event["setup"]) < 1e-6
    assert observed == set(events)


def beam(plans, width):
    selected, seen = [], set()
    for plan in sorted(plans, key=lambda p: p["objective"]):
        identity = tuple(np.asarray(plan[key]).tobytes() for key in ("assignment", "starts", "ends"))
        if identity not in seen:
            seen.add(identity)
            selected.append(plan)
        if len(selected) == width:
            break
    return selected


def audit(out):
    out = out.resolve()
    registration = read(out / "protocol_before_run.json")
    protocol, phase = registration["protocol"], registration["phase"]
    assert phase in ("calibration", "primary", "retry")
    for name, recorded in registration["sources"].items():
        saved = out / "source_snapshot" / name
        assert saved.is_file() and digest(saved) == recorded, f"source snapshot mismatch: {name}"
        if name.startswith("src/placement/radical/"):
            assert digest(ROOT / name) == recorded, f"loaded replay source mismatch: {name}"
    assert protocol == read(out / "source_snapshot/experiments/dag_resume_s0_v1.json")
    manifest = read(out / "artifacts.json")
    excluded = {"artifacts.json", "AUDIT.json"}
    present = {str(p.relative_to(out)) for p in out.rglob("*")
               if p.is_file() and p.suffix in (".json", ".jsonl") and p.name not in excluded}
    assert set(manifest) == present, "artifact manifest coverage differs"
    for name, recorded in manifest.items():
        assert digest(out / name) == recorded, f"artifact mismatch: {name}"
    engine = BridgedDag(out / "build")
    assert engine.provenance == read(out / "native.json"), "native provenance mismatch"
    if phase == "calibration":
        seeds = protocol["calibration_seeds"]
    else:
        first, last = protocol["primary_seeds" if phase == "primary" else "retry_seeds"]
        seeds = list(range(first, last + 1))
    assert {p.parent.name for p in out.glob("*/read.json")} == {str(s) for s in seeds}
    seen, rows = set(), []
    evaluated = feasible = infeasible = runs = operations = 0
    missing_second_decisions = []
    for seed in seeds:
        cell = out / str(seed)
        row = read(cell / "read.json")
        assert row["seed"] == seed and row["depth"] == (1 if phase == "primary" else 2)
        b = problem(seed, *protocol["shape"])
        assert read(cell / "input.json") == serialized(b)
        identity = hashlib.sha256(pack(b).tobytes() + b["predecessors"].tobytes()).hexdigest()
        assert row["identity"] == identity and identity not in seen
        seen.add(identity)
        assert set(row["arms"]) == ARMS and set(row["search"]) == SEARCHES
        incumbent = row["arms"]["incumbent"]
        a, rank, release = arrays(incumbent)
        at = float(np.sort(np.asarray(incumbent["ends"]).ravel())[
            int(a.size * protocol["checkpoint_completed_fraction"]) - 1])
        assert row["checkpoint_time"] == at
        root = snapshot(b, a, rank, release, at, include_starts_at_cut=False)
        assert row["started"] == int(root.started.sum())
        assert row["completed"] == int((root.ends <= at).sum())
        compare(incumbent, resume(b, root, a, rank, release))
        assert row["common_start_ms"] <= protocol["common_start_budget_ms"]
        assert row["snapshot_ms"] >= 0
        assert {p.stem for p in (cell / "placements").glob("*.jsonl")} == SEARCHES
        per_search_costs = {}
        for name in sorted(SEARCHES):
            search = row["search"][name]
            records = [json.loads(line) for line in (cell / "placements" / f"{name}.jsonl").read_text().splitlines()]
            assert len(records) == search["evaluations"]
            cap = protocol["decision_evaluation_cap"] if name != "reference" else protocol[
                "reference_evaluation_cap" if row["depth"] == 1 else "retry_reference_evaluation_cap"]
            assert len(records) <= cap
            assert search["elapsed_ms"] >= 0
            expected_valid = name == "reference" or search["elapsed_ms"] <= protocol["decision_budget_ms"]
            assert search["timing_valid"] == expected_valid
            assert sum(record["evaluation_ms"] for record in records) <= search["elapsed_ms"] + 1e-6
            indices = [record["evaluation_index"] for record in records]
            assert indices == list(range(len(records))), "missing or duplicate evaluation index"
            costs = [incumbent["objective"]]
            stages = {"first": 0, "second": 0}
            first_plans = []
            for record in records:
                evaluated += 1
                stage = record["label"].split(":", 1)[0]
                if stage in stages:
                    stages[stage] += 1
                source = record.get("source_state")
                if stage == "second":
                    assert source is not None, "second-stage evaluation lacks its source state"
                    assert source["include_starts_at_cut"] is False
                    sa, sr, sl = arrays(source)
                    state = snapshot(b, sa, sr, sl, source["now"], committed=source["committed"],
                                     include_starts_at_cut=False)
                    assert state.now > root.now
                    assert state.committed == root.committed
                    assert any(np.array_equal(sa, saved["assignment"]) and
                               np.array_equal(sr, saved["rank"]) and
                               np.array_equal(state.starts, saved["starts"]) and
                               np.array_equal(state.ends, saved["ends"])
                               for saved in beam(first_plans, protocol["beam_width"])), \
                        "second-stage source is not in the distinct first-stage beam"
                    assert state.now == float(state.ends[state.ends > root.now].min())
                    source_plan = {"assignment": sa, "starts": state.starts, "ends": state.ends}
                    history(root, source_plan, {"events": state.events})
                else:
                    assert source is None
                    state = root
                assert record["checkpoint_time"] == state.now and record["evaluation_ms"] >= 0
                ca, cr, cl = arrays(record)
                try:
                    result = resume(b, state, ca, cr, cl)
                except InfeasibleContinuation as error:
                    assert record["status"] == "infeasible" and record["reason"] == str(error)
                    infeasible += 1
                    continue
                assert record["status"] == "feasible"
                if stage in stages:
                    action_text, rule = record["label"].split(":", 1)[1].rsplit(":", 1)
                    action = ast.literal_eval(re.sub(r"np\.int64\((\d+)\)", r"\1", action_text))
                    assert rule in protocol["continuation_rules"]
                    mutable = ~state.started.copy()
                    mutable.ravel()[list(state.committed)] = False
                    new_starts = [e for e in result["events"] if e["event"] == "start"
                                  and mutable[e["job"], e["operation"]]]
                    if action[0] == "wait":
                        assert all(e["start"] >= action[1] for e in new_starts)
                    else:
                        first_event = new_starts[0]
                        assert first_event["job"] * a.shape[1] + first_event["operation"] == action[0]
                        assert first_event["host"] == action[1] and first_event["start"] == state.now
                compare(record, result)
                history(root, record, result)
                history(state, record, result)
                verify_plan(engine, b, record)
                feasible += 1
                costs.append(result["objective"])
                if stage == "first":
                    first_plans.append({**record, **result})
            if name != "hand":
                assert stages["first"] == search["first_evaluations"]
                assert stages["second"] == search["second_evaluations"]
                assert sum(stages.values()) == len(records)
                assert 0 <= search["distinct_beam_schedules"] <= protocol["beam_width"]
                if row["depth"] == 1:
                    assert stages["second"] == search["distinct_beam_schedules"] == 0
                else:
                    assert search["distinct_beam_schedules"] == len(beam(first_plans, protocol["beam_width"]))
                    if name == "candidate" and stages["second"] == 0:
                        missing_second_decisions.append(seed)
            per_search_costs[name] = min(costs)
            assert search["plan"] == row["arms"][name]
        for name in ("hand", "candidate"):
            assert row["arms"][name]["objective"] == per_search_costs[name]
        assert row["arms"]["reference"]["objective"] == min(per_search_costs.values())
        assert {p.name for p in cell.glob("*_live.json")} == {f"{name}_live.json" for name in ARMS}
        for name, plan in row["arms"].items():
            expected = verify_plan(engine, b, plan)
            history(root, plan, expected)
            verify_live(read(cell / f"{name}_live.json"), plan, expected)
            runs += 1
            operations += a.size
        rows.append(row)
    report = read(out / "report.json")
    gains = {name: [100 * (r["arms"]["hand"]["objective"] - r["arms"][name]["objective"]) /
                    r["arms"]["hand"]["objective"] for r in rows] for name in ("candidate", "reference")}
    timing_valid = all(r["search"][name]["elapsed_ms"] <= protocol["decision_budget_ms"]
                       for r in rows for name in ("hand", "candidate"))
    advance = bool(timing_valid and np.median(gains["candidate"]) >= protocol["candidate_gain_bar_pct"]
                   and np.median(gains["reference"]) >= protocol["reference_headroom_bar_pct"])
    expected_report = {
        "cases": len(rows), "candidate_gain_pct": gains["candidate"], "reference_gain_pct": gains["reference"],
        "median_candidate_gain_pct": float(np.median(gains["candidate"])),
        "median_reference_gain_pct": float(np.median(gains["reference"])),
        "candidate_wins": sum(value > 0 for value in gains["candidate"]),
        "timing_valid": timing_valid, "advance": advance, "live_runs": runs, "live_operations": operations,
        "phase": phase, "training_allowed": False, "sources_unchanged": True,
        "median_gain_over_incumbent_pct": {name: float(np.median([
            100 * (r["arms"]["incumbent"]["objective"] - r["arms"][name]["objective"]) /
            r["arms"]["incumbent"]["objective"] for r in rows])) for name in SEARCHES}}
    assert report == expected_report, "report does not reproduce from retained case records"
    result = {"status": "PASS", "phase": phase, "unique_inputs": len(seen), "evaluations": evaluated,
              "feasible_evaluations": feasible, "infeasible_evaluations": infeasible,
              "live_runs": runs, "live_operations": operations, "advance": advance,
              "two_step_mechanism_valid": not missing_second_decisions,
              "cases_without_served_second_decision": missing_second_decisions,
              "auditor_sha256": digest(Path(__file__)), "artifact_manifest_sha256": digest(out / "artifacts.json")}
    (out / "AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().out), indent=2))
