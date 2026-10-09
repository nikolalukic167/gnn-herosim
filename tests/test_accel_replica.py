"""accel_replica_v1: preinit.replica_placement_rule. first_compatible (default) is byte-identical to the generator before the flag; fastest_compatible
puts each type's per_server replica on the compatible platform with the lowest executionTime in the simulator's task-types table, ties by platform id."""
import copy
import json
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.generate_infrastructure import generate_replica_placements_deterministic  # noqa: E402

TASK_TYPES = json.loads((REPO / "data/nofs-ids/task-types.json").read_text())
SERVER = ["rpiCpu", "xavierCpu", "xavierGpu", "xavierDla", "pynqFpga"]
NODES = [{"node_name": f"server{i}", "platforms": list(SERVER)} for i in range(3)]


def _gen(rule=None, overlap=True, task_types=TASK_TYPES, nodes=NODES):
    preinit = {"servers": [n["node_name"] for n in nodes], "clients": [], "replica_overlap": overlap}
    if rule is not None:
        preinit["replica_placement_rule"] = rule
    config = {"preinit": preinit, "replicas": {t: {"per_server": 1, "per_client": 0} for t in task_types}}
    return generate_replica_placements_deterministic(nodes, config, {"task_types": task_types}, random.Random(0))


def _types_on(placements, platform_type):
    return sorted(t for t, ps in placements.items() if any(p["platform_type"] == platform_type for p in ps))


def test_default_is_byte_identical_to_the_absent_key_and_to_first_compatible():
    a, b, c = _gen(None), _gen("first_compatible"), _gen("first_compatible", overlap=True)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True) == json.dumps(c, sort_keys=True)
    # first compatible in the node's list order: rpiCpu hosts all four types, xavierGpu / xavierDla / xavierCpu none (dnn1 stops at rpiCpu too)
    assert _types_on(a, "rpiCpu") == ["cnn", "dnn1", "dnn2", "rf"]
    assert _types_on(a, "xavierGpu") == _types_on(a, "xavierDla") == []


def test_fastest_compatible_follows_the_execution_time_table():
    p = _gen("fastest_compatible")
    assert _types_on(p, "pynqFpga") == ["dnn1"]          # 0.00057 s < xavierCpu 0.00105 s
    assert _types_on(p, "xavierCpu") == ["dnn2"]          # 0.0239 s < xavierGpu 0.0362 s
    assert _types_on(p, "xavierGpu") == ["cnn", "rf"]     # 0.104 s (cnn) and 0.0005 s (rf)
    assert _types_on(p, "xavierDla") == []                # never the fastest for any type under overlap
    assert _types_on(p, "rpiCpu") == []
    assert all(len(ps) == len(NODES) for ps in p.values())


def test_without_overlap_a_taken_platform_pushes_the_later_type_to_its_next_fastest():
    p = _gen("fastest_compatible", overlap=False)
    assert _types_on(p, "xavierGpu") == ["rf"] and _types_on(p, "xavierDla") == ["cnn"]   # rf (before cnn) takes the GPU; cnn falls to the DLA
    keys = [(q["node_name"], q["platform_id"]) for ps in p.values() for q in ps]
    assert len(keys) == len(set(keys))


def test_ties_break_by_platform_id():
    tt = copy.deepcopy(TASK_TYPES)
    for name in tt:
        tt[name]["executionTime"] = {pl: 1.0 for pl in tt[name]["platforms"]}
    nodes = [{"node_name": "server0", "platforms": ["xavierGpu", "xavierCpu", "rpiCpu", "xavierDla"]}]
    p = _gen("fastest_compatible", task_types=tt, nodes=nodes)
    first_id = {t: ps[0]["platform_id"] for t, ps in p.items()}
    assert set(first_id.values()) == {0}                  # platform id 0 is xavierGpu, supported by dnn1 / dnn2 / rf / cnn


def test_a_missing_execution_time_and_an_unknown_rule_fail_loud():
    tt = copy.deepcopy(TASK_TYPES)
    del tt["cnn"]["executionTime"]["xavierDla"]
    with pytest.raises(ValueError, match="executionTime"):
        _gen("fastest_compatible", task_types=tt)
    _gen("first_compatible", task_types=tt)               # the default never reads the table
    with pytest.raises(ValueError, match="replica_placement_rule"):
        _gen("fastest")


def test_the_legacy_path_is_refused_under_a_non_default_rule():
    from src.generate_infrastructure import require_generated_replicas_for_rule as guard

    cfg = {"preinit": {"replica_placement_rule": "fastest_compatible"}}
    with pytest.raises(RuntimeError, match="legacy path would re-derive"):
        guard(cfg, None)
    guard(cfg, {"nodes": []})                                   # generated infrastructure present: fine
    guard({"preinit": {}}, None)                                # default rule: the legacy path is what it always was
    guard({"preinit": {"replica_placement_rule": "first_compatible"}}, None)

