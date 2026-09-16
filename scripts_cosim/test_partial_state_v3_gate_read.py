"""Tests for the P2 / P3 gate-summary glue."""
import json
import os
import tempfile

from scripts_cosim.partial_state_v3_gate_read import (
    format_p2, format_p3, p2_inputs, p3_inputs, read_p2_dir, read_p3_dir,
)
from scripts_cosim.partial_state_v3_read import (
    P2_CELLS, P2_V2_CONTROL_MEDIANS, P3_CHECKPOINT_SEEDS, V_DEGRADES, V_GENERALISES,
    V_STILL_PINNED, V_TIE,
)


def _dir(docs):
    d = tempfile.mkdtemp()
    for doc in docs:
        json.dump(doc, open(os.path.join(d, doc["arm"] + ".summary.json"), "w"))
    return d


# --- P2 ------------------------------------------------------------------------------------

def _p2_docs(v3_scale=1.0):
    docs = []
    for cell in P2_CELLS:
        base = P2_V2_CONTROL_MEDIANS[cell]
        for kind in ("gnn", "mpoff"):
            seeds = [s for s in range(1, 17) if not (kind == "gnn" and s == 3)]
            for i, seed in enumerate(seeds):
                # v2 values symmetric around the queue_range median (15 gnn seeds -> the 8th
                # is the median), so the P2-a control reproduces it exactly
                v2 = base + (i - (len(seeds) - 1) / 2) * 0.1
                for rep, scale in (("v2", 1.0), ("v3", v3_scale)):
                    docs.append({"arm": f"{cell}__{rep}_{kind}_s{seed}", "cell": cell,
                                 "representation": rep, "arm_kind": kind, "training_seed": seed,
                                 "averageElapsedTime": v2 * scale, "wallclock_s": 100})
    return docs


def test_p2_inputs_group_by_kind_rep_cell_and_recover_the_control_medians():
    inp = p2_inputs(_p2_docs())
    assert set(inp["table"]) == {"gnn", "mpoff"}
    assert set(inp["table"]["gnn"]) == {"v3", "v2"}
    assert len(inp["table"]["gnn"]["v2"]["cell_s9001_f4000_pg16"]) == 15   # seed 3 excluded
    assert len(inp["table"]["mpoff"]["v2"]["cell_s9001_f4000_pg16"]) == 16
    for cell, want in P2_V2_CONTROL_MEDIANS.items():
        assert abs(inp["v2_gnn_medians"][cell] - want) < 1e-9


def test_p2_tie_when_v3_equals_v2():
    res = read_p2_dir(_dir(_p2_docs(1.0)))
    assert res["gnn"]["verdict"] == V_TIE and res["mpoff"]["verdict"] == V_TIE
    assert res["n_arms"] == 3 * (15 + 16) * 2
    assert "P2-a control" in format_p2(res)


def test_p2_reads_a_live_cost_when_v3_is_slower():
    res = read_p2_dir(_dir(_p2_docs(1.15)))
    assert res["gnn"]["verdict"] == "REPRESENTATION-COSTS-LIVE"


# --- P3 ------------------------------------------------------------------------------------

def _p3_docs(d_by_rung, drop=(), hang=()):
    """d_by_rung: rung -> deficit for gnn (mpoff at +0.5). drop: (rung, cell, kind, seed)
    learned arms omitted (P3-a). hang: (rung, cell) whose reactive arm is missing (attrition)."""
    servers = {"R0": 6, "R1": 12, "R2": 24, "R3": 80}
    docs = []
    for rung, d in d_by_rung.items():
        for tseed in (9001, 9002, 9003, 9005):
            cell = f"cs{servers[rung]}s{tseed}"
            react = 20.0 + tseed % 7
            if (rung, cell) not in hang:
                docs.append({"arm": f"{cell}__{rung}__reactive_s0", "cell": cell, "rung": rung,
                             "arm_kind": "reactive", "checkpoint_seed": 0,
                             "averageElapsedTime": react, "wallclock_s": 300})
            for kind, dd in (("gnn", d), ("mpoff", 0.5)):
                for seed in P3_CHECKPOINT_SEEDS:
                    if (rung, cell, kind, seed) in drop:
                        continue
                    docs.append({"arm": f"{cell}__{rung}__{kind}_s{seed}", "cell": cell, "rung": rung,
                                 "arm_kind": kind, "checkpoint_seed": seed,
                                 "averageElapsedTime": react * (1 + dd), "wallclock_s": 300})
    return docs


def test_p3_inputs_drop_hung_cells_and_count_completion():
    inp = p3_inputs(_p3_docs({"R0": 0.3, "R1": 0.3}, hang={("R1", "cs12s9001")},
                             drop={("R1", "cs12s9002", "gnn", 4)}))
    assert set(inp["R0"]["cells"]) == {"cs6s9001", "cs6s9002", "cs6s9003", "cs6s9005"}
    assert "cs12s9001" not in inp["R1"]["cells"]
    assert inp["R1"]["cells"]["cs12s9002"]["completed"] == {"gnn": 3, "mpoff": 4}


def test_p3_generalises_end_to_end():
    res = read_p3_dir(_dir(_p3_docs({"R0": 0.30, "R1": 0.33, "R2": 0.36, "R3": 0.39})))
    assert res["p3"]["verdict"] == V_GENERALISES
    assert res["p4"]["verdict"] in ("SCALE-HELPS", "SCALE-DOES-NOT-HELP")
    assert "P3 VERDICT: GENERALISES" in format_p3(res)


def test_p3_degrades_end_to_end():
    res = read_p3_dir(_dir(_p3_docs({"R0": 0.30, "R1": 0.35, "R2": 0.60, "R3": 1.20})))
    assert res["p3"]["verdict"] == V_DEGRADES and res["p3"]["first_break"] == "R2"
    assert res["p4"]["verdict"] == "NOT-READ"


def test_p3_a_a_missing_learned_arm_is_pinned_only_when_its_cause_is_the_representation():
    docs = _p3_docs({"R0": 0.3, "R1": 0.3, "R2": 0.3, "R3": 0.3}, drop={("R3", "cs80s9001", "gnn", 1)})
    # cause recorded as a krank raise -> P3-a fires
    res = read_p3_dir(_dir(docs), failures={"cs80s9001__R3__gnn_s1": "krank_node_order"})
    assert res["p3"]["verdict"] == V_STILL_PINNED and "pinned" in format_p3(res)
    # cause recorded as an OOM (job 769872's actual loss) -> disclosed, rung read on 3 seeds
    res = read_p3_dir(_dir(docs), failures={"cs80s9001__R3__gnn_s1": "oom"})
    assert res["p3"]["verdict"] == V_GENERALISES
    assert res["p3"]["incomplete_other"] == [("R3", "cs80s9001", "gnn", ["oom"])]
    assert "NOT a representation failure" in format_p3(res)
    # no failures map at all -> unclassified, still not pinned
    res = read_p3_dir(_dir(docs))
    assert res["p3"]["verdict"] == V_GENERALISES and res["p3"]["incomplete_other"][0][3] == ["unclassified"]


def test_p3_attrition_below_min_cells_makes_the_rung_unreadable_not_pinned():
    res = read_p3_dir(_dir(_p3_docs({"R0": 0.3, "R1": 0.9, "R2": 0.3, "R3": 0.3},
                                    hang={("R1", "cs12s9001"), ("R1", "cs12s9002")})))
    assert res["p3"]["verdict"] == V_GENERALISES and "R1" not in res["p3"]["readable_scaled"]
