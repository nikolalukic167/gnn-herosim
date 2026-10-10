import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_cosim"))
import reactive_conc_read as RC


def _cell(d, kind, topo, win, rung, elapsed, end=1.0, share=0.1, seed=0):
    json.dump({"averageElapsedTime": elapsed, "arrival_end": {"end_over_last_arrival": end}, "effective_queue_share": share,
               "latency_percentiles": {"p95": 2 * elapsed}},
              open(os.path.join(d, f"cc40s{topo}__{win}{rung}__{kind}_s{seed}.summary.json"), "w"))


def test_paired_contrast_and_collapse_counts(tmp_path):
    d = str(tmp_path)
    for topo in (1, 2, 3):
        for win in ("g0", "g1"):
            _cell(d, "reactive_conc", topo, win, "moderate", 9.0)
            _cell(d, "cd", topo, win, "moderate", 10.0)
            _cell(d, "reactive", topo, win, "moderate", 500.0, end=2.0, share=0.99)
    r = RC.read([d])
    assert r["cells"]["reactive_conc"] == 6 and r["topologies"] == [1, 2, 3]
    c = r["contrasts"]["reactive_conc vs cd|moderate"]
    assert c["median_pct"] == -10.0 and c["wins"] == 3 and c["n_topologies"] == 3
    assert r["contrasts"]["reactive_conc vs cd|heavy"]["median_pct"] is None
    assert r["collapse"]["reactive|moderate"]["end_gt_1.5"] == 6 and r["collapse"]["reactive_conc|moderate"]["end_gt_1.5"] == 0
    assert r["contrasts"]["reactive_conc vs reactive|moderate"]["median_pct"] < -90
