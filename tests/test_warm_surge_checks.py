import pytest

from scripts_cosim.cd_gap_v1_build_b import surge_after
from scripts_cosim.warm_surge_checks import check_workload


def wl():
    ev = [{"timestamp": t, "peer_group": g, "qos": "m"} for g, ts in ((0, [1.0, 1.001]), (1, [12.0, 12.002]), (2, [20.0])) for t in ts]
    ev.sort(key=lambda e: e["timestamp"])
    return {"events": ev, "duration": 30.0, "peer_exchange": [[0, 1, 5.0]]}


def test_minted_surge_passes_and_tamper_fails():
    w = wl()
    o = surge_after(w, 10.0, 3.0)
    o["warm_surge_s0_v1"]["source_sha256"] = "x"
    assert check_workload(w, o, 10.0, 3.0, "x")["n_post"] == 3
    o["events"][-1]["timestamp"] += 1e-3
    with pytest.raises(SystemExit):
        check_workload(w, o, 10.0, 3.0, "x")
    o2 = surge_after(w, 10.0, 3.0)
    o2["warm_surge_s0_v1"]["source_sha256"] = "x"
    o2["events"][0]["qos"] = "other"
    with pytest.raises(SystemExit):
        check_workload(w, o2, 10.0, 3.0, "x")
