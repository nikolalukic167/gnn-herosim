import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))


def _fake_sys(tmp_path, smt="off", cores=4, tpc=1):
    d = tmp_path / "cpu"
    for c in range(cores * tpc):
        (d / f"cpu{c}" / "topology").mkdir(parents=True)
        (d / f"cpu{c}" / "topology" / "physical_package_id").write_text("0\n")
        (d / f"cpu{c}" / "topology" / "core_id").write_text(f"{c % cores}\n")
    (d / "smt").mkdir()
    (d / "smt" / "control").write_text(smt + "\n")
    (d / "smt" / "active").write_text(("1" if smt == "on" else "0") + "\n")
    (d / "online").write_text(f"0-{cores * tpc - 1}\n")
    info = tmp_path / "cpuinfo"
    info.write_text("processor\t: 0\nmodel name\t: AMD EPYC 9335 32-Core Processor\n")
    return str(d), str(info)


def test_host_block_and_checks(tmp_path, monkeypatch):
    from scripts_cosim import fresh_topo_burst_v1_gate as G
    d, info = _fake_sys(tmp_path, smt="on", cores=4, tpc=2)
    monkeypatch.setattr(G.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3, 4, 5, 6, 7})
    monkeypatch.setattr(G, "PARALLEL", 5)
    h = G.host_block(d, info)
    assert h["nproc"] == 8 and h["physical_cores"] == 4 and h["cpuset"] == "0-7" and h["smt_control"] == "on" and h["smt_active"] == "1"
    assert h["cpu_model"] == "AMD EPYC 9335 32-Core Processor" and h["parallel"] == 5 and h["cpus_online"] == "0-7"
    with pytest.raises(SystemExit, match="physical cores"):
        G.check_parallel(h)
    G.check_parallel({**h, "parallel": 4})
    G.check_host(h)
    monkeypatch.setenv("R1A_EXPECT_SMT", "off")
    with pytest.raises(SystemExit, match="R1A_EXPECT_SMT"):
        G.check_host(h)
    monkeypatch.setenv("R1A_EXPECT_SMT", "on")
    monkeypatch.setenv("R1A_EXPECT_CPU_MODEL", "AMD EPYC 9475F 48-Core Processor")
    with pytest.raises(SystemExit, match="R1A_EXPECT_CPU_MODEL"):
        G.check_host(h)
    assert G._cpulist([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"


def test_host_block_on_this_machine():
    from scripts_cosim import fresh_topo_burst_v1_gate as G
    h = G.host_block()
    assert h["nproc"] == len(os.sched_getaffinity(0)) and 1 <= h["physical_cores"] <= h["nproc"] and h["hostname"]
