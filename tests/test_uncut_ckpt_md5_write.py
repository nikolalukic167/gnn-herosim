"""uncut_ckpt_md5_write: contract checks, chained-run provenance, the submission warning."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import uncut_ckpt_md5_write as W  # noqa: E402


def _ckpts(tmp_path, btf=None):
    for arm in ("gnn-eng", "twin-eng"):
        for seed in (1, 2):
            f = tmp_path / f"m-{arm}-seed{seed}-final.pt"
            f.write_bytes(f"{arm}{seed}".encode())
            c = {"physics_env": {"warmth_physics": "node_disk_v2"}}
            if btf is not None:
                c["batched_teacher_forced"] = btf
            (tmp_path / f"m-{arm}-seed{seed}-final.contract.json").write_text(json.dumps(c))


def _link(job, host="n1", cpu="CPU X"):
    return {"job": job, "host": host, "cpu": cpu}


def _run(tmp_path, prov, extra=()):
    p = tmp_path / "prov.json"
    p.write_text(json.dumps(prov))
    out = tmp_path / "o.json"
    W.main(["--dir", str(tmp_path), "--name", "m", "--prov", str(p), "--out", str(out), *extra])
    return json.loads(out.read_text())


def _flat(tmp_path):
    return {f"{a}|s{s}": _link(f"9{i}_{s}") for i, a in enumerate(("gnn_eng", "twin_eng")) for s in (1, 2)}


def test_flat_provenance_round_trips(tmp_path):
    _ckpts(tmp_path)
    o = _run(tmp_path, _flat(tmp_path))
    assert o["gnn_eng|s1"] and o["_provenance"]["gnn_eng|s1"]["file"] == "m-gnn-eng-seed1-final.pt"


def test_chained_links_are_listed_and_not_split_by_job_id(tmp_path, capsys):
    _ckpts(tmp_path)
    prov = {k: {"links": [_link("100"), _link("101", cpu="CPU Y")], "submission": "arm2@abc"} for k in _flat(tmp_path)}
    o = _run(tmp_path, prov)
    e = o["_provenance"]["twin_eng|s2"]
    assert e["job"] == "100,101" and e["cpu"] == "CPU X; CPU Y" and len(e["links"]) == 2
    assert "WARNING" not in capsys.readouterr().err


def test_seeds_from_different_submissions_warn(tmp_path, capsys):
    _ckpts(tmp_path)
    prov = {k: {"links": [_link("100")], "submission": "arm2@abc"} for k in _flat(tmp_path)}
    prov["gnn_eng|s2"]["submission"] = "arm2@def"
    _run(tmp_path, prov)
    assert "gnn_eng" in capsys.readouterr().err


def test_missing_cpu_fails_loudly(tmp_path):
    _ckpts(tmp_path)
    prov = _flat(tmp_path)
    prov["gnn_eng|s1"]["cpu"] = ""
    with pytest.raises(SystemExit, match="FAIL LOUD"):
        _run(tmp_path, prov)


def test_batched_flag_must_match(tmp_path):
    _ckpts(tmp_path, btf=True)
    with pytest.raises(SystemExit, match="batched_teacher_forced"):
        _run(tmp_path, _flat(tmp_path))
    assert _run(tmp_path, _flat(tmp_path), ["--batched", "true"])["twin_eng|s1"]
