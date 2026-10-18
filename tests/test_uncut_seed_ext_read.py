import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
sys.path.insert(0, os.path.dirname(__file__))
import uncut_seed_ext_read as X  # noqa: E402
from test_uncut_arm_gate_read import cell  # noqa: E402

TOPOS = list(range(1, 13))
TOPS = ",".join(map(str, TOPOS))


def test_sign_ci_index_for_twelve_topologies():
    assert X.ci_index(12) == 2  # [x(3), x(10)], coverage 96.1 %
    with pytest.raises(SystemExit):
        X.ci_index(5)


@pytest.mark.parametrize("vals,p,sign,want", [
    ([-3.0 - 0.1 * i for i in range(12)], 0.001, -1, "CLAIM"),
    ([3.0 + 0.1 * i for i in range(12)], 0.001, -1, "REVERSED"),
    ([3.0 + 0.1 * i for i in range(12)], 0.001, +1, "CLAIM"),
    ([0.2 * ((-1) ** i) for i in range(12)], 0.9, -1, "CLEAN-NEGATIVE"),
    ([4.0 * ((-1) ** i) for i in range(12)], 0.9, -1, "UNRESOLVED"),
    ([-0.95 + 0.0 * i for i in range(12)], 0.2, -1, "CLEAN-NEGATIVE"),
    ([-1.2 + 0.0 * i for i in range(12)], 0.2, -1, "UNRESOLVED"),
    ([0.2 * ((-1) ** i) for i in range(12)], 0.9, +1, "CLEAN-NEGATIVE"),
])
def test_labels(vals, p, sign, want):
    assert X.label(vals, p, sign) == want


def build(tmp_path, effects, kr_effects=None):
    """effects: arm -> mean latency (bar is 2.0); one cell per (topology, window, rung, seed); _kr cells at kr_effects (default the same)."""
    d = tmp_path / "g"
    d.mkdir()
    for t in TOPOS:
        for w in ("g0", "g1", "g2", "g3"):
            for r in ("moderate", "heavy"):
                cell(str(d), "cd_ib0", 0, t, w, r, 2.0)
                for arm, m in effects.items():
                    for sd in X.SEEDS[arm]:
                        jit = 0.002 * ((t * 7 + sd) % 5)
                        cell(str(d), arm, sd, t, w, r, m + jit)
                        cell(str(d), arm, sd, t, w, r, (kr_effects or effects)[arm] + jit, sfx="_kr")
    reads = []
    for pr, tw in X.READS.items():
        p = tmp_path / f"read_{pr}.json"
        p.write_text(json.dumps({"primary": pr, "twin": tw, "seeds": list(X.SEEDS[pr]), "combined": {"F1": "MIXED"}}))
        reads.append(str(p))
    return str(d), reads


def run(tmp_path, d, reads, **kw):
    args = ["--gate", d, "--read", ",".join(reads), "--topos", TOPS, "--topos-sha256", hashlib.sha256(TOPS.encode()).hexdigest(),
            "--out", str(tmp_path / "ext.json")]
    for k, v in kw.items():
        args += [f"--{k}", v]
    return X.main(args)


BASE = {X.A3G: 1.90, X.A1BG: 1.96, X.K0G: 1.96, X.A2G: 2.06}


def test_families_and_labels(tmp_path):
    d, reads = build(tmp_path, BASE)
    res = run(tmp_path, d, reads)
    assert sorted(res["families"]) == ["E1", "E3", "E4"]  # E2 withdrawn by the user's direction
    assert [res["families"][f]["holm_family"] for f in ("E1", "E3", "E4")] == [4, 6, 2]  # E4b withdrawn: E4 is k = 2
    fd = res["findings"]
    assert fd["E1"][f"{X.A3G} vs {X.A1BG} moderate"] == "CLAIM"
    assert fd["E3"][f"{X.K0G} vs cd_ib0 heavy"] == "CLAIM"
    assert fd["E4"][f"{X.A2G} vs {X.K0G} heavy"] == "CLAIM"  # K0 the faster, as claimed
    assert list(fd["E4"]) == [f"{X.A2G} vs {X.K0G} heavy"]


def test_frozen_and_cotenancy_must_agree(tmp_path):
    kr = dict(BASE, **{X.A3G: 1.96})
    d, reads = build(tmp_path, BASE, kr)
    res = run(tmp_path, d, reads)
    e = [x for x in res["families"]["E1"]["combined"] if x["rung"] == "moderate"][0]
    assert e["frozen"] == "CLAIM" and e["cotenancy"] != "CLAIM" and e["combined"] == "UNRESOLVED"


def test_reads_must_be_the_registered_seeds_and_all_present(tmp_path):
    d, reads = build(tmp_path, BASE)
    with pytest.raises(SystemExit, match="no extension read"):
        run(tmp_path, d, reads[:-1])
    bad = json.loads(open(reads[0]).read())
    bad["seeds"] = [11, 12]
    open(reads[0], "w").write(json.dumps(bad))
    with pytest.raises(SystemExit, match="registered"):
        run(tmp_path, d, reads)


def test_old_seed_cells_in_the_new_set_fail_loud(tmp_path):
    d, reads = build(tmp_path, BASE)
    cell(d, X.A3G, 1, 1, "g0", "heavy", 1.9)
    with pytest.raises(SystemExit, match="other seeds"):
        run(tmp_path, d, reads)
