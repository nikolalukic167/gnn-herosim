"""load_repr_v1's L1/L2 labels are the registered table (docs/lineages/load_repr_v1.md)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_cosim"))
import load_repr_v1_read as R  # noqa: E402


def _r(med, p, verdict):
    return {"read": {"median_pct": med, "p": p, "verdict": verdict}}


def test_l2_labels():
    assert R.l2_label(_r(-6.0, 0.01, "V4LOAD-FASTER")) == "LOAD-HELPS"
    assert R.l2_label(_r(-2.0, 0.01, "V4LOAD-FASTER (direction only, |median| < 5 %)")) == \
        "LOAD-HELPS (direction only)"
    assert R.l2_label(_r(3.0, 0.01, "V4TWIN-FASTER (direction only, |median| < 5 %)")) == "LOAD-HURTS"
    assert R.l2_label(_r(-2.0, 0.2, "NOT-SEPARATED")) == "NOT-SEPARATED"
    assert R.l2_label({"verdict": "DESIGN-SHORT"}) == "DESIGN-SHORT"


def test_l1_labels():
    assert R.l1_label(_r(-1.0, 0.01, "V4LOAD-FASTER"), "NOT-SEPARATED") == "CLOSES-GAP"
    assert R.l1_label(_r(4.0, 0.3, "NOT-SEPARATED"), "NOT-SEPARATED") == "CLOSES-GAP"
    assert R.l1_label(_r(8.0, 0.002, "CD-FASTER"), "LOAD-HELPS (direction only)") == "NARROWS"
    assert R.l1_label(_r(8.0, 0.002, "CD-FASTER"), "NOT-SEPARATED") == "NO-EFFECT"
