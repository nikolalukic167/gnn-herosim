"""x11_confirm_v1 reader: per-topology median over (seed, draw) paired against CD on the same draw."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import x11_confirm_v1_read as X  # noqa: E402

TOPOS = list(range(9401, 9411))


def _s(learned_factor, seeds=(1, 2, 3, 4), drop=None):
    s = {}
    for t in TOPOS:
        for k in X.DRAWS:
            w = X._win(k)
            s[(t, w, "cd", 0)] = {"averageElapsedTime": 10.0 + k}
            for sd in seeds:
                if (t, w, sd) == drop:
                    continue
                s[(t, w, X.LEARNED, sd)] = {"averageElapsedTime": (10.0 + k) * learned_factor(t, sd)}
    return s


def test_confirmed_when_every_topology_is_ten_percent_faster():
    c = X.contrast(_s(lambda t, sd: 0.9), TOPOS, (2, 3, 4))
    assert abs(c["read"]["median_pct"] + 10.0) < 1e-9 and c["read"]["a_faster"] == 10
    assert X.label(c) == "CONFIRMED"


def test_q1_ignores_seed_one():
    # seed 1 is 50 % faster, seeds 2-4 are 1 % slower: Q1 must not see seed 1
    c = X.contrast(_s(lambda t, sd: 0.5 if sd == 1 else 1.01), TOPOS, (2, 3, 4))
    assert c["read"]["median_pct"] > 0 and X.label(c) == "NOT-CONFIRMED"


def test_missing_run_drops_the_topology_by_name():
    c = X.contrast(_s(lambda t, sd: 0.9, drop=(9401, "w0x11d3", 2)), TOPOS, (2, 3, 4))
    assert "9401" in c["dropped"] and c["read"]["n"] == 9
