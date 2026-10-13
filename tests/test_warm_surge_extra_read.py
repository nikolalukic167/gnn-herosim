from scripts_cosim.warm_surge_extra_read import is_collapse


def test_collapse_rule():
    assert is_collapse(3.1, 1.0, 10.0) and not is_collapse(2.9, 1.0, 10.0)
    assert is_collapse(1.0, 1.0, 121.0) and not is_collapse(1.0, 1.0, 120.0)
