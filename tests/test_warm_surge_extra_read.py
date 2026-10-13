from scripts_cosim.warm_surge_extra_read import is_collapse, tail_flag


def test_collapse_rule():
    assert is_collapse(3.1, 1.0, 10.0) and not is_collapse(2.9, 1.0, 10.0)
    assert not is_collapse(1.0, 1.0, 500.0)
    assert tail_flag(121.0) and not tail_flag(120.0)
