import numpy as np

from scripts_cosim.warm_surge_0b_read import backlog_ratio, moved, pct, summarize, window_mean


def test_window_mean_uses_dispatch_time():
    rows = np.array([[0.0, 5.0], [10.0, 12.0], [20.0, 26.0]], dtype=np.float32)
    assert window_mean(rows, 10.0) == 4.0
    assert window_mean(rows, 10.0, 20.0) == 2.0
    assert window_mean(rows, 99.0) is None


def test_pct_and_move_rule():
    assert pct(1.1, 1.0) > 9.99 and pct(None, 1.0) is None
    s = summarize([-2.0] * 9 + [1.0] * 3)
    assert s["wins"] == 9 and moved(s, 12)
    assert not moved(summarize([-2.0] * 8 + [1.0] * 4), 12)
    assert not moved(summarize([-0.5] * 12), 12)


def test_backlog_ratio_flags_growth():
    t = np.linspace(0, 100, 2000)
    growing = np.stack([t, t + 1 + 0.5 * t], 1)
    steady = np.stack([t, t + 2.0], 1)
    assert backlog_ratio(growing, 0.0, 100.0) > 1.5
    assert abs(backlog_ratio(steady, 0.0, 100.0) - 1.0) < 0.2
