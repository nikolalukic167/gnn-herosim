import json
import subprocess
import sys
import textwrap

from src.placement.progress_watchdog import PROJECTED, STALL, ProgressWatchdog


def run(wd, rate_fn, t_end, step=30.0):
    t, out = 0.0, None
    while t < t_end and out is None:
        t += step
        out = wd.update(t, rate_fn(t))
    return out, t


def test_a_cell_that_will_finish_is_left_alone():
    wd = ProgressWatchdog(50000, 7200)
    out, _ = run(wd, lambda t: 10.0 * t, 3600)  # 10 trigger/s: done at 5,000 s
    assert out is None


def test_a_creeping_cell_is_killed_after_the_warmup_not_before():
    wd = ProgressWatchdog(50000, 7200)
    out, t = run(wd, lambda t: 1.4 * t, 3600)  # 1.4/s projects to ~36,000 s > 1.5 x 7,200
    assert out == PROJECTED and 600 <= t <= 660


def test_the_margin_spares_a_slow_but_in_time_cell():
    wd = ProgressWatchdog(50000, 7200)
    out, _ = run(wd, lambda t: 5.0 * t, 3600)  # projects to 10,000 s: over the limit but under 1.5 x
    assert out is None


def test_no_progress_for_five_minutes_is_a_stall_at_any_time():
    wd = ProgressWatchdog(50000, 7200)
    out, t = run(wd, lambda t: min(100.0 * t, 3000.0), 3600)  # moves for 30 s then stops
    assert out == STALL and 330 <= t <= 360


def test_a_finished_run_is_never_killed_and_progress_never_runs_backwards():
    wd = ProgressWatchdog(100, 10, warmup_s=0)
    assert wd.update(1, 100) is None and wd.update(9999, 100) is None
    wd = ProgressWatchdog(50000, 7200)
    wd.update(10, 500)
    wd.update(20, 100)
    assert wd.samples[-1][1] == 500


def test_the_runner_kills_a_stalled_command_and_returns_137(tmp_path):
    prog = tmp_path / "p.jsonl"
    script = tmp_path / "w.py"
    script.write_text(textwrap.dedent(f"""
        import json, time
        open({str(prog)!r}, "a").write(json.dumps({{"trigger_task_id": 5}}) + "\\n")
        time.sleep(60)
    """))
    r = subprocess.run([sys.executable, "scripts_cosim/watchdog_run.py", "--progress-file", str(prog), "--total", "50000", "--limit", "600",
                        "--stall", "1", "--poll", "0.3", "--", sys.executable, str(script)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 137 and "stall" in r.stderr


def test_a_sweep_that_cannot_finish_in_its_budget_is_stopped_early():
    import concurrent.futures
    import time

    from src.executecosimulation import _completed_or_stalled

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        futs = {ex.submit(time.sleep, 0.2): i for i in range(30)}  # 6 s of work
        stall = {"pending": 0}
        wd = ProgressWatchdog(len(futs), 0.5, warmup_s=0.0, stall_s=1e9)  # budget 0.5 s x 1.5 margin
        got = list(_completed_or_stalled(futs, None, stall, wd, poll_s=0.1))
        for f in futs:
            f.cancel()
    assert 0 < len(got) < 30 and stall["pending"] == 30 - len(got) and "projected" in stall["watchdog"]


def _lumpy(t):
    """a cell on track (1.6 sim-s per wall-s) whose clock sits almost still for 400 s inside one big ILS burst (0.0223/s, the rate that killed the PER 200 cells)"""
    return 1.6 * t if t < 3000 else 4800 + 0.0223 * (t - 3000) if t < 3400 else 4809 + 1.6 * (t - 3400)


def test_a_frozen_sim_window_inside_an_on_track_run_kills_the_default_rule_but_not_the_whole_run_rate():
    out, _ = run(ProgressWatchdog(9354, 72000, stall_s=7200), _lumpy, 3400)
    assert out == PROJECTED  # 300 s window: the projected finish is ~120,000 s
    out, _ = run(ProgressWatchdog(9354, 72000, stall_s=7200, window_s=1e12), _lumpy, 4800)
    assert out is None


def test_the_whole_run_rate_still_kills_a_genuine_sustained_slowdown():
    out, t = run(ProgressWatchdog(9354, 72000, stall_s=7200, window_s=1e12), lambda t: 0.02 * t, 4800)  # 0.02 sim-s/s: finish at ~468,000 s
    assert out == PROJECTED and 600 <= t <= 660


def test_the_progress_reporter_appends_a_series(tmp_path):
    import time

    import simpy

    from src.placement.progress import start_progress_reporter

    p = tmp_path / "x.progress"
    stop = start_progress_reporter(simpy.Environment(), str(p), interval=0.05)
    time.sleep(0.4)
    stop.set()
    rows = (tmp_path / "x.progress.series").read_text().split()
    assert len(rows) >= 4 and len(rows) % 2 == 0
