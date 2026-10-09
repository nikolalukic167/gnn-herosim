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
