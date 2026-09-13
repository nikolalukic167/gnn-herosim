"""The simulator's log configuration must win over a module that configured logging on import.

2026-09-13. `logging.basicConfig` does nothing when the root logger already has a handler.
`src/notebooks/prepare_graphs_cache.py` calls `basicConfig(level=INFO)` at module import, and
loading a prefix-conditioned checkpoint imports it (`prefix_serving._task_type_vocab`). So by
the time `start_simulation` asked for "ERROR only, on stdout", the root logger already carried
an INFO handler on stderr and the request was discarded in silence.

Consequence, measured: every per-event `logging.info` in `infrastructure.py` -- six or so per
task -- reached stderr, ~390 MB of log for one 450,729-task arm. A 102-arm gate wrote ~40 GB,
exhausted the 250 GiB /home quota on datalab and killed 128 arms with exit 1 and no traceback,
because the traceback could not be written either.

These tests pin the configuration, not the volume: whatever a previously-imported module did,
`start_simulation`'s handler must be the one that serves, and INFO must not reach stderr.
"""

import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _configure_like_start_simulation() -> None:
    """The exact call `start_simulation` makes (src/placement/simulation.py)."""
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.ERROR)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s [%(funcName)18s() ] %(message)s",
        handlers=[console_handler],
        force=True,
    )


def test_an_import_time_basicconfig_does_not_survive_the_simulator_config():
    root = logging.getLogger()
    saved = list(root.handlers), root.level
    try:
        # what prepare_graphs_cache does at import
        logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
        assert root.handlers, "precondition: the import-time config installed a handler"

        _configure_like_start_simulation()

        assert len(root.handlers) == 1, [type(h) for h in root.handlers]
        assert root.handlers[0].level == logging.ERROR
        assert root.handlers[0].stream is sys.stdout
    finally:
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])


def test_per_event_info_never_reaches_the_installed_handler():
    """The 390 MB/arm regression, as an assertion: an INFO record like the simulator's own
    per-task lines must be filtered by the handler the simulator installed."""
    root = logging.getLogger()
    saved = list(root.handlers), root.level
    try:
        logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
        _configure_like_start_simulation()

        captured = io.StringIO()
        root.handlers[0].stream = captured
        logging.info("[ 102380.8 ] Task 352694 (dnn2) arrived on Node 21")
        logging.error("a real failure")

        out = captured.getvalue()
        assert "arrived on Node" not in out, out
        assert "a real failure" in out
    finally:
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])
