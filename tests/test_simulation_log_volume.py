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

2026-09-14 adds the second half of the same defect. The handler filtered INFO, but the ROOT
logger sat at DEBUG, so every dropped call still built a full `LogRecord` -- including the
stack walk `%(funcName)s` forces -- before the handler threw it away. py-spy put 48 % of
samples in that dead path. The root level must therefore be at least as high as the handler's,
and `logging.info(...)` must return before a record exists.
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
        level=logging.ERROR,
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


def test_the_root_level_is_not_below_the_handler_level():
    """The 2026-09-14 half: a root logger below its only handler pays to build every record
    it then discards. Whatever the levels are, a per-event INFO call must be refused by
    `isEnabledFor` -- i.e. before `LogRecord.__init__` and its stack walk ever run."""
    root = logging.getLogger()
    saved = list(root.handlers), root.level
    try:
        logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
        _configure_like_start_simulation()

        handler_level = min(h.level for h in root.handlers)
        assert root.level >= handler_level, (
            f"root at {root.level} below handler at {handler_level}: every dropped record "
            "is still fully constructed"
        )
        assert not root.isEnabledFor(logging.INFO)
        assert not root.isEnabledFor(logging.DEBUG)
        assert root.isEnabledFor(logging.ERROR)
    finally:
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])


def test_the_source_matches_the_configuration_this_file_pins():
    """`_configure_like_start_simulation` is a copy of the real block; if the source drifts,
    every assertion here goes on testing the copy. Pin the two levels in the source itself."""
    import re

    src = (Path(__file__).resolve().parents[1] / "src/placement/simulation.py").read_text()
    call = re.search(r"console_handler\.setLevel\((logging\.\w+)\)", src)
    basic = re.search(r"logging\.basicConfig\(\s*level=(logging\.\w+)", src)
    assert call and basic, "the logging block in simulation.py no longer has the shape this test reads"
    assert getattr(logging, basic.group(1).split(".")[1]) >= getattr(
        logging, call.group(1).split(".")[1]
    ), f"source has root={basic.group(1)} below handler={call.group(1)}"
