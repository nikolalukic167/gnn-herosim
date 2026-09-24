#!/usr/bin/env python3
"""Advisory check after an edit to the research record.

The routing rule ("LINEAGES.md is an index only", "one fact, one home") is stated in
AGENTS.md, in LINEAGES.md, in the experiment-gate agent and in the doc-helper agent, and
on 2026-09-16 was found violated in all four: index rows up to 9,838 bytes, twelve
statuses disagreeing with their own node, and an auto-loaded summary that had stopped
tracking the work six lineages earlier.

A fifth copy of the rule would not have helped. This runs the check instead.

Advisory on purpose: it reports, it does not block. An edit in progress is allowed to be
half-finished; what must not happen is finishing and never noticing. Exit code is always
0 so a mid-edit failure never interrupts the work.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Only these files are governed by the check; editing a lineage node or a test is not
# itself a reason to run it.
WATCHED = {
    "LINEAGES.md",
    "AGENTS.md",
    "docs/lessons.md",
    "docs/hard-stops.md",
    "docs/gates/gate-tools.md",
}
WATCHED_PREFIXES = ("docs/lineages/",)


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    if data.get("tool_name") not in ("Write", "Edit", "MultiEdit"):
        return 0

    repo = Path(os.environ.get("CLAUDE_PROJECT_DIR", ".")).resolve()
    raw = (data.get("tool_input") or {}).get("file_path")
    if not raw:
        return 0
    try:
        rel = Path(raw).resolve().relative_to(repo).as_posix()
    except ValueError:
        return 0

    if rel not in WATCHED and not rel.startswith(WATCHED_PREFIXES):
        return 0

    # All Python in this repo goes through pipenv; a stray local .venv hijacks `pipenv run`,
    # hence PIPENV_IGNORE_VIRTUALENVS. Bare python3 has no pytest here.
    try:
        proc = subprocess.run(
            ["pipenv", "run", "python3", "-m", "pytest",
             "tests/test_record_hygiene.py", "-q", "--no-header"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PIPENV_IGNORE_VIRTUALENVS": "1", "VIRTUAL_ENV": ""},
        )
    except Exception:
        return 0  # never let the reminder break the session

    combined = proc.stdout + proc.stderr
    # An environment that cannot run the check is not a failing record. Staying quiet here
    # is the point: a reminder that fires when nothing is wrong gets ignored when it is.
    if "No module named pytest" in combined or "no such command" in combined.lower():
        return 0

    if proc.returncode != 0:
        failures = [
            ln.split("::")[-1].split(" ")[0]
            for ln in proc.stdout.splitlines()
            if ln.startswith("FAILED")
        ]
        print(
            "[record-hygiene] "
            + (", ".join(failures[:6]) if failures else "checks failing")
            + "\n  The research record is inconsistent. Run:\n"
            "    PIPENV_IGNORE_VIRTUALENVS=1 pipenv run python3 -m pytest "
            "tests/test_record_hygiene.py -q\n"
            "  Closing a lineage? Use the close-a-lineage skill.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
