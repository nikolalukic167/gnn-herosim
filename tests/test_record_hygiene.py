"""Mechanical checks on the research record.

The routing rule ("one fact, one home"; `LINEAGES.md` is an index only) is stated in
AGENTS.md, in LINEAGES.md's own preamble, in the `experiment-gate` agent and in the
`doc-helper` agent -- and was violated in all four places anyway. Restating it a fifth
time does nothing. These tests make it fail loudly instead.

Run:
    PIPENV_IGNORE_VIRTUALENVS=1 pipenv run python3 -m pytest tests/test_record_hygiene.py -q

Each test names the defect it exists because of. Do not relax a bound to make a test
pass -- fix the document, or make the exemption explicit and dated below.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

INDEX = REPO / "LINEAGES.md"
NODES_DIR = REPO / "docs" / "lineages"
AGENTS_MD = REPO / "AGENTS.md"
CLAUDE_MD = REPO / ".claude" / "CLAUDE.md"

# Record files whose cited paths must resolve in the live tree.
RECORD_FILES = [
    INDEX,
    REPO / "docs" / "lessons.md",
    REPO / "docs" / "hard-stops.md",
    REPO / "docs" / "gates" / "gate-tools.md",
    AGENTS_MD,
]

# --- bounds -----------------------------------------------------------------------
# An index row is a status and a one-line outcome. 400 B is ~2 printed lines and is
# generous: the 23 rows that already complied when this test was written averaged 230 B.
ROW_MAX_BYTES = 400

# A reference file you cannot slice is a file you must read whole. Three separate defects,
# because they have three different cures:
#
#  1. A physical line so long you cannot read part of it. hard-stops.md held 31 closed
#     directions on ONE 12,775 B line, and LINEAGES.md rows reached 9,838 B.
#  2. A prose wall: a run of paragraphs with no heading, list or table to grep for.
#  3. A file so large that even well-formed sections are too far apart to find.
#
# A long bulleted list or a dated table is NOT a defect: both are greppable and loadable in
# part. An earlier version of this test measured only (3) and so demanded headings inside a
# table, which is the wrong cure for the right worry.
MAX_PHYSICAL_LINE_BYTES = 5_000
MAX_PROSE_RUN_BYTES = 8_000
HEADING_REQUIRED_OVER_BYTES = 60_000
MAX_BYTES_BETWEEN_HEADINGS = 50_000

_STRUCTURED = re.compile(r"^(#{1,6} |[-*+] |\d+\. |\| |```|> )")

VALID_STATUSES = {
    "ACTIVE",
    "REGISTERED",
    "PRE-REGISTERED",
    "CLOSED",
    "PARKED",
    "SUPERSEDED",
    "FAILED",
    "FALSIFIED",
    "SYNTHESIS",
    "PAPER",
    "REFERENCE",
}

# Paths that are illustrative rather than real: format templates, glob patterns and
# names that stand for a family of files rather than one file on disk.
PATH_CITATION_EXEMPT = re.compile(
    r"(^|/)(<|\{|\*|\.\.\.)"  # placeholders: <name>.md, {a,b}.py, *.py
    r"|^[A-Za-z0-9_.-]*\*"
    r"|^/tmp/"
    r"|^~"
)


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


# --- LINEAGES.md is an index -------------------------------------------------------


def _index_rows() -> list[tuple[int, str, str, str]]:
    """(line_no, lineage_name, status, outcome_cell) for every lineage row."""
    rows = []
    for i, line in enumerate(_read(INDEX).splitlines(), 1):
        m = re.match(
            r"^\|\s*\[\*\*(?P<name>[A-Za-z0-9_]+)\*\*\]\((?P<link>[^)]+)\)\s*\|"
            r"\s*`(?P<status>[A-Z-]+)`\s*\|(?P<outcome>.*)$",
            line,
        )
        if m:
            rows.append(
                (i, m.group("name"), m.group("status"), m.group("outcome").rstrip("| "))
            )
    return rows


def test_index_has_rows():
    """Guards the parser itself: a format change must not silently disable every check."""
    rows = _index_rows()
    assert len(rows) >= 30, (
        f"Only parsed {len(rows)} lineage rows from LINEAGES.md. The row format "
        f"probably changed; fix the regex in _index_rows() rather than ignoring this."
    )


def test_index_rows_are_one_line_outcomes():
    """LINEAGES.md reached 88,893 B because 12 CLOSED rows carried up to 9,838 B of
    node narrative each. The index is a status and a one-line outcome; the record is
    the node."""
    offenders = [
        (ln, name, len(outcome))
        for ln, name, _status, outcome in _index_rows()
        if len(outcome.encode("utf-8")) > ROW_MAX_BYTES
    ]
    assert not offenders, "Index rows over {} B -- move the narrative into the node:\n{}".format(
        ROW_MAX_BYTES,
        "\n".join(f"  LINEAGES.md:{ln}  {name}  {n} B" for ln, name, n in offenders),
    )


def test_index_rows_have_no_embedded_tables():
    """A row containing a pipe table is a node section that was pasted into the index."""
    offenders = [
        (ln, name)
        for ln, name, _status, outcome in _index_rows()
        if "|" in outcome
    ]
    assert not offenders, "Index rows containing embedded tables:\n" + "\n".join(
        f"  LINEAGES.md:{ln}  {name}" for ln, name in offenders
    )


# --- rows and nodes agree ----------------------------------------------------------


def _node_status(node: Path) -> str | None:
    """The status the node declares for itself, from its own header block."""
    head = _read(node)[:4000]
    m = re.search(r"\*\*Status:?\*\*:?\s*`?([A-Z][A-Z-]{2,})`?", head)
    if m:
        return m.group(1)
    m = re.search(r"^#\s+\S+\s+[-—]+\s+`?([A-Z][A-Z-]{2,})`?", head, flags=re.M)
    return m.group(1) if m else None


def test_every_node_has_an_index_row():
    """Rule 3: a lineage is not done until it has a LINEAGES.md row AND a node.
    drainable_debug_v1 and drainable_serving_config_v1 had nodes and no row."""
    index_text = _read(INDEX)
    missing = [
        n.name
        for n in sorted(NODES_DIR.glob("*.md"))
        if f"docs/lineages/{n.name}" not in index_text
    ]
    assert not missing, "Lineage nodes with no row in LINEAGES.md:\n" + "\n".join(
        f"  docs/lineages/{m}" for m in missing
    )


def test_every_index_row_has_a_node():
    missing = [
        (ln, name)
        for ln, name, _s, _o in _index_rows()
        if not (NODES_DIR / f"{name}.md").exists()
    ]
    assert not missing, "Index rows pointing at a node that does not exist:\n" + "\n".join(
        f"  LINEAGES.md:{ln}  {name}" for ln, name in missing
    )


def test_index_status_matches_node_status():
    """Twelve lineages disagreed with their own node. Six closed on 2026-09-15 or later
    still said REGISTERED in their header, and drainable_regime_v1 was CLOSED in
    CLAUDE.md while the index said REGISTERED."""
    drift = []
    for _ln, name, status, _o in _index_rows():
        node = NODES_DIR / f"{name}.md"
        if not node.exists():
            continue
        declared = _node_status(node)
        if declared and declared != status:
            drift.append((name, status, declared))
    assert not drift, "Status drift between the index and the node's own header:\n" + "\n".join(
        f"  {n:<32} index={i:<14} node={d}" for n, i, d in drift
    )


def test_statuses_are_from_the_vocabulary():
    bad = [
        (ln, name, status)
        for ln, name, status, _o in _index_rows()
        if status not in VALID_STATUSES
    ]
    assert not bad, "Unknown statuses (see LINEAGES.md -> Statuses):\n" + "\n".join(
        f"  LINEAGES.md:{ln}  {name}  {s}" for ln, name, s in bad
    )


# --- the record points at code that exists -----------------------------------------


def _tracked_names() -> tuple[set[str], set[str]]:
    out = subprocess.check_output(["git", "ls-files"], cwd=REPO, text=True).split("\n")
    live, archived = set(), set()
    for f in out:
        if not f:
            continue
        (archived if f.startswith("archive/") else live).add(Path(f).name)
    return live, archived


@pytest.mark.parametrize("doc", RECORD_FILES, ids=lambda p: p.name)
def test_cited_scripts_exist_outside_archive(doc: Path):
    """CLAUDE.md and PARITY.md both named verify_code_identity.py as step 1 of the
    mandatory parity check order; it exists nowhere. lessons.md cited 175 scripts of
    which 74 were deleted or archive-only, and CLAUDE.md says to ignore archive/."""
    if not doc.exists():
        pytest.skip(f"{doc} not present")
    live, archived = _tracked_names()
    cited = set(re.findall(r"`([A-Za-z0-9_./-]+\.(?:py|sh|sbatch))`", _read(doc)))
    dead, only_archived = [], []
    for c in sorted(cited):
        if PATH_CITATION_EXEMPT.search(c):
            continue
        name = Path(c).name
        if name in live:
            continue
        # A file that exists at the cited path but is not yet tracked is new, not dead.
        if (REPO / c).exists():
            continue
        (only_archived if name in archived else dead).append(c)
    msg = []
    if dead:
        msg.append("cited but absent from the repo:\n" + "\n".join(f"    {d}" for d in dead))
    if only_archived:
        msg.append(
            "cited but archive-only (CLAUDE.md says ignore archive/):\n"
            + "\n".join(f"    {d}" for d in only_archived)
        )
    assert not msg, f"{doc.relative_to(REPO)} cites code that is not live:\n" + "\n".join(msg)


# --- a reference file must be sliceable --------------------------------------------


@pytest.mark.parametrize("doc", RECORD_FILES, ids=lambda p: p.name)
def test_large_record_files_are_sliceable(doc: Path):
    """lessons.md put 158,421 of its 171,206 B before the first heading and gate-tools.md
    84,584 of 100,964. A file with no headings has to be read whole."""
    if not doc.exists():
        pytest.skip(f"{doc} not present")
    text = _read(doc)
    rel = doc.relative_to(REPO)

    # (1) no unreadably long physical line
    for n, line in enumerate(text.split("\n"), 1):
        assert len(line.encode("utf-8")) <= MAX_PHYSICAL_LINE_BYTES, (
            f"{rel}:{n} is {len(line.encode('utf-8'))} B on one physical line "
            f"(max {MAX_PHYSICAL_LINE_BYTES}). Split it into one item per line -- you cannot "
            f"read, grep or diff part of a single line."
        )

    # (2) no prose wall
    run, worst, worst_line, start = 0, 0, 1, 1
    for n, line in enumerate(text.split("\n"), 1):
        if _STRUCTURED.match(line) or not line.strip():
            run, start = 0, n + 1
        else:
            run += len(line.encode("utf-8")) + 1
            if run > worst:
                worst, worst_line = run, start
    assert worst <= MAX_PROSE_RUN_BYTES, (
        f"{rel} has a {worst} B run of unstructured prose starting at line {worst_line} "
        f"(max {MAX_PROSE_RUN_BYTES}). Break it with headings or a list."
    )

    # (3) sections close enough to find in a large file
    if len(text.encode("utf-8")) <= HEADING_REQUIRED_OVER_BYTES:
        return
    offsets = [0] + [m.start() for m in re.finditer(r"^#{1,4} ", text, flags=re.M)] + [len(text)]
    gap, at = max(((b - a, a) for a, b in zip(offsets, offsets[1:])), default=(0, 0))
    assert gap <= MAX_BYTES_BETWEEN_HEADINGS, (
        f"{rel} has a {gap} B stretch with no heading (starts at line "
        f"{text[:at].count(chr(10)) + 1}); max is {MAX_BYTES_BETWEEN_HEADINGS} B."
    )


# --- node heads are the read path, so they must be current --------------------------


def _dates(s: str) -> list[str]:
    return re.findall(r"20\d\d-\d\d-\d\d", s)


def _head_of(node: Path) -> str:
    """Everything before the first dated record section."""
    text = _read(node)
    m = re.search(r"^#{2,3} .*(20\d\d-\d\d-\d\d|Record|Bars|Registration)", text, flags=re.M)
    return text[: m.start()] if m else text


@pytest.mark.parametrize(
    "node", sorted(NODES_DIR.glob("*.md")), ids=lambda p: p.stem
)
def test_node_head_is_as_current_as_its_record(node: Path):
    """The head is the orientation read: 12% of node bytes, and the only part most
    sessions need. peer_affinity_v1's head led with the +5.14 pp offline win and
    mentioned none of the reversals that followed, so a head-only read would have
    quoted the one number CLAUDE.md warns in bold not to quote. A head is REWRITTEN
    on every close, never only appended to."""
    text = _read(node)
    all_dates = _dates(text)
    if not all_dates:
        pytest.skip("node carries no dated record")
    head_dates = _dates(_head_of(node))
    assert head_dates, f"{node.name}: head carries no date; it cannot be checked for staleness"
    assert max(head_dates) >= max(all_dates), (
        f"{node.name}: head's newest date is {max(head_dates)} but the record runs to "
        f"{max(all_dates)}. Rewrite the head's standing answer to match the record, or "
        f"the head-only read path reports a superseded result."
    )


# --- ephemera never land in the tree ------------------------------------------------


# AGENTS.md is the canonical instructions file -- `.claude/CLAUDE.md` imports it -- and
# loads automatically on every session, so its size is a tax on all work. 2026-09-24: it
# absorbed CLAUDE.md's operational content (~13 KB), the research-stand narrative (~5 KB)
# and AGENTS.md's own research-entry-point routing (~6 KB) into one file, ~27.6 KB total.
# This leaves modest headroom and still catches a narrative growing unchecked.
AGENTS_MD_MAX_BYTES = 30_000


def test_agents_md_stays_within_budget():
    """Raising this bound is a decision, not a fix.

    If AGENTS.md no longer fits, the question is which section stopped being guidance and
    became a record. Records go in `docs/lineages/`; this file says how to work here.
    """
    n = len(_read(AGENTS_MD).encode("utf-8"))
    assert n <= AGENTS_MD_MAX_BYTES, (
        f"AGENTS.md is {n} B, over the {AGENTS_MD_MAX_BYTES} B budget. It loads on "
        f"every session. Move whatever grew into a lineage node rather than raising the cap."
    )


def test_agents_md_has_standing_answer_stamp():
    """AGENTS.md is the single canonical, continuously-updated research narrative -- the
    file `.claude/CLAUDE.md` imports for Claude Code sessions. A missing stamp means the
    close-a-lineage rewrite habit lapsed."""
    text = _read(AGENTS_MD)
    match = re.search(r"\*\*Where the research question stands \(rewritten (20\d\d-\d\d-\d\d)", text)
    assert match, "AGENTS.md lost its standing-answer stamp"


def test_agents_research_entry_points_resolve():
    """AGENTS.md's 'Current research entry points' section routes to lineage nodes outside
    the headline narrative (L2D / DAG / workflow tracks); every path it cites must exist."""
    text = _read(AGENTS_MD)
    section = text.split("## Current research entry points", 1)[1].split("\n## ", 1)[0]
    paths = re.findall(r"`(docs/lineages/[^`]+\.md)`", section)
    assert paths, "operational guidance must route readers to current research nodes"
    for path in paths:
        assert (REPO / path).is_file(), f"missing research entry point: {path}"


def test_claude_md_is_a_thin_import():
    """`.claude/CLAUDE.md` must stay a thin `@AGENTS.md` import plus a short Claude-only
    appendix, never grow back into a second copy of the research narrative -- the exact
    cross-file duplication this migration removed."""
    text = _read(CLAUDE_MD)
    assert "@../AGENTS.md" in text or "@AGENTS.md" in text, (
        ".claude/CLAUDE.md must import AGENTS.md"
    )
    assert "**Where the research question stands" not in text, (
        ".claude/CLAUDE.md must not duplicate AGENTS.md's research narrative -- "
        "add Claude-specific notes only, below the import"
    )
    n = len(text.encode("utf-8"))
    assert n <= 2_000, (
        f".claude/CLAUDE.md is {n} B, over the 2,000 B thin-import budget. "
        f"Move substantive content into AGENTS.md."
    )



def test_no_live_file_references_an_archive_only_filename():
    """CLAUDE.md rule 1: the live tree is verified closed against `archive/`.

    This gate used to live as a paste-and-run Python block inside LINEAGES.md's
    Conventions section, which meant it only ran when someone remembered it. It is a
    check, so it belongs in the test suite.
    """
    tracked = [
        f
        for f in subprocess.check_output(["git", "ls-files"], cwd=REPO, text=True).split("\n")
        if f
    ]
    live = [f for f in tracked if not f.startswith("archive/")]
    code = (".py", ".sh", ".sbatch")
    archived_names = {
        Path(f).name for f in tracked if f.startswith("archive/") and f.endswith(code)
    }
    archived_names -= {Path(f).name for f in live}

    # Prose that explains why a guard exists may name a code path that no longer does.
    KNOWN_BENIGN = {"executeinitial.py"}
    archived_names -= KNOWN_BENIGN

    if not archived_names:
        return
    # One compiled alternation, not a loop per name: the original paste-in block was
    # O(files x lines x names) and took minutes.
    pattern = re.compile(
        r"(?<![\w.-])(" + "|".join(re.escape(n) for n in sorted(archived_names)) + r")(?![\w])"
    )
    bad = []
    for f in live:
        if not f.endswith(code):
            continue
        for lineno, line in enumerate(
            (REPO / f).read_text(errors="ignore").split("\n"), 1
        ):
            if "archive/" in line:
                continue
            m = pattern.search(line)
            if m:
                bad.append(f"{f}:{lineno} -> {m.group(1)}")
    assert not bad, "Live files referencing archive-only filenames:\n" + "\n".join(
        f"  {b}" for b in bad
    )


def test_no_handover_files_committed():
    """CLAUDE.md: session handovers are ephemeral and never committed."""
    out = subprocess.check_output(["git", "ls-files"], cwd=REPO, text=True).split("\n")
    bad = [
        f
        for f in out
        if f
        # An agent or slash-command *definition* named handover is the tool that writes
        # one; it is not itself a handover document.
        and not f.startswith((".claude/agents/", ".claude/commands/", ".cursor/commands/"))
        and re.search(r"(^|/)HANDOVER[^/]*\.md$", f, flags=re.I)
    ]
    assert not bad, "Committed handover files (these are ephemeral):\n" + "\n".join(
        f"  {b}" for b in bad
    )
