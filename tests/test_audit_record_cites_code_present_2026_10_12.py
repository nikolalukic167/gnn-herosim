"""Audit 2026-10-12: the record on this branch cites code that must exist on this branch.

LINEAGES.md and `docs/lineages/{ib0_confirm_v1,plan_rerank_v1,slate_bar_v1}.md` and the dated table in
`docs/gates/gate-tools.md` define the current hand bar (`cd_exactS_ib0` = `peer_greedy_network_cd` with
`HEROSIM_PG_BATCH_ALPHA=0` over the exact search `HEROSIM_PG_CD_EXACT=1`) and the slate fix
(`src/placement/declared_slate.py`, rule `declared_pruning_v1_uncapped`). A reader of this branch must be able to find
every one of those in `src/` or `scripts_cosim/`; otherwise the cited numbers are reproducible only from an unmerged
`rp/*` branch and the record's own rule ("a result never written down gets re-run") is met in letter, not in substance.

Expected on 2026-10-12: FAILS on `reference-physics` (the code is on rp/plan-rerank d192081f and rp/slate-fix 4d90119c,
unmerged); passes once those branches are merged. The failure message lists every missing citation.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ("src", "scripts_cosim")

# (where the record cites it, what must exist in the code tree)
CITATIONS = [
    ("docs/lineages/ib0_confirm_v1.md", "HEROSIM_PG_BATCH_ALPHA"),
    ("docs/lineages/plan_rerank_v1.md", "HEROSIM_PG_DRAIN_ALPHA"),
    ("docs/gates/gate-tools.md", "cd_exactS"),
    ("LINEAGES.md", "cd_exactS_ib0"),
    ("docs/gates/gate-tools.md", "cd_exactS_ib0"),
    ("docs/gates/gate-tools.md", "src/placement/declared_slate.py"),
    ("docs/lineages/slate_bar_v1.md", "declared_pruning_v1"),
]


def _record_mentions(doc: str, token: str) -> bool:
    text = (ROOT / doc).read_text(errors="replace")
    return token in text


def _code_has(token: str) -> bool:
    if token.endswith(".py"):
        return (ROOT / token).is_file()
    pat = re.compile(re.escape(token))
    for d in CODE_DIRS:
        for p in (ROOT / d).rglob("*.py"):
            if "archive" in p.parts:
                continue
            try:
                if pat.search(p.read_text(errors="replace")):
                    return True
            except OSError:
                continue
    return False


def test_every_cited_flag_module_and_arm_exists_in_this_branch_code():
    missing = []
    for doc, token in CITATIONS:
        if not (ROOT / doc).is_file():
            continue
        if not _record_mentions(doc, token):
            pytest.fail(f"{doc} no longer mentions {token!r}; update CITATIONS")
        if not _code_has(token):
            missing.append(f"{token!r} (cited in {doc})")
    assert not missing, (
        "the record on this branch cites code that is not on this branch (unmerged rp/* work):\n  "
        + "\n  ".join(missing)
    )
