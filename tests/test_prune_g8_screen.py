import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import prune_g8_screen as P  # noqa: E402


def row(rung, order, gain, pb=1, passg=5.0):
    return {"ds": "x", "rung": rung, "order": order, "prune": {"pg_exact_prune_batches": pb, "pg_exact_prune_gain_fb_seconds": gain * pb,
                                                                 "pg_exact_prune_gain_pass_seconds": passg * pb, "pg_exact_prune_differs": 1 if gain else 0,
                                                                 "pg_exact_prune_set_differs": pb}}


def test_over_cap_share_gain_shares_and_go(capsys):
    rows = []
    for rung in ("moderate", "heavy"):
        rows += [row(rung, "cost", g) for g in (-1.0, 0.0, 0.5)] + [{"ds": "y", "rung": rung, "order": "cost", "prune": {"pg_exact_prune_batches": 0}}]
        rows += [row(rung, "gnn", g) for g in (0.2, 1.0, 2.0)]
    s = P.summarise(rows)
    c = s["moderate|cost"]
    assert c["states"] == 4 and c["over_cap_states"] == 3 and c["over_cap_share"] == pytest.approx(0.75)
    assert c["gain_vs_expansion"]["median"] == pytest.approx(0.0) and c["gain_vs_expansion"]["neg"] == pytest.approx(1 / 3)
    assert s["heavy|gnn"]["gain_vs_expansion"]["median"] == pytest.approx(1.0) and s["heavy|gnn"]["gain_vs_expansion"]["pos"] == 1.0

    class A:
        rows = []
    import json, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write("\n".join(json.dumps(r) for r in rows))
    A.rows = [fh.name]
    P.read(A)
    out = capsys.readouterr().out
    assert "SCREEN_GO True" in out


def test_errors_are_counted_not_dropped():
    s = P.summarise([row("heavy", "cost", 1.0), {"ds": "z", "rung": "heavy", "order": "cost", "error": "boom"}])
    assert s["heavy|cost"]["errors"] == 1 and s["heavy|cost"]["states"] == 1
