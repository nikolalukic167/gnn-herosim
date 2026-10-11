import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import prune_g8_screen as P  # noqa: E402


def row(rung, order, gain, pb=1, passg=5.0, ds="x"):
    return {"ds": ds, "rung": rung, "order": order, "prune": {"pg_exact_prune_batches": pb, "pg_exact_prune_gain_fb_seconds": gain * pb,
                                                                 "pg_exact_prune_gain_pass_seconds": passg * pb, "pg_exact_prune_differs": 1 if gain else 0,
                                                                 "pg_exact_prune_set_differs": pb}}


def test_over_cap_share_gain_shares_and_go(capsys):
    rows = []
    for rung in ("moderate", "heavy"):
        rows += [row(rung, "cost", g, ds=f"{rung}{i}") for i, g in enumerate((-1.0, 0.0, 0.5))]
        rows += [{"ds": f"{rung}3", "rung": rung, "order": "cost", "prune": {"pg_exact_prune_batches": 0}},
                 {"ds": f"{rung}3", "rung": rung, "order": "gnn", "prune": {"pg_exact_prune_batches": 0}}]
        rows += [row(rung, "gnn", g, ds=f"{rung}{i}") for i, g in enumerate((0.2, 1.0, 2.0))]
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


def test_orders_are_compared_on_the_same_states_only():
    rows = [row("heavy", "cost", 1.0), row("heavy", "gnn", 2.0)]
    rows[0]["ds"], rows[1]["ds"] = "a", "a"
    b = row("heavy", "cost", -5.0)
    b["ds"] = "b"
    rows.append(b)
    rows.append({"ds": "b", "rung": "heavy", "order": "gnn", "error": "queued tasks did not run on their recorded replicas"})
    s = P.summarise(rows)
    assert s["heavy|cost"]["states"] == 1 and s["heavy|cost"]["dropped_unpaired"] == 1 and s["heavy|gnn"]["errors"] == 1
    assert s["heavy|cost"]["gain_vs_expansion"]["median"] == pytest.approx(1.0)
