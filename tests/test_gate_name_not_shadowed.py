import ast
import pathlib


def test_run_one_never_rebinds_name_in_a_loop():
    """`name` is the cell name that .tasks.npy / .summary.json are written under; a `for name in ...` check loop clobbered it (plan_rerank 6ad51eaf)."""
    src = pathlib.Path(__file__).resolve().parents[1] / "scripts_cosim" / "fresh_topo_burst_v1_gate.py"
    fn = next(n for n in ast.parse(src.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "run_one")
    bad = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "name"]
    assert not bad, bad
