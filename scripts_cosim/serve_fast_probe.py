"""rp/gnn-serve-fast identity probe: records every live GNN decision of one cell so two runs (flags off / on) can be compared.

Installed in a cell process by a one-line sitecustomize.py at the worktree root (not tracked: it is a measurement hook, never part of
serving). Active only when SERVEFAST_PROBE_OUT is set and the process is src/executesimulation.py. Per decision it keeps
  g       sha1 of the built graph (task / platform features, edge_index, edge_attr, node_edge_index, the candidate mapping),
  cols    sha1 of each step's partial_state_columns block,
  task    the task index of each step,
  logits  each step's logits as float32,
  plan    the decoded plan.
A wrapper reads values and returns them unchanged. scripts_cosim/serve_fast_compare.py compares two such files.
"""
from __future__ import annotations

import atexit
import hashlib
import importlib.abc
import importlib.util
import os
import pickle
import sys


def install(out_path: str) -> None:
    DEC: list = []
    CUR = {"rec": None, "g": None}

    def sha(*chunks: bytes) -> str:
        h = hashlib.sha1()
        for c in chunks:
            h.update(c)
        return h.hexdigest()

    def tb(x):
        return x.detach().cpu().numpy().tobytes() if hasattr(x, "detach") else bytes(repr(x), "utf8")

    def p_sched(m):
        orig = m.build_pyg_inference_graph

        def w(*a, **k):
            graph, mapping = orig(*a, **k)
            if graph is None:
                CUR["g"] = None
            else:
                CUR["g"] = sha(tb(graph.task_features), tb(graph.platform_features), tb(graph.edge_index), tb(graph.edge_attr),
                               tb(graph.node_edge_index), repr(sorted((int(k_), [tuple(map(int, c)) for c in v]) for k_, v in mapping.items())).encode())
            return graph, mapping
        m.build_pyg_inference_graph = w

    def p_prefix(m):
        orig_make = m.make_partial_state_score_fn

        def make(*a, **k):
            score = orig_make(*a, **k)

            def s(task_idx, committed):
                out = score(task_idx, committed)
                rec = CUR["rec"]
                if rec is not None:
                    rec["task"].append(int(task_idx))
                    rec["logits"].append(out.detach().float().cpu().numpy().copy())
                return out
            for attr in ("encode_count", "close", "anchor_logits", "snapshot_key"):
                if hasattr(score, attr):
                    setattr(s, attr, getattr(score, attr))
            return s
        m.make_partial_state_score_fn = make

        orig_dec = m.decode_prefix_conditioned

        def dec(*a, **k):
            rec = {"g": CUR["g"], "task": [], "cols": [], "logits": [], "plan": None}
            CUR["rec"] = rec
            try:
                combo = orig_dec(*a, **k)
            finally:
                CUR["rec"] = None
            rec["plan"] = None if combo is None else [tuple(int(v) for v in c) for c in combo]
            DEC.append(rec)
            return combo
        m.decode_prefix_conditioned = dec

    def p_pse(m):
        orig = m.partial_state_columns

        def w(*a, **k):
            out = orig(*a, **k)
            rec = CUR["rec"]
            if rec is not None:
                rec["cols"].append(sha(out.tobytes()))
            return out
        m.partial_state_columns = w

    PATCH = {"src.policy.gnn.scheduler": p_sched, "src.policy.gnn.prefix_serving": p_prefix,
             "src.policy.gnn.partial_state_edges": p_pse}
    seen: set = set()

    class Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in PATCH or name in seen:
                return None
            seen.add(name)
            spec = importlib.util.find_spec(name)
            if spec is None:
                return None
            orig = spec.loader.exec_module

            def exec_module(mod, _o=orig, _n=name):
                _o(mod)
                PATCH[_n](mod)
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, Hook())

    def dump() -> None:
        with open(out_path, "wb") as fh:
            pickle.dump(DEC, fh)
    atexit.register(dump)


def maybe_install() -> None:
    out = os.environ.get("SERVEFAST_PROBE_OUT")
    if not out:
        return
    try:
        cmd = open("/proc/self/cmdline", "rb").read().decode(errors="ignore")
    except Exception:
        return
    if "executesimulation.py" in cmd:
        argv = cmd.split("\0")
        tag = os.path.basename(argv[argv.index("--output") + 1]).replace(".raw.json", "") if "--output" in argv else str(os.getpid())
        install(f"{out}.{tag}.pkl")
