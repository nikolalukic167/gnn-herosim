"""literature_reeval_v1 -- import the upstream L2D code (Zhang et al., NeurIPS 2020) unmodified.

The upstream repo is NOT vendored (no licence file in it); it is expected at ``L2D_ROOT``
(default ``/root/projects/L2D``, override with the env var). Upstream ``Params.py`` parses
``sys.argv`` at import time, so this module temporarily swaps argv while importing it and
restores it afterwards. Nothing upstream is edited -- every arm difference lives in
``arm_features.py`` (what the policy is fed), never in the model code.

Upstream needs ``gym`` only for ``gym.Env`` / ``gym.utils.EzPickle``; ``gymshim/`` supplies
those two names so the herosim pipenv (torch 2.5) can run it without installing gym 0.17.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

L2D_ROOT = os.environ.get("L2D_ROOT", "/root/projects/L2D")
_HERE = os.path.dirname(os.path.abspath(__file__))

_loaded: SimpleNamespace | None = None


def load_l2d(n_j: int, n_m: int, device: str = "cpu") -> SimpleNamespace:
    """Import upstream modules with ``configs`` bound to (n_j, n_m, device). Idempotent per process."""
    global _loaded
    if _loaded is not None:
        if (_loaded.configs.n_j, _loaded.configs.n_m) != (n_j, n_m):
            raise RuntimeError(
                f"L2D already imported for {_loaded.configs.n_j}x{_loaded.configs.n_m}; "
                f"upstream Params is process-global, start a new process for {n_j}x{n_m}"
            )
        return _loaded
    if not os.path.isdir(L2D_ROOT):
        raise FileNotFoundError(
            f"L2D_ROOT={L2D_ROOT!r} not found. Clone https://github.com/zcaicaros/L2D there "
            f"(commit 7b2efbb used for literature_reeval_v1)."
        )
    for p in (os.path.join(_HERE, "gymshim"), L2D_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
    saved_argv = sys.argv
    sys.argv = ["l2d", "--device", device, "--n_j", str(n_j), "--n_m", str(n_m)]
    try:
        from Params import configs  # noqa: E402
        from JSSP_Env import SJSSP  # noqa: E402
        from models.actor_critic import ActorCritic  # noqa: E402
        from mb_agg import g_pool_cal, aggr_obs  # noqa: E402
        from agent_utils import select_action, greedy_select_action, eval_actions  # noqa: E402
        from uniform_instance_gen import uni_instance_gen  # noqa: E402
        import PPO_jssp_multiInstances as ppo_mod  # noqa: E402
    finally:
        sys.argv = saved_argv
    _loaded = SimpleNamespace(
        configs=configs, SJSSP=SJSSP, ActorCritic=ActorCritic, g_pool_cal=g_pool_cal,
        aggr_obs=aggr_obs, select_action=select_action, greedy_select_action=greedy_select_action,
        eval_actions=eval_actions, uni_instance_gen=uni_instance_gen, ppo_mod=ppo_mod,
        root=L2D_ROOT,
    )
    return _loaded


def upstream_commit() -> str:
    head = os.path.join(L2D_ROOT, ".git", "HEAD")
    try:
        with open(head) as fh:
            ref = fh.read().strip()
        if ref.startswith("ref:"):
            with open(os.path.join(L2D_ROOT, ".git", ref.split(" ", 1)[1])) as fh:
                return fh.read().strip()
        return ref
    except OSError:
        return "unknown"
