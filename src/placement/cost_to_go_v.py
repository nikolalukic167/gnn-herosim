"""cost_to_go_v1 gate: the value term that re-ranks cd_exactS's top-5 plans by dS + lambda * dV (HEROSIM_PG_CD_EXACT_V).

  HEROSIM_PG_CD_EXACT_V=hand   the hand V (own service + cold, cost_to_go_handv.hand_v on load_after), lambda default 0.1
  HEROSIM_PG_CD_EXACT_V=mlp    the rung-(b) MLP V (no in-flight variant), HEROSIM_PG_CD_EXACT_V_CKPT=<.pt with .contract.json>,
                               lambda from the contract
  HEROSIM_PG_CD_EXACT_V_LAMBDA overrides lambda; unset or empty: the default above.

The features are computed by the SAME functions that built the offline training and evaluation rows (scripts_cosim:
cost_to_go_handv_features.post_commit_features, cost_to_go_handv.type_rates / hand_v, cost_to_go_mlpv_data.replica_matrix), so a
live decision and its replay cannot read differently; the arrival rates come from the run's own trace file, HEROSIM_PG_CD_EXACT_V_TRACE
(events with timestamp <= now over the 60 s lookback, as offline). H = 5 s, fixed by cost_to_go_v1 S1.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

V_ENV = "HEROSIM_PG_CD_EXACT_V"
V_CKPT_ENV = "HEROSIM_PG_CD_EXACT_V_CKPT"
V_LAMBDA_ENV = "HEROSIM_PG_CD_EXACT_V_LAMBDA"
V_TRACE_ENV = "HEROSIM_PG_CD_EXACT_V_TRACE"  # the cell's workload JSON (the live run's trace_file is a name, not a path)
V_DUMP_ENV = "HEROSIM_PG_CD_EXACT_V_DUMP"  # parity check only: one JSON line per decision (time, batch ids, plans, features, rates)
H = 5.0
LOOKBACK = 60.0
K = 5


class ValueTerm:
    def __init__(self, kind: str):
        if kind not in ("hand", "mlp"):
            raise ValueError(f"FAIL LOUD: {V_ENV}={kind!r}; expected hand or mlp")
        self.kind = kind
        self.model = None
        lam = os.environ.get(V_LAMBDA_ENV, "").strip()
        if kind == "mlp":
            import torch

            from scripts_cosim.cost_to_go_mlpv_data import FEATURES
            from scripts_cosim.train_value_mlp import ValueMLP

            ckpt = os.environ.get(V_CKPT_ENV, "").strip()
            if not ckpt:
                raise ValueError(f"FAIL LOUD: {V_ENV}=mlp needs {V_CKPT_ENV}")
            contract = json.loads(Path(ckpt).with_suffix(".contract.json").read_text())
            if list(contract["features"]) != list(FEATURES) or not contract["no_inflight"]:
                raise ValueError(f"FAIL LOUD: {ckpt} is not the no-in-flight MLP V on the current features")
            self.model = ValueMLP(len(FEATURES))
            self.model.load_state_dict(torch.load(ckpt))
            self.model.eval()
            self.lam = float(lam) if lam else (math.inf if contract["lambda"] == "inf" else float(contract["lambda"]))
            self.contract = contract
        else:
            self.lam = float(lam) if lam else 0.1
        self._events = None
        self._trace = None

    def _rates(self, orch, now: float) -> Dict[str, float]:
        from scripts_cosim.cost_to_go_handv import type_rates

        trace = os.environ.get(V_TRACE_ENV, "").strip()
        if not trace or not Path(trace).is_file():
            raise RuntimeError(f"FAIL LOUD: the value term needs {V_TRACE_ENV} = the run's workload JSON for its arrival rates, got {trace!r}")
        if self._trace != trace:
            self._events = json.load(open(trace))["events"]
            self._trace = trace
        return type_rates(self._events, now, LOOKBACK)

    def values(self, sched, batch_tasks, plans: Sequence[List], system_state) -> List[float]:
        """V of each plan ([(Node, Platform)] aligned with batch_tasks) at this decision."""
        from scripts_cosim.cost_to_go_handv_features import post_commit_features

        orch = sched._pg_orchestrator()
        now = float(sched.env.now)
        rates = self._rates(orch, now)
        type_defs = orch.data.task_types
        stable = float(sched.autoscaler.kpa.config.stable_window)
        feats = [post_commit_features(sched, batch_tasks, p, system_state, now, type_defs, stable, ds=None, slot=str(i), t0=now,
                                      workload=None) for i, p in enumerate(plans)]
        dump = os.environ.get(V_DUMP_ENV, "").strip()
        if dump:
            with open(dump, "a") as fh:
                fh.write(json.dumps({"now": now, "batch_ids": [int(t.id) for t in batch_tasks], "rates": rates,
                                     "plans": [[[int(c[0].id), int(c[1].id)] for c in p] for p in plans],
                                     "features": [{k: f[k] for k in ("load_after", "drain", "replicas", "type_platforms", "scale_in_after_s",
                                                                    "pg_inflight")} for f in feats]}) + "\n")
        if self.kind == "hand":
            from scripts_cosim.cost_to_go_handv import hand_v

            return [hand_v(f, rates, H, "load_after")["v"] for f in feats]
        import torch

        from scripts_cosim.cost_to_go_mlpv_data import replica_matrix

        out = []
        with torch.no_grad():
            for f in feats:
                f = dict(f, inflight_b1={k: 0.0 for k in f["load_after"]})  # the no-in-flight variant: the feature is zeroed
                x = replica_matrix(f, rates, H)
                xt = torch.tensor([x], dtype=torch.float32)
                out.append(float(self.model(xt, torch.ones(1, len(x)))[0]))
        return out

    def choose(self, ds_: Sequence[float], dv: Sequence[float]) -> int:
        """argmin dS + lambda * dV; ties go to the decided plan (index 0), then list order (as offline)."""
        if math.isinf(self.lam):
            return min(range(len(dv)), key=lambda k: (dv[k], k != 0, k))
        return min(range(len(dv)), key=lambda k: (ds_[k] + self.lam * dv[k], k != 0, k))
