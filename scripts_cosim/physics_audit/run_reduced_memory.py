#!/usr/bin/env python3
"""physics_audit_v1 I6 refusal cell: run src/executesimulation.py with every node that has more than
HEROSIM_AUDIT_NODE_MEMORY_GB of memory capped to it, so the KPA memory refusals fire. A wrapper, not a flag in the
simulator: the default path does not know it exists. Arguments after the script name are executesimulation's."""
import os
import sys

sys.path.insert(0, os.getcwd())
import src.executesimulation as ex  # noqa: E402

cap = float(os.environ["HEROSIM_AUDIT_NODE_MEMORY_GB"])
_prepare = ex.prepare_infrastructure_for_real_simulation


def prepare(*a, **kw):
    infra = _prepare(*a, **kw)
    for node in infra["nodes"]:
        node["memory"] = min(node["memory"], cap)
    return infra


ex.prepare_infrastructure_for_real_simulation = prepare
if __name__ == "__main__":
    sys.argv[0] = "executesimulation.py"
    ex.main()
