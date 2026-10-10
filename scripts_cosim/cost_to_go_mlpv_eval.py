#!/usr/bin/env python3
"""cost_to_go_v1 rung (b): the ONE evaluation of a trained MLP V on the S0 held-out states (cost_to_go_mlpv_data build output).

  cost_to_go_mlpv_eval.py CKPT.pt S0_STATES.jsonl [--eval-topos 16252,16254,16256,16258]

lambda is the checkpoint's (fit on train-topology SELECTION, never on S0). Reports on all S0 states and on the hand V's EVAL half:
pooled and per-state Spearman of dV vs dQ5, hit rate (lambda and lambda=0), Q5 saved vs cd_exactS as a share of the headroom
(best of the plan set), and the registered rule: the GNN rung runs only if the MLP leaves >= half the headroom.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cost_to_go_mlpv_data import FEATURES  # noqa: E402
from train_value_mlp import ValueMLP, predict, pooled_spearman, saved, spearman, tensors  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("states")
    ap.add_argument("--eval-topos", default="16252,16254,16256,16258")
    a = ap.parse_args()
    c = json.loads(Path(a.ckpt).with_suffix(".contract.json").read_text())
    if list(c["features"]) != list(FEATURES):
        raise SystemExit(f"FAIL LOUD: checkpoint features {c['features']} != {FEATURES}")
    lam = math.inf if c["lambda"] == "inf" else float(c["lambda"])
    model = ValueMLP(len(FEATURES))
    model.load_state_dict(torch.load(a.ckpt))
    states = [json.loads(l) for l in open(a.states)]
    ev_t = {int(x) for x in a.eval_topos.split(",")}
    print(f"{a.ckpt}: variant {'no in-flight' if c['no_inflight'] else 'in-flight b1'}, seed {c['seed']}, epoch {c['selected_epoch']}, "
          f"lambda {c['lambda']} (fit on train-topology SELECTION {c['selection_topos']}; the hand V's lambda was fit on S0's FIT half)")
    for name, sub in (("all S0 states", states), (f"EVAL half {sorted(ev_t)} (the hand V's 56.6 % comparator)", [s for s in states if s["topo"] in ev_t])):
        pred = predict(model, [tensors(s, c["no_inflight"]) for s in sub])
        per = [spearman(dv[1:], dq[1:]) for dv, dq, _ in pred if len(dq) >= 4 and len(set(dq[1:])) > 1 and len(set(dv[1:])) > 1]
        per = [x for x in per if not math.isnan(x)]
        got, head, hit = saved(pred, lam)
        got0, _h, hit0 = saved(pred, 0.0)
        print(f"\n[{name}] {len(sub)} states")
        print(f"  pooled Spearman dV vs dQ5 {pooled_spearman(pred):+.3f}; per-state mean {sum(per) / max(1, len(per)):+.3f} (n={len(per)})")
        print(f"  lambda {c['lambda']}: hit {100 * hit:.1f} % (argmin S {100 * hit0:.1f} %); Q5 saved vs cd_exactS {got:.2f} s of {head:.2f} s "
              f"headroom ({100 * got / head if head > 0 else float('nan'):.1f} %); lambda=0 saves {got0:.2f} s")
        if name.startswith("all"):
            print(f"  registered rule: GNN rung only if the MLP leaves >= half the headroom -> "
                  f"{'GNN rung RUNS' if head > 0 and got / head < 0.5 else 'GNN rung does NOT run'}")


if __name__ == "__main__":
    main()
