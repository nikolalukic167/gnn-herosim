"""Fresh stronger-control challenge using the frozen dispatch screen/live harness."""
import argparse
from pathlib import Path
import scripts_cosim.mixed_dispatch_gate as harness
from src.placement.radical.dispatch_adaptive import AdaptiveDispatch
if __name__=='__main__':
    harness.DispatchNative=AdaptiveDispatch
    harness.SOURCES=['experiments/mixed_dispatch_adaptive_v1.json',*harness.SOURCES,
                     'src/placement/radical/dispatch_adaptive.py','src/placement/radical/dispatch_adaptive.cpp',
                     'scripts_cosim/mixed_dispatch_adaptive_gate.py']
    harness.main()
