"""Simulation constants shared by the live co-simulation and inference paths.

Moved here from ``src.motivational.constants``, which belonged to the pre-GNN
HeROsim proactive-autoscaling stack (now under ``archive/pre_gnn_herosim/``).
The archived copy is deliberately left frozen in place so archived experiments
reproduce with the values they actually ran under; this module is the live
source of truth going forward.
"""

KEEP_ALIVE = 30
QUEUE_LENGTH = 100  # Balanced target concurrency
RECONCILE_INTERVAL = 1
REACTIVE_RECONCILE_INTERVAL = 5
PROACTIVE_RECONCILE_INTERVAL = 15
PREDICTION_WINDOW_SIZE = 60
PREPARE_PREDICTION_WINDOW_SIZE = 5
