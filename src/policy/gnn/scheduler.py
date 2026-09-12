"""
GNN-based Scheduler for Task-to-Platform Placement (NON-UNIQUE VERSION)

This scheduler uses a trained GNN model to make placement decisions.
It processes batches of 2-4 tasks together and decodes placements via
sequential GNN argmax with live queue roll-forward between tasks.

Fallback to shortest-queue for:
- Single task batches (model not trained on 1 task)
- Large batches > 4 tasks (model not trained on 5+ tasks)
"""

from __future__ import annotations

import copy
import logging
import json
import os
from timeit import default_timer
from typing import Callable, Generator, List, Optional, Set, Tuple, TYPE_CHECKING, Dict, Any

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

from src.policy.gnn.seq_decode import (
    KNOWN_DECODE_MODES,
    reset_episode_trajectory,
    get_episode_trajectory,
    PlacementCombo,
    get_run_decode_stats,
    record_queue_feature_discrimination,
    reset_run_decode_stats,
    run_decode_with_timing,
)
from src.policy.tabular.feature_builder import build_pyg_inference_graph
from src.placement.live_audit import maybe_capture_batch_live_audit_snapshot
from src.placement.model import SystemState
from src.placement.scheduler import Scheduler
from src.policy.state_capture import StateCaptureHelper

def _detached_graph_copy(graph: Data) -> Data:
    """A CPU, grad-free copy of an inference graph, for the Phase 3 replay reservoir.

    The stored graph outlives the decode by an entire episode and is then pickled to
    disk, so it must not hold device memory or reference the live `Data` the scheduler
    keeps mutating (`queue_snapshot` and `task_logit_to_placement` are rewritten in
    place every batch). Non-tensor attributes are copied by value for the same reason.
    """
    out = Data()
    for key, value in graph:
        if torch.is_tensor(value):
            out[key] = value.detach().to("cpu").clone()
        else:
            out[key] = copy.deepcopy(value)
    # `for key, value in graph` skips the underscore-prefixed private attrs that
    # feature_builder attaches; the decode needs none of them, but the candidate map
    # is what lets a replay assert its logit rows still line up with the same
    # placements, so it travels explicitly.
    tl2p = getattr(graph, "_task_logit_to_placement", None)
    if tl2p is not None:
        out.task_logit_to_placement = copy.deepcopy(tl2p)
    return out


def move_graph_tensors_(graph: Data, device: torch.device) -> Data:
    """Move a graph's tensor fields to `device`, in place.

    `Data.to()` recurses through every stored attribute, including the plain-Python
    dicts the scheduler attaches (`queue_snapshot`, `task_logit_to_placement`). Those
    are decode bookkeeping the forward pass never reads, and walking them cost ~26%
    of a live episode. Only tensors are moved; everything else is left alone.
    """
    # `Tensor.to` returns self when the tensor is already on `device`, so it is its own
    # short-circuit. An explicit `value.device != device` guard would be wrong anyway:
    # torch.device('cuda') never compares equal to the cuda:0 a tensor reports.
    moved = [
        (key, value.to(device))
        for key, value in list(graph._store.items())
        if isinstance(value, torch.Tensor)
    ]
    for key, value in moved:
        graph._store[key] = value
    return graph


# Task-platform compatibility (same as training)
TASK_PLATFORM_COMPATIBILITY = {
    'dnn1': ['rpiCpu', 'xavierGpu', 'xavierCpu', 'pynqFpga'],
    'dnn2': ['rpiCpu', 'xavierGpu', 'xavierCpu']
}

# Queue normalization constant (same as training)
QUEUE_NORM_FACTOR = 50.0

# GNN model training range: 2-4 tasks (co-sim cache uses up to 4-task batches)
# Use fallback (shortest queue) for batches outside this range
MIN_BATCH_SIZE_FOR_GNN = 2
MAX_BATCH_SIZE_FOR_GNN = 4
# peer_affinity_v1 stage 3 (2026-09-11): the prefix-conditioned decoder
# (GNN_DECODE_MODE=masked_topo) was trained on 10-task batches and its MASK — not the
# batch width — is what makes a plan valid, so it decodes any size >= 1 (a lone task is
# a one-step decode with an all-zero prefix). The ceiling is the corpus's own batch
# width with headroom for the denser-graph rungs; it is a guard against a misconfigured
# env, not a model limit.
MIN_BATCH_SIZE_FOR_PREFIX_GNN = 1
MAX_BATCH_SIZE_FOR_PREFIX_GNN = 16
PREFIX_DECODE_MODE = "masked_topo"


def _gnn_batch_range() -> Tuple[int, int]:
    mode = os.environ.get("GNN_DECODE_MODE", "argmax").strip().lower()
    if mode == PREFIX_DECODE_MODE:
        return MIN_BATCH_SIZE_FOR_PREFIX_GNN, MAX_BATCH_SIZE_FOR_PREFIX_GNN
    return MIN_BATCH_SIZE_FOR_GNN, MAX_BATCH_SIZE_FOR_GNN


def _read_gnn_batch_size() -> int:
    raw = os.environ.get("GNN_BATCH_SIZE", "4")
    try:
        size = int(raw)
    except ValueError as exc:
        raise ValueError(f"GNN_BATCH_SIZE must be an integer, got {raw!r}") from exc
    if size < 1:
        raise ValueError(f"GNN_BATCH_SIZE must be >= 1, got {size}")
    lo, hi = _gnn_batch_range()
    if size > hi:
        raise ValueError(
            f"GNN_BATCH_SIZE={size} exceeds the GNN batch ceiling {hi} for decode mode "
            f"{os.environ.get('GNN_DECODE_MODE', 'argmax')!r}; batches outside [{lo},{hi}] use "
            "shortest-queue fallback and corrupt GNN/MLP comparisons"
        )
    return size


def _read_batch_by_peer_group() -> bool:
    """GNN_BATCH_BY_PEER_GROUP=1: collect the first task's whole peer group (transitive
    closure of the trace's peer_exchange table) into one batch, waiting up to
    GNN_BATCH_TIMEOUT for members that have not arrived yet; tasks of other groups stay
    queued. This is the live analogue of the co-sim batch (= the peer group). The
    reactive baselines keep placing per arrival, peer-blind, and are charged the same
    exchange physics. Requires masked_topo."""
    raw = os.environ.get("GNN_BATCH_BY_PEER_GROUP", "0").strip().lower()
    if raw not in ("", "0", "false", "no", "1", "true", "yes"):
        raise ValueError(f"GNN_BATCH_BY_PEER_GROUP must be a boolean, got {raw!r}")
    on = raw in ("1", "true", "yes")
    if on and os.environ.get("GNN_DECODE_MODE", "argmax").strip().lower() != PREFIX_DECODE_MODE:
        raise ValueError(
            "FAIL LOUD: GNN_BATCH_BY_PEER_GROUP=1 needs GNN_DECODE_MODE=masked_topo — only the "
            "prefix-conditioned decoder scores a batch against its peer edges"
        )
    return on


def _read_gnn_batch_timeout() -> float:
    raw = os.environ.get("GNN_BATCH_TIMEOUT", "0.002")
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError(f"GNN_BATCH_TIMEOUT must be a float, got {raw!r}") from exc
    if timeout <= 0:
        raise ValueError(f"GNN_BATCH_TIMEOUT must be > 0, got {timeout}")
    return timeout


class GNNScheduler(Scheduler):
    # Snapshot "policy" field for live-audit capture; subclasses override.
    _live_audit_policy_name = "gnn"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Batch collection window — override via GNN_BATCH_SIZE / GNN_BATCH_TIMEOUT env vars.
        # MLP/XGB batch schedulers inherit these settings via GNNScheduler.
        self.batch_size = _read_gnn_batch_size()
        self.batch_timeout = _read_gnn_batch_timeout()
        self.batch_min, self.batch_max = _gnn_batch_range()
        # peer_affinity_v1 stage 3: prefix-conditioned serving state. The decode options
        # come off the checkpoint sidecar (set_models); the counters make a gate report
        # say how many batches were peer-complete and how many peers the decoder could
        # not see, instead of leaving that to be inferred.
        self.batch_by_peer_group = _read_batch_by_peer_group()
        self._prefix_options = None
        self.prefix_batches = 0
        self.prefix_tasks_decoded = 0
        self.prefix_tasks_deferred = 0
        self.prefix_pairs_in_batch = 0
        self.prefix_peers_outside_batch = 0
        self.peer_group_incomplete_batches = 0
        # serving knobs (default off): how often a batch actually had standing load to
        # charge, so a gate arm cannot claim the seed was active when it never fired
        self.prefix_batches_load_seeded = 0

        # GNN model will be set via models dict from orchestrator
        self.gnn_model = None
        self.device = None
        self.task_types_data = None
        self.dataset_id = None
        self.models_dict = None  # Store full models dict
        
        # === GNN Configuration ===
        # Use pure GNN with simple fallback (no soft blending)
        # Adaptive queue normalization will be calculated per batch
        
        # Stats tracking for debugging
        self.gnn_pure_decisions = 0
        self.fallback_decisions = 0
        self.decode_stats = reset_run_decode_stats()
        self._decode_mode = os.environ.get("GNN_DECODE_MODE", "argmax").strip().lower()
        # A typo used to fall through to plain argmax silently, so an ablation could
        # report "seq_reforwrd" results that were really the control.
        if self._decode_mode not in KNOWN_DECODE_MODES:
            raise ValueError(
                f"FAIL LOUD: GNN_DECODE_MODE={self._decode_mode!r} is not a known decode mode. "
                f"Known: {sorted(KNOWN_DECODE_MODES)}"
            )
        # objective_pivot_v1 Phase 3: the sampled decode is the one stochastic mode, so
        # it opens a seeded episode trajectory here. Seeding at scheduler construction
        # (not per batch) makes the whole episode a deterministic function of
        # GNN_SAMPLE_SEED, which is what lets an arm be paired against another under
        # common random numbers.
        if self._decode_mode in ("sample", "sampled"):
            temp_env = os.environ.get("GNN_SAMPLE_TEMPERATURE", "").strip()
            seed_env = os.environ.get("GNN_SAMPLE_SEED", "").strip()
            if not temp_env or not seed_env:
                raise ValueError(
                    "FAIL LOUD: decode mode 'sample' requires GNN_SAMPLE_TEMPERATURE and "
                    "GNN_SAMPLE_SEED. Defaults would fix an unregistered exploration "
                    "level and an unreproducible episode into every run."
                )
            # k = 0 keeps the probe's exact behaviour (no replay, no payload cost);
            # the closed-loop trainer sets it to the number of decode batches whose
            # gradient it can afford to replay.
            reservoir_k = int(os.environ.get("GNN_SAMPLE_RESERVOIR_K", "0"))
            reset_episode_trajectory(float(temp_env), int(seed_env), reservoir_k)
            print(
                f"[GNN] Decode mode: sample (T={temp_env}, episode seed={seed_env}, "
                f"replay_k={reservoir_k}) -- STOCHASTIC policy, not a gate configuration",
                flush=True,
            )
        self._decode_seqblend = self._decode_mode in ("seqblend", "seqblend_p1", "1")
        self._decode_queue_margin = int(os.environ.get("GNN_SEQBLEND_QUEUE_MARGIN", "1"))
        if self._decode_mode in ("frozen", "frozen_argmax", "frozen_topk", "topk", "topk_joint"):
            top_k = os.environ.get("GNN_DECODE_TOP_K", "10")
            print(
                f"[GNN] Decode mode: {self._decode_mode}"
                + (f" (top_k={top_k})" if "topk" in self._decode_mode or self._decode_mode == "topk" else ""),
                flush=True,
            )
        
        # State capture helper (initialized lazily when env/nodes are available)
        self._state_capture: Optional[StateCaptureHelper] = None
        # Phase 3: sim-lifetime FilterStore pull ledger for seq_reforward_pull
        # (ect_pull persists across decisions; per-batch reset was the Phase 1 gap).
        self._pulls_committed: Dict[str, int] = {}

    def set_models(self, models: dict):
        """
        Set GNN models from orchestrator.
        
        Expected models dict structure:
        {
            'gnn_model': trained PyTorch model,
            'device': torch device (cuda/cpu),
            'task_types_data': task type metadata for feature computation,
            'dataset_id': optional dataset identifier
        }
        """
        print(f"[GNN Scheduler] set_models called with keys: {list(models.keys()) if models else 'None'}", flush=True)
        self.models_dict = models
        
        if models is None:
            print("[GNN Scheduler] WARNING: set_models called with None", flush=True)
            return
        
        if 'gnn_model' in models:
            self.gnn_model = models['gnn_model']
            print(f"[GNN Scheduler] Model loaded: {type(self.gnn_model).__name__}", flush=True)
            # peer_affinity_v1 stage 3: load_gnn_model pins the sidecar's decode options
            # on the module; masked_topo cannot run without them.
            self._prefix_options = getattr(self.gnn_model, "prefix_serving_options", None)
            if self._decode_mode == PREFIX_DECODE_MODE and self._prefix_options is None:
                raise ValueError(
                    "FAIL LOUD: GNN_DECODE_MODE=masked_topo but the loaded model carries no "
                    "prefix_serving_options — it is not a prefix-conditioned checkpoint "
                    "(src/policy/gnn/prefix_serving.py)"
                )
            if self._prefix_options is not None and self._decode_mode != PREFIX_DECODE_MODE:
                raise ValueError(
                    f"FAIL LOUD: {type(self.gnn_model).__name__} is prefix-conditioned "
                    f"(partial_state_edge_features) but GNN_DECODE_MODE={self._decode_mode!r}; "
                    "only masked_topo builds the prefix it scores against"
                )
        else:
            print("[GNN Scheduler] WARNING: 'gnn_model' not found in models dict", flush=True)
        
        if 'device' in models:
            self.device = models['device']
            print(f"[GNN Scheduler] Device: {self.device}", flush=True)
        else:
            self.device = torch.device('cpu')
            print("[GNN Scheduler] No device specified, using CPU", flush=True)
        
        if 'task_types_data' in models:
            self.task_types_data = models['task_types_data']
            print(f"[GNN Scheduler] Task types data loaded: {list(self.task_types_data.keys()) if self.task_types_data else 'None'}", flush=True)
        
        if 'dataset_id' in models:
            self.dataset_id = models['dataset_id']
        
        # Put model in eval mode
        if self.gnn_model is not None:
            self.gnn_model.eval()
            print("[GNN Scheduler] Model set to eval mode", flush=True)
        
        print(f"[GNN Scheduler] After set_models: gnn_model is None = {self.gnn_model is None}", flush=True)

    def scheduler_process(self) -> Generator:
        if False:
            yield

        logging.info(
            f"[ {self.env.now} ] GNN Scheduler started with policy {self.policy}"
            f" (batch_size={self.batch_size}, gnn_range=[{self.batch_min},{self.batch_max}])"
        )

        while True:
            batch_tasks = yield self.env.process(self._collect_task_batch())
            
            if not batch_tasks:
                yield self.env.timeout(0.01)
                continue

            yield self.env.process(self._process_task_batch(batch_tasks))

    def _collect_task_batch(self) -> Generator[Any, Any, List[Task]]:
        """
        Collect tasks into a batch using timeout-based waiting.
        
        Strategy:
        1. Wait for at least one task (blocking)
        2. Wait for batch_timeout duration to collect more tasks
        3. Return whatever was collected (min 1, max batch_size)
        
        This ensures GNN gets batches of 2-3 tasks (its training range).
        """
        batch: List[Task] = []
        wait_start_time = self.env.now
        
        def task_filter(queued_task):
            return all(dependency.finished for dependency in queued_task.dependencies)
        
        # First task: wait indefinitely (blocking is expected)
        task: Task = yield self.tasks.get(task_filter)
        batch.append(task)

        # Wait for batch_timeout to collect more tasks
        # Use small increments to be responsive while still batching
        timeout_remaining = self.batch_timeout
        poll_interval = 0.001  # 1ms polling interval

        if self.batch_by_peer_group:
            # peer_affinity_v1 stage 3: the batch IS the first task's peer group. Members
            # already placed by an earlier (incomplete) batch are not waited for; members
            # that have not arrived yet are, up to the batch timeout, which is a real
            # queueing cost this arm pays and the reactive arms do not.
            orch = self._orchestrator()
            remaining = {
                j for j in self._peer_group(task)
                if j != task.id and not self._peer_already_scheduled(orch, j)
            }
            while remaining and len(batch) < self.batch_size and timeout_remaining > 0:
                ready = [t for t in self.tasks.items if t.id in remaining and task_filter(t)]
                if ready:
                    member: Task = yield self.tasks.get(
                        lambda queued, _r=remaining: queued.id in _r and task_filter(queued)
                    )
                    batch.append(member)
                    remaining.discard(member.id)
                else:
                    wait_time = min(poll_interval, timeout_remaining)
                    yield self.env.timeout(wait_time)
                    timeout_remaining -= wait_time
            if remaining:
                self.peer_group_incomplete_batches += 1
            return batch

        while len(batch) < self.batch_size and timeout_remaining > 0:
            # Check if there are any ready tasks in the queue
            ready_tasks = [t for t in self.tasks.items if task_filter(t)]
            
            if ready_tasks:
                # Get the ready task immediately
                task = yield self.tasks.get(task_filter)
                batch.append(task)
            else:
                # Wait a small interval for more tasks to arrive
                wait_time = min(poll_interval, timeout_remaining)
                yield self.env.timeout(wait_time)
                timeout_remaining -= wait_time
        
        # Calculate actual wait time
        actual_wait_time = self.env.now - wait_start_time
        
        # Simple logging for batch size and wait time
        batch_size = len(batch)
        if batch_size == 2:
            print(f"[GNN Batch] Batch size: 2 tasks, wait time: {actual_wait_time*1000:.2f}ms", flush=True)
        elif batch_size == 3:
            print(f"[GNN Batch] Batch size: 3 tasks, wait time: {actual_wait_time*1000:.2f}ms", flush=True)
        else:
            print(f"[GNN Batch] Batch size: {batch_size} tasks, wait time: {actual_wait_time*1000:.2f}ms", flush=True)
        
        # Log batch size for debugging
        if len(batch) >= MIN_BATCH_SIZE_FOR_GNN:
            logging.debug(f"[ {self.env.now} ] GNN: Collected batch of {len(batch)} tasks (will use GNN)")
        else:
            logging.debug(f"[ {self.env.now} ] GNN: Collected batch of {len(batch)} tasks (will use fallback)")
        
        return batch

    def _process_task_batch(self, batch_tasks: List[Task]) -> Generator:
        """
        Process a batch of tasks using GNN placement.
        
        Optimized version: single mutex acquisition per batch, no proactive replica creation.
        """
        batch_start = default_timer()
        batch_size = len(batch_tasks)

        # peer_affinity_v1 stage 3: the prefix-conditioned decoder has its own batch
        # path — joint decode first, node pre-pass for the peer physics, then enqueue;
        # no shortest-queue fallback anywhere in it.
        if self._decode_mode == PREFIX_DECODE_MODE:
            yield from self._process_task_batch_prefix(batch_tasks)
            return

        # Get system state once for the entire batch
        system_state: SystemState = yield self.mutex.get()

        # Same oracle-audit capture the knative_network_batch arm has, so collapse-moment
        # states from ML arms are capturable (off unless LIVE_AUDIT_SNAPSHOT_PATH is set).
        maybe_capture_batch_live_audit_snapshot(
            self, system_state, batch_tasks, self._live_audit_policy_name
        )

        # Full-infra queue + temporal snapshot at batch start (matches SSC/cache graph build)
        queue_snapshot = self._capture_full_queue_snapshot()
        temporal_state = self._capture_temporal_state_snapshot()
        
        # Skip GNN for batches outside training range [2, 3]
        if batch_size < self.batch_min or batch_size > self.batch_max:
            placements = None  # Will trigger fallback to shortest queue
            inference_time = 0.0
            logging.info(f"[ {self.env.now} ] GNN: Batch size {batch_size} outside GNN range [{self.batch_min},{self.batch_max}], using fallback")
        else:
            # Build graph and run GNN inference
            inference_start = default_timer()
            placements = self._gnn_inference(
                batch_tasks, system_state, queue_snapshot, temporal_state
            )
            inference_time = default_timer() - inference_start
            if placements:
                logging.info(f"[ {self.env.now} ] GNN: Batch of {batch_size} tasks, GNN returned {len(placements)} placements in {inference_time*1000:.2f}ms")
            else:
                logging.info(f"[ {self.env.now} ] GNN: Batch of {batch_size} tasks, GNN inference failed (model not loaded?)")
        
        # Release mutex before processing tasks (allows monitor/autoscaler to run)
        yield self.mutex.put(system_state)
        
        # Process each task in batch
        for task_idx, task in enumerate(batch_tasks):
            task_start = default_timer()
            
            # Get fresh system state for this task
            current_system_state: SystemState = yield self.mutex.get()
            
            task_replicas = current_system_state.replicas.get(task.type["name"], set())
            valid_replicas = self._get_valid_replicas(task_replicas, task)

            # If no valid replicas, request autoscaling (reactive, like knative_network)
            if not valid_replicas:
                logging.warning(
                    f"[ {self.env.now} ] GNN Scheduler: no network-accessible replica for {task}"
                )
                
                task.postponed_count += 1
                yield self.tasks.put(task)

                # Request replica from autoscaler
                stop = yield self.env.process(
                    self.autoscaler.create_first_replica(
                        current_system_state, 
                        task.type,
                        source_node_name=task.node_name
                    )
                )
                
                yield self.mutex.put(current_system_state)
                continue

            # Capture scheduling snapshots only when generating GNN training datasets.
            if os.environ.get("GNN_CAPTURE_DATASET_STATE", "0") == "1":
                task.queue_snapshot_at_scheduling = self._capture_queue_snapshot_for_replicas(valid_replicas)
                task.full_queue_snapshot = self._capture_full_queue_snapshot()
                task.temporal_state_at_scheduling = self._capture_temporal_state_for_replicas(valid_replicas)

            # Select placement using GNN with fallback to shortest queue
            target_node, target_platform = self._select_placement_pure_gnn(
                task, task_idx, placements, valid_replicas
            )

            # Fallback to shortest queue if GNN placement is invalid
            if target_node is None or target_platform is None:
                target_node, target_platform = min(
                    valid_replicas, key=lambda couple: len(couple[1].queue.items)
                )
                self.fallback_decisions += 1

            # Deferred cold pull at place time (scarce-preinit stubs)
            from src.placement.replica_seeding import start_deferred_cold_init

            start_deferred_cold_init(
                self.env,
                self.autoscaler,
                target_node,
                target_platform,
                task_replicas,
                task.type,
                current_system_state,
            )

            task.execution_node = target_node.node_name
            task.execution_platform = str(target_platform.id)
            task.gnn_decision_time = inference_time / batch_size  # Amortized

            # Update node
            node: Node = yield self.nodes.get(lambda node: node.id == target_node.id)
            task.node = node
            node.unused = False
            
            # Update platform
            platform: Platform = yield node.platforms.get(lambda platform: platform.id == target_platform.id)
            task.platform = platform

            # End wall-clock time measurement
            task_end = default_timer()
            elapsed_clock_time = task_end - task_start
            node.wall_clock_scheduling_time += elapsed_clock_time

            # Put task in platform queue
            yield platform.queue.put(task)
            yield task.scheduled.succeed()

            yield node.platforms.put(platform)
            yield self.nodes.put(node)
            
            yield self.mutex.put(current_system_state)

    # ------------------------------------------------------------------
    # peer_affinity_v1 stage 3: prefix-conditioned serving (GNN_DECODE_MODE=masked_topo)
    # ------------------------------------------------------------------

    def _orchestrator(self):
        for node in self.nodes.items:
            orch = getattr(node, "orchestrator_ref", None)
            if orch is not None:
                return orch
        raise RuntimeError("GNN scheduler: no node carries an orchestrator_ref yet")

    def _peer_group(self, task: Task) -> Set[int]:
        """Transitive closure of the peer table from `task` (global task ids, incl. its own)."""
        table = getattr(self._orchestrator(), "peer_exchange", None) or {}
        seen = {int(task.id)}
        stack = [int(task.id)]
        while stack:
            i = stack.pop()
            for j in table.get(i, {}):
                if j not in seen:
                    seen.add(int(j))
                    stack.append(int(j))
        return seen

    @staticmethod
    def _peer_already_scheduled(orch, task_id: int) -> bool:
        peer = getattr(orch, "task_by_id", {}).get(int(task_id))
        return peer is not None and bool(peer.scheduled.triggered)

    def _prefix_inference(
        self,
        batch_tasks: List[Task],
        system_state: SystemState,
        queue_snapshot: Dict[str, int],
        temporal_state: Optional[Dict[str, Dict[str, float]]],
    ) -> Dict[int, Tuple[int, int]]:
        """Build the live graph, attach the prefix block, run the registered decoder.

        Strict by construction: every exception propagates. A prefix-conditioned gate
        that silently served shortest-queue for a batch would report a number that is
        not the model's, so there is no try/except here at all.
        """
        from src.policy.gnn.prefix_serving import (
            attach_live_prefix_block,
            decode_prefix_conditioned,
        )

        if self.gnn_model is None or self._prefix_options is None:
            raise RuntimeError("masked_topo: model / prefix options not set (set_models)")
        graph, task_logit_to_placement = self._build_inference_graph(
            batch_tasks, system_state, queue_snapshot, temporal_state
        )
        if graph is None:
            raise RuntimeError("masked_topo: the live graph builder returned no feasible edges")
        for idx, task in enumerate(batch_tasks):
            if not task_logit_to_placement.get(idx):
                raise RuntimeError(
                    f"masked_topo: task {task.id} ({task.type['name']}) has no candidate edge "
                    "in the live graph although it has network-accessible replicas"
                )
        graph.queue_snapshot = dict(queue_snapshot)
        graph.task_logit_to_placement = task_logit_to_placement
        graph._task_logit_to_placement = task_logit_to_placement
        diag = attach_live_prefix_block(
            graph,
            batch_tasks,
            nodes=list(self.nodes.items),
            peer_table=getattr(self._orchestrator(), "peer_exchange", None) or {},
            options=self._prefix_options,
        )
        self.prefix_pairs_in_batch += int(diag["n_pairs_in_batch"])
        self.prefix_peers_outside_batch += int(diag["peers_outside_batch"])
        if int(diag.get("seeded_nodes", 0)):
            self.prefix_batches_load_seeded += 1
        graph = move_graph_tensors_(graph, self.device)
        with torch.no_grad():
            combo = decode_prefix_conditioned(
                self.gnn_model, graph, self._prefix_options, stats=self.decode_stats
            )
        self.last_prefix_graph = graph  # parity checks read the served graph
        trace_path = os.environ.get("GNN_PREFIX_TRACE_PATH", "").strip()
        if trace_path:
            # One pickled record per served batch: the served graph (CPU tensors + the
            # prefix context) and the decoded plan, so a parity check can hold the live
            # graph up against the cache graph attribute by attribute.
            import pickle

            record = {
                "sim_time": float(self.env.now),
                "task_ids": [int(t.id) for t in batch_tasks],
                "task_types": [str(t.type["name"]) for t in batch_tasks],
                "combo": [list(c) for c in combo],
                "diag": dict(diag),
            }
            # serving_gap_v1: the served graph is ~50 kB, so a full production trace
            # (45,375 batches) is ~2.3 GB and 96 of them do not fit the /home quota.
            # GNN_PREFIX_TRACE_SLIM=1 drops it; the plan, the ids and the diagnostics are
            # what the herding statistics need. Default keeps the graph, so the parity
            # check (peer_affinity_live_serve_check.py) is unaffected.
            if os.environ.get("GNN_PREFIX_TRACE_SLIM", "").strip() != "1":
                record["graph"] = _detached_graph_copy(graph)
            with open(trace_path, "ab") as fh:
                pickle.dump(record, fh)
        return {idx: combo[idx] for idx in range(len(combo))}

    def _process_task_batch_prefix(self, batch_tasks: List[Task]) -> Generator:
        """masked_topo batch path.

        1. Under one mutex hold: snapshot the state, split the batch into tasks WITH a
           network-accessible replica (decodable) and tasks without (deferred), decode
           the decodable ones jointly, and set every decodable task's
           `planned_node_name` BEFORE anything is enqueued — the peer physics reads it
           the moment a peer's input stage starts (the same pre-pass the determined
           scheduler performs for forced batches).
        2. Deferred tasks go back to the queue behind an autoscaler request, exactly as
           the reactive path does. Their peers do not wait on them here: the input-stage
           rendezvous in Platform.platform_process waits for them.
        3. Decodable tasks are enqueued at their decoded placement. A placement that is
           no longer a replica when its turn comes is an error, not a fallback.
        """
        system_state: SystemState = yield self.mutex.get()
        maybe_capture_batch_live_audit_snapshot(
            self, system_state, batch_tasks, self._live_audit_policy_name
        )
        queue_snapshot = self._capture_full_queue_snapshot()
        temporal_state = self._capture_temporal_state_snapshot()

        decodable: List[Task] = []
        deferred: List[Task] = []
        for task in batch_tasks:
            task_replicas = system_state.replicas.get(task.type["name"], set())
            if self._get_valid_replicas(task_replicas, task):
                decodable.append(task)
            else:
                deferred.append(task)

        placements: Dict[int, Tuple[int, int]] = {}
        inference_time = 0.0
        if decodable:
            inference_start = default_timer()
            placements = self._prefix_inference(
                decodable, system_state, queue_snapshot, temporal_state
            )
            inference_time = default_timer() - inference_start
            node_by_id = {node.id: node for node in self.nodes.items}
            for idx, task in enumerate(decodable):
                task.planned_node_name = node_by_id[placements[idx][0]].node_name
            self.prefix_batches += 1
            self.prefix_tasks_decoded += len(decodable)
            logging.info(
                f"[ {self.env.now} ] GNN masked_topo: batch of {len(batch_tasks)} "
                f"({len(decodable)} decoded, {len(deferred)} deferred) in {inference_time*1000:.2f}ms"
            )
        self.prefix_tasks_deferred += len(deferred)
        yield self.mutex.put(system_state)

        for task in deferred:
            current_system_state: SystemState = yield self.mutex.get()
            logging.warning(
                f"[ {self.env.now} ] GNN masked_topo: no network-accessible replica for {task}"
            )
            task.postponed_count += 1
            yield self.tasks.put(task)
            yield self.env.process(
                self.autoscaler.create_first_replica(
                    current_system_state, task.type, source_node_name=task.node_name
                )
            )
            yield self.mutex.put(current_system_state)

        for idx, task in enumerate(decodable):
            task_start = default_timer()
            current_system_state = yield self.mutex.get()
            target_node_id, target_plat_id = placements[idx]
            task_replicas = current_system_state.replicas.get(task.type["name"], set())
            match = [
                (node, plat) for node, plat in task_replicas
                if node.id == target_node_id and plat.id == target_plat_id
            ]
            if not match:
                raise RuntimeError(
                    f"masked_topo: decoded placement ({target_node_id}, {target_plat_id}) for "
                    f"task {task.id} is no longer a replica of {task.type['name']} — refusing "
                    "to substitute a fallback"
                )
            target_node, target_platform = match[0]
            self.gnn_pure_decisions += 1

            from src.placement.replica_seeding import start_deferred_cold_init

            start_deferred_cold_init(
                self.env, self.autoscaler, target_node, target_platform,
                task_replicas, task.type, current_system_state,
            )
            task.execution_node = target_node.node_name
            task.execution_platform = str(target_platform.id)
            task.gnn_decision_time = inference_time / max(1, len(decodable))

            node: Node = yield self.nodes.get(lambda n, _id=target_node.id: n.id == _id)
            task.node = node
            node.unused = False
            platform: Platform = yield node.platforms.get(
                lambda p, _id=target_platform.id: p.id == _id
            )
            task.platform = platform
            node.wall_clock_scheduling_time += default_timer() - task_start
            yield platform.queue.put(task)
            yield task.scheduled.succeed()
            yield node.platforms.put(platform)
            yield self.nodes.put(node)
            yield self.mutex.put(current_system_state)

    def _gnn_inference(
        self,
        batch_tasks: List[Task],
        system_state: SystemState,
        queue_snapshot: Dict[str, int],
        temporal_state: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Optional[Dict[int, Tuple[int, int]]]:
        """
        Run GNN inference on the batch of tasks.
        
        Returns: Dict mapping task_idx -> (node_id, platform_id)
        """
        if self.gnn_model is None:
            print("[GNN] Model not loaded, using fallback")
            return None
        
        try:
            # Build graph from current system state
            graph, task_logit_to_placement = self._build_inference_graph(
                batch_tasks, system_state, queue_snapshot, temporal_state
            )
            
            if graph is None:
                return None
            
            task_logit_to_queue_key = getattr(graph, "_task_logit_to_queue_key", None)
            graph.queue_snapshot = dict(queue_snapshot)
            if task_logit_to_placement:
                graph.task_logit_to_placement = task_logit_to_placement
                graph._task_logit_to_placement = task_logit_to_placement

            decode_mode = os.environ.get("GNN_DECODE_MODE", "argmax").strip().lower()
            if decode_mode in ("seq_reforward", "seq_reforward_argmax"):
                from src.policy.gnn.seq_decode import decode_sequential_reforward_placement

                graph = move_graph_tensors_(graph, self.device)
                combo = decode_sequential_reforward_placement(
                    self.gnn_model,
                    graph,
                    len(batch_tasks),
                    queue_snapshot,
                    stats=self.decode_stats,
                )
                if combo is None:
                    return None
                return {t_idx: combo[t_idx] for t_idx in range(len(combo))}

            if decode_mode in (
                "seq_reforward_pull",
                "seq_reforward_pulls",
                "pulls_committed",
                "pull_ledger",
            ):
                from src.policy.gnn.seq_decode import (
                    decode_sequential_reforward_pull_placement,
                )

                # Live initialized flags — source of truth for FilterStore pull need.
                platform_needs_pull: Dict[str, bool] = {}
                for node in self.nodes.items:
                    for platform in node.platforms.items:
                        key = f"{node.node_name}:{platform.id}"
                        platform_needs_pull[key] = not bool(
                            platform.initialized.triggered
                        )

                graph = move_graph_tensors_(graph, self.device)
                combo = decode_sequential_reforward_pull_placement(
                    self.gnn_model,
                    graph,
                    len(batch_tasks),
                    queue_snapshot,
                    platform_needs_pull=platform_needs_pull,
                    pulls_committed=self._pulls_committed,
                    stats=self.decode_stats,
                )
                if combo is None:
                    return None
                return {t_idx: combo[t_idx] for t_idx in range(len(combo))}

            # Move to device
            graph = move_graph_tensors_(graph, self.device)
            
            # Run inference
            with torch.no_grad():
                logits_per_task = self.gnn_model(graph)
            
            # Decode placements sequentially with live queue state (matches online scheduling)
            placements = self._decode_placements(
                logits_per_task,
                task_logit_to_placement,
                len(batch_tasks),
                queue_snapshot,
                task_logit_to_queue_key,
                replay_payload_factory=lambda g=graph: _detached_graph_copy(g),
            )
            if placements and self.decode_stats is not None:
                missing = [i for i in range(len(batch_tasks)) if i not in placements]
                if missing:
                    raise RuntimeError(
                        f"GNN decode returned a partial placement map (missing {missing}); "
                        "refusing to probe a truncated combo"
                    )
                combo = tuple(placements[i] for i in range(len(batch_tasks)))
                # Pure instrumentation: a bug here must never discard an already-computed,
                # valid placement (that would silently fall back to shortest-queue for the
                # whole batch just because a diagnostic probe tripped).
                try:
                    record_queue_feature_discrimination(
                        self.decode_stats,
                        combo=combo,
                        logits_per_task=logits_per_task,
                        task_logit_to_placement=task_logit_to_placement,
                        queue_snapshot=queue_snapshot,
                        task_logit_to_queue_key=task_logit_to_queue_key,
                        platform_features=graph.platform_features,
                        queue_key_to_platform_meta=getattr(
                            graph, "queue_key_to_platform_meta", None
                        ),
                    )
                except Exception as probe_exc:
                    print(f"[GNN] queue-feature-discrimination probe failed (non-fatal): {probe_exc}")
            return placements
            
        except Exception as e:
            print(f"[GNN] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_adaptive_queue_norm(self, queue_snapshot: Dict[str, int]) -> float:
        """
        Calculate adaptive queue normalization factor.

        Two modes controlled by GNN_QUEUE_NORM_MODE env var:
          'scheduler_adaptive' (default): p90 of ALL platforms, cap [1, 100].
            Collapses to 1.0 when most platforms are idle (p90 of zeros = 0).
          'adaptive_nonzero': p90 of NON-ZERO queue platforms only.
            Robust against sparse-heavy-tailed distributions. Must match the
            queue_norm_mode used when building the training cache.
        """
        if not queue_snapshot:
            return QUEUE_NORM_FACTOR

        queue_values = sorted(queue_snapshot.values())
        if not queue_values:
            return QUEUE_NORM_FACTOR

        norm_mode = os.environ.get("GNN_QUEUE_NORM_MODE", "scheduler_adaptive").strip()

        if norm_mode == "adaptive_nonzero":
            non_zero = [v for v in queue_values if v > 0]
            if not non_zero:
                return 1.0
            idx = int(len(non_zero) * 0.9)
            p90 = non_zero[min(idx, len(non_zero) - 1)]
        else:
            idx = int(len(queue_values) * 0.9)
            p90 = queue_values[min(idx, len(queue_values) - 1)]

        return float(min(max(1.0, p90), 100.0))

    def _build_inference_graph(
        self,
        batch_tasks: List[Task],
        system_state: SystemState,
        queue_snapshot: Dict[str, int],
        temporal_state: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[Optional[Data], Optional[Dict[int, List[Tuple[int, int]]]]]:
        """
        Build a PyG graph from current system state for GNN inference.
        
        Returns: (graph, task_logit_to_placement mapping)
        """
        norm_mode = os.environ.get("GNN_QUEUE_NORM_MODE", "adaptive").strip().lower()
        return build_pyg_inference_graph(
            batch_tasks,
            system_state,
            queue_snapshot,
            nodes=list(self.nodes.items),
            task_types_data=self.task_types_data,
            queue_norm_mode=norm_mode,
            temporal_state=temporal_state,
        )

    def _decode_placements(
        self,
        logits_per_task: List[torch.Tensor],
        task_logit_to_placement: Dict[int, List[Tuple[int, int]]],
        n_tasks: int,
        queue_snapshot: Optional[Dict[str, int]] = None,
        task_logit_to_queue_key: Optional[Dict[int, List[str]]] = None,
        replay_payload_factory: Optional[Callable[[], Any]] = None,
    ) -> Dict[int, Tuple[int, int]]:
        """
        GNN decode modes (GNN_DECODE_MODE env):

        argmax (default): sequential per-task argmax with live queue roll-forward.
        argmax_uniq: sequential argmax with intra-batch platform uniqueness mask.
        seq_reforward: per-task argmax with platform queue-feature refresh + GNN re-forward each task.
        seq_reforward_pull: Phase 1 — seq_reforward + pulls_committed ledger updating dim24
            node_cold_count / estimated_pull_remaining before each re-forward (ect_pull bookkeeping).
        seqblend: sequential argmax + min-queue override when queue > min + margin.
        frozen: per-task argmax from one snapshot (no roll-forward; matches offline greedy).
        frozen_topk: joint top-k by summed logits from one snapshot (matches near-RTT eval).

        frozen_topk uses GNN_DECODE_TOP_K (default 10).
        """
        decode_mode = os.environ.get("GNN_DECODE_MODE", "argmax").strip().lower()
        top_k = max(1, int(os.environ.get("GNN_DECODE_TOP_K", "10")))
        seqblend = decode_mode in ("seqblend", "seqblend_p1", "1")
        queue_margin = int(os.environ.get("GNN_SEQBLEND_QUEUE_MARGIN", "1"))

        temperature = float(os.environ.get("GNN_LOGIT_TEMPERATURE", "1.0"))
        if temperature != 1.0:
            logits_per_task = [lgt / temperature for lgt in logits_per_task]

        combo = run_decode_with_timing(
            decode_mode,
            logits_per_task,
            task_logit_to_placement,
            n_tasks,
            queue_snapshot=queue_snapshot,
            task_logit_to_queue_key=task_logit_to_queue_key,
            seqblend=seqblend,
            queue_margin=queue_margin,
            top_k=top_k,
            stats=self.decode_stats,
        )
        # Phase 3 closed loop: close the sampled batch exactly once, either way. A
        # decode that returned None recorded log-probs for placements the simulator
        # never executed, so those records are dropped rather than credited with the
        # episode's return.
        if decode_mode in ("sample", "sampled"):
            traj = get_episode_trajectory()
            if traj is not None:
                if combo is None:
                    traj.abandon_open_batch()
                elif replay_payload_factory is not None:
                    traj.offer_replay(replay_payload_factory)
                else:
                    traj.offer_replay(lambda: None)
        if combo is None:
            return {}
        return {t_idx: combo[t_idx] for t_idx in range(len(combo))}

    def _capture_batch_queue_snapshot(self, system_state: SystemState, batch_tasks: List[Task]) -> Dict[str, int]:
        queue_snapshot = {}
        task_types = set(task.type["name"] for task in batch_tasks)
        for task_type in task_types:
            replicas = system_state.replicas.get(task_type, set())
            for node, platform in replicas:
                key = f"{node.node_name}:{platform.id}"
                if key not in queue_snapshot:
                    queue_snapshot[key] = platform.queue_length()
        return queue_snapshot

    def _capture_full_queue_snapshot(self) -> Dict[str, int]:
        queue_snapshot = {}
        for node in self.nodes.items:
            for platform in node.platforms.items:
                key = f"{node.node_name}:{platform.id}"
                queue_snapshot[key] = platform.queue_length()
        return queue_snapshot

    def placement(self, system_state: SystemState, task: Task) -> Generator:
        if False:
            yield
        return None

    def _get_valid_replicas(self, replicas: Set[Tuple[Node, Platform]], task: Task) -> List[Tuple[Node, Platform]]:
        """Get valid replicas: task's source node + server nodes with network connectivity.
        
        Matches herocache_network logic: only servers (non-client nodes) can receive remote tasks.
        """
        valid_replicas = []
        
        # Find source node to check its network_map
        source_node = None
        for n in self.nodes.items:
            if n.node_name == task.node_name:
                source_node = n
                break
        
        for node, platform in replicas:
            # Include if it's the task's source node (local execution)
            if node.node_name == task.node_name:
                valid_replicas.append((node, platform))
            # Include if it's a server node AND has network connectivity to task source
            elif not node.node_name.startswith('client_node'):
                # Check if source node has network connectivity to this server
                if source_node is not None and hasattr(source_node, 'network_map'):
                    if node.node_name in source_node.network_map:
                        valid_replicas.append((node, platform))
                # Fallback: check bidirectional connectivity
                elif hasattr(node, 'network_map') and task.node_name in node.network_map:
                    valid_replicas.append((node, platform))

        # `replicas` is a set, so its iteration order (and therefore this list's order,
        # which feeds candidate/graph-node ordering for the GNN's tie-break-sensitive
        # argmax decode) is not reproducible across processes (PYTHONHASHSEED).
        valid_replicas.sort(key=lambda couple: (couple[0].id, couple[1].id))
        return valid_replicas

    def _capture_temporal_state_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Temporal state (remaining times) for all platforms — THE capture the cache
        was built from.

        Until 2026-09-11 this method carried its own copy of the formula, and the copy
        drifted: it read `comm_remaining` off the node's real local-storage throughput
        and latency, while `StateCaptureHelper.capture_temporal_state_for_replicas` —
        the function that writes `full_temporal_state_at_scheduling` into every
        system_state_captured_unique.json, i.e. what every cache and every checkpoint
        saw — uses fixed defaults (100 MB/s, 1 ms). On a flashCard node the two differ
        by ~4e4x on platform column 11 whenever a task is mid-execution there. Found by
        scripts_cosim/peer_affinity_live_serve_check.py (4/34 held-out batches served a
        column the model was never trained on). One definition now; the helper's extra
        `previous_task_type_name` key is ignored by the feature builder.
        """
        return self.state_capture.capture_full_temporal_state()

    def _select_placement_pure_gnn(
        self,
        task: Task,
        task_idx: int,
        placements: Optional[Dict[int, Tuple[int, int]]],
        available_replicas: List[Tuple[Node, Platform]]
    ) -> Tuple[Node, Platform]:
        """
        Select placement using pure GNN decision with shortest-queue fallback.
        This is the original logic, preserved when soft blending is disabled.
        """
        target_node, target_platform = None, None
        
        # Get GNN's placement decision
        gnn_placement = placements.get(task_idx) if placements else None
        
        if gnn_placement:
            target_node_id, target_plat_id = gnn_placement
            # Find the actual node/platform objects
            for node, plat in available_replicas:
                if node.id == target_node_id and plat.id == target_plat_id:
                    target_node, target_platform = node, plat
                    self.gnn_pure_decisions += 1
                    break
        
        # Fallback to shortest queue if GNN placement is invalid
        if target_node is None or target_platform is None:
            print(f"[ {self.env.now} ] GNN: Fallback to shortest queue for task {task.id}")
            initialized_replicas = [
                replica for replica in available_replicas if replica[1].initialized.triggered
            ]
            candidates = initialized_replicas if initialized_replicas else available_replicas
            target_node, target_platform = min(
                candidates, key=lambda couple: len(couple[1].queue.items)
            )
            self.fallback_decisions += 1
        
        return target_node, target_platform

    # ==================== State Capture Methods ====================
    
    @property
    def state_capture(self) -> StateCaptureHelper:
        """Lazy initialization of state capture helper."""
        if self._state_capture is None:
            self._state_capture = StateCaptureHelper(self.env, self.nodes)
        return self._state_capture
    
    def enable_state_capture(self, output_path: str):
        """Enable state capture and set output path."""
        self.state_capture.enable_capture(output_path)
    
    def disable_state_capture(self):
        """Disable state capture."""
        self.state_capture.disable_capture()
    
    def capture_task_placement(
        self,
        task: 'Task',
        execution_node: str,
        execution_platform: str,
        elapsed_time: float,
        valid_replicas: List[Tuple['Node', 'Platform']]
    ) -> Dict[str, Any]:
        """
        Capture a task placement decision with full state information.
        
        Args:
            task: The task being placed
            execution_node: Node where task will execute
            execution_platform: Platform ID where task will execute
            elapsed_time: Wall-clock time for scheduling decision
            valid_replicas: Set of valid replicas for this task
            
        Returns:
            Dict with placement information
        """
        # Calculate queue time
        queue_time = self.env.now - task.arrived_time if hasattr(task, 'arrived_time') else 0.0
        
        # Capture queue snapshots
        valid_replicas_set = set(valid_replicas)
        queue_snapshot_at_scheduling = self.state_capture.capture_queue_snapshot_for_replicas(valid_replicas_set)
        full_queue_snapshot = self.state_capture.capture_full_queue_snapshot()
        
        # Capture temporal state
        temporal_state_at_scheduling = self.state_capture.capture_temporal_state_for_replicas(valid_replicas_set)

        # Capture initialized state for all platforms (hidden FilterStore pull state)
        initialized_snapshot = self.state_capture.capture_initialized_snapshot()

        return self.state_capture.capture_task_placement(
            task=task,
            execution_node=execution_node,
            execution_platform=execution_platform,
            elapsed_time=elapsed_time,
            queue_time=queue_time,
            queue_snapshot_at_scheduling=queue_snapshot_at_scheduling,
            full_queue_snapshot=full_queue_snapshot,
            temporal_state_at_scheduling=temporal_state_at_scheduling,
            initialized_snapshot=initialized_snapshot,
        )
    
    def save_captured_state(self, system_state: 'SystemState', total_rtt: float = 0.0, output_path: Optional[str] = None):
        """Save captured state to JSON file."""
        self.state_capture.save_captured_state(system_state, total_rtt, output_path)
    
    def get_captured_state(self, system_state: 'SystemState', total_rtt: float = 0.0) -> Dict[str, Any]:
        """Get captured state as dictionary."""
        return self.state_capture.get_captured_state(system_state, total_rtt)
    
    def reset_state_capture(self):
        """Reset captured placements for a new simulation run."""
        self.state_capture.reset()

    # ==================== Direct State Capture (for task results) ====================
    
    def _capture_queue_snapshot_for_replicas(self, replicas: List[Tuple['Node', 'Platform']]) -> Dict[str, int]:
        """Capture queue lengths for a specific set of replicas."""
        queue_snapshot = {}
        for node, platform in replicas:
            key = f"{node.node_name}:{platform.id}"
            queue_snapshot[key] = len(platform.queue.items)
        return queue_snapshot
    
    def _capture_temporal_state_for_replicas(self, replicas: List[Tuple['Node', 'Platform']]) -> Dict[str, Dict[str, float]]:
        """Capture temporal state (remaining times) for a set of replicas."""
        temporal_state = {}
        now = self.env.now
        
        for node, platform in replicas:
            key = f"{node.node_name}:{platform.id}"
            
            current_task_remaining = 0.0
            cold_start_remaining = 0.0
            comm_remaining = 0.0
            
            if platform.current_task is not None:
                current_task = platform.current_task
                
                # Check if task is in cold start phase
                if current_task.cold_started and not hasattr(current_task, "started_time"):
                    cold_start_duration = current_task.type["coldStartDuration"].get(
                        platform.type["shortName"], 0.0
                    )
                    elapsed_cold_start = now - current_task.arrived_time
                    cold_start_remaining = max(0.0, cold_start_duration - elapsed_cold_start)
                
                # Check if task is executing
                if hasattr(current_task, "started_time") and current_task.started_time is not None:
                    exec_duration = current_task.type["executionTime"].get(
                        platform.type["shortName"], 0.0
                    )
                    elapsed_exec = now - current_task.started_time
                    current_task_remaining = max(0.0, exec_duration - elapsed_exec)
                    
                    # Estimate communication remaining
                    if current_task.application:
                        state_size_map = current_task.type.get("stateSize", {})
                        app_name = current_task.application.type.get("name", "")
                        if isinstance(state_size_map, dict) and app_name in state_size_map:
                            output_size = state_size_map[app_name].get("output", 0)
                            if isinstance(output_size, (int, float)) and output_size > 0:
                                throughput = 100.0 * 1024 * 1024  # 100 MB/s
                                latency = 0.001  # 1ms
                                comm_remaining = (output_size / throughput) + latency
            
            temporal_state[key] = {
                "current_task_remaining": current_task_remaining,
                "cold_start_remaining": cold_start_remaining,
                "comm_remaining": comm_remaining,
            }
        
        return temporal_state
