"""Request timeout (R1.1-T): a task placed on a replica and still waiting for a peer that has arrived but is unplaced,
REQUEST_TIMEOUT_S after the later of its placement and that peer's arrival, fails (Platform._fail_task). It never fires
while a partner has not arrived. Knative's default revision `timeoutSeconds`.

The deadline of each rendezvous wait is registered in `env.request_waits` and swept by the autoscaler tick; it is never a
SimPy event. The tick's own `env.step()` consumes the next queued event, so one extra scheduled event shifts which
event it consumes and changes every run, whether or not it fires.
"""
from typing import Any

REQUEST_TIMEOUT = "request_timeout"
REQUEST_TIMEOUT_S = 300.0


def interrupt_if_waiting(proc: Any, cause: str) -> bool:
    """Interrupt `proc` unless it was already interrupted at this instant: a second interrupt (a rendezvous release and
    a request timeout together) would be thrown into whatever the process yields after the first."""
    env = proc.env
    if not proc.is_alive or getattr(proc, "_interrupted_at", None) == env.now:
        return False
    proc._interrupted_at = env.now
    proc.interrupt(cause)
    return True


def request_deadline(placed_at: float, unplaced: Any) -> Any:
    """When a request waiting on `unplaced` partners times out, or None while it cannot: never while a partner has not
    arrived (None, or no dispatch time yet); otherwise REQUEST_TIMEOUT_S after the later of its placement and the last
    arrival among the partners it still waits for (R1.1-T)."""
    start = placed_at
    for peer in unplaced:
        arrived = getattr(peer, "dispatched_time", None) if peer is not None else None
        if arrived is None:
            return None
        start = max(start, arrived)
    return start + REQUEST_TIMEOUT_S if unplaced else None


def expire_requests(env: Any) -> int:
    """Interrupt every rendezvous wait whose request deadline has passed. Called once per autoscaler tick."""
    waits = env.__dict__.get("request_waits")
    if not waits:
        return 0
    n = 0
    for placed_at, proc, unplaced in list(waits.values()):
        deadline = request_deadline(placed_at, unplaced())
        if deadline is not None and env.now >= deadline:
            n += interrupt_if_waiting(proc, REQUEST_TIMEOUT)
    return n
