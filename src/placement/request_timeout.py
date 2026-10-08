"""Request timeout (R1.1): a task placed on a replica and still waiting for an unplaced peer REQUEST_TIMEOUT_S after its
placement fails (Platform._fail_task). Knative's default revision `timeoutSeconds`.

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


def expire_requests(env: Any) -> int:
    """Interrupt every rendezvous wait whose request deadline has passed. Called once per autoscaler tick."""
    waits = env.__dict__.get("request_waits")
    if not waits:
        return 0
    return sum(interrupt_if_waiting(proc, REQUEST_TIMEOUT) for deadline, proc in list(waits.values()) if env.now >= deadline)
