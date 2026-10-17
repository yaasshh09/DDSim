from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from ddsim.core.config import CONFIG
from ddsim.device.builder import Device
from ddsim.device.state import DeviceState
from ddsim.device.transport import TimeStep, TransportModels, solve_bias_newton

MAX_HALVINGS = 10


@dataclass(frozen=True)
class TransientPoint:
    time: float

    state: DeviceState


@dataclass(frozen=True)
class Transient:
    points: tuple[TransientPoint, ...]

    complete: bool

    message: str = ""


def _advance(device: Device, models: TransportModels, state: DeviceState, dt: float, max_iterations: int, on_frame: Callable[[object], None] | None) -> DeviceState | None:
    s = solve_bias_newton(device, models, guess=state, max_iterations=max_iterations, on_frame=on_frame, step=TimeStep(previous=state, dt=dt))
    return s if s.newton is not None and s.newton.converged else None


def backward_euler(device: Device, initial: DeviceState, times: Sequence[float], models: TransportModels | None = None, max_iterations: int = CONFIG.bias.max_iterations, on_frame: Callable[[object], None] | None = None,) -> Transient:
    before = 0.0
    for t in times:
        if not t > before:
            raise ValueError(f"times have to increase from zero, and {t:g} s comes after {before:g} s")
        before = t
    if models is None:
        models = TransportModels.for_device(device)
    now, state = 0.0, initial
    out: list[TransientPoint] = []
    for target in times:
        while now < target:
            dt = target - now
            for _ in range(MAX_HALVINGS + 1):
                nxt = _advance(device, models, state, dt, max_iterations, on_frame)
                if nxt is not None:
                    break
                dt /= 2.0
            else:
                return Transient(points=tuple(out), complete=False, message=f"stalled at {now:g} s: the step was halved {MAX_HALVINGS} times to {dt:g} s and Newton still did not converge")
            now = target if dt == target - now else now + dt
            state = nxt
        out.append(TransientPoint(time=target, state=state))
    return Transient(points=tuple(out), complete=True)
