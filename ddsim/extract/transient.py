from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from ddsim.core.config import CONFIG
from ddsim.device.builder import Device
from ddsim.device.state import DeviceState
from ddsim.device.transport import TimeStep, TransportModels, solve_bias_newton
from ddsim.discretize.coupled import Unknown, assemble_coupled_terms, pack
from ddsim.extract.iv import terminal_currents

MAX_HALVINGS = 10


@dataclass(frozen=True)
class TransientPoint:
    time: float

    state: DeviceState

    currents: dict[str, float]


@dataclass(frozen=True)
class Transient:
    points: tuple[TransientPoint, ...]

    complete: bool

    message: str = ""


def _advance(device: Device, models: TransportModels, state: DeviceState, dt: float, max_iterations: int, on_frame: Callable[[object], None] | None) -> DeviceState | None:
    s = solve_bias_newton(device, models, guess=state, max_iterations=max_iterations, on_frame=on_frame, step=TimeStep(previous=state, dt=dt))
    return s if s.newton is not None and s.newton.converged else None


def _charges(device: Device, state: DeviceState, models: TransportModels) -> dict[str, float]:
    mesh = device.scaled_mesh
    x = pack(state.psi.data, state.n.data, state.p.data)
    bare, _ = assemble_coupled_terms(mesh.h, device.charge_volume_scaled, x, device.poisson_doping_scaled.data, models.Dn, models.Dp, models.recombination, mesh.geometry, device.degeneracy, device.ionization, device.net_doping_scaled.data)
    psi = bare.residual[Unknown.PSI::3]
    return {c.name: float(sum(psi[i] for i in c.nodes)) for c in device.contacts}


def _currents(device: Device, models: TransportModels, before: DeviceState, after: DeviceState, dt: float) -> dict[str, float]:
    s = device.scale
    unit = s.J_0 * s.x_0 ** (device.dimension - 1)
    conduction = terminal_currents(device, after, models)
    q0, q1 = _charges(device, before, models), _charges(device, after, models)
    return {k: conduction[k] + (q1[k] - q0[k]) / (dt / s.t_0) * unit for k in conduction}


def backward_euler(device: Device, initial: DeviceState, times: Sequence[float], models: TransportModels | None = None, max_iterations: int = CONFIG.bias.max_iterations, on_frame: Callable[[object], None] | None = None,) -> Transient:
    before = 0.0
    for t in times:
        if not t > before:
            raise ValueError(f"times have to increase from zero, and {t:g} s comes after {before:g} s")
        before = t
    if models is None:
        models = TransportModels.for_device(device)
    now, state = 0.0, initial
    currents: dict[str, float] = {}
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
            currents = _currents(device, models, state, nxt, dt)
            state = nxt
        out.append(TransientPoint(time=target, state=state, currents=currents))
    return Transient(points=tuple(out), complete=True)
