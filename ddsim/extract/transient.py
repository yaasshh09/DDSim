from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from ddsim.core.config import CONFIG
from ddsim.device.builder import Device
from ddsim.device.state import DeviceState
from ddsim.device.transport import TimeStep, TransportModels, solve_bias_newton
import numpy as np
import numpy.typing as npt
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

    steps: int = 0

    rejected: int = 0


def _advance(device: Device, models: TransportModels, step: TimeStep, max_iterations: int, on_frame: Callable[[object], None] | None) -> DeviceState | None:
    s = solve_bias_newton(device, models, guess=step.previous, max_iterations=max_iterations, on_frame=on_frame, step=step)
    return s if s.newton is not None and s.newton.converged else None


def _charges(device: Device, state: DeviceState, models: TransportModels) -> dict[str, float]:
    mesh = device.scaled_mesh
    x = pack(state.psi.data, state.n.data, state.p.data)
    bare, _ = assemble_coupled_terms(mesh.h, device.charge_volume_scaled, x, device.poisson_doping_scaled.data, models.Dn, models.Dp, models.recombination, mesh.geometry, device.degeneracy, device.ionization, device.net_doping_scaled.data)
    psi = bare.residual[Unknown.PSI::3]
    return {c.name: float(sum(psi[i] for i in c.nodes)) for c in device.contacts}


def _currents(device: Device, models: TransportModels, step: TimeStep, after: DeviceState, q: list[dict[str, float]]) -> dict[str, float]:
    s = device.scale
    unit = s.J_0 * s.x_0 ** (device.dimension - 1)
    conduction = terminal_currents(device, after, models)
    a0, a1, a2 = step.weights
    q1 = _charges(device, after, models)
    rate = {k: a0 * q1[k] + a1 * q[-1][k] + (a2 * q[-2][k] if step.older is not None else 0.0) for k in conduction}
    q.append(q1)
    return {k: conduction[k] + rate[k] / (step.dt / s.t_0) * unit for k in conduction}


def _packed(state: DeviceState) -> npt.NDArray[np.float64]:
    return pack(state.psi.data, state.n.data, state.p.data)


def _error(device: Device, rtol: float, t: list[float], x: list[npt.NDArray[np.float64]], now: float, corrected: npt.NDArray[np.float64]) -> float:
    predicted = np.zeros_like(corrected)
    for i in range(3):
        w = 1.0
        for j in range(3):
            if j != i:
                w *= (now - t[j]) / (t[i] - t[j])
        predicted += w * x[i]
    lte = (now - t[2]) / (now - t[0]) * np.abs(corrected - predicted)
    floor = np.full(corrected.size, rtol * device.material.n_i / device.scale.C_0)
    floor[Unknown.PSI::3] = rtol
    return float(np.max(lte / (floor + rtol * np.abs(corrected))))


def _walk(device: Device, initial: DeviceState, times: Sequence[float], models: TransportModels | None, max_iterations: int, on_frame: Callable[[object], None] | None, bias: Callable[[float], dict[str, float]] | None, order: int, rtol: float | None = None) -> Transient:
    before = 0.0
    for t in times:
        if not t > before:
            raise ValueError(f"times have to increase from zero, and {t:g} s comes after {before:g} s")
        before = t
    if models is None:
        models = TransportModels.for_device(device)
    now, state = 0.0, initial
    older: DeviceState | None = None
    last = 0.0
    q = [_charges(device, initial, models)]
    currents: dict[str, float] = {}
    out: list[TransientPoint] = []
    ht, hx = [0.0], [_packed(initial)]
    planned = times[0] / 100.0
    accepted = rejected = 0
    for target in times:
        while now < target:
            dt = target - now if rtol is None else min(planned, target - now)
            for _ in range(MAX_HALVINGS + 1):
                step = TimeStep(previous=state, dt=dt) if order == 1 or older is None else TimeStep(previous=state, dt=dt, older=older, ratio=dt / last)
                at = device if bias is None else device.with_bias(**bias(now + dt))
                nxt = _advance(at, models, step, max_iterations, on_frame)
                if nxt is not None:
                    break
                dt /= 2.0
            else:
                return Transient(points=tuple(out), complete=False, message=f"stalled at {now:g} s: the step was halved {MAX_HALVINGS} times to {dt:g} s and Newton still did not converge", steps=accepted, rejected=rejected)
            if rtol is not None and len(ht) >= 3:
                err = _error(device, rtol, ht[-3:], hx[-3:], now + dt, _packed(nxt))
                planned = dt * min(2.0, max(0.2, 0.9 * err ** (-1.0 / 3.0))) if err > 0.0 else 2.0 * dt
                if err > 1.0:
                    rejected += 1
                    continue
            elif rtol is not None:
                planned = dt
            now = target if dt == target - now else now + dt
            accepted += 1
            ht, hx = (ht + [now])[-3:], (hx + [_packed(nxt)])[-3:]
            currents = _currents(device, models, step, nxt, q)
            older, state, last = state, nxt, dt
        out.append(TransientPoint(time=target, state=state, currents=currents))
    return Transient(points=tuple(out), complete=True, steps=accepted, rejected=rejected)


def backward_euler(device: Device, initial: DeviceState, times: Sequence[float], models: TransportModels | None = None, max_iterations: int = CONFIG.bias.max_iterations, on_frame: Callable[[object], None] | None = None, bias: Callable[[float], dict[str, float]] | None = None,) -> Transient:
    return _walk(device, initial, times, models, max_iterations, on_frame, bias, order=1)


def bdf2(device: Device, initial: DeviceState, times: Sequence[float], models: TransportModels | None = None, max_iterations: int = CONFIG.bias.max_iterations, on_frame: Callable[[object], None] | None = None, bias: Callable[[float], dict[str, float]] | None = None, rtol: float | None = None,) -> Transient:
    return _walk(device, initial, times, models, max_iterations, on_frame, bias, order=2, rtol=rtol)
