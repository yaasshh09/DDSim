from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from ddsim.core.config import CONFIG
from ddsim.device.builder import Device
from ddsim.device.state import DeviceState
from ddsim.device.transport import TimeStep, TransportModels, solve_bias_newton
import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq
from ddsim.discretize.coupled import Unknown, assemble_coupled_terms, pack
from ddsim.extract.ac import _dc
from ddsim.extract.iv import terminal_currents
import math

MAX_HALVINGS = 10


@dataclass(frozen=True)
class TransientPoint:
    time: float

    state: DeviceState

    currents: dict[str, float]

    voltages: dict[str, float]


@dataclass(frozen=True)
class Circuit:
    contact: str

    resistance: float = 0.0

    voltage: Callable[[float], float] | None = None

    current: Callable[[float], float] | None = None


@dataclass(frozen=True)
class TransientFrame:
    step: int

    time: float

    dt: float

    voltages: dict[str, float]

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


def _currents(device: Device, models: TransportModels, step: TimeStep, after: DeviceState, q: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    s = device.scale
    unit = s.J_0 * s.x_0 ** (device.dimension - 1)
    conduction = terminal_currents(device, after, models)
    a0, a1, a2 = step.weights
    q1 = _charges(device, after, models)
    rate = {k: a0 * q1[k] + a1 * q[-1][k] + (a2 * q[-2][k] if step.older is not None else 0.0) for k in conduction}
    return {k: conduction[k] + rate[k] / (step.dt / s.t_0) * unit for k in conduction}, q1


Solved = tuple[Device, DeviceState, dict[str, float], dict[str, float]]


def _plain(at: Device, models: TransportModels, step: TimeStep, max_iterations: int, on_frame: Callable[[object], None] | None, q: list[dict[str, float]]) -> Solved | None:
    nxt = _advance(at, models, step, max_iterations, on_frame)
    if nxt is None:
        return None
    i, q1 = _currents(at, models, step, nxt, q)
    return at, nxt, i, q1


class _DivergedError(Exception):
    pass


def _through(at: Device, models: TransportModels, step: TimeStep, max_iterations: int, on_frame: Callable[[object], None] | None, q: list[dict[str, float]], circuit: Circuit, t: float, guess: float) -> Solved | None:
    if circuit.voltage is not None and circuit.resistance == 0.0:
        return _plain(at.with_bias(**{circuit.contact: circuit.voltage(t)}), models, step, max_iterations, on_frame, q)
    cache: dict[float, Solved] = {}

    def wanted(v: float) -> float:
        if circuit.current is not None:
            return circuit.current(t)
        assert circuit.voltage is not None
        return (circuit.voltage(t) - v) / circuit.resistance

    def gap(v: float) -> float:
        if v not in cache:
            got = _plain(at.with_bias(**{circuit.contact: v}), models, step, max_iterations, on_frame, q)
            if got is None:
                raise _DivergedError
            cache[v] = got
        return cache[v][2][circuit.contact] - wanted(v)

    try:
        a, ga = guess, gap(guess)
        if ga == 0.0:
            return cache[a]
        dv = 1e-3 if ga < 0.0 else -1e-3
        b, gb = a + dv, gap(a + dv)
        for _ in range(14):
            if ga * gb <= 0.0:
                break
            a, ga = b, gb
            dv *= 2.0
            b, gb = a + dv, gap(a + dv)
        else:
            return None
        r = float(brentq(gap, min(a, b), max(a, b), xtol=1e-13, rtol=1e-15))
        gap(r)
    except _DivergedError:
        return None
    return cache[r]


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


def _check_circuit(device: Device, circuit: Circuit) -> None:
    names = [c.name for c in device.contacts]
    if circuit.contact not in names:
        raise KeyError(f"no contact named {circuit.contact!r} for the circuit, the contacts are {names}")
    if (circuit.voltage is None) == (circuit.current is None):
        raise ValueError("a circuit drives its contact from exactly one source, a voltage or a current")
    if not circuit.resistance >= 0.0:
        raise ValueError(f"a series resistance can't be negative, got {circuit.resistance:g}")


def _walk(device: Device, initial: DeviceState, times: Sequence[float], models: TransportModels | None, max_iterations: int, on_frame: Callable[[object], None] | None, bias: Callable[[float], dict[str, float]] | None, order: int, rtol: float | None = None, circuit: Circuit | None = None) -> Transient:
    if circuit is not None:
        _check_circuit(device, circuit)
    before = 0.0
    for t in times:
        if not t > before:
            raise ValueError(f"times have to increase from zero, and {t:g} s comes after {before:g} s")
        before = t
    if models is None:
        models = TransportModels.for_device(device)
    now, state = 0.0, initial
    held = {c.name: c.voltage for c in device.contacts}
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
                got = _plain(at, models, step, max_iterations, on_frame, q) if circuit is None else _through(at, models, step, max_iterations, on_frame, q, circuit, now + dt, held[circuit.contact])
                if got is not None:
                    break
                dt /= 2.0
            else:
                return Transient(points=tuple(out), complete=False, message=f"stalled at {now:g} s: the step was halved {MAX_HALVINGS} times to {dt:g} s and Newton still did not converge", steps=accepted, rejected=rejected)
            used, nxt, trial, q1 = got
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
            currents = trial
            q.append(q1)
            held = {c.name: c.voltage for c in used.contacts}
            older, state, last = state, nxt, dt
            if on_frame is not None:
                on_frame(TransientFrame(step=accepted, time=now, dt=dt, voltages=held, currents=currents))
        out.append(TransientPoint(time=target, state=state, currents=currents, voltages=held))
    return Transient(points=tuple(out), complete=True, steps=accepted, rejected=rejected)


def backward_euler(device: Device, initial: DeviceState, times: Sequence[float], models: TransportModels | None = None, max_iterations: int = CONFIG.bias.max_iterations, on_frame: Callable[[object], None] | None = None, bias: Callable[[float], dict[str, float]] | None = None, circuit: Circuit | None = None,) -> Transient:
    return _walk(device, initial, times, models, max_iterations, on_frame, bias, order=1, circuit=circuit)


def bdf2(device: Device, initial: DeviceState, times: Sequence[float], models: TransportModels | None = None, max_iterations: int = CONFIG.bias.max_iterations, on_frame: Callable[[object], None] | None = None, bias: Callable[[float], dict[str, float]] | None = None, rtol: float | None = None, circuit: Circuit | None = None,) -> Transient:
    return _walk(device, initial, times, models, max_iterations, on_frame, bias, order=2, rtol=rtol, circuit=circuit)


def check_times(times: Sequence[float]) -> None:
    if not times:
        raise ValueError("a transient needs at least one sample time [s]")
    before = 0.0
    for t in times:
        if not (math.isfinite(t) and t > before):
            raise ValueError(f"sample times have to be finite and rise from zero, and {t!r} s comes after {before:g} s")
        before = t


@dataclass(frozen=True)
class TransientCurve:
    contact: str

    points: tuple[TransientPoint, ...]

    complete: bool

    message: str = ""

    steps: int = 0

    rejected: int = 0


def transient_sweep(device: Device, contact: str, voltages: list[float], times: list[float], models: TransportModels | None = None, resistance: float = 0.0, rtol: float = CONFIG.sweeps.transient_rtol, max_iterations: int = CONFIG.bias.max_iterations, on_frame: Callable[[object], None] | None = None,) -> TransientCurve:
    if len(voltages) != 2:
        raise ValueError(f"a step takes two voltages, the one before and the one after, and got {len(voltages)}")
    before, after = voltages
    circuit = Circuit(contact=contact, resistance=resistance, voltage=lambda t: after)
    _check_circuit(device, circuit)
    check_times(times)
    if not (math.isfinite(rtol) and rtol > 0.0):
        raise ValueError(f"rtol has to be a positive number, got {rtol!r}")
    if models is None:
        models = TransportModels.for_device(device)
    start = device.with_bias(**{contact: before})
    s = _dc(start, models, max_iterations, on_frame)
    if s is None:
        return TransientCurve(contact=contact, points=(), complete=False, message=f"the DC state at {before:+g} V did not converge, so there is nothing to step from")
    t = bdf2(start, s, times, models, max_iterations, on_frame, rtol=rtol, circuit=circuit)
    return TransientCurve(contact=contact, points=t.points, complete=t.complete, message=t.message, steps=t.steps, rejected=t.rejected)


def _crossing(t0: float, y0: float, t1: float, y1: float, level: float) -> float:
    return t0 + (t1 - t0) * (y0 - level) / (y0 - y1)


def _waveform(transient: Transient, contact: str, field: str) -> list[tuple[float, float]]:
    if not transient.points or contact not in getattr(transient.points[0], field):
        raise KeyError(f"no contact named {contact!r} in this transient")
    return [(p.time, getattr(p, field)[contact]) for p in transient.points]


def storage_time(transient: Transient, contact: str) -> float:
    w = _waveform(transient, contact, "voltages")
    for (t0, v0), (t1, v1) in zip(w[:-1], w[1:], strict=True):
        if v0 > 0.0 >= v1:
            return _crossing(t0, v0, t1, v1, 0.0)
    raise ValueError(f"the {contact} voltage never crosses zero going down, so there is no storage time in this run")


def recovery_time(transient: Transient, contact: str, fraction: float = 0.1) -> float:
    ts = storage_time(transient, contact)
    w = _waveform(transient, contact, "currents")
    level = fraction * abs(w[0][1])
    for (t0, i0), (t1, i1) in zip(w[:-1], w[1:], strict=True):
        if t1 > ts and abs(i0) > level >= abs(i1):
            return _crossing(t0, abs(i0), t1, abs(i1), level)
    raise ValueError(f"the {contact} current never falls to {fraction:g} of its reverse value in this run")
