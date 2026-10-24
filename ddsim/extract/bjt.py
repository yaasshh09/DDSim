from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq
from ddsim.core.config import CONFIG
from ddsim.device.bjt import BASE, COLLECTOR, EMITTER
from ddsim.device.builder import Device
from ddsim.device.state import DeviceState
from ddsim.device.transport import TransportModels, solve_bias_newton, solve_bias_ramped
from ddsim.extract.ac import admittance
from ddsim.extract.iv import IVFrame, gate_sweep, terminal_currents
from ddsim.extract.params import saturation_current

Array = npt.NDArray[np.float64]

FT_FREQUENCY = 1e6


@dataclass(frozen=True)
class GummelPoint:
    v_be: float

    collector: float

    base: float

    f_T: float | None

    state: DeviceState


@dataclass(frozen=True)
class GummelPlot:
    points: tuple[GummelPoint, ...]

    complete: bool

    message: str = ""

    @property
    def v_be(self) -> Array:
        return np.array([p.v_be for p in self.points])

    @property
    def collector(self) -> Array:
        return np.array([p.collector for p in self.points])

    @property
    def base(self) -> Array:
        return np.array([p.base for p in self.points])

    @property
    def beta(self) -> Array:
        return self.collector / self.base

    @property
    def f_T(self) -> Array:
        return np.array([np.nan if p.f_T is None else p.f_T for p in self.points])


def gummel_plot(device: Device, v_be: Sequence[float], models: TransportModels | None = None, step: float = CONFIG.sweeps.iv_step, f_t: bool = True, on_frame: Callable[[object], None] | None = None) -> GummelPlot:
    if models is None:
        models = TransportModels.for_device(device)
    c = gate_sweep(device, [-v for v in v_be], contact=EMITTER, measure_at=COLLECTOR, models=models, step=step, on_frame=on_frame)
    out = []
    for p in c.points:
        d = device.with_bias(**{EMITTER: p.voltage})
        i = terminal_currents(d, p.state, models)
        f = admittance(d, p.state, FT_FREQUENCY, models).f_T(BASE, COLLECTOR) if f_t else None
        out.append(GummelPoint(v_be=-p.voltage, collector=i[COLLECTOR], base=i[BASE], f_T=f, state=p.state))
    return GummelPlot(points=tuple(out), complete=c.complete, message=c.message)


def ideality(v_be: Array, current: Array, window: tuple[float, float]) -> float:
    return saturation_current(np.asarray(v_be), np.asarray(current), window=window)[1]


def peak_ft(plot: GummelPlot) -> tuple[float, float]:
    f = plot.f_T
    if np.all(np.isnan(f)):
        raise ValueError("the Gummel plot was run without f_T")
    k = int(np.nanargmax(f))
    return float(f[k]), float(plot.collector[k])


def early_voltage(device: Device, v_be: float, v_ce: Sequence[float], models: TransportModels | None = None) -> float:
    if len(v_ce) < 2:
        raise ValueError(f"an Early voltage needs at least two collector voltages, got {len(v_ce)}")
    d = device.with_bias(**{BASE: v_be})
    if models is None:
        models = TransportModels.for_device(d)
    c = gate_sweep(d, list(v_ce), contact=COLLECTOR, measure_at=COLLECTOR, models=models)
    if not c.complete:
        raise RuntimeError(f"the output curve stopped early: {c.message}")
    slope, cut = np.polyfit(c.voltage, c.current, 1)
    return float(cut / slope)


@dataclass(frozen=True)
class OutputPoint:
    v_ce: float

    v_be: float

    collector: float

    base: float

    state: DeviceState


@dataclass(frozen=True)
class OutputCurve:
    i_b: float

    points: tuple[OutputPoint, ...]

    complete: bool

    message: str = ""

    @property
    def v_ce(self) -> Array:
        return np.array([p.v_ce for p in self.points])

    @property
    def v_be(self) -> Array:
        return np.array([p.v_be for p in self.points])

    @property
    def collector(self) -> Array:
        return np.array([p.collector for p in self.points])

    @property
    def base(self) -> Array:
        return np.array([p.base for p in self.points])


V_BE_BRACKET = (0.1, 1.3)


def output_curve(device: Device, i_b: float, v_ce: Sequence[float], models: TransportModels | None = None, rtol: float = 1e-9, on_frame: Callable[[object], None] | None = None) -> OutputCurve:
    if not i_b > 0.0:
        raise ValueError(f"the base current has to be above zero, got {i_b:g} A/cm^2")
    if models is None:
        models = TransportModels.for_device(device)
    m = models
    known: list[tuple[float, float, DeviceState]] = []

    def at(v: float, u: float) -> tuple[DeviceState, dict[str, float]]:
        d = device.with_bias(**{BASE: v, COLLECTOR: u})
        s = None
        if known:
            g = min(known, key=lambda k: abs(k[0] - u) + abs(k[1] - v))[2]
            s = solve_bias_newton(d, m, guess=g)
        if s is None or s.newton is None or not s.newton.converged:
            s = solve_bias_ramped(d, m)
        if s.newton is None or not s.newton.converged:
            raise RuntimeError(f"no solve at V_BE {v:g} V and V_CE {u:g} V")
        known.append((u, v, s))
        return s, terminal_currents(d, s, m)

    out: list[OutputPoint] = []
    for u in v_ce:
        try:
            def miss(x: float, u: float = u) -> float:
                j = at(x, u)[1][BASE]
                return math.log(j / i_b) if j > 0.0 else -50.0

            v = brentq(miss, *V_BE_BRACKET, xtol=1e-12, rtol=rtol)
            s, i = at(v, u)
        except (RuntimeError, ValueError) as e:
            return OutputCurve(i_b=i_b, points=tuple(out), complete=False, message=f"stopped at V_CE {u:g} V: {e}")
        out.append(OutputPoint(v_ce=u, v_be=v, collector=i[COLLECTOR], base=i[BASE], state=s))
        if on_frame is not None:
            on_frame(IVFrame(index=len(out) - 1, voltage=u, current=i[COLLECTOR]))
    return OutputCurve(i_b=i_b, points=tuple(out), complete=True)
