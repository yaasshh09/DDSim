from __future__ import annotations
import math
from dataclasses import replace
from functools import cache
import numpy as np
import pytest
from ddsim.core import constants as C
from ddsim.device.bjt import BASE, COLLECTOR, EMITTER, npn_1d
from ddsim.device.builder import Device
from ddsim.device.state import DeviceState
from ddsim.device.transport import TransportModels, solve_bias_newton, solve_bias_ramped
from ddsim.extract.iv import terminal_currents
from ddsim.physics.recombination import NoRecombination

LOW_INJECTION = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


def ideal(d: Device) -> TransportModels:
    return replace(TransportModels.for_device(d), recombination=NoRecombination())


@cache
def biased(v: float) -> tuple[Device, DeviceState]:
    d = npn_1d().with_bias(base=v, collector=v)
    m = ideal(d)
    s = solve_bias_ramped(d, m)
    assert s.newton is not None and s.newton.converged
    return d, s


def gummel_current(d: Device, s: DeviceState, v: float) -> float:
    x = d.mesh_1d.x
    w = d.mesh_1d.volume
    b = d.net_doping.data < 0.0
    p = s.p.to_physical(d.scale).data
    T = d.material.T
    integral = float(np.sum((p / (C.D_n(T) * d.material.n_i ** 2) * w)[b]))
    assert x[b].min() > 0.0 and x[b].max() < x[-1]
    return C.q * math.exp(v / C.V_T(T)) / integral


def test_the_base_contact_lets_electrons_through() -> None:
    d, s = biased(0.6)
    k = [c for c in d.contacts if c.name == BASE][0].nodes[0]
    n = s.n.to_physical(d.scale).data[k]
    Na = -d.net_doping.data[k]
    assert n > 1e3 * d.material.n_i ** 2 / Na, "the base contact pins the electrons to equilibrium, so no electron reaches the collector"
    assert s.p.to_physical(d.scale).data[k] == pytest.approx(Na, rel=1e-9)


def test_the_three_terminal_currents_add_up_to_zero() -> None:
    d, s = biased(0.6)
    i = terminal_currents(d, s, ideal(d))
    assert abs(i[EMITTER] + i[BASE] + i[COLLECTOR]) <= 1e-9 * abs(i[COLLECTOR])


@pytest.mark.parametrize("v", LOW_INJECTION)
def test_the_collector_current_is_gummels(v: float) -> None:
    d, s = biased(v)
    got = terminal_currents(d, s, ideal(d))[COLLECTOR]
    assert got > 0.0
    assert got == pytest.approx(gummel_current(d, s, v), rel=0.02)


def test_the_collector_current_has_an_ideality_of_one() -> None:
    i = [terminal_currents(*biased(v), ideal(biased(v)[0]))[COLLECTOR] for v in LOW_INJECTION]
    slope = np.polyfit(LOW_INJECTION, np.log(i), 1)[0]
    assert 1.0 / (slope * C.V_T()) == pytest.approx(1.0, abs=0.01)


def test_with_no_recombination_the_base_current_is_the_holes_into_the_emitter() -> None:
    d, s = biased(0.6)
    i = terminal_currents(d, s, ideal(d))
    assert i[BASE] > 0.0
    assert i[BASE] < 0.1 * i[COLLECTOR], "an emitter ten times heavier than the base should inject far more electrons than it takes holes"


def test_a_cold_start_converges_without_a_ramp() -> None:
    d = npn_1d().with_bias(base=0.3, collector=0.3)
    s = solve_bias_newton(d, ideal(d))
    assert s.newton is not None and s.newton.converged
