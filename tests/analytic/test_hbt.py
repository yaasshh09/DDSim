from __future__ import annotations
import math
from dataclasses import replace
from functools import cache
import numpy as np
import pytest
from ddsim.core import constants as C
from ddsim.device.bjt import BASE, COLLECTOR, npn_1d
from ddsim.device.builder import Device
from ddsim.device.state import DeviceState
from ddsim.device.transport import TransportModels, solve_bias_ramped
from ddsim.extract.iv import terminal_currents
from ddsim.physics.bands import Semiconductor
from ddsim.physics.recombination import NoRecombination

V_BE = (0.40, 0.45, 0.50)


def ideal(d: Device) -> TransportModels:
    return replace(TransportModels.for_device(d), recombination=NoRecombination())


@cache
def currents(x: float | None, v: float) -> tuple[float, float]:
    base = None if x is None else Semiconductor.sige(x)
    d = npn_1d(base_material=base).with_bias(base=v, collector=v)
    s = solve_bias_ramped(d, ideal(d))
    assert s.newton is not None and s.newton.converged
    i = terminal_currents(d, s, ideal(d))
    return i[COLLECTOR], i[BASE]


@pytest.mark.parametrize("v", V_BE)
def test_a_silicon_base_through_the_band_path_is_the_plain_npn(v: float) -> None:
    a, b = currents(None, v), currents(0.0, v)
    assert b[0] == pytest.approx(a[0], rel=1e-9, abs=0)
    assert b[1] == pytest.approx(a[1], rel=1e-9, abs=0)


@pytest.mark.parametrize("x", [0.1, 0.2])
@pytest.mark.parametrize("v", V_BE)
def test_the_gain_rises_as_exp_of_the_base_gap_drop(x: float, v: float) -> None:
    drop = Semiconductor.silicon().Eg - Semiconductor.sige(x).Eg
    want = math.exp(drop / C.V_T())
    si, ge = currents(None, v), currents(x, v)
    assert ge[0] / si[0] == pytest.approx(want, rel=0.10)
    assert (ge[0] / ge[1]) / (si[0] / si[1]) == pytest.approx(want, rel=0.10)


@cache
def graded(top: float, v: float) -> tuple[Device, DeviceState, float]:
    d = npn_1d(base_material=lambda t: Semiconductor.sige(top * t)).with_bias(base=v, collector=v)
    s = solve_bias_ramped(d, ideal(d))
    assert s.newton is not None and s.newton.converged
    return d, s, terminal_currents(d, s, ideal(d))[COLLECTOR]


@pytest.mark.parametrize("top", [0.1, 0.2])
@pytest.mark.parametrize("v", [0.40, 0.50])
def test_a_graded_base_carries_gummels_current_with_its_local_n_i(top: float, v: float) -> None:
    d, s, got = graded(top, v)
    assert d.bands is not None
    b = d.net_doping.data < 0.0
    p = s.p.to_physical(d.scale).data
    ni2 = d.material.n_i ** 2 * d.bands.n_local ** 2
    gummel = C.q * math.exp(v / C.V_T()) / float(np.sum((p / (C.D_n() * ni2) * d.mesh_1d.volume)[b]))
    assert got == pytest.approx(gummel, rel=0.01)


def test_a_flat_grade_is_the_uniform_base() -> None:
    d = npn_1d(base_material=lambda t: Semiconductor.sige(0.1)).with_bias(base=0.45, collector=0.45)
    s = solve_bias_ramped(d, ideal(d))
    assert s.newton is not None and s.newton.converged
    assert terminal_currents(d, s, ideal(d))[COLLECTOR] == currents(0.1, 0.45)[0]


@pytest.mark.parametrize("top", [0.1, 0.2])
def test_a_graded_base_gains_less_than_the_top_of_its_grade(top: float) -> None:
    ratio = graded(top, 0.45)[2] / currents(None, 0.45)[0]
    assert 1.5 < ratio < currents(top, 0.45)[0] / currents(None, 0.45)[0]


def test_the_base_current_does_not_see_the_base_gap() -> None:
    si, ge = currents(None, 0.45), currents(0.2, 0.45)
    assert ge[1] == pytest.approx(si[1], rel=0.05)
