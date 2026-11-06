from __future__ import annotations
import math
from dataclasses import replace
from functools import cache
import pytest
from ddsim.core import constants as C
from ddsim.device.bjt import BASE, COLLECTOR, npn_1d
from ddsim.device.builder import Device
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


def test_the_base_current_does_not_see_the_base_gap() -> None:
    si, ge = currents(None, 0.45), currents(0.2, 0.45)
    assert ge[1] == pytest.approx(si[1], rel=0.05)
