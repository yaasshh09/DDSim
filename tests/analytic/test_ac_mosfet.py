from __future__ import annotations
import pytest
from ddsim.device.mosfet import DRAIN, GATE, nmos
from ddsim.device.transport import TransportModels, solve_bias_newton, solve_bias_ramped
from ddsim.extract.ac import admittance
from ddsim.extract.iv import terminal_currents

VG = 1.0


@pytest.mark.parametrize("field_dependent", [False, True])
@pytest.mark.parametrize("vd", [0.05, 1.0])
@pytest.mark.parametrize(("driven", "H", "rel"), [(GATE, 4e-3, 1e-4), (DRAIN, 2e-2, 1e-3)])
def test_low_frequency_y_is_the_dc_slope(driven, H, rel, vd, field_dependent):
    d = nmos(gate_voltage=VG, drain_voltage=vd)
    m = TransportModels.for_device(d, field_dependent=field_dependent)
    s = solve_bias_ramped(d, m)
    assert s.newton.converged

    def current(dv):
        e = nmos(gate_voltage=VG + dv * (driven == GATE), drain_voltage=vd + dv * (driven == DRAIN))
        return terminal_currents(e, solve_bias_newton(e, m, guess=s), m)[DRAIN]

    a = (current(H) - current(-H)) / (2 * H)
    b = (current(2 * H) - current(-2 * H)) / (4 * H)
    y = admittance(d, s, 1e3, m).y(DRAIN, driven)
    assert y.real == pytest.approx((4 * a - b) / 3, rel=rel)
