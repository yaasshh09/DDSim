from __future__ import annotations
import pytest
from ddsim.device.equilibrium import frozen_quasi_fermi, solve_equilibrium
from ddsim.device.mosfet import DRAIN, GATE, SOURCE, nmos
from ddsim.device.transport import TransportModels, solve_bias_newton, solve_bias_ramped
from ddsim.extract import cv
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


@pytest.mark.parametrize("vg", [-0.5, 0.5, 1.5])
def test_at_zero_drain_bias_cgg_is_the_quasi_static_gate_capacitance(vg):
    d = nmos(gate_voltage=vg)
    m = TransportModels.for_device(d)
    q = frozen_quasi_fermi(d)
    eq = solve_equilibrium(d, quasi_fermi=q)
    a = admittance(d, solve_bias_newton(d, m, guess=eq), 1e3, m)
    g = a.contacts.index(GATE)
    assert a.C[g, g] == pytest.approx(cv.small_signal_capacitance(d, eq, GATE, cv.Response.LOW_FREQUENCY, quasi_fermi=q, width=1.0), rel=1e-7, abs=0)
    assert a.C[g, a.contacts.index(SOURCE)] == pytest.approx(a.C[g, a.contacts.index(DRAIN)], rel=1e-6, abs=0)
