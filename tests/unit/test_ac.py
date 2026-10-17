from __future__ import annotations
from dataclasses import replace
import numpy as np, pytest
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import TransportModels, solve_bias_newton, solve_bias_ramped
from ddsim.extract.ac import admittance
from ddsim.extract.iv import terminal_currents

V = 0.4


@pytest.fixture(scope="module")
def diode():
    d = pn_diode(anode_voltage=V)
    m = TransportModels.for_device(d)
    return d, m, solve_bias_ramped(d, m)


def dc_current(diode, at):
    d, m, s = diode
    e = d.with_bias(anode=at)
    return terminal_currents(e, solve_bias_newton(e, m, guess=s), m)


def test_zero_frequency_is_the_dc_conductance(diode):
    d, m, s = diode
    y = admittance(d, s, 0.0, m)
    h = 5e-4

    def slope(step, name):
        return (dc_current(diode, V + step)[name] - dc_current(diode, V - step)[name]) / (2 * step)

    for name in ("anode", "cathode"):
        a, b = slope(h, name), slope(2 * h, name)
        want = (4 * a - b) / 3
        assert y.y(name, "anode") == pytest.approx(want, rel=1e-6)


def test_zero_frequency_is_real(diode):
    d, m, s = diode
    assert np.all(admittance(d, s, 0.0, m).Y.imag == 0.0)


@pytest.mark.parametrize("f", [1e3, 1e7, 1e10])
def test_currents_and_voltages_both_sum_to_zero(diode, f):
    d, m, s = diode
    Y = admittance(d, s, f, m).Y
    g = float(np.max(np.abs(Y.real)))
    c = float(np.max(np.abs(Y.imag)))
    for k in (0, 1):
        assert np.max(np.abs(Y.sum(axis=k).real)) < 1e-8 * g
        assert np.max(np.abs(Y.sum(axis=k).imag)) < 1e-11 * c


def test_refuses_a_state_newton_did_not_converge(diode):
    d, m, s = diode
    bad = replace(s, newton=replace(s.newton, converged=False))
    with pytest.raises(ValueError, match="converge"):
        admittance(d, bad, 1e6, m)


def test_capacitance_needs_a_frequency(diode):
    d, m, s = diode
    with pytest.raises(ValueError, match="frequency"):
        _ = admittance(d, s, 0.0, m).C
