from __future__ import annotations
import math
from dataclasses import replace
import numpy as np
import pytest
from ddsim.core import constants as C
from ddsim.device.schottky import schottky_diode
from ddsim.device.transport import TransportModels
from ddsim.extract.ac import ac_sweep, admittance
from ddsim.extract.iv import gate_sweep, iv_sweep, terminal_currents
from ddsim.physics.recombination import NoRecombination

BARRIER = 0.7


def models(d, faster=1.0):
    m = TransportModels.for_device(d, recombination=NoRecombination())
    return replace(m, Dn=m.Dn * faster)


def thermionic(v, T=C.T_ROOM):
    return C.A_STAR_N * T**2 * math.exp(-BARRIER / C.V_T(T)) * math.expm1(v / C.V_T(T))


def test_the_contact_holds_the_barrier_density_and_no_current_at_zero_bias() -> None:
    d = schottky_diode(barrier=BARRIER)
    m = models(d)
    c = iv_sweep(d, "anode", [0.0], models=m)
    s = c.points[0].state
    n0 = C.Nc() * math.exp(-BARRIER / C.V_T())
    assert s.n.data[0] * d.scale.C_0 == pytest.approx(n0, rel=1e-9)
    assert abs(c.points[0].current) < 1e-6 * abs(thermionic(0.2))


@pytest.mark.parametrize("v", [0.1, 0.2, 0.3])
def test_fast_diffusion_leaves_the_thermionic_emission_current(v) -> None:
    d = schottky_diode(barrier=BARRIER)
    c = iv_sweep(d, "anode", [0.0, v], models=models(d, 1e3))
    assert c.complete, c.message
    assert c.points[-1].current == pytest.approx(thermionic(v), rel=0.02)


def diffusion_velocity(d, state, faster):
    psi = state.psi.data
    f = np.exp(psi[0] - psi) / (faster * C.D_n())
    return 1.0 / float(np.trapezoid(f, d.mesh.x))


@pytest.mark.parametrize("faster", [1e-2, 1e-1, 1.0, 1e3])
@pytest.mark.parametrize("v", [-0.2, 0.15, 0.3])
def test_the_current_follows_thermionic_emission_diffusion(v, faster) -> None:
    d = schottky_diode(barrier=BARRIER)
    m = models(d, faster)
    c = iv_sweep(d, "anode", [0.0, v], models=m)
    assert c.complete, c.message
    s = c.points[-1].state
    vD = diffusion_velocity(d, s, faster)
    vR = C.A_STAR_N * C.T_ROOM**2 / (C.q * C.Nc())
    want = thermionic(v) / (1.0 + vR / vD)
    assert c.points[-1].current == pytest.approx(want, rel=0.05)
    assert terminal_currents(d.with_bias(anode=v), s, m)["anode"] == c.points[-1].current


def test_the_small_signal_conductance_is_the_slope_of_the_dc_current() -> None:
    d = schottky_diode(barrier=BARRIER)
    m = models(d)
    v, dv = 0.15, 1e-4
    c = iv_sweep(d, "anode", [0.0, v - dv, v, v + dv], models=m)
    assert c.complete, c.message
    slope = (c.points[3].current - c.points[1].current) / (2 * dv)
    a = admittance(d.with_bias(anode=v), c.points[2].state, 0.0, m)
    assert a.G[0, 0] == pytest.approx(slope, rel=1e-4)
    assert a.G[1, 0] == pytest.approx(-slope, rel=1e-4)


def test_the_reverse_capacitance_gives_the_built_in_voltage() -> None:
    d = schottky_diode(barrier=BARRIER)
    v = [-1.0, -0.8, -0.6, -0.4, -0.2]
    c = ac_sweep(d, "anode", v, [1e6], models=models(d))
    assert c.complete, c.message
    y = [1.0 / p.admittances[0].C[0, 0] ** 2 for p in c.points]
    slope, cut = np.polyfit(v, y, 1)
    intercept = -cut / slope
    built_in = BARRIER - C.V_T() * math.log(C.Nc() / 1e16)
    assert intercept + C.V_T() == pytest.approx(built_in, rel=0.02)
    assert -2.0 / (C.q * C.eps_Si() * slope) == pytest.approx(1e16, rel=0.02)


def contact_field(d, state):
    return abs(state.psi.data[1] - state.psi.data[0]) * d.scale.psi_0 / (d.mesh.x[1] - d.mesh.x[0])


@pytest.mark.parametrize("v", [-1.0, -0.2, 0.1, 0.2])
def test_image_force_lowers_the_barrier_by_the_contact_field(v) -> None:
    d = schottky_diode(barrier=BARRIER, image_force=True)
    c = iv_sweep(d, "anode", [0.0, v], models=models(d, 1e3))
    assert c.complete, c.message
    E = contact_field(d, c.points[-1].state)
    lowered = math.sqrt(C.q * E / (4 * math.pi * C.eps_Si()))
    assert 0.005 < lowered < 0.1
    assert c.points[-1].current == pytest.approx(thermionic(v) * math.exp(lowered / C.V_T()), rel=0.02)


def test_the_reverse_current_grows_with_the_lowering() -> None:
    d = schottky_diode(barrier=BARRIER, image_force=True)
    c = iv_sweep(d, "anode", [0.0, -0.5, -2.0], models=models(d))
    assert c.complete, c.message
    assert abs(c.points[2].current) > 1.1 * abs(c.points[1].current)
    plain = iv_sweep(schottky_diode(barrier=BARRIER), "anode", [0.0, -0.5, -2.0], models=models(d))
    assert abs(plain.points[2].current) < 1.02 * abs(plain.points[1].current)


def test_newton_and_gummel_agree_with_image_force() -> None:
    d = schottky_diode(barrier=BARRIER, image_force=True)
    m = models(d)
    v = [0.0, -1.0, -0.5, 0.1, 0.2]
    a = iv_sweep(d, "anode", v, models=m)
    b = gate_sweep(d, v, contact="anode", measure_at="anode", models=m)
    assert a.complete and b.complete
    for p, q in zip(a.points[1:], b.points[1:], strict=True):
        assert q.current == pytest.approx(p.current, rel=1e-6, abs=0)


def test_the_image_force_conductance_is_the_slope_of_the_dc_current() -> None:
    d = schottky_diode(barrier=BARRIER, image_force=True)
    m = models(d)
    v, dv = -0.5, 1e-3
    c = iv_sweep(d, "anode", [0.0, v + dv, v, v - dv], models=m)
    assert c.complete, c.message
    slope = (c.points[1].current - c.points[3].current) / (2 * dv)
    a = admittance(d.with_bias(anode=v), c.points[2].state, 0.0, m)
    assert a.G[0, 0] == pytest.approx(slope, rel=1e-3)
