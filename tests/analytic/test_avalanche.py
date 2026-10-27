from __future__ import annotations
import numpy as np
import pytest
from ddsim.core import constants as C
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import TransportModels
from ddsim.extract.iv import gate_sweep, iv_sweep
from ddsim.physics.recombination import SRHRecombination


def short_lived(d):
    s = d.scale
    ni = d.material.n_i / s.C_0
    return SRHRecombination(tau_n=1e-12 / s.t_0, tau_p=1e-12 / s.t_0, ni2=ni * ni, n1=ni, p1=ni)


def one_minus_integral(d, state):
    z = d.scaled_mesh
    m = TransportModels.for_device(d, impact=True).impact
    E = np.abs(np.diff(state.psi.data)) / z.h
    an = m.alpha_n(E)[0] * z.h
    ap = m.alpha_p(E)[0] * z.h
    inner = np.cumsum(an - ap) - 0.5 * (an - ap)
    return 1.0 - float(np.sum(an * np.exp(-inner)))


def crossing(v, y):
    v, y = np.asarray(v), np.asarray(y)
    k = int(np.flatnonzero(np.diff(np.sign(y)))[0])
    return float(v[k] - y[k] * (v[k + 1] - v[k]) / (y[k + 1] - y[k]))


def integral_breakdown(Nd, length, top):
    d = pn_diode(Na=1e19, Nd=Nd, length=length, junction=0.1 * length, n_nodes=401, h_min=1e-7)
    c = iv_sweep(d, "anode", [-x for x in np.linspace(0.0, top, 41)], models=TransportModels.for_device(d))
    assert c.complete, c.message
    return crossing([-p.voltage for p in c.points], [one_minus_integral(d, p.state) for p in c.points])


def test_nothing_ionizes_in_a_forward_biased_diode() -> None:
    d = pn_diode()
    v = [0.0, 0.25, 0.5]
    a = iv_sweep(d, "anode", v, models=TransportModels.for_device(d))
    b = iv_sweep(d, "anode", v, models=TransportModels.for_device(d, impact=True))
    assert b.complete, b.message
    assert abs(b.points[-1].current / a.points[-1].current - 1.0) < 1e-12


def test_breakdown_matches_the_ionization_integral() -> None:
    d = pn_diode(Na=1e19, Nd=1e17, length=1.5e-4, junction=0.3e-4, n_nodes=301, h_min=1e-7)
    R = short_lived(d)
    v = [0.0, -6.0, -10.0, -12.0, -13.0] + [-round(x, 2) for x in np.arange(13.2, 13.81, 0.1)]
    a = iv_sweep(d, "anode", v + [-13.9, -14.0, -14.1], models=TransportModels.for_device(d, recombination=R))
    assert a.complete, a.message
    b = gate_sweep(d, v, contact="anode", measure_at="anode", models=TransportModels.for_device(d, recombination=R, impact=True))
    assert b.complete, b.message
    V = [-p.voltage for p in a.points]
    by_integral = crossing(V, [one_minus_integral(d, p.state) for p in a.points])
    inverse_m = [p.current / q.current for p, q in zip(a.points[1:], b.points[1:], strict=False)]
    assert inverse_m[-1] < 0.06, f"the drift-diffusion sweep stopped at M = {1 / inverse_m[-1]:.3g}, too far from the knee to extrapolate"
    r = np.roots(np.polyfit(V[len(inverse_m) - 3:len(inverse_m) + 1], inverse_m[-4:], 2))
    r = r[np.isreal(r)].real
    by_dd = float(r[np.argmin(np.abs(r - by_integral))])
    assert by_dd == pytest.approx(by_integral, rel=0.02)


def test_breakdown_falls_with_doping() -> None:
    low = integral_breakdown(1e16, 8e-4, 90.0)
    high = integral_breakdown(1e17, 1.5e-4, 20.0)
    assert high < low
    for v, N in ((low, 1e16), (high, 1e17)):
        assert v == pytest.approx(60.0 * (C.Eg() / 1.1) ** 1.5 * (N / 1e16) ** -0.75, rel=0.30)


def test_the_sweep_goes_through_the_knee() -> None:
    d = pn_diode(Na=1e19, Nd=1e17, length=1.5e-4, junction=0.3e-4, n_nodes=301, h_min=1e-7)
    v = [0.0, -10.0, -13.0, -13.5, -13.8, -13.9, -14.0, -14.05, -14.1, -14.2, -14.5]
    b = gate_sweep(d, v, contact="anode", measure_at="anode", models=TransportModels.for_device(d, recombination=short_lived(d), impact=True))
    assert b.complete, b.message
    i = [abs(p.current) for p in b.points]
    assert all(x < y for x, y in zip(i[1:], i[2:], strict=False))
    assert i[-1] > 1e5 * i[2]
