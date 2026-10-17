from __future__ import annotations
import numpy as np
import pytest
from ddsim.device.equilibrium import frozen_quasi_fermi, solve_equilibrium
from ddsim.device.mos_cap import mos_cap
from ddsim.device.transport import TransportModels, solve_bias_newton
from ddsim.extract import cv
from ddsim.extract.ac import admittance
from ddsim.physics.recombination import SRHRecombination

HF = 1e6


def solved(vg, tau=None):
    d = mos_cap(gate_voltage=vg)
    s = d.scale
    ni = d.material.n_i / s.C_0
    m = TransportModels.for_device(d) if tau is None else TransportModels.for_device(d, recombination=SRHRecombination(tau_n=tau / s.t_0, tau_p=tau / s.t_0, ni2=ni * ni, n1=ni, p1=ni))
    q = frozen_quasi_fermi(d)
    eq = solve_equilibrium(d, quasi_fermi=q)
    return d, m, q, eq, solve_bias_newton(d, m, guess=eq)


def gate_c(d, m, st, f):
    a = admittance(d, st, f, m, width=d.mesh.x_axis.length)
    return float(a.C[a.contacts.index("gate"), a.contacts.index("gate")])


def floating_inversion(d, eq, q):
    jj = cv._bare_poisson(d, eq, q)
    u = cv._frozen_diagonal(d, eq) * (d.net_doping_scaled.data < 0)
    N = d.mesh.n_nodes
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    k = np.concatenate([jj.rows, i.ravel()])
    o = np.concatenate([jj.cols, j.ravel()])
    v = np.concatenate([jj.values, (-np.outer(u, u) / u.sum()).ravel()])
    t = cv._potential_derivative(d, "gate", k, o, v)
    tmp = np.zeros(N)
    np.add.at(tmp, k, v * t[o])
    return float(np.sum(tmp[list(cv._contact_nodes(d, "gate"))])) * cv._charge_unit(d, None)


@pytest.mark.parametrize("vg", [-2.0, -0.5, 0.0, 0.5, 2.0])
def test_far_below_the_generation_rate_c_is_quasi_static(vg):
    d, m, q, eq, st = solved(vg, tau=1e-12)
    lf = cv.small_signal_capacitance(d, eq, "gate", cv.Response.LOW_FREQUENCY, quasi_fermi=q)
    assert gate_c(d, m, st, 1e-2) == pytest.approx(lf, rel=1e-4)


@pytest.mark.parametrize("vg", [-2.0, -1.0, -0.5])
def test_without_an_inversion_layer_high_frequency_is_the_frozen_curve(vg):
    d, m, q, eq, st = solved(vg)
    hf = cv.small_signal_capacitance(d, eq, "gate", cv.Response.HIGH_FREQUENCY, quasi_fermi=q)
    assert gate_c(d, m, st, HF) == pytest.approx(hf, rel=1e-5)


@pytest.mark.parametrize("vg", [0.0, 0.5, 1.0, 2.0])
def test_at_high_frequency_the_inversion_layer_keeps_its_count_but_not_its_shape(vg):
    d, m, q, eq, st = solved(vg)
    c = gate_c(d, m, st, HF)
    hf = cv.small_signal_capacitance(d, eq, "gate", cv.Response.HIGH_FREQUENCY, quasi_fermi=q)
    assert c == pytest.approx(floating_inversion(d, eq, q), rel=1e-5)
    assert 0.0 < c / hf - 1 < 0.03
