from __future__ import annotations
import math
import numpy as np
import pytest
from ddsim.core import constants as C
from ddsim.device.builder import build_device
from ddsim.device.doping import Uniform
from ddsim.device.equilibrium import solve_equilibrium
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import TransportModels, solve_bias_newton, solve_bias_ramped
from ddsim.discretize.boundary import OhmicContact
from ddsim.extract.ac import noise
from ddsim.extract.iv import terminal_currents
from ddsim.mesh.mesh1d import graded_mesh_1d
from ddsim.physics.recombination import SRHRecombination

KT = C.q * C.V_T(C.T_ROOM)


@pytest.fixture(scope="module")
def bar():
    m = graded_mesh_1d(1e-4, 40, refine_at=0.3e-4, h_min=2e-7)
    d = build_device(mesh=m, doping=Uniform(1e16), contacts=(OhmicContact(name="left", node=0, voltage=0.0), OhmicContact(name="right", node=39, voltage=0.0)))
    md = TransportModels.for_device(d)
    return d, md, solve_bias_newton(d, md, guess=solve_equilibrium(d))


@pytest.fixture(scope="module")
def diode():
    d = pn_diode()
    s = d.scale
    ni = d.material.n_i / s.C_0
    tau = 1e-9
    md = TransportModels.for_device(d, recombination=SRHRecombination(tau_n=tau / s.t_0, tau_p=tau / s.t_0, ni2=ni * ni, n1=ni, p1=ni))
    return d, md, solve_bias_newton(d, md, guess=solve_equilibrium(d)), tau


@pytest.fixture(scope="module", params=[0.5, 0.776])
def forward(request):
    tau = 1e-3
    vt = C.V_T(C.T_ROOM)
    ln = math.sqrt(C.MU_N_300 * vt * tau)
    lp = math.sqrt(C.MU_P_300 * vt * tau)
    d = pn_diode(Na=1e19, Nd=1e18, length=5 * ln + 5 * lp, junction=5 * ln, n_nodes=1500, h_min=2e-7, anode_voltage=request.param)
    s = d.scale
    ni = d.material.n_i / s.C_0
    md = TransportModels.for_device(d, recombination=SRHRecombination(tau_n=tau / s.t_0, tau_p=tau / s.t_0, ni2=ni * ni, n1=ni, p1=ni))
    st = solve_bias_ramped(d, md)
    return d, md, st, tau, request.param, terminal_currents(d, st, md)["anode"]


def twiss(nz):
    return 2 * KT * (nz.Y + nz.Y.conj().T)


@pytest.mark.parametrize("f", [0.0, 1e9, 1e12])
def test_a_bar_at_equilibrium_has_nyquist_voltage_noise(bar, f):
    d, md, s = bar
    nz = noise(d, s, f, md)
    z = 1 / nz.Y[1, 1]
    assert nz.open_circuit("left")[1, 1].real == pytest.approx(4 * KT * z.real, rel=0.01, abs=0)
    np.testing.assert_allclose(nz.S_I, twiss(nz), rtol=0, atol=1e-6 * float(np.max(np.abs(twiss(nz)))))


@pytest.mark.parametrize("wt", [0.0, 1e-3, 0.1, 1.0, 10.0])
def test_a_diode_at_zero_bias_obeys_nyquist_with_recombination(diode, wt):
    d, md, s, tau = diode
    nz = noise(d, s, wt / (2 * math.pi * tau), md)
    np.testing.assert_allclose(np.diag(nz.S_I).real, 4 * KT * np.diag(nz.Y).real, rtol=1e-4, atol=0)


def test_a_forward_diode_has_van_der_ziel_shot_noise(forward):
    d, md, s, tau, v, current = forward
    nz = noise(d, s, 1e-3 / (2 * math.pi * tau), md)
    want = 2 * C.q * current / math.tanh(v / (2 * C.V_T(C.T_ROOM)))
    assert nz.S_I[0, 0].real == pytest.approx(want, rel=0.01, abs=0)
