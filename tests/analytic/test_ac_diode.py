from __future__ import annotations
import math
import numpy as np
import pytest
from ddsim.core import constants as C
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import TransportModels, solve_bias_ramped
from ddsim.extract.ac import admittance
from ddsim.extract.iv import terminal_currents
from ddsim.physics.recombination import SRHRecombination

TAU = 1e-3
V_T = C.V_T(C.T_ROOM)


def uniform_lifetime(device, tau):
    s = device.scale
    ni = device.material.n_i / s.C_0
    return TransportModels.for_device(device, recombination=SRHRecombination(tau_n=tau / s.t_0, tau_p=tau / s.t_0, ni2=ni * ni, n1=ni, p1=ni))


@pytest.fixture(scope="module")
def long_diode():
    Ln = math.sqrt(C.MU_N_300 * V_T * TAU)
    Lp = math.sqrt(C.MU_P_300 * V_T * TAU)
    d = pn_diode(Na=1e19, Nd=1e18, length=5 * Ln + 5 * Lp, junction=5 * Ln, n_nodes=1500, h_min=2e-7, anode_voltage=0.776)
    m = uniform_lifetime(d, TAU)
    s = solve_bias_ramped(d, m)
    return d, m, s, terminal_currents(d, s, m)["anode"] / V_T


@pytest.fixture(scope="module")
def short_diode():
    d = pn_diode(Na=1e20, Nd=1e17, length=0.06, junction=0.03, n_nodes=1500, h_min=2e-7, anode_voltage=0.715)
    m = uniform_lifetime(d, 1.0)
    s = solve_bias_ramped(d, m)
    return d, m, s, terminal_currents(d, s, m)["anode"] / V_T


@pytest.mark.parametrize("wt", [1e-4, 1e-2])
def test_short_base_diffusion_capacitance_is_g0_w_squared_over_3d(short_diode, wt):
    d, m, s, G0 = short_diode
    W, D = 0.03, C.MU_P_300 * V_T
    a = admittance(d, s, wt / (2 * math.pi * W**2 / (2 * D)), m)
    assert a.C[0, 0] == pytest.approx(G0 * W**2 / (3 * D), rel=0.03)


@pytest.mark.parametrize("wt", np.logspace(-1.5, 1.5, 7))
def test_long_diode_admittance_is_g0_root_one_plus_i_omega_tau(long_diode, wt):
    d, m, s, G0 = long_diode
    y = admittance(d, s, wt / (2 * math.pi * TAU), m).y("anode", "anode")
    want = G0 * np.sqrt(1 + 1j * wt)
    assert abs(y) / abs(want) == pytest.approx(1.0, abs=0.02)
    assert np.angle(y) / np.angle(want) == pytest.approx(1.0, abs=0.02)
