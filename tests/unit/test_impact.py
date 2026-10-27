from __future__ import annotations
import math
import numpy as np
import pytest
from ddsim.core import constants as C
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import TransportModels
from ddsim.discretize.assembly import SparseAssembly
from ddsim.discretize.coupled import pack
from ddsim.discretize.impact import ImpactIonization, add_impact, impact_residual
from ddsim.device.equilibrium import frozen_quasi_fermi, solve_equilibrium


@pytest.mark.parametrize("E", [2e5, 3.9e5, 4.1e5, 6e5])
def test_alpha_is_the_van_overstraeten_de_man_fit_in_physical_units(E) -> None:
    d = pn_diode()
    s = d.scale
    m = ImpactIonization.for_silicon(s)
    e = np.array([E * s.x_0 / s.psi_0])
    an, _ = m.alpha_n(e)
    ap, _ = m.alpha_p(e)
    assert an[0] / s.x_0 == pytest.approx(C.II_A_N * math.exp(-C.II_B_N / E), rel=1e-12)
    a, b = (C.II_A_P_LOW, C.II_B_P_LOW) if E < C.II_E_P_SWITCH else (C.II_A_P_HIGH, C.II_B_P_HIGH)
    assert ap[0] / s.x_0 == pytest.approx(a * math.exp(-b / E), rel=1e-12)


def test_no_field_ionizes_nothing() -> None:
    d = pn_diode()
    m = ImpactIonization.for_silicon(d.scale)
    a, da = m.alpha_n(np.array([0.0, 1e-30]))
    assert list(a) == [0.0, 0.0]
    assert list(da) == [0.0, 0.0]


def high_field(field_dependent):
    d = pn_diode(n_nodes=41)
    s = d.scale
    mesh = d.scaled_mesh
    x_cm = d.mesh.x
    rng = np.random.default_rng(3)
    psi = -3e5 * x_cm / s.psi_0 + 0.3 * rng.standard_normal(x_cm.size)
    n = np.exp(rng.uniform(-2, 8, x_cm.size))
    p = np.exp(rng.uniform(-2, 8, x_cm.size))
    m = TransportModels.for_device(d, field_dependent=field_dependent)
    return d, mesh, pack(psi, n, p), m


def assembled(d, mesh, m, x):
    bare = SparseAssembly(residual=np.zeros(x.size), rows=np.zeros(0, dtype=np.int64), cols=np.zeros(0, dtype=np.int64), values=np.zeros(0), shape=(x.size, x.size))
    return add_impact(bare, x, mesh.h, mesh.geometry, m.Dn, m.Dp, ImpactIonization.for_silicon(d.scale), d.dimension)


@pytest.mark.parametrize("field_dependent", [False, True])
def test_the_jacobian_matches_central_differences(field_dependent) -> None:
    d, mesh, x, m = high_field(field_dependent)
    a = assembled(d, mesh, m, x)
    assert np.abs(a.residual).max() > 0.0
    J = np.zeros((x.size, x.size))
    np.add.at(J, (a.rows, a.cols), a.values)
    scale = np.abs(J).max()
    for k in range(x.size):
        step = 1e-6 * max(1.0, abs(x[k]))
        up, dn = x.copy(), x.copy()
        up[k] += step
        dn[k] -= step
        col = (assembled(d, mesh, m, up).residual - assembled(d, mesh, m, dn).residual) / (2 * step)
        top = max(np.abs(col).max(), np.abs(J[:, k]).max(), 1e-300)
        assert np.abs(col - J[:, k]).max() <= 1e-5 * top + 1e-9 * scale, f"column {k}"


def test_an_equilibrium_junction_generates_nothing() -> None:
    d = pn_diode(Na=1e19, Nd=1e17, length=1.5e-4, junction=0.3e-4, n_nodes=301, h_min=1e-7)
    m = TransportModels.for_device(d, impact=True)
    eq = solve_equilibrium(d, quasi_fermi=frozen_quasi_fermi(d))
    z = d.scaled_mesh
    r = impact_residual(pack(eq.psi.data, eq.n.data, eq.p.data), z.h, z.geometry, m.Dn, m.Dp, m.impact, 1)
    flux = np.abs(m.Dn / z.h * eq.n.data[1:]).max()
    assert np.abs(r).max() < 1e-12 * flux
