from __future__ import annotations
import math
import numpy as np
import pytest
from ddsim.core import constants as C
from ddsim.core.field import Field, Location, ScalingState
from ddsim.device.builder import Material, build_device
from ddsim.device.doping import Uniform, abrupt_junction
from ddsim.device.equilibrium import solve_equilibrium
from ddsim.device.transport import solve_bias_newton
from ddsim.discretize.boundary import OhmicContact
from ddsim.discretize.poisson import assemble_poisson
from ddsim.mesh.mesh1d import uniform_mesh_1d

MICRON = 1e-4
TEMPERATURES = [100.0, 200.0, 300.0]


def bar(doping, T, n_nodes=21, **kw):
    contacts = (OhmicContact("left", 0, 0.0), OhmicContact("right", n_nodes - 1, 0.0))
    return build_device(uniform_mesh_1d(MICRON, n_nodes), doping, contacts, material=Material.silicon(T), **kw)


def frozen(N, N1, g):
    return (N1 / (2.0 * g)) * (math.sqrt(1.0 + 4.0 * g * N / N1) - 1.0)


@pytest.mark.parametrize("T", TEMPERATURES)
def test_donor_bar_freezes_out_to_the_closed_form(T: float) -> None:
    Nd = 1e17
    N1 = C.Nc(T) * math.exp(-C.E_D / C.V_T(T))
    d = bar(Uniform(Nd), T, incomplete_ionization=True)
    n = solve_equilibrium(d).n.to_physical(d.scale).data
    np.testing.assert_allclose(n, (N1 / 4.0) * (math.sqrt(1.0 + 8.0 * Nd / N1) - 1.0), rtol=1e-10)


@pytest.mark.parametrize("T", TEMPERATURES)
def test_acceptor_bar_freezes_out_with_degeneracy_four(T: float) -> None:
    Na = 1e17
    N1 = C.Nv(T) * math.exp(-C.E_A / C.V_T(T))
    d = bar(Uniform(-Na), T, incomplete_ionization=True)
    p = solve_equilibrium(d).p.to_physical(d.scale).data
    np.testing.assert_allclose(p, frozen(Na, N1, 4.0), rtol=1e-10)


def test_freeze_out_is_not_a_no_op_at_100_k() -> None:
    d = bar(Uniform(1e17), 100.0, incomplete_ionization=True)
    n = solve_equilibrium(d).n.to_physical(d.scale).data
    assert np.max(n) < 0.5e17


def test_off_by_default_is_full_ionization() -> None:
    d = bar(Uniform(1e17), 100.0)
    n = solve_equilibrium(d).n.to_physical(d.scale).data
    np.testing.assert_allclose(n, 1e17, rtol=1e-10)


def test_the_poisson_jacobian_carries_the_ionization_derivative() -> None:
    m = uniform_mesh_1d(2 * MICRON, 41)
    contacts = (OhmicContact("anode", 0, 0.0), OhmicContact("cathode", 40, 0.0))
    d = build_device(m, abrupt_junction(Na=3e17, Nd=1e17, position=MICRON), contacts, material=Material.silicon(150.0), incomplete_ionization=True)
    psi = solve_equilibrium(d).psi.data + 0.3 * np.sin(np.linspace(0.0, 3.0, 41))

    def assemble(x):
        f = Field(x, "V", ScalingState.SCALED, Location.NODE, name="psi")
        return assemble_poisson(d.scaled_mesh, f, d.net_doping_scaled, ionization=d.ionization)

    a = assemble(psi)
    J = np.zeros(a.shape)
    np.add.at(J, (a.rows, a.cols), a.values)
    for k in range(1, 40):
        e = np.zeros(41)
        e[k] = 1e-6
        fd = (assemble(psi + e).residual - assemble(psi - e).residual) / 2e-6
        np.testing.assert_allclose(J[:, k], fd, rtol=1e-6, atol=1e-6 * np.max(np.abs(J[:, k])))


def test_ionization_needs_boltzmann_statistics() -> None:
    with pytest.raises(ValueError, match="Boltzmann"):
        bar(Uniform(1e17), 300.0, incomplete_ionization=True, degenerate=True)


def test_the_coupled_solve_refuses_rather_than_ignores_ionization() -> None:
    with pytest.raises(NotImplementedError, match="ionization"):
        solve_bias_newton(bar(Uniform(1e17), 300.0, incomplete_ionization=True))
