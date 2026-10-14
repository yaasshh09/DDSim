from __future__ import annotations
import math
import numpy as np
import pytest
from ddsim.core import constants as C
from ddsim.core.field import Field, Location, ScalingState
from ddsim.device.builder import Material, build_device
from ddsim.device.doping import Uniform, abrupt_junction
from ddsim.device.equilibrium import frozen_quasi_fermi, solve_equilibrium
from ddsim.device.mos_cap import mos_cap
from ddsim.extract.cv import small_signal_capacitance, terminal_charge
from ddsim.device.transport import TransportModels, solve_bias, solve_bias_newton
from ddsim.discretize.coupled import assemble_coupled_terms, pack
from ddsim.extract.iv import terminal_currents
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


@pytest.mark.parametrize("T", [100.0, 200.0])
@pytest.mark.parametrize("solver", ["gummel", "newton"])
def test_a_frozen_out_bar_obeys_ohms_law_with_the_frozen_density(T: float, solver: str) -> None:
    Nd, V = 1e17, 1e-3
    N1 = C.Nc(T) * math.exp(-C.E_D / C.V_T(T))
    n = (N1 / 4.0) * (math.sqrt(1.0 + 8.0 * Nd / N1) - 1.0)
    d = bar(Uniform(Nd), T, n_nodes=51, incomplete_ionization=True).with_bias(left=V)
    st = solve_bias(d) if solver == "gummel" else solve_bias_newton(d)
    assert terminal_currents(d, st)["left"] == pytest.approx(C.q * C.mu_n(T) * n * V / MICRON, rel=1e-7)


def test_the_coupled_jacobian_carries_the_ionization_derivative() -> None:
    m = uniform_mesh_1d(2 * MICRON, 21)
    contacts = (OhmicContact("anode", 0, 0.0), OhmicContact("cathode", 20, 0.0))
    d = build_device(m, abrupt_junction(Na=3e17, Nd=1e17, position=MICRON), contacts, material=Material.silicon(150.0), incomplete_ionization=True)
    st = solve_equilibrium(d)
    rng = np.random.default_rng(1)
    x = pack(st.psi.data + 0.2 * rng.standard_normal(21), st.n.data * np.exp(0.3 * rng.standard_normal(21)), st.p.data * np.exp(0.3 * rng.standard_normal(21)))
    mo = TransportModels.for_device(d)
    sm = d.scaled_mesh

    def assemble(y):
        return assemble_coupled_terms(sm.h, d.charge_volume_scaled, y, d.poisson_doping_scaled.data, mo.Dn, mo.Dp, mo.recombination, sm.geometry, ionization=d.ionization, dopants=d.net_doping_scaled.data).assembly

    a = assemble(x)
    J = np.zeros(a.shape)
    np.add.at(J, (a.rows, a.cols), a.values)
    for k in range(3, x.size - 3):
        h = 1e-7 * max(abs(x[k]), 1.0)
        e = np.zeros(x.size)
        e[k] = h
        fd = (assemble(x + e).residual - assemble(x - e).residual) / (2.0 * h)
        floor = 10.0 * np.finfo(float).eps * np.max(np.abs(a.residual[0::3])) / h
        np.testing.assert_allclose(J[0::3, k], fd[0::3], rtol=1e-5, atol=1e-6 * np.max(np.abs(J[0::3, k])) + floor)


@pytest.mark.parametrize("vg", [-1.5, -0.5, 0.5])
def test_the_small_signal_capacitance_is_the_slope_of_the_frozen_out_charge(vg: float) -> None:
    def cap(v):
        return mos_cap(gate_voltage=v, T=150.0, incomplete_ionization=True)

    d = cap(vg)
    st = solve_equilibrium(d, frozen_quasi_fermi(d))
    c = small_signal_capacitance(d, st, "gate", quasi_fermi=frozen_quasi_fermi(d))
    h = 1e-4
    q = [terminal_charge(cap(v), solve_equilibrium(cap(v), frozen_quasi_fermi(cap(v))), "gate", quasi_fermi=frozen_quasi_fermi(cap(v))) for v in (vg - h, vg + h)]
    assert c == pytest.approx((q[1] - q[0]) / (2.0 * h), rel=1e-5)
