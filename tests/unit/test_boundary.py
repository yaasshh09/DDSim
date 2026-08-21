"""Tests for discretize/boundary.py.

Ohmic contacts impose charge neutrality plus thermal equilibrium at the contact
node, which fixes psi there. docs/01-physics.md gives

    psi_contact = V_applied + V_T * asinh(N / (2 * n_i))

and is emphatic about asinh rather than the naive V_T*ln(N/n_i), because the
log form breaks wherever N is near zero or negative. In scaled units with
C_0 = n_i that is psi = V_applied_scaled + asinh(N_scaled / 2).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.assembly import SparseAssembly
from ddsim.discretize.boundary import (
    Carrier,
    OhmicContact,
    apply_dirichlet,
    apply_ohmic_contacts,
    apply_ohmic_densities,
    ohmic_psi_scaled,
)
from ddsim.discretize.poisson import poisson_jacobian, poisson_residual
from ddsim.mesh.mesh1d import uniform_mesh_1d
from ddsim.physics.statistics import psi_equilibrium_scaled

MICRON = 1e-4
"""One micron [cm]."""


def sample_assembly(n_nodes: int = 11, doping: float = 1e6) -> tuple:
    """A small assembled Poisson system plus the psi it was built at."""
    scale = ScaleFactors.for_silicon()
    mesh = uniform_mesh_1d(MICRON, n_nodes)
    h = mesh.h / scale.x_0
    volume = mesh.volume / scale.x_0

    psi = np.full(n_nodes, float(psi_equilibrium_scaled(doping)))
    net_doping = np.full(n_nodes, doping)

    rows, cols, values = poisson_jacobian(h, volume, psi, net_doping)
    assembly = SparseAssembly(
        residual=poisson_residual(h, volume, psi, net_doping),
        rows=rows,
        cols=cols,
        values=values,
        shape=(n_nodes, n_nodes),
    )
    return assembly, psi


def dense(assembly: SparseAssembly) -> np.ndarray:
    """The Jacobian as a dense array, for inspection."""
    return sp.coo_matrix(
        (assembly.values, (assembly.rows, assembly.cols)), shape=assembly.shape
    ).toarray()


# ------------------------------------------------------------ contact potential


def test_contact_potential_at_zero_bias_is_the_equilibrium_potential() -> None:
    assert ohmic_psi_scaled(1e6, 0.0) == pytest.approx(
        float(psi_equilibrium_scaled(1e6)), rel=1e-15
    )


def test_contact_potential_adds_the_applied_bias() -> None:
    """The bias enters as a rigid shift of psi, nothing more."""
    equilibrium = ohmic_psi_scaled(1e6, 0.0)
    biased = ohmic_psi_scaled(1e6, 5.0)
    assert biased - equilibrium == pytest.approx(5.0, rel=1e-14)


def test_contact_potential_uses_asinh_not_log() -> None:
    net = 1e6
    assert ohmic_psi_scaled(net, 0.0) == pytest.approx(
        math.asinh(net / 2.0), rel=1e-15
    )


def test_contact_potential_is_finite_in_compensated_material() -> None:
    """The bug source called out in 01-physics: the log form dies here."""
    for net in (-1e-8, 0.0, 1e-8, -1e6):
        assert math.isfinite(ohmic_psi_scaled(net, 0.0))


def test_contact_potential_is_negative_for_p_type() -> None:
    assert ohmic_psi_scaled(-1e6, 0.0) < 0.0


# ---------------------------------------------------------------- Dirichlet rows


def test_dirichlet_residual_is_the_potential_error() -> None:
    assembly, psi = sample_assembly()
    constrained = apply_dirichlet(assembly, psi, node=0, target=2.5)
    assert constrained.residual[0] == pytest.approx(psi[0] - 2.5, rel=1e-15)


def test_dirichlet_residual_is_zero_when_psi_already_matches() -> None:
    assembly, psi = sample_assembly()
    constrained = apply_dirichlet(assembly, psi, node=0, target=float(psi[0]))
    assert constrained.residual[0] == 0.0


def test_dirichlet_row_becomes_the_identity() -> None:
    assembly, psi = sample_assembly()
    matrix = dense(apply_dirichlet(assembly, psi, node=3, target=0.0))
    expected = np.zeros(assembly.shape[1])
    expected[3] = 1.0
    np.testing.assert_allclose(matrix[3, :], expected, atol=0.0)


def test_dirichlet_leaves_other_rows_untouched() -> None:
    assembly, psi = sample_assembly()
    before = dense(assembly)
    after = dense(apply_dirichlet(assembly, psi, node=3, target=0.0))
    others = [i for i in range(assembly.shape[0]) if i != 3]
    np.testing.assert_allclose(after[others, :], before[others, :], rtol=1e-15)


def test_dirichlet_leaves_other_residuals_untouched() -> None:
    assembly, psi = sample_assembly()
    constrained = apply_dirichlet(assembly, psi, node=3, target=0.0)
    others = [i for i in range(assembly.shape[0]) if i != 3]
    np.testing.assert_allclose(
        constrained.residual[others], assembly.residual[others], rtol=1e-15
    )


def test_dirichlet_does_not_mutate_the_original_assembly() -> None:
    assembly, psi = sample_assembly()
    before = assembly.residual.copy()
    apply_dirichlet(assembly, psi, node=0, target=9.0)
    np.testing.assert_array_equal(assembly.residual, before)


def test_dirichlet_rejects_a_node_outside_the_mesh() -> None:
    assembly, psi = sample_assembly(n_nodes=11)
    with pytest.raises(IndexError, match="node"):
        apply_dirichlet(assembly, psi, node=11, target=0.0)


def test_solving_a_dirichlet_row_reproduces_the_target_exactly() -> None:
    """The point of the whole exercise: one Newton step lands psi on target."""
    assembly, psi = sample_assembly()
    target = 3.0
    constrained = apply_dirichlet(assembly, psi, node=0, target=target)

    matrix = dense(constrained)
    delta = np.linalg.solve(matrix, -constrained.residual)
    assert psi[0] + delta[0] == pytest.approx(target, rel=1e-12)


# ------------------------------------------------------------------- contacts


def test_two_contacts_pin_both_ends() -> None:
    scale = ScaleFactors.for_silicon()
    assembly, psi = sample_assembly(n_nodes=11)
    net_doping = np.full(11, 1e6)

    contacts = (
        OhmicContact(name="anode", node=0, voltage=0.0),
        OhmicContact(name="cathode", node=10, voltage=0.0),
    )
    constrained = apply_ohmic_contacts(assembly, psi, net_doping, contacts, scale)

    matrix = dense(constrained)
    for node in (0, 10):
        expected = np.zeros(11)
        expected[node] = 1.0
        np.testing.assert_allclose(matrix[node, :], expected, atol=0.0)


def test_contact_voltage_is_converted_from_volts_to_scaled_units() -> None:
    """A contact is specified in volts. The solver works in units of V_T."""
    scale = ScaleFactors.for_silicon()
    assembly, psi = sample_assembly(n_nodes=11)
    net_doping = np.full(11, 1e6)

    contacts = (OhmicContact(name="anode", node=0, voltage=1.0),)
    constrained = apply_ohmic_contacts(assembly, psi, net_doping, contacts, scale)

    target = 1.0 / scale.psi_0 + float(psi_equilibrium_scaled(1e6))
    assert constrained.residual[0] == pytest.approx(psi[0] - target, rel=1e-12)


def test_contact_reads_the_doping_at_its_own_node() -> None:
    """A contact on the n side must not use the p side doping."""
    scale = ScaleFactors.for_silicon()
    assembly, psi = sample_assembly(n_nodes=11)
    net_doping = np.concatenate((np.full(5, -1e6), np.full(6, 1e6)))

    contacts = (
        OhmicContact(name="anode", node=0, voltage=0.0),
        OhmicContact(name="cathode", node=10, voltage=0.0),
    )
    constrained = apply_ohmic_contacts(assembly, psi, net_doping, contacts, scale)

    anode_target = psi[0] - constrained.residual[0]
    cathode_target = psi[10] - constrained.residual[10]
    assert anode_target < 0.0, "p side contact must sit at negative psi"
    assert cathode_target > 0.0, "n side contact must sit at positive psi"


def test_contact_names_must_be_unique() -> None:
    scale = ScaleFactors.for_silicon()
    assembly, psi = sample_assembly(n_nodes=11)
    contacts = (
        OhmicContact(name="anode", node=0, voltage=0.0),
        OhmicContact(name="anode", node=10, voltage=0.0),
    )
    with pytest.raises(ValueError, match="unique|duplicate"):
        apply_ohmic_contacts(assembly, psi, np.full(11, 1e6), contacts, scale)


def test_built_in_potential_is_the_difference_between_the_two_contacts() -> None:
    """The headline analytic result, before any solve.

    For a 1e16/1e16 junction, V_bi = V_T*ln(Na*Nd/n_i^2). The two ohmic
    contact potentials differ by exactly that, which is a useful check that
    the asinh form is right before trusting a full solve.
    """
    scale = ScaleFactors.for_silicon()
    doping = 1e16 / scale.C_0

    anode = ohmic_psi_scaled(-doping, 0.0)
    cathode = ohmic_psi_scaled(doping, 0.0)
    V_bi = (cathode - anode) * scale.psi_0  # [V]

    expected = scale.psi_0 * math.log(1e16 * 1e16 / scale.C_0**2)  # [V]
    assert V_bi == pytest.approx(expected, rel=1e-6)


# ------------------------------------------------- ohmic contacts on n and p


def continuity_assembly(n_nodes: int = 5) -> SparseAssembly:
    """A stand in continuity system: identity Jacobian, arbitrary residual."""
    index = np.arange(n_nodes, dtype=np.int64)
    return SparseAssembly(
        residual=np.full(n_nodes, 3.0),
        rows=index,
        cols=index,
        values=np.full(n_nodes, 2.0),
        shape=(n_nodes, n_nodes),
    )


def test_ohmic_densities_pin_the_majority_carrier_to_the_doping() -> None:
    """In n-type material the contact holds n at N, to twelve digits.

    Neutrality plus mass action give n - p = N and n*p = 1, so with N = 1e6 the
    majority carrier is 1e6 to within the minority correction of 1e-6.
    """
    doping = np.array([1e6, 1e6, 0.0, -1e6, -1e6])
    density = np.zeros(5)
    contacts = (OhmicContact("cathode", 0, 0.0),)

    pinned = apply_ohmic_densities(
        continuity_assembly(), density, doping, contacts, Carrier.ELECTRON
    )

    # residual = density - target, and density is zero here.
    np.testing.assert_allclose(-pinned.residual[0], 1e6, rtol=1e-6)


def test_ohmic_densities_pin_the_minority_carrier_by_mass_action() -> None:
    """The minority carrier is 1/N, not N, and getting it right matters.

    It is the minority density at the contact that sets the saturation current
    of a diode, so an error here is an error in every I-V curve.
    """
    doping = np.array([1e6, 1e6, 0.0, -1e6, -1e6])
    density = np.zeros(5)
    contacts = (OhmicContact("cathode", 0, 0.0),)

    pinned = apply_ohmic_densities(
        continuity_assembly(), density, doping, contacts, Carrier.HOLE
    )

    np.testing.assert_allclose(-pinned.residual[0], 1e-6, rtol=1e-6)


def test_the_two_pinned_densities_satisfy_mass_action_exactly() -> None:
    doping = np.array([-1e6, 0.0, 1e6])
    density = np.zeros(3)
    contacts = (OhmicContact("anode", 0, 0.0),)

    n = -apply_ohmic_densities(
        continuity_assembly(3), density, doping, contacts, Carrier.ELECTRON
    ).residual[0]
    p = -apply_ohmic_densities(
        continuity_assembly(3), density, doping, contacts, Carrier.HOLE
    ).residual[0]

    np.testing.assert_allclose(n * p, 1.0, rtol=1e-15)
    np.testing.assert_allclose(p - n, 1e6, rtol=1e-12)


def test_the_pinned_density_agrees_with_the_pinned_potential() -> None:
    """n = exp(psi - phi_n) at the contact, with phi_n the applied bias.

    The two boundary conditions are written independently, one from asinh and
    one from the quadratic, so their agreement is a real check rather than a
    tautology. If they disagreed, the first Gummel cycle would fight itself at
    the contact node forever.
    """
    doping_value = -1e6
    applied = 0.4 / ScaleFactors.for_silicon().psi_0

    psi = ohmic_psi_scaled(doping_value, applied)
    doping = np.array([doping_value, doping_value])
    contacts = (OhmicContact("anode", 0, 0.0),)
    n = -apply_ohmic_densities(
        continuity_assembly(2), np.zeros(2), doping, contacts, Carrier.ELECTRON
    ).residual[0]

    np.testing.assert_allclose(n, math.exp(psi - applied), rtol=1e-12)


def test_ohmic_densities_apply_at_every_contact() -> None:
    doping = np.array([-1e6, 0.0, 1e6])
    contacts = (OhmicContact("anode", 0, 0.0), OhmicContact("cathode", 2, 0.0))

    pinned = apply_ohmic_densities(
        continuity_assembly(3), np.zeros(3), doping, contacts, Carrier.ELECTRON
    )

    np.testing.assert_allclose(-pinned.residual[0], 1e-6, rtol=1e-6)
    np.testing.assert_allclose(-pinned.residual[2], 1e6, rtol=1e-6)
    assert pinned.residual[1] == 3.0
