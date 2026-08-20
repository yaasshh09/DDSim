"""Tests for discretize/poisson.py.

Scaled units throughout: psi in units of V_T, densities in units of n_i,
lengths in units of the Debye length x_0. In those units Poisson is

    lap(psi) = -(p - n + N)

with no coefficient at all, which is the entire reason for the scaling.

The residual is defined with the sign that makes the Jacobian diagonal
positive and the off-diagonals negative, so the matrix is an M-matrix:

    F_i = (psi_i - psi_{i-1})/h_{i-1} - (psi_{i+1} - psi_i)/h_i
          - (p_i - n_i + N_i) * volume_i

The strongest test here is that uniform material at psi = asinh(N/2) gives an
exactly zero residual. The Laplacian vanishes because psi is flat, and the
charge term vanishes because that psi is the neutrality solution. Any sign
error in either half breaks it.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from ddsim.core.field import Field, Location, ScalingState
from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.poisson import (
    assemble_poisson,
    poisson_jacobian,
    poisson_residual,
)
from ddsim.mesh.mesh1d import uniform_mesh_1d
from ddsim.physics.statistics import psi_equilibrium_scaled

MICRON = 1e-4
"""One micron [cm]."""


def scaled_mesh(n_nodes: int = 21, length: float = MICRON) -> tuple:
    """A uniform mesh with its edge lengths and volumes in scaled units."""
    scale = ScaleFactors.for_silicon()
    mesh = uniform_mesh_1d(length, n_nodes)
    return mesh.h / scale.x_0, mesh.volume / scale.x_0


# ------------------------------------------------------- the equilibrium limit


@pytest.mark.parametrize("doping_scaled", [-1e6, -1.0, 0.0, 1.0, 1e6, 1e8])
def test_residual_is_zero_for_uniform_material_at_equilibrium(
    doping_scaled: float,
) -> None:
    """The single strongest check on the signs in this module.

    Flat psi kills the Laplacian, and psi = asinh(N/2) kills the charge term.
    If either sign is wrong the two no longer cancel.
    """
    h, volume = scaled_mesh()
    net_doping = np.full(h.size + 1, doping_scaled)
    psi = np.full(h.size + 1, float(psi_equilibrium_scaled(doping_scaled)))

    residual = poisson_residual(h, volume, psi, net_doping)
    np.testing.assert_allclose(residual, 0.0, atol=1e-9 * max(1.0, abs(doping_scaled)))


def test_residual_is_nonzero_when_psi_is_off_equilibrium() -> None:
    """Guards the test above against passing for a trivial reason."""
    h, volume = scaled_mesh()
    net_doping = np.full(h.size + 1, 1e6)
    psi = np.full(h.size + 1, float(psi_equilibrium_scaled(1e6)) + 0.1)

    assert np.max(np.abs(poisson_residual(h, volume, psi, net_doping))) > 1.0


# ----------------------------------------------------------------- the operator


def test_laplacian_of_a_linear_potential_vanishes_in_the_interior() -> None:
    """A linear psi has zero second derivative, so only the charge term is left.

    The charge term has to be evaluated at the same psi, not at zero. n and p
    are exponentials of psi, so a linear psi still carries charge.
    """
    h, volume = scaled_mesh(n_nodes=21)
    psi = np.concatenate(([0.0], np.cumsum(h))) * 3.0
    net_doping = np.zeros(psi.size)

    residual = poisson_residual(h, volume, psi, net_doping)
    charge_only = -(np.exp(-psi) - np.exp(psi) + net_doping) * volume

    # Interior nodes see only the charge term. Boundary nodes do not, because
    # a reflecting boundary is not satisfied by a linear profile.
    np.testing.assert_allclose(residual[1:-1], charge_only[1:-1], atol=1e-14)


def test_boundary_rows_use_a_reflecting_condition() -> None:
    """Nodes that are not contacts get homogeneous Neumann, per 01-physics.

    Node 0 has no left face, so its row carries only the right flux.
    """
    h, volume = scaled_mesh(n_nodes=5)
    psi = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    net_doping = np.zeros(5)

    residual = poisson_residual(h, volume, psi, net_doping)
    n0 = np.exp(1.0)
    p0 = np.exp(-1.0)
    expected = -(0.0 - 1.0) / h[0] - (p0 - n0) * volume[0]
    assert residual[0] == pytest.approx(expected, rel=1e-14)


def test_charge_term_has_the_sign_that_pulls_psi_toward_neutrality() -> None:
    """Net donors must push psi positive, per the sign convention."""
    h, volume = scaled_mesh(n_nodes=11)
    psi = np.zeros(11)

    donors = poisson_residual(h, volume, psi, np.full(11, 1e6))
    acceptors = poisson_residual(h, volume, psi, np.full(11, -1e6))

    assert np.all(donors[1:-1] < 0.0)
    assert np.all(acceptors[1:-1] > 0.0)


# ------------------------------------------------------------------- Jacobian


def test_jacobian_matches_complex_step_differentiation() -> None:
    """Exact to machine precision, unlike finite differences.

    The residual is built from sums, quotients and exp, all analytic, so the
    complex step trick applies cleanly here. It does not suffer the
    cancellation that limits it for the Bernoulli function.
    """
    h, volume = scaled_mesh(n_nodes=20)
    rng = np.random.default_rng(0)
    psi = rng.uniform(-8.0, 8.0, 20)
    net_doping = rng.uniform(-1e5, 1e5, 20)

    rows, cols, values = poisson_jacobian(h, volume, psi, net_doping)
    assembled = sp.coo_matrix((values, (rows, cols)), shape=(20, 20)).toarray()

    step = 1e-30
    for column in range(20):
        perturbed = psi.astype(np.complex128)
        perturbed[column] += 1j * step
        derivative = poisson_residual(h, volume, perturbed, net_doping).imag / step
        np.testing.assert_allclose(
            assembled[:, column], derivative, rtol=1e-12, atol=1e-12
        )


def test_jacobian_matches_finite_difference_on_a_20_node_mesh() -> None:
    """The check named in phases/PHASE-1.md, kept alongside the exact one."""
    h, volume = scaled_mesh(n_nodes=20)
    rng = np.random.default_rng(1)
    psi = rng.uniform(-5.0, 5.0, 20)
    net_doping = rng.uniform(-1e4, 1e4, 20)

    rows, cols, values = poisson_jacobian(h, volume, psi, net_doping)
    assembled = sp.coo_matrix((values, (rows, cols)), shape=(20, 20)).toarray()

    delta = 1e-6
    for column in range(20):
        forward = psi.copy()
        backward = psi.copy()
        forward[column] += delta
        backward[column] -= delta
        derivative = (
            poisson_residual(h, volume, forward, net_doping)
            - poisson_residual(h, volume, backward, net_doping)
        ) / (2.0 * delta)
        np.testing.assert_allclose(
            assembled[:, column], derivative, rtol=1e-5, atol=1e-5
        )


def test_jacobian_is_tridiagonal() -> None:
    h, volume = scaled_mesh(n_nodes=15)
    rows, cols, _ = poisson_jacobian(h, volume, np.zeros(15), np.zeros(15))
    assert np.all(np.abs(rows - cols) <= 1)


def test_jacobian_diagonal_is_strictly_positive() -> None:
    """docs/02-numerics.md: this is what makes equilibrium Poisson easy."""
    h, volume = scaled_mesh(n_nodes=15)
    psi = np.linspace(-10.0, 10.0, 15)
    rows, cols, values = poisson_jacobian(h, volume, psi, np.zeros(15))
    diagonal = values[rows == cols]
    assert np.all(diagonal > 0.0)


def test_jacobian_off_diagonals_are_negative() -> None:
    """Together with a positive diagonal this makes it an M-matrix."""
    h, volume = scaled_mesh(n_nodes=15)
    rows, cols, values = poisson_jacobian(h, volume, np.zeros(15), np.zeros(15))
    assert np.all(values[rows != cols] < 0.0)


def test_jacobian_is_symmetric() -> None:
    """Box integration of the Laplacian is symmetric, and the charge term is
    diagonal, so the whole matrix is."""
    h, volume = scaled_mesh(n_nodes=15)
    psi = np.linspace(-3.0, 3.0, 15)
    rows, cols, values = poisson_jacobian(h, volume, psi, np.zeros(15))
    matrix = sp.coo_matrix((values, (rows, cols)), shape=(15, 15)).toarray()
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-14)


def test_jacobian_is_diagonally_dominant() -> None:
    h, volume = scaled_mesh(n_nodes=15)
    rows, cols, values = poisson_jacobian(h, volume, np.zeros(15), np.zeros(15))
    matrix = sp.coo_matrix((values, (rows, cols)), shape=(15, 15)).toarray()
    diagonal = np.abs(np.diag(matrix))
    off_diagonal = np.abs(matrix).sum(axis=1) - diagonal
    assert np.all(diagonal >= off_diagonal)


def test_jacobian_laplacian_block_matches_the_uniform_mesh_stencil() -> None:
    """On a uniform mesh the interior stencil is exactly (-1, 2, -1)/h."""
    h, volume = scaled_mesh(n_nodes=11)
    spacing = h[0]
    rows, cols, values = poisson_jacobian(h, volume, np.zeros(11), np.zeros(11))
    matrix = sp.coo_matrix((values, (rows, cols)), shape=(11, 11)).toarray()

    charge_diagonal = 2.0 * volume  # (n + p) * volume at psi = 0, n = p = 1
    for i in range(1, 10):
        assert matrix[i, i - 1] == pytest.approx(-1.0 / spacing, rel=1e-14)
        assert matrix[i, i + 1] == pytest.approx(-1.0 / spacing, rel=1e-14)
        assert matrix[i, i] == pytest.approx(
            2.0 / spacing + charge_diagonal[i], rel=1e-14
        )


# ------------------------------------------------------------ the Field wrapper


def test_assemble_checks_scaling_state_at_entry() -> None:
    """A physical psi here would be wrong by a factor of 38.7 and still solve."""
    mesh = uniform_mesh_1d(MICRON, 11)
    scale = ScaleFactors.for_silicon()
    psi = Field(np.zeros(11), "V", ScalingState.PHYSICAL, Location.NODE)
    doping = Field(np.zeros(11), "cm^-3", ScalingState.SCALED, Location.NODE)

    with pytest.raises(ValueError, match="SCALED"):
        assemble_poisson(mesh, psi, doping, scale)


def test_assemble_rejects_edge_located_fields() -> None:
    mesh = uniform_mesh_1d(MICRON, 11)
    scale = ScaleFactors.for_silicon()
    psi = Field(np.zeros(10), "V", ScalingState.SCALED, Location.EDGE)
    doping = Field(np.zeros(11), "cm^-3", ScalingState.SCALED, Location.NODE)

    with pytest.raises(ValueError, match="NODE"):
        assemble_poisson(mesh, psi, doping, scale)


def test_assemble_rejects_a_field_of_the_wrong_length() -> None:
    mesh = uniform_mesh_1d(MICRON, 11)
    scale = ScaleFactors.for_silicon()
    psi = Field(np.zeros(9), "V", ScalingState.SCALED, Location.NODE)
    doping = Field(np.zeros(11), "cm^-3", ScalingState.SCALED, Location.NODE)

    with pytest.raises(ValueError, match="length|nodes"):
        assemble_poisson(mesh, psi, doping, scale)


def test_assemble_scales_the_mesh_by_the_debye_length() -> None:
    """The mesh is in cm but the equation is in units of x_0.

    Forgetting this is a silent error of many orders of magnitude, since
    x_0 is 40.9 um at intrinsic doping and a device is 1 um across.
    """
    mesh = uniform_mesh_1d(MICRON, 11)
    scale = ScaleFactors.for_silicon()
    psi = Field(np.zeros(11), "V", ScalingState.SCALED, Location.NODE)
    doping = Field(np.zeros(11), "cm^-3", ScalingState.SCALED, Location.NODE)

    assembly = assemble_poisson(mesh, psi, doping, scale)
    expected = poisson_residual(
        mesh.h / scale.x_0, mesh.volume / scale.x_0, np.zeros(11), np.zeros(11)
    )
    np.testing.assert_allclose(assembly.residual, expected, rtol=1e-14)


def test_assemble_returns_a_square_system_of_the_right_size() -> None:
    mesh = uniform_mesh_1d(MICRON, 11)
    scale = ScaleFactors.for_silicon()
    psi = Field(np.zeros(11), "V", ScalingState.SCALED, Location.NODE)
    doping = Field(np.zeros(11), "cm^-3", ScalingState.SCALED, Location.NODE)

    assembly = assemble_poisson(mesh, psi, doping, scale)
    assert assembly.shape == (11, 11)
    assert assembly.residual.shape == (11,)
