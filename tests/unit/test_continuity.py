"""Tests for discretize/continuity.py, the Scharfetter-Gummel assembly.

Everything is scaled: psi in units of V_T, densities in units of C_0, lengths
in units of x_0, diffusivities in units of D_0. In those units the edge fluxes
carry no constants at all, from docs/02-numerics.md:

    Jn = (Dn/h) * (B(X)*n_right - B(-X)*n_left)
    Jp = (Dp/h) * (B(X)*p_left  - B(-X)*p_right)

with X = psi_right - psi_left.

The asymmetry between those two lines is the single most dangerous detail in
the project. docs/05-pitfalls.md: reversing it produces a solver that converges
cleanly to a physically wrong answer with reversed current. It is tested here
three separate ways: the low field limit, the drift limit, and which node
dominates at high field.

The strongest test in the file is the conservation one. With recombination off,
solving the electron equation and then measuring Jn on every edge must give the
same number everywhere, to machine precision rather than to a tolerance. That
property is what Scharfetter-Gummel is for.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.core.field import Field, Location, ScalingState
from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.assembly import SparseAssembly
from ddsim.discretize.boundary import apply_dirichlet
from ddsim.discretize.continuity import (
    assemble_electron_continuity,
    assemble_hole_continuity,
    electron_continuity_jacobian,
    electron_continuity_residual,
    electron_current,
    hole_continuity_jacobian,
    hole_continuity_residual,
    hole_current,
)
from ddsim.mesh.mesh1d import Mesh1D, graded_mesh_1d, uniform_mesh_1d
from ddsim.physics.bernoulli import B
from ddsim.physics.recombination import NoRecombination, SRHRecombination
from ddsim.solve.linear import SparseLU

MICRON = 1e-4
"""One micron [cm]."""

D_N = 1.0
"""Scaled electron diffusivity [1]. D_0 is max(Dn, Dp), which is Dn."""

D_P = 0.3317
"""Scaled hole diffusivity [1], 470/1417 by the Einstein relation."""

V_BI = 27.6
"""A built-in potential [1], about 0.714 V in units of V_T."""


def scaled_mesh(
    n_nodes: int = 41, length: float = MICRON, graded: bool = False
) -> tuple[Mesh1D, np.ndarray, np.ndarray]:
    """A mesh with its edge lengths and dual volumes in units of x_0."""
    scale = ScaleFactors.for_silicon()
    if graded:
        mesh = graded_mesh_1d(length, n_nodes, refine_at=0.5 * length, h_min=2e-7)
    else:
        mesh = uniform_mesh_1d(length, n_nodes)
    return mesh, mesh.h / scale.x_0, mesh.volume / scale.x_0


def junction_potential(mesh: Mesh1D) -> np.ndarray:
    """A junction shaped psi [1], p-type on the left, n-type on the right.

    Not a solution of anything. It only has to be a realistic profile with
    strong fields in the middle, so that the edge arguments X span the regime
    where exponential fitting matters.
    """
    middle = 0.5 * mesh.length
    width = 0.05 * mesh.length
    return 0.5 * V_BI * np.tanh((mesh.x - middle) / width)


def solve_block(
    residual: np.ndarray,
    triplets: tuple[np.ndarray, np.ndarray, np.ndarray],
    density: np.ndarray,
    targets: tuple[float, float],
) -> np.ndarray:
    """One block solve: pin both ends, take the Newton step, return the density.

    The continuity equation is linear in its own carrier once the fluxes and
    the recombination linearization are fixed, so this single step is the exact
    solution rather than an iteration towards it.
    """
    rows, cols, values = triplets
    n_nodes = density.size
    assembly = SparseAssembly(residual, rows, cols, values, (n_nodes, n_nodes))
    assembly = apply_dirichlet(assembly, density, 0, targets[0])
    assembly = apply_dirichlet(assembly, density, n_nodes - 1, targets[1])

    solver = SparseLU()
    solver.factorize(assembly.rows, assembly.cols, assembly.values, assembly.shape)
    return density + solver.solve(-assembly.residual)


def as_field(values: np.ndarray, unit: str, name: str) -> Field:
    """A scaled node Field, the form the assemble functions demand."""
    return Field(values, unit, ScalingState.SCALED, Location.NODE, name=name)


# ------------------------------------------------------------- the edge flux


def test_zero_field_reduces_to_plain_diffusion() -> None:
    """B(0) = 1 on both sides, so the flux is Dn * dn/dx and nothing else.

    This is the limit in which Scharfetter-Gummel must agree with central
    differencing. If it does not, the scheme is not consistent.
    """
    h = np.array([0.5, 0.25])
    psi = np.zeros(3)
    n = np.array([1.0, 3.0, 4.0])

    expected = D_N * np.array([(3.0 - 1.0) / 0.5, (4.0 - 3.0) / 0.25])
    np.testing.assert_allclose(electron_current(h, D_N, psi, n), expected, rtol=1e-15)


def test_zero_field_hole_flux_is_minus_the_gradient() -> None:
    """Jp = -Dp * dp/dx at zero field. The sign flip against Jn is physical:
    holes diffuse down the gradient carrying positive charge with them."""
    h = np.array([0.5, 0.25])
    psi = np.zeros(3)
    p = np.array([1.0, 3.0, 4.0])

    expected = -D_P * np.array([(3.0 - 1.0) / 0.5, (4.0 - 3.0) / 0.25])
    np.testing.assert_allclose(hole_current(h, D_P, psi, p), expected, rtol=1e-14)


def test_uniform_density_gives_pure_drift() -> None:
    """With n flat, B(X) - B(-X) = -X exactly, so Jn = -Dn * n * dpsi/dx.

    That is q*mu_n*n*E with E = -dpsi/dx and D = mu*V_T, in scaled units. The
    identity B(-x) = B(x) + x is what makes it exact rather than approximate,
    which is why docs/02-numerics.md calls it the strongest Bernoulli test.
    """
    h = np.array([0.4, 0.4])
    psi = np.array([0.0, 1.3, 2.9])
    n = np.full(3, 7.0)

    gradient = np.diff(psi) / h
    np.testing.assert_allclose(
        electron_current(h, D_N, psi, n), -D_N * 7.0 * gradient, rtol=1e-14
    )


def test_uniform_hole_density_gives_pure_drift_with_the_same_sign() -> None:
    """Jp = -Dp * p * dpsi/dx too.

    Both carriers drift in the same direction as conventional current under a
    field, which is the whole reason a semiconductor conducts with either.
    """
    h = np.array([0.4, 0.4])
    psi = np.array([0.0, 1.3, 2.9])
    p = np.full(3, 7.0)

    gradient = np.diff(psi) / h
    np.testing.assert_allclose(
        hole_current(h, D_P, psi, p), -D_P * 7.0 * gradient, rtol=1e-14
    )


def test_electron_current_flows_the_right_way_down_a_potential_drop() -> None:
    """A uniform n-type bar with psi falling to the right.

    E = -dpsi/dx points in +x, so conventional current flows in +x and Jn is
    positive. This is the sign check docs/05-pitfalls.md puts first in the
    debugging order, in its smallest possible form.
    """
    h = np.array([1.0, 1.0])
    psi = np.array([2.0, 1.0, 0.0])
    n = np.full(3, 1e6)

    assert np.all(electron_current(h, D_N, psi, n) > 0.0)


def test_hole_current_flows_the_same_way() -> None:
    h = np.array([1.0, 1.0])
    psi = np.array([2.0, 1.0, 0.0])
    p = np.full(3, 1e6)

    assert np.all(hole_current(h, D_P, psi, p) > 0.0)


def test_high_field_upwinds_the_electron_flux_to_the_left_node() -> None:
    """The Bernoulli asymmetry, stated as directly as it can be stated.

    With X = +10, psi rises to the right, so E points in -x and electrons drift
    in +x. The upwind node for electrons is then the left one, and the flux
    must be dominated by n_left. The ratio of the two sensitivities is
    B(-X)/B(X) = exp(X), which is 22026 here.
    """
    h = np.array([1.0])
    psi = np.array([0.0, 10.0])

    from_left = electron_current(h, D_N, psi, np.array([1.0, 0.0]))
    from_right = electron_current(h, D_N, psi, np.array([0.0, 1.0]))

    np.testing.assert_allclose(abs(from_left / from_right), np.exp(10.0), rtol=1e-12)
    assert abs(from_left) > abs(from_right)


def test_high_field_upwinds_the_hole_flux_to_the_right_node() -> None:
    """The other half of the asymmetry, and the reason it is a trap.

    Same field, opposite answer. Holes drift along E, which points in -x, so
    the upwind node for holes is the right one. An implementation that treats
    both carriers alike passes every symmetric test and fails this one.
    """
    h = np.array([1.0])
    psi = np.array([0.0, 10.0])

    from_left = hole_current(h, D_P, psi, np.array([1.0, 0.0]))
    from_right = hole_current(h, D_P, psi, np.array([0.0, 1.0]))

    np.testing.assert_allclose(abs(from_right / from_left), np.exp(10.0), rtol=1e-12)
    assert abs(from_right) > abs(from_left)


def test_hole_flux_is_the_electron_flux_with_the_potential_reversed() -> None:
    """Jp(psi) = -Jn(-psi) with the same density profile.

    A structural identity rather than a physical one. It pins the two
    implementations to each other, so a later edit to one of them cannot drift
    away from the other unnoticed.
    """
    mesh, h, _ = scaled_mesh()
    psi = junction_potential(mesh)
    density = np.exp(np.linspace(-8.0, 8.0, mesh.n_nodes))

    np.testing.assert_allclose(
        hole_current(h, D_P, psi, density),
        -electron_current(h, D_P, -psi, density),
        rtol=1e-13,
    )


def test_flux_survives_a_field_large_enough_to_overflow_exp() -> None:
    """X = 800 is past the point where exp(X) is representable.

    A depletion region on a coarse mesh reaches arguments like this, and the
    branch structure in physics/bernoulli.py is what keeps the flux finite.
    """
    h = np.array([1.0])
    psi = np.array([0.0, 800.0])
    n = np.array([1e6, 1e6])

    flux = electron_current(h, D_N, psi, n)
    assert np.all(np.isfinite(flux))
    np.testing.assert_allclose(flux, -D_N * 1e6 * 800.0, rtol=1e-12)


def test_diffusivity_may_vary_per_edge() -> None:
    """Phase 3 makes mobility doping dependent, so D lives on edges."""
    h = np.ones(2)
    psi = np.zeros(3)
    n = np.array([0.0, 1.0, 3.0])
    Dn = np.array([2.0, 5.0])

    np.testing.assert_allclose(
        electron_current(h, Dn, psi, n), np.array([2.0 * 1.0, 5.0 * 2.0]), rtol=1e-15
    )


# ----------------------------------------------------------------- residuals


def test_electron_residual_is_zero_for_a_constant_current_solution() -> None:
    """Any n making Jn constant solves div(Jn) = 0 at every interior node.

    Built by walking the flux relation backwards from a chosen current, so the
    profile is an exact solution of the discrete equation and the residual has
    to vanish for a reason that does not involve the solver.

    The potential ramps gently here on purpose. Solving the flux relation
    forwards divides by B(X), which is 5e-11 at X = 27, so a steep profile
    turns this construction into an error amplifier and tests the recursion
    rather than the residual.
    """
    mesh, h, volume = scaled_mesh(n_nodes=11)
    psi = np.linspace(0.0, 2.0, mesh.n_nodes)

    X = np.diff(psi)
    current = 3.0
    n = np.empty(mesh.n_nodes)
    n[0] = 5.0
    for edge in range(mesh.n_edges):
        # Jn = (Dn/h)*(B(X)*n_right - B(-X)*n_left), solved for n_right.
        n[edge + 1] = (
            current * h[edge] / D_N + float(np.asarray(B(-X[edge]))) * n[edge]
        ) / float(np.asarray(B(X[edge])))

    residual = electron_continuity_residual(
        h, volume, D_N, psi, n, np.zeros(mesh.n_nodes)
    )
    np.testing.assert_allclose(residual[1:-1], 0.0, atol=1e-9 * current)


def test_electron_residual_picks_up_recombination() -> None:
    """With no current flowing, the residual is exactly R * volume."""
    mesh, h, volume = scaled_mesh(n_nodes=11)
    psi = np.zeros(mesh.n_nodes)
    n = np.ones(mesh.n_nodes)
    R = np.full(mesh.n_nodes, 0.25)

    np.testing.assert_allclose(
        electron_continuity_residual(h, volume, D_N, psi, n, R),
        R * volume,
        rtol=1e-14,
    )


def test_hole_residual_picks_up_recombination_with_the_same_sign() -> None:
    """div(Jp) = -R, and the residual is written as div(Jp) + R*volume.

    Both residuals therefore reduce to +R*volume when no current flows. That
    shared sign is not a coincidence: recombination removes an electron and a
    hole together, so it enters both equations as a sink.
    """
    mesh, h, volume = scaled_mesh(n_nodes=11)
    psi = np.zeros(mesh.n_nodes)
    p = np.ones(mesh.n_nodes)
    R = np.full(mesh.n_nodes, 0.25)

    np.testing.assert_allclose(
        hole_continuity_residual(h, volume, D_P, psi, p, R), R * volume, rtol=1e-14
    )


# ----------------------------------------------------------------- Jacobians


def dense(
    triplets: tuple[np.ndarray, np.ndarray, np.ndarray], n_nodes: int
) -> np.ndarray:
    """The Jacobian as a dense array, for structure checks."""
    rows, cols, values = triplets
    matrix = np.zeros((n_nodes, n_nodes))
    np.add.at(matrix, (rows, cols), values)
    return matrix


def test_electron_jacobian_is_the_exact_derivative_of_the_residual() -> None:
    """The residual is linear in n when R is held fixed, so J*n reproduces it.

    Exact rather than approximate, which is a stronger statement than any
    finite difference check can make. With R = 0 and dR/dn = 0 the residual is
    exactly -div(Jn), a linear function of n with no constant term.
    """
    mesh, h, volume = scaled_mesh(graded=True)
    psi = junction_potential(mesh)
    n = np.exp(np.linspace(-14.0, 14.0, mesh.n_nodes))
    zero = np.zeros(mesh.n_nodes)

    residual = electron_continuity_residual(h, volume, D_N, psi, n, zero)
    jacobian = dense(
        electron_continuity_jacobian(h, volume, D_N, psi, zero), mesh.n_nodes
    )

    np.testing.assert_allclose(jacobian @ n, residual, rtol=1e-12)


def test_hole_jacobian_is_the_exact_derivative_of_the_residual() -> None:
    mesh, h, volume = scaled_mesh(graded=True)
    psi = junction_potential(mesh)
    p = np.exp(np.linspace(14.0, -14.0, mesh.n_nodes))
    zero = np.zeros(mesh.n_nodes)

    residual = hole_continuity_residual(h, volume, D_P, psi, p, zero)
    jacobian = dense(
        hole_continuity_jacobian(h, volume, D_P, psi, zero), mesh.n_nodes
    )

    np.testing.assert_allclose(jacobian @ p, residual, rtol=1e-12)


def test_recombination_slope_lands_on_the_diagonal_scaled_by_volume() -> None:
    """dR/dn enters as dR_dn * volume and nowhere else."""
    mesh, h, volume = scaled_mesh(n_nodes=11)
    psi = junction_potential(mesh)
    zero = np.zeros(mesh.n_nodes)
    slope = np.linspace(1.0, 2.0, mesh.n_nodes)

    without = dense(
        electron_continuity_jacobian(h, volume, D_N, psi, zero), mesh.n_nodes
    )
    with_slope = dense(
        electron_continuity_jacobian(h, volume, D_N, psi, slope), mesh.n_nodes
    )

    # The flux entries on that diagonal are six orders of magnitude larger
    # than the recombination term, so differencing them costs six digits.
    np.testing.assert_allclose(
        np.diag(with_slope - without), slope * volume, rtol=1e-10
    )
    difference = with_slope - without
    np.fill_diagonal(difference, 0.0)
    np.testing.assert_array_equal(difference, np.zeros_like(difference))


@pytest.mark.parametrize("graded", [False, True])
def test_electron_jacobian_is_an_m_matrix(graded: bool) -> None:
    """Positive diagonal, non-positive off diagonals.

    This is what guarantees a non-negative inverse, and with a non-negative
    right hand side that is what guarantees the solved density is positive.
    Lose it and carrier densities go negative in regions with no physical
    reason, which docs/05-pitfalls.md lists as a symptom of a sign error.
    """
    mesh, h, volume = scaled_mesh(graded=graded)
    psi = junction_potential(mesh)
    slope = np.full(mesh.n_nodes, 0.5)
    matrix = dense(
        electron_continuity_jacobian(h, volume, D_N, psi, slope), mesh.n_nodes
    )

    assert np.all(np.diag(matrix) > 0.0)
    off_diagonal = matrix - np.diag(np.diag(matrix))
    assert np.all(off_diagonal <= 0.0)


@pytest.mark.parametrize("graded", [False, True])
def test_hole_jacobian_is_an_m_matrix(graded: bool) -> None:
    mesh, h, volume = scaled_mesh(graded=graded)
    psi = junction_potential(mesh)
    slope = np.full(mesh.n_nodes, 0.5)
    matrix = dense(hole_continuity_jacobian(h, volume, D_P, psi, slope), mesh.n_nodes)

    assert np.all(np.diag(matrix) > 0.0)
    off_diagonal = matrix - np.diag(np.diag(matrix))
    assert np.all(off_diagonal <= 0.0)


# ------------------------------------------------- the conservation invariant


def ramp_potential(mesh: Mesh1D) -> np.ndarray:
    """A uniform field across the whole device [1], resistor fashion."""
    return 10.0 * (1.0 - mesh.x / mesh.length)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize(
    ("potential", "targets"),
    [
        (ramp_potential, (1e-6, 1e6)),
        (junction_potential, (1e2, 1e6)),
    ],
    ids=["uniform field", "junction under injection"],
)
def test_solved_electron_current_is_constant_across_every_edge(
    graded: bool, potential, targets: tuple[float, float]
) -> None:
    """The primary Phase 2 gate, in its smallest form.

    With recombination off, div(Jn) = 0, so the solved profile carries the same
    current through every edge. Scharfetter-Gummel makes that exact on the mesh
    rather than approximate, so what limits the number below is arithmetic, not
    discretization. Twelve decades of density between the two contacts and it
    still holds.

    Both cases carry a large current on purpose. Jn is the difference of two
    edge terms of size (Dn/h)*n, and near equilibrium those cancel to nothing,
    so the ratio of a flux term to the current sets how many digits survive.
    That is a property of measuring Jn, not of solving for n. The zero current
    case is checked separately below, against the flux scale rather than
    against a current that is not there.
    """
    mesh, h, volume = scaled_mesh(graded=graded)
    psi = potential(mesh)
    n = np.full(mesh.n_nodes, 1.0)
    zero = np.zeros(mesh.n_nodes)

    solved = solve_block(
        electron_continuity_residual(h, volume, D_N, psi, n, zero),
        electron_continuity_jacobian(h, volume, D_N, psi, zero),
        n,
        targets=targets,
    )

    current = electron_current(h, D_N, psi, solved)
    spread = np.max(np.abs(current - current.mean())) / abs(current.mean())
    assert spread < 1e-9, f"Jn varies by {spread:.2e} across the mesh"


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize(
    ("potential", "targets"),
    [
        (ramp_potential, (1e6, 1e-6)),
        (junction_potential, (1e6, 1e2)),
    ],
    ids=["uniform field", "junction under injection"],
)
def test_solved_hole_current_is_constant_across_every_edge(
    graded: bool, potential, targets: tuple[float, float]
) -> None:
    mesh, h, volume = scaled_mesh(graded=graded)
    psi = potential(mesh)
    p = np.full(mesh.n_nodes, 1.0)
    zero = np.zeros(mesh.n_nodes)

    solved = solve_block(
        hole_continuity_residual(h, volume, D_P, psi, p, zero),
        hole_continuity_jacobian(h, volume, D_P, psi, zero),
        p,
        targets=targets,
    )

    current = hole_current(h, D_P, psi, solved)
    spread = np.max(np.abs(current - current.mean())) / abs(current.mean())
    assert spread < 1e-9, f"Jp varies by {spread:.2e} across the mesh"


def test_the_equilibrium_profile_is_an_exact_solution_carrying_no_current() -> None:
    """n = exp(psi) makes every edge flux vanish identically.

    B(-X)*exp(psi_left) and B(X)*exp(psi_right) are equal term by term, because
    B(-X) = B(X)*exp(X) and X is exactly the difference of the two exponents.
    So the current is zero on every edge for any psi whatsoever, on any mesh,
    with no solve involved. That identity is what makes thermal equilibrium a
    fixed point of the whole Gummel cycle.

    The residual left over is bounded against the size of the terms being
    differenced, not against a current, because there is no current here to be
    relative to.
    """
    mesh, h, _ = scaled_mesh(graded=True)
    psi = junction_potential(mesh)
    n = np.exp(psi)

    current = electron_current(h, D_N, psi, n)
    flux_scale = np.max(np.abs((D_N / h) * n[1:]))
    assert np.max(np.abs(current)) < 1e-15 * flux_scale


def test_the_equilibrium_profile_survives_a_block_solve() -> None:
    """Solving from the exact answer must return the exact answer.

    Componentwise, including the minority end where n is 1e-6 while the other
    contact sits at 1e6. Twelve decades apart in one linear system, and the
    small end has to keep its digits, because the minority carrier density is
    precisely what sets the saturation current of a diode. Solving for the
    Newton correction rather than for the density itself is what buys that:
    the roundoff is relative to the update, which is zero here.
    """
    mesh, h, volume = scaled_mesh(graded=True)
    psi = junction_potential(mesh)
    exact = np.exp(psi)
    zero = np.zeros(mesh.n_nodes)

    solved = solve_block(
        electron_continuity_residual(h, volume, D_N, psi, exact, zero),
        electron_continuity_jacobian(h, volume, D_N, psi, zero),
        exact,
        targets=(exact[0], exact[-1]),
    )

    np.testing.assert_allclose(solved, exact, rtol=1e-12)


def test_solved_electron_density_stays_positive_everywhere() -> None:
    """Guaranteed by the M-matrix, and worth checking rather than assuming.

    The generation half of the recombination linearization is the only source
    term, and it is non-negative, so the solution of an M-matrix system with
    positive contact densities cannot dip below zero.
    """
    mesh, h, volume = scaled_mesh(graded=True)
    psi = junction_potential(mesh)
    n = np.full(mesh.n_nodes, 1.0)
    p = np.full(mesh.n_nodes, 1.0)

    model = SRHRecombination(tau_n=1e-3, tau_p=1e-3)
    R = np.asarray(model.rate(n, p))
    slope, _ = model.electron_linearization(n, p)

    solved = solve_block(
        electron_continuity_residual(h, volume, D_N, psi, n, R),
        electron_continuity_jacobian(h, volume, D_N, psi, np.asarray(slope)),
        n,
        targets=(1e-6, 1e6),
    )

    assert np.all(solved > 0.0)


def test_recombination_bends_the_current_the_way_it_should() -> None:
    """With a sink present, Jn is no longer constant, and it rises along x.

    div(Jn) = +R, so a positive recombination rate makes Jn increase from left
    to right. Physically the electrons flow inward from both contacts to be
    consumed in the middle, and conventional electron current runs against the
    electron flow, so Jn is negative at the left contact and positive at the
    right one. Getting that backwards is the sign error this test exists for.

    It also guards the conservation tests above against passing for the trivial
    reason that the recombination term was dropped in the assembly.
    """
    mesh, h, volume = scaled_mesh()
    psi = np.zeros(mesh.n_nodes)
    n = np.full(mesh.n_nodes, 1.0)
    R = np.full(mesh.n_nodes, 1.0)
    slope = np.full(mesh.n_nodes, 1.0)

    solved = solve_block(
        electron_continuity_residual(h, volume, D_N, psi, n, R),
        electron_continuity_jacobian(h, volume, D_N, psi, slope),
        n,
        targets=(10.0, 10.0),
    )
    current = electron_current(h, D_N, psi, solved)

    assert np.max(np.abs(current - current.mean())) / abs(current).max() > 1e-3
    assert current[0] < 0.0 < current[-1]
    assert np.all(np.diff(current) > 0.0)


# ------------------------------------------------------------ the Field layer


def continuity_inputs(n_nodes: int = 21) -> tuple:
    """A mesh, a device state and a model, ready for the assemble functions."""
    mesh = uniform_mesh_1d(MICRON, n_nodes)
    scale = ScaleFactors.for_silicon()
    psi = as_field(junction_potential(mesh), "V", "psi")
    n = as_field(np.full(n_nodes, 1.0), "cm^-3", "n")
    p = as_field(np.full(n_nodes, 1.0), "cm^-3", "p")
    return mesh, psi, n, p, scale


def test_assemble_electron_matches_the_array_level_functions() -> None:
    """The Field wrapper adds unit checking and the division by x_0, nothing else."""
    mesh, psi, n, p, scale = continuity_inputs()
    model = NoRecombination()

    assembly = assemble_electron_continuity(mesh, psi, n, p, model, scale, D_N)
    expected = electron_continuity_residual(
        mesh.h / scale.x_0,
        mesh.volume / scale.x_0,
        D_N,
        psi.data,
        n.data,
        np.zeros(mesh.n_nodes),
    )
    np.testing.assert_allclose(assembly.residual, expected, rtol=1e-14)
    assert assembly.shape == (mesh.n_nodes, mesh.n_nodes)


def test_assemble_hole_matches_the_array_level_functions() -> None:
    mesh, psi, n, p, scale = continuity_inputs()
    model = NoRecombination()

    assembly = assemble_hole_continuity(mesh, psi, n, p, model, scale, D_P)
    expected = hole_continuity_residual(
        mesh.h / scale.x_0,
        mesh.volume / scale.x_0,
        D_P,
        psi.data,
        p.data,
        np.zeros(mesh.n_nodes),
    )
    np.testing.assert_allclose(assembly.residual, expected, rtol=1e-14)


def test_assemble_uses_the_recombination_model() -> None:
    """A model with a real rate must move the residual off the no-model answer."""
    mesh, psi, n, p, scale = continuity_inputs()
    hot = as_field(np.full(mesh.n_nodes, 1e3), "cm^-3", "n")

    without = assemble_electron_continuity(
        mesh, psi, hot, p, NoRecombination(), scale, D_N
    )
    with_srh = assemble_electron_continuity(
        mesh, psi, hot, p, SRHRecombination(tau_n=1.0, tau_p=1.0), scale, D_N
    )

    assert np.max(np.abs(with_srh.residual - without.residual)) > 0.0


def test_assemble_rejects_physical_fields() -> None:
    """A physical psi here is wrong by a factor of 1/V_T and still converges."""
    mesh, psi, n, p, scale = continuity_inputs()
    physical = Field(psi.data, "V", ScalingState.PHYSICAL, Location.NODE, name="psi")

    with pytest.raises(ValueError, match="SCALED"):
        assemble_electron_continuity(
            mesh, physical, n, p, NoRecombination(), scale, D_N
        )


def test_assemble_rejects_edge_fields() -> None:
    mesh, psi, n, p, scale = continuity_inputs()
    on_edges = Field(
        np.zeros(mesh.n_edges), "cm^-3", ScalingState.SCALED, Location.EDGE, name="n"
    )

    with pytest.raises(ValueError, match="NODE"):
        assemble_electron_continuity(
            mesh, psi, on_edges, p, NoRecombination(), scale, D_N
        )


def test_assemble_rejects_a_field_of_the_wrong_length() -> None:
    mesh, psi, n, p, scale = continuity_inputs()
    short = as_field(np.ones(mesh.n_nodes - 1), "cm^-3", "p")

    with pytest.raises(ValueError, match="length"):
        assemble_hole_continuity(mesh, psi, n, short, NoRecombination(), scale, D_P)

