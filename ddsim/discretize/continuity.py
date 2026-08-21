"""Scharfetter-Gummel discretization of the two continuity equations.

This is the module docs/02-numerics.md says to get right before anything else,
so the derivation is repeated here rather than referred to.

On the edge between nodes i and i+1, assume Jn and the field are constant and
integrate the current relation analytically. With X = psi_{i+1} - psi_i in
scaled units, where psi is already measured in units of V_T:

    Jn_{i+1/2} = (Dn/h) * ( B(X)*n_{i+1} - B(-X)*n_i )
    Jp_{i+1/2} = (Dp/h) * ( B(X)*p_i     - B(-X)*p_{i+1} )

**The B(X) factor attaches to the right node for electrons and to the left node
for holes.** That asymmetry is the single most dangerous detail in the project.
Reversing it gives a solver that converges cleanly to a physically wrong answer
with the current running backwards, which no convergence diagnostic will ever
report. It is physical, not arbitrary: for X > 0 the field points in -x, so
electrons drift in +x and holes drift in -x, and each flux is dominated by its
own upwind node. Those are opposite ends of the same edge.

Discrete equations, box integrated over the dual cell of each node exactly as
in poisson.py, so that the two share their conservation structure:

    div(Jn)_i = Jn_{i+1/2} - Jn_{i-1/2}      steady state: div(Jn) = R
    div(Jp)_i = Jp_{i+1/2} - Jp_{i-1/2}      steady state: div(Jp) = -R

The residuals are written with the sign that makes the diagonal positive and
the off diagonals negative, matching poisson.py:

    F_n,i = R_i*volume_i - (Jn_{i+1/2} - Jn_{i-1/2})
    F_p,i = (Jp_{i+1/2} - Jp_{i-1/2}) + R_i*volume_i

Both then reduce to +R*volume when no current flows, which is right: an
electron and a hole recombine together, so recombination is a sink in both
equations.

Why the matrix is an M-matrix, and why that matters
--------------------------------------------------
B(x) > 0 everywhere, so every off diagonal entry is strictly negative and every
diagonal entry is a sum of positive terms. Adding dR/dn to the diagonal can
only strengthen it, since dR/dn > 0 at every density. The result is a
non-singular M-matrix, whose inverse is non-negative. Feed it a non-negative
right hand side and the solved density is non-negative, with no clamping
anywhere. docs/05-pitfalls.md is emphatic that clamping a negative density is
never the repair, and this is what makes clamping unnecessary in 1D.

The recombination arguments
---------------------------
The array level functions take R and dR_dn as plain arrays rather than a model.
The caller decides what goes in them, and the two phases want different things:

- Gummel (Phase 2) passes the frozen denominator slope c = p/D from
  physics/recombination.py, which keeps the right hand side non-negative and so
  keeps densities positive.
- Full Newton (Phase 3) will pass the exact tangent dR/dn.

Either way the residual is evaluated with the true R at the current state, so
the converged answer solves the true equation and not a linearized substitute.

Conservation, and the one thing that limits it
----------------------------------------------
Because the scheme is built from edge fluxes, the discrete divergence of the
discrete current is exactly zero in a source free steady state. With
recombination off, the solved profile carries the same Jn through every edge to
machine precision, not to a discretization tolerance. That is the strongest
single correctness check in the project and it is the primary gate for Phase 2.

What limits it in practice is not the solve but the measurement. Jn is the
difference of two edge terms each of size (Dn/h)*n, and near equilibrium those
two cancel to almost nothing: on a 1e16 junction the terms reach 5e9 in scaled
units while the current is zero, so the answer comes out at the 1e-7 level
rather than at zero. The relative spread of Jn across the mesh is therefore
about eps times the ratio of a flux term to the current, and it degrades at low
bias exactly where the current is small. The current itself is right, and the
solved densities are right to twelve digits; only the subtraction is lossy.

Where that matters, measure the terminal current from the residual instead. The
residual form telescopes over the whole device and carries no cancellation, so
it stays accurate where the edge difference does not.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ddsim.core.field import Field, Location, ScalingState
from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.assembly import SparseAssembly
from ddsim.mesh.mesh1d import Mesh1D
from ddsim.physics.bernoulli import B
from ddsim.physics.recombination import RecombinationModel

Diffusivity = float | npt.NDArray[np.float64]
"""Scaled diffusivity [1], one value or one per edge."""


def _bernoulli_pair(
    psi: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """(B(X), B(-X)) on every edge, with X = psi_right - psi_left [1]."""
    X = psi[1:] - psi[:-1]
    return np.asarray(B(X), dtype=np.float64), np.asarray(B(-X), dtype=np.float64)


# --------------------------------------------------------------- edge fluxes


def electron_current(
    h: npt.NDArray[np.float64],
    Dn: Diffusivity,
    psi: npt.NDArray[np.float64],
    n: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Scharfetter-Gummel electron current on every edge [1].

    Args:
        h: scaled edge lengths [1], length n_nodes - 1.
        Dn: scaled electron diffusivity [1], scalar or per edge.
        psi: scaled potential on nodes [1].
        n: scaled electron density on nodes [1].

    Positive means conventional current flowing in +x. B(X) multiplies the
    right hand node. See the module docstring before changing that.
    """
    b_plus, b_minus = _bernoulli_pair(psi)
    return np.asarray((Dn / h) * (b_plus * n[1:] - b_minus * n[:-1]))


def hole_current(
    h: npt.NDArray[np.float64],
    Dp: Diffusivity,
    psi: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Scharfetter-Gummel hole current on every edge [1].

    B(X) multiplies the left hand node here, the mirror image of the electron
    flux. That is the asymmetry docs/05-pitfalls.md warns about.
    """
    b_plus, b_minus = _bernoulli_pair(psi)
    return np.asarray((Dp / h) * (b_plus * p[:-1] - b_minus * p[1:]))


# ----------------------------------------------------------------- residuals


def electron_continuity_residual(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    Dn: Diffusivity,
    psi: npt.NDArray[np.float64],
    n: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Residual of the scaled electron continuity equation [1].

        F_i = R_i*volume_i - (Jn_{i+1/2} - Jn_{i-1/2})

    Args:
        h: scaled edge lengths [1].
        volume: scaled dual cell widths [1].
        Dn: scaled electron diffusivity [1].
        psi: scaled potential on nodes [1].
        n: scaled electron density on nodes [1].
        R: scaled net recombination rate on nodes [1].

    Boundary nodes get the reflecting condition for free, by having no face on
    the outward side. Contacts overwrite those rows afterwards.
    """
    current = electron_current(h, Dn, psi, n)

    residual = R * volume
    residual[:-1] -= current
    residual[1:] += current
    return np.asarray(residual)


def hole_continuity_residual(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    Dp: Diffusivity,
    psi: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Residual of the scaled hole continuity equation [1].

        F_i = (Jp_{i+1/2} - Jp_{i-1/2}) + R_i*volume_i

    The divergence enters with the opposite sign to the electron equation,
    because div(Jp) = -R while div(Jn) = +R. Both residuals still reduce to
    +R*volume with no current flowing.
    """
    current = hole_current(h, Dp, psi, p)

    residual = R * volume
    residual[:-1] += current
    residual[1:] -= current
    return np.asarray(residual)


# ----------------------------------------------------------------- Jacobians


def electron_continuity_jacobian(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    Dn: Diffusivity,
    psi: npt.NDArray[np.float64],
    dR_dn: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """dF_n/dn in COO form, with psi and p held fixed.

    Written term by term against the residual above:

        dF_i/dn_{i-1} = -(Dn/h_{i-1}) * B(-X_{i-1})
        dF_i/dn_{i+1} = -(Dn/h_i)     * B(X_i)
        dF_i/dn_i     =  (Dn/h_i)*B(-X_i) + (Dn/h_{i-1})*B(X_{i-1})
                         + dR_dn_i * volume_i

    dR_dn is whatever the caller supplies. See the module docstring for why
    Gummel and Newton want different things there.
    """
    n_nodes = psi.size
    b_plus, b_minus = _bernoulli_pair(psi)

    # Coefficients of the two nodes in the edge flux Jn = right*n_right -
    # left*n_left. Both are strictly positive because B(x) > 0 everywhere.
    right = np.asarray((Dn / h) * b_plus)
    left = np.asarray((Dn / h) * b_minus)

    nodes = np.arange(n_nodes, dtype=np.int64)
    edges = np.arange(h.size, dtype=np.int64)

    diagonal = dR_dn * volume
    np.add.at(diagonal, edges, left)
    np.add.at(diagonal, edges + 1, right)

    rows = np.concatenate([nodes, edges, edges + 1])
    cols = np.concatenate([nodes, edges + 1, edges])
    values = np.concatenate([diagonal, -right, -left])

    return rows, cols, values


def hole_continuity_jacobian(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    Dp: Diffusivity,
    psi: npt.NDArray[np.float64],
    dR_dp: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """dF_p/dp in COO form, with psi and n held fixed.

        dF_i/dp_{i-1} = -(Dp/h_{i-1}) * B(X_{i-1})
        dF_i/dp_{i+1} = -(Dp/h_i)     * B(-X_i)
        dF_i/dp_i     =  (Dp/h_i)*B(X_i) + (Dp/h_{i-1})*B(-X_{i-1})
                         + dR_dp_i * volume_i

    The two off diagonal factors have swapped places relative to the electron
    Jacobian, for the same reason the fluxes do.
    """
    n_nodes = psi.size
    b_plus, b_minus = _bernoulli_pair(psi)

    # Jp = left*p_left - right*p_right, the mirror of the electron flux.
    left = np.asarray((Dp / h) * b_plus)
    right = np.asarray((Dp / h) * b_minus)

    nodes = np.arange(n_nodes, dtype=np.int64)
    edges = np.arange(h.size, dtype=np.int64)

    diagonal = dR_dp * volume
    np.add.at(diagonal, edges, left)
    np.add.at(diagonal, edges + 1, right)

    rows = np.concatenate([nodes, edges, edges + 1])
    cols = np.concatenate([nodes, edges + 1, edges])
    values = np.concatenate([diagonal, -right, -left])

    return rows, cols, values


# --------------------------------------------------------------- Field layer


def _check(mesh: Mesh1D, named: tuple[tuple[str, Field], ...]) -> None:
    """Scaling state, mesh location and length, checked once at entry."""
    for name, field in named:
        if field.scaling is not ScalingState.SCALED:
            raise ValueError(
                f"{name} must be SCALED before assembly, got {field.scaling.name}. "
                "A physical density here is wrong by a factor of C_0 and would "
                "still converge."
            )
        if field.location is not Location.NODE:
            raise ValueError(f"{name} must live on NODE, got {field.location.name}.")
        if field.size != mesh.n_nodes:
            raise ValueError(
                f"{name} has length {field.size} but the mesh has "
                f"{mesh.n_nodes} nodes."
            )


def assemble_electron_continuity(
    mesh: Mesh1D,
    psi: Field,
    n: Field,
    p: Field,
    recombination: RecombinationModel,
    scale: ScaleFactors,
    Dn: Diffusivity,
) -> SparseAssembly:
    """Assemble the electron continuity system for a 1D mesh.

    Args:
        mesh: the 1D mesh, positions in cm.
        psi: scaled potential on nodes [V].
        n: scaled electron density on nodes [cm^-3].
        p: scaled hole density on nodes [cm^-3], held fixed.
        recombination: a model built in scaled units.
        scale: de Mari scale factors, used to put the mesh in units of x_0.
        Dn: scaled electron diffusivity [1].

    Uses the Gummel linearization of the recombination term, R = c*n - g with
    the denominator frozen, which is what keeps the solved density positive.
    Phase 3 will want the exact tangent here instead.

    The mesh arrives in cm and the equation is in units of the Debye length, so
    it is divided by x_0 here, exactly as in assemble_poisson.
    """
    _check(mesh, (("psi", psi), ("n", n), ("p", p)))

    h = mesh.h / scale.x_0
    volume = mesh.volume / scale.x_0

    R = np.asarray(recombination.rate(n.data, p.data), dtype=np.float64)
    slope, _ = recombination.electron_linearization(n.data, p.data)

    residual = electron_continuity_residual(h, volume, Dn, psi.data, n.data, R)
    rows, cols, values = electron_continuity_jacobian(
        h, volume, Dn, psi.data, np.asarray(slope, dtype=np.float64)
    )

    return SparseAssembly(
        residual=residual,
        rows=rows,
        cols=cols,
        values=values,
        shape=(mesh.n_nodes, mesh.n_nodes),
    )


def assemble_hole_continuity(
    mesh: Mesh1D,
    psi: Field,
    n: Field,
    p: Field,
    recombination: RecombinationModel,
    scale: ScaleFactors,
    Dp: Diffusivity,
) -> SparseAssembly:
    """Assemble the hole continuity system for a 1D mesh.

    Same contract as assemble_electron_continuity, with n held fixed instead.
    """
    _check(mesh, (("psi", psi), ("n", n), ("p", p)))

    h = mesh.h / scale.x_0
    volume = mesh.volume / scale.x_0

    R = np.asarray(recombination.rate(n.data, p.data), dtype=np.float64)
    slope, _ = recombination.hole_linearization(n.data, p.data)

    residual = hole_continuity_residual(h, volume, Dp, psi.data, p.data, R)
    rows, cols, values = hole_continuity_jacobian(
        h, volume, Dp, psi.data, np.asarray(slope, dtype=np.float64)
    )

    return SparseAssembly(
        residual=residual,
        rows=rows,
        cols=cols,
        values=values,
        shape=(mesh.n_nodes, mesh.n_nodes),
    )
