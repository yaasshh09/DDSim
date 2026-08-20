"""Nonlinear Poisson at equilibrium, 1D box integration.

One unknown per node. Boltzmann statistics substituted in to eliminate n and p,
per docs/02-numerics.md:

    lap(psi) = -(n_i*exp(-psi/V_T) - n_i*exp(psi/V_T) + N)

In scaled units that is just

    lap(psi) = -(p - n + N)     with     n = exp(psi),  p = exp(-psi)

Discretization is box integration over the dual cell of each node, which is
what makes the scheme conservative and what carries over unchanged to the
Scharfetter-Gummel fluxes in Phase 2. Integrating the Laplacian over the cell
around node i turns it into the difference of the two face fluxes:

    integral(lap psi) = (psi_{i+1} - psi_i)/h_i - (psi_i - psi_{i-1})/h_{i-1}

and the charge term picks up the cell volume. The residual is written with the
overall sign that makes the Jacobian diagonal positive:

    F_i = (psi_i - psi_{i-1})/h_{i-1} - (psi_{i+1} - psi_i)/h_i
          - (p_i - n_i + N_i) * volume_i

    dF_i/dpsi_{i-1} = -1/h_{i-1}
    dF_i/dpsi_{i+1} = -1/h_i
    dF_i/dpsi_i     =  1/h_{i-1} + 1/h_i + (n_i + p_i) * volume_i

The diagonal is strictly positive and the off-diagonals strictly negative, so
the matrix is a symmetric M-matrix and Newton on it is reliably convergent.
That is why Phase 1 comes before everything else.

Boundary nodes get the natural reflecting condition, homogeneous Neumann, by
simply having no face on the outward side. Contacts overwrite those rows
afterwards, in discretize/boundary.py. docs/01-physics.md: every boundary that
is not a contact is reflecting.

The array level functions are dtype preserving so that complex step
differentiation works on them, which is how the Jacobian is verified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ddsim.core.field import Field, Location, ScalingState
from ddsim.core.scaling import ScaleFactors
from ddsim.mesh.mesh1d import Mesh1D


@dataclass(frozen=True)
class PoissonAssembly:
    """A residual vector and a Jacobian in COO form, ready for solve/linear.py."""

    residual: npt.NDArray[np.float64]
    """F(psi) [1], one entry per node."""

    rows: npt.NDArray[np.int64]
    """Jacobian row indices."""

    cols: npt.NDArray[np.int64]
    """Jacobian column indices."""

    values: npt.NDArray[np.float64]
    """Jacobian values, same length as rows and cols."""

    shape: tuple[int, int]
    """Jacobian shape, (n_nodes, n_nodes)."""


def poisson_residual(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    psi: npt.NDArray[np.float64],
    net_doping: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Residual of the scaled nonlinear Poisson equation [1].

    Args:
        h: scaled edge lengths [1], length n_nodes - 1.
        volume: scaled dual cell widths [1], length n_nodes.
        psi: scaled potential [1], length n_nodes.
        net_doping: scaled net doping N = (Nd - Na)/C_0 [1], length n_nodes.

    Reflecting at both ends. Contacts are applied separately.

    Does not force a dtype, so passing a complex psi gives a complex residual
    and complex step differentiation works directly on this function.
    """
    n = np.exp(psi)
    p = np.exp(-psi)

    residual = np.zeros_like(psi)

    # Flux through each interior face, from the left node to the right node.
    # face_flux[e] is the flux across edge e, positive when psi decreases
    # to the right.
    face_flux = (psi[:-1] - psi[1:]) / h

    # Each face contributes with opposite sign to the two cells it separates.
    residual[:-1] += face_flux
    residual[1:] -= face_flux

    residual -= (p - n + net_doping) * volume
    return residual


def poisson_jacobian(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    psi: npt.NDArray[np.float64],
    net_doping: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Jacobian of poisson_residual, in COO form.

    Returns (rows, cols, values). Written term by term rather than assembled
    with a stencil helper, so that a device engineer can check each derivative
    against the residual above by eye.
    """
    n_nodes = psi.size
    n = np.exp(psi)
    p = np.exp(-psi)

    nodes = np.arange(n_nodes, dtype=np.int64)
    edges = np.arange(h.size, dtype=np.int64)
    conductance = 1.0 / h

    # Diagonal: both adjacent face conductances, plus the charge derivative.
    # d/dpsi of -(p - n) is (p + n), and both are positive, so the charge term
    # can only strengthen the diagonal.
    diagonal = (n + p) * volume
    np.add.at(diagonal, edges, conductance)
    np.add.at(diagonal, edges + 1, conductance)

    # Off diagonals: one entry per face, in each direction. Symmetric.
    rows = np.concatenate([nodes, edges, edges + 1])
    cols = np.concatenate([nodes, edges + 1, edges])
    values = np.concatenate([diagonal, -conductance, -conductance])

    return rows, cols, values


def assemble_poisson(
    mesh: Mesh1D,
    psi: Field,
    net_doping: Field,
    scale: ScaleFactors,
) -> PoissonAssembly:
    """Assemble the equilibrium Poisson system for a 1D mesh.

    Args:
        mesh: the 1D mesh, positions in cm.
        psi: scaled potential on nodes [V], must be SCALED.
        net_doping: scaled net doping on nodes [cm^-3], must be SCALED.
        scale: the de Mari scale factors, used to put the mesh in units of x_0.

    Checks the scaling state and mesh location once here, then works on raw
    arrays, which is the pattern docs/03-architecture.md prescribes.

    The mesh arrives in cm and the equation is in units of the Debye length, so
    it is divided by x_0 here. Forgetting that is a silent error of several
    orders of magnitude, since x_0 is 40.9 um at intrinsic doping while a
    device is 1 um across.
    """
    for name, field in (("psi", psi), ("net_doping", net_doping)):
        if field.scaling is not ScalingState.SCALED:
            raise ValueError(
                f"{name} must be SCALED before assembly, got {field.scaling.name}. "
                "A physical potential here is wrong by a factor of 1/V_T and "
                "would still converge."
            )
        if field.location is not Location.NODE:
            raise ValueError(
                f"{name} must live on NODE, got {field.location.name}."
            )
        if field.size != mesh.n_nodes:
            raise ValueError(
                f"{name} has length {field.size} but the mesh has "
                f"{mesh.n_nodes} nodes."
            )

    h = mesh.h / scale.x_0
    volume = mesh.volume / scale.x_0

    residual = poisson_residual(h, volume, psi.data, net_doping.data)
    rows, cols, values = poisson_jacobian(h, volume, psi.data, net_doping.data)

    return PoissonAssembly(
        residual=residual,
        rows=rows,
        cols=cols,
        values=values,
        shape=(mesh.n_nodes, mesh.n_nodes),
    )
