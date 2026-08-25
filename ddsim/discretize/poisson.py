"""Nonlinear Poisson at equilibrium, 1D box integration.

One unknown per node. Boltzmann statistics substituted in to eliminate n and p,
per docs/02-numerics.md:

    lap(psi) = -(n_i*exp(-psi/V_T) - n_i*exp(psi/V_T) + N)

In scaled units that is just

    lap(psi) = -(p - n + N)

    n = exp(psi - phi_n),   p = exp(phi_p - psi)

phi_n and phi_p are the quasi-Fermi potentials, held fixed here. At true
thermal equilibrium both are zero and the densities reduce to exp(+/- psi).

They are carried because without them an applied bias cannot reach a junction.
With the quasi-Fermi levels pinned at zero the densities are tied absolutely to
psi, so a quasi-neutral region cannot shift its potential without changing p by
exp(38.7) per volt. The bias piles up in a thin layer at the contact instead:
measured on a 1e16 diode at -1 V, the whole volt falls across 0.05 um at the
contact with a 2e5 V/cm field there, while the junction field stays at its zero
bias value.

They have to be separate, not one common phi. Under reverse bias phi_n and
phi_p are split by exactly the applied bias throughout the depletion region,
which is what reverse bias means. Forcing a single phi with a step at the
metallurgical junction makes n = exp(psi - phi) blow up to 1e26 cm^-3 on the p
side of the junction, which screens the field and gives a depletion width three
times too small.

Shifting psi, phi_n and phi_p together by the same amount leaves n and p
unchanged, which is the freedom a biased neutral region needs.

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

import numpy as np
import numpy.typing as npt

from ddsim.core.field import Field, Location, ScalingState
from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.assembly import SparseAssembly
from ddsim.discretize.geometry import UNIFORM_1D, EdgeGeometry
from ddsim.mesh.mesh1d import Mesh1D

BoltzmannDensities = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
"""(n, p) on nodes [1], from psi at fixed quasi-Fermi levels."""


def _boltzmann_densities(
    psi: npt.NDArray[np.float64],
    phi_n: npt.NDArray[np.float64] | None,
    phi_p: npt.NDArray[np.float64] | None,
) -> BoltzmannDensities:
    """(n, p) from the potential at fixed quasi-Fermi levels [1].

    n = exp(psi - phi_n) and p = exp(phi_p - psi), with None meaning a level
    pinned at zero, which is true thermal equilibrium.

    Does not force a dtype, so a complex psi gives complex densities and
    complex step differentiation works through here.
    """
    n = np.exp(psi if phi_n is None else psi - phi_n)
    p = np.exp(-psi if phi_p is None else phi_p - psi)
    return n, p


def poisson_residual(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    psi: npt.NDArray[np.float64],
    net_doping: npt.NDArray[np.float64],
    phi_n: npt.NDArray[np.float64] | None = None,
    phi_p: npt.NDArray[np.float64] | None = None,
    geometry: EdgeGeometry = UNIFORM_1D,
) -> npt.NDArray[np.float64]:
    """Residual of the scaled nonlinear Poisson equation [1].

    Args:
        h: scaled edge lengths [1], one per edge.
        volume: scaled dual cell volumes [1], length n_nodes.
        psi: scaled potential [1], length n_nodes.
        net_doping: scaled net doping N = (Nd - Na)/C_0 [1], length n_nodes.
        phi_n: electron quasi-Fermi potential [1], length n_nodes. None means
            zero, which is true thermal equilibrium.
        phi_p: hole quasi-Fermi potential [1], length n_nodes. None means zero.
        geometry: which nodes each edge joins and what it carries. The default
            is the contiguous 1D chain in silicon.

    Reflecting on every boundary that is not a contact, which in box
    integration means doing nothing at all: a node simply has no face on the
    outward side. Contacts are applied separately.

    Does not force a dtype, so passing a complex psi gives a complex residual
    and complex step differentiation works directly on this function.
    """
    return _poisson_residual(
        h,
        volume,
        psi,
        net_doping,
        _boltzmann_densities(psi, phi_n, phi_p),
        geometry,
    )


def _poisson_residual(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    psi: npt.NDArray[np.float64],
    net_doping: npt.NDArray[np.float64],
    densities: BoltzmannDensities,
    geometry: EdgeGeometry = UNIFORM_1D,
) -> npt.NDArray[np.float64]:
    """poisson_residual with the Boltzmann densities already in hand [1].

    The residual and the Jacobian are built from the same n and p, and two
    exponentials over every node is the most expensive thing in either, so
    the assembly evaluates them once and hands them to both. Private because
    the densities have to be the ones belonging to this psi and nothing
    outside can check that.
    """
    n, p = densities

    residual = np.zeros_like(psi)
    left, right = geometry.ends(h.size)

    # Flux through each interior face, from the left node to the right node.
    # face_flux[e] is the flux across edge e, positive when psi decreases
    # towards the right node. The permittivity rides on the face rather than
    # on the cell, which is what makes the normal component of D continuous
    # across a material interface instead of E. See docs/01-physics.md.
    face_flux = geometry.weight * (psi[left] - psi[right]) / h

    # Each face contributes with opposite sign to the two cells it separates.
    np.add.at(residual, left, face_flux)
    np.add.at(residual, right, -face_flux)

    residual -= (p - n + net_doping) * volume
    return residual


def poisson_jacobian(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    psi: npt.NDArray[np.float64],
    net_doping: npt.NDArray[np.float64],
    phi_n: npt.NDArray[np.float64] | None = None,
    phi_p: npt.NDArray[np.float64] | None = None,
    geometry: EdgeGeometry = UNIFORM_1D,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Jacobian of poisson_residual, in COO form.

    Returns (rows, cols, values). Written term by term rather than assembled
    with a stencil helper, so that a device engineer can check each derivative
    against the residual above by eye.

    The quasi-Fermi levels are held fixed, so dn/dpsi is still n and dp/dpsi is
    still -p and the diagonal keeps its form.
    """
    return _poisson_jacobian(
        h, volume, psi.size, _boltzmann_densities(psi, phi_n, phi_p), geometry
    )


def _poisson_jacobian(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    n_nodes: int,
    densities: BoltzmannDensities,
    geometry: EdgeGeometry = UNIFORM_1D,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """poisson_jacobian with the Boltzmann densities already in hand."""
    n, p = densities

    nodes = np.arange(n_nodes, dtype=np.int64)
    left, right = geometry.ends(h.size)
    conductance = geometry.weight / h

    # Diagonal: both adjacent face conductances, plus the charge derivative.
    # d/dpsi of -(p - n) is (p + n), and both are positive, so the charge term
    # can only strengthen the diagonal.
    # Every node picks up the conductance of each face it touches. Scattered
    # rather than sliced: in 1D the edges touching a node are contiguous and a
    # pair of slice additions would do, but in 2D a node has four of them and
    # they are not adjacent in the ordering. See geometry.py on why this is
    # add.at and not bincount.
    diagonal = (n + p) * volume
    np.add.at(diagonal, left, conductance)
    np.add.at(diagonal, right, conductance)

    # Off diagonals: one entry per face, in each direction. Symmetric.
    rows = np.concatenate([nodes, left, right])
    cols = np.concatenate([nodes, right, left])
    values = np.concatenate([diagonal, -conductance, -conductance])

    return rows, cols, values


def assemble_poisson(
    mesh: Mesh1D,
    psi: Field,
    net_doping: Field,
    scale: ScaleFactors,
    phi_n: Field | None = None,
    phi_p: Field | None = None,
    geometry: EdgeGeometry = UNIFORM_1D,
) -> SparseAssembly:
    """Assemble the equilibrium Poisson system for a 1D mesh.

    Args:
        mesh: the 1D mesh, positions in cm.
        psi: scaled potential on nodes [V], must be SCALED.
        net_doping: scaled net doping on nodes [cm^-3], must be SCALED.
        scale: the de Mari scale factors, used to put the mesh in units of x_0.
        phi_n: electron quasi-Fermi potential on nodes [V], must be SCALED.
            None means true equilibrium.
        phi_p: hole quasi-Fermi potential on nodes [V], must be SCALED.
        geometry: which nodes each edge joins and what it carries. The default
            is the contiguous 1D chain in silicon.

    Checks the scaling state and mesh location once here, then works on raw
    arrays, which is the pattern docs/03-architecture.md prescribes.

    The mesh arrives in cm and the equation is in units of the Debye length, so
    it is divided by x_0 here. Forgetting that is a silent error of several
    orders of magnitude, since x_0 is 40.9 um at intrinsic doping while a
    device is 1 um across.
    """
    checked = [("psi", psi), ("net_doping", net_doping)]
    if phi_n is not None:
        checked.append(("phi_n", phi_n))
    if phi_p is not None:
        checked.append(("phi_p", phi_p))

    for name, field in checked:
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

    n_values = None if phi_n is None else phi_n.data
    p_values = None if phi_p is None else phi_p.data

    # One pair of exponentials for both halves of the system.
    densities = _boltzmann_densities(psi.data, n_values, p_values)
    residual = _poisson_residual(
        h, volume, psi.data, net_doping.data, densities, geometry
    )
    rows, cols, values = _poisson_jacobian(
        h, volume, mesh.n_nodes, densities, geometry
    )

    return SparseAssembly(
        residual=residual,
        rows=rows,
        cols=cols,
        values=values,
        shape=(mesh.n_nodes, mesh.n_nodes),
    )
