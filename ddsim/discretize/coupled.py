"""The full 3N coupled system for (psi, n, p), assembled simultaneously.

This is the Phase 3 discretization. The three equations are the same ones
poisson.py and continuity.py already assemble separately; what changes is that
psi, n and p are now independent unknowns solved together, so every derivative
that Gummel throws away by holding two of them fixed has to be written down.

Ordering
--------
Node interleaved, per docs/02-numerics.md:

    x = [psi_0, n_0, p_0, psi_1, n_1, p_1, ...]

so the unknown for component c at node i sits at index 3*i + c. Blocking by
variable instead would put a node's three unknowns 2N apart and hand the
factorization a band 2N wide to fill, when the physical coupling is entirely
local.

The one term that changes meaning
---------------------------------
In Phase 1 the Poisson equation carries Boltzmann statistics substituted in, so
n and p are functions of psi and the diagonal picks up (n + p)*volume. That
term is what makes the Phase 1 matrix an M-matrix and it is why that phase
converges so reliably.

**It is not in dF_psi/dpsi here.** n and p are separate unknowns now, so the
same physics lives in dF_psi/dn = +volume and dF_psi/dp = -volume instead.
Carrying it in both places is the most natural way to get this wrong, because
the Phase 1 Jacobian is sitting right there to copy from, and the result would
be a Jacobian wrong by exactly the coupling the phase exists to represent.
Newton would still converge, more slowly, to the right answer, which is the
worst possible symptom.

The nine blocks
---------------
Residuals, identical to the uncoupled ones:

    F_psi,i = (psi_i - psi_{i-1})/h_{i-1} - (psi_{i+1} - psi_i)/h_i
              - (p_i - n_i + N_i)*volume_i
    F_n,i   = R_i*volume_i - (Jn_{i+1/2} - Jn_{i-1/2})
    F_p,i   = (Jp_{i+1/2} - Jp_{i-1/2}) + R_i*volume_i

with the Scharfetter-Gummel fluxes on edge e between nodes e and e+1, and
X_e = psi_{e+1} - psi_e:

    Jn_e = (Dn/h_e) * ( B(X_e)*n_{e+1} - B(-X_e)*n_e )
    Jp_e = (Dp/h_e) * ( B(X_e)*p_e     - B(-X_e)*p_{e+1} )

Differentiating the fluxes with respect to the potential is the only new
algebra in the phase. Both B factors move, and the minus sign in front of the
second one cancels against the minus sign from d(-X)/dX, so the two terms add
rather than subtract:

    dJn_e/dX_e = (Dn/h_e) * ( B'(X_e)*n_{e+1} + B'(-X_e)*n_e )   =: G_e
    dJp_e/dX_e = (Dp/h_e) * ( B'(X_e)*p_e     + B'(-X_e)*p_{e+1} ) =: H_e

and X_e = psi_{e+1} - psi_e gives d/dpsi_{e+1} = +1 and d/dpsi_e = -1.

That makes dF_n/dpsi a Laplacian-shaped stencil with conductance G_e, and
dF_p/dpsi the same stencil with -H_e, the sign flip coming from the two
currents entering their residuals oppositely. Both are the physical statement
that raising the potential at one end of an edge pushes electrons one way and
holes the other.

The recombination cross terms are diagonal, because R is a point function of
the two densities at one node with no edge in it:

    dF_n/dp = dR/dp * volume        dF_p/dn = dR/dn * volume

**These use the exact tangent, not the Gummel linearization.** Phase 2 froze
the SRH denominator to keep the continuity matrix an M-matrix and so keep
solved densities positive. That guarantee does not survive coupling anyway,
since the coupled matrix is not an M-matrix in any case, and a frozen
denominator would cost the quadratic convergence that is the entire reason for
this phase. Positivity is enforced by damping instead. See PROGRESS.md.

Complex step
------------
coupled_residual preserves the dtype of x, which is what lets the verification
in tests/unit/test_coupled.py differentiate it directly. Every array it builds
is derived from x rather than allocated as float64, and physics/bernoulli.py
grew a complex branch for the same reason. A residual that quietly casts to
float returns a complex step Jacobian of exactly zero, which reads as
agreement with any block that happens to be missing a term.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt

from ddsim.core.field import Field, Location, ScalingState
from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.assembly import SparseAssembly
from ddsim.discretize.continuity import Diffusivity
from ddsim.mesh.mesh1d import Mesh1D
from ddsim.physics.bernoulli import B, dB_dx
from ddsim.physics.recombination import Density, RecombinationModel


class Unknown(IntEnum):
    """Which of a node's three unknowns, and its offset within the node."""

    PSI = 0
    """Electrostatic potential."""

    N = 1
    """Electron density."""

    P = 2
    """Hole density."""


UNKNOWNS_PER_NODE = len(Unknown)
"""Three: psi, n and p at every node."""

Number = TypeVar("Number", np.float64, np.complex128)
"""The dtype of an unknown vector.

Constrained to the two dtypes that actually occur rather than left open, so
that the arithmetic in the residual still type checks. A device is always
float64; complex128 appears only under the complex step verification.
coupled_jacobian is real only and says so.
"""


def unknown_index(node: int, component: Unknown) -> int:
    """Index of one unknown in the interleaved vector."""
    return UNKNOWNS_PER_NODE * node + int(component)


def pack(
    psi: npt.NDArray[Number],
    n: npt.NDArray[Number],
    p: npt.NDArray[Number],
) -> npt.NDArray[Number]:
    """Interleave the three node arrays into one unknown vector [1]."""
    if not psi.size == n.size == p.size:
        raise ValueError(
            "psi, n and p must have the same length, got "
            f"{psi.size}, {n.size} and {p.size}."
        )

    x = np.empty(UNKNOWNS_PER_NODE * psi.size, dtype=np.result_type(psi, n, p))
    x[Unknown.PSI :: UNKNOWNS_PER_NODE] = psi
    x[Unknown.N :: UNKNOWNS_PER_NODE] = n
    x[Unknown.P :: UNKNOWNS_PER_NODE] = p
    return x


def unpack(
    x: npt.NDArray[Number],
) -> tuple[npt.NDArray[Number], npt.NDArray[Number], npt.NDArray[Number]]:
    """Split the unknown vector into (psi, n, p) [1].

    Strided views, not copies. The residual is evaluated once per Newton step
    and 3N times per Jacobian verification, and none of it needs a copy.
    """
    return (
        x[Unknown.PSI :: UNKNOWNS_PER_NODE],
        x[Unknown.N :: UNKNOWNS_PER_NODE],
        x[Unknown.P :: UNKNOWNS_PER_NODE],
    )


def _bernoulli_pair(
    psi: npt.NDArray[Number],
) -> tuple[npt.NDArray[Number], npt.NDArray[Number]]:
    """(B(X), B(-X)) on every edge, with X = psi_right - psi_left [1].

    Unlike the one in continuity.py this does not force float64, because the
    residual it feeds has to survive a complex step.
    """
    X = psi[1:] - psi[:-1]
    return np.asarray(B(X)), np.asarray(B(-X))


def _bernoulli_derivative_pair(
    psi: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """(B'(X), B'(-X)) on every edge [1].

    Real only. The Jacobian is assembled at a real state; it is the residual
    that gets differentiated, never this.
    """
    X = psi[1:] - psi[:-1]
    return (
        np.asarray(dB_dx(X), dtype=np.float64),
        np.asarray(dB_dx(-X), dtype=np.float64),
    )


# ----------------------------------------------------------------- residual


def coupled_residual(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    x: npt.NDArray[Number],
    net_doping: npt.NDArray[np.float64],
    Dn: Diffusivity,
    Dp: Diffusivity,
    recombination: RecombinationModel,
) -> npt.NDArray[Number]:
    """Residual of the coupled system, interleaved by node [1].

    Args:
        h: scaled edge lengths [1], length n_nodes - 1.
        volume: scaled dual cell widths [1], length n_nodes.
        x: the interleaved unknown vector [1], length 3*n_nodes.
        net_doping: scaled net doping N = (Nd - Na)/C_0 [1], on nodes.
        Dn: scaled electron diffusivity [1], scalar or per edge.
        Dp: scaled hole diffusivity [1], scalar or per edge.
        recombination: net recombination model, in scaled units.

    Reflecting at both ends, by having no face on the outward side. Contacts
    overwrite those rows afterwards, in discretize/boundary.py.

    Preserves the dtype of x, so complex step differentiation works directly
    on this function. That is how every Jacobian block below is verified.
    """
    psi, n, p = unpack(x)
    bernoulli = _bernoulli_pair(psi)
    b_plus, b_minus = bernoulli
    # The protocol is typed for the physical case, which is float64. Every
    # implementation is dtype preserving and the complex step verification
    # depends on that, but declaring it in the protocol would put complex in
    # every signature in physics/recombination.py to serve this one line.
    R = recombination.rate(cast(Density, n), cast(Density, p))

    out = np.zeros_like(x)
    F_psi, F_n, F_p = unpack(out)

    # Poisson. The flux through each interior face, positive when psi falls to
    # the right, contributing with opposite sign to the two cells it separates.
    face_flux = (psi[:-1] - psi[1:]) / h
    F_psi[:-1] += face_flux
    F_psi[1:] -= face_flux
    F_psi -= (p - n + net_doping) * volume

    # Electron continuity. B(X) multiplies the right hand node. See the
    # docstring of discretize/continuity.py before changing that.
    Jn = (Dn / h) * (b_plus * n[1:] - b_minus * n[:-1])
    F_n += R * volume
    F_n[:-1] -= Jn
    F_n[1:] += Jn

    # Hole continuity. B(X) multiplies the left hand node, the mirror image,
    # and the divergence enters with the opposite sign.
    Jp = (Dp / h) * (b_plus * p[:-1] - b_minus * p[1:])
    F_p += R * volume
    F_p[:-1] += Jp
    F_p[1:] -= Jp

    return out


# ----------------------------------------------------------------- Jacobian


class _Triplets:
    """A COO accumulator that names the block every entry belongs to.

    The point is that the assembly below reads as nine named blocks rather
    than as index arithmetic. Every add call says which equation and which
    unknown it is differentiating, which is the thing a reader has to be able
    to check against the residual by eye.
    """

    def __init__(self) -> None:
        self._rows: list[npt.NDArray[np.int64]] = []
        self._cols: list[npt.NDArray[np.int64]] = []
        self._values: list[npt.NDArray[np.float64]] = []

    def add(
        self,
        equation: Unknown,
        at_nodes: npt.NDArray[np.int64],
        unknown: Unknown,
        of_nodes: npt.NDArray[np.int64],
        values: npt.NDArray[np.float64],
    ) -> None:
        """dF_equation at at_nodes, with respect to unknown at of_nodes."""
        self._rows.append(UNKNOWNS_PER_NODE * at_nodes + int(equation))
        self._cols.append(UNKNOWNS_PER_NODE * of_nodes + int(unknown))
        self._values.append(np.asarray(values, dtype=np.float64))

    def build(
        self,
    ) -> tuple[
        npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]
    ]:
        """(rows, cols, values), with duplicates left for the CSC conversion."""
        return (
            np.concatenate(self._rows),
            np.concatenate(self._cols),
            np.concatenate(self._values),
        )


def coupled_jacobian(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    Dn: Diffusivity,
    Dp: Diffusivity,
    recombination: RecombinationModel,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Jacobian of coupled_residual, in COO form.

    Returns (rows, cols, values). Written as nine named blocks, each one
    checkable against the residual above by eye, per the working agreement in
    CLAUDE.md. Every block is verified against complex step differentiation in
    tests/unit/test_coupled.py, which phases/PHASE-3.md makes non-negotiable.
    """
    psi, n, p = unpack(x)
    n_nodes = psi.size

    b_plus, b_minus = _bernoulli_pair(psi)
    db_plus, db_minus = _bernoulli_derivative_pair(psi)

    dR_dn = np.asarray(recombination.d_rate_dn(n, p), dtype=np.float64)
    dR_dp = np.asarray(recombination.d_rate_dp(n, p), dtype=np.float64)


    nodes = np.arange(n_nodes, dtype=np.int64)
    left = np.arange(h.size, dtype=np.int64)
    right = left + 1

    J = _Triplets()

    # --- dF_psi/dpsi. The bare Laplacian. No charge term: see the module
    # docstring, this is the block the Phase 1 Jacobian would corrupt.
    conductance = 1.0 / h
    diagonal = np.zeros(n_nodes)
    diagonal[:-1] += conductance
    diagonal[1:] += conductance
    J.add(Unknown.PSI, nodes, Unknown.PSI, nodes, diagonal)
    J.add(Unknown.PSI, left, Unknown.PSI, right, -conductance)
    J.add(Unknown.PSI, right, Unknown.PSI, left, -conductance)

    # --- dF_psi/dn and dF_psi/dp. From -(p - n + N)*volume, diagonal only.
    J.add(Unknown.PSI, nodes, Unknown.N, nodes, volume)
    J.add(Unknown.PSI, nodes, Unknown.P, nodes, -volume)

    # --- dF_n/dpsi. Laplacian shaped, with conductance G on each edge.
    G = (Dn / h) * (db_plus * n[1:] + db_minus * n[:-1])
    diagonal = np.zeros(n_nodes)
    diagonal[:-1] += G
    diagonal[1:] += G
    J.add(Unknown.N, nodes, Unknown.PSI, nodes, diagonal)
    J.add(Unknown.N, left, Unknown.PSI, right, -G)
    J.add(Unknown.N, right, Unknown.PSI, left, -G)

    # --- dF_n/dn. The Scharfetter-Gummel stencil plus the exact SRH tangent.
    to_right = (Dn / h) * b_plus
    to_left = (Dn / h) * b_minus
    diagonal = dR_dn * volume
    diagonal[:-1] += to_left
    diagonal[1:] += to_right
    J.add(Unknown.N, nodes, Unknown.N, nodes, diagonal)
    J.add(Unknown.N, left, Unknown.N, right, -to_right)
    J.add(Unknown.N, right, Unknown.N, left, -to_left)

    # --- dF_n/dp. R is a point function, so this touches one node only.
    J.add(Unknown.N, nodes, Unknown.P, nodes, dR_dp * volume)

    # --- dF_p/dpsi. The same stencil as the electron block with the opposite
    # sign, because Jp enters its residual with the opposite sign.
    H = (Dp / h) * (db_plus * p[:-1] + db_minus * p[1:])
    diagonal = np.zeros(n_nodes)
    diagonal[:-1] -= H
    diagonal[1:] -= H
    J.add(Unknown.P, nodes, Unknown.PSI, nodes, diagonal)
    J.add(Unknown.P, left, Unknown.PSI, right, H)
    J.add(Unknown.P, right, Unknown.PSI, left, H)

    # --- dF_p/dn.
    J.add(Unknown.P, nodes, Unknown.N, nodes, dR_dn * volume)

    # --- dF_p/dp. The mirror of the electron block: the two flux coefficients
    # have swapped nodes, for the same reason the fluxes do.
    to_left = (Dp / h) * b_plus
    to_right = (Dp / h) * b_minus
    diagonal = dR_dp * volume
    diagonal[:-1] += to_left
    diagonal[1:] += to_right
    J.add(Unknown.P, nodes, Unknown.P, nodes, diagonal)
    J.add(Unknown.P, left, Unknown.P, right, -to_right)
    J.add(Unknown.P, right, Unknown.P, left, -to_left)

    return J.build()


# --------------------------------------------------------------- Field layer


def assemble_coupled(
    mesh: Mesh1D,
    psi: Field,
    n: Field,
    p: Field,
    net_doping: Field,
    recombination: RecombinationModel,
    scale: ScaleFactors,
    Dn: Diffusivity,
    Dp: Diffusivity,
) -> SparseAssembly:
    """Assemble the coupled 3N system for a 1D mesh.

    Args:
        mesh: the 1D mesh, positions in cm.
        psi: scaled potential on nodes [V].
        n: scaled electron density on nodes [cm^-3].
        p: scaled hole density on nodes [cm^-3].
        net_doping: scaled net doping on nodes [cm^-3].
        recombination: a model built in scaled units.
        scale: de Mari scale factors, used to put the mesh in units of x_0.
        Dn: scaled electron diffusivity [1].
        Dp: scaled hole diffusivity [1].

    Checks the scaling state and mesh location once here, then works on raw
    arrays, exactly as assemble_poisson and the two continuity assemblies do.
    """
    for name, field in (
        ("psi", psi),
        ("n", n),
        ("p", p),
        ("net_doping", net_doping),
    ):
        if field.scaling is not ScalingState.SCALED:
            raise ValueError(
                f"{name} must be SCALED before assembly, got {field.scaling.name}. "
                "A physical value here is wrong by a fixed factor and would "
                "still converge."
            )
        if field.location is not Location.NODE:
            raise ValueError(f"{name} must live on NODE, got {field.location.name}.")
        if field.size != mesh.n_nodes:
            raise ValueError(
                f"{name} has length {field.size} but the mesh has "
                f"{mesh.n_nodes} nodes."
            )

    h = mesh.h / scale.x_0
    volume = mesh.volume / scale.x_0
    x: npt.NDArray[np.float64] = pack(psi.data, n.data, p.data)

    residual = coupled_residual(
        h, volume, x, net_doping.data, Dn, Dp, recombination
    )
    rows, cols, values = coupled_jacobian(h, volume, x, Dn, Dp, recombination)

    size = UNKNOWNS_PER_NODE * mesh.n_nodes
    return SparseAssembly(
        residual=np.asarray(residual, dtype=np.float64),
        rows=rows,
        cols=cols,
        values=values,
        shape=(size, size),
    )
