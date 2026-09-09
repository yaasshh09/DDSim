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
this phase. Positivity is enforced by damping instead. See
docs/07-decisions.md.

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

from collections.abc import Sequence
from enum import Enum, IntEnum
from typing import NamedTuple, TypeVar, cast

import numpy as np
import numpy.typing as npt

from ddsim.core import constants as C
from ddsim.core.field import Field, Location, ScalingState
from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.assembly import SparseAssembly
from ddsim.discretize.boundary import (
    Carrier,
    Contact,
    GateContact,
    apply_dirichlet_nodes,
    gate_psi_scaled,
    ohmic_density_scaled,
    ohmic_psi_scaled,
)
from ddsim.discretize.continuity import Diffusivity
from ddsim.discretize.geometry import UNIFORM_1D, EdgeGeometry
from ddsim.mesh.mesh1d import Mesh1D
from ddsim.physics.bernoulli import B, dB_dx
from ddsim.physics.mobility import (
    EdgeDiffusivity,
    diffusivity_at,
    diffusivity_tangent,
)
from ddsim.physics.recombination import Density, RecombinationModel
from ddsim.physics.statistics import Degeneracy


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


TermScales = tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]
"""The size of the terms every row is assembled from [1], by family.

Three arrays of one entry per node, not three numbers. A residual is a
difference of terms and cannot be resolved below machine epsilon times the
things being differenced, so the terms are what sets the floor, and they are a
property of a row rather than of an equation family. On a 1e17 / 1e20 junction
the electron flux terms span twelve decades between the two sides. Measured
against a single number for the whole family, the lightly doped rows are
divided by something set on the degenerate side and land below any threshold
whatever they say: a cold solve at 0.4 V reported convergence after zero
Newton steps, still sitting on the equilibrium guess, with a terminal current
seven decades below the answer. See docs/07-decisions.md.
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


def edge_drop(
    psi: npt.NDArray[Number], geometry: EdgeGeometry = UNIFORM_1D
) -> npt.NDArray[Number]:
    """psi_right - psi_left on every edge [1], the X the Bernoulli pair uses.

    The field along an edge is this divided by the edge length, with a minus
    sign. A field dependent mobility wants exactly the same quantity the
    Scharfetter-Gummel argument does, which is why they are computed the same
    way here rather than each in its own convention.
    """
    node_left, node_right = geometry.ends_of(psi.size)
    return psi[node_right] - psi[node_left]


def effective_potentials(
    psi: npt.NDArray[Number],
    n: npt.NDArray[Number],
    p: npt.NDArray[Number],
    degeneracy: Degeneracy | None = None,
) -> tuple[npt.NDArray[Number], npt.NDArray[Number]]:
    """The potentials the two carriers are Boltzmann in [1], on nodes.

    Under Boltzmann both are psi itself, returned as the same array, so every
    device solved before Phase 5 goes down a path that computes nothing extra
    and gets the same bits.

    Under Fermi-Dirac they are psi + ln(gamma) with the two gammas, so
    n = exp(psi_eff_n - phi_n) holds exactly and the Scharfetter-Gummel
    exponential fit stays exactly valid in psi_eff. That is why degeneracy
    enters the transport equations here, inside the Bernoulli argument, rather
    than as a generalized Einstein ratio multiplying D: the two are the same
    physics, and only this one leaves the discrete current conservative. See
    docs/07-decisions.md.

    The electron potential falls below psi and the hole one rises above it, by
    30.5 mV at 1e20. The two are different arrays, so a degenerate device
    evaluates two Bernoulli pairs per edge where a Boltzmann one evaluates
    one.
    """
    if degeneracy is None:
        return psi, psi

    # Degeneracy is typed for the physical case, which is float64. Its
    # arithmetic is a polynomial and a comparison on the real part, so it is
    # dtype preserving and the complex step verification depends on that.
    # Declaring it would put complex into every signature in
    # physics/statistics.py to serve these two lines, which is the same trade
    # coupled_residual already makes for the recombination protocol.
    potential = cast("npt.NDArray[np.float64]", psi)
    electrons = cast("npt.NDArray[np.float64]", n)
    holes = cast("npt.NDArray[np.float64]", p)
    return (
        cast(
            "npt.NDArray[Number]",
            degeneracy.electron_potential(potential, electrons),
        ),
        cast("npt.NDArray[Number]", degeneracy.hole_potential(potential, holes)),
    )


def _bernoulli_pair(
    psi: npt.NDArray[Number], geometry: EdgeGeometry = UNIFORM_1D
) -> tuple[npt.NDArray[Number], npt.NDArray[Number]]:
    """(B(X), B(-X)) on every edge, with X = psi_right - psi_left [1].

    Unlike the one in continuity.py this does not force float64, because the
    residual it feeds has to survive a complex step.
    """
    X = edge_drop(psi, geometry)
    return np.asarray(B(X)), np.asarray(B(-X))


def _diffusivity_at(
    D: EdgeDiffusivity,
    psi: npt.NDArray[Number],
    h: npt.NDArray[np.float64],
    geometry: EdgeGeometry = UNIFORM_1D,
) -> Diffusivity:
    """One diffusivity per edge at this state [1].

    The drop across each edge is worked out here and the dispatch on whether
    the model needs it lives in physics/mobility.py, so this module keeps
    knowing about geometry and stays ignorant of mobility.
    """
    return diffusivity_at(D, edge_drop(psi, geometry), h)


def _diffusivity_tangent(
    D: EdgeDiffusivity,
    psi: npt.NDArray[np.float64],
    h: npt.NDArray[np.float64],
    geometry: EdgeGeometry = UNIFORM_1D,
) -> npt.NDArray[np.float64] | None:
    """dD/dX on every edge [1], or None where D does not depend on X."""
    return diffusivity_tangent(D, edge_drop(psi, geometry), h)


def _bernoulli_derivative_pair(
    psi: npt.NDArray[np.float64], geometry: EdgeGeometry = UNIFORM_1D
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """(B'(X), B'(-X)) on every edge [1].

    Real only. The Jacobian is assembled at a real state; it is the residual
    that gets differentiated, never this.
    """
    X = edge_drop(psi, geometry)
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
    Dn: EdgeDiffusivity,
    Dp: EdgeDiffusivity,
    recombination: RecombinationModel,
    geometry: EdgeGeometry = UNIFORM_1D,
    degeneracy: Degeneracy | None = None,
) -> npt.NDArray[Number]:
    """Residual of the coupled system, interleaved by node [1].

    Args:
        h: scaled edge lengths [1], length n_nodes - 1.
        volume: the part of each dual cell that holds semiconductor [1],
            scaled, length n_nodes. The whole dual cell on a device made of
            one material. It multiplies the charge term and the two
            recombination terms, which are the three things that exist only
            in silicon, and it is zero in an insulator.
        x: the interleaved unknown vector [1], length 3*n_nodes.
        net_doping: scaled net doping N = (Nd - Na)/C_0 [1], on nodes.
        Dn: scaled electron diffusivity [1], scalar or per edge.
        Dp: scaled hole diffusivity [1], scalar or per edge.
        recombination: net recombination model, in scaled units.
        geometry: which nodes each edge joins and what it carries. The
            default is the contiguous 1D chain in silicon.
        degeneracy: the statistics, or None for Boltzmann. It enters the two
            Bernoulli arguments and nothing else here, because the charge term
            carries n and p as unknowns and never substitutes them.

    Reflecting at both ends, by having no face on the outward side. Contacts
    overwrite those rows afterwards, in discretize/boundary.py.

    Preserves the dtype of x, so complex step differentiation works directly
    on this function. That is how every Jacobian block below is verified.
    """
    psi, n, p = unpack(x)
    # The protocol is typed for the physical case, which is float64. Every
    # implementation is dtype preserving and the complex step verification
    # depends on that, but declaring it in the protocol would put complex in
    # every signature in physics/recombination.py to serve this one line.
    # The cast back is the same statement in the other direction: the rate has
    # whatever dtype the densities had, and only the annotation says float64.
    R = cast(
        "npt.NDArray[Number]",
        recombination.rate(cast(Density, n), cast(Density, p)),
    )
    psi_n, psi_p = effective_potentials(psi, n, p, degeneracy)
    return _residual_from(
        h,
        volume,
        x,
        net_doping,
        _diffusivity_at(Dn, psi, h, geometry),
        _diffusivity_at(Dp, psi, h, geometry),
        _bernoulli_pair(psi_n, geometry),
        _bernoulli_pair(psi_p, geometry),
        R,
        geometry,
    )


def _residual_from(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    x: npt.NDArray[Number],
    net_doping: npt.NDArray[np.float64],
    Dn: Diffusivity,
    Dp: Diffusivity,
    bernoulli_n: tuple[npt.NDArray[Number], npt.NDArray[Number]],
    bernoulli_p: tuple[npt.NDArray[Number], npt.NDArray[Number]],
    R: npt.NDArray[Number],
    geometry: EdgeGeometry = UNIFORM_1D,
) -> npt.NDArray[Number]:
    """coupled_residual with both Bernoulli pairs and the rate in hand.

    A pair is the most expensive thing in an assembly, and the residual, the
    Jacobian and the row scales all need the same ones. Evaluating them once
    and handing them round is the pattern poisson.py and continuity.py already
    use; the coupled module was evaluating B four times per Newton step before
    this existed, which was a quarter of the assembly cost.

    Two pairs rather than one, because the two carriers see different
    effective potentials once the material is degenerate. Under Boltzmann the
    caller passes the same pair twice and nothing is computed twice.

    Private because the pairs have to be the ones belonging to this state and
    nothing outside can check that.
    """
    psi, n, p = unpack(x)
    bn_plus, bn_minus = bernoulli_n
    bp_plus, bp_minus = bernoulli_p

    out = np.zeros_like(x)
    F_psi, F_n, F_p = unpack(out)
    node_left, node_right = geometry.ends(h.size)

    # Poisson. The flux through each interior face, positive when psi falls to
    # the right, contributing with opposite sign to the two cells it separates.
    # The permittivity rides on the face, which is what makes the normal
    # component of D continuous across a material interface rather than E.
    face_flux = geometry.weight * (psi[node_left] - psi[node_right]) / h
    np.add.at(F_psi, node_left, face_flux)
    np.add.at(F_psi, node_right, -face_flux)
    F_psi -= (p - n + net_doping) * volume

    # Electron continuity. B(X) multiplies the right hand node. See the
    # docstring of discretize/continuity.py before changing that. The carrier
    # face enters here, never the permittivity and never the whole dual face:
    # see geometry.py.
    Jn = (Dn * geometry.carrier_face / h) * (
        bn_plus * n[node_right] - bn_minus * n[node_left]
    )
    F_n += R * volume
    np.add.at(F_n, node_left, -Jn)
    np.add.at(F_n, node_right, Jn)

    # Hole continuity. B(X) multiplies the left hand node, the mirror image,
    # and the divergence enters with the opposite sign.
    Jp = (Dp * geometry.carrier_face / h) * (
        bp_plus * p[node_left] - bp_minus * p[node_right]
    )
    F_p += R * volume
    np.add.at(F_p, node_left, Jp)
    np.add.at(F_p, node_right, -Jp)

    return out


# ----------------------------------------------------------------- Jacobian


class NodeRange(Enum):
    """Which nodes a block of Jacobian entries attaches to.

    Three, in any dimension: every node, and the two endpoints of every edge.
    Naming them reads better at the call site than a bare slice would, and it
    is what lets the assembly below stay legible as nine named blocks.
    """

    ALL = "all"
    """Every node. The diagonal blocks."""

    LEFT = "left"
    """Node e of edge e, for every edge."""

    RIGHT = "right"
    """Node e+1 of edge e, for every edge."""


class _Triplets:
    """A COO accumulator that names the block every entry belongs to.

    The point is that the assembly below reads as nine named blocks rather
    than as index arithmetic. Every add call says which equation and which
    unknown it is differentiating, which is the thing a reader has to be able
    to check against the residual by eye.
    """

    def __init__(
        self, n_nodes: int, geometry: EdgeGeometry = UNIFORM_1D
    ) -> None:
        node_left, node_right = geometry.ends_of(n_nodes)
        self._nodes = {
            NodeRange.ALL: np.arange(n_nodes, dtype=np.int64),
            NodeRange.LEFT: node_left,
            NodeRange.RIGHT: node_right,
        }
        self._rows: list[npt.NDArray[np.int64]] = []
        self._cols: list[npt.NDArray[np.int64]] = []
        self._values: list[npt.NDArray[np.float64]] = []

    def add(
        self,
        equation: Unknown,
        at_nodes: NodeRange,
        unknown: Unknown,
        of_nodes: NodeRange,
        values: npt.NDArray[np.float64],
    ) -> None:
        """dF_equation at at_nodes, with respect to unknown at of_nodes.

        The index arithmetic used to be cached across Newton steps, keyed on
        the node count, which an arbitrary edge list cannot be. It is done per
        call again. That was measured at ten percent of the coupled solve when
        it was first cached, so this is a real cost, knowingly paid:
        correctness in any dimension first, and docs/03-architecture.md says
        not to optimize before Phase 5.
        """
        self._rows.append(
            UNKNOWNS_PER_NODE * self._nodes[at_nodes] + int(equation)
        )
        self._cols.append(
            UNKNOWNS_PER_NODE * self._nodes[of_nodes] + int(unknown)
        )
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
    Dn: EdgeDiffusivity,
    Dp: EdgeDiffusivity,
    recombination: RecombinationModel,
    geometry: EdgeGeometry = UNIFORM_1D,
    degeneracy: Degeneracy | None = None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Jacobian of coupled_residual, in COO form.

    Returns (rows, cols, values). Written as nine named blocks, each one
    checkable against the residual above by eye, which is the whole point of
    spelling it out. Every block is verified against complex step
    differentiation in tests/unit/test_coupled.py, which phases/PHASE-3.md
    makes non-negotiable.
    """
    psi, n, p = unpack(x)
    psi_n, psi_p = effective_potentials(psi, n, p, degeneracy)
    return _jacobian_from(
        h,
        volume,
        x,
        _diffusivity_at(Dn, psi, h, geometry),
        _diffusivity_at(Dp, psi, h, geometry),
        _bernoulli_pair(psi_n, geometry),
        _bernoulli_pair(psi_p, geometry),
        _bernoulli_derivative_pair(psi_n, geometry),
        _bernoulli_derivative_pair(psi_p, geometry),
        np.asarray(recombination.d_rate_dn(n, p), dtype=np.float64),
        np.asarray(recombination.d_rate_dp(n, p), dtype=np.float64),
        geometry,
        _diffusivity_tangent(Dn, psi, h, geometry),
        _diffusivity_tangent(Dp, psi, h, geometry),
        *_potential_tangents(n, p, degeneracy),
    )


def _potential_tangents(
    n: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    degeneracy: Degeneracy | None,
) -> tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None]:
    """(d psi_eff_n/dn, d psi_eff_p/dp) on nodes [1], or None under Boltzmann.

    None rather than an array of zeros, so the Boltzmann Jacobian skips the
    terms entirely instead of adding zero to each of six blocks. It is the
    same convention dDn_dX already uses for a diffusivity that does not move
    with the field.
    """
    if degeneracy is None:
        return None, None
    return degeneracy.d_electron_potential_dn(n), degeneracy.d_hole_potential_dp(p)


def _jacobian_from(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    Dn: Diffusivity,
    Dp: Diffusivity,
    bernoulli_n: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    bernoulli_p: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    dbernoulli_n: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    dbernoulli_p: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    dR_dn: npt.NDArray[np.float64],
    dR_dp: npt.NDArray[np.float64],
    geometry: EdgeGeometry = UNIFORM_1D,
    dDn_dX: npt.NDArray[np.float64] | None = None,
    dDp_dX: npt.NDArray[np.float64] | None = None,
    dpsi_n_dn: npt.NDArray[np.float64] | None = None,
    dpsi_p_dp: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """coupled_jacobian with all four Bernoulli pairs and every tangent in hand.

    dDn_dX and dDp_dX are None unless the diffusivity depends on the potential
    drop across the edge, which is Caughey-Thomas and nothing before it.

    dpsi_n_dn and dpsi_p_dp are None unless the material is degenerate. They
    are what makes the Bernoulli argument depend on the densities as well as
    on the potential, which turns the two diagonal continuity blocks from the
    Scharfetter-Gummel stencil alone into that stencil plus the same edge
    conductance the potential block already carries.
    """
    psi, n, p = unpack(x)
    n_nodes = psi.size

    b_plus, b_minus = bernoulli_n
    db_plus, db_minus = dbernoulli_n
    bp_plus, bp_minus = bernoulli_p
    dbp_plus, dbp_minus = dbernoulli_p

    nodes = NodeRange.ALL
    left = NodeRange.LEFT
    right = NodeRange.RIGHT

    J = _Triplets(n_nodes, geometry)
    node_left, node_right = geometry.ends(h.size)

    # --- dF_psi/dpsi. The bare Laplacian. No charge term: see the module
    # docstring, this is the block the Phase 1 Jacobian would corrupt.
    conductance = geometry.weight / h
    diagonal = np.zeros(n_nodes)
    np.add.at(diagonal, node_left, conductance)
    np.add.at(diagonal, node_right, conductance)
    J.add(Unknown.PSI, nodes, Unknown.PSI, nodes, diagonal)
    J.add(Unknown.PSI, left, Unknown.PSI, right, -conductance)
    J.add(Unknown.PSI, right, Unknown.PSI, left, -conductance)

    # --- dF_psi/dn and dF_psi/dp. From -(p - n + N)*volume, diagonal only.
    J.add(Unknown.PSI, nodes, Unknown.N, nodes, volume)
    J.add(Unknown.PSI, nodes, Unknown.P, nodes, -volume)

    # --- dF_n/dpsi. Laplacian shaped, with conductance G on each edge.
    G_sg = (Dn * geometry.carrier_face / h) * (
        db_plus * n[node_right] + db_minus * n[node_left]
    )
    G = G_sg
    if dDn_dX is not None:
        # The flux carries a factor of D, so a D that moves with the drop
        # across the edge differentiates into a second term of the same shape.
        # Written as the flux over D rather than as a logarithm, so nothing
        # divides by a diffusivity that is allowed to become small.
        G = G + (dDn_dX * geometry.carrier_face / h) * (
            b_plus * n[node_right] - b_minus * n[node_left]
        )
    diagonal = np.zeros(n_nodes)
    np.add.at(diagonal, node_left, G)
    np.add.at(diagonal, node_right, G)
    J.add(Unknown.N, nodes, Unknown.PSI, nodes, diagonal)
    J.add(Unknown.N, left, Unknown.PSI, right, -G)
    J.add(Unknown.N, right, Unknown.PSI, left, -G)

    # --- dF_n/dn. The Scharfetter-Gummel stencil plus the exact SRH tangent.
    to_right = (Dn * geometry.carrier_face / h) * b_plus
    to_left = (Dn * geometry.carrier_face / h) * b_minus
    if dpsi_n_dn is not None:
        # The Bernoulli argument is the drop in the effective potential, and
        # that depends on the density at each end as well as on psi. So the
        # same edge conductance the dF_n/dpsi block carries reappears here,
        # once per end, weighted by how far the effective potential moves per
        # electron. Only the Scharfetter-Gummel part of it: a field dependent
        # diffusivity reads the real potential drop and does not move when a
        # density does.
        to_right = to_right + G_sg * dpsi_n_dn[node_right]
        to_left = to_left + G_sg * dpsi_n_dn[node_left]
    diagonal = dR_dn * volume
    np.add.at(diagonal, node_left, to_left)
    np.add.at(diagonal, node_right, to_right)
    J.add(Unknown.N, nodes, Unknown.N, nodes, diagonal)
    J.add(Unknown.N, left, Unknown.N, right, -to_right)
    J.add(Unknown.N, right, Unknown.N, left, -to_left)

    # --- dF_n/dp. R is a point function, so this touches one node only.
    J.add(Unknown.N, nodes, Unknown.P, nodes, dR_dp * volume)

    # --- dF_p/dpsi. The same stencil as the electron block with the opposite
    # sign, because Jp enters its residual with the opposite sign.
    H_sg = (Dp * geometry.carrier_face / h) * (
        dbp_plus * p[node_left] + dbp_minus * p[node_right]
    )
    H = H_sg
    if dDp_dX is not None:
        H = H + (dDp_dX * geometry.carrier_face / h) * (
            bp_plus * p[node_left] - bp_minus * p[node_right]
        )
    diagonal = np.zeros(n_nodes)
    np.add.at(diagonal, node_left, -H)
    np.add.at(diagonal, node_right, -H)
    J.add(Unknown.P, nodes, Unknown.PSI, nodes, diagonal)
    J.add(Unknown.P, left, Unknown.PSI, right, H)
    J.add(Unknown.P, right, Unknown.PSI, left, H)

    # --- dF_p/dn.
    J.add(Unknown.P, nodes, Unknown.N, nodes, dR_dn * volume)

    # --- dF_p/dp. The mirror of the electron block: the two flux coefficients
    # have swapped nodes, for the same reason the fluxes do.
    to_left = (Dp * geometry.carrier_face / h) * bp_plus
    to_right = (Dp * geometry.carrier_face / h) * bp_minus
    if dpsi_p_dp is not None:
        # The mirror of the electron block. The sign is opposite because the
        # hole effective potential rises where the electron one falls.
        to_left = to_left - H_sg * dpsi_p_dp[node_left]
        to_right = to_right - H_sg * dpsi_p_dp[node_right]
    diagonal = dR_dp * volume
    np.add.at(diagonal, node_left, to_left)
    np.add.at(diagonal, node_right, to_right)
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
    Dn: EdgeDiffusivity,
    Dp: EdgeDiffusivity,
    degeneracy: Degeneracy | None = None,
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

    x: npt.NDArray[np.float64] = pack(psi.data, n.data, p.data)
    return assemble_coupled_arrays(
        h=mesh.h / scale.x_0,
        volume=mesh.volume / scale.x_0,
        x=x,
        net_doping=net_doping.data,
        Dn=Dn,
        Dp=Dp,
        recombination=recombination,
        degeneracy=degeneracy,
    )


def assemble_coupled_arrays(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    net_doping: npt.NDArray[np.float64],
    Dn: EdgeDiffusivity,
    Dp: EdgeDiffusivity,
    recombination: RecombinationModel,
    degeneracy: Degeneracy | None = None,
) -> SparseAssembly:
    """assemble_coupled with the scaling and location already checked.

    The Field level checks belong once at the entry to a solve, not once per
    iteration, and building three Fields per step only to unwrap them again is
    work a solver does not need.

    Unscaled. A Newton loop wants assemble_coupled_scaled instead; this is the
    form the block verification checks and the form the Field level wrapper
    returns.
    """
    return assemble_coupled_terms(
        h, volume, x, net_doping, Dn, Dp, recombination, degeneracy=degeneracy
    ).assembly


class CoupledAssembly(NamedTuple):
    """An assembled coupled system and the term scales of the same state."""

    assembly: SparseAssembly
    """Residual and Jacobian, unscaled."""

    scales: TermScales
    """Per node term scale for the psi, n and p rows at this state [1]."""


def assemble_coupled_terms(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    net_doping: npt.NDArray[np.float64],
    Dn: EdgeDiffusivity,
    Dp: EdgeDiffusivity,
    recombination: RecombinationModel,
    geometry: EdgeGeometry = UNIFORM_1D,
    degeneracy: Degeneracy | None = None,
) -> CoupledAssembly:
    """Residual, Jacobian and term scales, with the shared work done once.

    What a Newton loop calls. The residual, the Jacobian and the scales all
    want the same Bernoulli pair and the same recombination rate at the same
    state, and going through the three public functions separately evaluates B
    four times per Newton step and the SRH denominator three times. Measured,
    that was a quarter of the cost of a coupled solve.

    Returns the assembly unscaled and the scales beside it, rather than a
    scaled assembly, because the contacts have to be applied in between: a
    pinned row has to become the identity and then be divided like every other
    row. Scaling first would leave the pinned rows at one while everything
    around them moved.
    """
    psi, n, p = unpack(x)

    psi_n, psi_p = effective_potentials(psi, n, p, degeneracy)
    bernoulli_n = _bernoulli_pair(psi_n, geometry)
    bernoulli_p = (
        bernoulli_n if degeneracy is None else _bernoulli_pair(psi_p, geometry)
    )
    dbernoulli_n = _bernoulli_derivative_pair(psi_n, geometry)
    dbernoulli_p = (
        dbernoulli_n
        if degeneracy is None
        else _bernoulli_derivative_pair(psi_p, geometry)
    )
    dpsi_n_dn, dpsi_p_dp = _potential_tangents(n, p, degeneracy)
    R = np.asarray(recombination.rate(n, p), dtype=np.float64)
    dR_dn = np.asarray(recombination.d_rate_dn(n, p), dtype=np.float64)
    dR_dp = np.asarray(recombination.d_rate_dp(n, p), dtype=np.float64)

    # A field dependent diffusivity is a function of this state, so it is
    # evaluated here with everything else that is, and once rather than three
    # times: the residual, the Jacobian and the term scales all want the same
    # one, for the same reason they all want the same Bernoulli pair.
    Dn_edge = _diffusivity_at(Dn, psi, h, geometry)
    Dp_edge = _diffusivity_at(Dp, psi, h, geometry)
    dDn_dX = _diffusivity_tangent(Dn, psi, h, geometry)
    dDp_dX = _diffusivity_tangent(Dp, psi, h, geometry)

    residual = _residual_from(
        h,
        volume,
        x,
        net_doping,
        Dn_edge,
        Dp_edge,
        bernoulli_n,
        bernoulli_p,
        R,
        geometry,
    )
    rows, cols, values = _jacobian_from(
        h,
        volume,
        x,
        Dn_edge,
        Dp_edge,
        bernoulli_n,
        bernoulli_p,
        dbernoulli_n,
        dbernoulli_p,
        dR_dn,
        dR_dp,
        geometry,
        dDn_dX,
        dDp_dX,
        dpsi_n_dn,
        dpsi_p_dp,
    )
    scales = _term_scales_from(
        h,
        volume,
        x,
        net_doping,
        Dn_edge,
        Dp_edge,
        bernoulli_n,
        bernoulli_p,
        R,
        geometry,
    )

    size = x.size
    return CoupledAssembly(
        assembly=SparseAssembly(
            residual=np.asarray(residual, dtype=np.float64),
            rows=rows,
            cols=cols,
            values=values,
            shape=(size, size),
        ),
        scales=scales,
    )


def limit_psi_step(
    delta: npt.NDArray[np.float64], max_psi_step: float
) -> npt.NDArray[np.float64]:
    """Cap the potential update and take the density updates in full.

    docs/02-numerics.md prescribes exactly this: dpsi_max of 5*V_T per Newton
    step, which is 5.0 in scaled units, with the full n and p updates
    accepted. docs/05-pitfalls.md says the same thing from the other side,
    that damping the density updates slows convergence without buying
    robustness.

    The psi sub-vector is scaled by one factor rather than clipped entry by
    entry, so the potential update keeps its own direction. Between psi and
    the densities the direction does rotate, and that is deliberate: they
    differ by six decades in scaled units, so one factor over the whole vector
    would be set entirely by the density update and would leave psi frozen.

    Returns the argument itself when nothing needs capping, so that
    newton_solve does not count an inactive limiter as a limited step.
    """
    dpsi = delta[Unknown.PSI :: UNKNOWNS_PER_NODE]
    peak = float(np.max(np.abs(dpsi)))
    if peak <= max_psi_step:
        return delta

    limited = delta.copy()
    limited[Unknown.PSI :: UNKNOWNS_PER_NODE] = dpsi * (max_psi_step / peak)
    return limited


# ------------------------------------------------------------------ contacts


def apply_contacts_coupled(
    assembly: SparseAssembly,
    x: npt.NDArray[np.float64],
    net_doping: npt.NDArray[np.float64],
    contacts: Sequence[Contact],
    scale: ScaleFactors,
    carrier_free_nodes: Sequence[int] = (),
    T: float = C.T_ROOM,
    degeneracy: Degeneracy | None = None,
) -> SparseAssembly:
    """Pin every contact on a device, of whatever kind, returning a new assembly.

    Args:
        assembly: the assembled coupled system.
        x: the current interleaved unknown vector [1].
        net_doping: scaled net doping on nodes [1].
        contacts: the contacts to apply, ohmic points, ohmic plates and gates.
            A point contact pins one node and a plate pins every node it
            covers, which is the same statement: both are asked for their
            `nodes`.
        scale: scale factors, used to convert contact voltages from V.
        carrier_free_nodes: nodes with no semiconductor in them, whose n and
            p rows are singular and have to be pinned. See below.
        T: temperature [K], which the gate potential needs for the band gap.
        degeneracy: the statistics, or None for Boltzmann. All three contact
            values come from it together, which is what keeps psi, n and p at
            a contact node one state rather than three near agreements.

    Three Dirichlet conditions per ohmic contact node rather than one. In the
    uncoupled solve the potential and the two densities are pinned in three
    separate systems, by three separate calls; here they are three unknowns of
    one system and go in together.

    **A gate pins one, not three.** It is metal on an insulator, so there is
    no doping under it to solve neutrality against and no carrier population
    to hold. Its potential comes from its own work function through
    gate_psi_scaled, and its two density rows are already spoken for: gate
    nodes sit on the oxide, so they arrive in carrier_free_nodes and are
    pinned at zero there. Pinning them here as well would hand
    apply_dirichlet_nodes the same unknown twice, which it refuses, and that
    refusal is the check that the two halves agree about which nodes are
    metal.

    The values are the same ones the uncoupled path uses, and they have to be,
    or the two solvers would answer different problems and their agreement at
    every bias would mean nothing. psi carries the applied bias, both
    densities are at their equilibrium values whatever the terminal voltage,
    which is what makes an ohmic contact a perfect sink.

    **Why an insulator node needs pinning at all.** Its Poisson row is a
    perfectly good Laplacian and wants no help. Its two continuity rows are
    another matter: the charge volume there is zero, so the recombination term
    goes, and the carrier face there is zero, so every flux goes too. What is
    left is the row 0 = 0, with nothing on the diagonal, and a singular matrix
    is not something to discover inside a factorization. Pinning both
    densities at zero says the true thing, that an insulator holds no free
    carriers, and leaves the potential alone.

    apply_dirichlet_nodes does the work and needs nothing taught about the
    coupling: it takes unknown indices, not node indices, so the contact
    simply hands it three indices per node. It eliminates the column as well
    as the row, which is what makes the pinned value come back exactly rather
    than to within the conditioning of the whole system. See its docstring for
    the measurement that forced that. Everything goes in one call, so a
    contact sitting on an insulator node is refused there rather than resolved
    by whichever pass went last.
    """
    names = [contact.name for contact in contacts]
    if len(set(names)) != len(names):
        raise ValueError(f"contact names must be unique, got {names}")

    indices: list[int] = []
    targets: list[float] = []

    for contact in contacts:
        applied = contact.voltage / scale.psi_0

        if isinstance(contact, GateContact):
            target = gate_psi_scaled(applied, contact.work_function, T)
            for node in contact.nodes:
                indices.append(unknown_index(node, Unknown.PSI))
                targets.append(target)
            continue

        for node in contact.nodes:
            doping = float(net_doping[node])

            indices.append(unknown_index(node, Unknown.PSI))
            targets.append(ohmic_psi_scaled(doping, applied, degeneracy))

            indices.append(unknown_index(node, Unknown.N))
            targets.append(
                ohmic_density_scaled(doping, Carrier.ELECTRON, degeneracy)
            )

            indices.append(unknown_index(node, Unknown.P))
            targets.append(ohmic_density_scaled(doping, Carrier.HOLE, degeneracy))

    for node in carrier_free_nodes:
        indices.append(unknown_index(node, Unknown.N))
        targets.append(0.0)
        indices.append(unknown_index(node, Unknown.P))
        targets.append(0.0)

    return apply_dirichlet_nodes(assembly, x, indices, targets)


# ------------------------------------------------------------- row scaling


def residual_term_scales(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    net_doping: npt.NDArray[np.float64],
    Dn: EdgeDiffusivity,
    Dp: EdgeDiffusivity,
    R: npt.NDArray[np.float64] | None = None,
    geometry: EdgeGeometry = UNIFORM_1D,
    degeneracy: Degeneracy | None = None,
) -> TermScales:
    """The size of the terms each row is assembled from [1], one per node.

    Args:
        h: scaled edge lengths [1].
        volume: the part of each dual cell that holds semiconductor [1],
            scaled. Zero in an insulator. See coupled_residual.
        x: the interleaved unknown vector [1].
        net_doping: scaled net doping on nodes [1].
        Dn: scaled electron diffusivity [1].
        Dp: scaled hole diffusivity [1].
        R: scaled net recombination rate on nodes [1]. None leaves it out,
            which only ever lowers a scale and is right for a caller that has
            not built a model. A solve always passes it.

    Returns (psi, n, p). docs/02-numerics.md asks for a residual threshold
    built from exactly this, and for a scale that does not depend on the
    starting iterate.

    **That means it must not depend on how converged the start is. It does not
    mean freezing it at the guess, and freezing it at the guess is wrong.** The
    terms a residual is built from are a property of the state, and on a
    forward biased junction they grow with the injected density. Measured on a
    1e16 diode at 1 V the electron term scale is 28 times larger at the answer
    than at the equilibrium guess; on a 1e20 / 1e14 junction it is 660000 times
    larger, because the minority electron density on the heavily doped side is
    injected up by exp(V/V_T). A scale frozen at the guess describes a
    different problem from the one being solved, and it made that device
    report failure at a residual of 2.8e-9 while it was in fact converged to
    4.5e-15 against the terms it actually had.

    So a coupled solve re-evaluates this at every iterate. The threshold itself
    stays fixed, at residual_rtol against a scale of one, so nothing drifts:
    what is held constant is the question being asked, which is how large the
    residual is next to the terms it is currently made of.

    Why three numbers and not one. The Poisson residual is a charge, of order
    N*volume, and the continuity residuals are currents, of order (D/h)*n.
    On a 1e16 diode in scaled units those differ by six decades. A single
    threshold over the whole vector is set by the larger one, and then the
    Poisson equation is declared converged at a residual a million times above
    its own floor. Every family is measured against its own terms instead.

    Each scale is the largest single term that goes into the sum, not the
    largest sum. A residual is a difference of terms and cannot be resolved
    below machine epsilon times the things being differenced, so the terms are
    what sets the floor.
    """
    psi, n, p = unpack(x)
    psi_n, psi_p = effective_potentials(psi, n, p, degeneracy)
    return _term_scales_from(
        h,
        volume,
        x,
        net_doping,
        _diffusivity_at(Dn, psi, h, geometry),
        _diffusivity_at(Dp, psi, h, geometry),
        _bernoulli_pair(psi_n, geometry),
        _bernoulli_pair(psi_p, geometry),
        R,
        geometry,
    )


def _term_scales_from(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    net_doping: npt.NDArray[np.float64],
    Dn: Diffusivity,
    Dp: Diffusivity,
    bernoulli_n: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    bernoulli_p: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    R: npt.NDArray[np.float64] | None,
    geometry: EdgeGeometry = UNIFORM_1D,
) -> TermScales:
    """residual_term_scales with both Bernoulli pairs already in hand."""
    psi, n, p = unpack(x)
    bn_plus, bn_minus = bernoulli_n
    bp_plus, bp_minus = bernoulli_p
    node_left, node_right = geometry.ends(h.size)
    n_nodes = volume.size

    # Poisson: the face fluxes psi/h, and the three charges that make up
    # -(p - n + N)*volume. All three, not only the doping. On intrinsic
    # material the sum p - n + N is exactly zero while n*volume and p*volume
    # are a whole dual cell each, and a scale built from the doping alone
    # reports zero there and makes every row nan on division. Measured before
    # this counted the carriers: an undoped 41 node bar returned a residual of
    # nan and the message blamed the LU factorization for being singular.
    psi_edge = (geometry.weight / h) * np.maximum(
        np.abs(psi[node_left]), np.abs(psi[node_right])
    )
    psi_scale = np.maximum(
        _largest_at_each_node(psi_edge, node_left, node_right, n_nodes),
        (np.abs(p) + np.abs(n) + np.abs(net_doping)) * volume,
    )

    # Continuity: the two halves of each Scharfetter-Gummel flux, and the
    # recombination that sits alongside them in the same row.
    recombined = (
        np.zeros(n_nodes) if R is None else np.abs(R) * volume
    )
    gn = Dn * geometry.carrier_face / h
    gp = Dp * geometry.carrier_face / h
    electron_scale = np.maximum(
        _largest_at_each_node(
            np.maximum(gn * bn_plus * n[node_right], gn * bn_minus * n[node_left]),
            node_left,
            node_right,
            n_nodes,
        ),
        recombined,
    )
    hole_scale = np.maximum(
        _largest_at_each_node(
            np.maximum(gp * bp_plus * p[node_left], gp * bp_minus * p[node_right]),
            node_left,
            node_right,
            n_nodes,
        ),
        recombined,
    )

    scales = (psi_scale, electron_scale, hole_scale)
    if not all(
        np.any(scale > 0.0) and bool(np.all(np.isfinite(scale))) for scale in scales
    ):
        raise ValueError(
            f"the state has no terms to measure a residual against: term "
            f"scales max to "
            f"{tuple(float(np.max(scale)) for scale in scales)} for (psi, n, p). "
            "Every equation is identically zero, which a device never is. "
            "Dividing by these would rename the problem as a singular matrix "
            "three call frames later."
        )
    return scales


def _largest_at_each_node(
    edge_term: npt.NDArray[np.float64],
    node_left: npt.NDArray[np.int64],
    node_right: npt.NDArray[np.int64],
    n_nodes: int,
) -> npt.NDArray[np.float64]:
    """The largest incident edge term at every node [1].

    A node's equation sums the terms on the edges that touch it, so those are
    the terms its own residual is a difference of, and the largest of them is
    what sets the floor that residual can be resolved against.
    """
    largest = np.zeros(n_nodes)
    np.maximum.at(largest, node_left, edge_term)
    np.maximum.at(largest, node_right, edge_term)
    return largest


def row_weights(scales: TermScales, n_nodes: int) -> npt.NDArray[np.float64]:
    """One weight per unknown, for the preconditioner [1]: per family.

    The largest term anywhere in each family, broadcast over every row of it.
    That is deliberately not the per row scale `residual_measure` uses, and
    the two are not interchangeable, because they are answering different
    questions.

    A row of dF_n/dn holds edge conductances of size D*face/h. The residual on
    that row is a difference of terms of size D*face/h*n. The two differ by n
    itself, which spans fourteen decades across a depleted device. Dividing
    the Jacobian by the per row residual scale therefore leaves row entries of
    size 1/n and hands the factorization a matrix fourteen decades worse
    conditioned than the one it started with. Measured on the 41 node MOS
    capacitor driven into accumulation: the Newton step came out 81 percent
    different from the correctly scaled one and the solve walked off to a
    potential of -27 V.

    So the preconditioner keeps one number per family, which is flat across
    the rows and cannot do that, and the convergence test gets its own
    measure.
    """
    weights = np.empty(UNKNOWNS_PER_NODE * n_nodes)
    for component, scale in zip(Unknown, scales, strict=True):
        if scale.size != n_nodes:
            raise ValueError(
                f"the {component.name} term scale has {scale.size} entries "
                f"for a mesh of {n_nodes} nodes"
            )
        weights[component::UNKNOWNS_PER_NODE] = np.max(scale)
    return weights


def residual_measure(
    residual: npt.NDArray[np.float64], scales: TermScales, n_nodes: int
) -> float:
    """How large a scaled residual is against the terms of its own row [1].

    Args:
        residual: the residual of the assembly newton_solve is driving, which
            is the one `scale_rows` has already divided by `row_weights`.
        scales: the per node term scales of the same state.
        n_nodes: mesh node count.

    The number a coupled solve tests for convergence. `row_weights` divided
    every row of a family by one number, so the residual arrives measured
    against the largest terms anywhere in that family rather than against its
    own. Multiplying that back out and dividing by the row's own terms is what
    makes max |F| mean the same thing in every row.

    Why it has to. On a 1e17 / 1e20 junction the electron flux terms span
    twelve decades between the two sides, so a row in the lightly doped side
    is divided by something set on the degenerate side and lands twelve
    decades below any threshold whatever its own residual is doing. Measured:
    a cold solve at 0.4 V reported convergence after zero Newton steps, still
    sitting on the equilibrium guess, with a terminal current of 1.2e-10
    against the 7.4e-4 the Gummel path gives. Under this measure the same
    guess reads 1.4e-3 and the answer reads 4.2e-16.

    A row with no terms in it at all is skipped rather than divided by zero.
    The only rows that has ever meant are the two continuity rows of a node
    holding no semiconductor, which carry no flux and no recombination because
    there is nothing there to carry them. They are pinned to the identity, so
    their residual is the pinning error and goes to exactly zero in one step
    whatever it is measured against.
    """
    weights = row_weights(scales, n_nodes)
    raw = np.abs(residual) * weights
    largest = 0.0
    for component, scale in zip(Unknown, scales, strict=True):
        rows = raw[component::UNKNOWNS_PER_NODE]
        measured = np.divide(
            rows, scale, out=np.zeros_like(rows), where=scale > 0.0
        )
        largest = max(largest, float(np.max(measured)))
    return largest


def scale_rows(
    assembly: SparseAssembly, weights: npt.NDArray[np.float64]
) -> SparseAssembly:
    """Divide every equation by its own term scale, returning a new assembly.

    A diagonal left preconditioner. J*dx = -F row-divided by w is the same
    linear system with the same solution, so this changes no answer; it makes
    max |F| a number that means the same thing in every row, and it takes six
    decades out of the row norms, which the factorization is happier with.

    Applied outside assemble_coupled_arrays rather than inside it, so that the
    Jacobian the block verification checks is the unweighted one.
    """
    return SparseAssembly(
        residual=assembly.residual / weights,
        rows=assembly.rows,
        cols=assembly.cols,
        values=assembly.values / weights[assembly.rows],
        shape=assembly.shape,
    )


def coupled_update_norm(
    delta: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> float:
    """Size of a coupled Newton update, measured per family [1].

        max( |dpsi|,  |dn|/(n + 1),  |dp|/(p + 1) )

    docs/02-numerics.md asks for the potential update absolutely and the
    carrier change as max |dn| / (n + n_i), which is n + 1 in scaled units
    with C_0 = n_i. The three are combined by taking the largest, so one
    threshold still means one thing.

    Absolute for psi and relative for the densities is not an inconsistency.
    psi is a logarithmic quantity already, order ten across a whole device, so
    an absolute change in it is a relative change in everything it drives. The
    densities run over twenty five decades and an absolute change in them
    means nothing at all.

    The floor at 1 matters as much as the ratio. Without it the measure is
    dominated by nodes where the density is 1e-15 and physically irrelevant,
    and a solve that has converged everywhere that carries charge reports a
    huge update from a node that carries none.

    max |dx| over the raw vector, which is what newton_solve does by default,
    cannot work here. Measured on a 1e16 diode at 1 V: the residual reaches
    1.8e-16 at step eight and the raw update sits at 8.9e-10 forever, because
    n is 1e6 in scaled units and its last bit is 1e-10. Every solve above
    0.2 V reported failure while sitting on the exact answer.
    """
    dpsi, dn, dp = unpack(delta)
    _, n, p = unpack(x)

    return max(
        float(np.max(np.abs(dpsi))),
        float(np.max(np.abs(dn) / (np.abs(n) + 1.0))),
        float(np.max(np.abs(dp) / (np.abs(p) + 1.0))),
    )
