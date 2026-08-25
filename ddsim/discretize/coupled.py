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

import math
from enum import Enum, IntEnum
from typing import NamedTuple, TypeVar, cast

import numpy as np
import numpy.typing as npt

from ddsim.core.field import Field, Location, ScalingState
from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.assembly import SparseAssembly
from ddsim.discretize.boundary import (
    Carrier,
    OhmicContact,
    apply_dirichlet_nodes,
    ohmic_density_scaled,
    ohmic_psi_scaled,
)
from ddsim.discretize.continuity import Diffusivity
from ddsim.discretize.geometry import UNIFORM_1D, EdgeGeometry
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
    psi: npt.NDArray[Number], geometry: EdgeGeometry = UNIFORM_1D
) -> tuple[npt.NDArray[Number], npt.NDArray[Number]]:
    """(B(X), B(-X)) on every edge, with X = psi_right - psi_left [1].

    Unlike the one in continuity.py this does not force float64, because the
    residual it feeds has to survive a complex step.
    """
    node_left, node_right = geometry.ends_of(psi.size)
    X = psi[node_right] - psi[node_left]
    return np.asarray(B(X)), np.asarray(B(-X))


def _bernoulli_derivative_pair(
    psi: npt.NDArray[np.float64], geometry: EdgeGeometry = UNIFORM_1D
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """(B'(X), B'(-X)) on every edge [1].

    Real only. The Jacobian is assembled at a real state; it is the residual
    that gets differentiated, never this.
    """
    node_left, node_right = geometry.ends_of(psi.size)
    X = psi[node_right] - psi[node_left]
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
    geometry: EdgeGeometry = UNIFORM_1D,
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
        geometry: which nodes each edge joins and what it carries. The
            default is the contiguous 1D chain in silicon.

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
    return _residual_from(
        h, volume, x, net_doping, Dn, Dp, _bernoulli_pair(psi, geometry), R,
        geometry,
    )


def _residual_from(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    x: npt.NDArray[Number],
    net_doping: npt.NDArray[np.float64],
    Dn: Diffusivity,
    Dp: Diffusivity,
    bernoulli: tuple[npt.NDArray[Number], npt.NDArray[Number]],
    R: npt.NDArray[Number],
    geometry: EdgeGeometry = UNIFORM_1D,
) -> npt.NDArray[Number]:
    """coupled_residual with the Bernoulli pair and the rate already in hand.

    The pair is the most expensive thing in an assembly, and the residual, the
    Jacobian and the row scales all need the same one. Evaluating it once and
    handing it round is the pattern poisson.py and continuity.py already use;
    the coupled module was evaluating B four times per Newton step before this
    existed, which was a quarter of the assembly cost.

    Private because the pair has to be the one belonging to this psi and
    nothing outside can check that.
    """
    psi, n, p = unpack(x)
    b_plus, b_minus = bernoulli

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
    # docstring of discretize/continuity.py before changing that. Only the
    # dual face area enters here, never the permittivity: see geometry.py.
    Jn = (Dn * geometry.dual_face / h) * (
        b_plus * n[node_right] - b_minus * n[node_left]
    )
    F_n += R * volume
    np.add.at(F_n, node_left, -Jn)
    np.add.at(F_n, node_right, Jn)

    # Hole continuity. B(X) multiplies the left hand node, the mirror image,
    # and the divergence enters with the opposite sign.
    Jp = (Dp * geometry.dual_face / h) * (
        b_plus * p[node_left] - b_minus * p[node_right]
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
    Dn: Diffusivity,
    Dp: Diffusivity,
    recombination: RecombinationModel,
    geometry: EdgeGeometry = UNIFORM_1D,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Jacobian of coupled_residual, in COO form.

    Returns (rows, cols, values). Written as nine named blocks, each one
    checkable against the residual above by eye, per the working agreement in
    CLAUDE.md. Every block is verified against complex step differentiation in
    tests/unit/test_coupled.py, which phases/PHASE-3.md makes non-negotiable.
    """
    psi, n, p = unpack(x)
    return _jacobian_from(
        h,
        volume,
        x,
        Dn,
        Dp,
        _bernoulli_pair(psi, geometry),
        _bernoulli_derivative_pair(psi, geometry),
        np.asarray(recombination.d_rate_dn(n, p), dtype=np.float64),
        np.asarray(recombination.d_rate_dp(n, p), dtype=np.float64),
        geometry,
    )


def _jacobian_from(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    Dn: Diffusivity,
    Dp: Diffusivity,
    bernoulli: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    dbernoulli: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    dR_dn: npt.NDArray[np.float64],
    dR_dp: npt.NDArray[np.float64],
    geometry: EdgeGeometry = UNIFORM_1D,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """coupled_jacobian with both Bernoulli pairs and both tangents in hand."""
    psi, n, p = unpack(x)
    n_nodes = psi.size

    b_plus, b_minus = bernoulli
    db_plus, db_minus = dbernoulli

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
    G = (Dn * geometry.dual_face / h) * (
        db_plus * n[node_right] + db_minus * n[node_left]
    )
    diagonal = np.zeros(n_nodes)
    np.add.at(diagonal, node_left, G)
    np.add.at(diagonal, node_right, G)
    J.add(Unknown.N, nodes, Unknown.PSI, nodes, diagonal)
    J.add(Unknown.N, left, Unknown.PSI, right, -G)
    J.add(Unknown.N, right, Unknown.PSI, left, -G)

    # --- dF_n/dn. The Scharfetter-Gummel stencil plus the exact SRH tangent.
    to_right = (Dn * geometry.dual_face / h) * b_plus
    to_left = (Dn * geometry.dual_face / h) * b_minus
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
    H = (Dp * geometry.dual_face / h) * (
        db_plus * p[node_left] + db_minus * p[node_right]
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
    to_left = (Dp * geometry.dual_face / h) * b_plus
    to_right = (Dp * geometry.dual_face / h) * b_minus
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

    x: npt.NDArray[np.float64] = pack(psi.data, n.data, p.data)
    return assemble_coupled_arrays(
        h=mesh.h / scale.x_0,
        volume=mesh.volume / scale.x_0,
        x=x,
        net_doping=net_doping.data,
        Dn=Dn,
        Dp=Dp,
        recombination=recombination,
    )


def assemble_coupled_arrays(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    net_doping: npt.NDArray[np.float64],
    Dn: Diffusivity,
    Dp: Diffusivity,
    recombination: RecombinationModel,
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
        h, volume, x, net_doping, Dn, Dp, recombination
    ).assembly


class CoupledAssembly(NamedTuple):
    """An assembled coupled system and the term scales of the same state."""

    assembly: SparseAssembly
    """Residual and Jacobian, unscaled."""

    scales: tuple[float, float, float]
    """Term scale for the psi, n and p families at this state [1]."""


def assemble_coupled_terms(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    net_doping: npt.NDArray[np.float64],
    Dn: Diffusivity,
    Dp: Diffusivity,
    recombination: RecombinationModel,
    geometry: EdgeGeometry = UNIFORM_1D,
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

    bernoulli = _bernoulli_pair(psi, geometry)
    dbernoulli = _bernoulli_derivative_pair(psi, geometry)
    R = np.asarray(recombination.rate(n, p), dtype=np.float64)
    dR_dn = np.asarray(recombination.d_rate_dn(n, p), dtype=np.float64)
    dR_dp = np.asarray(recombination.d_rate_dp(n, p), dtype=np.float64)

    residual = _residual_from(
        h, volume, x, net_doping, Dn, Dp, bernoulli, R, geometry
    )
    rows, cols, values = _jacobian_from(
        h, volume, x, Dn, Dp, bernoulli, dbernoulli, dR_dn, dR_dp, geometry
    )
    scales = _term_scales_from(
        h, volume, x, net_doping, Dn, Dp, bernoulli, R, geometry
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


def apply_ohmic_contacts_coupled(
    assembly: SparseAssembly,
    x: npt.NDArray[np.float64],
    net_doping: npt.NDArray[np.float64],
    contacts: tuple[OhmicContact, ...],
    scale: ScaleFactors,
) -> SparseAssembly:
    """Pin psi, n and p at every ohmic contact, returning a new assembly.

    Args:
        assembly: the assembled coupled system.
        x: the current interleaved unknown vector [1].
        net_doping: scaled net doping on nodes [1].
        contacts: the contacts to apply.
        scale: scale factors, used to convert contact voltages from V.

    Three Dirichlet conditions per contact rather than one. In the uncoupled
    solve the potential and the two densities are pinned in three separate
    systems, by three separate calls; here they are three unknowns of one
    system and go in together.

    The values are the same ones the uncoupled path uses, and they have to be,
    or the two solvers would answer different problems and their agreement at
    every bias would mean nothing. psi carries the applied bias, both
    densities are at their equilibrium values whatever the terminal voltage,
    which is what makes an ohmic contact a perfect sink.

    apply_dirichlet_nodes does the work and needs nothing taught about the
    coupling: it takes unknown indices, not node indices, so the contact
    simply hands it three indices per node. It eliminates the column as well
    as the row, which is what makes the pinned value come back exactly rather
    than to within the conditioning of the whole system. See its docstring for
    the measurement that forced that.
    """
    names = [contact.name for contact in contacts]
    if len(set(names)) != len(names):
        raise ValueError(f"contact names must be unique, got {names}")

    indices: list[int] = []
    targets: list[float] = []

    for contact in contacts:
        doping = float(net_doping[contact.node])
        applied = contact.voltage / scale.psi_0

        indices.append(unknown_index(contact.node, Unknown.PSI))
        targets.append(ohmic_psi_scaled(doping, applied))

        indices.append(unknown_index(contact.node, Unknown.N))
        targets.append(ohmic_density_scaled(doping, Carrier.ELECTRON))

        indices.append(unknown_index(contact.node, Unknown.P))
        targets.append(ohmic_density_scaled(doping, Carrier.HOLE))

    return apply_dirichlet_nodes(assembly, x, indices, targets)


# ------------------------------------------------------------- row scaling


def residual_term_scales(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    net_doping: npt.NDArray[np.float64],
    Dn: Diffusivity,
    Dp: Diffusivity,
    R: npt.NDArray[np.float64] | None = None,
    geometry: EdgeGeometry = UNIFORM_1D,
) -> tuple[float, float, float]:
    """The size of the terms each equation family is assembled from [1].

    Args:
        h: scaled edge lengths [1].
        volume: scaled dual cell widths [1].
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
    psi, _, _ = unpack(x)
    return _term_scales_from(
        h, volume, x, net_doping, Dn, Dp, _bernoulli_pair(psi, geometry), R,
        geometry,
    )


def _term_scales_from(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    net_doping: npt.NDArray[np.float64],
    Dn: Diffusivity,
    Dp: Diffusivity,
    bernoulli: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    R: npt.NDArray[np.float64] | None,
    geometry: EdgeGeometry = UNIFORM_1D,
) -> tuple[float, float, float]:
    """residual_term_scales with the Bernoulli pair already in hand."""
    psi, n, p = unpack(x)
    b_plus, b_minus = bernoulli
    node_left, node_right = geometry.ends(h.size)

    # Poisson: the face fluxes psi/h, and the three charges that make up
    # -(p - n + N)*volume. All three, not only the doping. On intrinsic
    # material the sum p - n + N is exactly zero while n*volume and p*volume
    # are a whole dual cell each, and a scale built from the doping alone
    # reports zero there and makes every row nan on division. Measured before
    # this counted the carriers: an undoped 41 node bar returned a residual of
    # nan and a message blaming the LU factorization for being singular.
    psi_scale = max(
        float(np.max(np.abs(psi)) * np.max(geometry.weight / h)),
        float(np.max((np.abs(p) + np.abs(n) + np.abs(net_doping)) * volume)),
    )

    # Continuity: the two halves of each Scharfetter-Gummel flux, and the
    # recombination that sits alongside them in the same row.
    recombined = 0.0 if R is None else float(np.max(np.abs(R) * volume))
    electron_scale = max(
        float(
            np.max(
                np.maximum(
                    (Dn * geometry.dual_face / h) * b_plus * n[node_right],
                    (Dn * geometry.dual_face / h) * b_minus * n[node_left],
                )
            )
        ),
        recombined,
    )
    hole_scale = max(
        float(
            np.max(
                np.maximum(
                    (Dp * geometry.dual_face / h) * b_plus * p[node_left],
                    (Dp * geometry.dual_face / h) * b_minus * p[node_right],
                )
            )
        ),
        recombined,
    )

    scales = (psi_scale, electron_scale, hole_scale)
    if not all(scale > 0.0 and math.isfinite(scale) for scale in scales):
        raise ValueError(
            f"the state has no terms to measure a residual against: term "
            f"scales {scales} for (psi, n, p). Every equation is identically "
            "zero, which a device never is. Dividing by these would rename the "
            "problem as a singular matrix three call frames later."
        )
    return scales


def row_weights(
    scales: tuple[float, float, float], n_nodes: int
) -> npt.NDArray[np.float64]:
    """One weight per unknown, from the three per family term scales [1]."""
    weights = np.empty(UNKNOWNS_PER_NODE * n_nodes)
    for component, scale in zip(Unknown, scales, strict=True):
        weights[component::UNKNOWNS_PER_NODE] = scale
    return weights


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
