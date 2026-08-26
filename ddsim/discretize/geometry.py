"""Edge geometry, the one thing an assembly needs to stop being 1D.

Box integration does not care about dimensionality. It cares about three things:
which two nodes an edge joins, how much dual face area that edge carries the
flux through, and what the material between them is. In 1D all three are
trivial, and the 1D assemblies were written with them inlined: edge e joins node
e and node e+1 by contiguous slicing, the dual face of a 1D edge is the unit
cross section, and there is only ever one material. None of the three survives
into 2D.

This type carries them instead, so that `discretize/` can be handed a 2D mesh
without knowing that is what happened.

    flux through edge e = eps_r[e] * dual_face[e] * (difference) / h[e]

The defaults reproduce 1D exactly. `eps_r` and `dual_face` both default to
**exactly** 1.0, which matters more than it looks: multiplication by 1.0 is
exact in IEEE754, so `eps_r * dual_face * d / h` collapses to `d / h` with the
same bit pattern, and every number the 1D test suite pins stays where it is. A
default that was merely close to one would move all of them at the last digit
and the suite would light up for no physical reason.

On scatter: the 1D code accumulated face fluxes into nodes with a pair of slice
additions, and said so, calling `np.add.at` a needless generality when there are
no repeated indices. That was right in 1D. In 2D a node has four neighbours
rather than two and the edges touching it are not contiguous, so the slice form
has nothing to slice and `np.add.at` becomes the only option. It is also the
only one that works here at all: `np.bincount` is faster on paper but refuses
complex weights, and `coupled.py` is deliberately generic over complex128 so
that complex step differentiation and the Phase 4 AC solve can both run through
it. Measured on 200k edges into 20k nodes, `add.at` came out at 0.51 ms against
`bincount` at 0.80 ms, so nothing is being given up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

EdgeQuantity = float | npt.NDArray[np.float64]
"""A per edge quantity, or a scalar standing for the same value on every edge."""


@dataclass(frozen=True)
class EdgeGeometry:
    """Which nodes each edge joins, and what its flux carries.

    Attributes:
        edge_nodes: shape (n_edges, 2), the two nodes of each edge. None means
            the contiguous 1D ordering, edge e joining node e and node e+1.
        dual_face: dual face area of each edge [1], scaled. 1.0 in 1D, where
            the cross section is the unit area.
        eps_r: permittivity of each edge relative to the permittivity the
            device is scaled by [1]. 1.0 in silicon, since the scaling uses
            eps_Si, and about 0.333 in silicon dioxide.
        semiconductor_face: the part of each edge's dual face that carriers
            can cross [1], scaled. None means all of it, which is right for a
            device made of one semiconductor. See the `carrier_face` property.
    """

    edge_nodes: npt.NDArray[np.int64] | None = None
    dual_face: EdgeQuantity = 1.0
    eps_r: EdgeQuantity = 1.0
    semiconductor_face: EdgeQuantity | None = None

    def __post_init__(self) -> None:
        if self.edge_nodes is None:
            return

        if self.edge_nodes.ndim != 2 or self.edge_nodes.shape[1] != 2:
            raise ValueError(
                "edge_nodes must have shape (n_edges, 2), got "
                f"{self.edge_nodes.shape}. Transposing it is the usual slip."
            )

        if np.any(self.edge_nodes[:, 0] == self.edge_nodes[:, 1]):
            raise ValueError(
                "an edge joins a node to itself, which has zero length and "
                "would divide by zero in every flux that crosses it"
            )

    def ends(self, n_edges: int) -> tuple[
        npt.NDArray[np.int64], npt.NDArray[np.int64]
    ]:
        """The (left, right) node index of every edge.

        Args:
            n_edges: how many edges the mesh has, for the consistency check.

        Gathering with these two arrays is what replaces `[:-1]` and `[1:]`.
        In the default 1D case they are exactly those two slices expressed as
        indices, so the gathered values are identical.
        """
        if self.edge_nodes is None:
            left = np.arange(n_edges, dtype=np.int64)
            return left, left + 1

        if self.edge_nodes.shape[0] != n_edges:
            raise ValueError(
                f"the edge list has {self.edge_nodes.shape[0]} edges but the "
                f"mesh has {n_edges}"
            )

        return self.edge_nodes[:, 0], self.edge_nodes[:, 1]

    def edge_count(self, n_nodes: int) -> int:
        """How many edges the mesh has, given how many nodes it has.

        Only the default can answer this from the node count alone, because a
        1D chain of N nodes has exactly N-1 edges. Any other mesh has to be
        asked, which is why an explicit edge list reports its own length.
        """
        if self.edge_nodes is None:
            return n_nodes - 1
        return int(self.edge_nodes.shape[0])

    def ends_of(self, n_nodes: int) -> tuple[
        npt.NDArray[np.int64], npt.NDArray[np.int64]
    ]:
        """`ends`, for callers that know the node count rather than the edge one.

        The flux kernels are the reason this exists: they are handed psi and
        nothing else, so the node count is all they have.
        """
        return self.ends(self.edge_count(n_nodes))

    @property
    def weight(self) -> EdgeQuantity:
        """The combined `eps_r * dual_face` prefactor, for Poisson only [1].

        **Carrier fluxes use `dual_face` alone, not this.** Permittivity scales
        the electric displacement and has no business in a Scharfetter-Gummel
        current: an electron crossing an edge does not care what the edge is
        made of, only how much face area it crosses. Multiplying a carrier flux
        by eps_r would quietly rescale every current in the oxide by 0.333, and
        since the oxide carries no carriers at all it would look like it worked.

        Exactly 1.0 in the 1D silicon default, which is what keeps the existing
        results bit identical.
        """
        return self.eps_r * self.dual_face

    @property
    def carrier_face(self) -> EdgeQuantity:
        """The face area a carrier flux crosses [1]. Poisson does not use this.

        An insulator carries no current, so the face an edge offers to
        electrons and holes is only the part of it made of semiconductor.
        Three cases, and the third is the one that is easy to miss:

        - an edge with silicon on both sides offers its whole face
        - an edge inside the oxide offers none of it, so no current crosses
          the Si/SiO2 interface and none flows through the insulator
        - **an edge lying along the interface offers half of it**, because the
          face a horizontal edge crosses spans half a cell above and half a
          cell below, and at the interface those two halves are different
          materials. That edge is the channel of a MOSFET, so getting it wrong
          is not a corner case: it doubles the conductance of the inversion
          layer.

        Zeroing the volume afterwards does not do this job. The volume
        multiplies the recombination term, not the flux, so an oxide edge with
        a full face still moves carriers between the interface node and a node
        with no carriers in it, and on a MOSFET that drains the inversion layer
        into the gate dielectric.

        Defaults to the whole dual face, so a single material device is bit
        for bit what it was before this existed.
        """
        if self.semiconductor_face is None:
            return self.dual_face
        return self.semiconductor_face


UNIFORM_1D = EdgeGeometry()
"""The 1D contiguous, unit cross section, all silicon case.

The default argument of every assembly. Named rather than written inline so
that a call site reads as a deliberate choice of geometry.
"""


@dataclass(frozen=True)
class ScaledMesh:
    """What an assembly needs from a mesh, already in scaled units.

    Every caller used to write `mesh.h / scale.x_0` and `mesh.volume /
    scale.x_0` by hand. The second is wrong in 2D, where the dual volume is an
    area and wants x_0 squared, and it is wrong quietly: the device simply
    comes out the wrong size by a factor of the Debye length, converges, and
    reports a capacitance that is off by orders of magnitude with no symptom
    pointing at the cause.

    The general rule is `volume / x_0^d` and `face / x_0^(d-1)`. Rather than
    write d anywhere, each mesh answers for itself, and no caller has to know
    which dimension it is in.
    """

    h: npt.NDArray[np.float64]
    """Edge lengths [1], scaled by x_0."""

    volume: npt.NDArray[np.float64]
    """Dual cell volumes [1], scaled by x_0^d."""

    geometry: EdgeGeometry
    """Edge list, dual faces and permittivities, all scaled."""

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return int(self.volume.size)

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return int(self.h.size)
