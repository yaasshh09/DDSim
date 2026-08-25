"""Structured tensor product 2D mesh, box integration finite volume.

phases/PHASE-4.md is explicit about starting here rather than with Delaunay: a
rectangular MOS capacitor does not need unstructured meshing, and a rectangle
cannot produce an obtuse triangle, so the negative dual area failure mode that
docs/05-pitfalls.md describes cannot arise. mesh/quality.py exists for the day
unstructured meshing does arrive, and refuses a mesh that would hit it.

Built as the tensor product of two Mesh1D axes. That is deliberate reuse and
not a shortcut. The grading solver and the dual grid construction are
inherited from the 1D mesh rather than written a second time, and grading a 2D
mesh towards a junction is then just grading the axis.

The dual cells sum to the domain area to about 1e-16 relative, not exactly.
Summing the outer product reassociates the terms compared with multiplying the
two 1D sums. The 1D sum is not unconditionally exact either: it is exact on
the Phase 0 acceptance case and off by one bit on other node counts.

Numbering
---------
Row major, x fastest:

    node(i, j) = j*nx + i

Edges come in two families, horizontal first then vertical. Within a family
they run in the same order as the nodes. Nothing downstream depends on that
ordering, and tests/unit/test_edge_list_assembly.py proves it: shuffling the
edge list does not change the answer. The ordering is chosen to be readable,
not to be relied on.

What box integration needs from an edge
---------------------------------------
Two lengths, and it is worth being careful about which is which.

    h          the distance between the two nodes, along the edge
    dual_face  the extent of the face the flux crosses, across the edge

For a horizontal edge, the face it crosses is vertical, so `dual_face` comes
from the **y** dual grid. For a vertical edge it comes from the x dual grid.
Swapping those two is the classic slip, and on a square mesh it makes no
difference at all, which is what makes it dangerous. Test on a rectangle.

Units and depth
---------------
Lengths in cm, following CLAUDE.md. A 2D simulation is per unit depth, so
`dual_face` is a length [cm] and `volume` is an area [cm^2]. The scaling that
the assemblies want is therefore

    h          / x_0
    dual_face  / x_0
    volume     / x_0^2

which is the d-dimensional rule `volume / x_0^d` and `face / x_0^(d-1)`
evaluated at d = 2. In 1D the same rule gives a dual_face of x_0^0 = 1, which
is exactly the default that discretize/geometry.py carries, so the two cases
are the same statement rather than two conventions.

A device uniform in y solves to the same psi, n and p as the 1D device with
the same x profile. The residual is a constant multiple of the 1D one, the
constant being the device height in scaled units, so the solution is identical
while the residual is not. That is the acceptance criterion in PHASE-4.md and
it is checked in tests/analytic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.geometry import EdgeGeometry, ScaledMesh
from ddsim.mesh.mesh1d import Mesh1D, uniform_mesh_1d


@dataclass(frozen=True)
class Mesh2D:
    """A structured 2D mesh with its dual grid and edge list."""

    x_axis: Mesh1D
    """The x axis, as a 1D mesh. Carries its own dual grid."""

    y_axis: Mesh1D
    """The y axis, as a 1D mesh."""

    node_x: npt.NDArray[np.float64]
    """x position of every node [cm], length n_nodes."""

    node_y: npt.NDArray[np.float64]
    """y position of every node [cm], length n_nodes."""

    h: npt.NDArray[np.float64]
    """Length of every edge [cm], the distance between its two nodes."""

    dual_face: npt.NDArray[np.float64]
    """Extent of the face every edge crosses [cm], per unit depth."""

    volume: npt.NDArray[np.float64]
    """Dual cell area of every node [cm^2], per unit depth."""

    edge_nodes: npt.NDArray[np.int64]
    """Shape (n_edges, 2). The two nodes of each edge."""

    @property
    def nx(self) -> int:
        """Nodes along x."""
        return self.x_axis.n_nodes

    @property
    def ny(self) -> int:
        """Nodes along y."""
        return self.y_axis.n_nodes

    @property
    def n_nodes(self) -> int:
        """Total nodes."""
        return int(self.node_x.size)

    @property
    def n_edges(self) -> int:
        """Total edges, both families."""
        return int(self.h.size)

    @property
    def n_horizontal(self) -> int:
        """How many edges are in the horizontal family, which come first."""
        return (self.nx - 1) * self.ny

    def node_at(self, i: int, j: int) -> int:
        """The node index at column i, row j."""
        return j * self.nx + i

    def edge_geometry(self, eps_r: npt.NDArray[np.float64] | float = 1.0) -> (
        EdgeGeometry
    ):
        """What the assemblies need to work on this mesh.

        Args:
            eps_r: permittivity of each edge relative to the scaling
                permittivity [1]. 1.0 everywhere in a single material device.

        The dual faces are handed over unscaled, in cm. The device layer
        divides them by x_0, exactly as it already does for h and volume. See
        the module docstring on why the power of x_0 differs between them.
        """
        return EdgeGeometry(
            edge_nodes=self.edge_nodes, dual_face=self.dual_face, eps_r=eps_r
        )

    def scaled(
        self,
        scale: ScaleFactors,
        eps_r: npt.NDArray[np.float64] | float = 1.0,
    ) -> ScaledMesh:
        """This mesh in the units the assemblies work in.

        Args:
            scale: the de Mari scale factors.
            eps_r: permittivity of each edge relative to the scaling
                permittivity [1]. A single material device leaves it at 1.0;
                a MOS stack passes what device/regions.py worked out.

        The two powers of x_0 differ, and that is the whole point of asking
        the mesh instead of doing it at the call site. See ScaledMesh.
        """
        return ScaledMesh(
            h=self.h / scale.x_0,
            volume=self.volume / scale.x_0**2,
            geometry=EdgeGeometry(
                edge_nodes=self.edge_nodes,
                dual_face=self.dual_face / scale.x_0,
                eps_r=eps_r,
            ),
        )

    def __repr__(self) -> str:
        return (
            f"Mesh2D {self.nx}x{self.ny} = {self.n_nodes} nodes, "
            f"{self.n_edges} edges, "
            f"{self.x_axis.length:.4e} by {self.y_axis.length:.4e} cm"
        )


def tensor_mesh_2d(x_axis: Mesh1D, y_axis: Mesh1D) -> Mesh2D:
    """The tensor product of two 1D meshes.

    Args:
        x_axis: the x direction, uniform or graded.
        y_axis: the y direction.

    Either axis may be graded, which is how a 2D mesh resolves a junction.
    """
    nx, ny = x_axis.n_nodes, y_axis.n_nodes

    # Node positions. Row major, x fastest, so a row of the 2D mesh is the
    # 1D mesh node for node.
    node_x = np.tile(x_axis.x, ny)
    node_y = np.repeat(y_axis.x, nx)

    columns = np.arange(nx, dtype=np.int64)
    rows = np.arange(ny, dtype=np.int64)

    # --- horizontal edges: (i, j) to (i+1, j), one per interior x interval
    # per row. The flux crosses a vertical face, whose extent is the y dual
    # width of the row. See the module docstring before swapping these.
    h_i, h_j = np.meshgrid(columns[:-1], rows, indexing="xy")
    h_from = (h_j * nx + h_i).ravel()
    horizontal = np.column_stack([h_from, h_from + 1])
    h_length = np.tile(x_axis.h, ny)
    h_face = np.repeat(y_axis.volume, nx - 1)

    # --- vertical edges: (i, j) to (i, j+1). The flux crosses a horizontal
    # face, whose extent is the x dual width of the column.
    v_i, v_j = np.meshgrid(columns, rows[:-1], indexing="xy")
    v_from = (v_j * nx + v_i).ravel()
    vertical = np.column_stack([v_from, v_from + nx])
    v_length = np.repeat(y_axis.h, nx)
    v_face = np.tile(x_axis.volume, ny - 1)

    # --- dual cell area of each node, the product of the two 1D dual widths.
    # The areas sum to the domain to about 1e-16 relative. See the module
    # docstring: summing the outer product is not the same association as
    # multiplying the two 1D sums, and neither is unconditionally exact.
    volume = np.outer(y_axis.volume, x_axis.volume).ravel()

    return Mesh2D(
        x_axis=x_axis,
        y_axis=y_axis,
        node_x=node_x,
        node_y=node_y,
        h=np.concatenate([h_length, v_length]),
        dual_face=np.concatenate([h_face, v_face]),
        volume=volume,
        edge_nodes=np.concatenate([horizontal, vertical]).astype(np.int64),
    )


def uniform_mesh_2d(width: float, height: float, nx: int, ny: int) -> Mesh2D:
    """A uniformly spaced rectangle, [0, width] by [0, height] [cm]."""
    return tensor_mesh_2d(
        uniform_mesh_1d(length=width, n_nodes=nx),
        uniform_mesh_1d(length=height, n_nodes=ny),
    )
