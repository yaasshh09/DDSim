"""Material regions on a structured mesh: which cell is what, and what follows.

A MOS capacitor is two materials stacked. The oxide carries Poisson only, with
a different permittivity and no carriers, and docs/01-physics.md asks for the
normal component of D to be continuous across the interface rather than E.

Box integration was chosen partly because that condition is not a special case
in it. Every edge carries its own permittivity, so the flux balance written at
an interface node **is** the statement that normal D is continuous. There is no
interface code anywhere in this project, and there should not be.

Materials belong to cells
-------------------------
The interface is required to lie on a line of mesh nodes, so every edge lies
wholly in one material. That is not enough on its own, because the face a
horizontal edge crosses is not an edge: it spans half a cell above the edge and
half a cell below, and at the interface those two halves are different
materials. So permittivity is assembled per edge as an area weighted sum over
the cells either side:

    eps_r[e] * dual_face[e] = sum over adjacent cells of eps_r[cell] * share

and the same argument says an interface node's dual cell is half semiconductor,
so only half of it carries charge and recombination.

Classifying nodes instead of cells is the easy version and it is wrong. It
gives the whole interface row one material or the other, which moves the
effective oxide thickness by half a mesh cell. On a coarse mesh that is a
percent level error in the accumulation capacitance, and it reads as a physics
problem rather than as bookkeeping.

What the semiconductor volume is for
------------------------------------
It replaces the plain dual volume everywhere the equations integrate something
that only exists in silicon: the charge term in Poisson, and the recombination
term in both continuity equations. An oxide node gets zero, which turns its
Poisson row into a bare Laplacian, exactly right for an insulator. Its two
continuity rows then say 0 = 0, which is singular, so they are pinned; see
`oxide_nodes`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ddsim.core import constants as C
from ddsim.discretize.geometry import EdgeGeometry
from ddsim.mesh.mesh2d import Mesh2D

SILICON = 0
"""Cell material tag for silicon."""

OXIDE = 1
"""Cell material tag for silicon dioxide."""

_RELATIVE_EPS = {
    SILICON: 1.0,
    OXIDE: C.EPS_R_OX / C.EPS_R_SI,
}
"""Permittivity of each material relative to the one the device is scaled by.

Silicon is exactly 1.0 because ScaleFactors uses eps_Si, which is what keeps a
single material device numerically identical to what it was before regions
existed. Oxide is about 0.333.
"""


_FULL_CELL_TOL = 1e-9
"""How far below a whole cell still counts as a whole cell [1].

The two volumes are summed by different associations, the dual volume from the
mesh axes and the semiconductor volume from quarters of cells, so on an all
silicon device they agree to rounding rather than exactly. Measured worst case
1.3e-16 relative on a uniform 4 by 5 mesh, which without a tolerance made two
bulk nodes report themselves as interface nodes.

The margin is enormous compared to anything real. A genuine interface node in a
structured stack holds about half its cell, and no grading takes that above
0.999, so nine orders of magnitude separate the tolerance from the physics.
"""


@dataclass(frozen=True)
class RegionMap:
    """Which material every cell is, and everything that follows from it."""

    cell_material: npt.NDArray[np.int64]
    """Material tag of every cell, shape (ny-1, nx-1), row major like the mesh."""

    eps_r: npt.NDArray[np.float64]
    """Effective relative permittivity of every edge [1], area weighted."""

    semiconductor_volume: npt.NDArray[np.float64]
    """The part of each node's dual cell that is semiconductor [cm^2].

    Equal to the full dual volume in the bulk, zero in the oxide, and a partial
    cell at the interface. This is what the charge and recombination terms
    integrate over, never the plain dual volume.
    """

    oxide_nodes: npt.NDArray[np.int64]
    """Nodes with no semiconductor at all, whose n and p rows must be pinned.

    Strictly inside the oxide. An interface node is a semiconductor node: it
    has silicon underneath it, and it is where the inversion layer forms, which
    is the entire point of the device.
    """

    def interface_nodes(self, mesh: Mesh2D) -> npt.NDArray[np.int64]:
        """Semiconductor nodes sitting on the boundary with the insulator.

        Defined by what their dual cell is made of rather than by where they
        are: an interface node holds some semiconductor and not all of it. A
        bulk node holds all of its cell and an oxide node holds none, so
        neither qualifies, and no geometry has to be described twice.

        This is the surface. It is where the inversion layer forms, so it is
        what a surface potential is read at and what a C-V curve is about.
        """
        fraction = self.semiconductor_volume / mesh.volume
        partial = (fraction > 0.0) & (fraction < 1.0 - _FULL_CELL_TOL)
        return np.flatnonzero(partial).astype(np.int64)

    def edge_geometry(self, mesh: Mesh2D) -> EdgeGeometry:
        """The geometry the assemblies take, carrying these permittivities."""
        return EdgeGeometry(
            edge_nodes=mesh.edge_nodes,
            dual_face=mesh.dual_face,
            eps_r=self.eps_r,
        )

    def __repr__(self) -> str:
        counts = {
            "silicon": int(np.sum(self.cell_material == SILICON)),
            "oxide": int(np.sum(self.cell_material == OXIDE)),
        }
        return (
            f"RegionMap {counts['silicon']} silicon cells, "
            f"{counts['oxide']} oxide cells, "
            f"{self.oxide_nodes.size} carrier free nodes"
        )


def _node_line_index(axis: npt.NDArray[np.float64], position: float) -> int:
    """Index of the node line at `position`, or raise if there is not one."""
    distance = np.abs(axis - position)
    nearest = int(np.argmin(distance))

    span = float(axis[-1] - axis[0])
    if distance[nearest] > 1e-9 * span:
        raise ValueError(
            f"the interface at y={position:g} cm does not lie on a node line; "
            f"the nearest is at y={axis[nearest]:g} cm. A cell that is half "
            "oxide and half silicon has no single permittivity, and rounding "
            "to the nearer line would move the oxide thickness by up to half a "
            "cell without saying so. Put a node line on the interface."
        )
    return nearest


def region_map(
    mesh: Mesh2D, cell_material: npt.NDArray[np.int64]
) -> RegionMap:
    """Build the derived quantities from a per cell material map.

    Args:
        mesh: the structured mesh.
        cell_material: shape (ny-1, nx-1), one tag per cell.

    Everything here is an area weighted sum over cells, done the same way
    twice: once onto edges for the permittivity, once onto nodes for the
    semiconductor volume.
    """
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.x_axis.h, mesh.y_axis.h
    eps_cell = np.vectorize(_RELATIVE_EPS.__getitem__)(cell_material)
    is_silicon = (cell_material == SILICON).astype(np.float64)

    # --- permittivity onto edges.
    # A horizontal edge at row j is bounded by the cell row below (j-1) and the
    # cell row above (j), each contributing half its height to the face.
    below = np.zeros((ny, nx - 1))
    above = np.zeros((ny, nx - 1))
    below[1:] = eps_cell * (0.5 * dy)[:, None]
    above[:-1] = eps_cell * (0.5 * dy)[:, None]
    horizontal = (below + above).ravel()

    # A vertical edge at column i is bounded by the cell column to its left
    # (i-1) and to its right (i), each contributing half its width.
    left = np.zeros((ny - 1, nx))
    right = np.zeros((ny - 1, nx))
    left[:, 1:] = eps_cell * (0.5 * dx)[None, :]
    right[:, :-1] = eps_cell * (0.5 * dx)[None, :]
    vertical = (left + right).ravel()

    # dual_face already holds the total extent of each face, so dividing gives
    # the area weighted mean permittivity rather than a sum of areas.
    eps_r = np.concatenate([horizontal, vertical]) / mesh.dual_face

    # --- semiconductor volume onto nodes. Every cell hands a quarter of its
    # area to each of its four corners.
    quarter = is_silicon * np.outer(0.5 * dy, 0.5 * dx)
    volume = np.zeros((ny, nx))
    volume[:-1, :-1] += quarter
    volume[:-1, 1:] += quarter
    volume[1:, :-1] += quarter
    volume[1:, 1:] += quarter

    semiconductor_volume = volume.ravel()
    oxide_nodes = np.flatnonzero(semiconductor_volume == 0.0).astype(np.int64)

    return RegionMap(
        cell_material=cell_material,
        eps_r=eps_r,
        semiconductor_volume=semiconductor_volume,
        oxide_nodes=oxide_nodes,
    )


def stacked_regions(mesh: Mesh2D, interface_y: float) -> RegionMap:
    """Silicon below `interface_y`, oxide above it.

    Args:
        mesh: the structured mesh.
        interface_y: height of the Si/SiO2 interface [cm]. Must lie on a line
            of mesh nodes.

    An interface above the top of the device gives a single material device,
    whose eps_r is 1.0 everywhere, which is the case that has to stay
    numerically identical to everything built before regions existed.
    """
    nx, ny = mesh.nx, mesh.ny

    if interface_y >= mesh.y_axis.x[-1]:
        cell_material = np.full((ny - 1, nx - 1), SILICON, dtype=np.int64)
        return region_map(mesh, cell_material)

    row = _node_line_index(mesh.y_axis.x, interface_y)

    cell_material = np.full((ny - 1, nx - 1), SILICON, dtype=np.int64)
    cell_material[row:] = OXIDE
    return region_map(mesh, cell_material)
