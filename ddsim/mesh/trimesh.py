from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from scipy.spatial import Delaunay
from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.geometry import EdgeGeometry, ScaledMesh
from ddsim.mesh.mesh2d import Mesh2D
from ddsim.mesh.quality import MeshQualityError, cotangent_edge_weights, triangle_areas

COUPLING_ROUNDOFF = 1e-12


@dataclass(frozen=True)
class TriMesh:
    node_x: npt.NDArray[np.float64]

    node_y: npt.NDArray[np.float64]

    triangles: npt.NDArray[np.int64]

    h: npt.NDArray[np.float64]

    dual_face: npt.NDArray[np.float64]

    volume: npt.NDArray[np.float64]

    edge_nodes: npt.NDArray[np.int64]

    @property
    def n_nodes(self) -> int:
        return int(self.node_x.size)

    @property
    def n_edges(self) -> int:
        return int(self.h.size)

    @property
    def width(self) -> float:
        return float(self.node_x.max() - self.node_x.min())

    def edge_geometry(self, eps_r: npt.NDArray[np.float64] | float = 1.0, semiconductor_face: npt.NDArray[np.float64] | None = None) -> EdgeGeometry:
        return EdgeGeometry(edge_nodes=self.edge_nodes, dual_face=self.dual_face, eps_r=eps_r, semiconductor_face=semiconductor_face)

    def scaled(self, scale: ScaleFactors, eps_r: npt.NDArray[np.float64] | float = 1.0, semiconductor_face: npt.NDArray[np.float64] | None = None) -> ScaledMesh:
        return ScaledMesh(h=self.h / scale.x_0, volume=self.volume / scale.x_0**2, geometry=EdgeGeometry(edge_nodes=self.edge_nodes, dual_face=self.dual_face / scale.x_0, eps_r=eps_r, semiconductor_face=None if semiconductor_face is None else semiconductor_face / scale.x_0))

    def __repr__(self) -> str:
        return f"TriMesh {self.n_nodes} nodes, {self.triangles.shape[0]} triangles, {self.n_edges} edges"


def triangle_mesh(points: npt.NDArray[np.float64], triangles: npt.NDArray[np.int64]) -> TriMesh:
    p = np.asarray(points, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)
    a = np.abs(triangle_areas(p, t))
    if np.any(a <= 0.0):
        k = int(np.flatnonzero(a <= 0.0)[0])
        raise MeshQualityError(f"triangle {k} has no area. Three collinear points have no dual cell to integrate over.")
    edges, w = cotangent_edge_weights(p, t)
    h = np.hypot(*(p[edges[:, 1]] - p[edges[:, 0]]).T)
    face = w * h
    bad = np.flatnonzero(face < -COUPLING_ROUNDOFF * np.abs(face).max())
    if bad.size:
        k = int(bad[np.argmin(face[bad])])
        raise MeshQualityError(
            f"{bad.size} edge(s) get a negative coupling, the worst being edge {edges[k].tolist()} at "
            f"{face[k]:.3e} cm. The angles facing an edge add up to more than 180 degrees there, or to more "
            "than 90 at a boundary, which lets carrier densities go negative for no physical reason. "
            "Re-triangulate or refine that corner."
        )
    face = np.maximum(face, 0.0)
    v = np.zeros(p.shape[0])
    np.add.at(v, edges[:, 0], 0.25 * h * face)
    np.add.at(v, edges[:, 1], 0.25 * h * face)
    return TriMesh(node_x=p[:, 0].copy(), node_y=p[:, 1].copy(), triangles=t, h=h, dual_face=face, volume=v, edge_nodes=edges)


def triangulated(mesh: Mesh2D) -> TriMesh:
    nx, ny = mesh.nx, mesh.ny
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="xy")
    a = (j * nx + i).ravel()
    b, c, d = a + 1, a + nx + 1, a + nx
    t = np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    return triangle_mesh(np.column_stack([mesh.node_x, mesh.node_y]), t)


def delaunay_mesh(points: npt.NDArray[np.float64]) -> TriMesh:
    p = np.asarray(points, dtype=np.float64)
    return triangle_mesh(p, Delaunay(p).simplices.astype(np.int64))
