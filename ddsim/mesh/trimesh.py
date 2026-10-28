from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from scipy.spatial import Delaunay
from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.geometry import EdgeGeometry, ScaledMesh
from ddsim.mesh.mesh2d import Mesh2D
from ddsim.mesh.quality import MeshQualityError, triangle_angles, triangle_areas

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


def triangle_mesh(points: npt.NDArray[np.float64], triangles: npt.NDArray[np.int64], cylindrical: bool = False) -> TriMesh:
    p = np.asarray(points, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)
    a = np.abs(triangle_areas(p, t))
    if np.any(a <= 0.0):
        k = int(np.flatnonzero(a <= 0.0)[0])
        raise MeshQualityError(f"triangle {k} has no area. Three collinear points have no dual cell to integrate over.")
    if cylindrical and np.any(p[:, 0] < 0.0):
        raise MeshQualityError("a cylindrical mesh has r along x, and r can't go below zero")
    angle = triangle_angles(p, t)
    c = _circumcentres(p, t)
    ends: list[npt.NDArray[np.int64]] = []
    faces: list[npt.NDArray[np.float64]] = []
    halves: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]] = []
    for k in range(3):
        u, v = t[:, (k + 1) % 3], t[:, (k + 2) % 3]
        m = 0.5 * (p[u] + p[v])
        h = np.hypot(*(p[v] - p[u]).T)
        with np.errstate(divide="ignore", invalid="ignore"):
            piece = 0.5 * h / np.tan(angle[:, k])
        if cylindrical:
            face = 2.0 * np.pi * piece * 0.5 * (m[:, 0] + c[:, 0])
            for w in (u, v):
                halves.append((w, 2.0 * np.pi * 0.25 * h * piece * (p[w, 0] + m[:, 0] + c[:, 0]) / 3.0))
        else:
            face = piece
            for w in (u, v):
                halves.append((w, 0.25 * h * piece))
        ends.append(np.sort(np.column_stack([u, v]), axis=1))
        faces.append(face)
    edges, at = np.unique(np.concatenate(ends), axis=0, return_inverse=True)
    face = np.zeros(edges.shape[0])
    np.add.at(face, at.ravel(), np.concatenate(faces))
    h = np.hypot(*(p[edges[:, 1]] - p[edges[:, 0]]).T)
    bad = np.flatnonzero(face < -COUPLING_ROUNDOFF * np.abs(face).max())
    if bad.size:
        k = int(bad[np.argmin(face[bad])])
        raise MeshQualityError(
            f"{bad.size} edge(s) get a negative coupling, the worst being edge {edges[k].tolist()} at "
            f"{face[k]:.3e} cm. The angles facing an edge add up to more than 180 degrees there, or to more "
            "than 90 at a boundary, which lets carrier densities go negative for no physical reason. "
            "Re-triangulate or refine that corner."
        )
    vol = np.zeros(p.shape[0])
    for w, piece_volume in halves:
        np.add.at(vol, w, piece_volume)
    return TriMesh(node_x=p[:, 0].copy(), node_y=p[:, 1].copy(), triangles=t, h=h, dual_face=np.maximum(face, 0.0), volume=vol, edge_nodes=edges.astype(np.int64))


def _circumcentres(p: npt.NDArray[np.float64], t: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    d = 2.0 * (a[:, 0] * (b[:, 1] - c[:, 1]) + b[:, 0] * (c[:, 1] - a[:, 1]) + c[:, 0] * (a[:, 1] - b[:, 1]))
    s = (a * a).sum(axis=1), (b * b).sum(axis=1), (c * c).sum(axis=1)
    x = (s[0] * (b[:, 1] - c[:, 1]) + s[1] * (c[:, 1] - a[:, 1]) + s[2] * (a[:, 1] - b[:, 1])) / d
    y = (s[0] * (c[:, 0] - b[:, 0]) + s[1] * (a[:, 0] - c[:, 0]) + s[2] * (b[:, 0] - a[:, 0])) / d
    return np.column_stack([x, y])


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
