from __future__ import annotations
from dataclasses import dataclass
from itertools import permutations
import numpy as np
import numpy.typing as npt
from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.geometry import EdgeGeometry, ScaledMesh
from ddsim.mesh.mesh2d import Mesh2D
from ddsim.mesh.mesh3d import Mesh3D
from ddsim.mesh.quality import MeshQualityError

COUPLING_ROUNDOFF = 1e-12
PAIRS = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1))


@dataclass(frozen=True)
class TetMesh:
    node_x: npt.NDArray[np.float64]

    node_y: npt.NDArray[np.float64]

    node_z: npt.NDArray[np.float64]

    tets: npt.NDArray[np.int64]

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
        return ScaledMesh(h=self.h / scale.x_0, volume=self.volume / scale.x_0**3, geometry=EdgeGeometry(edge_nodes=self.edge_nodes, dual_face=self.dual_face / scale.x_0**2, eps_r=eps_r, semiconductor_face=None if semiconductor_face is None else semiconductor_face / scale.x_0**2))

    def __repr__(self) -> str:
        return f"TetMesh {self.n_nodes} nodes, {self.tets.shape[0]} tetrahedra, {self.n_edges} edges"


def _sphere_centres(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], c: npt.NDArray[np.float64], d: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    m = 2.0 * np.stack([b - a, c - a, d - a], axis=1)
    r = np.stack([(b * b).sum(1) - (a * a).sum(1), (c * c).sum(1) - (a * a).sum(1), (d * d).sum(1) - (a * a).sum(1)], axis=1)
    return np.linalg.solve(m, r[:, :, None])[:, :, 0]


def _circle_centres(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    u, v = b - a, c - a
    w = np.cross(u, v)
    top = np.cross((u * u).sum(1)[:, None] * v - (v * v).sum(1)[:, None] * u, w)
    return a + top / (2.0 * (w * w).sum(1))[:, None]


def tet_volumes(points: npt.NDArray[np.float64], tets: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    p = points[tets]
    return np.asarray(np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) / 6.0)


def tet_mesh(points: npt.NDArray[np.float64], tets: npt.NDArray[np.int64]) -> TetMesh:
    p = np.asarray(points, dtype=np.float64)
    t = np.asarray(tets, dtype=np.int64)
    if p.ndim != 2 or p.shape[1] != 3 or t.ndim != 2 or t.shape[1] != 4:
        raise ValueError(f"a tetrahedral mesh takes points of shape (n, 3) and tetrahedra of shape (m, 4), got {p.shape} and {t.shape}")
    v = np.abs(tet_volumes(p, t))
    if np.any(v <= 1e-14 * v.max()):
        k = int(np.flatnonzero(v <= 1e-14 * v.max())[0])
        raise MeshQualityError(f"tetrahedron {k} has no volume. Four coplanar points have no dual cell to integrate over.")
    centre = _sphere_centres(p[t[:, 0]], p[t[:, 1]], p[t[:, 2]], p[t[:, 3]])
    ends: list[npt.NDArray[np.int64]] = []
    faces: list[npt.NDArray[np.float64]] = []
    for i, j, k, o in PAIRS:
        a, b, c, d = p[t[:, i]], p[t[:, j]], p[t[:, k]], p[t[:, o]]
        mid = 0.5 * (a + b)
        u = (b - a) / np.linalg.norm(b - a, axis=1)[:, None]
        f1, f2 = _circle_centres(a, b, c), _circle_centres(a, b, d)
        area = 0.5 * np.einsum("ij,ij->i", u, np.cross(f1 - mid, centre - mid) + np.cross(centre - mid, f2 - mid))
        sense = np.sign(np.einsum("ij,ij->i", u, np.cross(c - mid, d - mid)))
        ends.append(np.sort(np.column_stack([t[:, i], t[:, j]]), axis=1))
        faces.append(area * sense)
    edges, at = np.unique(np.concatenate(ends), axis=0, return_inverse=True)
    face = np.zeros(edges.shape[0])
    np.add.at(face, at.ravel(), np.concatenate(faces))
    h = np.linalg.norm(p[edges[:, 1]] - p[edges[:, 0]], axis=1)
    bad = np.flatnonzero(face < -COUPLING_ROUNDOFF * np.abs(face).max())
    if bad.size:
        k = int(bad[np.argmin(face[bad])])
        raise MeshQualityError(
            f"{bad.size} edge(s) get a negative coupling, the worst being edge {edges[k].tolist()} at {face[k]:.3e} cm^2. "
            "The circumcentres of the tetrahedra around that edge fall on the wrong side of it, which lets carrier "
            "densities go negative for no physical reason. A 3D Delaunay mesh can still do this at a boundary or "
            "around a sliver. Re-mesh or refine there."
        )
    face = np.maximum(face, 0.0)
    vol = np.zeros(p.shape[0])
    np.add.at(vol, edges[:, 0], face * h / 6.0)
    np.add.at(vol, edges[:, 1], face * h / 6.0)
    return TetMesh(node_x=p[:, 0].copy(), node_y=p[:, 1].copy(), node_z=p[:, 2].copy(), tets=t, h=h, dual_face=face, volume=vol, edge_nodes=edges.astype(np.int64))


def tetrahedralized(mesh: Mesh3D) -> TetMesh:
    if not isinstance(mesh.base, Mesh2D):
        raise ValueError("splitting into tetrahedra takes a box mesh, a tensor base extruded along z")
    nx, ny, nz = mesh.base.nx, mesh.base.ny, mesh.z_axis.n_nodes
    i, j, k = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij")
    corner = (k * ny * nx + j * nx + i).ravel()
    step = (1, nx, nx * ny)
    tets = []
    for order in permutations(range(3)):
        a = corner
        b = a + step[order[0]]
        c = b + step[order[1]]
        d = c + step[order[2]]
        tets.append(np.column_stack([a, b, c, d]))
    p = np.column_stack([mesh.node_x, mesh.node_y, mesh.node_z])
    return tet_mesh(p, np.concatenate(tets))
