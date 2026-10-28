from __future__ import annotations
from ddsim.core.config import CONFIG
from dataclasses import dataclass
import numpy as np, numpy.typing as npt
from ddsim.core import constants as C
from ddsim.discretize.geometry import EdgeGeometry
from ddsim.mesh.mesh2d import Mesh2D
from ddsim.mesh.quality import triangle_angles
from ddsim.mesh.trimesh import TriMesh

SILICON = 0

OXIDE = 1

_RELATIVE_EPS = {SILICON: 1.0, OXIDE: C.EPS_R_OX / C.EPS_R_SI,}

_FULL_CELL_TOL = CONFIG.mesh.full_cell_tol


@dataclass(frozen=True)
class RegionMap:
    cell_material: npt.NDArray[np.int64]

    eps_r: npt.NDArray[np.float64]

    semiconductor_volume:npt.NDArray[np.float64]
    semiconductor_face: npt.NDArray[np.float64]

    oxide_nodes: npt.NDArray[np.int64]
    def interface_nodes(self, mesh: Mesh2D) -> npt.NDArray[np.int64]:
        b = self.semiconductor_volume / mesh.volume
        w = (b > 0.0) & (b < 1.0 - _FULL_CELL_TOL)
        return np.flatnonzero(w).astype(np.int64)

    def interface_length(self, mesh: Mesh2D) -> npt.NDArray[np.float64]:
        s = self.cell_material == SILICON; o = self.cell_material == OXIDE
        L = np.zeros((mesh.ny, mesh.nx))
        a = ((s[:-1] & o[1:]) | (o[:-1] & s[1:])) * (0.5 * mesh.x_axis.h)[None, :]
        L[1:-1, :-1] += a; L[1:-1, 1:] += a
        b = ((s[:, :-1] & o[:, 1:]) | (o[:, :-1] & s[:, 1:])) * (0.5 * mesh.y_axis.h)[:, None]
        L[:-1, 1:-1] += b; L[1:, 1:-1] += b
        return L.ravel()

    def edge_geometry(self, mesh:Mesh2D) -> EdgeGeometry:
        return EdgeGeometry(edge_nodes=mesh.edge_nodes, dual_face=mesh.dual_face, eps_r=self.eps_r, semiconductor_face=self.semiconductor_face,)

    def __repr__(self) -> str:
        vals = {"silicon":int(np.sum(self.cell_material == SILICON)), "oxide":int(np.sum(self.cell_material == OXIDE)),}
        return (
            f"RegionMap {vals['silicon']} silicon cells, "
            f"{vals['oxide']} oxide cells, "
            f"{self.oxide_nodes.size} carrier free nodes"
        )

def _node_line_index(axis:npt.NDArray[np.float64], position:float) -> int:
    s = np.abs(axis-position)
    x = int(np.argmin(s))

    foo = float(axis[-1] - axis[0])
    if s[x] > 1e-9*foo:
        raise ValueError(
            f"the interface at y={position:g} cm does not lie on a node line; "
            f"the nearest is at y={axis[x]:g} cm. A cell that is half "
            'oxide and half silicon has no single permittivity, and rounding '
            'to the nearer line would move the oxide thickness by up to half a '
            "cell without saying so. Put a node line on the interface."
        )
    return x
def _onto_edges(
    mesh: Mesh2D, cell_value: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    info, x = mesh.nx, mesh.ny

    thing, g = mesh.x_axis.h, mesh.y_axis.h

    r = np.zeros((x, info - 1))

    w = np.zeros((x, info - 1))

    r[1:] = cell_value * (0.5*g)[:, None]
    w[:-1] = cell_value * (0.5 * g)[:, None]
    bb = (r + w).ravel()
    w2 = np.zeros((x - 1, info))
    m = np.zeros((x-1, info))
    w2[:, 1:] = cell_value * (0.5*thing)[None,:]
    m[:,:-1] = cell_value * (0.5 * thing)[None,:]
    k = (w2+m).ravel()

    return np.asarray(np.concatenate([bb, k]) / mesh.dual_face)

def region_map(
    mesh: Mesh2D, cell_material:npt.NDArray[np.int64]
) -> RegionMap:
    z, w = mesh.nx, mesh.ny
    c, v = mesh.x_axis.h, mesh.y_axis.h
    vals = np.vectorize(_RELATIVE_EPS.__getitem__)(cell_material); r = (cell_material == SILICON).astype(np.float64)
    buf = _onto_edges(mesh, vals)
    a2 = mesh.dual_face * _onto_edges(mesh, r)

    item = r * np.outer(0.5 * v, 0.5 * c)
    val = np.zeros((w, z)); val[:-1,:-1] += item
    val[:-1, 1:] += item
    val[1:,:-1] += item
    val[1:, 1:] += item
    f = val.ravel()

    u = np.flatnonzero(f == 0.0).astype(np.int64)
    return RegionMap(cell_material=cell_material, eps_r=buf, semiconductor_volume=f, semiconductor_face=a2, oxide_nodes=u,)


def stacked_regions(mesh:Mesh2D, interface_y:float) -> RegionMap:
    f, w2 = mesh.nx, mesh.ny

    if interface_y >= mesh.y_axis.x[-1]:
        m = np.full((w2 - 1, f - 1), SILICON, dtype=np.int64)
        return region_map(mesh, m)

    rows = _node_line_index(mesh.y_axis.x,
          interface_y)

    m = np.full((w2 - 1, f - 1), SILICON, dtype=np.int64)
    m[rows:] = OXIDE
    return region_map(mesh, m)


def triangle_regions(mesh: TriMesh, material: npt.NDArray[np.int64]) -> RegionMap:
    t = mesh.triangles
    m = np.asarray(material, dtype=np.int64)
    if m.shape != (t.shape[0],):
        raise ValueError(f"the material list has {m.shape} entries but the mesh has {t.shape[0]} triangles")
    if mesh.cylindrical:
        raise ValueError("regions on a cylindrical mesh need the revolved pieces, which aren't written yet")
    unknown = set(np.unique(m).tolist()) - set(_RELATIVE_EPS)
    if unknown:
        raise ValueError(f"no permittivity for material codes {sorted(unknown)}; the known ones are {sorted(_RELATIVE_EPS)}")
    p = np.column_stack([mesh.node_x, mesh.node_y])
    N = mesh.n_nodes
    keys = mesh.edge_nodes[:, 0] * N + mesh.edge_nodes[:, 1]
    eps_t = np.array([_RELATIVE_EPS[int(k)] for k in m])
    silicon = m == SILICON
    angle = triangle_angles(p, t)
    weighted = np.zeros(mesh.n_edges)
    total = np.zeros(mesh.n_edges)
    face = np.zeros(mesh.n_edges)
    volume = np.zeros(N)
    for k in range(3):
        u, v = t[:, (k + 1) % 3], t[:, (k + 2) % 3]
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        at = np.searchsorted(keys, lo * N + hi)
        h = np.hypot(*(p[v] - p[u]).T)
        with np.errstate(divide="ignore", invalid="ignore"):
            piece = 0.5 * h / np.tan(angle[:, k])
        np.add.at(weighted, at, eps_t * piece)
        np.add.at(total, at, piece)
        np.add.at(face, at, np.where(silicon, piece, 0.0))
        for w in (u, v):
            np.add.at(volume, w, np.where(silicon, 0.25 * h * piece, 0.0))
    eps = np.divide(weighted, total, out=np.ones_like(total), where=total > 0.0)
    return RegionMap(cell_material=m, eps_r=eps, semiconductor_volume=volume, semiconductor_face=np.maximum(face, 0.0), oxide_nodes=np.flatnonzero(volume <= 0.0).astype(np.int64))
