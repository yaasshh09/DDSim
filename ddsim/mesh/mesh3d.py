from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.geometry import EdgeGeometry, ScaledMesh
from ddsim.mesh.mesh1d import Mesh1D
from ddsim.mesh.mesh2d import Mesh2D
from ddsim.mesh.trimesh import TriMesh


@dataclass(frozen=True)
class Mesh3D:
    base: Mesh2D | TriMesh

    z_axis: Mesh1D

    node_x: npt.NDArray[np.float64]
    node_y: npt.NDArray[np.float64]
    node_z: npt.NDArray[np.float64]

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
        return self.base.width

    def edge_geometry(self, eps_r: npt.NDArray[np.float64] | float = 1.0, semiconductor_face: npt.NDArray[np.float64] | None = None) -> EdgeGeometry:
        return EdgeGeometry(edge_nodes=self.edge_nodes, dual_face=self.dual_face, eps_r=eps_r, semiconductor_face=semiconductor_face)

    def scaled(self, scale: ScaleFactors, eps_r: npt.NDArray[np.float64] | float = 1.0, semiconductor_face: npt.NDArray[np.float64] | None = None) -> ScaledMesh:
        return ScaledMesh(h=self.h / scale.x_0, volume=self.volume / scale.x_0**3, geometry=EdgeGeometry(edge_nodes=self.edge_nodes, dual_face=self.dual_face / scale.x_0**2, eps_r=eps_r, semiconductor_face=None if semiconductor_face is None else semiconductor_face / scale.x_0**2))

    def __repr__(self) -> str:
        return f"Mesh3D {self.base.n_nodes} x {self.z_axis.n_nodes} = {self.n_nodes} nodes, {self.n_edges} edges, {self.z_axis.length:.4e} cm deep"


def extruded(base: Mesh2D | TriMesh, z_axis: Mesh1D) -> Mesh3D:
    if isinstance(base, TriMesh) and base.cylindrical:
        raise ValueError("a cylindrical mesh is already a 3D body of revolution, so extruding it along z has no meaning")
    n, k = base.n_nodes, z_axis.n_nodes
    layer = np.arange(k, dtype=np.int64)[:, None, None] * n
    flat = (base.edge_nodes[None, :, :] + layer).reshape(-1, 2)
    up = np.arange(n * (k - 1), dtype=np.int64)
    rise = np.column_stack([up, up + n])
    return Mesh3D(
        base=base,
        z_axis=z_axis,
        node_x=np.tile(base.node_x, k),
        node_y=np.tile(base.node_y, k),
        node_z=np.repeat(z_axis.x, n),
        h=np.concatenate([np.tile(base.h, k), np.repeat(z_axis.h, n)]),
        dual_face=np.concatenate([np.outer(z_axis.volume, base.dual_face).ravel(), np.tile(base.volume, k - 1)]),
        volume=np.outer(z_axis.volume, base.volume).ravel(),
        edge_nodes=np.concatenate([flat, rise]),
    )
