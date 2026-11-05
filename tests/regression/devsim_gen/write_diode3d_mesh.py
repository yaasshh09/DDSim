from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import numpy.typing as npt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ddsim.mesh.gmsh import write_msh_volume
from ddsim.mesh.mesh1d import graded_mesh_1d, uniform_mesh_1d
from ddsim.mesh.mesh2d import tensor_mesh_2d
from ddsim.mesh.mesh3d import extruded
from ddsim.mesh.tetmesh import TetMesh, tetrahedralized
from tests.regression.devsim_gen import parameters as P

PATH = os.path.join("data", "golden", "diode3d_gmsh.msh")


def mesh() -> TetMesh:
    b = P.DIODE3D_BENCHMARK
    side = uniform_mesh_1d(length=P.DIODE3D_WIDTH, n_nodes=9)
    return tetrahedralized(extruded(tensor_mesh_2d(graded_mesh_1d(length=b.length, n_nodes=b.n_nodes, refine_at=b.junction, h_min=b.h_min), side), side))


def anode(m: TetMesh) -> npt.NDArray[np.bool_]:
    return np.asarray(m.node_x == 0.0)


def cathode(m: TetMesh) -> npt.NDArray[np.bool_]:
    return np.asarray((m.node_x == m.node_x.max()) & (m.node_y <= P.DIODE3D_PATCH * (1 + 1e-12)) & (m.node_z <= P.DIODE3D_PATCH * (1 + 1e-12)))


def faces(m: TetMesh, on: npt.NDArray[np.bool_]) -> npt.NDArray[np.int64]:
    out = []
    for skip in range(4):
        f = np.delete(m.tets, skip, axis=1)
        out.append(f[on[f].all(axis=1)])
    return np.unique(np.sort(np.concatenate(out), axis=1), axis=0)


def main() -> int:
    a = argparse.ArgumentParser(description="Write the 3D p+n diode as a tetrahedral mesh in Gmsh 4.1 format, for benchmark 19.")
    a.add_argument("--out", default=PATH)
    o = a.parse_args()
    m = mesh()
    contacts = {"anode": faces(m, anode(m)), "cathode": faces(m, cathode(m))}
    write_msh_volume(o.out, np.column_stack([m.node_x, m.node_y, m.node_z]), m.tets, {"bulk": np.arange(m.tets.shape[0])}, contacts)
    print(f"wrote {o.out}: {m.n_nodes} nodes, {m.tets.shape[0]} tetrahedra, {len(contacts['anode'])} anode and {len(contacts['cathode'])} cathode triangles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
