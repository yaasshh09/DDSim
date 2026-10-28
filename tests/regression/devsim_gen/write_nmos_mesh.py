from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import numpy.typing as npt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ddsim.device.builder import Device
from ddsim.device.mosfet import nmos
from ddsim.device.regions import OXIDE, SILICON
from ddsim.extract.rolloff import SHORT_CHANNEL_PROCESS
from ddsim.mesh.gmsh import write_msh
from ddsim.mesh.trimesh import TriMesh, triangulated

L_GATE = 1e-4

PATH = os.path.join("data", "golden", "nmos_1um_gmsh.msh")


def device() -> Device:
    return nmos(L_gate=L_GATE, degenerate=False, **SHORT_CHANNEL_PROCESS)


def materials(d: Device) -> npt.NDArray[np.int64]:
    assert d.regions is not None
    c = d.regions.cell_material.ravel()
    return np.concatenate([c, c])


def interface(m: TriMesh, material: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    seen: dict[tuple[int, int], set[int]] = {}
    for t, k in zip(m.triangles.tolist(), material.tolist(), strict=True):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            seen.setdefault((min(a, b), max(a, b)), set()).add(k)
    return np.array(sorted(e for e, s in seen.items() if s == {SILICON, OXIDE}), dtype=np.int64)


def lines(m: TriMesh, nodes: tuple[int, ...]) -> npt.NDArray[np.int64]:
    k = np.zeros(m.n_nodes, dtype=bool)
    k[list(nodes)] = True
    on = k[m.edge_nodes[:, 0]] & k[m.edge_nodes[:, 1]] & (m.dual_face > 0.0)
    return m.edge_nodes[on]


def main() -> int:
    a = argparse.ArgumentParser(description="Write the 1 um nmos as a triangle mesh in Gmsh 4.1 format, for benchmark 18.")
    a.add_argument("--out", default=PATH)
    o = a.parse_args()
    d = device()
    m = triangulated(d.mesh)
    mat = materials(d)
    contacts = {c.name: lines(m, c.nodes) for c in d.contacts}
    contacts["si_ox"] = interface(m, mat)
    write_msh(o.out, np.column_stack([m.node_x, m.node_y]), m.triangles, {"bulk": np.flatnonzero(mat == SILICON), "oxide": np.flatnonzero(mat == OXIDE)}, contacts)
    print(f"wrote {o.out}: {m.n_nodes} nodes, {m.triangles.shape[0]} triangles, groups {sorted(contacts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
