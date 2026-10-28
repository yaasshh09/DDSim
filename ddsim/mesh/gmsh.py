from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import numpy.typing as npt
from ddsim.mesh.msh41 import LINE, TRIANGLE, parse
from ddsim.mesh.trimesh import TriMesh, triangle_mesh


@dataclass(frozen=True)
class GmshMesh:
    points: npt.NDArray[np.float64]

    triangles: npt.NDArray[np.int64]

    region_of_triangles: tuple[str, ...]

    lines: npt.NDArray[np.int64]

    contact_of_lines: tuple[str, ...]

    def contact_nodes(self, name: str) -> tuple[int, ...]:
        if name not in self.contact_of_lines:
            raise KeyError(f"the mesh has no contact named {name!r}. Its line groups are {sorted(set(self.contact_of_lines))}")
        k = [i for i, c in enumerate(self.contact_of_lines) if c == name]
        return tuple(int(v) for v in np.unique(self.lines[k]))

    def mesh(self) -> TriMesh:
        return triangle_mesh(self.points, self.triangles)


def read_msh(path: str | Path) -> GmshMesh:
    xy, tri, region, seg, contact = parse(Path(path).read_text(encoding="utf-8"), str(path))
    return GmshMesh(points=np.asarray(xy, dtype=np.float64), triangles=np.asarray(tri, dtype=np.int64).reshape(-1, 3), region_of_triangles=tuple(region), lines=np.asarray(seg, dtype=np.int64).reshape(-1, 2), contact_of_lines=tuple(contact))


def write_msh(path: str | Path, points: npt.NDArray[np.float64], triangles: npt.NDArray[np.int64], regions: Mapping[str, npt.NDArray[np.int64]], contacts: Mapping[str, npt.NDArray[np.int64]]) -> None:
    p = np.asarray(points, dtype=np.float64)
    lo, hi = p.min(axis=0), p.max(axis=0)
    box = f"{lo[0]!r} {lo[1]!r} 0 {hi[0]!r} {hi[1]!r} 0"
    groups = [(1, name) for name in contacts] + [(2, name) for name in regions]
    out = ["$MeshFormat", "4.1 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(groups))]
    out += [f'{dim} {k + 1} "{name}"' for k, (dim, name) in enumerate(groups)]
    out += ["$EndPhysicalNames", "$Entities", f"0 {len(contacts)} {len(regions)} 0"]
    out += [f"{k + 1} {box} 1 {k + 1} 0" for k in range(len(contacts))]
    out += [f"{k + 1} {box} 1 {len(contacts) + k + 1} 0" for k in range(len(regions))]
    out += ["$EndEntities", "$Nodes", f"1 {p.shape[0]} 1 {p.shape[0]}", f"2 1 0 {p.shape[0]}"]
    out += [str(i + 1) for i in range(p.shape[0])]
    out += [f"{x!r} {y!r} 0" for x, y in p.tolist()]
    n_elements = sum(len(v) for v in contacts.values()) + sum(len(v) for v in regions.values())
    out += ["$EndNodes", "$Elements", f"{len(contacts) + len(regions)} {n_elements} 1 {n_elements}"]
    tag = 0
    for k, lines in enumerate(contacts.values()):
        out.append(f"1 {k + 1} {LINE} {len(lines)}")
        for a, b in np.asarray(lines).tolist():
            tag += 1
            out.append(f"{tag} {a + 1} {b + 1}")
    t = np.asarray(triangles)
    for k, which in enumerate(regions.values()):
        out.append(f"2 {k + 1} {TRIANGLE} {len(which)}")
        for a, b, c in t[np.asarray(which)].tolist():
            tag += 1
            out.append(f"{tag} {a + 1} {b + 1} {c + 1}")
    out.append("$EndElements")
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")
