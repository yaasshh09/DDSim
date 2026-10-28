from __future__ import annotations
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import numpy.typing as npt
from ddsim.mesh.trimesh import TriMesh, triangle_mesh

LINE = 1
TRIANGLE = 2


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


def _sections(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    name: str | None = None
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("$End"):
            name = None
        elif s.startswith("$"):
            name = s[1:]
            out[name] = []
        elif name is not None and s:
            out[name].append(s)
    return out


def _physical_of_entities(lines: list[str]) -> dict[tuple[int, int], int]:
    it: Iterator[str] = iter(lines)
    counts = [int(v) for v in next(it).split()]
    out: dict[tuple[int, int], int] = {}
    for dim, count in enumerate(counts):
        for _ in range(count):
            f = next(it).split()
            tag = int(f[0])
            k = 4 if dim == 0 else 7
            n = int(f[k])
            if n:
                out[(dim, tag)] = abs(int(f[k + 1]))
    return out


def read_msh(path: str | Path) -> GmshMesh:
    s = _sections(Path(path).read_text(encoding="utf-8"))
    version = s.get("MeshFormat", ["?"])[0].split()[0]
    if version != "4.1" or s["MeshFormat"][0].split()[1] != "0":
        raise ValueError(f"{path} is Gmsh format {version}, and this reader takes 4.1 ASCII only. Save it from Gmsh with Mesh.MshFileVersion = 4.1.")
    names = {}
    for line in s.get("PhysicalNames", [])[1:]:
        d, t, name = line.split(maxsplit=2)
        names[(int(d), int(t))] = name.strip('"')
    physical = _physical_of_entities(s.get("Entities", ["0 0 0 0"]))

    it = iter(s["Nodes"])
    blocks = int(next(it).split()[0])
    tags: list[int] = []
    xy: list[tuple[float, float]] = []
    for _ in range(blocks):
        _, _, parametric, n = (int(v) for v in next(it).split())
        if parametric:
            raise ValueError(f"{path} stores parametric node coordinates, which this reader does not take")
        tags += [int(next(it)) for _ in range(n)]
        for _ in range(n):
            x, y, _z = (float(v) for v in next(it).split())
            xy.append((x, y))
    index = {t: i for i, t in enumerate(tags)}

    it = iter(s["Elements"])
    blocks = int(next(it).split()[0])
    tri: list[list[int]] = []
    region: list[str] = []
    seg: list[list[int]] = []
    contact: list[str] = []
    for _ in range(blocks):
        dim, entity, kind, n = (int(v) for v in next(it).split())
        group = names.get((dim, physical.get((dim, entity), -1)), "")
        for _ in range(n):
            f = [index[int(v)] for v in next(it).split()[1:]]
            if kind == TRIANGLE:
                tri.append(f)
                region.append(group)
            elif kind == LINE:
                seg.append(f)
                contact.append(group)
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
