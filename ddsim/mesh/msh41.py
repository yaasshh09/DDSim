from __future__ import annotations
from collections.abc import Iterator

LINE = 1
TRIANGLE = 2


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


def parse(text: str, where: str = "the mesh") -> tuple[list[tuple[float, float]], list[list[int]], list[str], list[list[int]], list[str]]:
    s = _sections(text)
    version = s.get("MeshFormat", ["?"])[0].split()[0]
    if version != "4.1" or s["MeshFormat"][0].split()[1] != "0":
        raise ValueError(f"{where} is Gmsh format {version}, and this reader takes 4.1 ASCII only. Save it from Gmsh with Mesh.MshFileVersion = 4.1.")
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
            raise ValueError(f"{where} stores parametric node coordinates, which this reader does not take")
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
    return xy, tri, region, seg, contact
