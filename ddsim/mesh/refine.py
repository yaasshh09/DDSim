from __future__ import annotations
import math
import numpy as np
import numpy.typing as npt
from ddsim.mesh.trimesh import TriMesh, triangle_mesh

Edge = tuple[int, int]


def _key(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


class _Refiner:
    def __init__(self, mesh: TriMesh) -> None:
        self.p: list[tuple[float, float]] = list(zip(mesh.node_x.tolist(), mesh.node_y.tolist(), strict=True))
        self.t: dict[int, tuple[int, int, int]] = {i: (r[0], r[1], r[2]) for i, r in enumerate(mesh.triangles.tolist())}
        self.next = len(self.t)
        self.by_edge: dict[Edge, set[int]] = {}
        for i, r in self.t.items():
            self._link(i, r)

    def _link(self, i: int, r: tuple[int, int, int]) -> None:
        for a, b in ((r[0], r[1]), (r[1], r[2]), (r[2], r[0])):
            self.by_edge.setdefault(_key(a, b), set()).add(i)

    def _unlink(self, i: int) -> tuple[int, int, int]:
        r = self.t.pop(i)
        for a, b in ((r[0], r[1]), (r[1], r[2]), (r[2], r[0])):
            s = self.by_edge[_key(a, b)]
            s.discard(i)
            if not s:
                del self.by_edge[_key(a, b)]
        return r

    def _add(self, r: tuple[int, int, int]) -> int:
        i = self.next
        self.next += 1
        self.t[i] = r
        self._link(i, r)
        return i

    def _length(self, e: Edge) -> float:
        (x0, y0), (x1, y1) = self.p[e[0]], self.p[e[1]]
        return math.hypot(x1 - x0, y1 - y0)

    def longest(self, i: int) -> Edge:
        r = self.t[i]
        return max((_key(a, b) for a, b in ((r[0], r[1]), (r[1], r[2]), (r[2], r[0]))), key=lambda e: (self._length(e), e))

    def _across(self, i: int, e: Edge) -> int | None:
        o = self.by_edge[e] - {i}
        return next(iter(o)) if o else None

    def _bisect(self, e: Edge) -> None:
        (x0, y0), (x1, y1) = self.p[e[0]], self.p[e[1]]
        self.p.append((0.5 * (x0 + x1), 0.5 * (y0 + y1)))
        m = len(self.p) - 1
        for i in list(self.by_edge[e]):
            r = self._unlink(i)
            c = next(v for v in r if v not in e)
            a, b = e
            self._add((a, m, c))
            self._add((m, b, c))

    def split(self, i: int) -> None:
        stack = [i]
        while stack:
            k = stack[-1]
            if k not in self.t:
                stack.pop()
                continue
            e = self.longest(k)
            nb = self._across(k, e)
            if nb is not None and self.longest(nb) != e:
                stack.append(nb)
                continue
            self._bisect(e)
            stack.pop()

    def flip(self) -> None:
        changed = True
        while changed:
            changed = False
            for e, s in list(self.by_edge.items()):
                if len(s) != 2 or e not in self.by_edge:
                    continue
                i, j = tuple(s)
                c = next(v for v in self.t[i] if v not in e)
                d = next(v for v in self.t[j] if v not in e)
                if self._angle(c, e) + self._angle(d, e) > math.pi * (1 + 1e-12):
                    self._unlink(i)
                    self._unlink(j)
                    self._add((c, d, e[0]))
                    self._add((d, c, e[1]))
                    changed = True

    def encroached(self) -> bool:
        bad = []
        for e, k in self.by_edge.items():
            if len(k) == 1:
                c = next(v for v in self.t[next(iter(k))] if v not in e)
                if self._angle(c, e) > 0.5 * math.pi * (1 + 1e-12):
                    bad.append(e)
        for e in bad:
            if e in self.by_edge:
                self._bisect(e)
        return bool(bad)

    def _angle(self, c: int, e: Edge) -> float:
        (cx, cy), (ax, ay), (bx, by) = self.p[c], self.p[e[0]], self.p[e[1]]
        u, v = (ax - cx, ay - cy), (bx - cx, by - cy)
        return math.atan2(abs(u[0] * v[1] - u[1] * v[0]), u[0] * v[0] + u[1] * v[1])

    def mesh(self, cylindrical: bool) -> TriMesh:
        return triangle_mesh(np.asarray(self.p), np.asarray(list(self.t.values()), dtype=np.int64), cylindrical=cylindrical)


def refine(mesh: TriMesh, marked: npt.NDArray[np.int64]) -> TriMesh:
    r = _Refiner(mesh)
    for i in sorted(set(int(k) for k in np.asarray(marked).ravel())):
        if not 0 <= i < mesh.triangles.shape[0]:
            raise IndexError(f"triangle {i} is not in a mesh of {mesh.triangles.shape[0]} triangles")
        r.split(i)
    r.flip()
    while r.encroached():
        r.flip()
    return r.mesh(mesh.cylindrical)
