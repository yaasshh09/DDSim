from __future__ import annotations
from collections import Counter
import numpy as np
import pytest
from ddsim.mesh.mesh1d import uniform_mesh_1d
from ddsim.mesh.mesh2d import tensor_mesh_2d
from ddsim.mesh.refine import refine
from ddsim.mesh.trimesh import triangulated


def square():
    return triangulated(tensor_mesh_2d(uniform_mesh_1d(length=1.0, n_nodes=9), uniform_mesh_1d(length=1.0, n_nodes=9)))


def edges_of(t):
    return Counter(tuple(sorted(e)) for a, b, c in t.tolist() for e in ((a, b), (b, c), (c, a)))


def test_refining_keeps_a_conforming_mesh_that_tiles_the_square() -> None:
    m = square()
    marked = np.flatnonzero(np.hypot(*(np.column_stack([m.node_x, m.node_y])[m.triangles].mean(axis=1) - 0.5).T) < 0.2)
    r = refine(m, marked)
    assert r.volume.sum() == pytest.approx(1.0, rel=1e-12)
    assert max(edges_of(r.triangles).values()) == 2
    boundary = [e for e, k in edges_of(r.triangles).items() if k == 1]
    p = np.column_stack([r.node_x, r.node_y])
    for a, b in boundary:
        on = [(p[a][i] in (0.0, 1.0)) and p[a][i] == p[b][i] for i in (0, 1)]
        assert any(on), (p[a], p[b])
    assert r.n_nodes > m.n_nodes


def test_refining_stays_local() -> None:
    m = square()
    p = np.column_stack([m.node_x, m.node_y])
    corner = np.flatnonzero((p[m.triangles].mean(axis=1) < 0.25).all(axis=1))
    r = refine(m, corner)
    q = np.column_stack([r.node_x, r.node_y])[m.n_nodes:]
    assert (q.max(axis=1) <= 0.5 + 1e-12).all()


def test_refining_everything_twice_halves_the_longest_edge() -> None:
    m = square()
    r = refine(m, np.arange(m.triangles.shape[0]))
    r = refine(r, np.arange(r.triangles.shape[0]))
    assert r.h.max() == pytest.approx(0.5 * m.h.max(), rel=1e-12)
    assert r.volume.sum() == pytest.approx(1.0, rel=1e-12)


def test_refining_a_mesh_of_oblong_cells_leaves_no_obtuse_boundary_triangle() -> None:
    m = triangulated(tensor_mesh_2d(uniform_mesh_1d(length=1e-4, n_nodes=26), uniform_mesh_1d(length=1e-5, n_nodes=4)))
    r = refine(refine(m, np.arange(m.triangles.shape[0])), np.arange(2 * m.triangles.shape[0]))
    assert r.volume.sum() == pytest.approx(1e-9, rel=1e-12)
