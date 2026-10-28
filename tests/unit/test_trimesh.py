from __future__ import annotations
import numpy as np
import pytest
from ddsim.discretize.coupled import coupled_residual, pack
from ddsim.mesh.mesh1d import graded_mesh_1d, uniform_mesh_1d
from ddsim.mesh.mesh2d import tensor_mesh_2d
from ddsim.mesh.quality import MeshQualityError
from ddsim.mesh.trimesh import delaunay_mesh, triangle_mesh, triangulated
from ddsim.physics.recombination import NoRecombination


def tensor():
    return tensor_mesh_2d(graded_mesh_1d(length=1e-4, n_nodes=21, refine_at=0.4e-4, h_min=2e-6), uniform_mesh_1d(length=3e-5, n_nodes=5))


def by_edge(edges, values):
    return {tuple(sorted(e)): v for e, v in zip(edges.tolist(), values.tolist(), strict=True)}


def test_a_triangulated_tensor_mesh_keeps_every_volume() -> None:
    t = tensor()
    m = triangulated(t)
    np.testing.assert_allclose(m.volume, t.volume, rtol=1e-12, atol=0)
    assert m.volume.sum() == pytest.approx(1e-4 * 3e-5, rel=1e-12)


def test_a_triangulated_tensor_mesh_keeps_every_coupling_and_the_diagonals_carry_none() -> None:
    t = tensor()
    m = triangulated(t)
    want = by_edge(t.edge_nodes, t.dual_face)
    got = by_edge(m.edge_nodes, m.dual_face)
    for e, f in got.items():
        if e in want:
            assert f == pytest.approx(want[e], rel=1e-12)
        else:
            assert abs(f) < 1e-12 * t.dual_face.max()
    assert set(want) <= set(got)
    assert len(got) == len(want) + 20 * 4


def test_the_residual_on_the_triangulated_mesh_is_the_tensor_one() -> None:
    t = tensor()
    m = triangulated(t)
    rng = np.random.default_rng(7)
    N = t.n_nodes
    x = pack(rng.uniform(-5, 5, N), np.exp(rng.uniform(-3, 3, N)), np.exp(rng.uniform(-3, 3, N)))
    doping = rng.uniform(-10, 10, N)
    a = coupled_residual(t.h, t.volume, x, doping, 1.0, 0.4, NoRecombination(), t.edge_geometry())
    b = coupled_residual(m.h, m.volume, x, doping, 1.0, 0.4, NoRecombination(), m.edge_geometry())
    np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-12 * np.abs(a).max())


def test_an_obtuse_boundary_triangle_is_refused() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.1], [0.5, 1.0]])
    triangles = np.array([[0, 1, 2], [0, 2, 3], [1, 3, 2]])
    with pytest.raises(MeshQualityError, match="negative"):
        triangle_mesh(points, triangles)


def test_a_delaunay_mesh_of_a_rectangle_tiles_it() -> None:
    rng = np.random.default_rng(11)
    t = tensor()
    jiggle = np.column_stack([t.node_x, t.node_y])
    inside = (t.node_x > 0) & (t.node_x < 1e-4) & (t.node_y > 0) & (t.node_y < 3e-5)
    jiggle[inside] += rng.uniform(-0.15, 0.15, (int(inside.sum()), 2)) * np.array([1.5e-6, 1.5e-6])
    m = delaunay_mesh(jiggle)
    assert m.volume.sum() == pytest.approx(1e-4 * 3e-5, rel=1e-12)
    assert (m.dual_face >= -1e-12 * m.dual_face.max()).all()
