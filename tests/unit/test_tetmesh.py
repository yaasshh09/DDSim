from __future__ import annotations
import numpy as np
import pytest
from ddsim.mesh.mesh1d import graded_mesh_1d, uniform_mesh_1d
from ddsim.mesh.mesh2d import tensor_mesh_2d
from ddsim.mesh.mesh3d import extruded
from ddsim.mesh.quality import MeshQualityError
from ddsim.mesh.tetmesh import tet_mesh, tetrahedralized


def box():
    x = graded_mesh_1d(length=2.0, n_nodes=14, refine_at=0.5, h_min=0.05)
    y = uniform_mesh_1d(length=1.0, n_nodes=4)
    z = graded_mesh_1d(length=3.0, n_nodes=17, refine_at=1.0, h_min=0.1)
    return extruded(tensor_mesh_2d(x, y), z)


def test_a_box_split_into_tetrahedra_keeps_the_tensor_couplings() -> None:
    b = box()
    t = tetrahedralized(b)
    tensor = {tuple(sorted(e)): f for e, f in zip(b.edge_nodes.tolist(), b.dual_face, strict=True)}
    scale = b.dual_face.max()
    for e, f in zip(t.edge_nodes.tolist(), t.dual_face, strict=True):
        if tuple(e) in tensor:
            assert f == pytest.approx(tensor[tuple(e)], rel=1e-12)
        else:
            assert abs(f) < 1e-12 * scale
    np.testing.assert_allclose(t.volume, b.volume, rtol=1e-12)
    assert len(tensor) == int(np.count_nonzero(t.dual_face > 1e-12 * scale))


def test_a_regular_tetrahedron_couples_every_edge_equally() -> None:
    p = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    m = tet_mesh(p, np.array([[0, 1, 2, 3]]))
    assert m.n_edges == 6
    np.testing.assert_allclose(m.dual_face, m.dual_face[0], rtol=1e-12)
    assert m.dual_face[0] > 0.0
    assert m.volume.sum() == pytest.approx(8.0 / 3.0, rel=1e-12)
    np.testing.assert_allclose(m.volume, m.volume[0], rtol=1e-12)


def test_the_orientation_of_a_tetrahedron_does_not_matter() -> None:
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.2, 1.1, 0.1], [0.3, 0.2, 0.9]])
    a = tet_mesh(p, np.array([[0, 1, 2, 3]]))
    b = tet_mesh(p, np.array([[1, 0, 2, 3]]))
    np.testing.assert_allclose(a.dual_face, b.dual_face, rtol=1e-12)
    np.testing.assert_allclose(a.volume, b.volume, rtol=1e-12)


def test_a_flat_tetrahedron_with_its_centre_outside_is_refused() -> None:
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.2, 0.05]])
    with pytest.raises(MeshQualityError, match="negative coupling"):
        tet_mesh(p, np.array([[0, 1, 2, 3]]))


def test_a_tetrahedron_without_volume_is_refused() -> None:
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    with pytest.raises(MeshQualityError, match="no volume"):
        tet_mesh(p, np.array([[0, 1, 2, 3]]))


def test_a_corner_tetrahedron_couples_its_hypotenuses_negatively_on_its_own() -> None:
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(MeshQualityError, match="3 edge"):
        tet_mesh(p, np.array([[0, 1, 2, 3]]))
