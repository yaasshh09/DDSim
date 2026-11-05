from __future__ import annotations
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg
from ddsim.mesh.mesh1d import graded_mesh_1d, uniform_mesh_1d
from ddsim.mesh.mesh2d import tensor_mesh_2d
from ddsim.mesh.mesh3d import extruded


def capacitance(m, high, low) -> float:
    w = m.dual_face / m.h
    e = m.edge_nodes
    N = m.n_nodes
    K = coo_matrix((np.concatenate([w, w, -w, -w]), (np.concatenate([e[:, 0], e[:, 1], e[:, 0], e[:, 1]]), np.concatenate([e[:, 0], e[:, 1], e[:, 1], e[:, 0]]))), shape=(N, N)).tocsr()
    fixed = np.concatenate([high, low])
    psi = np.zeros(N)
    psi[high] = 1.0
    free = np.setdiff1d(np.arange(N), fixed)
    x, info = cg(K[free][:, free], -K[free][:, fixed] @ psi[fixed], rtol=1e-13, maxiter=50000)
    assert info == 0
    psi[free] = x
    return float((K @ psi)[high].sum())


def test_plates_across_the_whole_box_give_eps_a_over_d() -> None:
    x = graded_mesh_1d(length=2.0, n_nodes=14, refine_at=0.5, h_min=0.05)
    y = uniform_mesh_1d(length=3.0, n_nodes=5)
    z = graded_mesh_1d(length=0.5, n_nodes=17, refine_at=0.1, h_min=0.01)
    m = extruded(tensor_mesh_2d(x, y), z)
    c = capacitance(m, np.flatnonzero(m.node_z == 0.5), np.flatnonzero(m.node_z == 0.0))
    assert c == pytest.approx(2.0 * 3.0 / 0.5, rel=1e-10)


def cube(n: int) -> float:
    a = uniform_mesh_1d(length=3.0, n_nodes=n)
    m = extruded(tensor_mesh_2d(a, a), a)
    p = np.column_stack([m.node_x, m.node_y, m.node_z])
    inside = np.all(np.abs(p - 1.5) <= 0.5 + 1e-12, axis=1)
    wall = np.any((p == 0.0) | (p == 3.0), axis=1)
    return capacitance(m, np.flatnonzero(inside), np.flatnonzero(wall)) / (4 * np.pi)


def test_a_cube_in_a_grounded_box_converges_between_its_sphere_bounds() -> None:
    c = [cube(n) for n in (13, 25, 49)]
    low = 0.5 * 1.5 * np.sqrt(3) / (1.5 * np.sqrt(3) - 0.5)
    high = 0.5 * np.sqrt(3) * 1.5 / (1.5 - 0.5 * np.sqrt(3))
    assert all(low < v < high for v in c)
    assert abs(c[2] - c[1]) < 0.5 * abs(c[1] - c[0])


def plates(ratio: float, per_gap: int) -> float:
    W, margin = ratio, 3.0
    L, H = W + 2 * margin, 1.0 + 2 * margin
    a = uniform_mesh_1d(length=L, n_nodes=int(round(L * per_gap)) + 1)
    m = extruded(tensor_mesh_2d(a, a), uniform_mesh_1d(length=H, n_nodes=int(round(H * per_gap)) + 1))
    on = (np.abs(m.node_x - L / 2) <= W / 2 + 1e-9) & (np.abs(m.node_y - L / 2) <= W / 2 + 1e-9)
    low, high = np.flatnonzero(on & np.isclose(m.node_z, margin)), np.flatnonzero(on & np.isclose(m.node_z, margin + 1.0))
    return capacitance(m, high, low) / (W * W) - 1.0


def test_fringing_adds_capacitance_and_fades_as_the_plates_widen() -> None:
    e = [plates(4.0, g) for g in (2, 4, 8)]
    assert all(v > 0.0 for v in e)
    assert abs(e[2] - e[1]) < 0.5 * abs(e[1] - e[0])
    wide = plates(8.0, 4)
    assert 0.0 < wide < 0.6 * e[1]
