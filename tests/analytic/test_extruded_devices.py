from __future__ import annotations
import numpy as np
import pytest
from ddsim.device.builder import build_device
from ddsim.device.doping import abrupt_junction
from ddsim.device.transport import TransportModels
from ddsim.discretize.boundary import OhmicPlate
from ddsim.extract.iv import gate_sweep
from ddsim.mesh.mesh1d import graded_mesh_1d, uniform_mesh_1d
from ddsim.mesh.mesh2d import tensor_mesh_2d
from ddsim.mesh.mesh3d import extruded
from ddsim.mesh.trimesh import delaunay_mesh

V = [0.0, -0.5, 0.3, 0.5]


def diode(mesh):
    left = tuple(np.flatnonzero(mesh.node_x == mesh.node_x.min()).tolist())
    right = tuple(np.flatnonzero(mesh.node_x == mesh.node_x.max()).tolist())
    return build_device(mesh=mesh, doping=abrupt_junction(Na=1e16, Nd=1e16, position=0.5e-4), contacts=(OhmicPlate(name="anode", nodes=left, voltage=0.0), OhmicPlate(name="cathode", nodes=right, voltage=0.0)))


def sweep(d):
    return gate_sweep(d, V, contact="anode", measure_at="anode", models=TransportModels.for_device(d), iterative=False)


def test_an_extruded_mesh_tiles_the_box() -> None:
    base = tensor_mesh_2d(uniform_mesh_1d(length=2e-4, n_nodes=5), uniform_mesh_1d(length=1e-4, n_nodes=4))
    z = graded_mesh_1d(length=3e-4, n_nodes=17, refine_at=1e-4, h_min=1e-5)
    m = extruded(base, z)
    assert m.n_nodes == base.n_nodes * z.n_nodes
    assert m.n_edges == base.n_edges * z.n_nodes + base.n_nodes * (z.n_nodes - 1)
    assert m.volume.sum() == pytest.approx(2e-4 * 1e-4 * 3e-4, rel=1e-12)
    vertical = m.dual_face[base.n_edges * z.n_nodes:]
    assert vertical.sum() == pytest.approx(2e-4 * 1e-4 * (z.n_nodes - 1), rel=1e-12)
    assert np.all(m.h > 0.0) and np.all(m.dual_face > 0.0)


def test_a_box_is_the_same_box_whichever_axis_is_extruded() -> None:
    x = graded_mesh_1d(length=2e-4, n_nodes=14, refine_at=0.5e-4, h_min=5e-6)
    y = uniform_mesh_1d(length=1e-4, n_nodes=4)
    z = graded_mesh_1d(length=3e-4, n_nodes=17, refine_at=1e-4, h_min=1e-5)
    a = extruded(tensor_mesh_2d(x, y), z)
    b = extruded(tensor_mesh_2d(x, z), y)

    def table(m, p):
        at = np.round(np.column_stack(p) / 1e-9).astype(np.int64)
        e = {tuple(sorted((tuple(at[u]), tuple(at[v])))): (h, f) for (u, v), h, f in zip(m.edge_nodes.tolist(), m.h, m.dual_face, strict=True)}
        n = {tuple(at[k]): w for k, w in enumerate(m.volume)}
        return e, n

    ea, na = table(a, (a.node_x, a.node_y, a.node_z))
    eb, nb = table(b, (b.node_x, b.node_z, b.node_y))
    assert ea.keys() == eb.keys() and na.keys() == nb.keys()
    for k in ea:
        assert ea[k][0] == pytest.approx(eb[k][0], rel=1e-12)
        assert ea[k][1] == pytest.approx(eb[k][1], rel=1e-12, abs=0)
    for k in na:
        assert na[k] == pytest.approx(nb[k], rel=1e-12, abs=0)


def test_an_extruded_diode_carries_the_2d_current_times_its_depth() -> None:
    nx, ny = 81, 4
    flat = tensor_mesh_2d(graded_mesh_1d(length=1e-4, n_nodes=nx, refine_at=0.5e-4, h_min=2e-7), uniform_mesh_1d(length=1e-5, n_nodes=ny))
    z = graded_mesh_1d(length=3e-5, n_nodes=8, refine_at=1e-5, h_min=3e-6)
    x, y = sweep(diode(flat)), sweep(diode(extruded(flat, z)))
    assert x.complete and y.complete
    for p, q in zip(x.points[1:], y.points[1:], strict=True):
        assert q.current == pytest.approx(p.current * z.length, rel=1e-10, abs=0)


def test_an_extruded_triangle_mesh_carries_the_2d_current_times_its_depth() -> None:
    rng = np.random.default_rng(3)
    gx = graded_mesh_1d(length=1e-4, n_nodes=61, refine_at=0.5e-4, h_min=4e-7).x
    gy = uniform_mesh_1d(length=1e-5, n_nodes=4).x
    px, py = np.meshgrid(gx, gy)
    inner = (px > 0) & (px < 1e-4) & (py > 0) & (py < 1e-5)
    px = px + inner * rng.uniform(-0.1, 0.1, px.shape) * np.gradient(gx)[None, :]
    flat = delaunay_mesh(np.column_stack([px.ravel(), py.ravel()]))
    z = uniform_mesh_1d(length=2e-5, n_nodes=3)
    x, y = sweep(diode(flat)), sweep(diode(extruded(flat, z)))
    assert x.complete and y.complete
    for p, q in zip(x.points[1:], y.points[1:], strict=True):
        assert q.current == pytest.approx(p.current * z.length, rel=1e-10, abs=0)


def test_a_cut_through_an_extruded_diode_carries_the_terminal_current() -> None:
    from ddsim.device.transport import solve_bias_newton
    from ddsim.extract.iv import current_densities, edge_current_face, terminal_currents
    from ddsim.physics.recombination import NoRecombination
    flat = tensor_mesh_2d(graded_mesh_1d(length=1e-4, n_nodes=41, refine_at=0.5e-4, h_min=1e-6), uniform_mesh_1d(length=1e-5, n_nodes=3))
    d = diode(extruded(flat, uniform_mesh_1d(length=2e-5, n_nodes=3))).with_bias(anode=0.4)
    models = TransportModels.for_device(d, recombination=NoRecombination())
    s = solve_bias_newton(d, models=models)
    assert s.newton is not None and s.newton.converged
    jn, jp = current_densities(d, s, models)
    flux = (jn.data + jp.data) * edge_current_face(d)
    u, v = d.mesh.edge_nodes.T
    for cut in (0.2137e-4, 0.5013e-4, 0.7911e-4):
        across = (np.minimum(d.mesh.node_x[u], d.mesh.node_x[v]) < cut) & (np.maximum(d.mesh.node_x[u], d.mesh.node_x[v]) > cut)
        assert across.sum() == 3 * 3
        assert abs(flux[across].sum()) == pytest.approx(abs(terminal_currents(d, s, models)["anode"]), rel=1e-6)


def test_a_diode_split_into_tetrahedra_carries_the_box_current() -> None:
    from ddsim.mesh.tetmesh import tetrahedralized
    flat = tensor_mesh_2d(graded_mesh_1d(length=1e-4, n_nodes=41, refine_at=0.5e-4, h_min=1e-6), uniform_mesh_1d(length=1e-5, n_nodes=3))
    box = extruded(flat, uniform_mesh_1d(length=2e-5, n_nodes=3))
    x, y = sweep(diode(box)), sweep(diode(tetrahedralized(box)))
    assert x.complete and y.complete
    for p, q in zip(x.points[1:], y.points[1:], strict=True):
        assert q.current == pytest.approx(p.current, rel=1e-9, abs=0)


def test_box_regions_split_every_face_and_volume_between_their_cells() -> None:
    from ddsim.device.regions import OXIDE, SILICON, box_regions
    base = tensor_mesh_2d(graded_mesh_1d(length=2e-4, n_nodes=14, refine_at=0.5e-4, h_min=5e-6), uniform_mesh_1d(length=1e-4, n_nodes=5))
    m = extruded(base, graded_mesh_1d(length=3e-4, n_nodes=17, refine_at=1e-4, h_min=1e-5))
    cells = np.full((16, 4, 13), SILICON)
    r = box_regions(m, cells)
    np.testing.assert_allclose(r.semiconductor_volume, m.volume, rtol=1e-12)
    np.testing.assert_allclose(r.semiconductor_face, m.dual_face, rtol=1e-12)
    np.testing.assert_allclose(r.eps_r, 1.0, rtol=1e-12)
    cells[:, 2:, :] = OXIDE
    r = box_regions(m, cells)
    assert r.semiconductor_volume.sum() == pytest.approx(2e-4 * base.y_axis.x[2] * 3e-4, rel=1e-12)
    top = np.flatnonzero(m.node_y > base.y_axis.x[2])
    assert set(r.oxide_nodes.tolist()) == set(top.tolist())


def test_an_extruded_mosfet_carries_the_2d_drain_current_times_its_width() -> None:
    from dataclasses import replace
    from ddsim.device.mosfet import nmos
    from ddsim.device.regions import box_regions
    d = nmos(L_gate=1e-4, drain_voltage=0.1, degenerate=False, n_contact=3, n_sd=8, n_channel=10, n_silicon=16, n_oxide=3, h_min_x=1e-6, h_min_y=2e-7)
    z = graded_mesh_1d(length=2e-5, n_nodes=6, refine_at=0.0, h_min=2e-6)
    m = extruded(d.mesh, z)
    n = d.mesh.n_nodes
    contacts = tuple(replace(c, nodes=tuple(v + k * n for k in range(z.n_nodes) for v in c.nodes)) for c in d.contacts)
    e = build_device(mesh=m, doping=d.doping, contacts=contacts, regions=box_regions(m, np.tile(d.regions.cell_material[None], (z.n_nodes - 1, 1, 1))), degenerate=False)
    models = TransportModels.for_device(d, mobility="constant")
    x = gate_sweep(d, [0.0, 0.5, 1.0], models=models, iterative=False)
    y = gate_sweep(e, [0.0, 0.5, 1.0], models=TransportModels.for_device(e, mobility="constant"), iterative=False)
    assert x.complete and y.complete
    for p, q in zip(x.points[1:], y.points[1:], strict=True):
        assert q.current == pytest.approx(p.current * z.length, rel=1e-9, abs=0)
