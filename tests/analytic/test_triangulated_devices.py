from __future__ import annotations
import pytest
from ddsim.device.builder import build_device
from ddsim.device.doping import abrupt_junction
from ddsim.device.transport import TransportModels
from ddsim.discretize.boundary import OhmicPlate
from ddsim.extract.iv import gate_sweep
from ddsim.mesh.mesh1d import graded_mesh_1d, uniform_mesh_1d
from ddsim.mesh.mesh2d import tensor_mesh_2d
from ddsim.mesh.trimesh import triangulated

V = [0.0, -0.5, 0.3, 0.5]


def diode(mesh, nx, ny):
    left = tuple(j * nx for j in range(ny))
    right = tuple(j * nx + nx - 1 for j in range(ny))
    return build_device(mesh=mesh, doping=abrupt_junction(Na=1e16, Nd=1e16, position=0.5e-4), contacts=(OhmicPlate(name="anode", nodes=left, voltage=0.0), OhmicPlate(name="cathode", nodes=right, voltage=0.0)))


def test_a_triangulated_diode_carries_the_tensor_current() -> None:
    nx, ny = 81, 4
    t = tensor_mesh_2d(graded_mesh_1d(length=1e-4, n_nodes=nx, refine_at=0.5e-4, h_min=2e-7), uniform_mesh_1d(length=1e-5, n_nodes=ny))
    a, b = diode(t, nx, ny), diode(triangulated(t), nx, ny)
    x = gate_sweep(a, V, contact="anode", measure_at="anode", models=TransportModels.for_device(a))
    y = gate_sweep(b, V, contact="anode", measure_at="anode", models=TransportModels.for_device(b))
    assert x.complete and y.complete
    for p, q in zip(x.points[1:], y.points[1:], strict=True):
        assert q.current == pytest.approx(p.current, rel=1e-7, abs=0)


def test_a_triangulated_mos_capacitor_keeps_its_regions_and_its_capacitance() -> None:
    import numpy as np
    from ddsim.device.doping import Uniform
    from ddsim.device.mos_cap import mos_cap
    from ddsim.device.regions import triangle_regions
    from ddsim.extract.cv import cv_sweep
    d = mos_cap()
    m = triangulated(d.mesh)
    cells = d.regions.cell_material.ravel()
    r = triangle_regions(m, np.concatenate([cells, cells]))
    np.testing.assert_allclose(r.semiconductor_volume, d.regions.semiconductor_volume, rtol=1e-12, atol=1e-12 * d.regions.semiconductor_volume.max())
    assert r.oxide_nodes.tolist() == d.regions.oxide_nodes.tolist()
    tensor = {tuple(sorted(e)): (eps * f, s) for e, eps, f, s in zip(d.mesh.edge_nodes.tolist(), d.regions.eps_r, d.mesh.dual_face, d.regions.semiconductor_face, strict=True)}
    for e, eps, f, s in zip(m.edge_nodes.tolist(), r.eps_r, m.dual_face, r.semiconductor_face, strict=True):
        if tuple(e) in tensor:
            assert eps * f == pytest.approx(tensor[tuple(e)][0], rel=1e-12)
            assert s == pytest.approx(tensor[tuple(e)][1], rel=1e-12, abs=1e-12 * m.dual_face.max())
    t = build_device(mesh=m, doping=Uniform(d.doping.value), contacts=d.contacts, regions=r)
    a = cv_sweep(d, "gate", [-1.0, 0.0, 1.0])
    b = cv_sweep(t, "gate", [-1.0, 0.0, 1.0])
    for p, q in zip(a.points, b.points, strict=True):
        assert q.capacitance == pytest.approx(p.capacitance, rel=1e-7, abs=0)
