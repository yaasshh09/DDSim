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
