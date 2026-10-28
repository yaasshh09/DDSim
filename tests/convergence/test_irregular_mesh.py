from __future__ import annotations
import math
from functools import cache
import numpy as np
from ddsim.device.builder import build_device
from ddsim.device.doping import abrupt_junction
from ddsim.device.transport import TransportModels
from ddsim.discretize.boundary import OhmicPlate
from ddsim.extract.iv import gate_sweep
from ddsim.mesh.mesh1d import uniform_mesh_1d
from ddsim.mesh.mesh2d import tensor_mesh_2d
from ddsim.mesh.trimesh import delaunay_mesh

ROWS = 4
JIGGLE = 0.15


def current(mesh) -> float:
    left = tuple(np.flatnonzero(mesh.node_x == 0.0))
    right = tuple(np.flatnonzero(mesh.node_x == mesh.node_x.max()))
    d = build_device(mesh=mesh, doping=abrupt_junction(Na=1e16, Nd=1e16, position=0.5e-4), contacts=(OhmicPlate(name="anode", nodes=left, voltage=0.0), OhmicPlate(name="cathode", nodes=right, voltage=0.0)))
    c = gate_sweep(d, [0.0, 0.3, 0.5], contact="anode", measure_at="anode", models=TransportModels.for_device(d))
    assert c.complete, c.message
    return c.points[-1].current


def tensor(nx: int):
    return tensor_mesh_2d(uniform_mesh_1d(length=1e-4, n_nodes=nx), uniform_mesh_1d(length=1e-5, n_nodes=ROWS))


@cache
def tensor_current(nx: int) -> float:
    return current(tensor(nx))


def jiggled_current(nx: int) -> float:
    t = tensor(nx)
    p = np.column_stack([t.node_x, t.node_y])
    inside = (t.node_x > 0) & (t.node_x < 1e-4) & (t.node_y > 0) & (t.node_y < 1e-5)
    p[inside] += np.random.default_rng(nx).uniform(-JIGGLE, JIGGLE, (int(inside.sum()), 2)) * (1e-4 / (nx - 1))
    return current(delaunay_mesh(p))


def test_a_jiggled_delaunay_mesh_converges_to_the_tensor_richardson_value() -> None:
    a, b, c = (tensor_current(n) for n in (101, 201, 401))
    p = math.log2((a - b) / (b - c))
    assert 0.7 < p < 1.2, f"the tensor order moved to {p:.2f}, so the reference this test leans on has changed"
    ref = c + (c - b) / (2**p - 1)
    errors = {n: abs(jiggled_current(n) / ref - 1) for n in (51, 101, 201)}
    for n, e in errors.items():
        assert e <= 1.5 * abs(tensor_current(n) / ref - 1), (n, e)
    assert errors[51] / errors[201] > 2.5, errors
