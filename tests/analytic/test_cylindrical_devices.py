from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import pytest
from ddsim.device.builder import build_device
from ddsim.device.doping import Coordinates, DopingProfile, Position, abrupt_junction
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import TransportModels
from ddsim.discretize.boundary import OhmicPlate
from ddsim.extract.iv import gate_sweep, iv_sweep
from ddsim.mesh.mesh1d import graded_mesh_1d
from ddsim.mesh.trimesh import triangle_mesh

RADIUS = 2e-5


def grid(x, y):
    X, Y = np.meshgrid(x, y, indexing="xy")
    nx = x.size
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(y.size - 1), indexing="xy")
    a = (j * nx + i).ravel()
    t = np.concatenate([np.column_stack([a, a + 1, a + nx + 1]), np.column_stack([a, a + nx + 1, a + nx])])
    return np.column_stack([X.ravel(), Y.ravel()]), t


@dataclass(frozen=True)
class AlongY(DopingProfile):
    inner: DopingProfile

    def __call__(self, at: Position) -> np.ndarray:
        c = Coordinates.of(at)
        assert c.y is not None
        return self.inner(Coordinates(c.y))


def test_a_junction_across_a_cylinder_carries_the_planar_current_times_its_area() -> None:
    z = graded_mesh_1d(length=1e-4, n_nodes=101, refine_at=0.5e-4, h_min=2e-7)
    nr = 5
    p, t = grid(np.linspace(0.0, RADIUS, nr), z.x)
    m = triangle_mesh(p, t, cylindrical=True)
    d = build_device(mesh=m, doping=AlongY(abrupt_junction(Na=1e16, Nd=1e16, position=0.5e-4)), contacts=(OhmicPlate(name="anode", nodes=tuple(range(nr)), voltage=0.0), OhmicPlate(name="cathode", nodes=tuple(range(m.n_nodes - nr, m.n_nodes)), voltage=0.0)))
    assert d.dimension == 3
    c = gate_sweep(d, [0.0, 0.3, 0.5], contact="anode", measure_at="anode", models=TransportModels.for_device(d))
    flat = pn_diode(n_nodes=101, h_min=2e-7)
    want = iv_sweep(flat, "anode", [0.0, 0.3, 0.5], models=TransportModels.for_device(flat))
    assert c.complete and want.complete
    for a, b in zip(c.points[1:], want.points[1:], strict=True):
        assert a.current == pytest.approx(b.current * math.pi * RADIUS**2, rel=1e-6)


def cylindrical_breakdown(r_j: float) -> float:
    from ddsim.device.doping import Step
    from ddsim.discretize.impact import ImpactIonization
    from tests.analytic.test_avalanche import crossing
    r = graded_mesh_1d(length=1.5e-4, n_nodes=401, refine_at=r_j, h_min=1e-7).x
    p, t = grid(r, np.array([0.0, 2e-6]))
    m = triangle_mesh(p, t, cylindrical=True)
    n = r.size
    d = build_device(mesh=m, doping=Step(left=-1e19, right=1e17, position=r_j), contacts=(OhmicPlate(name="anode", nodes=(0, n), voltage=0.0), OhmicPlate(name="cathode", nodes=(n - 1, 2 * n - 1), voltage=0.0)))
    c = gate_sweep(d, [0.0] + [-v for v in np.arange(1.0, 15.01, 0.5)], contact="anode", measure_at="anode", models=TransportModels.for_device(d))
    assert c.complete, c.message
    model = ImpactIonization.for_silicon(d.scale)
    h = np.diff(r) / d.scale.x_0
    left = []
    for q in c.points:
        E = np.abs(np.diff(q.state.psi.data[:n])) / h
        an = model.alpha_n(E)[0] * h
        ap = model.alpha_p(E)[0] * h
        left.append(1.0 - float(np.sum(an * np.exp(-(np.cumsum(an - ap) - 0.5 * (an - ap))))))
    return crossing([-q.voltage for q in c.points], left)


def test_a_curved_junction_breaks_down_before_a_planar_one() -> None:
    from tests.analytic.test_avalanche import integral_breakdown
    planar = integral_breakdown(1e17, 1.5e-4, 20.0)
    wide, tight = cylindrical_breakdown(0.3e-4), cylindrical_breakdown(0.1e-4)
    assert tight < wide < planar, (tight, wide, planar)
