from __future__ import annotations
from dataclasses import replace
import numpy as np
import pytest
from ddsim.device.builder import build_device
from ddsim.device.doping import Step, abrupt_junction
from ddsim.device.pn_diode import hetero_bands
from ddsim.device.transport import TransportModels, solve_bias_newton
from ddsim.discretize.boundary import OhmicContact
from ddsim.extract.iv import gate_sweep
from ddsim.mesh.mesh1d import graded_mesh_1d
from ddsim.physics.bands import Semiconductor

SILICON = Semiconductor.silicon()
L, X = 4e-4, 2e-4


def mesh():
    return graded_mesh_1d(length=L, n_nodes=401, refine_at=X, h_min=2e-7)


def device(doping, bands=None):
    m = mesh()
    contacts = (OhmicContact(name="anode", node=0, voltage=0.0), OhmicContact(name="cathode", node=m.n_nodes - 1, voltage=0.0))
    return build_device(mesh=m, doping=doping, contacts=contacts, bands=bands)


def test_a_junction_of_silicon_with_itself_is_the_silicon_diode() -> None:
    doping = abrupt_junction(Na=1e16, Nd=1e16, position=X)
    a = device(doping)
    b = device(doping, hetero_bands(mesh(), X, SILICON, SILICON))
    V = [0.0, -0.5, 0.3, 0.6]
    x = gate_sweep(a, V, contact="anode", measure_at="anode", models=TransportModels.for_device(a))
    y = gate_sweep(b, V, contact="anode", measure_at="anode", models=TransportModels.for_device(b))
    assert x.complete and y.complete
    for p, q in zip(x.points, y.points, strict=True):
        assert q.current == pytest.approx(p.current, rel=1e-10, abs=0)


def test_an_isotype_junction_holds_a_flat_fermi_level_and_piles_electrons_into_the_well() -> None:
    deep = replace(SILICON, name="deep", chi=SILICON.chi + 0.2)
    d = device(Step(left=1e16, right=1e16, position=X), hetero_bands(mesh(), X, SILICON, deep))
    s = solve_bias_newton(d, models=TransportModels.for_device(d))
    assert s.newton is not None and s.newton.converged
    assert d.bands is not None
    phi_n = s.psi.data + d.bands.d_n - np.log(s.n.data)
    assert np.max(np.abs(phi_n)) < 1e-8
    phi_p = s.psi.data - d.bands.d_p + np.log(s.p.data)
    assert np.max(np.abs(phi_p)) < 1e-8
    x = d.mesh.x
    peak = int(np.argmax(s.n.data))
    assert x[peak] >= X and s.n.data[peak] > 2 * d.net_doping_scaled.data[peak]
    assert s.n.data[np.searchsorted(x, X) - 1] < 0.5 * d.net_doping_scaled.data[0]
    assert s.n.data[0] == pytest.approx(d.net_doping_scaled.data[0], rel=1e-3)
    assert s.n.data[-1] == pytest.approx(d.net_doping_scaled.data[-1], rel=1e-3)


@pytest.mark.parametrize("Na,Nd", [(1e16, 3e16), (1e16, 1e16), (1e15, 3e15)])
def test_an_anisotype_junction_splits_its_built_in_potential_by_eps_times_doping(Na: float, Nd: float) -> None:
    wide = replace(SILICON, name="wide", eps_r=1.2 * SILICON.eps_r, chi=SILICON.chi - 0.1, Eg=SILICON.Eg + 0.3)
    d = device(abrupt_junction(Na=Na, Nd=Nd, position=X), hetero_bands(mesh(), X, SILICON, wide))
    s = solve_bias_newton(d, models=TransportModels.for_device(d))
    assert s.newton is not None and s.newton.converged
    psi = s.psi.to_physical(d.scale).data
    at = int(np.searchsorted(d.mesh.x, X))
    V_T = d.scale.psi_0
    left, right = psi[at] - psi[0] - V_T, psi[-1] - psi[at] - V_T
    want = (wide.eps_r * Nd) / (SILICON.eps_r * Na)
    assert left / right == pytest.approx(want, rel=0.03), f"depletion drops {left:.4f} V and {right:.4f} V after the kT/q Debye tail on each side, ratio {left / right:.4f} against {want:.4f}"
