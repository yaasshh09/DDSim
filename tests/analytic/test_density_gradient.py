from __future__ import annotations
import importlib.util
from functools import cache
import numpy as np
import pytest
from ddsim.core import constants as C
from ddsim.device.density_gradient import DensityGradient, calibrate_gamma, residual, solve_density_gradient
from ddsim.device.mos_cap import mos_cap
from ddsim.device.schrodinger_poisson import column, solve_schrodinger_poisson


needs_atomsim = pytest.mark.skipif(importlib.util.find_spec("atomsim") is None, reason="Schrodinger-Poisson needs AtomSIM's eigen solver")


def capacitor(gate: float):
    return mos_cap(gate_voltage=gate, t_ox=2e-7, substrate_doping=-1e17, h_min=1e-8, n_silicon=161)


def test_the_jacobian_is_the_derivative_of_the_residual() -> None:
    d = capacitor(1.5)
    c = column(d)
    V_T = C.k_B * 300.0 / C.q
    top = int(np.max(np.flatnonzero(c.silicon > 0.0)))
    rng = np.random.default_rng(0)
    x = np.concatenate([c.psi_classical, 0.05 * rng.standard_normal(top + 1)])
    _, J = residual(c, x, top, DensityGradient(), d.material.n_i, V_T)
    for j in rng.choice(x.size, 60, replace=False):
        e = np.zeros(x.size)
        e[j] = 1e-7
        fd = (residual(c, x + e, top, DensityGradient(), d.material.n_i, V_T)[0] - residual(c, x - e, top, DensityGradient(), d.material.n_i, V_T)[0]) / 2e-7
        col = J[:, j].toarray().ravel()
        assert np.max(np.abs(fd - col)) <= 1e-5 * (np.abs(col).max() + np.abs(fd).max())


def test_no_gamma_is_the_classical_capacitor() -> None:
    d = capacitor(1.5)
    q = solve_density_gradient(d, DensityGradient(gamma=0.0, gamma_ox=0.0))
    np.testing.assert_allclose(q.psi, column(d).psi_classical, rtol=0, atol=1e-10)


@cache
def calibrated() -> float:
    return calibrate_gamma([capacitor(1.5)], [solve_schrodinger_poisson(capacitor(1.5)).sheet_charge])


@needs_atomsim
@pytest.mark.parametrize("gate", [1.0, 2.0])
def test_a_gamma_calibrated_at_one_bias_tracks_schrodinger_poisson_at_others(gate: float) -> None:
    g = calibrated()
    assert 3.0 < g < 20.0, f"calibrated gamma {g:.3f}"
    d = capacitor(gate)
    want = solve_schrodinger_poisson(d).sheet_charge
    classical = solve_schrodinger_poisson(d, quantum=False).sheet_charge
    got = solve_density_gradient(d, DensityGradient(gamma=g)).sheet_charge
    assert abs(got - want) < 0.15 * abs(classical - want), f"density gradient {got:.4e}, Schrodinger-Poisson {want:.4e}, classical {classical:.4e} C/cm^2"


@needs_atomsim
def test_density_gradient_moves_the_charge_off_the_interface() -> None:
    d = capacitor(1.5)
    q = solve_density_gradient(d, DensityGradient(gamma=calibrated()))
    c = solve_schrodinger_poisson(d, quantum=False)
    assert q.centroid > 2 * c.centroid, f"density gradient {q.centroid * 1e7:.3f} nm, classical {c.centroid * 1e7:.3f} nm"
