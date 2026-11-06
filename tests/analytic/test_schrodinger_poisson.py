from __future__ import annotations
import math
from functools import cache
import numpy as np
import pytest
from ddsim.core import constants as C
from ddsim.device.mos_cap import mos_cap

pytest.importorskip("atomsim")

from ddsim.device.schrodinger_poisson import column, solve_schrodinger_poisson, subband_density, valley_Nc  # noqa: E402


def capacitor(gate: float, t_ox: float = 2e-7):
    return mos_cap(gate_voltage=gate, t_ox=t_ox, substrate_doping=-1e17, h_min=1e-8, n_silicon=161)


@cache
def sheet(gate: float, t_ox: float, quantum: bool) -> float:
    return solve_schrodinger_poisson(capacitor(gate, t_ox), quantum=quantum).sheet_charge


def inversion_capacitance(t_ox: float, quantum: bool) -> float:
    return -(sheet(1.52, t_ox, quantum) - sheet(1.48, t_ox, quantum)) / 0.04


def test_the_classical_column_is_ddsims_own_equilibrium() -> None:
    d = capacitor(1.5)
    np.testing.assert_allclose(solve_schrodinger_poisson(d, quantum=False).psi, column(d).psi_classical, rtol=0, atol=1e-12)


def test_a_wide_flat_well_holds_the_boltzmann_density() -> None:
    T = 300.0
    depth = np.linspace(0.0, 6e-6, 1201)
    n, _ = subband_density(depth, np.full(depth.size, 0.2), T, 80, 6000)
    want = valley_Nc(T) * math.exp(-0.2 / (C.k_B * T / C.q))
    middle = (depth > 0.3 * depth[-1]) & (depth < 0.7 * depth[-1])
    np.testing.assert_allclose(n[middle], want, rtol=1e-3)


def test_confinement_pushes_the_inversion_charge_off_the_interface() -> None:
    d = capacitor(1.5)
    q, c = solve_schrodinger_poisson(d), solve_schrodinger_poisson(d, quantum=False)
    assert 0.5e-7 < q.centroid < 2e-7, f"quantum centroid {q.centroid * 1e7:.3f} nm"
    assert q.centroid > 2 * c.centroid, f"quantum {q.centroid * 1e7:.3f} nm against classical {c.centroid * 1e7:.3f} nm"
    assert q.levels[0][0] < q.levels[1][0], "the two valleys that confine with the heavy mass should sit lowest"


def test_thin_oxides_lose_inversion_capacitance_to_confinement() -> None:
    thin = 1 - inversion_capacitance(2e-7, True) / inversion_capacitance(2e-7, False)
    thick = 1 - inversion_capacitance(5e-7, True) / inversion_capacitance(5e-7, False)
    assert thin > 0.05, f"on 2 nm the quantum inversion capacitance should sit clearly below the classical one, got {thin:.3%}"
    assert 0.0 < thick < thin, f"the gap should shrink as the oxide thickens: {thin:.3%} at 2 nm, {thick:.3%} at 5 nm"


def test_the_valley_masses_imply_the_tabulated_conduction_band_density() -> None:
    assert valley_Nc(300.0) == pytest.approx(C.Nc(300.0), rel=0.05), "six valleys of m_l and m_t should give Nc near the table's 2.86e19; the 4.3 percent left is the table's density of states mass, 1.09 against 1.06"
