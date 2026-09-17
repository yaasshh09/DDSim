"""Band edges from a solved state, ddsim/extract/bands.py.

The band diagram the page draws. Energies in eV with the equilibrium Fermi
level at zero, so the intrinsic level sits at -psi. Every test is an analytic
limit the edges have to meet.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ddsim.core import constants as C
from ddsim.device.equilibrium import solve_equilibrium
from ddsim.device.mos_cap import mos_cap
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import solve_bias
from ddsim.extract.bands import band_edges

DOPING = 1e16


@pytest.fixture(scope="module")
def equilibrium():
    device = pn_diode(Na=DOPING, Nd=DOPING, n_nodes=201)
    return device, solve_equilibrium(device)


def test_the_gap_is_the_same_everywhere(equilibrium) -> None:
    """Ec - Ev = kT ln(Nc Nv / n_i^2), the gap the anchored n_i implies. It
    differs from Eg(300) by a few meV, recorded in docs/07-decisions.md."""
    device, state = equilibrium
    bands = band_edges(device, state)
    implied = C.V_T() * math.log(C.Nc() * C.Nv() / C.n_i() ** 2)

    np.testing.assert_allclose(bands.Ec - bands.Ev, implied, rtol=1e-12)
    assert abs(implied - C.Eg()) < 5e-3


def test_in_equilibrium_both_quasi_fermi_levels_are_flat_at_zero(equilibrium) -> None:
    device, state = equilibrium
    bands = band_edges(device, state)

    np.testing.assert_allclose(bands.Efn, 0.0, atol=1e-9)
    np.testing.assert_allclose(bands.Efp, 0.0, atol=1e-9)


def test_at_the_p_contact_the_valence_band_sits_where_boltzmann_puts_it(
    equilibrium,
) -> None:
    """Ef - Ev = kT ln(Nv / Na), non degenerate, at the ohmic anode."""
    device, state = equilibrium
    bands = band_edges(device, state)

    expected = C.V_T() * math.log(C.Nv() / DOPING)
    assert 0.0 - bands.Ev[0] == pytest.approx(expected, rel=1e-3)


def test_under_forward_bias_the_quasi_fermi_levels_split_by_the_bias() -> None:
    device = pn_diode(Na=DOPING, Nd=DOPING, n_nodes=201, anode_voltage=0.4)
    bands = band_edges(device, solve_bias(device))

    middle = len(bands.Efn) // 2
    assert bands.Efn[middle] - bands.Efp[middle] == pytest.approx(0.4, abs=0.02)


def test_the_oxide_has_no_silicon_band_edges() -> None:
    device = mos_cap(gate_voltage=0.0)
    bands = band_edges(device, solve_equilibrium(device))
    oxide = device.regions.oxide_nodes

    assert np.all(np.isnan(bands.Ec[oxide]))
    assert np.all(np.isfinite(np.delete(bands.Ec, oxide)))
