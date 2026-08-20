"""Tier 3 invariants, from docs/04-validation.md.

Cheap, and they catch the most bugs per line of code. These hold for every
solve regardless of doping, geometry or bias, so they are the checks that would
catch a sign error that happened to leave the built-in potential looking right.

Current continuity and the terminal current sum are the other two Tier 3
invariants. Both need currents, so they arrive with the continuity equations in
Phase 2.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ddsim.core import constants as C
from ddsim.device.builder import build_device
from ddsim.device.doping import Gaussian, Step, Uniform
from ddsim.device.equilibrium import frozen_quasi_fermi, solve_equilibrium
from ddsim.device.pn_diode import pn_diode
from ddsim.discretize.boundary import OhmicContact
from ddsim.mesh.mesh1d import uniform_mesh_1d

MICRON = 1e-4
"""One micron [cm]."""

DEVICES = {
    "symmetric-1e16": lambda: pn_diode(Na=1e16, Nd=1e16, length=12e-4, junction=6e-4),
    "asymmetric-1e15-1e18": lambda: pn_diode(
        Na=1e15, Nd=1e18, length=12e-4, junction=6e-4
    ),
    "heavy-1e19": lambda: pn_diode(Na=1e19, Nd=1e19, length=12e-4, junction=6e-4),
    "light-1e14": lambda: pn_diode(Na=1e14, Nd=1e14, length=40e-4, junction=20e-4),
}


@pytest.fixture(params=sorted(DEVICES), ids=sorted(DEVICES))
def solved(request):
    """A converged device across a wide span of doping."""
    device = DEVICES[request.param]()
    return device, solve_equilibrium(device)


# ------------------------------------------------------------- mass action


def test_np_equals_n_i_squared_everywhere(solved) -> None:
    """docs/04-validation.md: under 1e-8 relative, everywhere, all doping.

    In scaled units with C_0 = n_i this is just n*p = 1.
    """
    _, state = solved
    product = state.n.data * state.p.data
    np.testing.assert_allclose(product, 1.0, rtol=1e-8)


def test_np_equals_n_i_squared_in_physical_units(solved) -> None:
    """The same statement after converting out of scaled units."""
    device, state = solved
    n = state.n.to_physical(device.scale).data  # [cm^-3]
    p = state.p.to_physical(device.scale).data  # [cm^-3]
    np.testing.assert_allclose(n * p, device.material.n_i**2, rtol=1e-8)


# -------------------------------------------------------------- positivity


def test_carrier_densities_are_strictly_positive(solved) -> None:
    """docs/04-validation.md: a negative density means a broken M-matrix.

    In 1D that means a sign error. Do not paper over it by clamping.
    """
    _, state = solved
    assert np.all(state.n.data > 0.0)
    assert np.all(state.p.data > 0.0)


def test_carrier_densities_are_finite(solved) -> None:
    _, state = solved
    assert np.all(np.isfinite(state.n.data))
    assert np.all(np.isfinite(state.p.data))


# ------------------------------------------------------- charge neutrality


def test_bulk_is_charge_neutral(solved) -> None:
    """docs/04-validation.md: |p - n + N| / N under 1e-6, far from a junction.

    Far means far. The deviation from neutrality decays exponentially over the
    local Debye length, so reaching 1e-6 takes about 14 Debye lengths beyond
    the depletion edge. A 1 um diode at 1e16 has a 0.43 um depletion region and
    no room left to be neutral in, which is why the devices here are 12 um.
    """
    device, state = solved
    doping = device.net_doping_scaled.data
    net_charge = state.p.data - state.n.data + doping

    # The binding constraint is the lightly doped side, which has the longest
    # Debye length and therefore the slowest approach to neutrality. Measured
    # across four decades of doping, the deviation falls off cleanly in units
    # of that length: 15 L_D gives 1e-5, 20 gives 1e-6, 25 gives 1e-8 or
    # better. 25 is used here so the margin does not depend on the device.
    physical = np.abs(device.net_doping.data)
    lightest = float(np.min(physical[physical > 0.0]))  # [cm^-3]
    local_debye = math.sqrt(C.eps_Si() * C.V_T() / (C.q * lightest))  # [cm]

    junction = device.mesh.x[int(np.argmax(np.abs(np.diff(np.sign(doping)))))]
    far = np.abs(device.mesh.x - junction) > 25.0 * local_debye

    assert far.sum() > 10, "device is too short to have a neutral bulk"
    relative = np.abs(net_charge[far]) / np.abs(doping[far])
    assert relative.max() < 1e-6, (
        f"worst {relative.max():.3e}, local L_D = {local_debye * 1e7:.1f} nm"
    )


def test_total_charge_in_the_device_is_conserved(solved) -> None:
    """A junction separates charge, it does not create any.

    The integrated net charge must vanish, because both contacts are neutral
    and nothing else adds charge.
    """
    device, state = solved
    doping = device.net_doping_scaled.data
    net_charge = state.p.data - state.n.data + doping
    volume = device.mesh.volume / device.scale.x_0

    total = float(np.sum(net_charge * volume))
    reference = float(np.sum(np.abs(net_charge) * volume))
    assert abs(total) / reference < 1e-6


# ----------------------------------------------------- consistency of state


def test_densities_agree_with_boltzmann_applied_to_psi(solved) -> None:
    """n and p must be the ones psi implies, not a stale copy."""
    _, state = solved
    np.testing.assert_allclose(state.n.data, np.exp(state.psi.data), rtol=1e-12)
    np.testing.assert_allclose(state.p.data, np.exp(-state.psi.data), rtol=1e-12)


def test_majority_carrier_matches_the_doping_in_the_bulk(solved) -> None:
    """n = Nd in the neutral n region, p = Na in the neutral p region."""
    device, state = solved
    doping = device.net_doping_scaled.data

    n_bulk = doping > 0.0
    p_bulk = doping < 0.0
    at_n_contact = int(np.flatnonzero(n_bulk)[-1])
    at_p_contact = int(np.flatnonzero(p_bulk)[0])

    assert state.n.data[at_n_contact] == pytest.approx(doping[at_n_contact], rel=1e-6)
    assert state.p.data[at_p_contact] == pytest.approx(-doping[at_p_contact], rel=1e-6)


def test_potential_is_monotonic_across_the_junction(solved) -> None:
    """psi rises from the p side to the n side and never turns back.

    A non-monotonic potential in a two region diode at equilibrium would mean
    a spurious internal field, which is what a broken M-matrix produces.
    """
    _, state = solved
    assert np.all(np.diff(state.psi.data) > -1e-12)


# --------------------------------------------------------- other geometries


def test_invariants_hold_for_a_gaussian_profile() -> None:
    """A smoothly varying profile, not just an abrupt step."""
    mesh = uniform_mesh_1d(8.0 * MICRON, 601)
    device = build_device(
        mesh=mesh,
        doping=Uniform(-1e16) + Gaussian(peak=5e17, centre=0.0, sigma=0.5 * MICRON),
        contacts=(
            OhmicContact("anode", 0, 0.0),
            OhmicContact("cathode", mesh.n_nodes - 1, 0.0),
        ),
    )
    state = solve_equilibrium(device)

    np.testing.assert_allclose(state.n.data * state.p.data, 1.0, rtol=1e-8)
    assert np.all(state.n.data > 0.0)
    assert np.all(state.p.data > 0.0)


def test_invariants_hold_for_a_compensated_profile() -> None:
    """Net doping passes exactly through zero, where the log form would die."""
    mesh = uniform_mesh_1d(8.0 * MICRON, 601)
    device = build_device(
        mesh=mesh,
        doping=Step(left=-1e16, right=1e16, position=4.0 * MICRON) + Uniform(0.0),
        contacts=(
            OhmicContact("anode", 0, 0.0),
            OhmicContact("cathode", mesh.n_nodes - 1, 0.0),
        ),
    )
    state = solve_equilibrium(device)

    np.testing.assert_allclose(state.n.data * state.p.data, 1.0, rtol=1e-8)
    assert np.all(np.isfinite(state.psi.data))


def test_invariants_hold_under_reverse_bias() -> None:
    """np = n_i^2 holds only at equilibrium. Under bias it must equal
    exp(phi_p - phi_n), which for -1 V is exp(-38.7)."""
    device = pn_diode(Na=1e16, Nd=1e16, length=12e-4, junction=6e-4, anode_voltage=-1.0)
    quasi_fermi = solve_equilibrium(device, frozen_quasi_fermi(device))
    phi_n, phi_p = frozen_quasi_fermi(device)

    expected = np.exp(phi_p.data - phi_n.data)
    product = quasi_fermi.n.data * quasi_fermi.p.data
    np.testing.assert_allclose(product, expected, rtol=1e-8)
    assert np.all(quasi_fermi.n.data > 0.0)
    assert np.all(quasi_fermi.p.data > 0.0)


def test_intrinsic_material_stays_intrinsic() -> None:
    """Zero doping means psi = 0 and n = p = n_i, exactly."""
    mesh = uniform_mesh_1d(MICRON, 51)
    device = build_device(
        mesh=mesh,
        doping=Uniform(0.0),
        contacts=(
            OhmicContact("left", 0, 0.0),
            OhmicContact("right", mesh.n_nodes - 1, 0.0),
        ),
    )
    state = solve_equilibrium(device)

    np.testing.assert_allclose(state.psi.data, 0.0, atol=1e-12)
    np.testing.assert_allclose(state.n.data, 1.0, rtol=1e-12)
    np.testing.assert_allclose(state.p.data, 1.0, rtol=1e-12)
