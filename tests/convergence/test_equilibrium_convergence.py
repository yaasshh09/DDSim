"""Mesh refinement and Newton convergence studies.

Two of the Phase 1 acceptance criteria live here:

- error against the analytic V_bi decreases with h
- Newton converges in under 10 iterations from the charge neutral guess, and
  quadratically over the last few

The refinement study is the one that would catch a discretization that is
consistent but not convergent, which no single mesh can.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ddsim.core import constants as C
from ddsim.device.equilibrium import solve_equilibrium, solve_poisson
from ddsim.device.pn_diode import pn_diode

MICRON = 1e-4
"""One micron [cm]."""


def diode_on(n_nodes: int, Na: float = 1e16, Nd: float = 1e16):
    """A uniformly refined diode, so that h halves as n_nodes doubles."""
    return pn_diode(
        Na=Na,
        Nd=Nd,
        length=4.0 * MICRON,
        junction=2.0 * MICRON,
        n_nodes=n_nodes,
        h_min=4.0 * MICRON / (n_nodes - 1),
    )


def peak_field(device, state) -> float:
    """Largest field magnitude in the device [V/cm]."""
    psi = state.psi.to_physical(device.scale).data
    return float(np.max(np.abs(-np.diff(psi) / device.mesh.h)))


# --------------------------------------------------------- mesh refinement


def test_peak_field_converges_under_mesh_refinement() -> None:
    """The field is where discretization error shows up first.

    V_bi itself is pinned by the contacts, so it cannot be used to measure
    convergence of the interior scheme. The peak field is genuinely computed
    and it is the quantity a coarse mesh gets worst, since it lives at the
    junction where the curvature is highest.
    """
    counts = [101, 201, 401, 801, 1601]
    fields = []
    for n_nodes in counts:
        device = diode_on(n_nodes)
        fields.append(peak_field(device, solve_equilibrium(device)))

    reference = fields[-1]
    errors = [abs(value - reference) / reference for value in fields[:-1]]

    assert all(
        later < earlier for earlier, later in zip(errors[:-1], errors[1:], strict=True)
    ), f"errors must shrink monotonically, got {errors}"


def test_peak_field_converges_at_second_order() -> None:
    """Box integration of the Laplacian is second order on a uniform mesh.

    Halving h should quarter the error. Anything close to first order would
    mean the dual cell volumes or the face fluxes are subtly wrong.
    """
    counts = [201, 401, 801, 1601]
    fields = []
    for n_nodes in counts:
        device = diode_on(n_nodes)
        fields.append(peak_field(device, solve_equilibrium(device)))

    reference = fields[-1]
    errors = [abs(value - reference) / reference for value in fields[:-1]]

    orders = [
        math.log2(earlier / later)
        for earlier, later in zip(errors[:-1], errors[1:], strict=True)
    ]
    assert all(order > 1.5 for order in orders), f"observed orders {orders}"


def test_built_in_potential_is_mesh_independent() -> None:
    """It is set by the contacts, so refinement must not move it at all."""
    potentials = []
    for n_nodes in (51, 201, 801):
        device = diode_on(n_nodes)
        psi = solve_equilibrium(device).psi.to_physical(device.scale).data
        potentials.append(psi[-1] - psi[0])

    expected = C.V_T() * math.log(1e16 * 1e16 / C.n_i() ** 2)
    for value in potentials:
        assert value == pytest.approx(expected, rel=1e-9)


def test_refinement_does_not_change_the_invariants() -> None:
    for n_nodes in (51, 201, 801):
        device = diode_on(n_nodes)
        state = solve_equilibrium(device)
        np.testing.assert_allclose(state.n.data * state.p.data, 1.0, rtol=1e-8)
        assert np.all(state.n.data > 0.0)


# ------------------------------------------------------- Newton convergence


def test_newton_converges_in_under_ten_iterations_across_doping() -> None:
    """The acceptance criterion, over the full doping range Phase 5 will need."""
    for doping in (1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20):
        device = pn_diode(
            Na=doping, Nd=doping, length=4.0 * MICRON, junction=2.0 * MICRON
        )
        state = solve_equilibrium(device)
        assert state.newton.iterations < 10, (
            f"{doping:.0e} took {state.newton.iterations}: "
            f"{state.newton.residual_history}"
        )


def test_newton_residual_tail_is_quadratic() -> None:
    """Each step squares the residual once inside the basin of attraction.

    A typical history for this device is

        1.13e6, 7.16e5, 3.06e5, 46.3, 4.18, 2.93e-2, 1.27e-6, 2.09e-11

    The first three steps are step limited and only linear. The tail is the
    last three above the roundoff floor, where the residual falls by five
    orders of magnitude per step.

    Quadratic is checked without having to guess the constant: if
    r_{k+1} = C * r_k^2 then C is the same at every step, so the test asserts
    that the measured ratio stays put rather than that it hits some value.
    Linear convergence would make the ratio grow by orders of magnitude.
    """
    device = pn_diode(Na=1e16, Nd=1e16, length=4.0 * MICRON, junction=2.0 * MICRON)
    state = solve_equilibrium(device)

    history = np.array(state.newton.residual_history)
    relative = history / history[0]

    # Above the roundoff floor, which the final residual sits on.
    usable = relative[relative > 100.0 * relative[-1]]
    tail = usable[-3:]
    assert len(tail) == 3, f"no usable tail in {history}"

    for previous, current in zip(tail[:-1], tail[1:], strict=True):
        assert current < previous / 100.0, "a quadratic tail step gains many digits"

    ratios = [
        current / previous**2
        for previous, current in zip(tail[:-1], tail[1:], strict=True)
    ]
    assert max(ratios) / min(ratios) < 10.0, f"C is not constant: {ratios}"


def test_newton_ends_with_unlimited_steps() -> None:
    """A converged solve must finish with real Newton steps, not clamped ones.

    If the step limiter were still active at the end, the tail would be linear
    and the quadratic claim would be false.
    """
    device = pn_diode(Na=1e16, Nd=1e16, length=4.0 * MICRON, junction=2.0 * MICRON)
    state = solve_equilibrium(device)

    assert state.newton.limited_steps < state.newton.iterations


def test_residual_falls_by_many_orders_of_magnitude() -> None:
    device = pn_diode(Na=1e16, Nd=1e16, length=4.0 * MICRON, junction=2.0 * MICRON)
    state = solve_equilibrium(device)

    first = state.newton.residual_history[0]
    last = state.newton.residual_history[-1]
    assert last / first < 1e-14


def test_the_charge_neutral_guess_is_a_good_starting_point() -> None:
    """psi = asinh(N/2) should already be right everywhere except the junction.

    If the initial residual were large across the whole device rather than
    concentrated near the junction, the guess would not be doing its job.
    """
    device = pn_diode(Na=1e16, Nd=1e16, length=4.0 * MICRON, junction=2.0 * MICRON)
    state = solve_equilibrium(device)

    psi = state.psi.data
    from ddsim.physics.statistics import psi_equilibrium_scaled

    guess = np.asarray(psi_equilibrium_scaled(device.net_doping_scaled.data))
    difference = np.abs(psi - guess)

    junction = 2.0 * MICRON
    L_D = device.scale.x_0
    far = np.abs(device.mesh.x - junction) > 40.0 * math.sqrt(
        C.eps_Si() * C.V_T() / (C.q * 1e16)
    )
    assert difference[far].max() < 1e-6, f"worst {difference[far].max():.3e}"
    assert difference.max() > 1.0, "the junction must actually need solving"
    assert L_D > 0.0


# ------------------------------------------- the residual threshold has a floor


def test_newton_converges_on_lightly_doped_material() -> None:
    """The doping range above stops at 1e14, and below it the solve used to fail.

    The residual threshold is built from the doping charge in the largest dual
    cell, because that is the size of the terms the residual is made of and it
    does not depend on where the iteration started. But the residual has a
    second half, the difference of the two face fluxes, and that difference
    cannot be resolved below machine epsilon times the size of the fluxes
    themselves. The two scale in opposite directions: the charge falls with
    the doping while psi is only logarithmic in it, so the flux floor stays
    put.

    Below about 1e13 the charge threshold sinks under the flux floor, and then
    nothing can meet it. Measured on the 1e12 bar before the fix: the residual
    reached 6.8e-12 at the third iteration and sat there, unchanged to the last
    bit, for the remaining forty-seven, with an update of 4.4e-16 the whole
    time. Newton had solved it at step three and then reported failure.

    High resistivity substrates run at exactly these dopings, so this is a
    range the project needs rather than a curiosity.
    """
    for doping in (1e13, 1e12, 1e11, 1e10):
        device = pn_diode(
            Na=doping, Nd=doping, length=4.0 * MICRON, junction=2.0 * MICRON
        )
        state = solve_equilibrium(device)

        assert state.newton is not None
        assert state.newton.converged, (
            f"{doping:.0e} did not converge: {state.newton.message}"
        )
        assert state.newton.iterations < 10, (
            f"{doping:.0e} took {state.newton.iterations} iterations, which "
            "means the threshold is sitting on the floor rather than above it"
        )


def test_the_threshold_floor_does_not_loosen_a_normally_doped_solve() -> None:
    """The floor is a floor, not an addition.

    Raising the threshold to clear the flux floor must not touch any device
    where the doping charge already clears it, otherwise every diode in the
    project silently converges to a looser answer than it used to.
    """
    for doping in (1e15, 1e16, 1e18):
        device = pn_diode(Na=doping, Nd=doping)
        state = solve_equilibrium(device)

        assert state.newton is not None
        # The charge threshold is rtol times the doping charge in the largest
        # cell, and the solve has to beat it rather than some raised version.
        charge = float(
            np.max(
                np.abs(device.net_doping_scaled.data)
                * device.mesh.volume
                / device.scale.x_0
            )
        )
        assert state.newton.residual_history[-1] < 1e-12 + 1e-10 * charge


def test_a_stalled_solve_says_what_it_was_aiming_for() -> None:
    """The message has to carry the threshold, not just the residual.

    A residual that has stopped moving while the update is already tiny means
    the threshold is under the arithmetic floor. Without the number to compare
    against, that reads exactly like a solve that is merely slow, which is a
    different problem with a different fix.
    """
    device = pn_diode(Na=1e16, Nd=1e16)

    # A flat start is many volts from the answer and the step limiter allows
    # 5 * V_T, so one iteration cannot possibly land.
    stalled = solve_poisson(
        device, np.zeros(device.mesh.n_nodes), max_iterations=1
    )

    assert not stalled.converged
    assert "threshold" in stalled.message
    assert f"{stalled.residual_history[-1]:.3e}" in stalled.message
