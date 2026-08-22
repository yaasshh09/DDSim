"""Tests for extract/iv.py, terminal current and I-V sweeps.

Sign convention, fixed here and used everywhere downstream: a terminal current
is positive when conventional current flows from the contact into the device.
A forward biased diode therefore has a positive anode current, which is the
convention every diode I-V plot in the literature uses.

Terminal currents are taken from the continuity residual at the contact node
rather than from the edge flux next to it. The two are the same number, since
the residual at a contact is exactly the current that has to be injected there
to satisfy the equation, but the residual form is what makes the terminal
currents sum to zero identically rather than approximately.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.core.field import Location, ScalingState
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import TransportModels, solve_bias
from ddsim.extract.iv import (
    IVCurve,
    continuity_residuals,
    current_densities,
    iv_sweep,
    terminal_currents,
    total_current,
)
from ddsim.physics.recombination import NoRecombination

MICRON = 1e-4
"""One micron [cm]."""


def diode(**overrides: float):
    """The Phase 2 test diode: 1e16 / 1e16, 12 um, graded to 5 nm."""
    settings: dict = {
        "Na": 1e16,
        "Nd": 1e16,
        "length": 12 * MICRON,
        "junction": 6 * MICRON,
        "n_nodes": 201,
        "h_min": 5e-7,
    }
    settings.update(overrides)
    return pn_diode(**settings)


# ------------------------------------------------------------ current density


def test_current_densities_come_back_physical_and_on_edges() -> None:
    """A current density lives on edges, not nodes, and the type says so."""
    device = diode()
    state = solve_bias(device)
    Jn, Jp = current_densities(device, state)

    for current in (Jn, Jp):
        assert current.location is Location.EDGE
        assert current.scaling is ScalingState.PHYSICAL
        assert current.unit == "A/cm^2"
        assert current.size == device.mesh.n_edges


def test_equilibrium_carries_no_current() -> None:
    """Zero to the cancellation floor, which is what zero looks like here.

    Jn is the difference of two edge terms of size (Dn/h)*n, and at equilibrium
    those cancel completely. The floor is therefore set by the size of the
    terms rather than by anything physical, and it is many decades below any
    current this device passes under bias.
    """
    device = diode()
    state = solve_bias(device)
    Jn, Jp = current_densities(device, state)

    assert np.max(np.abs(Jn.data + Jp.data)) < 1e-10
    assert abs(total_current(device, state)) < 1e-10


# ---------------------------------------------------------- terminal currents


def test_forward_bias_drives_current_into_the_anode() -> None:
    """The sign check docs/05-pitfalls.md puts first in the debugging order.

    A diode biased positive on its p side conducts, and the current enters at
    the anode. Reversing the Bernoulli asymmetry in the flux would give a
    solver that converges just as cleanly with this number negative.
    """
    device = diode().with_bias(anode=0.4)
    state = solve_bias(device)

    assert state.gummel is not None and state.gummel.converged
    currents = terminal_currents(device, state)
    assert currents["anode"] > 0.0


def test_reverse_bias_gives_a_small_negative_current() -> None:
    """Saturation, plus depletion region generation, and far below forward."""
    device = diode().with_bias(anode=-1.0)
    state = solve_bias(device)

    reverse = terminal_currents(device, state)["anode"]
    forward = terminal_currents(
        diode().with_bias(anode=0.4), solve_bias(diode().with_bias(anode=0.4))
    )["anode"]

    assert reverse < 0.0
    assert abs(reverse) < 1e-3 * forward


def test_terminal_currents_sum_to_zero() -> None:
    """Kirchhoff, and a check on every boundary condition at once.

    docs/04-validation.md asks for 1e-8 relative to the largest terminal
    current. The residual form makes it exact rather than merely close: the
    discrete divergence telescopes over the whole device, so what is left is
    the interior residual, and the solve drove that to zero.
    """
    device = diode().with_bias(anode=0.4)
    state = solve_bias(device)

    currents = terminal_currents(device, state)
    largest = max(abs(value) for value in currents.values())
    assert abs(sum(currents.values())) < 1e-8 * largest


def test_total_current_is_the_anode_current() -> None:
    device = diode().with_bias(anode=0.3)
    state = solve_bias(device)

    np.testing.assert_allclose(
        total_current(device, state),
        terminal_currents(device, state)["anode"],
        rtol=1e-12,
    )


def test_current_rises_steeply_with_forward_bias() -> None:
    """Roughly a decade per 60 to 120 mV, which is the whole point of a diode."""
    low = diode().with_bias(anode=0.2)
    high = diode().with_bias(anode=0.3)

    ratio = total_current(high, solve_bias(high)) / total_current(
        low, solve_bias(low)
    )
    assert 10.0 < ratio < 100.0


def test_recombination_drops_out_of_the_terminal_current() -> None:
    """The current at a contact does not depend on the recombination model.

    Twice over. An electron and a hole recombine as a pair and carry no net
    charge away, so the R*volume term enters the two residuals with opposite
    signs and cancels. And at an ohmic contact it is not merely cancelled but
    identically zero, because the contact pins n and p at their equilibrium
    values and np = n_i^2 is exactly where SRH vanishes.

    Measured by running one solved state through two different models. Even the
    electron residual alone, where no cancellation is available, is unchanged.
    """
    device = diode().with_bias(anode=0.3)
    models = TransportModels.for_device(device, recombination=NoRecombination())
    state = solve_bias(device)

    assert (
        terminal_currents(device, state, models=models)["anode"]
        == terminal_currents(device, state)["anode"]
    )

    node = device.contacts[0].node
    assert (
        continuity_residuals(device, state, models)[0][node]
        == continuity_residuals(device, state)[0][node]
    )

    default = TransportModels.for_device(device)
    rate = np.asarray(default.recombination.rate(state.n.data, state.p.data))
    assert rate[node] == 0.0


# ------------------------------------------------------------------- sweeps


def test_a_sweep_lands_on_every_requested_voltage() -> None:
    voltages = [0.0, 0.1, 0.2, 0.3]
    curve = iv_sweep(diode(), "anode", voltages)

    assert curve.complete
    np.testing.assert_allclose(curve.voltage, voltages, atol=0.0)


def test_a_sweep_current_increases_with_forward_bias() -> None:
    curve = iv_sweep(diode(), "anode", [0.1, 0.2, 0.3, 0.4])

    assert np.all(np.diff(curve.current) > 0.0)


def test_a_sweep_can_run_into_reverse_bias() -> None:
    curve = iv_sweep(diode(), "anode", [-0.5, -0.25, 0.0, 0.25])

    assert curve.complete
    assert curve.current[0] < 0.0
    assert curve.current[-1] > 0.0


def test_a_sweep_carries_the_state_at_every_point() -> None:
    """The states are what the band diagrams and the plots are drawn from."""
    curve = iv_sweep(diode(), "anode", [0.0, 0.2])

    assert len(curve.points) == 2
    for point, voltage in zip(curve.points, [0.0, 0.2], strict=True):
        assert point.state.gummel is not None
        assert point.state.gummel.converged
        assert point.voltage == voltage


def test_a_sweep_stops_and_says_where_when_it_stalls() -> None:
    """A stalled sweep returns everything it reached plus the reason.

    phases/PHASE-2.md asks for the bias where Gummel gives up to be documented
    rather than fought, and this is the reporting half of that. Where it
    actually gives up, and the fact that it degrades rather than failing, is
    measured in tests/analytic/test_shockley_diode.py.

    Deliberately cheap. Inducing a stall costs one failed solve per
    continuation halving, and the default min_step is ten of them; on a 201
    node mesh with a 40 cycle budget this one test was 14 s of a 17 s suite,
    which is 84 percent of it to reach an assertion about a message. A coarse
    mesh, a small budget and a floor on the step reproduce the same path, the
    same two accepted points and the same stall at 0.4 V, in under a tenth of
    the time.
    """
    curve = iv_sweep(
        diode(n_nodes=41, h_min=2e-6),
        "anode",
        [0.2, 0.4, 5.0],
        step=0.05,
        min_step=0.01,
        max_iterations=8,
    )

    assert not curve.complete
    assert curve.message
    assert len(curve.points) >= 1
    assert curve.voltage[-1] < 5.0


def test_an_unknown_contact_is_rejected_before_any_solving() -> None:
    with pytest.raises(KeyError, match="gate"):
        iv_sweep(diode(), "gate", [0.1])


def test_curve_repr_reports_the_range() -> None:
    curve = iv_sweep(diode(), "anode", [0.0, 0.2])

    assert "anode" in repr(curve)
    assert "complete" in repr(curve)


def test_an_empty_curve_still_has_a_repr() -> None:
    """A sweep that stalls before its first requested point returns no points."""
    curve = IVCurve(contact="anode", points=(), complete=False, message="stalled")

    assert "empty" in repr(curve)


def test_a_sweep_with_no_starting_guess_raises() -> None:
    """Every point is continued from the starting bias, so that one must solve.

    Starting cold at 5 V is far outside any basin of attraction. The fallback
    guess is the Phase 1 equilibrium solve, which raises rather than returning
    something unconverged, and the sweep says what that means for the sweep.
    """
    with pytest.raises(RuntimeError, match="could not be started"):
        iv_sweep(diode(), "anode", [5.1], start=5.0, max_iterations=5)


def test_a_sweep_whose_first_point_stalls_raises() -> None:
    """The other way the same thing happens: a guess exists but will not solve.

    Forced here with a tolerance no solve can meet, which is the cleanest way
    to reach the branch without inventing a device that cannot be solved.
    """
    with pytest.raises(RuntimeError, match="did not converge"):
        iv_sweep(
            diode(), "anode", [0.1], max_iterations=1, update_tol=1e-30
        )


def test_a_floor_on_the_continuation_step_bounds_the_cost_of_a_stall() -> None:
    """min_step decides how hard a failing sweep tries before giving up.

    Every halving below the floor is one more full failed solve at the Gummel
    budget, and the default floor is a thousandth of the first step, which is
    ten of them. A caller who only wants to know roughly where a sweep dies
    should not have to pay for ten refinements of the answer.

    The coarse sweep has to stop no later than the fine one, and both have to
    report the same kind of failure.
    """
    coarse = iv_sweep(
        diode(n_nodes=41, h_min=2e-6),
        "anode",
        [0.2, 5.0],
        step=0.05,
        min_step=0.025,
        max_iterations=8,
    )
    fine = iv_sweep(
        diode(n_nodes=41, h_min=2e-6),
        "anode",
        [0.2, 5.0],
        step=0.05,
        min_step=0.0005,
        max_iterations=8,
    )

    assert not coarse.complete and not fine.complete
    assert coarse.message and fine.message
    assert coarse.voltage[-1] <= fine.voltage[-1]
