"""The coupled Newton driver, at the level of one bias point.

The convergence claims phases/PHASE-3.md is actually graded on live in
tests/convergence/test_newton_convergence.py. This file covers the wiring:
that the contacts arrive imposed rather than approached, that the limiter
does what docs/02-numerics.md prescribes, that a failure is reported rather
than swallowed, and that the state handed back is a state and not a vector.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import (
    TransportModels,
    initial_state,
    solve_bias_hybrid,
    solve_bias_newton,
)
from ddsim.discretize.boundary import Carrier, ohmic_density_scaled, ohmic_psi_scaled
from ddsim.discretize.coupled import (
    UNKNOWNS_PER_NODE,
    Unknown,
    limit_psi_step,
    pack,
)


@pytest.fixture
def diode():
    """The Phase 2 device, at a small forward bias the Poisson guess reaches."""
    return pn_diode(n_nodes=81, anode_voltage=0.2)


# ------------------------------------------------------------- the limiter


def test_the_limiter_caps_the_potential_update():
    """5*V_T per step, which is 5.0 scaled. docs/02-numerics.md."""
    delta = pack(
        np.array([40.0, -80.0, 1.0]), np.zeros(3), np.zeros(3)
    )

    limited = limit_psi_step(delta, 5.0)

    dpsi = limited[Unknown.PSI :: UNKNOWNS_PER_NODE]
    assert np.max(np.abs(dpsi)) == pytest.approx(5.0)


def test_the_limiter_takes_the_density_updates_in_full():
    """Damping n and p slows convergence without buying robustness.

    docs/05-pitfalls.md is explicit about this. The density update is six
    decades larger than the potential update in scaled units, so a single
    factor over the whole vector would be set by the densities and would
    leave psi effectively frozen.
    """
    dn = np.array([1e6, -2e6, 3e6])
    delta = pack(np.array([40.0, -80.0, 1.0]), dn, -dn)

    limited = limit_psi_step(delta, 5.0)

    np.testing.assert_array_equal(limited[Unknown.N :: UNKNOWNS_PER_NODE], dn)
    np.testing.assert_array_equal(limited[Unknown.P :: UNKNOWNS_PER_NODE], -dn)


def test_the_limiter_preserves_the_direction_within_psi():
    """Scaled by one factor, not clipped entry by entry."""
    psi_update = np.array([40.0, -80.0, 1.0])
    delta = pack(psi_update, np.zeros(3), np.zeros(3))

    limited = limit_psi_step(delta, 5.0)[Unknown.PSI :: UNKNOWNS_PER_NODE]

    np.testing.assert_allclose(
        limited / psi_update, np.full(3, 5.0 / 80.0), rtol=1e-15
    )


def test_the_limiter_returns_its_argument_when_nothing_needs_capping():
    """Identity, so that newton_solve does not count an inactive limiter."""
    delta = pack(np.array([1.0, -2.0]), np.array([9.0, 9.0]), np.array([9.0, 9.0]))

    assert limit_psi_step(delta, 5.0) is delta


# --------------------------------------------------------------- the driver


def test_returns_a_converged_device_state(diode):
    """The happy path, at a bias the Poisson guess already reaches."""
    state = solve_bias_newton(diode)

    assert state.newton is not None
    assert state.newton.converged, state.newton.message
    assert state.psi.size == diode.mesh.n_nodes


def test_the_contact_values_are_imposed_exactly(diode):
    """A Dirichlet condition is a statement about the answer, not a target.

    Checked to the last bit rather than to a tolerance, because
    apply_dirichlet_nodes eliminates the column as well as the row and that
    is the whole reason it does.
    """
    state = solve_bias_newton(diode)
    doping = diode.net_doping_scaled.data

    for contact in diode.contacts:
        node = contact.node
        assert state.psi.data[node] == pytest.approx(
            ohmic_psi_scaled(
                float(doping[node]), contact.voltage / diode.scale.psi_0
            ),
            rel=1e-14,
        )
        assert state.n.data[node] == pytest.approx(
            ohmic_density_scaled(float(doping[node]), Carrier.ELECTRON), rel=1e-14
        )
        assert state.p.data[node] == pytest.approx(
            ohmic_density_scaled(float(doping[node]), Carrier.HOLE), rel=1e-14
        )


def test_no_density_is_negative(diode):
    """phases/PHASE-3.md, and docs/05-pitfalls.md forbids clamping to get it."""
    state = solve_bias_newton(diode)

    assert np.all(state.n.data > 0.0)
    assert np.all(state.p.data > 0.0)


def test_a_starting_guess_is_used_rather_than_recomputed(diode):
    """Continuation depends on this. The guess is the whole mechanism."""
    warm = solve_bias_newton(diode)
    again = solve_bias_newton(diode, guess=warm)

    assert again.newton is not None
    assert again.newton.iterations < warm.newton.iterations


def test_the_initial_guess_is_not_mutated(diode):
    """A driver that edits its guess makes continuation unrepeatable."""
    guess = initial_state(diode)
    before = guess.psi.data.copy()

    solve_bias_newton(diode, guess=guess)

    np.testing.assert_array_equal(guess.psi.data, before)


def test_a_failed_solve_is_reported_not_raised(diode):
    """A failed solve is information to inspect, matching newton_solve."""
    state = solve_bias_newton(diode, max_iterations=1)

    assert state.newton is not None
    assert not state.newton.converged
    assert state.newton.message


def test_models_can_be_supplied(diode):
    """Phase 4 swaps the mobility model in here, so it has to be a seam."""
    models = TransportModels.for_device(diode)
    state = solve_bias_newton(diode, models=models)

    assert state.newton is not None
    assert state.newton.converged, state.newton.message


# ---------------------------------------------------------------- the hybrid


@pytest.fixture
def hard_case():
    """A device and guess where cold Newton diverges, so the hybrid has a job.

    1e15 doping on 41 nodes at 1.2 V. Coarse enough that the starting guess is
    a long way from the answer and lightly doped enough that the depletion
    region is a large fraction of the device. Measured bare: 30 steps, 28 of
    them against the limiter, 42 nodes with a negative density, and a final
    residual of 1.3e5.

    Not a contrived case. It is the same diode as everywhere else, one decade
    lighter and five times coarser.

    The guess is the equilibrium of the unbiased device, which is what the
    first step of any ramp starts from. It cannot be initial_state of the
    biased device: the Poisson solve at frozen quasi-Fermi levels does not
    itself converge at 1.2 V here, so there is no cold guess at that bias to
    be had. That is a separate limit and it is recorded in PROGRESS.md.

    Returns (device at 1.2 V, guess at 0 V).
    """
    unbiased = pn_diode(Na=1e15, Nd=1e15, n_nodes=41, h_min=2e-7)
    return unbiased.with_bias(anode=1.2, cathode=0.0), initial_state(unbiased)


def test_bare_newton_diverges_on_the_hard_case(hard_case):
    """Pins the premise the hybrid test rests on.

    Without this, a hybrid that passes proves nothing: the case has to be one
    that bare Newton actually fails. If this ever starts passing, the cold
    start got better and the hybrid case needs to be made harder or retired.
    """
    device, guess = hard_case
    state = solve_bias_newton(device, guess=guess)

    assert state.newton is not None
    assert not state.newton.converged
    assert np.any(state.n.data <= 0.0) or np.any(state.p.data <= 0.0)


def test_the_hybrid_converges_where_bare_newton_diverges(hard_case):
    """docs/02-numerics.md: Gummel for 3 to 5 cycles, then switch to Newton.

    Measured on this device: two cycles are enough to turn a divergence into
    an eight step Newton solve, three into seven, five into six.
    """
    device, guess = hard_case
    state = solve_bias_hybrid(device, guess=guess)

    assert state.newton is not None
    assert state.newton.converged, state.newton.message
    assert np.all(state.n.data > 0.0)
    assert np.all(state.p.data > 0.0)


def test_the_hybrid_keeps_its_quadratic_tail(hard_case):
    """The prelude buys the basin. It must not cost the convergence rate."""
    device, guess = hard_case
    state = solve_bias_hybrid(device, guess=guess)

    assert state.newton is not None
    assert state.newton.residual_history[-1] < 1e-13
    assert state.newton.iterations < 15


def test_the_hybrid_reports_the_prelude_it_ran(hard_case):
    """Log every switch, per docs/02-numerics.md.

    A solve that needed a prelude and a solve that did not are different
    events, and only one of them is a sign the guess is getting thin.
    """
    device, guess = hard_case
    # Models supplied rather than rebuilt, which is what continuation does:
    # the lifetimes come from the doping and do not move with the bias.
    state = solve_bias_hybrid(
        device, models=TransportModels.for_device(device), guess=guess
    )

    assert state.gummel is not None
    assert state.gummel.iterations > 0


def test_the_hybrid_agrees_with_bare_newton_where_both_converge(diode):
    """One set of equations, so the route to the answer cannot change it."""
    hybrid = solve_bias_hybrid(diode)
    bare = solve_bias_newton(diode)

    assert hybrid.newton is not None and hybrid.newton.converged
    assert bare.newton is not None and bare.newton.converged

    assert np.max(np.abs(hybrid.psi.data - bare.psi.data)) < 1e-8
    assert (
        np.max(np.abs(hybrid.n.data - bare.n.data) / (np.abs(bare.n.data) + 1.0))
        < 1e-8
    )


def test_no_prelude_reduces_to_bare_newton(hard_case):
    """The prelude is the only difference, so switching it off removes it."""
    device, guess = hard_case
    state = solve_bias_hybrid(
        device, guess=guess, gummel_cycles=0, retry_cycles=0
    )

    assert state.newton is not None
    assert not state.newton.converged


def test_the_hybrid_retries_with_more_gummel_after_a_newton_failure(hard_case):
    """The fallback half of the prescription, exercised rather than assumed.

    One prelude cycle is measured to be not enough on this device: Newton
    still diverges to 41 negative densities. The retry has to notice that and
    run more Gummel from the state before Newton touched it, never from the
    diverged one.
    """
    device, guess = hard_case
    state = solve_bias_hybrid(device, guess=guess, gummel_cycles=1, retry_cycles=4)

    assert state.newton is not None
    assert state.newton.converged, state.newton.message
    assert np.all(state.n.data > 0.0)


def test_the_hybrid_does_not_mutate_its_guess(diode):
    """Continuation calls this in a loop and reuses the state it passed in."""
    guess = initial_state(diode)
    before = guess.psi.data.copy()

    solve_bias_hybrid(diode, guess=guess)

    np.testing.assert_array_equal(guess.psi.data, before)


def test_a_prelude_that_fails_still_hands_its_state_to_newton():
    """A prelude is not a solve, so a block giving up inside it is not fatal.

    At 10 V forward on this device the Poisson block of the first Gummel cycle
    exhausts its own budget and raises. The state at that point is still a
    better guess than the one the cycle started from, and Newton is entitled
    to try it and fail in its own way. Swallowing the exception here rather
    than propagating it is what keeps continuation able to read a failure and
    halve its step.
    """
    unbiased = pn_diode(Na=1e15, Nd=1e15, n_nodes=41, h_min=2e-7)
    device = unbiased.with_bias(anode=10.0, cathode=0.0)

    state = solve_bias_hybrid(device, guess=initial_state(unbiased))

    assert state.newton is not None
    assert not state.newton.converged
