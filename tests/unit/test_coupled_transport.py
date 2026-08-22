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
