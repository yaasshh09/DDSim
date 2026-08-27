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

from ddsim.device.equilibrium import solve_equilibrium
from ddsim.device.mos_cap import GATE, mos_cap
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import (
    TransportModels,
    initial_state,
    solve_bias_hybrid,
    solve_bias_newton,
)
from ddsim.discretize.boundary import (
    Carrier,
    gate_psi_scaled,
    ohmic_density_scaled,
    ohmic_psi_scaled,
)
from ddsim.discretize.coupled import (
    UNKNOWNS_PER_NODE,
    Unknown,
    limit_psi_step,
    pack,
)
from ddsim.extract.iv import terminal_currents
from ddsim.physics.recombination import SumOfRecombination


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
    be had. That is a separate limit and it is recorded in
    docs/07-decisions.md.

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


# ------------------------------------------------ the Phase 3 physics models


def test_doping_dependent_mobility_lowers_the_diffusivity(diode):
    """Arora at 1e16 gives about 1230 against the constant model's 1417.

    The seam works the way physics/mobility.py says it does: Dn becomes an
    array over edges and nothing in the assembly changes.
    """
    constant = TransportModels.for_device(diode)
    arora = TransportModels.for_device(diode, mobility="arora")

    assert np.isscalar(constant.Dn) or np.ndim(constant.Dn) == 0
    assert np.ndim(arora.Dn) == 1
    assert np.size(arora.Dn) == diode.mesh.n_edges
    assert np.all(np.asarray(arora.Dn) < constant.Dn)


def test_doping_dependent_mobility_still_converges(diode):
    """A per edge diffusivity has to solve exactly as a scalar one does."""
    state = solve_bias_newton(
        diode, models=TransportModels.for_device(diode, mobility="arora")
    )

    assert state.newton is not None
    assert state.newton.converged, state.newton.message
    assert np.all(state.n.data > 0.0)


def test_lower_mobility_gives_less_current(diode):
    """The whole reason the model matters, and a sign check on the wiring.

    Arora reduces the mobility at 1e16, so it has to reduce the current. If it
    raised it, the model would be inverted somewhere between the doping and
    the diffusivity and every quantitative result downstream would be wrong in
    a way no convergence check would notice.
    """
    biased = diode.with_bias(anode=0.4, cathode=0.0)

    constant = TransportModels.for_device(biased)
    arora = TransportModels.for_device(biased, mobility="arora")

    with_constant = terminal_currents(
        biased, solve_bias_newton(biased, models=constant), constant
    )["anode"]
    with_arora = terminal_currents(
        biased, solve_bias_newton(biased, models=arora), arora
    )["anode"]

    assert 0.0 < with_arora < with_constant


def test_auger_can_be_switched_on(diode):
    """Phase 3 scope item 6. SRH and Auger act in parallel, so they add."""
    models = TransportModels.for_device(diode, auger=True)

    assert isinstance(models.recombination, SumOfRecombination)
    assert len(models.recombination.models) == 2


def test_auger_raises_the_recombination_rate(diode):
    """Adding a parallel path can only add rate, never remove it."""
    plain = TransportModels.for_device(diode)
    with_auger = TransportModels.for_device(diode, auger=True)

    n = np.full(diode.mesh.n_nodes, 1e6)
    p = np.full(diode.mesh.n_nodes, 1e6)

    assert np.all(
        np.asarray(with_auger.recombination.rate(n, p))
        > np.asarray(plain.recombination.rate(n, p))
    )


def test_auger_overtakes_srh_as_the_square_of_the_density(diode):
    """Cubic against linear, which is why Auger is a high injection mechanism.

    The physical content of the model is not that Auger is large or small, it
    is how fast it takes over. SRH at high injection goes as n and Auger goes
    as n^3, so their ratio has to go as n^2: exactly a hundredfold per decade
    of density. Measured across nine decades it is 99.1, 99.9, then 100.0 to
    four figures the whole way, and the crossover where Auger overtakes SRH
    lands at about 6e17 cm^-3.

    A coefficient wrong by orders of magnitude would move the crossover and
    leave every other test here passing. A coefficient with the wrong power of
    the density scale would break this one.
    """
    models = TransportModels.for_device(diode, auger=True)
    srh, auger = models.recombination.models

    def ratio(density: float) -> float:
        srh_rate = float(np.max(np.asarray(srh.rate(density, density))))
        return float(np.asarray(auger.rate(density, density))) / srh_rate

    decades = [ratio(10.0**e) for e in range(3, 11)]

    for lower, upper in zip(decades[:-1], decades[1:], strict=True):
        assert upper / lower == pytest.approx(100.0, rel=0.02)

    # Negligible at low injection, dominant well above the crossover.
    assert ratio(1e2) < 1e-9
    assert ratio(1e10) > 1e3


def test_auger_still_converges(diode):
    """The exact Auger tangent is not sign definite, so this is worth running."""
    state = solve_bias_newton(
        diode, models=TransportModels.for_device(diode, auger=True)
    )

    assert state.newton is not None
    assert state.newton.converged, state.newton.message
    assert np.all(state.n.data > 0.0)
    assert np.all(state.p.data > 0.0)


def test_both_models_together_converge(diode):
    """Scope items 6 and 7 at once, which is how Phase 4 will run."""
    models = TransportModels.for_device(diode, mobility="arora", auger=True)
    state = solve_bias_newton(diode, models=models)

    assert state.newton is not None
    assert state.newton.converged, state.newton.message


def test_an_unknown_mobility_model_is_rejected(diode):
    """A typo must not silently fall back to the constant model."""
    with pytest.raises(ValueError, match="mobility"):
        TransportModels.for_device(diode, mobility="arorra")


# ------------------------------------------------------- a gate on the coupled path


def capacitor(gate_voltage: float = 1.0):
    """A small MOS capacitor. Coarse on purpose: nothing here is a physics
    claim about the capacitor, only about what the coupled path does with a
    contact that pins psi alone."""
    return mos_cap(gate_voltage=gate_voltage, n_silicon=41, n_oxide=3)


def test_the_coupled_solve_reproduces_the_capacitor_at_equilibrium():
    """The gating test for the gate, and the sharpest form available.

    No current flows through an ideal insulator, so the semiconductor of a MOS
    capacitor stays in equilibrium with its body contact at every gate bias.
    Equilibrium is an exact fixed point of the coupled system: the continuity
    residuals vanish because there is no flux and no net recombination, and
    what is left is the Poisson equation the equilibrium path already solved.

    So the coupled solve must not move off the equilibrium answer. If the gate
    row were left out, mispinned, or pinned at the ohmic target instead of the
    work function one, the coupled residual there would not vanish and Newton
    would walk away from it. One check covering the gate boundary condition,
    the carrier free pinning and the charge volume together.
    """
    device = capacitor(gate_voltage=1.0)
    reference = solve_equilibrium(device)

    state = solve_bias_newton(device)

    assert state.newton.converged, state.newton.message
    np.testing.assert_allclose(
        state.psi.data, reference.psi.data, rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(state.n.data, reference.n.data, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(state.p.data, reference.p.data, rtol=1e-8, atol=1e-12)


def test_the_gate_potential_is_imposed_exactly():
    """A gate pins psi at its own work function, not at the doping under it.

    ohmic_psi_scaled would read the net doping at a node in the middle of an
    insulator, which is zero, and return the applied bias alone. That is a
    plausible looking number and it is wrong by Phi_MS, which slides the whole
    curve sideways with every regime still looking correct.
    """
    device = capacitor(gate_voltage=1.0)
    gate = next(c for c in device.contacts if c.name == GATE)

    state = solve_bias_newton(device)

    expected = gate_psi_scaled(
        gate.voltage / device.scale.psi_0, gate.work_function
    )
    for node in gate.nodes:
        assert state.psi.data[node] == pytest.approx(expected, rel=1e-14)


def test_the_oxide_holds_no_carriers_on_the_coupled_path():
    """Including the gate nodes, which are metal sitting on the insulator.

    Their continuity rows have no flux and no volume, so they read 0 = 0 and
    the matrix is singular unless something pins them. carrier_free_nodes
    already does, and a gate must not fight it for those rows.
    """
    device = capacitor(gate_voltage=1.0)

    state = solve_bias_newton(device)

    for node in device.carrier_free_nodes:
        assert state.n.data[node] == 0.0
        assert state.p.data[node] == 0.0


def test_the_gate_bias_reaches_the_silicon():
    """Guards the case where the gate is accepted and then ignored.

    A gate quietly left out of the coupled assembly gives a floating oxide,
    which converges perfectly happily and reports a surface that does not care
    what the gate is doing.

    Both solves start from the same guess, which is what makes this a
    statement about the assembly. Started from their own guesses the two would
    differ whatever the assembly did, because initial_state is the equilibrium
    Poisson solve and that path has applied the gate correctly since Phase 4.
    The test would pass with the gate row deleted, which is the mutation it
    exists to catch.
    """
    guess = solve_equilibrium(capacitor(gate_voltage=0.0))

    # Both on the accumulation side, where one Newton solve reaches the answer
    # from this guess. Driving the same guess into inversion in one step does
    # not converge, which is what continuation is for and is not what this
    # test is about.
    held = solve_bias_newton(capacitor(gate_voltage=0.0), guess=guess)
    accumulated = solve_bias_newton(capacitor(gate_voltage=-1.0), guess=guess)

    assert held.newton.converged, held.newton.message
    assert accumulated.newton.converged, accumulated.newton.message
    assert not np.allclose(held.psi.data, accumulated.psi.data)
