"""Lombardi surface mobility, wired into a MOSFET solve.

physics/mobility.py is tested on its own against analytic limits. This file is
about the two things that only exist once the model meets a device: the field
normal to the interface, and the outer fixed point that lets a frozen mobility
converge to a self consistent one.

Why the mobility is frozen inside a Newton solve
------------------------------------------------
Lombardi reads the field normal to the Si/SiO2 interface. For a horizontal
channel edge, which is the edge the whole model is about, that field lives on
the vertical edges above and below the edge's two endpoints rather than on the
edge itself. Carrying the dependence exactly would widen the Jacobian stencil
past the edge based pattern every assembly in `discretize/` is built on, and
would rewrite code that four phases of results rest on.

So it is frozen within each Newton solve and an outer loop refreshes it. That
buys two properties, and both are asserted below rather than asserted about:

- inside a sweep the residual and the Jacobian come from the same frozen
  mobility, so they agree exactly and the nine block complex step verification
  in test_coupled.py is untouched
- at the fixed point the mobility the solve used is the mobility the answer
  implies, so the converged state solves the true equations

The second is the one that matters and it is the sharpest test here: re-solving
from the answer, with the mobility that answer implies, has to cost nothing and
move nothing. See test_the_converged_state_is_self_consistent.

The device is deliberately coarse. Nothing asserted here is a converged number
against a reference; they are all ratios, counts and invariants, and a mesh
fine enough for a quantitative claim would put a minute into the suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.core import constants as C
from ddsim.device.mosfet import nmos
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import (
    TransportModels,
    _low_field_models,
    _needs_a_low_field_prelude,
    _surface_fixed_point,
    _surface_moved,
    _surface_scattering,
    initial_state,
    solve_bias_newton,
)
from ddsim.discretize.coupled import pack, unpack
from ddsim.extract.iv import gate_sweep, terminal_currents
from ddsim.mesh.mesh2d import normal_field
from ddsim.physics.mobility import CaugheyThomas
from ddsim.solve.newton import NewtonResult

COARSE = {
    "n_contact": 4,
    "n_sd": 10,
    "n_channel": 12,
    "n_silicon": 29,
    "n_oxide": 4,
    "h_min_x": 5e-7,
    "h_min_y": 1e-7,
    "drain_voltage": 0.05,
}
"""A MOSFET small enough to solve several times in a unit test."""


def fet(gate_voltage: float = 0.0):
    """The coarse NMOS at one gate bias."""
    return nmos(gate_voltage=gate_voltage, **COARSE)


@pytest.fixture(scope="module")
def device():
    return fet()


@pytest.fixture(scope="module")
def bulk_models(device):
    """Arora alone, which is where Phase 3 left the mobility."""
    return TransportModels.for_device(device, mobility="arora")


@pytest.fixture(scope="module")
def surface_models(device):
    """Arora with Lombardi scattering on top."""
    return TransportModels.for_device(device, mobility="arora", surface=True)


@pytest.fixture(scope="module")
def inverted(surface_models):
    """The device solved up to 2 V of gate bias, by continuation.

    Module scoped because every test below that needs an inversion layer needs
    the same one, and each solve is seconds rather than milliseconds.
    """
    state = None
    for gate_voltage in (0.0, 1.0, 2.0):
        state = solve_bias_newton(
            fet(gate_voltage), surface_models, guess=state, max_iterations=60
        )
    return state


def sweeps_taken(result) -> int:
    """How many Newton solves the fixed point needed.

    Each contributes its iterations plus one final residual to the history, so
    the difference between the two lengths counts the solves. Reading it this
    way rather than reporting it separately keeps NewtonResult the shape every
    other caller already knows.
    """
    return len(result.residual_history) - result.iterations


# ------------------------------------------- nothing before Phase 5 moved


def test_a_device_has_no_surface_model_unless_it_is_asked_for(bulk_models):
    """Off by default, like Arora and Auger before it, and for the same
    reason: every result recorded before this existed was taken without it."""
    assert bulk_models.surface is None


def test_a_solve_without_one_runs_a_single_newton_and_no_outer_loop(
    device, bulk_models
):
    """The fixed point is not merely inert without a surface model, it does
    not run. A solve that took one Newton before this feature still takes
    exactly one, so no earlier number can have moved."""
    state = solve_bias_newton(device, bulk_models)

    assert state.newton.converged
    assert sweeps_taken(state.newton) == 1


def test_surface_mobility_on_a_line_is_refused():
    """A normal direction is what the model reads, and a 1D mesh has none.
    Refused where it is asked for rather than returning a mobility computed
    from whichever axis happened to be there."""
    diode = pn_diode(Na=1e16, Nd=1e16, length=2e-4, n_nodes=41)
    models = TransportModels.for_device(diode, mobility="arora", surface=True)

    with pytest.raises(TypeError, match="Mesh2D"):
        solve_bias_newton(diode, models)


# ------------------------------------------------------- the fixed point


def test_the_converged_state_is_self_consistent(inverted, surface_models):
    """The sharpest test in this file, and the one that says the frozen
    mobility is not an approximation.

    Take the answer. Build the mobility that answer implies. Solve again from
    it. If the frozen mobility the last sweep used had disagreed with the one
    its own answer implies, this second solve would have somewhere to go.

    It has nowhere to go: zero Newton steps, and psi and n come back bit for
    bit identical rather than merely within a tolerance.
    """
    device = fet(2.0)
    refreshed = surface_models.at_state(
        device, inverted.psi.data, inverted.n.data, inverted.p.data
    )

    again = solve_bias_newton(
        device, refreshed, guess=inverted, max_iterations=60
    )

    assert again.newton.iterations == 0
    np.testing.assert_array_equal(again.psi.data, inverted.psi.data)
    np.testing.assert_array_equal(again.n.data, inverted.n.data)
    np.testing.assert_array_equal(again.p.data, inverted.p.data)


def test_reaching_it_takes_more_than_one_sweep(inverted):
    """If one sweep were always enough the loop would be doing nothing and
    the test above would be proving nothing. An inverted channel moves the
    normal field enough that the first frozen mobility is not the answer."""
    assert sweeps_taken(inverted.newton) > 1


def test_the_reported_cost_is_the_whole_cost(inverted):
    """Newton steps summed over every sweep, not the last one's. A solve that
    reported only its final sweep would look cheaper than it is, and the
    convergence rate this method gives up is the one thing worth watching."""
    result = inverted.newton

    assert result.converged
    assert result.iterations >= sweeps_taken(result)
    assert len(result.update_history) == result.iterations


def test_an_exhausted_sweep_budget_is_reported_as_not_converged(surface_models):
    """The inner Newton converges every time here; what does not converge is
    the fixed point between the mobility and the state it is read from.
    Reporting the inner result would claim a convergence the solve does not
    have, which is exactly the dishonesty docs/07-decisions.md keeps catching
    elsewhere."""
    state = solve_bias_newton(
        fet(2.0),
        surface_models,
        max_iterations=60,
        max_surface_sweeps=1,
        surface_rtol=1e-14,
    )

    assert not state.newton.converged
    assert "surface mobility" in state.newton.message


# ---------------------------------------------------- where it applies


def test_the_oxide_keeps_its_bulk_mobility(inverted, surface_models):
    """An insulator has no surface mobility and its doping is exactly zero,
    which the roughness exponent would raise to a negative power. Both are
    handled at once by leaving those nodes alone.

    Not harmless if it were wrong: an edge from the interface into the oxide
    averages its two ends, so an infinity there reaches the channel row.
    """
    device = fet(2.0)
    surface = surface_models.surface
    mu_n, mu_p = surface.corrected(
        device, inverted.psi.data, inverted.n.data, inverted.p.data
    )
    oxide = list(device.carrier_free_nodes)

    assert oxide, "this device is supposed to have an oxide"
    assert np.all(np.isfinite(mu_n)) and np.all(np.isfinite(mu_p))
    np.testing.assert_array_equal(mu_n[oxide], surface.mu_bulk_n[oxide])
    np.testing.assert_array_equal(mu_p[oxide], surface.mu_bulk_p[oxide])


def test_the_silicon_does_not_keep_its_bulk_mobility(inverted, surface_models):
    """The other half of the test above. A correction applied nowhere would
    pass every oxide assertion and would be worth nothing."""
    surface = surface_models.surface
    mu_n, _ = surface.corrected(
        fet(2.0), inverted.psi.data, inverted.n.data, inverted.p.data
    )
    silicon = surface.semiconductor

    assert np.min(mu_n[silicon] / surface.mu_bulk_n[silicon]) < 0.5


# ------------------------------------------------ what it does to a current


@pytest.fixture(scope="module")
def transfer_curves(device):
    """The same transfer curve with and without the surface model."""
    gates = [0.0, 1.0, 2.0]
    curves = {}
    for surface in (False, True):
        models = TransportModels.for_device(
            device, mobility="arora", surface=surface
        )
        curve = gate_sweep(device, gates, models=models, max_iterations=60)
        curves[surface] = np.asarray(curve.current.data)
    return curves


def test_the_correction_dies_away_from_the_interface(inverted, surface_models):
    """What says the reduction below is a surface term and not a mobility
    knocked down everywhere by a constant.

    Read as a depth profile down the middle of the channel, at 2 V of gate
    bias. Measured: 0.37 of the bulk mobility on the interface row, back
    inside a percent of it one third of the way into the substrate, and
    exactly 1 in the oxide.

    This replaces a comparison of the two off state currents, which is the
    same claim measured somewhere it cannot be measured. At zero gate bias on
    this mesh the drain current is 5e-8 while the sum of the terminal
    currents, which Kirchhoff says is zero, is 3e-8. The quantity is a
    cancellation between fluxes ten decades larger and it carries no
    information: the two models agreed there to 0.8 percent under Boltzmann
    and to 59 percent under Fermi-Dirac, and neither number was about
    mobility. See docs/07-decisions.md.
    """
    device = fet(2.0)
    surface = surface_models.surface
    mu_n, _ = surface.corrected(
        device, inverted.psi.data, inverted.n.data, inverted.p.data
    )
    ratio = (mu_n / surface.mu_bulk_n).reshape(
        device.mesh.y_axis.n_nodes, device.mesh.x_axis.n_nodes
    )
    column = ratio[:, device.mesh.x_axis.n_nodes // 2]

    # The interface is the last silicon row, the oxide sitting above it.
    interface = int(np.argmin(column))
    assert column[interface] < 0.5
    assert column[0] == pytest.approx(1.0, abs=0.01)
    assert np.all(np.diff(column[: interface + 1]) <= 0.0)


def test_the_on_current_falls_by_the_factor_the_physics_doc_names(
    transfer_curves,
):
    """docs/01-physics.md: "Without this your inversion-layer mobility is too
    high by a factor of 2 to 3 and your Id is correspondingly wrong." At 2 V
    on a 20 nm oxide the factor is about two, which is the low end of that
    band and is where a 20 nm oxide should put it: the normal field at a given
    overdrive is half what a 10 nm stack would produce."""
    ratio = transfer_curves[False][-1] / transfer_curves[True][-1]

    assert 1.7 < ratio < 3.0, f"on current fell by {ratio:.2f}x"


def test_the_reduction_grows_with_gate_bias(transfer_curves):
    """The signature that this is a field dependent model and not a constant
    factor. More overdrive presses the inversion layer harder against the
    interface, so it scatters more, so the ratio climbs. A fitted prefactor
    would give the same reduction at every bias."""
    ratios = transfer_curves[False] / transfer_curves[True]

    assert np.all(np.diff(ratios) > 0.0)


# --------------------------------------------------------------- the units


def test_the_normal_field_at_the_channel_is_a_physical_number(inverted):
    """The one place this could be quietly wrong with no other symptom.

    The state is scaled and Lombardi's parameters are physical, so psi is
    multiplied by psi_0 before the field is taken. Drop that and the field is
    39 times too large; confuse it with x_0 and it is decades out. Either way
    the mobility that comes back is positive, monotone in the field, and
    entirely wrong.

    A 2 V gate over 20 nm of oxide puts about 1e6 V/cm across the oxide, and
    the silicon side of the interface sees that reduced by the permittivity
    ratio, so a few times 1e5 V/cm is where the answer belongs. The band here
    is wide on purpose: it is a units check, not a measurement.
    """
    device = fet(2.0)
    mesh = device.mesh
    silicon = np.ones(mesh.n_nodes, dtype=bool)
    silicon[list(device.carrier_free_nodes)] = False

    E = normal_field(mesh, inverted.psi.data * device.scale.psi_0)

    middle = mesh.nx // 2
    column = [mesh.node_at(middle, row) for row in range(mesh.ny)]
    at_the_surface = max(node for node in column if silicon[node])

    assert 1e5 < E[at_the_surface] < 2e6


# ------------------------------------- the driver, with Newton stubbed out


def test_a_sweep_that_fails_still_reports_what_the_earlier_ones_cost(
    device, surface_models
):
    """The failure path, which no real device reaches on this device because
    each sweep starts from the last one's answer and so gets easier rather
    than harder. Newton is stubbed instead of contrived into failing.

    A sweep that fails ends the loop, and what comes back has to carry the
    cost of the sweeps before it. Returning the failing sweep's own result
    would report a solve that had already spent five steps as having spent
    two, and continuation reads that number.
    """
    start = initial_state(device)
    x0 = pack(start.psi.data, start.n.data, start.p.data)

    calls = []

    def run(models, x):
        calls.append(models)
        if len(calls) == 1:
            # Converged, but at a state far enough from the start that the
            # mobility refreshed from it will not have settled.
            psi, n, p = unpack(x)
            moved = pack(psi + 0.5 * np.cos(np.arange(psi.size)), n, p)
            return NewtonResult(
                x=moved,
                converged=True,
                iterations=5,
                residual_history=[1e-2, 1e-6, 1e-9, 1e-12, 1e-14, 1e-15],
                update_history=[1.0, 1e-3, 1e-6, 1e-9, 1e-12],
            )
        return NewtonResult(
            x=x,
            converged=False,
            iterations=2,
            residual_history=[1e-3, 1e-4, 1e-4],
            update_history=[1e-1, 1e-1],
            message="stub refused to converge",
        )

    result = _surface_fixed_point(
        device, surface_models, run, x0, max_sweeps=10, rtol=1e-8
    )

    assert len(calls) == 2, "the first sweep converged, so a second must run"
    assert not result.converged
    assert result.message == "stub refused to converge"
    assert result.iterations == 7
    assert len(result.update_history) == 7


# ------------------------------------------- building the correction


def test_at_state_leaves_models_without_a_surface_alone(bulk_models, device):
    """The identity that lets every caller ask unconditionally. Extraction
    calls it on whatever models it was handed, and a device with no interface
    has to come back untouched rather than through a second code path."""
    state = initial_state(device)

    assert (
        bulk_models.at_state(device, state.psi.data, state.n.data, state.p.data)
        is bulk_models
    )


def test_the_constant_mobility_can_be_corrected_too(device):
    """The surface term corrects whatever bulk model was chosen and does not
    know which. Worth pinning because the constant model is the only one that
    reaches the assembly as a scalar, and the surface path has to turn it into
    a nodal array before it can correct it."""
    models = TransportModels.for_device(
        device, mobility="constant", surface=True
    )

    bulk = models.surface.mu_bulk_n
    assert bulk.shape == (device.mesh.n_nodes,)
    np.testing.assert_allclose(bulk, C.mu_n(device.material.T), rtol=1e-14)


def test_an_unknown_mobility_model_is_refused_here_too(device):
    """The surface path picks its own bulk model, so it has its own place to
    get the name wrong. Refusing in one of the two and defaulting in the other
    is how a typo turns into a silently different device.

    Asserted twice on purpose. Through for_device it is the diffusivity that
    refuses first, so that path never reaches the surface builder's own check
    and would leave it dead. The second call is that check.
    """
    with pytest.raises(ValueError, match="unknown mobility model"):
        TransportModels.for_device(device, mobility="masetti", surface=True)

    with pytest.raises(ValueError, match="unknown mobility model"):
        _surface_scattering(device, "masetti", np.abs(device.net_doping.data))


# -------------------------------------- with velocity saturation as well


@pytest.fixture(scope="module")
def both_models(device):
    """Caughey-Thomas around a surface corrected mobility.

    The combination Phase 5 actually wants, and the one where the wrapping
    order matters: the surface correction is nodal and is applied before the
    average onto the edges, and the saturation factor is applied after it with
    that edge's own parallel drop. It is the order the reference uses.
    """
    return TransportModels.for_device(
        device, mobility="arora", field_dependent=True, surface=True
    )


def test_the_two_field_models_compose(both_models, transfer_curves, device):
    """Both at once, on the same device as the curves above. Velocity
    saturation is nearly idle at 50 mV across a 1 um channel, since the
    parallel field is far below the critical 7.6 kV/cm, so the reduction here
    is the surface term with a little more on top rather than a new effect."""
    curve = gate_sweep(
        device, [0.0, 1.0, 2.0], models=both_models, max_iterations=60
    )
    both = np.asarray(curve.current.data)
    surface_only = transfer_curves[True]

    assert np.all(both[1:] < surface_only[1:])
    assert np.all(both[1:] > 0.9 * surface_only[1:])


def test_the_fixed_point_reaches_through_the_saturation_wrapper(
    both_models, device
):
    """With velocity saturation on, the diffusivity is a model rather than an
    array, and the fixed point is on the low field mobility inside it. Reading
    the wrapper instead would compare two objects and never converge; reading
    only the array case would crash here."""
    state = None
    for gate_voltage in (0.0, 1.0):
        state = solve_bias_newton(
            fet(gate_voltage), both_models, guess=state, max_iterations=60
        )

    assert state.newton.converged
    assert isinstance(both_models.Dn, CaugheyThomas)
    refreshed = both_models.at_state(
        fet(1.0), state.psi.data, state.n.data, state.p.data
    )
    assert isinstance(refreshed.Dn, CaugheyThomas)
    assert _surface_moved(refreshed, refreshed) == 0.0


# --------------------------------------------------- the low field prelude


def test_a_cold_solve_with_velocity_saturation_converges():
    """The regression the prelude exists for. From the Poisson guess this
    device took 60 iterations, 55 of them against the step limiter, and never
    converged. Caughey-Thomas is inside the Jacobian and its mobility falls
    steeply through the critical field, so a step that overshoots lands
    somewhere the linearization did not predict."""
    device = fet()
    models = TransportModels.for_device(
        device, mobility="arora", field_dependent=True
    )

    state = solve_bias_newton(device, models, max_iterations=60)

    assert state.newton.converged
    assert state.newton.iterations < 30


def test_the_prelude_runs_only_where_it_is_needed(device):
    """Cold and field dependent, and neither of the other three. A prelude on
    a warm solve would throw away a guess from the neighbouring bias that is
    worth more than the one it makes, and surface scattering is outside the
    Jacobian and cold starts without help."""
    warm = initial_state(device)
    field = TransportModels.for_device(
        device, mobility="arora", field_dependent=True
    )
    surface = TransportModels.for_device(
        device, mobility="arora", surface=True
    )
    bulk = TransportModels.for_device(device, mobility="arora")

    assert _needs_a_low_field_prelude(field, None)
    assert not _needs_a_low_field_prelude(field, warm)
    assert not _needs_a_low_field_prelude(surface, None)
    assert not _needs_a_low_field_prelude(bulk, None)


def test_the_prelude_models_carry_no_state_dependence(both_models):
    """What the prelude solves is Phase 3's mobility: a fixed array the
    assembly reads rather than evaluates. The saturation wrapper unwraps to
    the low field diffusivity it was built around, so the prelude is solving
    the same device rather than a different one."""
    low = _low_field_models(both_models)

    assert low.surface is None
    assert not low.field_dependent
    assert not isinstance(low.Dn, CaugheyThomas)
    assert not isinstance(low.Dp, CaugheyThomas)
    np.testing.assert_array_equal(low.Dn, both_models.Dn.low_field)


def test_the_prelude_is_counted_in_what_the_solve_cost(device):
    """A two stage solve that reported only its second stage would look
    cheaper than it is, and the whole reason the prelude exists is that the
    second stage cannot be reached without it."""
    models = TransportModels.for_device(
        device, mobility="arora", field_dependent=True
    )

    cold = solve_bias_newton(device, models, max_iterations=60)
    low_field_alone = solve_bias_newton(
        device, _low_field_models(models), max_iterations=60
    )

    assert cold.newton.iterations >= low_field_alone.newton.iterations
    assert len(cold.newton.update_history) == cold.newton.iterations


# ------------------------------------- reading a current off the answer


def test_a_terminal_current_uses_the_diffusivity_the_answer_implies(
    surface_models,
):
    """A current has to be computed with the mobility the solve converged
    with. For a surface corrected one that is a function of the state, so
    reading it off the models as they were built uses the uncorrected bulk
    value instead. Measured, that puts the terminal current sum four decades
    further from zero.

    The threshold is loosened from the 1e-8 a diode is held to, and the reason
    is measured rather than assumed: the sum tracks the Newton residual, the
    last sweep of a fixed point starts warm and stops the moment it crosses
    the threshold rather than running into the quadratic tail, so the residual
    it stops at is the threshold and not machine precision. At
    residual_rtol=1e-12 the sum is 1.2e-10; at the 1e-10 default it is 7.6e-7.
    """
    state = None
    for gate_voltage in (0.0, 1.0, 2.0):
        state = solve_bias_newton(
            fet(gate_voltage),
            surface_models,
            guess=state,
            max_iterations=60,
            residual_rtol=1e-12,
        )

    currents = terminal_currents(fet(2.0), state, surface_models)
    total = abs(sum(currents.values()))

    assert total / abs(currents["drain"]) < 1e-8
