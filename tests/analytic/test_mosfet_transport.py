"""The MOSFET carrying current: the first transfer curve this project produces.

Everything up to here has been a structure that solves. This is the device
working as a transistor, and the checks are the ones that would catch it not
being one.

The subthreshold slope is the primary sanity gate of phases/PHASE-5.md, and it
is the sharpest test in this file for a reason that has nothing to do with
MOSFETs. Below threshold the channel is a barrier and the gate lowers it, so
the current is a Boltzmann population over that barrier. The gate cannot move
the barrier by more than the bias applied to it, so no gate can win more than
one decade of current per kT/q * ln(10), which at 300 K is 59.5 mV. A solver
reporting less than that has a sign error, a scaling error, or a boundary
condition that is feeding the channel from somewhere it should not be. Nothing
about the geometry can rescue it, which is what makes it a gate rather than a
guideline.

The linear region check is the other side of the same coin. At a drain bias
small compared with the overdrive the channel is a resistor, so the current is
proportional to the drain bias and the constant of proportionality is a
conductance. That one is easy to pass by accident and easy to fail loudly: a
channel that is being fed by the contacts rather than by the gate has a
conductance that barely moves with V_G.

What is deliberately not here
-----------------------------
Threshold voltage against gate length, DIBL, and velocity saturation. Those are
the headline results of the phase and they need field dependent mobility and
surface mobility first, because without them the inversion layer mobility is
too high by two to three times and every current is wrong by that factor. What
is claimed here is only what does not depend on the mobility model: that the
device switches, that the slope obeys the thermal limit, and that the terminal
currents balance.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.core import constants as C
from ddsim.device.mosfet import BODY, DRAIN, GATE, SOURCE, nmos
from ddsim.device.transport import TransportModels, solve_bias_newton
from ddsim.extract.iv import gate_sweep, terminal_currents
from ddsim.extract.params import subthreshold_slope

THERMAL_LIMIT = 59.5
"""kT/q * ln(10) at 300 K [mV/decade]. No gate can beat it."""

NA_SUBSTRATE = 1e17
"""Substrate doping the nmos default builds with [cm^-3]."""

T_OX = 2e-6
"""Oxide thickness the nmos default builds with [cm], 20 nm."""

V_DS_LINEAR = 0.05
"""Drain bias for a linear region measurement [V]."""


@pytest.fixture(scope="module")
def transfer():
    """Id against Vg at a small drain bias, from off to well past threshold."""
    device = nmos(drain_voltage=V_DS_LINEAR)
    return gate_sweep(
        device,
        voltages=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5],
        models=TransportModels.for_device(device),
        step=0.1,
    )


@pytest.fixture(scope="module")
def solved_on():
    """The device on, at 1.5 V of gate and a small drain bias."""
    device = nmos(gate_voltage=1.5, drain_voltage=V_DS_LINEAR)
    models = TransportModels.for_device(device)
    state = None
    for gate in (0.0, 0.5, 1.0, 1.5):
        stepped = nmos(gate_voltage=gate, drain_voltage=V_DS_LINEAR)
        state = solve_bias_newton(stepped, models=models, guess=state)
        assert state.newton.converged
    return device, state


# ------------------------------------------------------------ the terminals


def test_the_terminal_currents_sum_to_zero(solved_on):
    """Charge does not accumulate anywhere in a steady state, so what flows in
    at one terminal leaves at the others."""
    device, state = solved_on
    currents = terminal_currents(device, state)

    total = sum(currents.values())
    assert abs(total) < 1e-6 * abs(currents[DRAIN])


def test_the_gate_carries_no_dc_current(solved_on):
    """Exactly none. An ideal insulator passes nothing, so this is a statement
    about the model and not a small number that happens to round to zero."""
    device, state = solved_on

    assert terminal_currents(device, state)[GATE] == 0.0


def test_the_drain_current_comes_out_of_the_source(solved_on):
    """Not out of the body. A channel current runs source to drain, and a body
    current of the same order would mean the junctions are leaking, which at
    this bias would be a boundary condition error rather than physics."""
    device, state = solved_on
    currents = terminal_currents(device, state)

    assert currents[DRAIN] > 0.0
    assert currents[SOURCE] == pytest.approx(-currents[DRAIN], rel=1e-6)
    assert abs(currents[BODY]) < 1e-9 * abs(currents[DRAIN])


# ------------------------------------------------------------- it switches


def test_the_sweep_reaches_every_gate_bias(transfer):
    assert transfer.complete, transfer.message
    assert transfer.contact == GATE
    assert transfer.measured_at == DRAIN


def test_the_curve_says_which_terminal_it_measured(transfer):
    """A transfer curve is the case where the swept and measured terminals
    differ, and the source, drain and body currents of a MOSFET are three
    different curves, so a repr that showed only one name would be ambiguous
    exactly where it matters."""
    assert "gate into drain" in repr(transfer)


def test_the_drain_current_rises_with_every_step_of_gate_bias(transfer):
    assert np.all(np.diff(transfer.current) > 0.0)


def test_the_transistor_switches(transfer):
    """Five decades between the bottom of the sweep and the top of it. A
    resistor with a gate over it would give one."""
    ratio = transfer.current[-1] / transfer.current[0]

    assert ratio > 1e5


# -------------------------------------------------- the subthreshold slope


def test_the_subthreshold_slope_beats_no_thermal_limit(transfer):
    """The primary sanity gate of phases/PHASE-5.md.

    Below 59.5 mV/decade at 300 K is not a better transistor, it is a bug: the
    gate would be moving the channel barrier by more than the bias applied to
    it. Nothing in the solver contains that number. The curve is computed from
    Poisson and two continuity equations, and the limit is read off it.
    """
    assert subthreshold_slope(transfer.voltage, transfer.current) >= THERMAL_LIMIT


def depletion_approximation_slope(Na: float, t_ox: float) -> float:
    """SS from the textbook body factor [mV/decade].

    SS = V_T ln(10) (1 + C_dep / C_ox), with the depletion capacitance taken
    at the surface potential of 2 phi_F that defines strong inversion.

    Deliberately here and not in ddsim/. It is the depletion approximation,
    it is what the solver is being measured against, and a project whose
    success criterion is that short channel effects emerge rather than being
    fitted should not carry a threshold expression anywhere something could
    reach for one.
    """
    phi_F = C.V_T() * np.log(Na / C.n_i())
    W = np.sqrt(2.0 * C.eps_Si() * 2.0 * phi_F / (C.q * Na))
    return float(
        1e3 * C.V_T() * np.log(10.0) * (1.0 + (C.eps_Si() / W) / (C.eps_ox() / t_ox))
    )


def test_the_subthreshold_slope_matches_the_body_factor(transfer):
    """Against the textbook expression, not merely inside a bound.

    93.9 mV/decade at 1e17 under 20 nm of oxide. The solver reports about 8
    percent above that, consistently: 100.8, 101.7 and 102.0 on three
    different gate grids, so it is the physics and not the differencing.

    The sign of the gap is the part worth reading. The depletion capacitance
    above is evaluated at 2 phi_F, which is the surface potential at strong
    inversion, and the steepest part of a subthreshold curve happens below
    that. Less band bending means a thinner depletion layer, a larger C_dep
    and a larger body factor, so a solver that actually integrates Poisson
    through the subthreshold region should land above the formula rather than
    on it. One that landed below it would be the thing to worry about.
    """
    ideal = depletion_approximation_slope(NA_SUBSTRATE, T_OX)
    measured = subthreshold_slope(transfer.voltage, transfer.current)

    assert ideal == pytest.approx(93.9, abs=0.5)
    assert measured == pytest.approx(ideal, rel=0.15)
    assert measured > ideal


# ----------------------------------------------------------- linear region


def test_the_channel_is_ohmic_at_a_small_drain_bias():
    """Id proportional to Vd, which is what "linear region" means.

    Measured as a conductance at two drain biases. They do not agree exactly
    and should not: even at 20 and 40 mV the drain is already taking some
    charge out of the channel near it, which is the beginning of saturation.
    Three percent is what that costs here, and a channel fed by anything other
    than the gate would miss by far more.
    """
    models = TransportModels.for_device(nmos())
    state = None
    for gate in (0.0, 0.5, 1.0, 1.5):
        state = solve_bias_newton(
            nmos(gate_voltage=gate), models=models, guess=state
        )
        assert state.newton.converged

    conductance = []
    for drain in (0.02, 0.04):
        device = nmos(gate_voltage=1.5, drain_voltage=drain)
        solved = solve_bias_newton(device, models=models, guess=state)
        assert solved.newton.converged
        conductance.append(terminal_currents(device, solved)[DRAIN] / drain)

    assert conductance[1] == pytest.approx(conductance[0], rel=0.03)


def test_the_conductance_rises_with_gate_bias():
    """The gate is what makes the channel, so the channel resistance has to be
    something the gate sets. A conduction path the gate does not control, a
    leak through the substrate or along the oxide, would show up here as a
    conductance that hardly moves."""
    models = TransportModels.for_device(nmos())
    state = None
    conductance = {}
    for gate in (0.0, 0.5, 1.0, 1.5):
        device = nmos(gate_voltage=gate, drain_voltage=V_DS_LINEAR)
        state = solve_bias_newton(device, models=models, guess=state)
        assert state.newton.converged
        conductance[gate] = (
            terminal_currents(device, state)[DRAIN] / V_DS_LINEAR
        )

    assert conductance[1.5] > 1e4 * conductance[0.0]


# --------------------------------------------------------------- a stall


def test_a_gate_sweep_stops_and_says_where_when_it_stalls():
    """A sweep that cannot reach a bias returns what it did reach and why.

    That matters more than it sounds. The alternative is keeping the
    unconverged point, and nothing downstream can tell: a threshold voltage
    read off a curve with one bad point looks exactly like one read off a good
    curve. So a point is either converged or it is not in the curve.

    Stalled on purpose by starving Newton of iterations rather than by asking
    for impossible physics, so the failure is the solver giving up and not the
    device doing something interesting.
    """
    curve = gate_sweep(
        nmos(drain_voltage=V_DS_LINEAR),
        voltages=[0.4, 4.0],
        step=0.4,
        min_step=0.25,
        max_iterations=7,
    )

    assert not curve.complete
    assert "stalled on the way to +4 V" in curve.message
    assert list(curve.voltage) == [0.4]


def test_a_gate_sweep_that_cannot_even_start_raises():
    """Every point is continued from the starting bias, so a curve with no
    valid start is not a short curve, it is nothing at all."""
    with pytest.raises(RuntimeError, match="could not be started"):
        gate_sweep(
            nmos(drain_voltage=V_DS_LINEAR),
            voltages=[0.5],
            max_iterations=2,
        )


def test_a_gate_sweep_rejects_a_terminal_the_device_does_not_have():
    """Before any solving, since the alternative is finding out after the
    first bias point has been paid for."""
    with pytest.raises(KeyError, match="no contact named 'collector'"):
        gate_sweep(nmos(), voltages=[0.5], measure_at="collector")
