"""MOS C-V, all three regimes: Stage 6 of the Phase 4 plan.

The capacitance is a derivative of a curve tests/analytic/test_mos_cap.py
already computes, so everything here rests on the DC solve being right and asks
only whether the derivative and the charge are.

Why there is no frequency in the low frequency answer
-----------------------------------------------------
docs/02-numerics.md asks for (J_dc + i omega M) x = b. The mass matrix M holds
the dn/dt and dp/dt terms, and there are none: equilibrium Poisson substitutes
Boltzmann in and eliminates n and p as unknowns, so the carriers follow the
potential instantaneously by construction. What is left is the omega to zero
limit of that system, and it is exact rather than approximate. Frequency
dependence needs the coupled 3N system with its mass matrix, which needs 2D
transport, which is Phase 6.

The high frequency end is still reachable, and it is the standard way of
reaching it without time stepping: hold the minority carrier response fixed.
That is what a signal fast compared to minority carrier generation does. It
costs one term in the Jacobian and nothing in the DC solution.

What each regime is checked against
-----------------------------------
Accumulation is a parallel plate, C_ox = eps_ox/t_ox.

Flatband is the sharpest number in the file: C_FB is C_ox in series with
eps_Si/L_D, the extrinsic Debye length, and it comes out to four digits. There
is no fitted quantity anywhere in that statement.

The depletion minimum is the high frequency one, C_ox in series with
eps_Si/W_max, read at the bias where the surface has bent by 2 phi_F, which is
where W_max is defined.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.core import constants as C
from ddsim.device.equilibrium import solve_equilibrium
from ddsim.device.mos_cap import BODY, GATE, mos_cap
from ddsim.extract.cv import (
    Response,
    cv_sweep,
    small_signal_capacitance,
    terminal_charge,
)
from tests.analytic.test_mos_cap import (
    NA,
    T_OX,
    T_SI,
    flatband_voltage,
    gate_bias_for,
    max_depletion_width,
    oxide_capacitance,
    phi_F,
    solved,
)

V_FB = flatband_voltage(-NA, C.PHI_M_N_POLY)
C_OX = oxide_capacitance(T_OX)


def debye_length(net_doping: float) -> float:
    """Extrinsic Debye length [cm], sqrt(eps_Si V_T / (q Na)).

    The screening length of the majority carrier. At flatband it is the only
    length in the semiconductor, so it is what the semiconductor contributes to
    the capacitance there.
    """
    return float(np.sqrt(C.eps_Si() * C.V_T() / (C.q * abs(net_doping))))


def in_series(*capacitances: float) -> float:
    """Capacitors in series [F/cm^2], which is what a stack of layers is."""
    return 1.0 / sum(1.0 / value for value in capacitances)


def capacitance_at(v_gate: float, response=Response.LOW_FREQUENCY, **kwargs):
    device, state = solved(gate_voltage=v_gate, **kwargs)
    return small_signal_capacitance(device, state, GATE, response=response)


# ---------------------------------------------------------------- the charge


def test_the_gate_holds_no_charge_at_flatband():
    """No bending means no field, and no field means nothing on the plates.
    The measurement of a quantity that has to be exactly zero is worth more
    than the measurement of one that has to be nearly something."""
    device, state = solved(gate_voltage=V_FB)
    charge = terminal_charge(device, state, GATE)
    assert abs(charge) < 1e-20, f"{charge:g} C/cm^2"


def semiconductor_charge(device, state) -> float:
    """Total charge held inside the silicon [C/cm^2].

    The same integral the Poisson residual carries, done separately so that
    the neutrality check below compares two things rather than restating one.
    """
    doping = device.net_doping_scaled.data
    density = state.p.data - state.n.data + doping
    scaled = float(np.sum(density * device.charge_volume_scaled))
    scale = device.scale
    return (
        scaled * C.q * scale.C_0 * scale.x_0**2 / device.mesh.x_axis.length
    )


@pytest.mark.parametrize("v_gate", [-3.0, V_FB, 0.0, 2.0], ids=str)
def test_the_gate_charge_is_the_silicon_charge_turned_round(v_gate):
    """Every field line that leaves the gate lands on charge in the silicon or
    on what the body contact had to supply, and there is nowhere else for it
    to go. The three sum to zero to the last bit rather than to a few digits.

    Note which two terminals do not balance each other. The body is an ohmic
    contact sitting in neutral bulk, so it is not the other plate of the
    capacitor and it holds almost nothing; the gate's partner is the depletion
    and inversion charge spread through the silicon.
    """
    device, state = solved(gate_voltage=v_gate)
    gate = terminal_charge(device, state, GATE)
    body = terminal_charge(device, state, BODY)
    silicon = semiconductor_charge(device, state)

    # The floor is the depletion charge, which is what one unit of charge on
    # this device is. At flatband every term above is genuinely zero and a
    # tolerance relative to them alone would be a tolerance relative to noise.
    scale = C.q * NA * max_depletion_width(-NA)
    total = gate + body + silicon
    assert abs(total) < 1e-13 * (abs(gate) + abs(silicon) + scale)


def test_the_body_contact_is_not_the_other_plate(_v_gate=-3.0):
    """The guard on the test above. If the body did hold the balancing charge,
    the neutrality check would pass without the silicon term and would be
    saying something much weaker."""
    device, state = solved(gate_voltage=_v_gate)
    gate = terminal_charge(device, state, GATE)
    body = terminal_charge(device, state, BODY)
    assert abs(body) < 1e-6 * abs(gate)


def test_the_gate_charge_has_the_sign_the_bias_asks_for():
    """Above flatband the gate is positive and the silicon inverts; below it
    the gate is negative and holes accumulate. Getting this backwards leaves
    every capacitance right and every charge wrong."""
    device, state = solved(gate_voltage=V_FB + 1.0)
    assert terminal_charge(device, state, GATE) > 0.0
    device, state = solved(gate_voltage=V_FB - 1.0)
    assert terminal_charge(device, state, GATE) < 0.0


def test_the_accumulation_charge_is_the_oxide_drop():
    """Deep in accumulation almost the whole bias falls across the oxide, so
    the plate charge is C_ox times what is left after the surface takes its
    share. Within a percent once the surface stops moving."""
    v_gate = V_FB - 3.0
    device, state = solved(gate_voltage=v_gate)
    charge = terminal_charge(device, state, GATE)

    from tests.analytic.test_mos_cap import surface_potential

    psi_s = surface_potential(device, state)
    # V_G = V_FB + psi_s + Q/C_ox, so the plate charge is what is left of
    # the bias after flatband and the surface have taken their share.
    expected = C_OX * (v_gate - V_FB - psi_s)
    assert charge == pytest.approx(expected, rel=0.01)


# ----------------------------------------------------------- the three regimes


def test_accumulation_reaches_the_oxide_capacitance():
    """phases/PHASE-4.md gates this at 1 percent. The silicon surface is a
    sheet of holes, so it contributes almost nothing in series and what is
    left is the parallel plate."""
    measured = capacitance_at(V_FB - 5.0)
    assert measured == pytest.approx(C_OX, rel=0.01)
    assert measured < C_OX, "a series capacitance cannot exceed either member"


def test_the_flatband_capacitance_is_the_debye_length_in_series():
    """The sharpest analytic check available on a C-V curve.

    At flatband there is no depletion region and no accumulation layer, so the
    only length the semiconductor has is its extrinsic Debye length, and the
    stack is C_ox in series with eps_Si/L_D. Nothing in that statement is
    fitted or approximated, and the solver reproduces it to about one part in
    ten thousand.
    """
    expected = in_series(C_OX, C.eps_Si() / debye_length(-NA))
    assert capacitance_at(V_FB) == pytest.approx(expected, rel=1e-3)


def test_the_high_frequency_minimum_is_the_maximum_depletion_width():
    """phases/PHASE-4.md: the depletion minimum against the maximum depletion
    width calculation.

    Read at the bias where the surface has bent by 2 phi_F, because that is
    where W_max is defined. Past it the high frequency capacitance keeps
    creeping down as the surface potential creeps past 2 phi_F, so a minimum
    taken over a bias grid would depend on how far the grid went.
    """
    v_th = gate_bias_for(2.0 * phi_F(-NA))
    measured = capacitance_at(v_th, response=Response.HIGH_FREQUENCY)
    expected = in_series(C_OX, C.eps_Si() / max_depletion_width(-NA))
    assert measured == pytest.approx(expected, rel=0.02)


def test_the_low_frequency_curve_comes_back_up_in_inversion():
    """Given time to be generated, the inversion layer screens the gate as
    well as a metal plate does, so the capacitance returns to C_ox."""
    measured = capacitance_at(V_FB + 4.0)
    assert measured == pytest.approx(C_OX, rel=0.02)


def test_the_high_frequency_curve_stays_down_in_inversion():
    """And this is the whole difference between the two. If the minority
    freeze did nothing, both curves would return to C_ox and the test above
    would pass for both."""
    low = capacitance_at(V_FB + 4.0)
    high = capacitance_at(V_FB + 4.0, response=Response.HIGH_FREQUENCY)
    assert high < 0.15 * C_OX
    assert low > 8.0 * high


@pytest.mark.parametrize("v_gate", [-3.0, -2.0, V_FB, V_FB + 0.3], ids=str)
def test_the_two_responses_agree_wherever_there_is_no_inversion(v_gate):
    """Freezing the minority carrier can only matter where there is one. In
    accumulation and depletion the two curves have to be the same curve, and
    any difference is the freeze reaching somewhere it should not."""
    low = capacitance_at(v_gate)
    high = capacitance_at(v_gate, response=Response.HIGH_FREQUENCY)
    assert high == pytest.approx(low, rel=1e-6)


def test_every_capacitance_is_positive_and_below_the_oxide_value():
    """C_ox is one member of a series pair, so it is an upper bound on the
    whole curve whatever the silicon does."""
    curve = cv_sweep(
        mos_cap(substrate_doping=-NA, t_ox=T_OX, t_si=T_SI),
        GATE,
        list(np.linspace(V_FB - 2.0, V_FB + 2.0, 9)),
    )
    assert np.all(curve.capacitance > 0.0)
    assert np.all(curve.capacitance < C_OX)


# ------------------------------------------------------ the derivative itself


@pytest.mark.parametrize("v_gate", [-2.0, -0.5, 0.5], ids=str)
def test_the_exact_derivative_matches_a_central_difference(v_gate):
    """The capacitance is dQ/dV taken by differentiating the solved system
    rather than by differencing two solutions, so it has no step size and no
    truncation error. Checked against the thing it replaces.

    The step is 10 mV. A central difference is second order, so the two should
    part company at around a part in ten thousand, and they do.
    """
    step = 1e-2
    exact = capacitance_at(v_gate)
    ahead = terminal_charge(*solved(gate_voltage=v_gate + step), GATE)
    behind = terminal_charge(*solved(gate_voltage=v_gate - step), GATE)
    assert exact == pytest.approx((ahead - behind) / (2 * step), rel=1e-3)


def test_the_derivative_is_not_the_difference_it_is_checked_against():
    """The guard on the test above. If the capacitance were computed by
    differencing, the agreement would be exact and would prove nothing."""
    step = 1e-2
    v_gate = -0.5
    exact = capacitance_at(v_gate)
    ahead = terminal_charge(*solved(gate_voltage=v_gate + step), GATE)
    behind = terminal_charge(*solved(gate_voltage=v_gate - step), GATE)
    assert exact != (ahead - behind) / (2 * step)


# --------------------------------------------------------------- the other type


def test_an_n_type_substrate_mirrors_the_curve():
    """A PMOS capacitor is the same device with every sign turned round, so
    its C-V is the NMOS one reflected about flatband. Nothing in the solver
    knows which it is being handed, which is the point of the check."""
    metal = C.PHI_M_MIDGAP
    p_body = flatband_voltage(-NA, metal)
    n_body = flatband_voltage(+NA, metal)

    for offset in (-1.5, -0.5, 0.5, 1.5):
        p_type = capacitance_at(p_body + offset, net_doping=-NA,
                                work_function=metal)
        n_type = capacitance_at(n_body - offset, net_doping=+NA,
                                work_function=metal)
        assert p_type == pytest.approx(n_type, rel=1e-6)


# ------------------------------------------------------------------ the sweep


def test_a_sweep_returns_a_point_for_every_bias_asked_for():
    voltages = list(np.linspace(-2.0, 1.0, 7))
    curve = cv_sweep(mos_cap(substrate_doping=-NA, t_ox=T_OX), GATE, voltages)
    assert curve.complete
    np.testing.assert_allclose(curve.gate_voltage, voltages, rtol=1e-14)
    assert len(curve.points) == len(voltages)


def test_a_sweep_agrees_with_solving_each_point_alone():
    """A sweep is a convenience, not a different calculation."""
    voltages = [-2.0, -0.5, 1.0]
    curve = cv_sweep(mos_cap(substrate_doping=-NA, t_ox=T_OX), GATE, voltages)
    for voltage, measured in zip(voltages, curve.capacitance, strict=True):
        assert measured == pytest.approx(capacitance_at(voltage), rel=1e-12)


def test_a_thicker_oxide_lowers_the_whole_curve():
    """C_ox is in series with everything, so halving it halves the ceiling."""
    thin = capacitance_at(V_FB - 5.0)
    device = mos_cap(substrate_doping=-NA, t_ox=2 * T_OX, t_si=T_SI,
                     gate_voltage=V_FB - 5.0)
    thick = small_signal_capacitance(device, solve_equilibrium(device), GATE)
    assert thick == pytest.approx(0.5 * thin, rel=0.02)


# ------------------------------------------------------------- the body bias


def test_the_whole_curve_shifts_with_the_body_bias():
    """A capacitor passes no current, so biasing the body moves the entire
    device with it and the C-V curve translates rigidly. Nothing about its
    shape may change: the same capacitance appears at the same gate to body
    difference.

    This is the check that the sweep carries flat quasi-Fermi levels at the
    body potential. Without them the body bias never reaches the surface at
    all, because at true equilibrium a quasi-neutral region cannot shift its
    potential without shifting its majority carrier density by exp(38.7) per
    volt, so the bias is screened within a few Debye lengths of its own
    contact and the curve does not move.
    """
    offsets = [-2.0, -1.0, 0.0, 1.0]
    grounded = cv_sweep(
        mos_cap(substrate_doping=-NA, t_ox=T_OX),
        GATE,
        [V_FB + offset for offset in offsets],
    )
    body = 0.4
    lifted = cv_sweep(
        mos_cap(substrate_doping=-NA, t_ox=T_OX, body_voltage=body),
        GATE,
        [V_FB + body + offset for offset in offsets],
    )
    np.testing.assert_allclose(
        lifted.capacitance, grounded.capacitance, rtol=1e-9
    )


def test_the_shift_is_a_shift_and_not_a_no_op():
    """The guard on the test above. If the body bias did nothing at all, the
    two sweeps would have to be compared at the same gate voltages to agree,
    and they must not."""
    body = 0.4
    grounded = cv_sweep(mos_cap(substrate_doping=-NA), GATE, [V_FB])
    lifted = cv_sweep(
        mos_cap(substrate_doping=-NA, body_voltage=body), GATE, [V_FB]
    )
    assert lifted.capacitance[0] != pytest.approx(
        grounded.capacitance[0], rel=0.05
    )


def test_a_terminal_that_is_not_being_swept_does_not_move():
    """The boundary condition of the small signal solve, stated directly.

    dpsi/dV is one at the swept terminal and zero at every other, because the
    others are held at fixed bias. Worth asserting on its own rather than
    trusting a capacitance to notice: the body of this capacitor is screened
    from the gate by microns of neutral silicon, so giving it the swept
    terminal's derivative by mistake changes the gate capacitance by nothing
    measurable and every other test in this file still passes.
    """
    from ddsim.extract.cv import _bare_poisson, _potential_derivative

    device, state = solved(gate_voltage=-2.0)
    assembly = _bare_poisson(device, state)
    dpsi = _potential_derivative(
        device, GATE, assembly.rows, assembly.cols, assembly.values
    )

    gate = next(c for c in device.contacts if c.name == GATE)
    body = next(c for c in device.contacts if c.name == BODY)
    np.testing.assert_allclose(
        dpsi[list(gate.nodes)] * device.scale.psi_0, 1.0, rtol=1e-14
    )
    np.testing.assert_array_equal(dpsi[list(body.nodes)], 0.0)


# ------------------------------------------------- a thick oxide at a real bias


THICK_T_OX = 1e-5
"""A 100 nm oxide [cm]. Older technologies and power devices are thicker."""


def thick_oxide_solved(v_gate: float):
    """A solved capacitor with a thick oxide: (device, state).

    Built here rather than through test_mos_cap.solved, which pins t_ox to the
    thin oxide every other test in these two files wants.
    """
    device = mos_cap(
        substrate_doping=-NA,
        t_ox=THICK_T_OX,
        t_si=T_SI,
        work_function=C.PHI_M_N_POLY,
        gate_voltage=v_gate,
    )
    return device, solve_equilibrium(device)


def test_a_thick_oxide_still_reports_a_charge_at_twenty_volts():
    """The gate charge has to stay finite when the oxide carries a real bias.

    A 100 nm oxide is swept to twenty volts as a matter of routine, and the
    potential in it then passes the point where exp overflows in scaled units,
    which is 709, or 18.3 V. The carrier densities are evaluated on the oxide
    nodes too, so they overflow, and the zero charge volume that is supposed to
    make them harmless turns inf into nan rather than into zero.

    Nothing about that is loud. The overflow lands on the topmost oxide row,
    which is the gate contact, and the Dirichlet condition overwrites it before
    the nonlinear solve sees it, so the solve converges and reports success.
    terminal_charge deliberately assembles with no contacts applied, because a
    Dirichlet row would throw away the flux balance that is the charge, so it
    is the extraction and not the solve that returns nan.
    """
    biases = (10.0, 15.0, 20.0, 25.0)
    charges = []
    for v_gate in biases:
        device, state = thick_oxide_solved(v_gate)
        charge = terminal_charge(device, state, GATE)
        assert np.isfinite(charge), (
            f"gate charge at {v_gate:+g} V is {charge}, so the oxide's own "
            "carrier term reached the answer"
        )
        charges.append(charge)

    # Not merely finite. Deep in inversion the surface potential is pinned, so
    # every further volt falls across the oxide and the plate charge grows at
    # exactly C_ox. That is the parallel plate, and it is what says the number
    # is right rather than just present.
    c_ox = oxide_capacitance(THICK_T_OX)
    for index in range(1, len(biases)):
        span = biases[index] - biases[index - 1]
        slope = (charges[index] - charges[index - 1]) / span
        assert slope == pytest.approx(c_ox, rel=0.01), (
            f"between {biases[index - 1]:+g} and {biases[index]:+g} V the gate "
            f"charge grows at {slope:.6e} F/cm^2, not the {c_ox:.6e} a "
            "parallel plate gives"
        )


def test_a_thick_oxide_still_reports_a_capacitance_at_twenty_volts():
    """The same for the derivative, which goes through the same assembly.

    Separate from the charge test because they fail for the same reason but
    are different things to lose, and because a capacitance that quietly reads
    nan is what a C-V plot would show as a gap rather than as an error.
    """
    c_ox = oxide_capacitance(THICK_T_OX)
    for v_gate in (10.0, 15.0, 20.0, 25.0):
        device, state = thick_oxide_solved(v_gate)
        measured = small_signal_capacitance(device, state, GATE)
        assert np.isfinite(measured), f"capacitance at {v_gate:+g} V is {measured}"
        assert measured == pytest.approx(c_ox, rel=0.02), (
            f"deep in inversion the stack is the parallel plate, but at "
            f"{v_gate:+g} V it reads {measured:.6e} against {c_ox:.6e}"
        )
