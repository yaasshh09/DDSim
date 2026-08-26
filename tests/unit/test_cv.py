"""Mechanics of extract/cv.py: what it reports and what it refuses.

The physics is in tests/analytic/test_mos_cv.py. This covers the wiring around
it, including the two things the analytic file never exercises: a 1D device,
whose charge is already a density and needs no width, and a sweep that does not
finish.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.core import constants as C
from ddsim.device.equilibrium import solve_equilibrium
from ddsim.device.mos_cap import BODY, GATE, mos_cap
from ddsim.device.pn_diode import pn_diode
from ddsim.extract.cv import (
    CVCurve,
    Response,
    cv_sweep,
    small_signal_capacitance,
    terminal_charge,
)

NA = 1e16
T_OX = 1e-6


@pytest.fixture(scope="module")
def cap():
    device = mos_cap(substrate_doping=-NA, t_ox=T_OX, gate_voltage=-2.0)
    return device, solve_equilibrium(device)


# ------------------------------------------------------------ naming a contact


@pytest.mark.parametrize(
    "call",
    [
        lambda d, s: terminal_charge(d, s, "drain"),
        lambda d, s: small_signal_capacitance(d, s, "drain"),
    ],
    ids=["charge", "capacitance"],
)
def test_a_terminal_that_is_not_there_is_named_in_the_refusal(cap, call):
    device, state = cap
    with pytest.raises(KeyError, match="drain"):
        call(device, state)


def test_the_refusal_lists_the_terminals_that_are_there(cap):
    device, state = cap
    with pytest.raises(KeyError) as raised:
        terminal_charge(device, state, "drain")
    assert GATE in str(raised.value)
    assert BODY in str(raised.value)


def test_a_sweep_checks_the_contact_before_solving_anything(cap):
    """A thousand converged solves followed by a typo is a bad trade."""
    device, _ = cap
    with pytest.raises(KeyError, match="drain"):
        cv_sweep(device, "drain", [0.0])


# ------------------------------------------------------------ per unit what


def test_a_1d_device_needs_no_width():
    """A 1D device is a slab, so its charge is already per unit area and the
    junction capacitance of a diode comes out in F/cm^2 with nothing to divide
    by. The 2D path has to be told, because a plate has an extent."""
    diode = pn_diode(anode_voltage=-1.0)
    state = solve_equilibrium(diode)
    charge = terminal_charge(diode, state, "anode")
    assert np.isfinite(charge)
    assert charge != 0.0


def test_a_reverse_biased_diode_has_a_junction_capacitance():
    """Not a MOS quantity, and the reason cv.py is not written as MOS code:
    dQ/dV at a contact is a contact property, not a device type."""
    diode = pn_diode(anode_voltage=-1.0)
    capacitance = small_signal_capacitance(
        diode, solve_equilibrium(diode), "anode"
    )
    assert capacitance > 0.0


def test_giving_a_width_scales_the_answer_by_it(cap):
    """The width divides, so a device declared twice as wide reports half the
    charge per unit area."""
    device, state = cap
    natural = terminal_charge(device, state, GATE)
    doubled = terminal_charge(
        device, state, GATE, width=2 * device.mesh.x_axis.length
    )
    assert doubled == pytest.approx(0.5 * natural, rel=1e-14)


def test_the_capacitance_takes_the_same_width(cap):
    device, state = cap
    natural = small_signal_capacitance(device, state, GATE)
    doubled = small_signal_capacitance(
        device, state, GATE, width=2 * device.mesh.x_axis.length
    )
    assert doubled == pytest.approx(0.5 * natural, rel=1e-14)


# ------------------------------------------------------------------ the curve


def test_a_curve_reports_what_it_is():
    curve = cv_sweep(mos_cap(substrate_doping=-NA), GATE, [-1.0, 0.0])
    text = repr(curve)
    assert "2 points" in text and "complete" in text
    assert Response.LOW_FREQUENCY.value in text


def test_an_empty_curve_says_so():
    empty = CVCurve(
        contact=GATE,
        response=Response.LOW_FREQUENCY,
        points=(),
        complete=False,
    )
    assert "empty" in repr(empty)
    assert "stopped early" in repr(empty)
    assert empty.gate_voltage.size == 0
    assert empty.capacitance.size == 0
    assert empty.charge.size == 0


def test_a_sweep_that_cannot_converge_returns_what_it_reached():
    """A budget of one Newton step converges at flatband, where the filled
    guess is already the answer, and nowhere else. So the sweep gets its first
    point and stops, which is the behaviour a stalled sweep should have: keep
    the measurements, say where it stopped, do not raise.
    """
    v_fb = float(C.work_function_difference(C.PHI_M_N_POLY, -NA))
    curve = cv_sweep(
        mos_cap(substrate_doping=-NA),
        GATE,
        [v_fb, v_fb + 2.0],
        max_iterations=1,
    )
    assert not curve.complete
    assert len(curve.points) == 1
    assert "did not converge" in curve.message
    assert f"{v_fb + 2.0:+g}" in curve.message


def test_the_charge_on_the_curve_is_the_charge_at_the_point():
    """The sweep stores both, and they have to be the same two numbers the
    single point functions give."""
    device = mos_cap(substrate_doping=-NA)
    curve = cv_sweep(device, GATE, [-1.5])
    point = curve.points[0]
    biased = device.with_bias(**{GATE: -1.5})
    assert point.charge == pytest.approx(
        terminal_charge(biased, point.state, GATE), rel=1e-14
    )


def test_the_response_is_carried_onto_the_curve():
    curve = cv_sweep(
        mos_cap(substrate_doping=-NA),
        GATE,
        [1.0],
        response=Response.HIGH_FREQUENCY,
    )
    assert curve.response is Response.HIGH_FREQUENCY
    assert curve.capacitance[0] == pytest.approx(
        small_signal_capacitance(
            mos_cap(substrate_doping=-NA, gate_voltage=1.0),
            curve.points[0].state,
            GATE,
            response=Response.HIGH_FREQUENCY,
        ),
        rel=1e-14,
    )
