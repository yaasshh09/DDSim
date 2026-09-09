"""The gate length sweep, which is what phases/PHASE-5.md is graded on.

One process, several gate lengths. That is what roll-off means: the vertical
structure is a process, fixed once by an oxide thickness, a channel doping and
an implant, and the gate length is the one thing a designer draws differently
on the same wafer. Scaling the oxide or the junction with the gate as well
would still produce a curve, but it would be a curve of four things moving at
once and nothing in it could be attributed to the channel getting shorter.

So `SHORT_CHANNEL_PROCESS` is a 50 nm generation process, and it is run at 1 um
too, where it is simply a device with a very long channel. The threshold at
the long end is what that process gives with no short channel effect in it, and
every millivolt of difference at the short end came out of the two dimensional
solution rather than out of a parameter.

Cost
----
Every point here is two full coupled transfer curves, so this file uses two
gate lengths and a coarse gate list and asks the questions that need a sweep to
answer. The full 1 um to 50 nm sweep and its plot live in the validation
script, not in the test suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.device.mosfet import DRAIN, nmos
from ddsim.device.transport import TransportModels, solve_bias_newton
from ddsim.extract.iv import terminal_currents
from ddsim.extract.rolloff import (
    SHORT_CHANNEL_PROCESS,
    gate_length_sweep,
    usable_span,
)

LONG = 1e-4
"""The long channel end [cm], 1 um."""

SHORT = 1e-5
"""The short channel end of this test [cm], 100 nm."""

GATE_VOLTAGES = list(np.round(np.arange(-0.2, 0.101, 0.05), 4)) + list(
    np.round(np.arange(0.2, 1.101, 0.1), 4)
)
"""Fine through subthreshold, coarse above it, because each point is a solve.

The subthreshold slope is read over two decades of current, which at 70
mV/decade is 140 mV wide, so a 0.1 V grid puts one point in it and cannot
measure a slope at all. Above threshold the curve is a power law and 0.1 V
resolves it fine.
"""

THERMAL_LIMIT = 59.5
"""kT/q ln 10 at 300 K [mV/decade]. Nothing thermally activated beats it."""


@pytest.fixture(scope="module")
def sweep():
    return gate_length_sweep(
        gate_lengths=[LONG, SHORT],
        gate_voltages=GATE_VOLTAGES,
        overdrive_window=(0.4, 0.9),
    )


# ------------------------------------------------------------- what came back


def test_the_sweep_returns_a_point_per_gate_length(sweep):
    assert [point.L_gate for point in sweep] == [LONG, SHORT]


def test_every_point_carries_both_curves_it_was_extracted_from(sweep):
    """A number without the curve behind it cannot be argued with."""
    for point in sweep:
        assert point.linear.measured_at == "drain"
        assert point.saturated.measured_at == "drain"
        assert len(point.linear.voltage) == len(GATE_VOLTAGES)
        assert len(point.saturated.voltage) == len(GATE_VOLTAGES)


# --------------------------------------------------- the acceptance criteria


def test_the_subthreshold_slope_beats_no_thermal_limit(sweep):
    """The primary sanity gate. Below 59.5 mV/decade at 300 K is a bug, at
    every gate length, and a short channel makes it worse and never better."""
    for point in sweep:
        assert point.subthreshold_slope >= THERMAL_LIMIT


def test_the_subthreshold_slope_degrades_as_the_gate_shortens(sweep):
    """The gate loses control of the barrier to the drain, so the same decade
    of current costs more gate bias."""
    long_channel, short_channel = sweep

    assert short_channel.subthreshold_slope > long_channel.subthreshold_slope


def test_the_threshold_rolls_off(sweep):
    """The source and drain depletion regions share the channel charge the
    gate would otherwise have to deplete itself, so less gate is needed. The
    doping did not change between these two devices and neither did the oxide:
    only the distance between the two junctions did."""
    long_channel, short_channel = sweep

    assert short_channel.threshold_linear < long_channel.threshold_linear


def test_both_threshold_methods_roll_off_the_same_way(sweep):
    """Constant current and linear extrapolation measure different things, so
    they do not have to agree on a value. A roll-off visible in one and absent
    from the other would mean the extraction rather than the device."""
    long_channel, short_channel = sweep

    assert (
        short_channel.threshold_extrapolated
        < long_channel.threshold_extrapolated
    )


def test_drain_induced_barrier_lowering_is_positive_and_worsens(sweep):
    """Raising the drain lowers the source barrier, so the threshold falls,
    and the shorter the channel the further the drain reaches."""
    long_channel, short_channel = sweep

    assert long_channel.dibl > 0.0
    assert short_channel.dibl > long_channel.dibl


def test_the_saturation_exponent_falls_from_the_square_law(sweep):
    """A long channel is Id proportional to overdrive squared. Velocity
    saturation removes one of the two factors, so the exponent falls toward 1
    as the channel field passes the critical field."""
    long_channel, short_channel = sweep

    assert long_channel.saturation_exponent <= 2.0
    assert short_channel.saturation_exponent < long_channel.saturation_exponent
    assert short_channel.saturation_exponent > 1.0


def test_the_short_device_drives_more_current(sweep):
    """Same process, same bias, less channel to push through."""
    long_channel, short_channel = sweep

    assert short_channel.peak_transconductance > (
        long_channel.peak_transconductance
    )


# --------------------------------------------- attributing the exponent


def drain_current(L_gate: float, field_dependent: bool) -> float:
    """Id at one strong inversion bias [A/cm], ramped up to from off.

    Everything except `field_dependent` is identical between the two calls, so
    the ratio of the two answers is what Caughey-Thomas did and nothing else.
    Lombardi is left off in both, because the question here is what the lateral
    field model does and the surface model is a second thing moving.
    """
    models = TransportModels.for_device(
        nmos(L_gate=L_gate, **SHORT_CHANNEL_PROCESS),
        mobility="arora",
        field_dependent=field_dependent,
    )
    state = None
    for gate in (0.0, 0.4, 0.8, 1.2):
        device = nmos(
            L_gate=L_gate,
            gate_voltage=gate,
            drain_voltage=1.0,
            **SHORT_CHANNEL_PROCESS,
        )
        state = solve_bias_newton(device, models=models, guess=state)
        assert state.newton is not None and state.newton.converged

    return float(terminal_currents(device, state)[DRAIN])


def test_velocity_saturation_is_what_holds_the_short_device_back():
    """The exponent alone cannot say why it fell, so this says it.

    A falling saturation exponent is consistent with several things that
    travel together as the gate shortens: two dimensional field effects, the
    series resistance of the source and drain, and the threshold reference
    moving. So instead of reading the exponent, the same device is solved twice
    with only Caughey-Thomas switched, at one strong inversion bias.

    At 1 um the two answers agree to a few percent, because 1 V spread over a
    micron of channel is around the critical field and no further. At 50 nm the
    same switch changes the current by more than half, because 1 V over 50 nm
    is twenty times that field and the carriers stopped speeding up long
    before it. Same equations, same doping, same bias: only the lateral field
    model moved.
    """
    long_free = drain_current(1e-4, field_dependent=False)
    long_saturated = drain_current(1e-4, field_dependent=True)
    short_free = drain_current(5e-6, field_dependent=False)
    short_saturated = drain_current(5e-6, field_dependent=True)

    assert long_saturated == pytest.approx(long_free, rel=0.05)
    assert short_saturated < 0.75 * short_free


# ------------------------------------------------------------------ refusals


def test_a_sweep_with_no_gate_lengths_is_refused():
    with pytest.raises(ValueError, match="at least one gate length"):
        gate_length_sweep(gate_lengths=[], gate_voltages=GATE_VOLTAGES)


def test_two_equal_drain_biases_are_refused():
    """DIBL is a shift per volt of drain, and there is no volt here."""
    with pytest.raises(ValueError, match="two different drain biases"):
        gate_length_sweep(
            gate_lengths=[LONG],
            gate_voltages=GATE_VOLTAGES,
            drain_low=0.05,
            drain_high=0.05,
        )


def test_the_process_is_the_one_that_was_handed_in():
    """It is a plain mapping of nmos arguments on purpose: the sweep does not
    get to hold opinions about the process, and a reader can see the whole
    device in one dict."""
    assert SHORT_CHANNEL_PROCESS["t_ox"] == 2e-7
    assert SHORT_CHANNEL_PROCESS["substrate_doping"] < 0.0
    assert "L_gate" not in SHORT_CHANNEL_PROCESS


# ------------------------------------------------------- trimming the curve


def test_the_leakage_floor_at_the_head_of_a_curve_is_trimmed_off():
    """A long channel device in deep off is reverse junction leakage, which is
    small, negative and does not rise. The extractors take logarithms, so those
    points have to go, and dropping them is what lets one gate list serve a
    1 um device and a 50 nm one in the same sweep."""
    gate = np.array([-0.4, -0.3, -0.2, -0.1, 0.0])
    current = np.array([-6.8e-8, -6.4e-8, 5.9e-7, 1.5e-5, 3.7e-4])

    V, J = usable_span(gate, current)

    np.testing.assert_allclose(V, gate[2:])
    np.testing.assert_allclose(J, current[2:])


def test_a_curve_that_stops_rising_is_trimmed_from_there():
    """Not only the sign. A flat spot is a curve that has stopped resolving
    the current it is reporting, and a threshold interpolated across one is
    interpolated across nothing."""
    gate = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    current = np.array([1e-6, 4e-6, 4e-6, 3e-4, 9e-3, 2e-1])

    V, J = usable_span(gate, current)

    np.testing.assert_allclose(V, gate[2:])


def test_a_curve_that_never_turns_on_is_refused():
    """Three points is the fewest a slope and a crossing can both come from.
    Silently returning a shorter span would hand the extractors a fit through
    two points and a threshold with no error bar anywhere on it."""
    gate = np.array([-0.4, -0.3, -0.2, -0.1])
    current = np.array([-1e-8, -1e-8, -1e-8, 1e-9])

    with pytest.raises(ValueError, match="Extend the gate sweep"):
        usable_span(gate, current)
