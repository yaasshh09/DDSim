"""Tests for extract/params.py, parameters read off a measured curve.

Everything here runs on synthetic curves with known answers. An extractor
tested only against solver output cannot tell you whether it is the extractor
or the solver that is wrong, and the whole point of these functions is to be
trustworthy when the solver output is the thing under suspicion.

The ideality factor is defined from the local slope:

    n = (1 / V_T) * dV / d(ln I)

so n = 1 for pure diffusion current and n = 2 for recombination limited
current, with a smooth crossover when both are present.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ddsim.core import constants as C
from ddsim.extract.params import (
    dibl,
    ideality_factor,
    saturation_current,
    subthreshold_slope,
    threshold_constant_current,
    threshold_linear_extrapolation,
    transconductance,
)

VT = C.V_T()
"""Thermal voltage at 300 K [V]."""


def shockley(voltage: np.ndarray, I_s: float, n: float) -> np.ndarray:
    """The ideal diode equation [A/cm^2], for building known test curves."""
    return I_s * np.expm1(voltage / (n * VT))


def exponential(voltage: np.ndarray, I_s: float, n: float) -> np.ndarray:
    """The same curve without the -1 term [A/cm^2].

    Used wherever the extractor itself is under test, so that the answer it
    should return is exactly the n it was built with. The -1 is a real feature
    of a diode and it gets its own test rather than contaminating these.
    """
    return I_s * np.exp(voltage / (n * VT))


# ------------------------------------------------------------ ideality factor


@pytest.mark.parametrize("n", [1.0, 1.5, 2.0])
def test_ideality_is_recovered_from_an_ideal_curve(n: float) -> None:
    """An exact Shockley curve must give back the n it was built with."""
    voltage = np.linspace(0.2, 0.5, 13)
    current = exponential(voltage, 1e-12, n)

    midpoint, extracted = ideality_factor(voltage, current)

    np.testing.assert_allclose(extracted, n, rtol=1e-3)
    np.testing.assert_allclose(midpoint, 0.5 * (voltage[:-1] + voltage[1:]))


def test_ideality_crosses_over_when_two_currents_compete() -> None:
    """The crossover the phase asks for, in a curve where it is put there by hand.

    An n = 2 term dominating at low bias plus an n = 1 term dominating at high
    bias gives a local ideality that starts near 2 and ends near 1. If the
    extractor cannot see it here, it cannot be trusted to see it in a solve.
    """
    voltage = np.linspace(0.1, 1.0, 46)
    current = exponential(voltage, 1e-8, 2.0) + exponential(voltage, 1e-14, 1.0)

    _, extracted = ideality_factor(voltage, current)

    assert extracted[0] > 1.99
    assert extracted[-1] < 1.01
    assert np.all(np.diff(extracted) < 0.0)


def test_ideality_needs_positive_currents() -> None:
    """A reverse biased point has a negative current and no logarithm."""
    with pytest.raises(ValueError, match="positive"):
        ideality_factor(np.array([0.1, 0.2]), np.array([-1e-9, 1e-9]))


def test_ideality_needs_at_least_two_points() -> None:
    with pytest.raises(ValueError, match="two"):
        ideality_factor(np.array([0.1]), np.array([1e-9]))


def test_ideality_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="length"):
        ideality_factor(np.array([0.1, 0.2]), np.array([1e-9]))


def test_ideality_rejects_a_repeated_voltage() -> None:
    """Two points at the same bias give a zero denominator, not an answer."""
    with pytest.raises(ValueError, match="increasing"):
        ideality_factor(np.array([0.2, 0.2]), np.array([1e-9, 2e-9]))


# --------------------------------------------------------- saturation current


@pytest.mark.parametrize(("I_s", "n"), [(1e-12, 1.0), (3.7e-10, 1.0), (1e-9, 2.0)])
def test_saturation_current_is_recovered_from_an_ideal_curve(
    I_s: float, n: float
) -> None:
    """The fit returns both the prefactor and the slope it came from."""
    voltage = np.linspace(0.3, 0.5, 9)
    current = exponential(voltage, I_s, n)

    fitted_I_s, fitted_n = saturation_current(voltage, current)

    np.testing.assert_allclose(fitted_I_s, I_s, rtol=2e-3)
    np.testing.assert_allclose(fitted_n, n, rtol=2e-3)


def test_saturation_current_uses_only_the_requested_window() -> None:
    """The fit has to be able to skip the recombination limited low bias end.

    A diode is only Shockley-like where one mechanism dominates, so extracting
    I_s means choosing that range deliberately rather than fitting everything
    and hoping.
    """
    voltage = np.linspace(0.1, 1.0, 46)
    current = exponential(voltage, 1e-8, 2.0) + exponential(voltage, 1e-14, 1.0)

    whole, _ = saturation_current(voltage, current)
    windowed, fitted_n = saturation_current(voltage, current, window=(0.9, 1.0))

    np.testing.assert_allclose(fitted_n, 1.0, rtol=0.01)
    assert abs(windowed - 1e-14) < abs(whole - 1e-14)


def test_a_fitted_saturation_current_amplifies_the_slope_error() -> None:
    """Why the ideality is trustworthy and the prefactor beside it is not.

    In the window above, one percent of a second mechanism is left in the
    current. The fitted ideality absorbs it and comes back within half a
    percent, but extrapolating that line from 0.95 V back to zero covers 37
    units of exponent, so the same small slope error lands as a 25 percent
    error in I_s.

    The repair is not a better fit. It is to stop extrapolating: with n fixed
    at the value theory says it takes, I_s is measured rather than projected.
    """
    voltage = np.linspace(0.1, 1.0, 46)
    current = exponential(voltage, 1e-8, 2.0) + exponential(voltage, 1e-14, 1.0)

    fitted, _ = saturation_current(voltage, current, window=(0.9, 1.0))
    measured, reported_n = saturation_current(
        voltage, current, window=(0.9, 1.0), ideality=1.0
    )

    assert fitted > 1.2e-14
    np.testing.assert_allclose(measured, 1e-14, rtol=0.02)
    assert reported_n == 1.0


def test_a_fixed_ideality_recovers_I_s_from_a_pure_curve() -> None:
    """The direct measurement, with no fit anywhere in it."""
    voltage = np.linspace(0.3, 0.5, 9)
    current = shockley(voltage, 4.2e-11, 1.0)

    measured, _ = saturation_current(voltage, current, ideality=1.0)

    np.testing.assert_allclose(measured, 4.2e-11, rtol=1e-12)


def test_saturation_current_rejects_an_empty_window() -> None:
    voltage = np.linspace(0.3, 0.5, 5)
    with pytest.raises(ValueError, match="window"):
        saturation_current(voltage, shockley(voltage, 1e-12, 1.0), window=(1.0, 2.0))


def test_saturation_current_ignores_the_minus_one_term_by_choosing_the_window(
) -> None:
    """Above about four V_T the exponential dominates and the fit is clean.

    Below that the -1 in the Shockley equation bends the curve and a straight
    line through ln I is measuring the wrong thing. Rather than subtracting it,
    which would need the answer in advance, the window is chosen to avoid it.
    """
    voltage = np.linspace(0.005, 0.5, 60)
    current = shockley(voltage, 1e-12, 1.0)

    clean, _ = saturation_current(voltage, current, window=(0.2, 0.5))
    contaminated, _ = saturation_current(voltage, current, window=(0.005, 0.5))

    assert abs(clean - 1e-12) < abs(contaminated - 1e-12)


# ------------------------------------------------------------ MOSFET parameters
#
# Same discipline as above: every curve here is built from a closed form whose
# answer is known before the extractor is asked for it. phases/PHASE-5.md grades
# the phase on these numbers, so an extractor that is itself under suspicion is
# no use on the day a real device disagrees with DEVSIM.

THERMAL_LIMIT = 1e3 * VT * math.log(10.0)
"""The 300 K subthreshold floor [mV/decade].

kT/q times ln(10), which is 59.5. phases/PHASE-5.md makes being at or above it
the primary sanity gate, because below it is not reachable by a thermally
activated current and so means a bug.
"""


def subthreshold_curve(gate: np.ndarray, I_0: float, slope: float) -> np.ndarray:
    """An exponential subthreshold characteristic [A/cm].

    Built directly from the slope it is supposed to have, in [mV/decade], so
    the extractor has an exact answer to be right or wrong about.
    """
    return I_0 * 10.0 ** (gate / (slope * 1e-3))


def linear_region_curve(
    gate: np.ndarray, gain: float, threshold: float, drain: float
) -> np.ndarray:
    """Id above threshold in the linear region, zero below it [A/cm].

    Id = gain * (Vg - Vth - Vd/2) * Vd, the textbook form. Written this way on
    purpose: the tangent to it extrapolates to Vg = Vth + Vd/2, never to Vth,
    and the drain term is exactly what closes that gap. An extractor that drops
    the correction is wrong by half the drain bias, which at Vd = 0.05 V is
    25 mV and still looks like a plausible threshold.
    """
    overdrive = gate - threshold - 0.5 * drain
    return np.where(overdrive > 0.0, gain * overdrive * drain, 0.0)


@pytest.mark.parametrize("slope", [60.0, 80.0, 100.0], ids=str)
def test_the_subthreshold_slope_is_read_back_from_a_built_curve(
    slope: float,
) -> None:
    gate = np.linspace(0.0, 0.4, 41)

    measured = subthreshold_slope(gate, subthreshold_curve(gate, 1e-12, slope))

    assert measured == pytest.approx(slope, rel=1e-12)


def test_the_thermal_limit_falls_out_of_a_boltzmann_tail() -> None:
    """59.5 mV/decade is kT/q times ln(10) and is not typed in anywhere.

    A current going as exp(V/V_T) is what a thermally activated population over
    a barrier gives when the gate moves that barrier one for one. Reading the
    limit back off such a curve is the check that the number the phase is
    graded against is the one this function actually measures.
    """
    gate = np.linspace(0.0, 0.3, 61)

    measured = subthreshold_slope(gate, 1e-12 * np.exp(gate / VT))

    assert measured == pytest.approx(THERMAL_LIMIT, rel=1e-12)
    assert THERMAL_LIMIT == pytest.approx(59.5, abs=0.05)


def test_the_subthreshold_slope_reports_the_steepest_part() -> None:
    """A real curve has a different slope everywhere and only one of them is
    the subthreshold slope. The steepest is the one the thermal limit binds, so
    it is the one a broken solve would push below 59.5, and the one to report.
    """
    gate = np.linspace(0.0, 0.6, 61)
    # Offsets chosen so the steep branch is the smaller one at low bias and
    # the shallow branch takes over near 0.48 V. Give them the same prefactor
    # and the steep one is above the shallow one everywhere above zero, so the
    # minimum is the shallow branch alone and the curve has one slope.
    steep = subthreshold_curve(gate, 1e-14, 65.0)
    shallow = subthreshold_curve(gate, 1e-9, 200.0)
    both = np.minimum(steep, shallow)

    assert not np.allclose(both, shallow), "the steep branch has to show"
    assert not np.allclose(both, steep), "the shallow branch has to show"
    assert subthreshold_slope(gate, both) == pytest.approx(65.0, rel=1e-9)


def test_constant_current_threshold_finds_the_crossing() -> None:
    """The bias where the curve crosses the target is closed form on an
    exponential, and log-linear interpolation of a log-linear curve is exact.

    The target is deliberately one that lands between two sweep points. At
    1e-7 the crossing sits at exactly 0.35 V, which is on this grid, and then
    interpolating Id linearly would return the same answer and this test would
    pass without saying anything about how the interpolation is done.
    """
    gate = np.linspace(0.0, 0.5, 51)
    slope, I_0, target = 70.0, 1e-12, 3e-7
    expected = slope * 1e-3 * math.log10(target / I_0)

    assert not np.any(np.isclose(gate, expected)), "the crossing must be off grid"

    measured = threshold_constant_current(
        gate, subthreshold_curve(gate, I_0, slope), target=target
    )

    assert measured == pytest.approx(expected, rel=1e-12)


def test_constant_current_threshold_divides_by_the_width() -> None:
    """The target is quoted per unit width, 100 nA/um in the usual statement,
    so a device twice as wide reaches it at the same gate bias."""
    gate = np.linspace(0.0, 0.5, 51)
    current = subthreshold_curve(gate, 1e-12, 70.0)

    narrow = threshold_constant_current(gate, current, target=1e-7)
    wide = threshold_constant_current(gate, 2.0 * current, target=1e-7, width=2.0)

    assert wide == pytest.approx(narrow, rel=1e-12)


def test_constant_current_threshold_refuses_a_target_off_the_curve() -> None:
    gate = np.linspace(0.0, 0.5, 51)
    current = subthreshold_curve(gate, 1e-12, 70.0)

    with pytest.raises(ValueError, match="never reaches"):
        threshold_constant_current(gate, current, target=1.0)


def test_linear_extrapolation_finds_the_threshold_it_was_built_with() -> None:
    """The tangent at peak transconductance, run back to zero current, less
    half the drain bias. Exact on a curve that is a straight line."""
    gate = np.linspace(0.0, 1.2, 121)
    threshold, drain = 0.42, 0.05

    measured = threshold_linear_extrapolation(
        gate,
        linear_region_curve(gate, gain=1e-3, threshold=threshold, drain=drain),
        drain_voltage=drain,
    )

    assert measured == pytest.approx(threshold, abs=1e-9)


def test_linear_extrapolation_without_the_drain_correction_is_off_by_half() -> None:
    """A test rather than a footnote, because the correction is the easiest
    half of this method to drop and the answer still looks like a threshold."""
    gate = np.linspace(0.0, 1.2, 121)
    threshold, drain = 0.42, 0.05
    current = linear_region_curve(gate, gain=1e-3, threshold=threshold, drain=drain)

    uncorrected = threshold_linear_extrapolation(gate, current)

    assert uncorrected == pytest.approx(threshold + 0.5 * drain, abs=1e-9)


def test_transconductance_is_the_slope_of_the_curve() -> None:
    gate = np.linspace(0.5, 1.2, 71)
    gain, drain = 1e-3, 0.05
    current = linear_region_curve(gate, gain=gain, threshold=0.42, drain=drain)

    _, gm = transconductance(gate, current)

    np.testing.assert_allclose(gm, gain * drain, rtol=1e-12)


def test_transconductance_reports_midpoints_like_the_ideality_does() -> None:
    gate = np.linspace(0.5, 1.2, 71)
    current = linear_region_curve(gate, gain=1e-3, threshold=0.42, drain=0.05)

    midpoint, gm = transconductance(gate, current)

    assert midpoint.size == gate.size - 1 == gm.size
    np.testing.assert_allclose(midpoint, 0.5 * (gate[:-1] + gate[1:]), rtol=1e-14)


def test_dibl_is_the_threshold_shift_per_volt_of_drain() -> None:
    """Positive when the higher drain bias lowered the threshold, which is the
    direction the effect actually goes."""
    measured = dibl(
        threshold_low=0.45,
        threshold_high=0.40,
        drain_low=0.05,
        drain_high=1.0,
    )

    assert measured == pytest.approx(1e3 * 0.05 / 0.95, rel=1e-12)


def test_dibl_is_zero_when_the_threshold_does_not_move() -> None:
    """A long channel device, where the drain cannot reach the barrier."""
    assert dibl(0.45, 0.45, 0.05, 1.0) == 0.0


def test_dibl_refuses_two_equal_drain_biases() -> None:
    with pytest.raises(ValueError, match="two different drain"):
        dibl(0.45, 0.40, 0.05, 0.05)


# ------------------------------------------------- what the extractors refuse


def falling_then_rising(gate: np.ndarray) -> np.ndarray:
    """A curve that dips before it climbs [A/cm].

    Not a physical Id-Vg. It stands in for the thing a half converged sweep
    produces, where one bias point came back below its neighbour and every
    slope taken across it is meaningless.
    """
    return 1e-9 * (1.0 + (gate - 0.25) ** 2)


def test_the_subthreshold_slope_refuses_a_curve_that_doubles_back() -> None:
    """The steepest slope of a curve that is not monotonic is not the
    subthreshold slope, and would come back looking like one."""
    gate = np.linspace(0.0, 0.5, 51)

    with pytest.raises(ValueError, match="rises with the gate"):
        subthreshold_slope(gate, falling_then_rising(gate))


def test_the_constant_current_threshold_refuses_the_same_curve() -> None:
    """Same reason, and it matters more here: the interpolation would pick
    whichever of the two crossings numpy walked into first."""
    gate = np.linspace(0.0, 0.5, 51)

    with pytest.raises(ValueError, match="rises with the gate"):
        threshold_constant_current(gate, falling_then_rising(gate), target=1.2e-9)


def test_the_subthreshold_slope_uses_only_the_requested_window() -> None:
    """Two slopes on one curve, and the window picks which one is reported."""
    gate = np.linspace(0.0, 0.6, 61)
    steep = subthreshold_curve(gate, 1e-14, 65.0)
    shallow = subthreshold_curve(gate, 1e-9, 200.0)
    both = np.minimum(steep, shallow)

    assert subthreshold_slope(gate, both, window=(0.0, 0.3)) == pytest.approx(
        65.0, rel=1e-9
    )
    assert subthreshold_slope(gate, both, window=(0.55, 0.6)) == pytest.approx(
        200.0, rel=1e-9
    )


def test_the_subthreshold_slope_refuses_a_window_with_one_point() -> None:
    gate = np.linspace(0.0, 0.5, 51)
    current = subthreshold_curve(gate, 1e-12, 70.0)

    with pytest.raises(ValueError, match="fewer than two points"):
        subthreshold_slope(gate, current, window=(0.201, 0.209))


def test_linear_extrapolation_refuses_a_curve_that_only_falls() -> None:
    """No peak transconductance means no tangent to run back, and the sign of
    the sweep is the likely reason."""
    gate = np.linspace(0.0, 1.0, 21)

    with pytest.raises(ValueError, match="never rises"):
        threshold_linear_extrapolation(gate, 1e-6 * (1.0 - gate))
