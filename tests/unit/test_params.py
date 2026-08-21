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

import numpy as np
import pytest

from ddsim.core import constants as C
from ddsim.extract.params import ideality_factor, saturation_current

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
