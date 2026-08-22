"""Tests for physics/bernoulli.py.

Everything in the Scharfetter-Gummel discretization hinges on B(x) being right.
A branch boundary that is off by 1e-8 does not crash, it just makes the current
slightly wrong in the low field regions where the scheme is supposed to reduce
to central differencing.

The strongest single check here is the reflection identity B(-x) = B(x) + x.
Derivation: B(-x) = -x/(e^-x - 1) = x*e^x/(e^x - 1) = B(x)*e^x, and
B(x)*e^x - B(x) = B(x)*(e^x - 1) = x. Differentiating it gives a second free
identity, B'(-x) = -1 - B'(x), which is tested too.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ddsim.physics.bernoulli import (
    ASYMPTOTE_CUTOFF_DB,
    SERIES_CUTOFF_B,
    SERIES_CUTOFF_DB,
    B,
    _B_negative_branch,
    _B_positive_branch,
    _B_series,
    _dB_expm1_branch,
    _dB_series,
    dB_dx,
)
from tests.reference.complexstep import DEFAULT_STEP as CS_STEP
from tests.reference.complexstep import B_complex
from tests.reference.highprec import (
    COMPLEX_STEP_MAX_ABS_X,
    COMPLEX_STEP_MIN_ABS_X,
    B_reference,
    dB_complex_step,
    dB_reference,
    relative_error,
)

THRESHOLDS = (SERIES_CUTOFF_B, SERIES_CUTOFF_DB, ASYMPTOTE_CUTOFF_DB)


def probe_points() -> list[float]:
    """Points that straddle every branch boundary, plus a broad sweep.

    The threshold straddles are the whole point. A branch bug that only shows
    up one ulp from a boundary is exactly the kind that survives a coarse
    sweep and then corrupts a solve months later.
    """
    points: list[float] = []
    for threshold in THRESHOLDS:
        for offset in (-1e-8, -1e-13, 0.0, 1e-13, 1e-8):
            points += [threshold + offset, -threshold + offset]
        points += [
            math.nextafter(threshold, -math.inf),
            math.nextafter(threshold, math.inf),
            math.nextafter(-threshold, -math.inf),
            math.nextafter(-threshold, math.inf),
        ]
    points += [0.0, 1e-30, 1e-20, 1e-8, 0.5, 1.0, 5.0, 20.0, 40.0, 79.9, 100.0, 300.0]
    points += [-v for v in (1e-30, 1e-20, 1e-8, 0.5, 1.0, 5.0, 20.0, 40.0, 79.9, 300.0)]
    points += list(np.linspace(-100.0, 100.0, 401))
    return sorted(set(points))


PROBES = probe_points()
FINITE_REFERENCE_PROBES = [x for x in PROBES if 1e-30 < abs(x) <= 300.0]


# ------------------------------------------------------------ exact behaviour


def test_B_at_zero_is_exactly_one() -> None:
    assert B(0.0) == 1.0


def test_dB_at_zero_is_exactly_minus_one_half() -> None:
    assert dB_dx(0.0) == -0.5


def test_B_at_negative_zero_is_exactly_one() -> None:
    assert B(-0.0) == 1.0


# ------------------------------------------------- against 80 digit reference


def test_B_matches_high_precision_reference() -> None:
    worst = 0.0
    worst_at = 0.0
    for x in FINITE_REFERENCE_PROBES:
        error = relative_error(float(B(x)), B_reference(x))
        if error > worst:
            worst, worst_at = error, x
    assert worst < 1e-13, f"worst relative error {worst:.3e} at x={worst_at}"


def test_dB_matches_high_precision_reference() -> None:
    worst = 0.0
    worst_at = 0.0
    for x in FINITE_REFERENCE_PROBES:
        error = relative_error(float(dB_dx(x)), dB_reference(x))
        if error > worst:
            worst, worst_at = error, x
    assert worst < 1e-13, f"worst relative error {worst:.3e} at x={worst_at}"


# ---------------------------------------------------------------- identities


def test_B_reflection_identity() -> None:
    """B(-x) = B(x) + x, the strongest available check on the branch logic."""
    worst = 0.0
    worst_at = 0.0
    for x in PROBES:
        if not 0.0 < x <= 100.0:
            continue
        left = float(B(-x))
        right = float(B(x)) + x
        error = relative_error(left, right)
        if error > worst:
            worst, worst_at = error, x
    assert worst < 1e-14, f"worst relative error {worst:.3e} at x={worst_at}"


def test_dB_reflection_identity() -> None:
    """B'(-x) = -1 - B'(x), obtained by differentiating B(-x) = B(x) + x.

    Held to 1e-13 rather than the 1e-14 used for the value identity. The
    closed form multiplies expm1(x) by (1 - x), so one ulp in expm1 becomes
    roughly abs(x) ulps in the result. A dense 8000 point sweep puts the true
    worst at 7.1e-15 near abs(x) = 64, which clears 1e-14 by only 1.4x. That
    is too little margin to survive a different libm on a CI runner, and the
    quantity itself is accurate to 20x better than Phase 0 requires.
    """
    worst = 0.0
    worst_at = 0.0
    for x in PROBES:
        if not 0.0 < x <= 100.0:
            continue
        error = relative_error(float(dB_dx(-x)), -1.0 - float(dB_dx(x)))
        if error > worst:
            worst, worst_at = error, x
    assert worst < 1e-13, f"worst relative error {worst:.3e} at x={worst_at}"


# --------------------------------------------------------- branch continuity


@pytest.mark.parametrize("threshold", [SERIES_CUTOFF_B, -SERIES_CUTOFF_B])
def test_B_branches_agree_at_the_series_boundary(threshold: float) -> None:
    """Both branch kernels evaluated at the same x, not at neighbouring floats.

    Evaluating the function at nextafter(t) on each side measures the slope of
    B across two ulps, which at large x is 1e-14 and looks like a jump that is
    not there.
    """
    series = _B_series(np.array([threshold]))[0]
    closed = (
        _B_positive_branch(np.array([threshold]))[0]
        if threshold > 0
        else _B_negative_branch(np.array([threshold]))[0]
    )
    assert relative_error(float(series), float(closed)) < 1e-13


@pytest.mark.parametrize("threshold", [SERIES_CUTOFF_DB, -SERIES_CUTOFF_DB])
def test_dB_branches_agree_at_the_series_boundary(threshold: float) -> None:
    series = _dB_series(np.array([threshold]))[0]
    closed = _dB_expm1_branch(np.array([threshold]))[0]
    assert relative_error(float(series), float(closed)) < 1e-13


def test_dB_branches_agree_at_the_positive_asymptote_boundary() -> None:
    x = np.array([ASYMPTOTE_CUTOFF_DB])
    expm1_form = _dB_expm1_branch(x)[0]
    asymptote = (1.0 - x[0]) * math.exp(-x[0])
    assert relative_error(float(expm1_form), float(asymptote)) < 1e-13


def test_dB_branches_agree_at_the_negative_asymptote_boundary() -> None:
    """The far negative branch returns exactly -1.0, so the jump must be zero."""
    expm1_form = _dB_expm1_branch(np.array([-ASYMPTOTE_CUTOFF_DB]))[0]
    assert float(expm1_form) == -1.0


# ------------------------------------------------------------------ asymptotes


def test_B_tends_to_minus_x_for_large_negative_x() -> None:
    x = np.array([-50.0, -100.0, -500.0, -1e30])
    np.testing.assert_allclose(B(x), -x, rtol=1e-15)


def test_B_tends_to_x_exp_minus_x_for_large_positive_x() -> None:
    x = np.array([100.0, 300.0, 500.0, 700.0])
    np.testing.assert_allclose(B(x), x * np.exp(-x), rtol=1e-14)


def test_dB_tends_to_minus_one_for_large_negative_x() -> None:
    x = np.array([-50.0, -100.0, -500.0, -1e30])
    np.testing.assert_array_equal(dB_dx(x), np.full(x.shape, -1.0))


def test_dB_tends_to_one_minus_x_times_exp_minus_x_for_large_positive_x() -> None:
    x = np.array([100.0, 300.0, 500.0, 700.0])
    np.testing.assert_allclose(dB_dx(x), (1.0 - x) * np.exp(-x), rtol=1e-14)


# --------------------------------------------------------------- overflow safety


def test_B_underflows_to_zero_rather_than_overflowing() -> None:
    """x/expm1(x) overflows above x = 709. The form used here does not.

    docs/02-numerics.md says to return 0.0 above x = 80, which discards every
    value from B(80) = 1.44e-33 down to B(745) = 3.7e-321. Those are all
    representable, so we keep them.
    """
    assert B(80.0) == pytest.approx(1.443881e-33, rel=1e-6)
    assert B(700.0) > 0.0
    assert B(745.0) > 0.0
    assert B(760.0) == 0.0


def test_B_is_finite_across_the_whole_double_range() -> None:
    x = np.array([-1e300, -1e30, -1e3, 0.0, 1e3, 1e30, 1e300])
    result = B(x)
    assert np.all(np.isfinite(result))


def test_dB_is_finite_across_the_whole_double_range() -> None:
    x = np.array([-1e300, -1e30, -1e3, 0.0, 1e3, 1e30, 1e300])
    result = dB_dx(x)
    assert np.all(np.isfinite(result))


def test_no_divide_overflow_or_invalid_warnings() -> None:
    """Underflow is deliberate and excluded. The other three are always bugs."""
    x = np.array(PROBES + [-1e300, 1e300, 709.0, 710.0, 745.0, 760.0])
    with np.errstate(divide="raise", over="raise", invalid="raise"):
        B(x)
        dB_dx(x)


# ------------------------------------------------------------------- shape


def test_B_is_monotonically_decreasing() -> None:
    x = np.linspace(-100.0, 100.0, 2001)
    assert np.all(np.diff(B(x)) < 0.0)


def test_B_is_positive_everywhere() -> None:
    x = np.linspace(-200.0, 200.0, 2001)
    assert np.all(B(x) > 0.0)


def test_dB_is_negative_everywhere() -> None:
    x = np.linspace(-100.0, 100.0, 2001)
    assert np.all(dB_dx(x) < 0.0)


def test_dB_never_becomes_positive_even_in_the_tails() -> None:
    x = np.array([-1e300, -500.0, 0.0, 500.0, 1e300])
    assert np.all(dB_dx(x) <= 0.0)


# ----------------------------------------------------------------- interface


def test_B_accepts_a_python_float_and_returns_a_float() -> None:
    result = B(1.0)
    assert isinstance(result, float)


def test_dB_accepts_a_python_float_and_returns_a_float() -> None:
    assert isinstance(dB_dx(1.0), float)


def test_B_preserves_input_shape() -> None:
    x = np.linspace(-5.0, 5.0, 12).reshape(3, 4)
    assert B(x).shape == (3, 4)


def test_dB_preserves_input_shape() -> None:
    x = np.linspace(-5.0, 5.0, 12).reshape(3, 4)
    assert dB_dx(x).shape == (3, 4)


def test_vectorized_matches_scalar_evaluation() -> None:
    x = np.array(PROBES)
    vector = B(x)
    scalar = np.array([B(float(v)) for v in x])
    np.testing.assert_array_equal(vector, scalar)


def test_vectorized_derivative_matches_scalar_evaluation() -> None:
    x = np.array(PROBES)
    vector = dB_dx(x)
    scalar = np.array([dB_dx(float(v)) for v in x])
    np.testing.assert_array_equal(vector, scalar)


# ---------------------------------------------------- complex step, as briefed


def test_dB_matches_complex_step_differentiation() -> None:
    """The acceptance criterion, over the range where complex step is exact."""
    worst = 0.0
    worst_at = 0.0
    for x in PROBES:
        if not COMPLEX_STEP_MIN_ABS_X <= abs(x) <= COMPLEX_STEP_MAX_ABS_X:
            continue
        error = relative_error(float(dB_dx(x)), dB_complex_step(x))
        if error > worst:
            worst, worst_at = error, x
    assert worst < 1e-13, f"worst relative error {worst:.3e} at x={worst_at}"


def test_complex_step_reference_is_itself_accurate_where_it_is_used() -> None:
    """Guards the guard.

    Complex step is not exact for B near the origin, so its usable range is
    restricted. This pins that restriction down so nobody later widens the
    range and spends a day debugging a correct implementation.
    """
    for x in (0.1, 0.5, 1.0, 10.0, 100.0, 300.0, -0.1, -1.0, -100.0):
        error = relative_error(dB_complex_step(x), dB_reference(x))
        assert error < 1e-14, f"reference itself is off by {error:.3e} at x={x}"


def test_complex_step_is_untrustworthy_below_the_documented_cutoff() -> None:
    """Documents why COMPLEX_STEP_MIN_ABS_X exists, with a measurement.

    If this ever starts passing, complex step got better and the cutoff can be
    lowered. Until then it stands as the reason the range is restricted.
    """
    error = relative_error(dB_complex_step(1e-6), dB_reference(1e-6))
    assert error > 1e-13


# ------------------------------------------------------ complex arguments


def test_B_preserves_a_complex_dtype() -> None:
    """The continuity residual is verified by complex step, and B is in it.

    Without this, the Scharfetter-Gummel residual discards the imaginary part
    and the complex step Jacobian of every continuity block comes back as
    exactly zero, which reads as agreement rather than as breakage.
    """
    got = B(np.array([0.5 + 1e-20j, -0.5 + 1e-20j]))

    assert np.iscomplexobj(got)


def test_B_on_a_real_valued_complex_array_matches_the_real_branch() -> None:
    """A zero imaginary part must not change the answer."""
    x = np.array([-300.0, -37.0, -1.0, -0.05, 0.0, 0.05, 1.0, 37.0, 300.0])
    got = B(x.astype(np.complex128))

    np.testing.assert_allclose(
        np.asarray(got).real, np.asarray(B(x)), rtol=2e-15, atol=0.0
    )


def test_B_on_complex_input_matches_the_independent_reference() -> None:
    """Checked against the closed form reference, which shares no branch."""
    x = np.array([-300.0, -37.0, -1.0, -0.05, 0.0, 0.05, 1.0, 37.0, 300.0])
    z = x + 1e-20j

    got = np.asarray(B(z))
    expected = B_complex(z)

    np.testing.assert_allclose(got.real, expected.real, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(got.imag, expected.imag, rtol=2e-13, atol=0.0)


def test_B_complex_does_not_overflow_in_the_positive_tail() -> None:
    """The real branch avoids exp(x) past 710 and the complex one must too."""
    got = np.asarray(B(np.array([700.0 + 1e-20j])))[0]

    assert math.isfinite(got.real)
    assert math.isfinite(got.imag)


def test_complex_step_through_B_recovers_dB_dx_at_the_origin() -> None:
    """B'(0) = -1/2, the case the whole complex path exists to make work."""
    step = CS_STEP
    got = np.asarray(B(np.array([complex(0.0, step)])))[0].imag / step

    assert got == pytest.approx(-0.5, rel=1e-14)


@pytest.mark.parametrize("x", [-300.0, -37.0, -1.0, -0.1, 0.1, 1.0, 37.0, 300.0])
def test_complex_step_through_B_recovers_dB_dx(x: float) -> None:
    """The Phase 3 harness path, end to end, over the trustworthy range."""
    got = np.asarray(B(np.array([complex(x, CS_STEP)])))[0].imag / CS_STEP

    assert relative_error(got, dB_reference(x)) < 1e-13
