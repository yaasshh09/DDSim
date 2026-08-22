"""The complex step Jacobian harness, checked before anything trusts it.

phases/PHASE-3.md makes complex step differentiation the acceptance criterion
for every Jacobian block, which puts the harness itself on the critical path.
A harness that is quietly wrong reports a wrong Jacobian as correct, and that
failure is invisible: Newton then stagnates and the stagnation gets blamed on
conditioning. So the harness is validated against the 80 digit decimal
reference from Phase 0 before it is pointed at anything.

The load bearing case is x = 0. The Phase 0 scalar reference in
tests/reference/highprec.py assumes cos(h) rounds to exactly 1.0 for the tiny
imaginary step, which is true, and then drops the resulting cos(h) - 1 term as
negligible, which is true everywhere except at the origin. At x = 0 that term
is the entire real part of expm1(ih) and dropping it returns 0.0 instead of
-0.5. A device at equilibrium in a uniformly doped region has X = 0 on every
edge there, so this is the common case and not a corner.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ddsim.physics.bernoulli import B, dB_dx
from tests.reference.complexstep import (
    DEFAULT_STEP,
    B_complex,
    complex_expm1,
    complex_step_jacobian,
)
from tests.reference.highprec import dB_reference, relative_error

# --------------------------------------------------------- the complex expm1


def test_complex_expm1_matches_expm1_on_the_real_axis() -> None:
    """A zero imaginary part has to reproduce the real function exactly."""
    x = np.array([-40.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 40.0])
    got = complex_expm1(x.astype(np.complex128))

    assert np.all(got.imag == 0.0)
    np.testing.assert_allclose(got.real, np.expm1(x), rtol=1e-15, atol=0.0)


def test_complex_expm1_keeps_the_second_order_term_at_the_origin() -> None:
    """expm1(ih) has real part -h^2/2, and it is the whole answer at x = 0.

    Written as cos(h) - 1 this underflows to exactly 0.0 for h = 1e-20, which
    is what makes the Phase 0 scalar reference return 0 for B'(0). Written as
    -2*sin(h/2)^2 it survives, because the squaring happens after the sine
    rather than inside a subtraction against 1.
    """
    h = 1e-20
    got = complex_expm1(np.array([complex(0.0, h)]))[0]

    assert got.real == pytest.approx(-(h**2) / 2.0, rel=1e-12)
    assert got.imag == pytest.approx(h, rel=1e-15)


# ------------------------------------------------------- the complex Bernoulli


def test_B_complex_reproduces_B_on_the_real_axis() -> None:
    """The complex path is different algebra, so it is checked against B."""
    x = np.array([-300.0, -37.0, -1.0, -0.05, 0.0, 0.05, 1.0, 37.0, 300.0])
    got = B_complex(x.astype(np.complex128))

    assert np.all(got.imag == 0.0)
    np.testing.assert_allclose(
        got.real, np.asarray(B(x)), rtol=2e-15, atol=0.0
    )


def test_B_complex_does_not_overflow_in_the_positive_tail() -> None:
    """exp(x) overflows past 710. The positive branch never evaluates it."""
    got = B_complex(np.array([complex(700.0, 1e-20)]))[0]

    assert math.isfinite(got.real)
    assert math.isfinite(got.imag)


# ------------------------------------------------- complex step against Phase 0


@pytest.mark.parametrize(
    "x", [-300.0, -37.0, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 37.0, 300.0]
)
def test_complex_step_matches_the_decimal_reference_where_it_is_exact(
    x: float,
) -> None:
    """The range Phase 0 established complex step as trustworthy over."""
    got = B_complex(np.array([complex(x, DEFAULT_STEP)]))[0].imag / DEFAULT_STEP

    assert relative_error(got, dB_reference(x)) < 1e-13


def test_complex_step_is_exact_at_the_origin() -> None:
    """B'(0) = -1/2 exactly. This is what the cos(h) - 1 form loses."""
    got = B_complex(np.array([complex(0.0, DEFAULT_STEP)]))[0].imag / DEFAULT_STEP

    assert got == pytest.approx(-0.5, rel=1e-14)


def test_complex_step_recovers_dB_dx_at_the_origin() -> None:
    """The harness and the implementation agree at the case that matters."""
    got = B_complex(np.array([complex(0.0, DEFAULT_STEP)]))[0].imag / DEFAULT_STEP

    assert got == pytest.approx(float(dB_dx(0.0)), rel=1e-14)


# ------------------------------------------------------- the generic harness


def test_complex_step_jacobian_is_exact_for_a_linear_function() -> None:
    """No truncation, no cancellation, and a power of two step: bit for bit.

    Exactness here is the whole reason DEFAULT_STEP is 2**-70 rather than a
    round decimal. With 1e-20 the -7 entry comes back one ulp low.
    """
    A = np.array([[2.0, -1.0, 0.0], [0.5, 3.0, -7.0]])

    def f(x: np.ndarray) -> np.ndarray:
        return A @ x

    got = complex_step_jacobian(f, np.array([1.0, -2.0, 0.5]))

    np.testing.assert_array_equal(got, A)


def test_complex_step_jacobian_matches_an_analytic_jacobian() -> None:
    """A nonlinear case with a hand derived Jacobian."""

    def f(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2 * x[1], np.sin(x[0]) + x[1] ** 3])

    x = np.array([0.7, -1.3])
    expected = np.array(
        [
            [2.0 * x[0] * x[1], x[0] ** 2],
            [np.cos(x[0]), 3.0 * x[1] ** 2],
        ]
    )

    got = complex_step_jacobian(f, x)

    np.testing.assert_allclose(got, expected, rtol=1e-15, atol=0.0)


def test_complex_step_jacobian_rejects_a_residual_that_drops_the_dtype() -> None:
    """A residual that casts to float silently returns a zero Jacobian.

    That is the harness failure mode with the worst consequences, because a
    zero column looks like a correct derivative for any term that happens to
    be absent. Caught rather than reported.
    """

    def f(x: np.ndarray) -> np.ndarray:
        # .real rather than an astype, because filterwarnings = ["error"]
        # already turns the implicit complex to float cast into a raise. A
        # residual that takes the real part deliberately does not warn, and
        # that is the one this guard has to catch.
        return np.asarray(x).real * 2.0

    with pytest.raises(TypeError, match="complex"):
        complex_step_jacobian(f, np.array([1.0, 2.0]))
