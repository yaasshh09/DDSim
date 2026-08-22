"""The Bernoulli function B(x) = x / (exp(x) - 1) and its derivative.

Everything in the Scharfetter-Gummel discretization hinges on evaluating this
correctly. In scaled units the argument is the potential difference across a
mesh edge, so it runs from tiny (low field, where the scheme must reduce to
central differencing) to enormous (a depletion region on a coarse mesh).

Properties worth knowing:

    B(0)  = 1
    B'(0) = -1/2
    B(-x) = B(x) + x                 the strongest identity, used as a test
    B(x) -> -x           as x -> -inf
    B(x) -> x * exp(-x)  as x -> +inf

Branch structure, tuned against an 80 digit reference rather than taken on
faith. Worst measured relative error is 1.7e-16 for B and 3.3e-15 for B', and
the largest branch discontinuity is 1.3e-15.

    B(x)
        x < -1e-4          x / expm1(x)
        abs(x) <= 1e-4     1 - x/2 + x^2/12 - x^4/720
        x > 1e-4           -x * exp(-x) / expm1(-x)

    B'(x)
        x < -80            -1.0
        -80 <= x < -0.1    (E*(1 - x) - x) / E^2,  E = expm1(x)
        abs(x) <= 0.1      -1/2 + x/6 - x^3/180 + x^5/5040 - x^7/151200
        0.1 < x <= 80      (E*(1 - x) - x) / E^2
        x > 80             (1 - x) * exp(-x)

Three deliberate departures from the table in docs/02-numerics.md, all of them
measured:

1. No abs(x) > 80 branch for B, and the positive form is
   -x*exp(-x)/expm1(-x) rather than x/expm1(x). The doc says to return 0.0
   above x = 80. That discards every value from B(80) = 1.44e-33 down to
   B(745) = 3.7e-321, all of which are representable, and puts a 100 percent
   relative discontinuity at the boundary. The form used here never overflows
   (the denominator stays in (0, 1]) and underflows to 0.0 on its own above
   x = 745. On the negative side expm1 saturates to exactly -1.0, so
   x/expm1(x) returns exactly -x with no shortcut needed.

2. The derivative series runs to x^7 and covers abs(x) <= 0.1 rather than
   x^3 and abs(x) <= 1e-4. At x = 1e-4 the closed form in the doc has 1.8e-8
   relative error, five orders of magnitude outside the 1e-13 that Phase 0 is
   required to hit, because exp(x)*(1 - x) - 1 cancels two order one terms to
   produce an order x^2 result. At the wider boundary both branches sit near
   1e-15.

3. The derivative closed form is written with expm1 as (E*(1-x) - x)/E^2. It
   is the same algebra, but the cancellation moves from order one terms down
   to order x terms, worth about three orders of magnitude near the boundary.
   It also needs the x > 80 branch, which the doc does not have at all: the
   doc's form squares exp(x) and overflows for x > 354.9.

The branch kernels are module level rather than inlined so that the branch
continuity tests can evaluate two of them at the same x. Comparing the
assembled function at neighbouring floats instead measures the slope of B,
which at large x looks like a 1e-14 jump that is not there.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Argument = float | complex | npt.NDArray[np.float64] | npt.NDArray[np.complex128]
"""What B accepts and returns [1].

Complex is here for the complex step Jacobian verification, not for physics.
The device never has a complex potential.
"""

SERIES_CUTOFF_B = 1e-4
"""Half width of the Taylor window for B [1].

x/expm1(x) holds 1e-16 relative accuracy everywhere, so this window is not
correcting a precision loss. It exists because x = 0 is 0/0, and it makes
B(0) == 1.0 exact by construction. The 4 term series is good to 1e-17 here.
"""

SERIES_CUTOFF_DB = 0.1
"""Half width of the Taylor window for B' [1].

Load bearing, unlike the one for B. Outside this window the closed form is
accurate to better than 1e-15, inside it the 5 term series is accurate to
4e-16, and in between neither is. See departure 2 in the module docstring.
"""

ASYMPTOTE_CUTOFF_DB = 80.0
"""Where B' switches to (1 - x)*exp(-x) [1].

Any value between 37 and 354 works. Below 37 the asymptote drops a correction
larger than machine epsilon, above 354.9 the expm1 form squares into overflow.
80 measures a branch jump of exactly zero and matches the other threshold in
docs/02-numerics.md.
"""


def _B_series(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Taylor series for B near the origin [1].

    B(x) = 1 - x/2 + x^2/12 - x^4/720 + O(x^6)

    Truncation error at the cutoff is 3e-29 relative, far below double
    precision. There is no x^3 term, the odd Bernoulli numbers above the
    first all vanish.
    """
    return np.asarray(1.0 - x / 2.0 + x * x / 12.0 - x**4 / 720.0)


def _B_negative_branch(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """B for x below the series window [1].

    expm1(x) lives in [-1, 0) for negative x, so this cannot overflow, and for
    x below about -37 it is exactly -1.0, which makes the result exactly -x.
    """
    return x / np.expm1(x)


def _B_positive_branch(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """B for x above the series window [1].

    Algebraically x/(exp(x) - 1), rewritten using exp(x) - 1 = -exp(x)*expm1(-x)
    so that the denominator stays in (0, 1] and nothing ever overflows. Above
    x = 745 the numerator underflows to zero, which is the right answer.
    """
    return -x * np.exp(-x) / np.expm1(-x)


def _complex_expm1(z: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """exp(z) - 1 for complex z, for Re(z) <= 0 [1].

    numpy and math provide no complex expm1, and the naive exp(z) - 1 loses
    three orders of magnitude near the origin.

        exp(x + iy) - 1 = (expm1(x) + exp(x)*(cos(y) - 1)) + i*exp(x)*sin(y)

    cos(y) - 1 is evaluated as -2*sin(y/2)^2 rather than by subtracting from
    one. For the tiny imaginary step of a complex step derivative, cos(y)
    rounds to exactly 1.0 and the subtraction gives exactly 0.0, which throws
    away the entire real part when x is also zero. B(ih) then comes back as
    exactly 1 with no imaginary part and the derivative reads 0.0 instead of
    -0.5. Squaring after the sine keeps the term at any step size.

    Only called with Re(z) <= 0, so exp(x) is bounded by 1 and cannot
    overflow. _B_complex arranges that.
    """
    x = z.real
    y = z.imag

    half_sin = np.sin(y / 2.0)
    cos_minus_one = -2.0 * half_sin * half_sin

    exp_x = np.exp(x)
    real = np.expm1(x) + exp_x * cos_minus_one
    imag = exp_x * np.sin(y)
    return np.asarray(real + 1j * imag, dtype=np.complex128)


def _B_complex(z: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """B for complex arguments [1].

    Exists so that the Scharfetter-Gummel residual preserves the dtype and can
    be differentiated by complex step, which phases/PHASE-3.md makes the
    non-negotiable acceptance criterion for every Jacobian block. Without it
    the residual discards the imaginary part and every continuity block
    verifies against a Jacobian of exactly zero, which reads as agreement.

    Branched on the real part the same way the real function is, so neither
    side overflows. There is no series branch: the closed form z/expm1(z)
    holds full relative accuracy for any nonzero z, and the series in the real
    path exists only to make B(0) exact, which is filled in directly here.
    """
    out = np.empty_like(z)

    # 0/0 at the origin. The limit is exactly 1. A complex step argument is
    # never exactly zero, so this only fires for a real valued complex array.
    origin = z == 0.0
    out[origin] = 1.0

    positive = (z.real > 0.0) & ~origin
    negative = ~positive & ~origin

    if negative.any():
        w = z[negative]
        out[negative] = w / _complex_expm1(w)
    if positive.any():
        # B(z) = -z*exp(-z)/expm1(-z), which keeps the exponent negative.
        w = z[positive]
        out[positive] = -w * np.exp(-w) / _complex_expm1(-w)

    return out


def _dB_series(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Taylor series for B' near the origin [1].

    B'(x) = -1/2 + x/6 - x^3/180 + x^5/5040 - x^7/151200 + O(x^9)

    Truncation at the cutoff is 4e-16 relative. The x^5 and x^7 terms are what
    let the window be wide enough for the closed form to take over cleanly.
    """
    x2 = x * x
    x3 = x2 * x
    series = -0.5 + x / 6.0 - x3 / 180.0 + x3 * x2 / 5040.0 - x3 * x2 * x2 / 151200.0
    return np.asarray(series)


def _dB_expm1_branch(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """B' away from the origin and away from the positive tail [1].

    B'(x) = (E*(1 - x) - x) / E^2 with E = expm1(x).

    Same algebra as (exp(x)*(1 - x) - 1)/(exp(x) - 1)^2, but the subtraction is
    between terms of order x instead of order one, which is worth about three
    orders of magnitude in relative accuracy near the series boundary.
    """
    E = np.expm1(x)
    return np.asarray((E * (1.0 - x) - x) / (E * E))


def _dB_positive_asymptote(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """B' in the positive tail [1].

    B'(x) -> (1 - x)*exp(-x). The dropped correction is order exp(-x), below
    machine epsilon for x > 37. Underflows to zero rather than overflowing.
    """
    return np.asarray((1.0 - x) * np.exp(-x))


def B(x: Argument) -> Argument:
    """Bernoulli function B(x) = x / (exp(x) - 1) [1].

    Dimensionless in and dimensionless out. In the Scharfetter-Gummel flux the
    argument is a scaled potential difference across an edge, which is already
    measured in units of V_T, so no thermal voltage appears here.

    Returns a scalar for scalar input and an array of the same shape
    otherwise, and preserves a complex dtype so that a residual built on it
    can be differentiated by complex step. The real path is untouched by the
    complex one: one dtype test per call, and every array below is float64.
    """
    values = np.asarray(x)
    if np.iscomplexobj(values):
        is_scalar = values.ndim == 0
        out = _B_complex(np.atleast_1d(values).astype(np.complex128))
        return complex(out[0]) if is_scalar else out

    values = np.asarray(x, dtype=np.float64)
    is_scalar = values.ndim == 0
    values = np.atleast_1d(values)
    out = np.empty_like(values)

    # Masked assignment rather than np.where. np.where evaluates both arms, so
    # it would compute 0/0 at x = 0 and raise an invalid value warning even
    # though the result is discarded.
    #
    # Each branch is skipped when nothing falls in it. On a device most edges
    # sit in one branch: the potential is flat through the quasi-neutral
    # regions and rises steadily through the depletion region, so the sign of
    # X rarely changes. Asking is one comparison, and the gather, the
    # evaluation and the scatter it avoids are three passes.
    near_zero = np.abs(values) <= SERIES_CUTOFF_B
    negative = values < -SERIES_CUTOFF_B
    positive = values > SERIES_CUTOFF_B

    if near_zero.any():
        out[near_zero] = _B_series(values[near_zero])
    if negative.any():
        out[negative] = _B_negative_branch(values[negative])
    if positive.any():
        out[positive] = _B_positive_branch(values[positive])

    if is_scalar:
        return float(out[0])
    return out


def dB_dx(x: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
    """Derivative of the Bernoulli function, B'(x) [1].

    Needed for the Newton Jacobian of the continuity equations. Verify it
    against complex step differentiation rather than finite differences, but
    only for abs(x) >= 0.1. Complex step is not exact for this function near
    the origin, because B(z) = z/(exp(z) - 1) carries cancellation inside its
    own algebra rather than in the differencing.

    Returns a float for scalar input and an array of the same shape otherwise.
    """
    values = np.asarray(x, dtype=np.float64)
    is_scalar = values.ndim == 0
    values = np.atleast_1d(values)
    out = np.empty_like(values)

    near_zero = np.abs(values) <= SERIES_CUTOFF_DB
    far_negative = values < -ASYMPTOTE_CUTOFF_DB
    far_positive = values > ASYMPTOTE_CUTOFF_DB
    middle = ~near_zero & ~far_negative & ~far_positive

    # Skipped when empty, for the same reason as in B.
    if near_zero.any():
        out[near_zero] = _dB_series(values[near_zero])
    if far_negative.any():
        out[far_negative] = -1.0
    if middle.any():
        out[middle] = _dB_expm1_branch(values[middle])
    if far_positive.any():
        out[far_positive] = _dB_positive_asymptote(values[far_positive])

    if is_scalar:
        return float(out[0])
    return out
