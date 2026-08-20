"""High precision references for the Bernoulli function.

Two independent references, because neither one alone covers the whole range.

1. A 80 digit decimal evaluation of B and B'. Exact for our purposes anywhere
   the arguments are representable, including right next to the origin. Uses
   only the standard library.

2. Complex step differentiation, which the phase brief asks for. It is exact to
   machine precision for most smooth functions, but not for this one near zero.
   B(z) = z / (exp(z) - 1) has subtractive cancellation inside its own algebra,
   so Im(B(x + ih)) / h loses roughly eps / |x| of relative accuracy. Measured:
   9.3e-11 at x = 1e-6, 3.6e-12 at 1e-4, 3.0e-14 at 1e-2, 3.3e-15 at 0.1. Use
   it only for abs(x) >= COMPLEX_STEP_MIN_ABS_X.

The complex step reference also needs its own expm1, because numpy and math do
not provide one for complex arguments and the naive exp(z) - 1 throws away the
precision that makes complex step worth using in the first place.
"""

from __future__ import annotations

import math
from decimal import Decimal, getcontext

getcontext().prec = 80

COMPLEX_STEP_MIN_ABS_X = 0.1
"""Smallest abs(x) at which complex step differentiation is trustworthy to
1e-13. Below this the reference is worse than the implementation it checks."""

COMPLEX_STEP_MAX_ABS_X = 300.0
"""Largest abs(x) the complex step reference handles. Above roughly 355 the
squared magnitude of exp(z) overflows a double."""

DECIMAL_MIN_ABS_X = 1e-40
"""Below this, exp(x) - 1 underflows to zero even at 80 digits."""


def B_reference(x: float) -> float:
    """B(x) = x / (exp(x) - 1) evaluated at 80 decimal digits [1]."""
    d = Decimal(x)
    return float(d / (d.exp() - 1))


def dB_reference(x: float) -> float:
    """B'(x) = (exp(x)*(1 - x) - 1) / (exp(x) - 1)^2 at 80 digits [1]."""
    d = Decimal(x)
    e = d.exp()
    return float((e * (1 - d) - 1) / ((e - 1) ** 2))


def relative_error(approx: float, exact: float) -> float:
    """Relative error, falling back to absolute error when exact is zero."""
    if exact == 0.0:
        return abs(approx)
    return abs((approx - exact) / exact)


def _complex_expm1(z: complex) -> complex:
    """exp(z) - 1 for complex z, without losing precision near the origin.

    exp(x + iy) - 1 = (e^x - 1) cos y + (cos y - 1) + i e^x sin y

    Splitting it this way keeps expm1 on the real part. With the tiny imaginary
    step used for complex differentiation, cos y is exactly 1.0 and sin y is
    exactly y, so the real part reduces to expm1(x) with no error at all.
    """
    x, y = z.real, z.imag
    real = math.expm1(x) * math.cos(y) + (math.cos(y) - 1.0)
    imag = math.exp(x) * math.sin(y)
    return complex(real, imag)


def dB_complex_step(x: float, h: float = 1e-20) -> float:
    """B'(x) by complex step differentiation [1].

    Only valid for COMPLEX_STEP_MIN_ABS_X <= abs(x) <= COMPLEX_STEP_MAX_ABS_X.
    See the module docstring for why the lower bound exists.
    """
    z = complex(x, h)
    return (z / _complex_expm1(z)).imag / h
