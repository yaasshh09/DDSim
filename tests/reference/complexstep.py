"""Complex step differentiation over arrays, for verifying Jacobian blocks.

phases/PHASE-3.md makes this the acceptance criterion for the coupled Jacobian:
every block matched to 1e-10 on a 20 node mesh, all nine tested individually,
non-negotiable. This module is that reference.

Complex step works because the imaginary part of a real function never mixes
back into the real part under addition or subtraction. Feed a real function
x + ih and the imaginary part of the result is h*f'(x) with no differencing
and therefore no subtractive cancellation from the step size. It is not exact
for every function, only for the differencing. Any cancellation the function
carries in its own algebra survives, which is the Phase 0 result for the
Bernoulli function near the origin.

Two things have to be right for it to work on this code.

**The residual must preserve the dtype.** A residual that writes into a
float64 output array, or calls np.asarray(..., dtype=np.float64), throws the
imaginary part away and returns a Jacobian of exactly zero. A zero column is
indistinguishable from a term that is genuinely absent, so it reads as
agreement rather than as breakage. complex_step_jacobian raises instead.

**expm1 needs a complex implementation.** numpy and math provide none, and the
naive exp(z) - 1 loses three orders of magnitude near the origin, which looks
exactly like an implementation bug in the code under test. The Phase 0 scalar
version in highprec.py writes the real part as expm1(x)*cos(y) + (cos(y) - 1)
and notes that cos(y) rounds to exactly 1.0 for h = 1e-20. That is true, and
it makes the second term underflow to exactly 0.0, which is fine everywhere
except at x = 0 where that term is the entire real part. There B(ih) comes out
as exactly 1 with zero imaginary part and the reported B'(0) is 0.0 instead of
-0.5. Here cos(y) - 1 is written as -2*sin(y/2)^2, which squares after the
sine instead of subtracting against one, and survives at any step size.

That case is not a corner. A device at equilibrium in a uniformly doped region
has X = psi_right - psi_left equal to zero on every edge there, so most edges
of a 20 node mesh sit exactly on it.

Where complex step is still not exact
-------------------------------------
Phase 0 measured the Bernoulli derivative by complex step at 9.3e-11 relative
at x = 1e-6, 3.6e-12 at 1e-4 and 3.0e-14 at 1e-2. The cause is inside B, not
in the step: Im(B(x + ih)) forms h*expm1(x) - x*h*exp(x), two terms of order
h*x, to produce a result of order h*x^2/2, so it loses about 2*eps/abs(x).
The fix above rescues x = 0 exactly and does nothing for x between zero and
roughly 1e-5, where the harness is worse than the code it is checking. Blocks
that depend on psi through B are checked against the 80 digit decimal
reference as well, for that reason.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

ComplexArray = npt.NDArray[np.complex128]

DEFAULT_STEP = 2.0**-70
"""Imaginary step [1], about 8.5e-22.

Complex step has no subtractive cancellation, so the step only has to be small
enough that the O(h^2) truncation is below machine epsilon and large enough
that h^2 does not underflow. h^2 here is 7.2e-43, far above the smallest
normal double, and the truncation sits at that same level.

A power of two, deliberately. The last two operations of the harness are a
multiply by h inside the function and a divide by h outside it, and for a
power of two both are exact, so a linear function comes back bit for bit.
With h = 1e-20 it does not: a coefficient of -7 returns -6.999999999999999,
because 7*fl(1e-20) needs three mantissa bits it does not have. One ulp is
harmless against a 1e-10 criterion, but a reference that is exact where it
can be is easier to reason about when a block does disagree.
"""


def complex_expm1(z: ComplexArray) -> ComplexArray:
    """exp(z) - 1 for complex z, accurate at the origin [1].

        exp(x + iy) - 1 = (expm1(x)*cos(y) + (cos(y) - 1)) + i*exp(x)*sin(y)

    with cos(y) - 1 evaluated as -2*sin(y/2)^2 so that it does not underflow
    to zero for a differentiation step. See the module docstring.
    """
    x = z.real
    y = z.imag

    half_sin = np.sin(y / 2.0)
    cos_minus_one = -2.0 * half_sin * half_sin
    cos_y = 1.0 + cos_minus_one

    real = np.expm1(x) * cos_y + cos_minus_one
    imag = np.exp(x) * np.sin(y)
    return np.asarray(real + 1j * imag, dtype=np.complex128)


def B_complex(z: ComplexArray) -> ComplexArray:
    """Bernoulli function B(z) = z / (exp(z) - 1) for complex z [1].

    Branched on the real part exactly as ddsim/physics/bernoulli.py branches on
    x, so that neither side overflows. For Re(z) > 0 the identity

        B(z) = -z*exp(-z) / expm1(-z)

    keeps the exponential's argument negative and its magnitude in (0, 1].

    Deliberately has no series branch near the origin. The series would make
    the reference agree with the implementation by sharing its approximation,
    which is the one thing a reference must not do. The closed form is what
    the reference is for.
    """
    values = np.asarray(z, dtype=np.complex128)
    out = np.empty_like(values)

    # B has a removable singularity at the origin and the closed form is 0/0
    # there. The limit is exactly 1, so it is filled in rather than computed.
    # This is the exact value and not a series, so the reference still shares
    # no approximation with the implementation. Only an exactly zero argument
    # qualifies, which a differentiation step never produces.
    origin = values == 0.0
    out[origin] = 1.0

    positive = (values.real > 0.0) & ~origin
    negative = ~positive & ~origin

    if negative.any():
        w = values[negative]
        out[negative] = w / complex_expm1(w)
    if positive.any():
        w = values[positive]
        out[positive] = -w * np.exp(-w) / complex_expm1(-w)

    return out


def complex_step_jacobian(
    residual: Callable[[ComplexArray], ComplexArray],
    x: npt.NDArray[np.float64],
    step: float = DEFAULT_STEP,
) -> npt.NDArray[np.float64]:
    """Dense Jacobian of residual at x, one complex step per column.

    Args:
        residual: F(x), evaluated on a complex array and returning one. Must
            preserve the dtype; see the module docstring.
        x: the point to differentiate at [1].
        step: imaginary step [1].

    Returns a dense (m, n) array, so this is for small meshes only. A 20 node
    coupled system is 60 unknowns and 60 residual evaluations, which is the
    size phases/PHASE-3.md asks the verification to run at.
    """
    base = np.asarray(x, dtype=np.complex128)
    n = base.size

    first = residual(base.copy())
    if not np.iscomplexobj(first):
        raise TypeError(
            "the residual returned a real array for a complex input, so it "
            "discards the imaginary part and the complex step Jacobian would "
            f"be exactly zero. Got dtype {np.asarray(first).dtype}."
        )

    m = np.asarray(first).size
    jacobian = np.empty((m, n), dtype=np.float64)

    for column in range(n):
        perturbed = base.copy()
        perturbed[column] += 1j * step
        jacobian[:, column] = np.asarray(residual(perturbed)).imag / step

    return jacobian
