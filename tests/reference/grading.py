"""Scalar reference for the graded mesh ratio solve.

`ddsim.mesh.mesh1d` solves the growth ratio for every candidate split of the
node budget at once, because solving them one at a time spends more time in the
interpreter than in the arithmetic. Vectorised bisection is easy to get subtly
wrong: an element that keeps iterating after it should have stopped, or a mask
applied one step late, moves the answer in the last few bits and nothing about
the mesh looks wrong afterwards.

So the obvious version lives here. One ratio, one loop, no masks, written the
way the formula reads. It is the oracle the vectorised solver is tested
against, and the two are required to agree to the last bit rather than to a
tolerance, since they are meant to be the same bisection.

This is the same role tests/reference/highprec.py plays for the Bernoulli
function: a slow implementation whose only job is to be obviously right.
"""

from __future__ import annotations

import math

RATIO_TOLERANCE = 1e-14
"""Relative tolerance for the geometric ratio solve [1]."""

DEGENERATE_TOLERANCE = 1e-12
"""Below this relative difference, a side is treated as exactly uniform [1]."""

LOG_MAX_DOUBLE = 700.0
"""ln of a number comfortably below the largest double, about 1.8e308 [1]."""


def geometric_sum(h_min: float, ratio: float, n_intervals: int) -> float:
    """Total length of n_intervals spacings growing geometrically [cm].

    Saturates to infinity rather than overflowing. The bracketing search below
    starts at ratio 2 and doubles, and with a thousand cells 2^1000 is far
    past the double range. The bisection only ever asks whether the sum
    exceeds the side length, so infinity is a perfectly usable answer.
    """
    if ratio == 1.0:
        return h_min * n_intervals
    if n_intervals * math.log(ratio) > LOG_MAX_DOUBLE:
        return math.inf
    return h_min * (ratio**n_intervals - 1.0) / (ratio - 1.0)


def solve_ratio(side_length: float, h_min: float, n_intervals: int) -> float | None:
    """Growth ratio r such that h_min * (r^m - 1)/(r - 1) = side_length.

    Returns None if the side cannot be covered, which happens when even the
    minimum spacing repeated m times overshoots the available length.
    """
    if n_intervals <= 0:
        return None
    if h_min > side_length * (1.0 + DEGENERATE_TOLERANCE):
        return None

    uniform_total = h_min * n_intervals
    if uniform_total > side_length * (1.0 + DEGENERATE_TOLERANCE):
        return None
    if n_intervals == 1:
        return 1.0
    if abs(uniform_total - side_length) <= DEGENERATE_TOLERANCE * side_length:
        return 1.0

    # Bracket first. The sum grows monotonically with r, so doubling the upper
    # bound until it overshoots is enough.
    low, high = 1.0, 2.0
    while geometric_sum(h_min, high, n_intervals) < side_length:
        high *= 2.0
        if high > 1e6:
            return None

    for _ in range(200):
        middle = 0.5 * (low + high)
        if geometric_sum(h_min, middle, n_intervals) < side_length:
            low = middle
        else:
            high = middle
        if high - low <= RATIO_TOLERANCE * low:
            break

    return 0.5 * (low + high)
