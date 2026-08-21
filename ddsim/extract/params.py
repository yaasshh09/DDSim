"""Device parameters read off a computed curve. Post processing, no solving.

Phase 2 needs two of them, both from the forward I-V characteristic. Threshold
voltage, subthreshold swing and DIBL join them when there is a MOSFET to take
them from.

Ideality factor
---------------
    n = (1 / V_T) * dV / d(ln I)

n = 1 when the current is minority carrier diffusion into the quasi-neutral
regions, because the injected density goes as exp(V/V_T). n = 2 when it is
recombination inside the depletion region, because there n = p = n_i
exp(V/2V_T) at the peak of the rate. A real diode is a sum of the two, so the
local ideality starts near 2 at low bias and falls toward 1 as the diffusion
term takes over.

That crossover is an output, never an input. docs/01-physics.md lists it among
the things that must emerge, and phases/PHASE-2.md says the same in stronger
words. Nothing in this file, or anywhere upstream of it, contains a 2.

Saturation current
------------------
Fit ln I = ln I_s + V / (n V_T) over a chosen window and report both. The
window matters and is deliberately not defaulted to the whole curve: a diode is
only Shockley-like where one mechanism dominates, and below about four V_T the
-1 in the diode equation bends ln I away from a straight line.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ddsim.core import constants as C


def _checked(
    voltage: npt.NDArray[np.float64], current: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Both arrays as float64, with the shape and sign checks done once.

    The current is called J rather than I throughout this file. It is a current
    density in A/cm^2, which is what a 1D solve produces, and a bare I is hard
    to tell from a 1 or an l in a monospace font.
    """
    V = np.asarray(voltage, dtype=np.float64)
    J = np.asarray(current, dtype=np.float64)

    if V.shape != J.shape:
        raise ValueError(
            f"voltage and current must be the same length, got {V.shape} and {J.shape}"
        )
    if V.size < 2:
        raise ValueError("at least two points are needed to take a slope")
    if np.any(J <= 0.0):
        raise ValueError(
            "every current must be positive to take its logarithm. Trim the "
            "reverse biased end of the sweep before extracting."
        )
    if np.any(np.diff(V) <= 0.0):
        raise ValueError("voltage must be strictly increasing")

    return V, J


def ideality_factor(
    voltage: npt.NDArray[np.float64],
    current: npt.NDArray[np.float64],
    T: float = C.T_ROOM,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Local ideality factor between consecutive points [1].

    Args:
        voltage: applied bias [V], strictly increasing.
        current: terminal current [A/cm^2], all positive.
        T: temperature [K], for V_T.

    Returns (midpoint voltages [V], ideality [1]), one entry shorter than the
    input, since each value belongs between a pair of points rather than at
    one of them.
    """
    V, J = _checked(voltage, current)
    midpoint = 0.5 * (V[:-1] + V[1:])
    return midpoint, np.diff(V) / (C.V_T(T) * np.diff(np.log(J)))


def saturation_current(
    voltage: npt.NDArray[np.float64],
    current: npt.NDArray[np.float64],
    window: tuple[float, float] | None = None,
    ideality: float | None = None,
    T: float = C.T_ROOM,
) -> tuple[float, float]:
    """Fit I = I_s * (exp(V / (n V_T)) - 1), returning (I_s [A/cm^2], n [1]).

    Args:
        voltage: applied bias [V], strictly increasing.
        current: terminal current [A/cm^2], all positive.
        window: (low, high) bias range to use [V]. The whole curve if None,
            which is rarely what you want.
        ideality: hold n at this value instead of fitting it. See below.
        T: temperature [K].

    With n fitted, this is a straight line least squares fit to ln I against V,
    and both numbers come out of the same fit.

    **I_s is an extrapolation and n is a slope, and they are not equally
    trustworthy.** Reading I_s off a fit means running the line back from the
    fitting window to zero bias, a distance of V/V_T in the exponent, which is
    about 37 at 0.95 V. A half percent error in the slope therefore becomes a
    20 percent error in I_s. Measured on a synthetic curve with one percent of
    a second mechanism mixed in: n comes back within 0.6 percent and I_s lands
    25 percent high.

    So when theory fixes the ideality, say it. Passing ideality=1.0 measures
    I_s directly as the average of I / (exp(V/V_T) - 1) across the window and
    reports back the n it was told, with no extrapolation and no amplification.
    That is the right way to compare against an analytic saturation current
    built from lifetimes and diffusion lengths.
    """
    V, J = _checked(voltage, current)

    if window is not None:
        inside = (V >= window[0]) & (V <= window[1])
        if int(np.count_nonzero(inside)) < 2:
            raise ValueError(
                f"the window {window} holds fewer than two points of a sweep "
                f"running {V[0]:+g} to {V[-1]:+g} V"
            )
        V, J = V[inside], J[inside]

    if ideality is not None:
        return float(np.mean(J / np.expm1(V / (ideality * C.V_T(T))))), ideality

    slope, intercept = np.polyfit(V, np.log(J), 1)
    return float(np.exp(intercept)), float(1.0 / (slope * C.V_T(T)))
