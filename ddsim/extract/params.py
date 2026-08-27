"""Device parameters read off a computed curve. Post processing, no solving.

Phase 2 needs two of them, both from the forward I-V characteristic. Phase 5
adds the MOSFET set: threshold voltage by two methods, subthreshold slope,
transconductance and DIBL.

Nothing here solves anything, which is the point. phases/PHASE-5.md grades the
phase on these numbers, so they are built and tested against closed form curves
first, in tests/unit/test_params.py. An extractor validated only against solver
output cannot say whether the extractor or the solver is the thing that is
wrong, and on the day a device disagrees with DEVSIM that is the question.

Two ways to say threshold, on purpose
-------------------------------------
Neither is more correct than the other; they measure different things and a
device where they disagree is telling you something. Constant current picks the
bias at a chosen current and is what a roll-off plot usually reports, because it
survives a curve whose shape is changing with gate length. Linear extrapolation
runs the tangent at peak transconductance back to zero current, which is closer
to the textbook definition and is more sensitive to series resistance and to
mobility degradation. Reporting both is how phases/PHASE-5.md asks for it.

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
    voltage: npt.NDArray[np.float64],
    current: npt.NDArray[np.float64],
    positive: bool = True,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Both arrays as float64, with the shape and sign checks done once.

    Args:
        voltage: applied bias [V], strictly increasing.
        current: terminal current, in whatever unit the caller is working in.
        positive: require every current to be strictly positive. True wherever
            a logarithm is taken. The linear region extractors take none, and
            an Id-Vg curve that is exactly zero below threshold is an ordinary
            thing to hand them rather than an error.

    The current is called J rather than I throughout this file. It is a current
    density in A/cm^2, which is what a 1D solve produces, and a bare I is hard
    to tell from a 1 or an l in a monospace font. A 2D MOSFET solve produces a
    current per unit width in A/cm instead; nothing here cares which, only that
    a current and anything it is compared against carry the same unit.
    """
    V = np.asarray(voltage, dtype=np.float64)
    J = np.asarray(current, dtype=np.float64)

    if V.shape != J.shape:
        raise ValueError(
            f"voltage and current must be the same length, got {V.shape} and {J.shape}"
        )
    if V.size < 2:
        raise ValueError("at least two points are needed to take a slope")
    if positive and np.any(J <= 0.0):
        raise ValueError(
            "every current must be positive to take its logarithm. Trim the "
            "reverse biased end of the sweep before extracting."
        )
    if np.any(np.diff(V) <= 0.0):
        raise ValueError("voltage must be strictly increasing")

    return V, J


def _rising(J: npt.NDArray[np.float64], what: str) -> None:
    """Refuse a current that does not increase with the gate.

    Every extractor below reads a slope of the gate characteristic and reports
    one number off it. On a curve that doubles back, the steepest local slope
    and the peak transconductance both stop meaning what their names say, and
    the number would come back plausible rather than wrong. A subthreshold
    sweep that is not monotonic is a solver problem to look at, not an input to
    push through.
    """
    if np.any(np.diff(J) <= 0.0):
        raise ValueError(
            f"{what} needs a current that rises with the gate bias, and this "
            "one does not everywhere. A non monotonic Id-Vg is worth looking "
            "at rather than extracting from."
        )


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


# ------------------------------------------------------------ MOSFET parameters


def subthreshold_slope(
    voltage: npt.NDArray[np.float64],
    current: npt.NDArray[np.float64],
    window: tuple[float, float] | None = None,
) -> float:
    """The steepest part of the gate characteristic [mV/decade].

    Args:
        voltage: gate bias [V], strictly increasing.
        current: drain current [A/cm], all positive and rising.
        window: (low, high) gate range to search [V]. The whole curve if None.

    SS = dVg / d(log10 Id), reported as the minimum over the window.

    **The minimum, because that is the one the thermal limit binds.** A real
    curve has a different slope at every bias, steepest deep in subthreshold
    and flattening as it approaches threshold, so "the" subthreshold slope has
    to be a choice. At 300 K no thermally activated current can beat
    kT/q * ln(10) = 59.5 mV/decade: the gate cannot move a barrier by more than
    the bias applied to it, and the population over that barrier is Boltzmann.
    A device that reports less than 59.5 has a bug, not a feature, and taking
    the minimum is what makes that check able to fail.

    Nothing here contains a 59.5. The limit is a property of the physics being
    modelled, and this function measures a slope.
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

    _rising(J, "the subthreshold slope")
    return float(1e3 * np.min(np.diff(V) / np.diff(np.log10(J))))


def transconductance(
    voltage: npt.NDArray[np.float64],
    current: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """dId/dVg between consecutive points [A/(cm V)].

    Args:
        voltage: gate bias [V], strictly increasing.
        current: drain current [A/cm]. Zeros are allowed: no logarithm here.

    Returns (midpoint gate biases [V], gm), one entry shorter than the input,
    since each value belongs between a pair of points rather than at one of
    them. Same shape and same convention as ideality_factor above.
    """
    V, J = _checked(voltage, current, positive=False)
    return 0.5 * (V[:-1] + V[1:]), np.diff(J) / np.diff(V)


def threshold_constant_current(
    voltage: npt.NDArray[np.float64],
    current: npt.NDArray[np.float64],
    target: float,
    width: float = 1.0,
) -> float:
    """Gate bias at which the drain current crosses a fixed value [V].

    Args:
        voltage: gate bias [V], strictly increasing.
        current: drain current, all positive and rising.
        target: the current per unit width to cross, in the units `current`
            carries once divided by `width`. The usual statement is 100 nA/um.
        width: device width to normalise by, in the length unit `current` is
            already per. Defaults to 1.0, for a current that arrives
            normalised already.

    Interpolated in log current against gate bias rather than linearly.
    Subthreshold current is exponential in the gate bias, so a straight line
    through two points of Id would sit well above the curve between them, and
    the error is largest exactly where this method is usually applied.
    """
    V, J = _checked(voltage, current)
    _rising(J, "a constant current threshold")

    normalised = J / width
    if not normalised[0] <= target <= normalised[-1]:
        raise ValueError(
            f"the sweep never reaches {target:g}: it runs {normalised[0]:g} to "
            f"{normalised[-1]:g} over {V[0]:+g} to {V[-1]:+g} V. Extend the "
            "sweep or choose a target inside it."
        )

    return float(np.interp(np.log10(target), np.log10(normalised), V))


def threshold_linear_extrapolation(
    voltage: npt.NDArray[np.float64],
    current: npt.NDArray[np.float64],
    drain_voltage: float | None = None,
) -> float:
    """Threshold by the tangent at peak transconductance [V].

    Args:
        voltage: gate bias [V], strictly increasing.
        current: drain current [A/cm]. Zeros below threshold are fine.
        drain_voltage: drain bias the curve was taken at [V]. Applies the
            -Vd/2 correction. None reports the raw intercept and says so.

    Find the steepest point of Id against Vg, run its tangent back to zero
    current, and read off the bias.

    **The -Vd/2 is not a fudge.** In the linear region
    Id = k*(Vg - Vth - Vd/2)*Vd exactly, so the tangent to it meets zero at
    Vg = Vth + Vd/2 and never at Vth. Dropping the correction leaves a number
    that still looks like a threshold and is wrong by half the drain bias,
    which is 25 mV at the 0.05 V a roll-off plot is usually taken at, and 25 mV
    is the same size as the roll-off being measured. Passing the drain bias in
    is the only way this function can apply it, because a curve does not carry
    the bias it was taken at.
    """
    V, J = _checked(voltage, current, positive=False)
    midpoint, gm = transconductance(V, J)

    peak = int(np.argmax(gm))
    if gm[peak] <= 0.0:
        raise ValueError(
            "the current never rises with the gate bias, so there is no "
            "tangent to extrapolate. Check the sign of the sweep."
        )

    # Midpoint values, so the point and the slope belong to the same interval.
    # On a straight segment the average of the two currents is exactly the
    # value at the midpoint, which is what makes this exact in the linear
    # region rather than merely close.
    at_peak = 0.5 * (J[peak] + J[peak + 1])
    intercept = midpoint[peak] - at_peak / gm[peak]

    if drain_voltage is None:
        return float(intercept)
    return float(intercept - 0.5 * drain_voltage)


def dibl(
    threshold_low: float,
    threshold_high: float,
    drain_low: float,
    drain_high: float,
) -> float:
    """Drain induced barrier lowering [mV/V].

    Args:
        threshold_low: threshold measured at the low drain bias [V].
        threshold_high: threshold measured at the high drain bias [V].
        drain_low: the low drain bias [V].
        drain_high: the high drain bias [V].

    (Vth_low - Vth_high) / (Vd_high - Vd_low), in mV per V.

    Positive when raising the drain lowered the threshold, which is the
    direction the effect goes: the drain field reaches through to the source
    barrier and pulls it down, so less gate is needed to turn the device on.
    A long channel device returns zero because the drain cannot reach that far.
    The sign convention is chosen so that a bigger number means a worse short
    channel effect, which is how it is always quoted.

    Two extracted thresholds in, one number out. Both have to come from the
    same device, the same method and the same width, or this is subtracting two
    different quantities.
    """
    if drain_high == drain_low:
        raise ValueError(
            f"DIBL is a shift per volt of drain, so it needs two different "
            f"drain biases, and both are {drain_low:g} V"
        )
    return float(1e3 * (threshold_low - threshold_high) / (drain_high - drain_low))
