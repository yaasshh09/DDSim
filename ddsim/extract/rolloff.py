"""The gate length sweep: one process, many gate lengths, and what falls out.

This is the deliverable of phases/PHASE-5.md. Threshold roll-off, drain induced
barrier lowering and velocity saturation are not modelled anywhere in this
codebase, and there is no term in any equation that knows what a short channel
is. They appear here because a two dimensional Poisson solve on a device whose
two junctions are close together gives a different answer from one where they
are far apart, and because a carrier in a high field does not go faster when
the field is raised.

One process, several gate lengths
---------------------------------
Everything except `L_gate` is held fixed across the sweep: the oxide, the
channel doping, the implant depth, the lateral encroachment, the contacts. That
is what roll-off means. A process is fixed once, on a wafer, and gate length is
the number a designer draws differently from one transistor to the next.
Scaling the oxide and the junction depth alongside the gate would also produce
a curve, but it would be a curve of four things moving together, and no part of
it could be attributed to the channel getting shorter.

`SHORT_CHANNEL_PROCESS` is therefore a process built for the short end of the
sweep, a 2 nm oxide over a 1e18 channel with 25 nm junctions, and the long
devices are that same process drawn long. At 1 um it has no short channel
effect left in it, which is exactly what makes it the reference the short
devices are measured against.

What the constant current threshold is measured at
--------------------------------------------------
Id = I_ref * W / L, the usual 100 nA * W / L. The target moves with the gate
length on purpose: a shorter device drives proportionally more current at the
same gate overdrive, so a fixed target would read the geometry as a threshold
shift and hand back roll-off that was put there by the extraction. Dividing by
L is what removes the trivial part and leaves the part that is a real barrier
change.

Where the numbers stop meaning anything
---------------------------------------
Drift-diffusion assumes the local field sets the local velocity. Below roughly
50 nm that is not true: carriers cross the channel in less time than it takes
them to reach the steady velocity of the field they are in, so real devices
overshoot and this model cannot. Nothing below 50 nm is reported here for that
reason, and the limit is a property of the equations rather than of this file.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ddsim.device.mosfet import nmos
from ddsim.extract.iv import IVCurve, gate_sweep
from ddsim.extract.params import (
    dibl,
    saturation_exponent,
    subthreshold_slope,
    threshold_constant_current,
    threshold_linear_extrapolation,
    transconductance,
)

SHORT_CHANNEL_PROCESS: Mapping[str, Any] = {
    "substrate_doping": -1e18,
    "sd_peak": 1e20,
    "x_j": 2.5e-6,
    "lateral_diffusion": 1.0e-6,
    "t_ox": 2e-7,
    "sd_length": 4e-5,
    "contact_length": 2e-5,
    "t_si": 1e-4,
}
"""The vertical process every gate length in the sweep is built on.

A 2 nm oxide over a 1e18 cm^-3 channel, with source and drain 25 nm deep and
reaching 10 nm under each gate edge. The channel doping is what holds the
50 nm device out of punch-through, and the 10 nm encroachment is what leaves
30 nm of metallurgical channel under a 50 nm gate.

Nothing in here was chosen to produce a threshold voltage. It is a mapping of
`nmos` arguments and nothing else, so a reader can see the entire device
without reading any code.
"""

REFERENCE_CURRENT = 1e-7
"""The numerator of the Id = I_ref * W / L threshold criterion [A].

100 nA, the usual statement. With the current already per unit width and every
length in cm, the target at a gate length L is REFERENCE_CURRENT / L [A/cm].
"""


@dataclass(frozen=True)
class RollOffPoint:
    """One gate length, and everything read off its two transfer curves."""

    L_gate: float
    """Gate length this device was drawn at [cm]."""

    threshold_linear: float
    """Constant current threshold at the low drain bias [V]."""

    threshold_saturated: float
    """Constant current threshold at the high drain bias [V]."""

    threshold_extrapolated: float
    """Threshold by tangent at peak transconductance, low drain bias [V].

    A second opinion rather than a better one. The two methods measure
    different things and a device where they disagree about the direction of
    the roll-off is telling you the extraction is wrong, not the device.
    """

    subthreshold_slope: float
    """Steepest part of the low drain curve [mV/decade]."""

    dibl: float
    """Threshold shift per volt of drain [mV/V]."""

    saturation_exponent: float
    """The power the high drain current follows the overdrive to.

    2 for a long channel square law, falling toward 1 as velocity saturates.

    Overdrive here is measured from `threshold_extrapolated` and not from
    either constant current threshold, and that is not a free choice. The
    square law is a statement about strong inversion, where the tangent at
    peak transconductance is what defines the threshold. A constant current
    threshold sits several decades lower, in the knee where the curve is still
    leaving its subthreshold exponential, so overdrives measured from it are
    too large by a couple of hundred millivolts and the fitted power comes back
    near 3.5 on a device that is a clean square law.
    """

    peak_transconductance: float
    """Largest dId/dVg on the low drain curve [A/(cm V)]."""

    linear: IVCurve
    """The transfer curve at the low drain bias."""

    saturated: IVCurve
    """The transfer curve at the high drain bias."""


def usable_span(
    voltage: npt.NDArray[np.float64], current: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The longest tail of a transfer curve that is positive and rising.

    The extractors take a logarithm, so they need a curve that is on. The head
    of a real transfer curve is not: below a few hundred millivolts of gate the
    drain current of a long channel device is reverse junction leakage, which
    is small, negative and flat, and the same gate list that reaches deep
    subthreshold on a 50 nm device is well into that floor on a 1 um one.

    Trimming rather than refusing is what lets one gate list serve every length
    in a sweep, which is the point of the sweep being config driven.
    """
    V, J = np.asarray(voltage, dtype=np.float64), np.asarray(
        current, dtype=np.float64
    )
    start = 0
    for k in range(len(J)):
        if J[k] <= 0.0:
            start = k + 1
        elif k > 0 and J[k] <= J[k - 1]:
            start = k

    if len(J) - start < 3:
        raise ValueError(
            f"only {len(J) - start} points of this transfer curve are on: it "
            f"runs {V[0]:+g} to {V[-1]:+g} V and never becomes both positive "
            "and rising for long enough to extract from. Extend the gate "
            "sweep upward."
        )
    return V[start:], J[start:]


def gate_length_sweep(
    gate_lengths: list[float],
    gate_voltages: list[float],
    process: Mapping[str, Any] | None = None,
    drain_low: float = 0.05,
    drain_high: float = 1.0,
    reference_current: float = REFERENCE_CURRENT,
    overdrive_window: tuple[float, float] = (0.4, 1.0),
    step: float = 0.05,
) -> tuple[RollOffPoint, ...]:
    """Build one MOSFET per gate length and extract the short channel set.

    Args:
        gate_lengths: the gate lengths to draw [cm], in the order to report
            them. 1e-4 is 1 um and 5e-6 is 50 nm.
        gate_voltages: the gate biases to solve at [V], increasing. One list
            for every length: the head of it is trimmed per device by
            `usable_span`, since the off state of a 50 nm device sits well below
            the off state of a 1 um one.
        process: `nmos` arguments held fixed across the sweep.
            `SHORT_CHANNEL_PROCESS` if None. It must not carry `L_gate`,
            `drain_voltage` or `gate_voltage`, which the sweep sets.
        drain_low: drain bias of the linear curve [V], the one thresholds and
            the subthreshold slope are read off.
        drain_high: drain bias of the saturated curve [V], the one DIBL and the
            saturation exponent are read off.
        reference_current: numerator of the Id = I_ref * W / L criterion [A].
        overdrive_window: the gate overdrive range to fit the saturation
            exponent over [V], measured above `threshold_extrapolated`. It
            starts above zero deliberately: just above threshold a transfer
            curve is still leaving the subthreshold exponential, which is not
            a power law in overdrive at all and would drag the fit.
        step: first continuation step between gate biases [V].

    Returns one `RollOffPoint` per gate length, in the order requested.

    Two curves per device, because DIBL is a difference between them and one
    curve cannot say anything about it. Both are kept on the result.
    """
    if not gate_lengths:
        raise ValueError("a gate length sweep needs at least one gate length")

    if drain_high == drain_low:
        raise ValueError(
            f"the sweep needs two different drain biases to report DIBL, and "
            f"both are {drain_low:g} V"
        )

    settings = dict(SHORT_CHANNEL_PROCESS if process is None else process)

    points = []
    for L_gate in gate_lengths:
        curves = {}
        for label, drain in (("linear", drain_low), ("saturated", drain_high)):
            device = nmos(L_gate=L_gate, drain_voltage=drain, **settings)
            curves[label] = gate_sweep(
                device, voltages=list(gate_voltages), step=step
            )

        V_lin, J_lin = usable_span(
            curves["linear"].voltage, curves["linear"].current
        )
        V_sat, J_sat = usable_span(
            curves["saturated"].voltage, curves["saturated"].current
        )
        target = reference_current / L_gate

        threshold_linear = threshold_constant_current(V_lin, J_lin, target)
        threshold_saturated = threshold_constant_current(V_sat, J_sat, target)
        threshold_extrapolated = threshold_linear_extrapolation(
            V_lin, J_lin, drain_voltage=drain_low
        )
        _, gm = transconductance(V_lin, J_lin)

        points.append(
            RollOffPoint(
                L_gate=L_gate,
                threshold_linear=threshold_linear,
                threshold_saturated=threshold_saturated,
                threshold_extrapolated=threshold_extrapolated,
                subthreshold_slope=subthreshold_slope(V_lin, J_lin),
                dibl=dibl(
                    threshold_low=threshold_linear,
                    threshold_high=threshold_saturated,
                    drain_low=drain_low,
                    drain_high=drain_high,
                ),
                saturation_exponent=saturation_exponent(
                    V_sat,
                    J_sat,
                    threshold=threshold_extrapolated,
                    window=(
                        threshold_extrapolated + overdrive_window[0],
                        threshold_extrapolated + overdrive_window[1],
                    ),
                ),
                peak_transconductance=float(np.max(gm)),
                linear=curves["linear"],
                saturated=curves["saturated"],
            )
        )

    return tuple(points)
