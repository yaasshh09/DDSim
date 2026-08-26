"""Generates the MOS C-V plot named in the Phase 4 definition of done.

    A C-V curve from your own solver overlaid on DEVSIM's, committed to the
    README, with the three regimes annotated.

Headless matplotlib. The figure goes into docs/images so the README can point
at it, and it is produced by a test rather than a script so that it cannot
drift away from the code that makes it.

The device is benchmark 4 of docs/04-validation.md, the 5 nm capacitor, chosen
because that is one of the two stacks DEVSIM golden data exists for. A thin
oxide also puts the weight of the comparison on the semiconductor charge
rather than on the parallel plate, which is the half of the problem that is
actually hard.

What is compared, and what is only drawn
----------------------------------------
Three curves, and they are not all the same kind of thing.

**ddsim, low frequency.** The exact derivative of the solved system, taken
through the DC Jacobian. Every carrier follows the small signal, which is what
this formulation gives and what DEVSIM's equilibrium solve means.

**ddsim, high frequency.** The same solve with the minority carrier held
still, the standard model of a signal faster than minority carrier
generation. Drawn because a C-V curve without it is only half the picture, and
compared against nothing, because DEVSIM was not asked for it.

**DEVSIM.** A central difference on the golden gate charge. DEVSIM has no
exact derivative path here, so the comparison is made by applying the same
central difference to ddsim's charge and comparing those, which keeps the
question about the physics rather than about the differentiation. That number
is asserted below and printed on the figure. The circles themselves are drawn
at DEVSIM's central difference, so the eye is comparing a difference quotient
against an exact derivative and the small gap near threshold is the truncation
error of the operator rather than a disagreement between the codes.

The bias range runs wider than the golden data on purpose. Accumulation
approaches C_ox slowly, because the accumulation layer has a finite thickness,
and cutting the axis at -2 V would show a curve that never reaches the line it
is drawn against. See docs/07-decisions.md.

Everything annotated on the figure is a closed form with no fitted quantity in
it, and every one of them is asserted here before the figure is drawn, so a
plot that looks right cannot be produced by a solver that is not.
"""

from __future__ import annotations

import pathlib

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ddsim.core import constants as C  # noqa: E402
from ddsim.device.mos_cap import GATE, mos_cap  # noqa: E402
from ddsim.extract.cv import Response, cv_sweep  # noqa: E402
from tests.analytic.test_mos_cap import (  # noqa: E402
    flatband_voltage,
    max_depletion_width,
    oxide_capacitance,
    threshold_voltage,
)
from tests.analytic.test_mos_cv import debye_length, in_series  # noqa: E402
from tests.regression.devsim_gen import parameters as P  # noqa: E402

OUTPUT = pathlib.Path(__file__).parents[2] / "docs" / "images"
GOLDEN_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "golden"

BENCHMARK = P.MOS_BENCHMARKS[0]
"""Benchmark 4, the 5 nm capacitor. The one with golden data and a thin oxide."""

NA = -BENCHMARK.substrate_doping
T_OX = BENCHMARK.t_ox
T_SI = BENCHMARK.t_si
METAL = BENCHMARK.work_function

V_FB = flatband_voltage(-NA, METAL)
V_TH = threshold_voltage(-NA, T_OX, METAL)

C_OX = oxide_capacitance(T_OX)
C_FB = in_series(C_OX, C.eps_Si() / debye_length(-NA))
C_MIN = in_series(C_OX, C.eps_Si() / max_depletion_width(-NA))

NANO = 1e9
"""F to nF. A 5 nm oxide is 691 nF/cm^2, so nano keeps the axis readable."""

LOWEST = -3.5
"""Most negative gate bias to solve [V], about V_FB - 2.6.

Far enough into accumulation that the curve is within 2 percent of C_ox. It
does not get there any sooner: at V_FB - 1 V it is still 5 percent short, and
that is the accumulation layer having a thickness rather than a defect.
"""

STEP = 0.1
"""Bias step [V]. The golden data's own step, so its points are a subset."""


def bias_points() -> list[float]:
    """The sweep, from LOWEST up to the top of the golden range."""
    top = max(BENCHMARK.voltages)
    count = int(round((top - LOWEST) / STEP)) + 1
    return [round(LOWEST + STEP * index, 4) for index in range(count)]


@pytest.fixture(scope="module")
def device():
    """The benchmark stack, built once."""
    return mos_cap(
        substrate_doping=BENCHMARK.substrate_doping,
        t_ox=BENCHMARK.t_ox,
        t_si=BENCHMARK.t_si,
        n_silicon=BENCHMARK.n_silicon,
        n_oxide=BENCHMARK.n_oxide,
        h_min=BENCHMARK.h_min,
        work_function=BENCHMARK.work_function,
    )


@pytest.fixture(scope="module")
def curves(device):
    """Both responses over the same bias range, solved once."""
    voltages = bias_points()
    return {
        response: cv_sweep(device, GATE, voltages, response=response)
        for response in Response
    }


@pytest.fixture(scope="module")
def landmarks(device):
    """The capacitance solved at exactly V_FB and V_TH, not interpolated.

    The curve turns hardest between those two biases, so reading it off the
    0.1 V plotting grid by linear interpolation overshoots C_FB by 2.6
    percent, which is the chord of an arc and not anything the solver did.
    tests/analytic/test_mos_cv.py holds the same solve to one part in a
    thousand at flatband, which is what the number is actually worth.
    """
    return {
        response: cv_sweep(device, GATE, [V_FB, V_TH], response=response)
        for response in Response
    }


@pytest.fixture(scope="module")
def golden():
    """DEVSIM's charge, differenced into a capacitance."""
    curve = P.read_mos_golden(str(GOLDEN_DIR / f"{BENCHMARK.name}.csv"))
    voltage, capacitance = P.central_difference(curve.gate_voltage, curve.charge)
    return np.asarray(voltage), np.asarray(capacitance)


def ddsim_differenced(curves):
    """ddsim's own charge through the same operator, on the golden biases.

    Like for like. Comparing an exact derivative against a difference quotient
    would fold the truncation error of the quotient into the disagreement, and
    that error belongs to the operator rather than to either code.
    """
    low = curves[Response.LOW_FREQUENCY]
    inside = np.isin(np.round(low.gate_voltage, 4), BENCHMARK.voltages)

    voltage, capacitance = P.central_difference(
        list(low.gate_voltage[inside]), list(low.charge[inside])
    )
    return np.asarray(voltage), np.asarray(capacitance)


def test_both_sweeps_finish(curves):
    for response, curve in curves.items():
        assert curve.complete, f"{response.value}: {curve.message}"


def test_the_sweep_covers_every_golden_bias(curves):
    """The overlay is only honest if both codes were asked the same question.

    Interpolating ddsim onto DEVSIM's grid would hide a solver that stalled
    somewhere in the middle of the sweep.
    """
    solved = np.round(curves[Response.LOW_FREQUENCY].gate_voltage, 4)

    missing = sorted(set(BENCHMARK.voltages) - set(solved.tolist()))
    assert not missing, f"ddsim never reached {missing}"


def test_the_annotated_numbers_are_the_ones_the_plot_will_show(curves, landmarks):
    """Everything the figure claims, asserted before it is drawn.

    A plot is not evidence. These four are, and they are the four
    phases/PHASE-4.md gates the C-V curve on.
    """
    low = curves[Response.LOW_FREQUENCY]
    high = curves[Response.HIGH_FREQUENCY]

    accumulation = float(low.capacitance[0])
    assert accumulation == pytest.approx(C_OX, rel=0.02)

    at_flatband = float(landmarks[Response.LOW_FREQUENCY].capacitance[0])
    assert at_flatband == pytest.approx(C_FB, rel=1e-3)

    at_threshold = float(landmarks[Response.HIGH_FREQUENCY].capacitance[1])
    assert at_threshold == pytest.approx(C_MIN, rel=0.05)

    inversion = float(low.capacitance[-1])
    assert inversion == pytest.approx(C_OX, rel=0.05)
    assert float(high.capacitance[-1]) < 0.2 * C_OX


def test_the_overlaid_curves_agree(curves, golden):
    """The claim the figure makes, asserted at the benchmark's own tolerance.

    Same operator on both sides, so what is left is the physics. This is the
    same comparison tests/regression/test_devsim_mos.py runs and it is
    repeated here for a reason: this file draws a picture, and a picture of
    two curves lying on top of each other has to be backed by a number in the
    same place it was produced.
    """
    golden_voltage, golden_capacitance = golden
    ddsim_voltage, ddsim_capacitance = ddsim_differenced(curves)

    np.testing.assert_allclose(ddsim_voltage, golden_voltage, atol=1e-9)

    worst = float(
        np.max(
            np.abs(ddsim_capacitance - golden_capacitance)
            / np.abs(golden_capacitance)
        )
    )
    assert worst < BENCHMARK.tolerance, (
        f"worst disagreement {worst:.3%}, allowed {BENCHMARK.tolerance:.3%}"
    )


def test_mos_cv_plot_is_generated(curves, golden):
    """The Phase 4 deliverable: ddsim over DEVSIM, three regimes annotated."""
    low = curves[Response.LOW_FREQUENCY]
    high = curves[Response.HIGH_FREQUENCY]
    golden_voltage, golden_capacitance = golden
    _, ddsim_capacitance = ddsim_differenced(curves)

    worst = float(
        np.max(
            np.abs(ddsim_capacitance - golden_capacitance)
            / np.abs(golden_capacitance)
        )
    )

    figure, axis = plt.subplots(figsize=(8.6, 5.6))

    axis.plot(
        golden_voltage,
        golden_capacitance * NANO,
        "o",
        markersize=5.5,
        markerfacecolor="none",
        markeredgewidth=1.1,
        color="tab:orange",
        label="DEVSIM 2.11, central difference on the golden charge",
        zorder=3,
    )
    axis.plot(
        low.gate_voltage,
        low.capacitance * NANO,
        label="ddsim, low frequency, every carrier follows",
        linewidth=1.8,
        color="tab:blue",
        zorder=4,
    )
    axis.plot(
        high.gate_voltage,
        high.capacitance * NANO,
        "--",
        label="ddsim, high frequency, minority carrier held",
        linewidth=1.8,
        color="tab:blue",
        alpha=0.65,
        zorder=4,
    )

    ceiling = C_OX * NANO
    left = float(low.gate_voltage[0])

    # The label sits under its line, except for the lowest one, which would
    # fall off the bottom of the axes.
    for value, text, drop in (
        (C_OX, r"$C_{ox} = \varepsilon_{ox}/t_{ox}$", 0.055),
        (C_FB, r"$C_{FB} = C_{ox} \parallel \varepsilon_{Si}/L_D$", 0.055),
        (C_MIN, r"$C_{min} = C_{ox} \parallel \varepsilon_{Si}/W_{max}$", -0.022),
    ):
        axis.axhline(value * NANO, color="grey", linestyle=":", linewidth=0.9)
        axis.annotate(
            f"{text} = {value * NANO:.1f}",
            xy=(left, value * NANO),
            xytext=(left + 0.08, value * NANO - drop * ceiling),
            fontsize=8.5,
            color="dimgrey",
        )

    for bias, text in (
        (V_FB, f"$V_{{FB}}$ = {V_FB:.3f} V"),
        (V_TH, f"$V_{{TH}}$ = {V_TH:.3f} V"),
    ):
        axis.axvline(bias, color="grey", linestyle="-.", linewidth=0.8)
        axis.annotate(
            text,
            xy=(bias, 0.62 * ceiling),
            xytext=(bias - 0.05, 0.62 * ceiling),
            fontsize=8.5,
            rotation=90,
            va="bottom",
            ha="right",
        )

    for centre, label in (
        (0.5 * (left + V_FB), "accumulation"),
        (0.5 * (V_FB + V_TH), "depletion"),
        (0.5 * (V_TH + float(low.gate_voltage[-1])), "inversion"),
    ):
        axis.annotate(
            label,
            xy=(centre, ceiling * 1.10),
            ha="center",
            fontsize=11,
            color="tab:blue",
        )

    axis.annotate(
        f"worst disagreement with DEVSIM: {worst:.3%}\n"
        "(same central difference applied to both)",
        xy=(0.985, 0.28),
        xycoords="axes fraction",
        ha="right",
        fontsize=8.5,
        color="dimgrey",
    )

    axis.set_ylim(0.0, ceiling * 1.20)
    axis.set_xlim(low.gate_voltage[0], low.gate_voltage[-1])
    axis.set_xlabel("gate bias [V]")
    axis.set_ylabel("capacitance [nF/cm$^2$]")
    axis.set_title(
        f"MOS capacitor C-V, {NA:.0e} cm$^{{-3}}$ p-type, "
        f"{T_OX * 1e7:.0f} nm oxide, n+ poly gate"
    )
    # Underneath, outside the axes. There is no empty corner left inside: the
    # curve runs from C_ox to C_min and back, the three closed forms have
    # their own lines across the full width, and a legend that overlaps any of
    # them hides the thing the figure exists to show.
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=1,
        fontsize=9,
        frameon=False,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "mos_cap_cv.png"
    figure.savefig(target, dpi=140, bbox_inches="tight")
    plt.close(figure)

    assert target.exists()
    assert target.stat().st_size > 10_000
