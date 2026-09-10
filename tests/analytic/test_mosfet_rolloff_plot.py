"""Generates the gate length sweep plot named in the Phase 5 definition of done.

    Vth versus Lg from 1 um to 50 nm. This single plot is the argument that the
    project worked. Nothing in it was fitted; it came out of Poisson plus two
    continuity equations plus a doping profile.

Headless matplotlib, into docs/images so the README can point at it, and
produced by a test rather than a script so it cannot drift away from the code
that makes it.

What is on it, and why each panel is there
------------------------------------------
Three of the four acceptance criteria of phases/PHASE-5.md are visual, and the
fourth is a number this file asserts before drawing anything.

**Left, threshold against gate length.** Two curves, the same devices at two
drain biases. The fall from left to right is roll-off. The gap between the two
is drain induced barrier lowering, and it opens as the gate shortens because
the drain reaches further into a shorter channel.

**Right, subthreshold slope and the saturation exponent.** The slope against
the 59.5 mV/decade thermal limit, which nothing thermally activated can beat at
300 K, and the exponent against the 2 of an ideal long channel square law.
The first rises off its limit as the gate loses control of the barrier; the
second falls toward 1 as carriers stop going faster when the field is raised.

The models are the full Phase 5 stack: Arora doping dependent mobility inside
Lombardi surface scattering inside Caughey-Thomas, with Fermi-Dirac statistics.
Two of those are named in the scope of phases/PHASE-5.md as not optional, and
the third is what the exponent is measuring, so a sweep taken on the Phase 2
constant mobility would be measuring something else and reporting it under this
title.

DEVSIM is not on this figure yet
--------------------------------
phases/PHASE-5.md asks for the sweep overlaid on DEVSIM, and benchmark 9 of
docs/04-validation.md is the golden data for it. Benchmarks 6 to 9 have no
golden file yet, so what is drawn here is ddsim alone and the figure says so.
Adding the overlay is what closes the phase, and drawing an unlabelled curve
in the meantime would be worse than drawing none.

Nothing here is fitted
----------------------
Every device on this plot is the same process from ddsim/extract/rolloff.py.
The oxide, the channel doping, the implant depth and the lateral encroachment
are identical across all six, and `L_gate` is the only argument that changes.
No short channel term exists anywhere in the solver to be turned on.
"""

from __future__ import annotations

import pathlib

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ddsim.extract.rolloff import (  # noqa: E402
    SHORT_CHANNEL_PROCESS,
    gate_length_sweep,
)

OUTPUT = pathlib.Path(__file__).parents[2] / "docs" / "images"

GATE_LENGTHS = [1e-4, 2e-5, 1e-5, 7e-6, 5e-6]
"""1 um down to 50 nm [cm]. The bottom is where drift-diffusion stops meaning
anything, not where the solver stops converging."""

GATE_VOLTAGES = list(np.round(np.arange(-0.5, 0.351, 0.05), 4)) + list(
    np.round(np.arange(0.4, 1.401, 0.1), 4)
)
"""Fine through subthreshold, coarse above it. The slope is read over two
decades of current, which at 70 mV/decade is 140 mV wide and needs 0.05 V
steps to hold more than one point. Above threshold the curve is a power law
and 0.1 V resolves it, and every point is a coupled 2D solve.

The fine stretch used to stop at 0.15 V, which was right while every 2D drain
current was 1/x_0 too large and the constant current threshold therefore sat
about 170 mV low. With the current fixed the thresholds moved up and the window
moved with them, leaving exactly two points inside it. Two points still measure
a slope, but a slope from two points is one difference with no averaging in it,
and the subthreshold slope is a headline number of this phase. Reaching 0.35 V
puts four or five points in the window at every gate length in the sweep."""

DRAIN_LOW = 0.05
DRAIN_HIGH = 1.0

THERMAL_LIMIT = 59.5
"""kT/q ln 10 at 300 K [mV/decade]."""

SQUARE_LAW = 2.0
"""The exponent an ideal long channel MOSFET saturates with."""


@pytest.fixture(scope="module")
def sweep():
    return gate_length_sweep(
        gate_lengths=GATE_LENGTHS,
        gate_voltages=GATE_VOLTAGES,
        drain_low=DRAIN_LOW,
        drain_high=DRAIN_HIGH,
    )


def falling(values) -> bool:
    """True when every step of the sequence goes down."""
    return bool(np.all(np.diff(np.asarray(values)) < 0.0))


def label_lengths(axis, lengths) -> None:
    """Tick at the gate lengths that were solved, not at powers of ten.

    A log axis holding six points between 1000 and 50 nm labels two of them by
    default and leaves the reader counting minor ticks to find the rest.
    """
    axis.set_xticks(lengths)
    axis.set_xticklabels([f"{value:.0f}" for value in lengths])
    axis.set_xticks([], minor=True)


# ------------------------------------------------- the acceptance criteria


def test_every_device_in_the_sweep_solved(sweep):
    """A stalled sweep still returns a curve, and a threshold extracted off a
    short one is a threshold for a device that was never solved."""
    assert len(sweep) == len(GATE_LENGTHS)
    for point in sweep:
        assert point.linear.complete
        assert point.saturated.complete


def test_the_subthreshold_slope_beats_no_thermal_limit(sweep):
    """The primary sanity gate of phases/PHASE-5.md. Below 59.5 mV/decade at
    300 K is a bug at any gate length."""
    for point in sweep:
        assert point.subthreshold_slope >= THERMAL_LIMIT


def test_the_subthreshold_slope_degrades_as_the_gate_shortens(sweep):
    """It sits on its limit while the gate owns the barrier and lifts off it
    once the drain starts sharing control. Asserted end to end rather than
    step by step: between 1 um and 200 nm this process has no short channel
    effect left to lose, so those three lengths are equal to a millivolt and
    a strict ordering there would be asserting noise."""
    slopes = [point.subthreshold_slope for point in sweep]

    assert slopes[-1] > slopes[0] + 10.0
    assert np.all(np.diff(slopes) > -0.5)


def test_the_threshold_rolls_off(sweep):
    """Both extraction methods, both drain biases. The source and drain
    depletion regions share channel charge the gate would otherwise have to
    deplete itself, so a shorter channel needs less gate. Nothing about the
    doping or the oxide changed between these six devices."""
    assert falling([point.threshold_linear for point in sweep])
    assert falling([point.threshold_saturated for point in sweep])
    assert falling([point.threshold_extrapolated for point in sweep])


def test_drain_induced_barrier_lowering_widens_the_gap(sweep):
    """Raising the drain pulls the source barrier down, so the saturated
    threshold sits below the linear one at every length, and the gap between
    them opens as the drain gets closer to the source."""
    for point in sweep:
        assert point.threshold_saturated < point.threshold_linear
        assert point.dibl > 0.0

    assert falling([-point.dibl for point in sweep])


def test_velocity_saturation_pulls_the_exponent_off_the_square_law(sweep):
    """Id goes as overdrive squared while the inversion charge and the
    velocity that carries it both rise with the gate. Once the channel field
    passes the critical field the velocity stops rising, one factor drops out,
    and the exponent falls toward 1.

    It is already below 2 at 1 um, because 1 V across a 1 um channel is around
    the critical field for electrons in silicon on its own, so this sweep never
    contains a device that is purely square law.
    """
    exponents = [point.saturation_exponent for point in sweep]

    assert exponents[0] <= SQUARE_LAW
    assert exponents[-1] < 1.2
    assert exponents[-1] > 1.0
    # What the fall is made of is measured separately, in
    # tests/analytic/test_mosfet_rolloff.py, because the exponent alone cannot
    # tell velocity saturation from the geometry it travels with.
    # End to end, and monotone within a hundredth, for the same reason the
    # subthreshold slope is asserted that way: the long devices differ from
    # each other by less than the extraction resolves.
    assert np.all(np.diff(exponents) < 0.01)


def test_the_process_did_not_change_across_the_sweep(sweep):
    """The claim the whole figure rests on. If the oxide or the doping moved
    between devices, the roll-off would be a plot of the process rather than
    of the channel length."""
    assert "L_gate" not in SHORT_CHANNEL_PROCESS
    assert "drain_voltage" not in SHORT_CHANNEL_PROCESS
    assert "gate_voltage" not in SHORT_CHANNEL_PROCESS


# ---------------------------------------------------------------- the figure


def test_mosfet_rolloff_plot_is_generated(sweep):
    lengths = np.array([point.L_gate * 1e7 for point in sweep])
    linear = np.array([point.threshold_linear for point in sweep])
    saturated = np.array([point.threshold_saturated for point in sweep])
    slopes = np.array([point.subthreshold_slope for point in sweep])
    exponents = np.array([point.saturation_exponent for point in sweep])

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.4, 5.0))

    left.fill_between(
        lengths,
        saturated,
        linear,
        color="tab:blue",
        alpha=0.12,
        label=(
            f"drain induced barrier lowering, {sweep[0].dibl:.0f} to "
            f"{sweep[-1].dibl:.0f} mV/V"
        ),
    )
    left.plot(
        lengths,
        linear,
        "o-",
        color="tab:blue",
        linewidth=1.8,
        markersize=5.5,
        label=f"$V_d$ = {DRAIN_LOW} V",
    )
    left.plot(
        lengths,
        saturated,
        "s--",
        color="tab:red",
        linewidth=1.8,
        markersize=5.0,
        label=f"$V_d$ = {DRAIN_HIGH} V",
    )
    left.annotate(
        f"{sweep[-1].dibl:.0f} mV/V",
        xy=(lengths[-1], 0.5 * (linear[-1] + saturated[-1])),
        xytext=(1.6 * lengths[-1], 0.5 * (linear[-1] + saturated[-1])),
        fontsize=9,
        color="dimgrey",
        va="center",
    )
    left.set_xscale("log")
    left.invert_xaxis()
    label_lengths(left, lengths)
    left.set_xlabel("gate length [nm]")
    left.set_ylabel("threshold voltage [V]")
    left.set_title("threshold roll-off")
    left.legend(fontsize=9, frameon=False, loc="lower left")
    left.grid(alpha=0.25, which="both")

    right.plot(
        lengths,
        slopes,
        "o-",
        color="tab:green",
        linewidth=1.8,
        markersize=5.5,
        label="subthreshold slope",
    )
    right.axhline(
        THERMAL_LIMIT,
        color="dimgrey",
        linestyle=":",
        linewidth=1.2,
    )
    right.text(
        lengths[0],
        THERMAL_LIMIT + 1.0,
        f"$kT/q \\cdot \\ln 10$ = {THERMAL_LIMIT} mV/decade",
        fontsize=8.5,
        color="dimgrey",
        va="bottom",
    )
    right.set_xscale("log")
    right.invert_xaxis()
    label_lengths(right, lengths)
    right.set_xlabel("gate length [nm]")
    right.set_ylabel("subthreshold slope [mV/decade]", color="tab:green")
    right.tick_params(axis="y", labelcolor="tab:green")
    right.set_ylim(THERMAL_LIMIT - 4.0, max(slopes.max() + 6.0, 95.0))
    right.set_title("gate control and velocity saturation")
    right.grid(alpha=0.25, which="both")

    twin = right.twinx()
    twin.plot(
        lengths,
        exponents,
        "^--",
        color="tab:purple",
        linewidth=1.8,
        markersize=5.5,
        label="saturation exponent",
    )
    twin.axhline(SQUARE_LAW, color="tab:purple", linestyle=":", linewidth=1.0)
    twin.text(
        lengths[-1],
        SQUARE_LAW - 0.03,
        "long channel square law",
        fontsize=8.5,
        color="tab:purple",
        ha="right",
        va="top",
    )
    twin.set_ylabel(
        r"$I_d \propto (V_g - V_{th})^{\alpha}$", color="tab:purple"
    )
    twin.tick_params(axis="y", labelcolor="tab:purple")
    twin.set_ylim(1.0, 2.15)

    figure.suptitle(
        "NMOS gate length sweep, one process: "
        f"{SHORT_CHANNEL_PROCESS['t_ox'] * 1e7:.0f} nm oxide, "
        f"{-SHORT_CHANNEL_PROCESS['substrate_doping']:.0e} cm$^{{-3}}$ "
        "channel, ddsim only, no DEVSIM overlay yet",
        fontsize=11,
    )
    figure.tight_layout()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "mosfet_rolloff.png"
    figure.savefig(target, dpi=140, bbox_inches="tight")
    plt.close(figure)

    assert target.exists()
    assert target.stat().st_size > 10_000
