"""Generates the MOS C-V plot named in the Phase 4 definition of done.

Headless matplotlib, per CLAUDE.md. The figure goes into docs/images so the
README can point at it, and it is produced by a test rather than a script so
that it cannot drift away from the code that makes it.

Both curves come from the same DC solves. The low frequency one lets every
carrier follow the small signal, which is exact for this formulation; the high
frequency one holds the minority carrier still, which is the standard model of
a signal faster than minority carrier generation. Everything annotated on the
figure is a closed form with no fitted quantity in it, and every one of them is
asserted here before the figure is drawn, so a plot that looks right cannot be
produced by a solver that is not.
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
    NA,
    T_OX,
    T_SI,
    flatband_voltage,
    max_depletion_width,
    oxide_capacitance,
    threshold_voltage,
)
from tests.analytic.test_mos_cv import debye_length, in_series  # noqa: E402

OUTPUT = pathlib.Path(__file__).parents[2] / "docs" / "images"

METAL = C.PHI_M_N_POLY
V_FB = flatband_voltage(-NA, METAL)
V_TH = threshold_voltage(-NA, T_OX, METAL)

C_OX = oxide_capacitance(T_OX)
C_FB = in_series(C_OX, C.eps_Si() / debye_length(-NA))
C_MIN = in_series(C_OX, C.eps_Si() / max_depletion_width(-NA))

NANO = 1e9
"""F to nF. An area capacitance of a 10 nm oxide is 345 nF/cm^2, so nano is
the prefix that keeps the axis readable."""


@pytest.fixture(scope="module")
def curves():
    """Both responses over the same bias range."""
    device = mos_cap(substrate_doping=-NA, t_ox=T_OX, t_si=T_SI)
    voltages = [
        round(V_FB - 2.5 + 0.125 * index, 4) for index in range(41)
    ]
    return {
        response: cv_sweep(device, GATE, voltages, response=response)
        for response in Response
    }


def test_both_sweeps_finish(curves):
    for response, curve in curves.items():
        assert curve.complete, f"{response.value}: {curve.message}"


def test_the_annotated_numbers_are_the_ones_the_plot_will_show(curves):
    """Everything the figure claims, asserted before it is drawn.

    A plot is not evidence. These four are, and they are the four
    phases/PHASE-4.md gates the C-V curve on.
    """
    low = curves[Response.LOW_FREQUENCY]
    high = curves[Response.HIGH_FREQUENCY]

    accumulation = float(low.capacitance[0])
    assert accumulation == pytest.approx(C_OX, rel=0.02)

    at_flatband = float(np.interp(V_FB, low.gate_voltage, low.capacitance))
    assert at_flatband == pytest.approx(C_FB, rel=0.02)

    at_threshold = float(np.interp(V_TH, high.gate_voltage, high.capacitance))
    assert at_threshold == pytest.approx(C_MIN, rel=0.05)

    inversion = float(low.capacitance[-1])
    assert inversion == pytest.approx(C_OX, rel=0.05)
    assert float(high.capacitance[-1]) < 0.2 * C_OX


def test_mos_cv_plot_is_generated(curves):
    """The Phase 4 deliverable: C-V with the three regimes annotated."""
    low = curves[Response.LOW_FREQUENCY]
    high = curves[Response.HIGH_FREQUENCY]

    figure, axis = plt.subplots(figsize=(8.0, 5.5))

    axis.plot(
        low.gate_voltage,
        low.capacitance * NANO,
        label="low frequency, every carrier follows",
        linewidth=1.8,
    )
    axis.plot(
        high.gate_voltage,
        high.capacitance * NANO,
        "--",
        label="high frequency, minority carrier held",
        linewidth=1.8,
    )

    ceiling = C_OX * NANO
    left = float(low.gate_voltage[0])

    for value, text in (
        (C_OX, r"$C_{ox} = \varepsilon_{ox}/t_{ox}$"),
        (C_FB, r"$C_{FB} = C_{ox} \parallel \varepsilon_{Si}/L_D$"),
        (C_MIN, r"$C_{min} = C_{ox} \parallel \varepsilon_{Si}/W_{max}$"),
    ):
        axis.axhline(value * NANO, color="grey", linestyle=":", linewidth=0.9)
        axis.annotate(
            f"{text} = {value * NANO:.1f}",
            xy=(left, value * NANO),
            xytext=(left + 0.08, value * NANO - 0.06 * ceiling),
            fontsize=8.5,
            color="dimgrey",
        )

    for bias, text in ((V_FB, f"$V_{{FB}}$ = {V_FB:.3f} V"),
                       (V_TH, f"$V_{{TH}}$ = {V_TH:.3f} V")):
        axis.axvline(bias, color="grey", linestyle="-.", linewidth=0.8)
        axis.annotate(
            text,
            xy=(bias, 0.32 * ceiling),
            xytext=(bias - 0.11, 0.32 * ceiling),
            fontsize=8.5,
            rotation=90,
            va="bottom",
            ha="right",
        )

    for centre, label in (
        (V_FB - 1.3, "accumulation"),
        (0.5 * (V_FB + V_TH), "depletion"),
        (V_TH + 0.9, "inversion"),
    ):
        axis.annotate(
            label,
            xy=(centre, ceiling * 1.10),
            ha="center",
            fontsize=11,
            color="tab:blue",
        )

    axis.set_ylim(0.0, ceiling * 1.20)
    axis.set_xlim(low.gate_voltage[0], low.gate_voltage[-1])
    axis.set_xlabel("gate bias [V]")
    axis.set_ylabel("capacitance [nF/cm$^2$]")
    axis.set_title(
        f"MOS capacitor C-V, {abs(NA):.0e} cm$^{{-3}}$ p-type, "
        f"{T_OX * 1e7:.0f} nm oxide, n+ poly gate"
    )
    axis.legend(loc="center left", fontsize=9)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "mos_cap_cv.png"
    figure.tight_layout()
    figure.savefig(target, dpi=140)
    plt.close(figure)

    assert target.exists()
    assert target.stat().st_size > 10_000
