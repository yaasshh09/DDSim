"""Generates the diode I-V plot named in the Phase 2 definition of done.

Headless matplotlib. The figure goes into docs/images so the README can point
at it, and it is produced by a test rather than a script so that it cannot
drift away from the code that makes it.

Two devices, one figure. The 1e16 diode is diffusion limited and has an
ideality of 1 across the whole useful range. The 1e18 diode has enough
depletion region recombination to show the n = 2 region and the crossover out
of it. Nothing changes between them except the doping.
"""

from __future__ import annotations

import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ddsim.device.pn_diode import pn_diode  # noqa: E402
from ddsim.extract.iv import iv_sweep  # noqa: E402
from ddsim.extract.params import ideality_factor, saturation_current  # noqa: E402

OUTPUT = pathlib.Path(__file__).parents[2] / "docs" / "images"

MICRON = 1e-4
"""One micron [cm]."""


def sweep(doping: float, n_nodes: int, step: float, top: float):
    """A forward sweep from `step` to `top` volts on a symmetric diode."""
    device = pn_diode(
        Na=doping,
        Nd=doping,
        length=12 * MICRON,
        junction=6 * MICRON,
        n_nodes=n_nodes,
        h_min=5e-7 if doping <= 1e16 else 2e-7,
    )
    count = int(round(top / step))
    voltages = [round(step * index, 4) for index in range(1, count + 1)]
    curve = iv_sweep(device, "anode", voltages, step=step)
    assert curve.complete, curve.message
    return curve


def reverse_sweep(doping: float = 1e16, n_nodes: int = 201):
    """The reverse branch, walked down from zero."""
    device = pn_diode(
        Na=doping,
        Nd=doping,
        length=12 * MICRON,
        junction=6 * MICRON,
        n_nodes=n_nodes,
        h_min=5e-7,
    )
    voltages = [round(-0.1 * index, 3) for index in range(1, 11)]
    curve = iv_sweep(device, "anode", voltages, step=0.1)
    assert curve.complete, curve.message
    return curve


def test_iv_plot_is_generated() -> None:
    """Log scale I-V for two diodes, with the extracted ideality annotated."""
    diffusion = sweep(1e16, 201, 0.025, 0.6)
    recombination = sweep(1e18, 301, 0.025, 0.6)
    reverse = reverse_sweep()

    I_s, _ = saturation_current(
        diffusion.voltage, diffusion.current, window=(0.4, 0.5), ideality=1.0
    )
    low_bias, low_ideality = ideality_factor(
        recombination.voltage, recombination.current
    )
    peak = float(low_ideality.max())
    peak_at = float(low_bias[low_ideality.argmax()])

    figure, axes = plt.subplots(2, 1, figsize=(7.5, 8), sharex=True)

    axes[0].semilogy(
        diffusion.voltage, diffusion.current, label="1e16 / 1e16, forward"
    )
    axes[0].semilogy(
        recombination.voltage, recombination.current, label="1e18 / 1e18, forward"
    )
    axes[0].semilogy(
        reverse.voltage,
        np.abs(reverse.current),
        "--",
        linewidth=1.0,
        color="tab:green",
        label="1e16 / 1e16, reverse |I|",
    )
    axes[0].axhline(I_s, color="grey", linestyle=":", linewidth=0.9)
    axes[0].annotate(
        f"$I_s$ = {I_s:.3g} A/cm$^2$\nanalytic short base 1.30e-10",
        xy=(0.02, I_s),
        xytext=(-1.0, I_s * 3.0),
        fontsize=8,
    )
    axes[0].set_xlim(-1.05, 0.65)
    axes[0].set_ylabel("|current| [A/cm$^2$]")
    axes[0].set_title("PN diode I-V, 12 um, SRH with Scharfetter lifetimes")
    axes[0].legend(loc="upper left", fontsize=8)

    for curve, label in (
        (diffusion, "1e16 / 1e16"),
        (recombination, "1e18 / 1e18"),
    ):
        midpoint, ideality = ideality_factor(curve.voltage, curve.current)
        axes[1].plot(midpoint, ideality, label=label)

    axes[1].axhline(2.0, color="grey", linestyle=":", linewidth=0.9)
    axes[1].axhline(1.0, color="grey", linestyle=":", linewidth=0.9)
    axes[1].annotate(
        f"peak n = {peak:.2f} at {peak_at:.2f} V",
        xy=(peak_at, peak),
        xytext=(peak_at + 0.08, peak + 0.08),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        fontsize=8,
    )
    axes[1].set_ylim(0.75, 2.15)
    axes[1].annotate(
        "the -1 term, not a second mechanism",
        xy=(-0.15, 0.82),
        fontsize=7,
        color="grey",
    )
    axes[1].set_ylabel("ideality factor $n$")
    axes[1].set_xlabel("anode bias [V]")
    axes[1].legend(loc="center left", fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.5)

    figure.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "pn_diode_iv.png"
    figure.savefig(target, dpi=140)
    plt.close(figure)

    assert target.exists()
    assert target.stat().st_size > 10_000

    # The plot has to show real physics, not a flat line.
    assert diffusion.current[-1] / diffusion.current[0] > 1e6
    assert peak > 1.7
    assert abs(I_s - 1.30e-10) / 1.30e-10 < 0.10
