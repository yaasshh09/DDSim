"""Generates the PN diode band diagram named in the Phase 1 definition of done.

Headless matplotlib. The figure is written into docs/images so the README can
point at it. It is produced by a test rather than a script so that it cannot
drift away from the code that makes it.
"""

from __future__ import annotations

import math
import pathlib

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ddsim.core import constants as C  # noqa: E402
from ddsim.device.equilibrium import solve_equilibrium  # noqa: E402
from ddsim.device.pn_diode import pn_diode  # noqa: E402

OUTPUT = pathlib.Path(__file__).parents[2] / "docs" / "images"


def test_band_diagram_is_generated() -> None:
    """A 1e16 / 1e16 diode at equilibrium: bands, carriers and field."""
    device = pn_diode(Na=1e16, Nd=1e16, length=4e-4, junction=2e-4, n_nodes=801)
    state = solve_equilibrium(device)

    x = device.mesh.x * 1e4  # [um]
    psi = state.psi.to_physical(device.scale).data  # [V]
    n = state.n.to_physical(device.scale).data  # [cm^-3]
    p = state.p.to_physical(device.scale).data  # [cm^-3]

    # Band edges relative to the intrinsic level. E_c - E_i is half the gap.
    half_gap = 0.5 * C.Eg()  # [eV]
    E_i = -psi
    E_c = E_i + half_gap
    E_v = E_i - half_gap
    E_f = np.zeros_like(x)

    field = -np.diff(psi) / device.mesh.h  # [V/cm]
    centres = 0.5 * (x[:-1] + x[1:])

    figure, axes = plt.subplots(3, 1, figsize=(7.5, 9), sharex=True)

    axes[0].plot(x, E_c, label="$E_c$")
    axes[0].plot(x, E_v, label="$E_v$")
    axes[0].plot(x, E_i, "--", linewidth=0.9, label="$E_i$")
    axes[0].plot(x, E_f, ":", linewidth=1.2, label="$E_F$")
    axes[0].set_ylabel("energy [eV]")
    axes[0].legend(loc="center right", fontsize=8)
    V_bi = psi[-1] - psi[0]
    axes[0].set_title(
        f"PN diode at equilibrium, 1e16 / 1e16, $V_{{bi}}$ = {V_bi:.4f} V"
    )

    axes[1].semilogy(x, n, label="$n$")
    axes[1].semilogy(x, p, label="$p$")
    axes[1].axhline(C.n_i(), color="grey", linestyle=":", linewidth=0.9)
    axes[1].set_ylabel("density [cm$^{-3}$]")
    axes[1].set_ylim(1e2, 1e18)
    axes[1].legend(loc="center right", fontsize=8)

    axes[2].plot(centres, field * 1e-3)
    axes[2].set_ylabel("field [kV/cm]")
    axes[2].set_xlabel("position [um]")

    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.5)

    figure.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "pn_diode_equilibrium.png"
    figure.savefig(target, dpi=140)
    plt.close(figure)

    assert target.exists()
    assert target.stat().st_size > 10_000

    # The plot must show real physics, not a flat line.
    assert V_bi == pytest.approx(
        C.V_T() * math.log(1e16 * 1e16 / C.n_i() ** 2), rel=5e-3
    )
    assert n.max() / n.min() > 1e10
