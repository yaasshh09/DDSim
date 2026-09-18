"""The benchmark devices drawn from rectangles reproduce the constructors.

phases/PHASE-7.md Stage 5: a 2D drawing of the benchmark MOS capacitor and of
the benchmark nmos reproduces mos_cap C-V and nmos Id-Vg, to a recorded
tolerance. The tolerance and the reason it is not bit for bit are in
docs/07-decisions.md, 2026-09-18.

The doping is the same function, tests/unit/test_drawing.py checks that. The
mesh is not: a drawing grades one axis through every edge drawn, where the
constructors lay hand built segments. So the difference measured here is a
difference of meshes, and the argument for the tolerance is that it sits at
or under what each constructor moves by when its own mesh is refined.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.device.drawing import MOS_CAP_DRAWING, drawing
from ddsim.device.mos_cap import mos_cap
from ddsim.device.mosfet import nmos
from ddsim.device.transport import TransportModels
from ddsim.extract.cv import cv_sweep
from ddsim.extract.iv import gate_sweep

CV_BIASES = [round(-2.0 + 0.25 * k, 10) for k in range(17)]
"""Accumulation through inversion [V]."""

GATES = [round(0.1 * k, 10) for k in range(13)]
"""0 to 1.2 V [V], at 50 mV on the drain."""

CV_TOLERANCE = 2e-4
"""Relative [1]. Measured 2026-09-18: 8.7e-5 at worst, against 1.8e-3 for
mos_cap refined to n_silicon=241, n_oxide=9, h_min=2.5e-8."""

ID_TOLERANCE = 0.015
"""Relative, above the floor [1]. Measured 2026-09-18: 1.01e-2 at worst, at
0.6 V, against up to 1.6e-2 for nmos refined to h_min_y=3.125e-9 with
n_oxide=65 over the same points."""

ID_FLOOR = 1e-7
"""Below this drain current nothing is compared [A/cm]. The drain terminal
has a floor near 2e-9 A/cm, see tests/regression/test_devsim_mosfet.py, and
at 0 and 0.1 V nmos and nmos refined read inside it with opposite signs. At
1e-7 the floor is two percent of the current."""


def test_the_drawn_mos_capacitor_reproduces_mos_cap_c_v() -> None:
    """At mos_cap's own node counts and surface spacing."""
    blocks, implants, electrodes = MOS_CAP_DRAWING
    drawn = drawing(
        blocks, implants, electrodes, nx=3, ny=125, h_min_y=5e-8, degenerate=False
    )
    reference = cv_sweep(mos_cap(), "gate", CV_BIASES)
    got = cv_sweep(drawn, "gate", CV_BIASES)
    assert reference.complete and got.complete
    np.testing.assert_allclose(
        got.capacitance, reference.capacitance, rtol=CV_TOLERANCE
    )


@pytest.fixture(scope="module")
def transfer_curves() -> tuple[np.ndarray, np.ndarray]:
    """nmos and the default drawing, each at 50 mV of drain, on the models
    TransportModels.for_device defaults to, which is what the page sends."""
    currents = []
    for device in (nmos(drain_voltage=0.05), drawing().with_bias(drain=0.05)):
        curve = gate_sweep(device, GATES, models=TransportModels.for_device(device))
        assert curve.complete, curve.message
        currents.append(np.asarray(curve.current))
    return currents[0], currents[1]


def test_the_drawn_nmos_reproduces_nmos_id_vg_above_the_floor(
    transfer_curves,
) -> None:
    reference, got = transfer_curves
    above = np.abs(reference) >= ID_FLOOR
    assert above.sum() >= 9, "the comparison has to cover the curve above threshold"
    np.testing.assert_allclose(got[above], reference[above], rtol=ID_TOLERANCE)


def test_the_drawn_nmos_is_off_where_nmos_is_off(transfer_curves) -> None:
    """Under the floor the two are not compared to each other, but a drawing
    that conducted where nmos does not would be a different device."""
    reference, got = transfer_curves
    below = np.abs(reference) < ID_FLOOR
    assert below.any()
    assert np.all(np.abs(got[below]) < ID_FLOOR)
