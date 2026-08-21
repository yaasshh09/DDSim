"""Mesh refinement of the diode I-V, Tier 2 support for the I_s claim.

An agreement between a solve and a closed form is only worth quoting if the
solved number has stopped moving under refinement. Otherwise it is a
discretization error that happens to land near the analytic value, and it will
walk away from it the moment the mesh changes.

Measured across a factor of eight in node count:

    101 nodes, h_min 1 nm    I_s = 1.33256e-10 A/cm^2
    201 nodes, h_min 5 A     I_s = 1.33220e-10
    401 nodes, h_min 2 A     I_s = 1.33201e-10
    801 nodes, h_min 1 A     I_s = 1.33194e-10

so the answer is converged to 5e-4 relative by 201 nodes, which is the mesh the
rest of the Phase 2 tests use. The analytic expression it is compared against
is itself only defined to about 2 percent, because the quasi-neutral width
depends on where in the fitting window the depletion edge is taken.
"""

from __future__ import annotations

import pytest

from ddsim.device.pn_diode import pn_diode
from ddsim.extract.iv import iv_sweep
from ddsim.extract.params import saturation_current

MICRON = 1e-4
"""One micron [cm]."""

WINDOW = (0.4, 0.5)
"""Bias range the saturation current is measured over [V]."""


def measured_saturation_current(n_nodes: int, h_min: float) -> float:
    """I_s [A/cm^2] from a forward sweep on a mesh of the given resolution."""
    device = pn_diode(
        Na=1e16,
        Nd=1e16,
        length=12 * MICRON,
        junction=6 * MICRON,
        n_nodes=n_nodes,
        h_min=h_min,
    )
    voltages = [round(0.05 * step, 3) for step in range(1, 13)]
    curve = iv_sweep(device, "anode", voltages, step=0.05)
    assert curve.complete, curve.message

    value, _ = saturation_current(
        curve.voltage, curve.current, window=WINDOW, ideality=1.0
    )
    return value


@pytest.fixture(scope="module")
def refinement() -> list[float]:
    """I_s on three meshes, coarse to fine."""
    return [
        measured_saturation_current(101, 1e-6),
        measured_saturation_current(201, 5e-7),
        measured_saturation_current(401, 2e-7),
    ]


def test_the_saturation_current_stops_moving_under_refinement(
    refinement: list[float],
) -> None:
    """Doubling the node count twice must not move I_s by 0.1 percent.

    This is what licenses quoting the agreement with the analytic value. A
    number still drifting at the first refinement is a coincidence, not a
    measurement.
    """
    coarse, medium, fine = refinement

    assert abs(medium - coarse) / coarse < 1e-3
    assert abs(fine - medium) / medium < 1e-3


def test_the_refinement_converges_rather_than_wandering(
    refinement: list[float],
) -> None:
    """Successive changes shrink, which is what convergence means.

    A sequence that moves by the same amount at every refinement is not
    converging, it is tracking something that scales with h.
    """
    coarse, medium, fine = refinement

    assert abs(fine - medium) < abs(medium - coarse)


def test_the_working_mesh_is_already_converged(refinement: list[float]) -> None:
    """201 nodes is what the rest of the Phase 2 tests run on.

    If the working mesh were not converged, every number in those tests would
    carry a discretization error that nothing else in the suite would notice.
    """
    _, medium, fine = refinement

    assert abs(medium - fine) / fine < 5e-4
