"""Tier 2 analytic tests: a PN diode at equilibrium against closed form results.

From docs/04-validation.md. Each of these is a full solve compared against a
textbook expression, so they check the physics rather than the plumbing.

A note on what is and is not emergent here. The built-in potential is largely
imposed: ohmic contacts pin psi at both ends to asinh(N/2), so their difference
is V_bi by construction, and the test below mostly validates the boundary
condition. The genuinely emergent results in this file are the depletion width,
which comes out of solving Poisson through the junction, and the Debye decay of
the potential into the neutral bulk. Those are the ones to watch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ddsim.core import constants as C
from ddsim.device.builder import build_device
from ddsim.device.doping import Step
from ddsim.device.equilibrium import frozen_quasi_fermi, solve_equilibrium
from ddsim.device.pn_diode import pn_diode
from ddsim.discretize.boundary import OhmicContact
from ddsim.mesh.mesh1d import graded_mesh_1d

MICRON = 1e-4
"""One micron [cm]."""


def analytic_V_bi(Na: float, Nd: float, T: float = 300.0) -> float:
    """V_bi = V_T * ln(Na * Nd / n_i^2) [V], docs/04-validation.md."""
    return C.V_T(T) * math.log(Na * Nd / C.n_i(T) ** 2)


def analytic_depletion_width(Na: float, Nd: float, bias: float = 0.0) -> float:
    """W = sqrt(2*eps*(V_bi - V)/q * (1/Na + 1/Nd)) [cm]."""
    V_bi = analytic_V_bi(Na, Nd)
    return math.sqrt(
        2.0 * C.eps_Si() * (V_bi - bias) / C.q * (1.0 / Na + 1.0 / Nd)
    )


def long_diode(Na: float, Nd: float, bias: float = 0.0, length: float = 12.0 * MICRON):
    """A diode long enough that the depletion region does not reach a contact.

    At 1e16 / 1e16 the depletion region is 0.43 um wide at equilibrium and
    grows as sqrt(V_bi - V), so a 1 um device has no neutral bulk left at
    -5 V. Everything here uses a device with room.
    """
    return pn_diode(
        Na=Na,
        Nd=Nd,
        length=length,
        junction=0.5 * length,
        n_nodes=801,
        h_min=5e-8,
        anode_voltage=bias,
    )


# --------------------------------------------------------- built in potential


@pytest.mark.parametrize(
    ("Na", "Nd"),
    [(1e15, 1e15), (1e16, 1e16), (1e17, 1e17), (1e16, 1e18), (1e15, 1e17)],
)
def test_built_in_potential_matches_the_analytic_form(Na: float, Nd: float) -> None:
    """Under 0.5 percent, per phases/PHASE-1.md and docs/04-validation.md."""
    device = long_diode(Na, Nd)
    state = solve_equilibrium(device)

    psi = state.psi.to_physical(device.scale).data
    simulated = psi[-1] - psi[0]  # [V]
    expected = analytic_V_bi(Na, Nd)

    assert simulated == pytest.approx(expected, rel=5e-3)


def test_built_in_potential_is_not_the_value_quoted_in_the_docs() -> None:
    """phases/PHASE-1.md and docs/06-constants.md used to say expect 0.695 V
    for a 1e16 / 1e16 junction. That figure is the n_i = 1.45e10 answer and
    predates this project's decision to use 1.0e10.

    With n_i = 1.0e10 the same formula gives 0.7143 V. The two differ by
    2.8 percent, which is more than five times the 0.5 percent tolerance the
    phase asks for, so both cannot be satisfied. The formula wins, because it
    is the definition and it is consistent with the n_i in use.

    Both documents now carry 0.7143 V. This test stays anyway, because the
    number it rejects is the one every other reference for silicon prints, and
    a 19 mV offset in V_bi reads exactly like a boundary condition sign error.
    See the deviations table in docs/07-decisions.md.
    """
    device = long_diode(1e16, 1e16)
    state = solve_equilibrium(device)
    psi = state.psi.to_physical(device.scale).data
    simulated = psi[-1] - psi[0]

    assert simulated == pytest.approx(0.7143, abs=1e-3)
    assert abs(simulated - 0.695) / 0.695 > 0.02


def test_built_in_potential_grows_with_doping() -> None:
    potentials = []
    for doping in (1e15, 1e16, 1e17):
        device = long_diode(doping, doping)
        state = solve_equilibrium(device)
        psi = state.psi.to_physical(device.scale).data
        potentials.append(psi[-1] - psi[0])

    assert all(a < b for a, b in zip(potentials[:-1], potentials[1:], strict=True))


def test_built_in_potential_rises_by_two_v_t_per_decade_of_doping() -> None:
    """A decade on each side is ln(100) = 4.6 V_T. Emergent from the formula."""
    low = long_diode(1e15, 1e15)
    high = long_diode(1e16, 1e16)

    psi_low = solve_equilibrium(low).psi.to_physical(low.scale).data
    psi_high = solve_equilibrium(high).psi.to_physical(high.scale).data

    step = (psi_high[-1] - psi_high[0]) - (psi_low[-1] - psi_low[0])
    assert step == pytest.approx(C.V_T() * math.log(100.0), rel=0.02)


# ------------------------------------------------------------ depletion width


def depletion_edges(device, state) -> tuple[float, float]:
    """Extract the two depletion edges from the simulated field [cm].

    The depletion approximation predicts a triangular field: linear on each
    side, peaking at the junction, zero at the edges. So the edges come from
    fitting the linear part on each side and extrapolating to zero.

    A fixed threshold on the field does not work, and it is worth saying why.
    The real field does not stop at the depletion edge, it decays
    exponentially over the local Debye length. That tail is 10 times longer on
    a 1e15 side than on a 1e17 side, so any single threshold measures
    different things on the two sides. Measured on a symmetric 1e16 junction,
    thresholds from 1 to 50 percent of peak give widths between 0.21 and
    0.62 um against a true 0.43 um, while the linear extrapolation lands
    within 1.2 percent.

    The fitting window is the 20 to 80 percent band of the peak field, which
    is inside the triangle and outside both the rounded apex and the tails.
    """
    psi = state.psi.to_physical(device.scale).data  # [V]
    field = np.abs(-np.diff(psi) / device.mesh.h)  # [V/cm] on edges
    centres = 0.5 * (device.mesh.x[:-1] + device.mesh.x[1:])

    peak_value = field.max()
    peak_position = centres[int(np.argmax(field))]

    edges = []
    for side in (-1.0, 1.0):
        in_band = (field > 0.2 * peak_value) & (field < 0.8 * peak_value)
        window = in_band & (np.sign(centres - peak_position) == side)
        slope, intercept = np.polyfit(centres[window], field[window], 1)
        edges.append(float(-intercept / slope))

    return edges[0], edges[1]


def depletion_width_from_field(device, state) -> float:
    """Total depletion width [cm]."""
    left, right = depletion_edges(device, state)
    return right - left


@pytest.mark.parametrize("bias", [0.0, -1.0, -5.0])
def test_depletion_width_matches_the_depletion_approximation(bias: float) -> None:
    """Under 3 percent at 0, -1 and -5 V, per phases/PHASE-1.md.

    This one is genuinely emergent. Nothing in the code knows the depletion
    approximation; the width falls out of solving Poisson across the junction.
    """
    Na = Nd = 1e16
    device = long_diode(Na, Nd, bias=bias)
    state = solve_equilibrium(device, frozen_quasi_fermi(device))

    simulated = depletion_width_from_field(device, state)
    expected = analytic_depletion_width(Na, Nd, bias)

    assert simulated == pytest.approx(expected, rel=3e-2)


def test_depletion_width_grows_as_the_square_root_of_reverse_bias() -> None:
    """W scales as sqrt(V_bi - V). The shape matters more than any tolerance.

    Stops at -5 V, which is what the phase asks for. Going further is possible
    but the step limiter caps psi at 5 V_T per Newton step, so a 10 V shift
    needs 46 iterations against a budget of 50. Bias continuation in Phase 3
    is the proper fix, not a bigger budget.
    """
    Na = Nd = 1e16
    biases = (0.0, -1.0, -3.0, -5.0)
    widths = []
    for bias in biases:
        device = long_diode(Na, Nd, bias=bias)
        state = solve_equilibrium(device, frozen_quasi_fermi(device))
        widths.append(depletion_width_from_field(device, state))

    V_bi = analytic_V_bi(Na, Nd)
    predicted = [widths[0] * math.sqrt((V_bi - bias) / V_bi) for bias in biases]
    for simulated, expected in zip(widths, predicted, strict=True):
        assert simulated == pytest.approx(expected, rel=3e-2)


def test_depletion_region_sits_mostly_on_the_lightly_doped_side() -> None:
    """x_p / x_n = Nd / Na, so a one sided junction depletes into the light side.

    Nothing in the code knows this. It comes out of Poisson.

    Checked at a 10 to 1 doping ratio rather than something more dramatic,
    because the depletion approximation stops being a fair comparison once the
    depleted width on the heavy side falls below the local Debye length. That
    side is then entirely smeared and has no abrupt edge to find. Measured:
    at 10 to 1 the depleted width on the heavy side is 2.2 Debye lengths and
    the recovered ratio is 7.3 against a predicted 10, while at 100 to 1 it is
    0.74 Debye lengths and the ratio collapses to 29 against a predicted 100.
    The 30 percent tolerance below is what the approximation can actually
    deliver here, not slack for a bug.
    """
    Na, Nd = 1e15, 1e16
    device = long_diode(Na, Nd)
    state = solve_equilibrium(device)

    junction = 0.5 * device.mesh.length
    left, right = depletion_edges(device, state)
    p_side = junction - left
    n_side = right - junction

    assert p_side > n_side, "the light side must take most of the depletion"
    assert p_side / n_side == pytest.approx(Nd / Na, rel=0.30)


# ---------------------------------------------------------------- Debye decay


def test_potential_decays_into_the_bulk_with_the_local_debye_length() -> None:
    """docs/04-validation.md, under 1 percent on the fitted decay length.

    A doping step with no change of type gives a small potential difference
    that relaxes over the Debye length rather than a depletion region. Fitting
    the exponential tail recovers L_D, which is the cleanest available check
    that the length scaling in the assembly is right.
    """
    low, high = 1e16, 2e16
    length = 4.0 * MICRON
    step_position = 0.5 * length

    mesh = graded_mesh_1d(length, 1201, refine_at=step_position, h_min=2e-8)
    device = build_device(
        mesh=mesh,
        doping=Step(left=low, right=high, position=step_position),
        contacts=(
            OhmicContact("left", 0, 0.0),
            OhmicContact("right", mesh.n_nodes - 1, 0.0),
        ),
    )
    state = solve_equilibrium(device)
    psi = state.psi.to_physical(device.scale).data  # [V]

    # Fit the decay on the lightly doped side, away from both the step and the
    # contact, where the deviation from bulk is a clean exponential.
    psi_bulk = psi[0]
    L_D = math.sqrt(C.eps_Si() * C.V_T() / (C.q * low))  # [cm]

    x = device.mesh.x
    window = (x > step_position - 8.0 * L_D) & (x < step_position - 2.0 * L_D)
    deviation = np.abs(psi[window] - psi_bulk)

    slope, _ = np.polyfit(x[window], np.log(deviation), 1)
    fitted = 1.0 / slope  # [cm]

    assert fitted == pytest.approx(L_D, rel=1e-2)


def test_debye_length_scales_with_the_local_doping() -> None:
    """Doubling the doping shortens the decay by sqrt(2). Emergent."""
    length = 4.0 * MICRON
    step_position = 0.5 * length
    fitted_lengths = []

    for low in (1e16, 4e16):
        mesh = graded_mesh_1d(length, 1201, refine_at=step_position, h_min=2e-8)
        device = build_device(
            mesh=mesh,
            doping=Step(left=low, right=2.0 * low, position=step_position),
            contacts=(
                OhmicContact("left", 0, 0.0),
                OhmicContact("right", mesh.n_nodes - 1, 0.0),
            ),
        )
        state = solve_equilibrium(device)
        psi = state.psi.to_physical(device.scale).data

        L_D = math.sqrt(C.eps_Si() * C.V_T() / (C.q * low))
        x = device.mesh.x
        window = (x > step_position - 8.0 * L_D) & (x < step_position - 2.0 * L_D)
        deviation = np.abs(psi[window] - psi[0])
        slope, _ = np.polyfit(x[window], np.log(deviation), 1)
        fitted_lengths.append(1.0 / slope)

    assert fitted_lengths[0] / fitted_lengths[1] == pytest.approx(2.0, rel=0.02)
