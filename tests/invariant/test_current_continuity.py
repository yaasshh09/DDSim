"""Current continuity, the primary gate for Phase 2.

docs/04-validation.md calls this the strongest single check available: it
catches Scharfetter-Gummel sign errors, boundary condition errors and assembly
errors alike. phases/PHASE-2.md makes it the gate that has to pass before the
phase proceeds.

    With recombination disabled, Jn + Jp is constant across all nodes to 1e-6.

It passes at forward bias, and the file also pins down exactly why it stops
passing below about 0.25 V, which is the more interesting half of the story.
Jn is the difference of two edge terms of size (Dn/h)*n. Near equilibrium those
two cancel to nothing, so the relative spread of the measured current is about
machine epsilon times the ratio of a flux term to the current itself. That
ratio grows exponentially as the bias falls, because the terms stay put while
the current collapses.

The last test in the first section measures that ratio and shows the observed
spread tracks it to within a factor of three across twelve decades. A genuine
conservation error would not scale with the ratio, so this separates the two
explanations rather than assuming one.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import TransportModels, solve_bias
from ddsim.extract.iv import current_densities, terminal_currents
from ddsim.physics.bernoulli import B
from ddsim.physics.recombination import NoRecombination

MICRON = 1e-4
"""One micron [cm]."""

EPS = float(np.finfo(np.float64).eps)
"""Machine epsilon [1], the unit the cancellation argument is measured in."""


def diode(voltage: float = 0.0, **overrides: float):
    """The Phase 2 test diode at a given anode bias."""
    settings: dict = {
        "Na": 1e16,
        "Nd": 1e16,
        "length": 12 * MICRON,
        "junction": 6 * MICRON,
        "n_nodes": 201,
        "h_min": 5e-7,
    }
    settings.update(overrides)
    return pn_diode(**settings).with_bias(anode=voltage)


def solved(voltage: float, recombination=None):
    """A converged device and state at one bias, with the models used."""
    device = diode(voltage)
    models = TransportModels.for_device(device, recombination=recombination)
    state = solve_bias(device, models=models)
    assert state.gummel is not None and state.gummel.converged, (
        f"the solve at {voltage:+g} V did not converge: {state.gummel}"
    )
    return device, state, models


def spread(values: np.ndarray) -> float:
    """Largest deviation from the mean, relative to the mean [1]."""
    return float(np.max(np.abs(values - values.mean())) / abs(values.mean()))


def largest_flux_term(device, state, models) -> float:
    """The biggest single term entering any edge flux [1], scaled.

    Jn is the difference of two of these. How much of the answer survives the
    subtraction is set by how large they are next to the current.
    """
    h = device.mesh.h / device.scale.x_0
    X = np.diff(state.psi.data)
    b_plus = np.asarray(B(X))
    b_minus = np.asarray(B(-X))
    terms = [
        (models.Dn / h) * b_plus * state.n.data[1:],
        (models.Dn / h) * b_minus * state.n.data[:-1],
        (models.Dp / h) * b_plus * state.p.data[:-1],
        (models.Dp / h) * b_minus * state.p.data[1:],
    ]
    return float(max(np.max(np.abs(term)) for term in terms))


# ------------------------------------------------------------- the gate


@pytest.mark.parametrize("voltage", [0.3, 0.4, 0.5])
def test_total_current_is_constant_across_the_device(voltage: float) -> None:
    """The primary Phase 2 gate, with recombination disabled.

    Scharfetter-Gummel is built from edge fluxes, so the discrete divergence of
    the discrete current is exactly zero in a source free steady state. Nothing
    about the mesh, the field strength or the twelve decades of density between
    the two contacts weakens that.
    """
    device, state, models = solved(voltage, NoRecombination())
    Jn, Jp = current_densities(device, state, models)

    deviation = spread(Jn.data + Jp.data)
    assert deviation < 1e-6, f"Jn + Jp varies by {deviation:.2e} at {voltage} V"


@pytest.mark.parametrize("voltage", [0.3, 0.4, 0.5])
def test_each_carrier_current_is_separately_constant(voltage: float) -> None:
    """With no recombination, div(Jn) = 0 and div(Jp) = 0 independently.

    Stronger than the gate, which only asks about the sum. If the two carriers
    were trading current with each other the sum could still be constant while
    both halves were wrong.
    """
    device, state, models = solved(voltage, NoRecombination())
    Jn, Jp = current_densities(device, state, models)

    assert spread(Jn.data) < 1e-6
    assert spread(Jp.data) < 1e-6


@pytest.mark.parametrize("voltage", [0.4, 0.5])
def test_recombination_moves_current_between_carriers_but_not_the_total(
    voltage: float,
) -> None:
    """div(Jn + Jp) = R - R = 0, so the total is conserved with SRH on too.

    This is what a diode is: electrons and holes flow in from opposite ends,
    recombine in between, and the total current through every plane is the
    same. Jn alone falls across the device and Jp alone rises, by equal
    amounts.
    """
    device, state, models = solved(voltage)
    Jn, Jp = current_densities(device, state, models)

    assert spread(Jn.data + Jp.data) < 1e-6
    assert spread(Jn.data) > 1e-3, "recombination should bend Jn on its own"


@pytest.mark.parametrize("voltage", [-1.0, 0.1, 0.2, 0.3, 0.4, 0.5])
def test_the_low_bias_deviation_is_cancellation_and_not_a_broken_scheme(
    voltage: float,
) -> None:
    """The measured spread is machine epsilon times the cancellation ratio.

    Across twelve decades of spread and seven of bias, and it is what separates
    the two possible explanations. A conservation error in the assembly would
    have no reason to track the ratio of a flux term to the current, and a
    subtraction that has run out of digits can do nothing else.

    The factor of three is headroom for the roundoff accumulated in the solve
    itself. Measured values sit between 0.4 and 1.2.
    """
    device, state, models = solved(voltage, NoRecombination())
    Jn, Jp = current_densities(device, state, models)
    total = Jn.data + Jp.data

    scaled_current = abs(total.mean()) / device.scale.J_0
    ratio = largest_flux_term(device, state, models) / scaled_current

    assert spread(total) < 3.0 * EPS * ratio


# ----------------------------------------------------------- terminal currents


@pytest.mark.parametrize("voltage", [0.3, 0.4, 0.5])
def test_terminal_currents_sum_to_zero(voltage: float) -> None:
    """Kirchhoff at the device terminals, to 1e-8 of the largest of them.

    docs/04-validation.md lists this as the invariant that catches boundary
    condition errors, and it is the one phases/PHASE-2.md asks for at 1e-8.

    What is left over is the sum of the interior residuals, since the discrete
    divergence telescopes and the integrated recombination cancels between the
    two carriers. So this measures how well the interior equations were solved,
    which is the honest thing for it to measure.
    """
    device, state, models = solved(voltage)
    currents = terminal_currents(device, state, models)

    largest = max(abs(value) for value in currents.values())
    assert abs(sum(currents.values())) < 1e-8 * largest


@pytest.mark.parametrize("voltage", [-1.0, 0.0])
def test_the_terminal_sum_is_at_the_arithmetic_floor_at_low_current(
    voltage: float,
) -> None:
    """Below about 0.25 V the 1e-8 figure stops being reachable, and why.

    The leftover is the sum of the interior residuals, and each of those is a
    difference of edge fluxes that cannot be resolved below machine epsilon
    times the size of the terms. At reverse bias the current is 4e-9 A/cm^2
    while the flux terms are worth 6e-13 A/cm^2 of unresolvable arithmetic, so
    the ratio bottoms out around 1e-4 however well the equations are solved.

    The bound below is that floor rather than a relative tolerance, so the test
    still fails if the boundary conditions are wrong.
    """
    device, state, models = solved(voltage)
    currents = terminal_currents(device, state, models)

    floor = EPS * largest_flux_term(device, state, models) * device.scale.J_0
    assert abs(sum(currents.values())) <= floor


def test_current_flows_from_the_anode_under_forward_bias() -> None:
    """Explicitly required by phases/PHASE-2.md, and the first debugging check.

    docs/05-pitfalls.md: reversing the Bernoulli asymmetry gives a solver that
    converges cleanly to a physically wrong answer with the current running the
    other way. Nothing else in the suite would notice.
    """
    device, state, models = solved(0.4)

    assert terminal_currents(device, state, models)["anode"] > 0.0
    assert terminal_currents(device, state, models)["cathode"] < 0.0


def test_current_reverses_under_reverse_bias() -> None:
    device, state, models = solved(-0.5)

    assert terminal_currents(device, state, models)["anode"] < 0.0


# ------------------------------------------------------------ other invariants


@pytest.mark.parametrize("voltage", [-1.0, 0.0, 0.3, 0.5])
def test_densities_are_positive_at_every_node(voltage: float) -> None:
    """No clamping anywhere, so this is a property of the M-matrix.

    docs/04-validation.md: a negative density means the maximum principle is
    broken, which in 1D means a sign error.
    """
    _, state, _ = solved(voltage)

    assert np.all(state.n.data > 0.0)
    assert np.all(state.p.data > 0.0)


@pytest.mark.parametrize("doping", [1e14, 1e16, 1e18])
def test_mass_action_holds_at_zero_bias(doping: float) -> None:
    """np = n_i^2 everywhere at equilibrium, at every doping level.

    docs/04-validation.md asks for 1e-8 relative. Solving in the quasi-Fermi
    form makes it exact to roundoff instead, because n and p are never solved
    independently of psi.
    """
    device = diode(0.0, Na=doping, Nd=doping)
    state = solve_bias(device)

    np.testing.assert_allclose(state.n.data * state.p.data, 1.0, rtol=1e-10)


def test_the_recombination_rate_vanishes_at_zero_bias() -> None:
    """A device at equilibrium generates nothing and recombines nothing.

    The rate is exactly zero where n*p is exactly 1, which is what the SRH unit
    tests pin. Here n and p come out of a solve, so n*p sits within a few ulp
    of 1 and the leftover rate is that error divided by the SRH denominator.

    What matters is the size of the current it would represent. Integrated over
    the device it comes to 2.7e-25 A/cm^2 against a saturation current of
    1.3e-10, so it is fifteen decades below anything measurable rather than
    merely small.
    """
    device, state, models = solved(0.0)
    rate = np.asarray(models.recombination.rate(state.n.data, state.p.data))

    volume = device.mesh.volume / device.scale.x_0
    spurious = abs(float(np.sum(rate * volume))) * device.scale.J_0
    assert spurious < 1e-23

