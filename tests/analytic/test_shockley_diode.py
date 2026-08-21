"""The Shockley diode, Tier 2 of docs/04-validation.md.

A full solve compared against closed form device physics. Nothing in the solver
knows any of the formulas in this file.

    I = I_s * (exp(V / (n V_T)) - 1)

Two things have to come out right, and they are checked separately because they
test different parts of the physics.

**The saturation current** is set by minority carrier diffusion into the two
quasi-neutral regions, so it tests the continuity equations, the contact
boundary conditions and the lifetimes together. This diode is short based: the
hole diffusion length is 55 um against a 6 um n side, so almost every injected
carrier reaches the contact rather than recombining on the way, and the
coth(W/L) factor in the general expression matters by a factor of nine.

**The ideality factor** is set by which mechanism dominates, and it has to move
from 2 to 1 on its own. At 1e16 with the documented lifetimes this diode is
diffusion limited almost everywhere, so its ideality sits near 1 and the
crossover is pushed below 50 mV. Raising the doping to 1e18 raises depletion
region recombination and lowers diffusion injection at the same time, and the
n = 2 region appears without a single fitted number changing. Both are measured
here, because the contrast is the actual physics.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ddsim.core import constants as C
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import solve_bias
from ddsim.extract.iv import iv_sweep, total_current
from ddsim.extract.params import ideality_factor, saturation_current
from ddsim.physics.recombination import scharfetter_lifetime

MICRON = 1e-4
"""One micron [cm]."""

LENGTH = 12 * MICRON
"""Device length [cm]. Long enough to have genuinely neutral bulk at 1e16."""

JUNCTION = 6 * MICRON
"""Junction position [cm]."""


def diode(doping: float = 1e16, n_nodes: int = 201):
    """A symmetric abrupt junction diode at the stated doping."""
    return pn_diode(
        Na=doping,
        Nd=doping,
        length=LENGTH,
        junction=JUNCTION,
        n_nodes=n_nodes,
        h_min=5e-7 if doping <= 1e16 else 2e-7,
    )


def built_in_potential(Na: float, Nd: float) -> float:
    """V_bi = V_T ln(Na Nd / n_i^2) [V]."""
    return C.V_T() * math.log(Na * Nd / C.n_i() ** 2)


def depletion_width(Na: float, Nd: float, bias: float) -> float:
    """Depletion approximation width [cm] at an applied bias [V]."""
    potential = built_in_potential(Na, Nd) - bias
    return math.sqrt(
        2.0 * C.eps_Si() * potential / C.q * (1.0 / Na + 1.0 / Nd)
    )


def analytic_saturation_current(doping: float, bias: float) -> float:
    """I_s [A/cm^2] for a short based symmetric diode, from the configured models.

    The general result, before any short or long base limit is taken:

        I_s = q n_i^2 [ Dp/(Lp Nd) coth(W_n/Lp) + Dn/(Ln Na) coth(W_p/Ln) ]

    W_n and W_p are the quasi-neutral widths, so the depletion region is taken
    off each side. That correction is worth about 2 percent here and it is
    cheap to include.

    The lifetimes are exactly the ones the solver was given: the Scharfetter
    relation evaluated at this doping, which at 1e16 gives 8.33 us for
    electrons and 2.5 us for holes.
    """
    tau_n = float(scharfetter_lifetime(doping, tau_max=C.TAU_N_MAX))
    tau_p = float(scharfetter_lifetime(doping, tau_max=C.TAU_P_MAX))
    Dn, Dp = C.D_n(), C.D_p()
    Ln, Lp = math.sqrt(Dn * tau_n), math.sqrt(Dp * tau_p)

    edge = 0.5 * depletion_width(doping, doping, bias)
    W_p = JUNCTION - edge
    W_n = (LENGTH - JUNCTION) - edge

    return (
        C.q
        * C.n_i() ** 2
        * (
            Dp / (Lp * doping) / math.tanh(W_n / Lp)
            + Dn / (Ln * doping) / math.tanh(W_p / Ln)
        )
    )


@pytest.fixture(scope="module")
def forward_curve():
    """One forward sweep of the 1e16 diode, reused by several tests."""
    voltages = [round(0.05 * step, 3) for step in range(1, 13)]
    curve = iv_sweep(diode(), "anode", voltages, step=0.05)
    assert curve.complete, curve.message
    return curve


@pytest.fixture(scope="module")
def recombination_curve():
    """The 1e18 diode, where depletion recombination is strong enough to see."""
    voltages = [round(0.04 * step, 3) for step in range(1, 16)]
    curve = iv_sweep(diode(1e18, n_nodes=301), "anode", voltages, step=0.04)
    assert curve.complete, curve.message
    return curve


# ------------------------------------------------------- saturation current


def test_saturation_current_matches_the_analytic_value(forward_curve) -> None:
    """docs/04-validation.md asks for 10 percent. This lands inside 3.

    Measured with the ideality held at 1, which is what diffusion theory says
    it is in this window, rather than fitted. Fitting both would extrapolate a
    slope from 0.45 V back to zero and turn a half percent slope error into a
    20 percent error in I_s.
    """
    measured, _ = saturation_current(
        forward_curve.voltage, forward_curve.current, window=(0.4, 0.5), ideality=1.0
    )
    analytic = analytic_saturation_current(1e16, 0.45)

    assert abs(measured - analytic) / analytic < 0.10, (
        f"I_s: simulated {measured:.4e}, analytic {analytic:.4e}"
    )


def test_the_short_base_correction_is_what_makes_it_agree(forward_curve) -> None:
    """Guards the test above against agreeing for the wrong reason.

    The long base expression, q n_i^2 (Dp/(Lp Nd) + Dn/(Ln Na)), drops the
    coth and is nine times too small here, because the diffusion lengths are
    55 and 175 um against a 6 um base. If the solver were somehow reproducing
    that instead, the test above would fail rather than pass quietly, and this
    records by how much.
    """
    measured, _ = saturation_current(
        forward_curve.voltage, forward_curve.current, window=(0.4, 0.5), ideality=1.0
    )

    tau_n = float(scharfetter_lifetime(1e16, tau_max=C.TAU_N_MAX))
    tau_p = float(scharfetter_lifetime(1e16, tau_max=C.TAU_P_MAX))
    long_base = (
        C.q
        * C.n_i() ** 2
        * (
            C.D_p() / (math.sqrt(C.D_p() * tau_p) * 1e16)
            + C.D_n() / (math.sqrt(C.D_n() * tau_n) * 1e16)
        )
    )

    assert measured / long_base > 5.0


def test_reverse_current_saturates_between_its_two_analytic_bounds() -> None:
    """Reverse current sits above diffusion alone and below full generation.

    The floor is I_s, the diffusion saturation current, which flows whatever
    the reverse bias. The ceiling is q n_i W / (tau_n + tau_p), the generation
    current if every point of the depletion region generated at the rate that
    holds where both densities are far below n_i. The true answer is inside
    that bracket, because near the depletion edges one carrier is still large
    and suppresses the rate.
    """
    device = diode().with_bias(anode=-1.0)
    state = solve_bias(device)
    assert state.gummel is not None and state.gummel.converged

    reverse = abs(total_current(device, state))

    floor = analytic_saturation_current(1e16, -1.0)
    tau_sum = float(
        scharfetter_lifetime(1e16, tau_max=C.TAU_N_MAX)
        + scharfetter_lifetime(1e16, tau_max=C.TAU_P_MAX)
    )
    ceiling = C.q * C.n_i() * depletion_width(1e16, 1e16, -1.0) / tau_sum

    assert floor < reverse < ceiling


def test_reverse_current_is_flat_with_bias() -> None:
    """Saturation, which is what the name says and worth checking.

    It is not perfectly flat: the depletion region widens with reverse bias, so
    the generation volume grows and the current grows slowly with it. That is
    physics rather than an artefact, so the tolerance is loose on purpose.
    """
    currents = []
    for bias in (-0.5, -1.0, -2.0):
        device = diode().with_bias(anode=bias)
        state = solve_bias(device)
        assert state.gummel is not None and state.gummel.converged
        currents.append(abs(total_current(device, state)))

    assert currents[0] < currents[1] < currents[2]
    assert currents[2] / currents[0] < 3.0


# ------------------------------------------------------------ ideality factor


def test_the_diffusion_limited_diode_has_ideality_one(forward_curve) -> None:
    """1e16 with the documented lifetimes is diffusion limited above 0.25 V."""
    midpoint, ideality = ideality_factor(
        forward_curve.voltage, forward_curve.current
    )
    above = ideality[midpoint > 0.25]

    assert np.all(above < 1.05)
    assert np.all(above > 0.95)


def test_the_ideality_crossover_emerges(recombination_curve) -> None:
    """From near 2 at low bias to 1 at moderate bias, with nothing fitted.

    phases/PHASE-2.md: the crossover must emerge, not be fitted. The only thing
    that changed from the diode above is the doping, which raises depletion
    region recombination and cuts minority injection at the same time. Both
    moves come out of the same equations.

    The peak lands at 1.78 rather than exactly 2, and that is correct rather
    than a shortfall. The recombination current is not exactly exp(V/2V_T): the
    depletion region narrows under forward bias, which speeds the rise slightly
    and pulls the apparent ideality below 2. Diffusion current also still
    contributes a few percent at the peak.
    """
    midpoint, ideality = ideality_factor(
        recombination_curve.voltage, recombination_curve.current
    )

    low = ideality[midpoint < 0.25]
    high = ideality[midpoint > 0.5]

    assert low.max() > 1.7, f"peak ideality only reached {low.max():.3f}"
    assert np.all(high < 1.1)
    assert ideality[-1] < ideality[0]


def test_the_ideality_never_exceeds_two(recombination_curve) -> None:
    """Above 2 would mean a mechanism that is not in the model.

    Series resistance and high injection both push it above 2 in a real diode,
    and neither is present here: there is no contact resistance, and the sweep
    stops below high injection. A value above 2 would be a bug.
    """
    _, ideality = ideality_factor(
        recombination_curve.voltage, recombination_curve.current
    )

    assert np.all(ideality < 2.0)


# ------------------------------------------------------------ Gummel behaviour


def test_gummel_converges_at_half_a_volt() -> None:
    """Named explicitly in the phases/PHASE-2.md acceptance list."""
    device = diode().with_bias(anode=0.5)
    state = solve_bias(device)

    assert state.gummel is not None
    assert state.gummel.converged
    assert state.gummel.iterations < 20


def test_gummel_degrades_as_injection_rises() -> None:
    """The expected failure mode, measured rather than fought.

    Gummel converges linearly and its rate is set by how strongly the three
    equations couple. Under low injection the coupling is weak and it takes a
    handful of cycles. As injection approaches the doping the rate climbs
    toward 1 and the cycle count climbs with it. That is the entire reason
    Phase 3 exists, and it is documented rather than damped away.
    """
    counts = []
    guess = None
    for bias in (0.3, 0.6, 0.9):
        device = diode().with_bias(anode=bias)
        state = solve_bias(device, guess=guess, max_iterations=400)
        assert state.gummel is not None and state.gummel.converged
        counts.append(state.gummel.iterations)
        guess = state

    assert counts[0] < counts[1] < counts[2]
    assert counts[2] > 5 * counts[0]


def test_high_injection_is_what_drives_the_degradation() -> None:
    """At 0.9 V the injected density has passed the doping, by construction.

    Worth asserting next to the test above, so that the degradation is tied to
    the physical condition that causes it rather than to the bias number.
    """
    device = diode().with_bias(anode=0.9)
    state = solve_bias(device, max_iterations=400)
    assert state.gummel is not None and state.gummel.converged

    junction = device.mesh.n_nodes // 2
    injected = state.n.data[junction] * device.scale.C_0

    assert injected > 1e16
