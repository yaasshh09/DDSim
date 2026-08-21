"""Tests for device/transport.py, the Gummel cycle wired to a real device.

This is where the pieces meet: the mesh and doping from device/, the residuals
and Jacobians from discretize/, the models from physics/, and the iteration
from solve/. Nothing here assembles or iterates, it only wires.

The load bearing test is the first one. Thermal equilibrium is an exact fixed
point of the whole Gummel cycle: with n = exp(psi) and p = exp(-psi) the
quasi-Fermi levels are flat at zero, Poisson is already solved, the
recombination rate is zero, and every edge flux cancels identically. So solving
a device at zero bias has to return the equilibrium solution unchanged. Any
sign error anywhere in the three blocks breaks it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.core import constants as C
from ddsim.device.equilibrium import solve_equilibrium
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import TransportModels, solve_bias
from ddsim.physics.recombination import NoRecombination
from ddsim.physics.statistics import equilibrium_densities_scaled

MICRON = 1e-4
"""One micron [cm]."""


def diode(**overrides: float):
    """A 1e16 / 1e16 abrupt junction, 12 um long, graded to 5 nm at the junction.

    Long enough that the quasi-neutral regions are genuinely neutral. Phase 1
    measured that a 1 um diode at 1e16 has essentially no neutral bulk at all,
    since the depletion region alone is 0.43 um wide.
    """
    settings: dict = {
        "Na": 1e16,
        "Nd": 1e16,
        "length": 12 * MICRON,
        "junction": 6 * MICRON,
        "n_nodes": 201,
        "h_min": 5e-7,
    }
    settings.update(overrides)
    return pn_diode(**settings)


# ------------------------------------------------------- the equilibrium limit


def test_zero_bias_reproduces_the_equilibrium_solution() -> None:
    """Equilibrium is an exact fixed point of the Gummel cycle.

    Not approximately. phi_n and phi_p are flat at zero, so nonlinear Poisson
    is already converged; n*p = 1 makes the recombination rate vanish; and
    B(-X)*exp(psi_left) equals B(X)*exp(psi_right) term by term, so every edge
    flux is zero. The cycle has nothing to do and must do nothing.
    """
    device = diode()
    reference = solve_equilibrium(device)
    solved = solve_bias(device)

    assert solved.gummel is not None
    assert solved.gummel.converged
    np.testing.assert_allclose(solved.psi.data, reference.psi.data, atol=1e-10)
    np.testing.assert_allclose(solved.n.data, reference.n.data, rtol=1e-9)
    np.testing.assert_allclose(solved.p.data, reference.p.data, rtol=1e-9)


def test_zero_bias_converges_immediately() -> None:
    """Being at the answer already, the cycle should stop after proving it."""
    solved = solve_bias(diode())

    assert solved.gummel is not None
    assert solved.gummel.iterations <= 2


def test_mass_action_holds_at_zero_bias() -> None:
    """np = n_i^2 everywhere, which is 1 in scaled units with C_0 = n_i.

    docs/04-validation.md asks for 1e-8 relative. The solve does far better,
    because the equilibrium densities come from the quasi-Fermi form rather
    than from two independent solves.
    """
    solved = solve_bias(diode())

    np.testing.assert_allclose(solved.n.data * solved.p.data, 1.0, rtol=1e-10)


def test_the_quasi_fermi_levels_are_flat_at_zero_bias() -> None:
    """Zero current means no gradient in either level, by definition."""
    solved = solve_bias(diode())

    assert np.max(np.abs(solved.phi_n.data)) < 1e-9
    assert np.max(np.abs(solved.phi_p.data)) < 1e-9


# --------------------------------------------------------------- forward bias


def test_forward_bias_splits_the_quasi_fermi_levels_by_the_applied_bias() -> None:
    """The definition of an applied bias, in the language of the solver.

    phi_n is pinned at the cathode bias and phi_p at the anode bias, and across
    the junction they separate by exactly the difference. Under forward bias
    that separation is what drives n*p above n_i^2 and makes recombination
    positive.
    """
    applied = 0.3
    device = diode().with_bias(anode=applied)
    solved = solve_bias(device)

    assert solved.gummel is not None and solved.gummel.converged

    scaled_bias = applied / device.scale.psi_0
    junction = device.mesh.n_nodes // 2
    separation = solved.phi_p.data[junction] - solved.phi_n.data[junction]
    np.testing.assert_allclose(separation, scaled_bias, rtol=1e-3)


def test_forward_bias_raises_np_above_equilibrium_in_the_junction() -> None:
    device = diode().with_bias(anode=0.3)
    solved = solve_bias(device)

    junction = device.mesh.n_nodes // 2
    assert solved.n.data[junction] * solved.p.data[junction] > 1e3


def test_densities_stay_positive_under_forward_bias() -> None:
    """No clamping anywhere, so this is a property of the discretization.

    docs/05-pitfalls.md: clamping a negative density masks a broken scheme and
    produces a solution that satisfies no equation. The M-matrix structure of
    the continuity assembly is what makes clamping unnecessary.
    """
    solved = solve_bias(diode().with_bias(anode=0.4))

    assert np.all(solved.n.data > 0.0)
    assert np.all(solved.p.data > 0.0)


def test_reverse_bias_depletes_the_junction() -> None:
    """np falls below n_i^2 in the depletion region, which is net generation."""
    device = diode().with_bias(anode=-1.0)
    solved = solve_bias(device)

    assert solved.gummel is not None and solved.gummel.converged
    junction = device.mesh.n_nodes // 2
    assert solved.n.data[junction] * solved.p.data[junction] < 1e-3


# ------------------------------------------------------------------- models


def test_lifetimes_follow_the_doping() -> None:
    """Scharfetter, evaluated on the total doping at each node.

    At 1e16 with N_ref = 5e16 the electron lifetime is 1e-5/1.2 = 8.33 us, and
    the scaled value is that divided by t_0.
    """
    device = diode()
    models = TransportModels.for_device(device)

    expected = (C.TAU_N_MAX / 1.2) / device.scale.t_0
    tau_n = np.asarray(models.recombination.tau_n)  # type: ignore[attr-defined]
    np.testing.assert_allclose(tau_n[0], expected, rtol=1e-12)


def test_diffusivities_are_scaled_by_D_0() -> None:
    """D_0 is max(Dn, Dp), which is Dn, so the electron value is exactly 1."""
    models = TransportModels.for_device(diode())

    np.testing.assert_allclose(models.Dn, 1.0, rtol=1e-14)
    np.testing.assert_allclose(models.Dp, C.MU_P_300 / C.MU_N_300, rtol=1e-12)


def test_recombination_can_be_switched_off() -> None:
    """The configuration the primary Phase 2 gate is measured under."""
    device = diode()
    models = TransportModels.for_device(device, recombination=NoRecombination())
    solved = solve_bias(device.with_bias(anode=0.3), models=models)

    assert solved.gummel is not None and solved.gummel.converged


def test_the_intrinsic_density_survives_a_different_C_0() -> None:
    """C_0 is not required to be n_i, and the models must not assume it is."""
    device = pn_diode(Na=1e16, Nd=1e16, n_nodes=101)
    models = TransportModels.for_device(device)

    ni2 = models.recombination.ni2  # type: ignore[attr-defined]
    np.testing.assert_allclose(ni2, (device.material.n_i / device.scale.C_0) ** 2)


# ------------------------------------------------------------------ plumbing


def test_a_guess_is_used_as_the_starting_point() -> None:
    """Continuation depends on this, so it is worth an explicit test."""
    device = diode()
    equilibrium = solve_bias(device)
    biased = device.with_bias(anode=0.2)

    cold = solve_bias(biased)
    warm = solve_bias(biased, guess=equilibrium)

    assert cold.gummel is not None and warm.gummel is not None
    np.testing.assert_allclose(warm.psi.data, cold.psi.data, atol=1e-6)


def test_with_bias_returns_a_new_device() -> None:
    device = diode()
    biased = device.with_bias(anode=0.5)

    assert device.contacts[0].voltage == 0.0
    assert biased.contacts[0].voltage == 0.5
    assert biased.contacts[1].voltage == device.contacts[1].voltage
    assert biased.mesh is device.mesh


def test_with_bias_rejects_an_unknown_contact() -> None:
    with pytest.raises(KeyError, match="drain"):
        diode().with_bias(drain=0.5)


def test_a_stalled_solve_is_reported_rather_than_raised() -> None:
    """At high injection Gummel is expected to fail, so failure is data.

    phases/PHASE-2.md asks for the bias at which it gives up to be documented.
    That number only exists if a stalled solve comes back as a result.
    """
    solved = solve_bias(diode().with_bias(anode=0.9), max_iterations=3)

    assert solved.gummel is not None
    assert not solved.gummel.converged
    assert solved.gummel.message


def test_the_update_history_falls_monotonically_at_low_bias() -> None:
    """Linear convergence, which is what Gummel promises and all it promises."""
    solved = solve_bias(diode().with_bias(anode=0.2))

    assert solved.gummel is not None
    history = solved.gummel.update_history
    assert history[-1] < history[0]


# ------------------------------------------------------- the contact condition


@pytest.mark.parametrize("voltage", [0.0, 0.4, -1.0])
def test_the_contact_densities_come_out_exact(voltage: float) -> None:
    """A Dirichlet value is imposed on the solution, not approached by it.

    Both densities at an ohmic contact are the neutrality and mass action
    values, whatever the terminal voltage, so they are known before the solve
    and have to appear in the answer to the last bit. Building them as
    old + update instead loses digits whenever the two are far apart, which on
    the first forward biased cycle they are by thirteen decades.
    """
    device = diode().with_bias(anode=voltage)
    state = solve_bias(device)
    doping = device.net_doping_scaled.data

    for contact in device.contacts:
        expected_n, expected_p = equilibrium_densities_scaled(
            float(doping[contact.node])
        )
        assert state.n.data[contact.node] == float(expected_n)
        assert state.p.data[contact.node] == float(expected_p)


def test_a_cold_start_at_high_forward_bias_still_converges() -> None:
    """No continuation, straight to 0.9 V from the flat level guess.

    docs/05-pitfalls.md says there is no such thing as a good initial guess at
    high forward bias, and it is right that continuation is the way to work.
    This is here because the case used to fail outright: the pinned minority
    density at the anode came back as -1.02e-6 and the solve stopped, and
    whether it did so depended on the last bit of the device length. Both the
    column elimination in apply_dirichlet and imposing the contact densities
    were needed to make it stop mattering.
    """
    state = solve_bias(diode().with_bias(anode=0.9), max_iterations=400)

    assert state.gummel is not None
    assert state.gummel.converged
    assert np.all(state.n.data > 0.0)
