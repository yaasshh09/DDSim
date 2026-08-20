"""Tests for physics/statistics.py.

Everything here is in de Mari scaled units unless the name says otherwise, so
psi is in units of V_T and densities are in units of C_0 = n_i. That makes
n = exp(psi - phi_n) with no constants floating around.

The invariant that matters most is n*p = 1 in scaled units, which is n*p = n_i^2
physically. docs/04-validation.md wants it to 1e-8 everywhere. The equilibrium
solver below holds it to machine precision by construction, which is the point
of not using the naive quadratic formula.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ddsim.core import constants as C
from ddsim.physics.statistics import (
    dn_dpsi_scaled,
    dp_dpsi_scaled,
    equilibrium_densities_scaled,
    n_boltzmann,
    n_boltzmann_scaled,
    p_boltzmann,
    p_boltzmann_scaled,
    psi_equilibrium_scaled,
)

# ----------------------------------------------------------- Boltzmann, scaled


def test_n_is_one_at_zero_potential() -> None:
    """Intrinsic material: n = n_i, which is 1 in scaled units."""
    assert n_boltzmann_scaled(0.0) == 1.0


def test_p_is_one_at_zero_potential() -> None:
    assert p_boltzmann_scaled(0.0) == 1.0


def test_n_rises_and_p_falls_with_potential() -> None:
    """psi increases toward n-type, per the sign convention in 01-physics."""
    psi = np.array([-5.0, 0.0, 5.0])
    assert np.all(np.diff(n_boltzmann_scaled(psi)) > 0.0)
    assert np.all(np.diff(p_boltzmann_scaled(psi)) < 0.0)


def test_mass_action_holds_at_equilibrium() -> None:
    """phi_n = phi_p = 0 gives n*p = n_i^2. The unit test named in 01-physics."""
    psi = np.linspace(-30.0, 30.0, 61)
    product = n_boltzmann_scaled(psi) * p_boltzmann_scaled(psi)
    np.testing.assert_allclose(product, 1.0, rtol=1e-14)


def test_quasi_fermi_splitting_breaks_mass_action() -> None:
    """Under bias np = n_i^2 * exp((phi_p - phi_n)/V_T), not n_i^2."""
    psi, phi_n, phi_p = 3.0, -0.4, 0.6
    product = n_boltzmann_scaled(psi, phi_n) * p_boltzmann_scaled(psi, phi_p)
    assert product == pytest.approx(math.exp(phi_p - phi_n), rel=1e-14)


def test_derivatives_match_the_analytic_forms() -> None:
    psi = np.linspace(-20.0, 20.0, 41)
    n = n_boltzmann_scaled(psi)
    p = p_boltzmann_scaled(psi)
    np.testing.assert_allclose(dn_dpsi_scaled(psi), n, rtol=1e-15)
    np.testing.assert_allclose(dp_dpsi_scaled(psi), -p, rtol=1e-15)


def test_derivatives_match_complex_step() -> None:
    """Exact here, unlike for the Bernoulli function, because exp is clean."""
    for psi in (-10.0, -1.0, 0.0, 1.0, 10.0):
        step = complex(psi, 1e-30)
        assert dn_dpsi_scaled(psi) == pytest.approx(
            (np.exp(step)).imag / 1e-30, rel=1e-14
        )


# --------------------------------------------------------- Boltzmann, physical


def test_physical_form_reduces_to_n_i_at_zero_potential() -> None:
    assert n_boltzmann(0.0, 0.0, C.n_i(), C.V_T()) == C.n_i()  # [cm^-3]
    assert p_boltzmann(0.0, 0.0, C.n_i(), C.V_T()) == C.n_i()  # [cm^-3]


def test_physical_and_scaled_forms_agree() -> None:
    """A scaled potential of 38.7 is 1.0 V, and both must give the same density."""
    V_T = C.V_T()
    n_i = C.n_i()
    psi_physical = 0.4  # [V]
    psi_scaled = psi_physical / V_T

    physical = n_boltzmann(psi_physical, 0.0, n_i, V_T)
    scaled = n_boltzmann_scaled(psi_scaled) * n_i
    assert physical == pytest.approx(scaled, rel=1e-12)


def test_one_volt_is_a_factor_of_exp_38_7() -> None:
    """The exponential stiffness that makes this problem hard, as a number."""
    ratio = n_boltzmann(1.0, 0.0, C.n_i(), C.V_T()) / C.n_i()
    assert ratio == pytest.approx(math.exp(1.0 / C.V_T()), rel=1e-12)
    assert ratio > 6e16


# ------------------------------------------------------ equilibrium from doping


def test_equilibrium_potential_is_zero_in_intrinsic_material() -> None:
    assert psi_equilibrium_scaled(0.0) == 0.0


def test_equilibrium_potential_is_positive_for_donors() -> None:
    """psi increases toward n-type."""
    assert psi_equilibrium_scaled(1e6) > 0.0
    assert psi_equilibrium_scaled(-1e6) < 0.0


def test_equilibrium_potential_matches_the_asinh_form() -> None:
    """psi = asinh(N/2) in scaled units, from neutrality plus mass action."""
    net = np.array([-1e8, -1e6, -1.0, 0.0, 1.0, 1e6, 1e8])
    np.testing.assert_allclose(
        psi_equilibrium_scaled(net), np.arcsinh(net / 2.0), rtol=1e-15
    )


def test_equilibrium_potential_matches_the_log_form_where_that_is_valid() -> None:
    """For heavy n-type doping asinh(N/2) tends to ln(N), the textbook result.

    1e16 cm^-3 is 1e6 in scaled units, so ln(1e6) = 13.8 which is 0.357 V.
    """
    net_scaled = 1e16 / C.n_i()
    assert psi_equilibrium_scaled(net_scaled) == pytest.approx(
        math.log(net_scaled), rel=1e-12
    )


def test_asinh_form_survives_compensated_material_where_log_would_not() -> None:
    """The frequent bug source called out in 01-physics: N near zero.

    ln(N/n_i) is -inf at N = 0 and NaN for N < 0. asinh is finite and smooth
    through both.
    """
    for net in (-1e-6, 0.0, 1e-30, -1e-30):
        assert math.isfinite(psi_equilibrium_scaled(net))


# ---------------------------------------------------- equilibrium densities


def test_equilibrium_densities_satisfy_neutrality() -> None:
    """p - n + N = 0, one of the two defining conditions."""
    net = np.array([-1e8, -1e6, -1.0, 0.0, 1.0, 1e6, 1e8])
    n, p = equilibrium_densities_scaled(net)
    np.testing.assert_allclose(p - n + net, 0.0, atol=1e-9 * np.abs(net).max())


def test_equilibrium_densities_satisfy_mass_action_exactly() -> None:
    """n*p = 1 to machine precision, not to 1e-8.

    docs/04-validation.md asks for 1e-8. Computing the minority carrier as the
    reciprocal of the majority instead of from the quadratic formula makes it
    exact, which removes the invariant as a source of noise later.
    """
    net = np.array([-1e12, -1e8, -1e6, -1.0, 0.0, 1.0, 1e6, 1e8, 1e12])
    n, p = equilibrium_densities_scaled(net)
    np.testing.assert_allclose(n * p, 1.0, rtol=1e-15)


def test_minority_carrier_does_not_lose_precision_at_heavy_doping() -> None:
    """The reason for the reciprocal trick, stated as a number.

    At N = 1e6 the quadratic formula computes p as (sqrt(1e12 + 4) - 1e6)/2.
    Both terms are 1e6 and the answer is 1e-6, so twelve digits cancel and
    almost nothing survives in double precision.
    """
    net = 1e6  # 1e16 cm^-3 donors, scaled by n_i
    n, p = equilibrium_densities_scaled(net)
    assert p == pytest.approx(1e-6, rel=1e-12)

    naive = (math.sqrt(net * net + 4.0) - net) / 2.0
    assert abs(naive - 1e-6) / 1e-6 > 1e-6, "naive form should be visibly wrong"


def test_equilibrium_densities_are_intrinsic_at_zero_doping() -> None:
    n, p = equilibrium_densities_scaled(0.0)
    assert n == pytest.approx(1.0, rel=1e-15)
    assert p == pytest.approx(1.0, rel=1e-15)


def test_equilibrium_densities_are_consistent_with_the_potential() -> None:
    """The two routes to the same equilibrium must agree."""
    net = np.array([-1e6, -1.0, 0.0, 1.0, 1e6])
    psi = psi_equilibrium_scaled(net)
    n, p = equilibrium_densities_scaled(net)
    np.testing.assert_allclose(n, n_boltzmann_scaled(psi), rtol=1e-12)
    np.testing.assert_allclose(p, p_boltzmann_scaled(psi), rtol=1e-12)


def test_equilibrium_densities_are_positive_everywhere() -> None:
    net = np.array([-1e12, -1e6, 0.0, 1e6, 1e12])
    n, p = equilibrium_densities_scaled(net)
    assert np.all(n > 0.0)
    assert np.all(p > 0.0)


def test_equilibrium_densities_preserve_shape() -> None:
    net = np.linspace(-10.0, 10.0, 12).reshape(3, 4)
    n, p = equilibrium_densities_scaled(net)
    assert n.shape == (3, 4)
    assert p.shape == (3, 4)
