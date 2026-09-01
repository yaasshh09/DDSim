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
from scipy.special import zeta

from ddsim.core import constants as C
from ddsim.physics.statistics import (
    JOYCE_DIXON_COEFFICIENTS,
    JOYCE_DIXON_MAX_U,
    degeneracy_factor,
    dn_dpsi_scaled,
    dp_dpsi_scaled,
    einstein_ratio,
    equilibrium_densities_scaled,
    fermi_dirac_half,
    fermi_dirac_minus_half,
    joyce_dixon_eta,
    n_boltzmann,
    n_boltzmann_scaled,
    p_boltzmann,
    p_boltzmann_scaled,
    psi_equilibrium_scaled,
)
from tests.reference import fermi as fermi_ref

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


# ------------------------------------------------------------- Fermi-Dirac
#
# docs/04-validation.md tier 1 asks for two things by name: Joyce-Dixon against
# tabulated F_{1/2} values, under 1 percent to n/Nc = 4, and Boltzmann against
# Fermi-Dirac agreeing to 1 percent when n/Nc < 0.01. Both are below, with the
# references in tests/reference/fermi.py, which shares no arithmetic with the
# implementation.
#
# The one tabulated value worth hardcoding is F_{1/2}(0) = 0.678094, because it
# also has a closed form, Gamma(3/2) * eta_dirichlet(3/2), so the table entry
# and the analytic identity check each other before either checks the code.


ETA_NEGATIVE = np.array([-40.0, -20.0, -10.0, -5.0, -2.0, -1.0, -0.5, -0.1])
"""Where the alternating series reference is exact [1]."""

ETA_POSITIVE = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 40.0])
"""Degenerate side, where only the adaptive quadrature reference speaks [1]."""

U_GRID = np.array([1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0, 2.0, 3.5, 4.0])
"""n/Nc values spanning nondegenerate to the 1e20 source and drain [1].

1e20 divided by Nc at 300 K is 3.4965, which is why 3.5 is in the list.
"""


def test_fermi_dirac_half_at_zero_matches_the_tabulated_value() -> None:
    """F_{1/2}(0) = 0.678094, the entry every table starts from."""
    assert fermi_dirac_half(0.0) == pytest.approx(0.678094, rel=1e-6)


def test_fermi_dirac_half_at_zero_matches_the_closed_form() -> None:
    """F_s(0) = Gamma(s+1) * (1 - 2^-s) * zeta(s+1), exactly."""
    closed = math.gamma(1.5) * (1.0 - 2.0**-0.5) * zeta(1.5)
    assert fermi_dirac_half(0.0) == pytest.approx(closed, rel=1e-13)


def test_fermi_dirac_minus_half_at_zero_matches_the_closed_form() -> None:
    closed = math.gamma(0.5) * (1.0 - 2.0**0.5) * zeta(0.5)
    assert fermi_dirac_minus_half(0.0) == pytest.approx(closed, rel=1e-13)


def test_fermi_dirac_half_matches_the_alternating_series() -> None:
    """Below zero the series is exact and the quadrature has to meet it."""
    for eta in ETA_NEGATIVE:
        assert fermi_dirac_half(eta) == pytest.approx(
            fermi_ref.F_series(float(eta), 0.5), rel=1e-12
        )


def test_fermi_dirac_minus_half_matches_the_alternating_series() -> None:
    for eta in ETA_NEGATIVE:
        assert fermi_dirac_minus_half(eta) == pytest.approx(
            fermi_ref.F_series(float(eta), -0.5), rel=1e-12
        )


def test_fermi_dirac_half_matches_adaptive_quadrature_when_degenerate() -> None:
    """Above zero the series says nothing, so a different rule has to agree."""
    for eta in ETA_POSITIVE:
        assert fermi_dirac_half(eta) == pytest.approx(
            fermi_ref.F_quad(float(eta), 0.5), rel=1e-11
        )


def test_fermi_dirac_half_reduces_to_boltzmann_far_below_the_band_edge() -> None:
    """F_{1/2}(eta) tends to Gamma(3/2) exp(eta), which gives n = Nc exp(eta)."""
    eta = np.array([-40.0, -30.0, -20.0])
    boltzmann = math.gamma(1.5) * np.exp(eta)
    np.testing.assert_allclose(fermi_dirac_half(eta), boltzmann, rtol=1e-8)


def test_fermi_dirac_minus_half_is_twice_the_slope_of_the_half_integral() -> None:
    """dF_{1/2}/deta = F_{-1/2}/2, the identity the Einstein ratio rests on."""
    eta = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    h = 1e-5
    slope = (fermi_dirac_half(eta + h) - fermi_dirac_half(eta - h)) / (2 * h)
    np.testing.assert_allclose(2.0 * slope, fermi_dirac_minus_half(eta), rtol=1e-9)


def test_fermi_dirac_half_is_vectorised_and_keeps_its_shape() -> None:
    eta = np.array([[-1.0, 0.0], [1.0, 2.0]])
    assert fermi_dirac_half(eta).shape == (2, 2)


def test_fermi_dirac_half_rises_with_eta() -> None:
    values = fermi_dirac_half(np.linspace(-20.0, 20.0, 81))
    assert np.all(np.diff(values) > 0.0)


# -------------------------------------------------------------- Joyce-Dixon


def test_joyce_dixon_coefficients_are_the_published_values() -> None:
    """Joyce and Dixon 1977. A1 and A2 have closed forms, A3 and A4 do not."""
    A1, A2, A3, A4 = JOYCE_DIXON_COEFFICIENTS
    assert A1 == pytest.approx(1.0 / math.sqrt(8.0), rel=1e-15)
    assert A2 == pytest.approx(3.0 / 16.0 - math.sqrt(3.0) / 9.0, rel=1e-15)
    assert A1 == pytest.approx(3.53553e-1, rel=1e-5)
    assert A2 == pytest.approx(-4.95009e-3, rel=1e-5)
    assert A3 == pytest.approx(1.48386e-4, rel=1e-5)
    assert A4 == pytest.approx(-4.42563e-6, rel=1e-5)


def test_joyce_dixon_reduces_to_the_logarithm_at_low_density() -> None:
    """eta tends to ln(n/Nc), which is Boltzmann and the first term."""
    u = 1e-8
    assert joyce_dixon_eta(u) == pytest.approx(math.log(u), abs=1e-8)


def test_joyce_dixon_inverts_the_integral_to_under_one_percent() -> None:
    """The tier 1 criterion in docs/04-validation.md, stated in density.

    An error in eta is an error in n through exp(eta), so the honest way to
    read "under 1 percent" is to push the returned eta back through F_{1/2}
    and compare densities.
    """
    for u in U_GRID:
        eta = joyce_dixon_eta(float(u))
        assert fermi_ref.u_reference(eta) == pytest.approx(float(u), rel=1e-2)


def test_joyce_dixon_is_far_better_than_its_one_percent_criterion() -> None:
    """Measured 4.4e-5 relative in n at u = 4, not 1e-2. Pin the real number."""
    eta = joyce_dixon_eta(4.0)
    assert fermi_ref.u_reference(eta) == pytest.approx(4.0, rel=1e-4)


def test_joyce_dixon_matches_the_reference_inversion_in_eta() -> None:
    for u in U_GRID:
        assert joyce_dixon_eta(float(u)) == pytest.approx(
            fermi_ref.eta_reference(float(u)), abs=1e-4
        )


def test_joyce_dixon_puts_the_fermi_level_above_the_boltzmann_estimate() -> None:
    """Degeneracy fills the band, so a given n needs a higher E_F than
    Boltzmann says. The correction is positive and grows with density.
    """
    u = np.array([0.1, 1.0, 3.5])
    correction = joyce_dixon_eta(u) - np.log(u)
    assert np.all(correction > 0.0)
    assert np.all(np.diff(correction) > 0.0)


def test_joyce_dixon_correction_at_1e20_is_thirty_millivolts() -> None:
    """docs/07-decisions.md quotes 30.5 mV as the cost of keeping Boltzmann in
    the source and drain. That number came from this series, so it is pinned
    here rather than left in prose.
    """
    u = 1e20 / C.Nc(C.T_ROOM)
    correction = joyce_dixon_eta(u) - math.log(u)
    assert correction * C.V_T(C.T_ROOM) * 1e3 == pytest.approx(30.5, abs=0.1)


def test_joyce_dixon_refuses_a_density_past_its_validated_range() -> None:
    with pytest.raises(ValueError, match="Joyce-Dixon"):
        joyce_dixon_eta(JOYCE_DIXON_MAX_U * 1.001)


def test_joyce_dixon_refuses_a_non_positive_density() -> None:
    with pytest.raises(ValueError, match="positive"):
        joyce_dixon_eta(0.0)


def test_joyce_dixon_refuses_an_array_with_one_bad_entry() -> None:
    """One bad element has to fail, not be silently averaged away."""
    with pytest.raises(ValueError, match="Joyce-Dixon"):
        joyce_dixon_eta(np.array([0.1, 1.0, 1e3]))


# --------------------------------------------------------- degeneracy factor


def test_degeneracy_factor_is_exactly_one_at_zero_density() -> None:
    """Not approximately. Boltzmann has to come back bit for bit, or every
    result this project already has moves in its last digits.
    """
    assert degeneracy_factor(0.0) == 1.0


def test_degeneracy_factor_is_consistent_with_joyce_dixon() -> None:
    """gamma is defined as n / (Nc exp(eta)), so gamma = u / exp(eta_JD(u)).

    Consistency is the point: the solver uses gamma and the Einstein ratio
    together, and two independently fitted approximations would not have a
    common eta between them.
    """
    u = np.array([1e-3, 0.1, 1.0, 3.5])
    np.testing.assert_allclose(
        degeneracy_factor(u), u / np.exp(joyce_dixon_eta(u)), rtol=1e-14
    )


def test_boltzmann_and_fermi_dirac_agree_to_one_percent_below_a_hundredth() -> None:
    """The second tier 1 criterion in docs/04-validation.md, verbatim."""
    u = np.linspace(1e-6, 1e-2, 50)
    np.testing.assert_allclose(degeneracy_factor(u), 1.0, rtol=1e-2)


def test_boltzmann_overestimates_density_threefold_at_1e20() -> None:
    """docs/05-pitfalls.md says "substantially" and this is the number.

    At a fixed Fermi level Boltzmann gives n_boltzmann = n_fermi / gamma, so
    1/gamma is the factor it is wrong by.
    """
    gamma = degeneracy_factor(1e20 / C.Nc(C.T_ROOM))
    assert 1.0 / gamma == pytest.approx(3.26, rel=1e-2)


def test_degeneracy_factor_falls_monotonically_with_density() -> None:
    values = degeneracy_factor(np.linspace(0.0, JOYCE_DIXON_MAX_U, 40))
    assert np.all(np.diff(values) < 0.0)
    assert np.all(values > 0.0)


# ------------------------------------------------------------ Einstein ratio


def test_einstein_ratio_is_exactly_one_at_zero_density() -> None:
    """D = mu * V_T exactly in the nondegenerate limit, so a device that is
    nowhere degenerate pays nothing at all for switching statistics on.
    """
    assert einstein_ratio(0.0) == 1.0


def test_einstein_ratio_matches_the_integral_form_to_one_percent() -> None:
    """2 F_{1/2}/F_{-1/2} at the eta the reference inversion gives."""
    for u in U_GRID:
        assert einstein_ratio(float(u)) == pytest.approx(
            fermi_ref.einstein_reference(float(u)), rel=1e-2
        )


def test_einstein_ratio_is_the_logarithmic_slope_of_joyce_dixon() -> None:
    """D/(mu V_T) = u * deta/du, which is what makes the ratio and the
    inversion the same approximation rather than two of them.
    """
    u = np.array([0.1, 0.5, 1.0, 2.0, 4.0])
    h = 1e-6
    slope = (joyce_dixon_eta(u + h) - joyce_dixon_eta(u - h)) / (2 * h)
    np.testing.assert_allclose(einstein_ratio(u), u * slope, rtol=1e-8)


def test_einstein_ratio_doubles_the_diffusivity_at_1e20() -> None:
    """Degeneracy raises D above mu*V_T. Measured 2.13 in the source and
    drain, which is the whole reason the generalized relation is in scope.
    """
    assert einstein_ratio(1e20 / C.Nc(C.T_ROOM)) == pytest.approx(2.13, rel=1e-2)


def test_einstein_ratio_rises_monotonically_with_density() -> None:
    values = einstein_ratio(np.linspace(0.0, JOYCE_DIXON_MAX_U, 40))
    assert np.all(np.diff(values) > 0.0)


def test_einstein_ratio_refuses_a_density_past_its_validated_range() -> None:
    """Past u = 20 the series turns over and the ratio goes negative, which is
    a diffusivity with the wrong sign. The guard is what stops that.
    """
    with pytest.raises(ValueError, match="Joyce-Dixon"):
        einstein_ratio(JOYCE_DIXON_MAX_U * 1.001)


def test_degeneracy_factor_refuses_a_negative_density() -> None:
    """gamma and the Einstein ratio have no logarithm in them, so a negative
    density gets past the positivity check the inversion uses. It is still not
    a density, and a caller who hands one over has a sign error upstream."""
    with pytest.raises(ValueError, match="negative"):
        degeneracy_factor(-1.0)
