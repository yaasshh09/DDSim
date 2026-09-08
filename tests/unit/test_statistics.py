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
    Degeneracy,
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


# ------------------------------------------- Fermi-Dirac, as the solver holds it
#
# Everything above tests the arithmetic on u = n/Nc. This section tests the
# object a solver holds: the same series in scaled units, answering the four
# questions transport asks. The sharpest test here is the round trip, because
# the forward direction and the inverse cap the density by two different
# routes, and the two caps have to land on the same number rather than nearly
# the same number. See test_the_inversion_is_exact_past_the_validated_density.

DEGENERACY = Degeneracy.for_silicon(C.n_i())
"""Silicon at 300 K, scaled by C_0 = n_i, which is what a device would build."""

PSI_GRID = np.array([-20.0, -5.0, 0.0, 10.0, 20.0, 24.0, 26.0, 30.0, 40.0])
"""Boltzmann exponents from empty to far past the cap [1].

The cap sits at u = 8, which is n = 2.29e10 scaled and psi = 26.4, so this
grid crosses it. That crossing is the whole point of the grid.
"""

COMPLEX_STEP = 1e-30
"""Imaginary step for differentiating the degenerate paths [1]."""


def test_degeneracy_refuses_a_nonpositive_band_density() -> None:
    """Nc appears in a denominator and in a logarithm. Zero is not a material."""
    with pytest.raises(ValueError, match="Nc"):
        Degeneracy(Nc=0.0, Nv=1.0)
    with pytest.raises(ValueError, match="Nv"):
        Degeneracy(Nc=1.0, Nv=-1.0)


def test_for_silicon_scales_both_band_densities_by_the_density_scale() -> None:
    """The class works in scaled units, so C_0 has to divide out of both."""
    scaled = Degeneracy.for_silicon(C.n_i())
    assert scaled.Nc == pytest.approx(C.Nc(C.T_ROOM) / C.n_i(), rel=1e-15)
    assert scaled.Nv == pytest.approx(C.Nv(C.T_ROOM) / C.n_i(), rel=1e-15)


# ------------------------------------------------------ the effective potential


def test_the_effective_potential_is_psi_itself_at_zero_density() -> None:
    """Not approximately. A Boltzmann device that switches statistics on has
    to keep every number it already had, and this is where that starts.
    """
    assert DEGENERACY.electron_potential(3.0, 0.0) == 3.0
    assert DEGENERACY.hole_potential(3.0, 0.0) == 3.0


def test_the_electron_potential_falls_below_psi_and_the_hole_one_rises() -> None:
    """gamma < 1 for both carriers, and ln(gamma) enters with opposite signs.

    Both statements say a filled band pushes its own carriers out.
    """
    n = np.array([1e8, 1e9, 1e10])
    assert np.all(DEGENERACY.electron_potential(0.0, n) < 0.0)
    assert np.all(DEGENERACY.hole_potential(0.0, n) > 0.0)


def test_the_effective_potential_carries_the_joyce_dixon_correction() -> None:
    """psi_eff = psi + ln(gamma), with gamma the same factor tested above."""
    n = np.array([1e6, 1e9, 2e10])
    expected = 2.0 + np.log(degeneracy_factor(n / DEGENERACY.Nc))
    np.testing.assert_allclose(
        DEGENERACY.electron_potential(2.0, n), expected, rtol=1e-14
    )


def test_the_correction_is_thirty_millivolts_at_1e20() -> None:
    """Row 119 of docs/07-decisions.md, in the units the solver works in."""
    n = 1e20 / C.n_i()
    shift = DEGENERACY.electron_potential(0.0, n) * C.V_T() * 1e3  # [mV]
    assert shift == pytest.approx(-30.5, rel=1e-2)


def test_the_potential_derivative_matches_complex_step() -> None:
    """The Bernoulli argument depends on n once psi_eff does, so this
    derivative becomes a Jacobian entry and has to be exact, not close.
    """
    for n in (1e-3, 1.0, 1e9, 1e10):
        step = complex(n, COMPLEX_STEP)
        assert DEGENERACY.d_electron_potential_dn(n) == pytest.approx(
            DEGENERACY.electron_potential(0.0, step).imag / COMPLEX_STEP, rel=1e-12
        )
        assert DEGENERACY.d_hole_potential_dp(n) == pytest.approx(
            DEGENERACY.hole_potential(0.0, step).imag / COMPLEX_STEP, rel=1e-12
        )


def test_the_potential_derivative_is_zero_above_the_cap() -> None:
    """The cap holds the correction constant, so its slope there is zero and
    not the slope the series would have had. A tangent three times too steep
    is what turns the inversion below from quadratic into a crawl.
    """
    n = DEGENERACY.Nc * JOYCE_DIXON_MAX_U * 2.0
    p = DEGENERACY.Nv * JOYCE_DIXON_MAX_U * 2.0
    assert DEGENERACY.d_electron_potential_dn(n) == 0.0
    assert DEGENERACY.d_hole_potential_dp(p) == 0.0


# ---------------------------------------------------------------- the inverse


def test_the_inversion_returns_the_density_the_potential_describes() -> None:
    """n solves psi_eff(psi, n) = ln(n) when phi_n is zero, which is the
    statement that the forward direction and the inverse are one function.
    """
    n = DEGENERACY.electron_density(PSI_GRID)
    np.testing.assert_allclose(
        DEGENERACY.electron_potential(PSI_GRID, n), np.log(n), atol=1e-13
    )
    p = DEGENERACY.hole_density(PSI_GRID)
    np.testing.assert_allclose(
        DEGENERACY.hole_potential(-PSI_GRID, p), -np.log(p), atol=1e-13
    )


def test_the_inversion_is_exact_past_the_validated_density() -> None:
    """Above u = 8 the correction is a constant, so the Newton lands in one
    step. It does not if the two caps disagree: capping ln(u) at ln(8) and
    exponentiating gives 7.999999999999998, which is below the ceiling by an
    ulp, so the slope comes back at its uncapped value, the tangent is 3.4
    times too steep and six steps leave the round trip 0.31 out. Measured.
    """
    psi = np.array([27.0, 30.0, 40.0, 60.0])
    n = DEGENERACY.electron_density(psi)
    assert np.all(n > DEGENERACY.Nc * JOYCE_DIXON_MAX_U)
    np.testing.assert_allclose(
        DEGENERACY.electron_potential(psi, n), np.log(n), atol=0.0
    )


def test_the_inversion_reduces_to_the_exponential_where_the_band_is_empty() -> None:
    """Far below the band edge there is nothing to correct, so n = exp(psi)."""
    psi = np.array([-20.0, -10.0, 0.0])
    np.testing.assert_allclose(DEGENERACY.electron_density(psi), np.exp(psi), rtol=1e-9)


def test_the_inversion_is_monotone_across_the_cap() -> None:
    """A density that stopped rising with the potential would make the Poisson
    diagonal vanish. The cap has to hold the correction constant, not turn it.
    """
    n = DEGENERACY.electron_density(np.linspace(-40.0, 60.0, 400))
    assert np.all(np.diff(n) > 0.0)


def test_a_node_with_no_carriers_comes_back_at_exactly_zero() -> None:
    """A carrier free node arrives with an exponent of -inf. Without the
    guard the Newton computes inf minus inf and returns nan, which poisons a
    whole assembly rather than one row.
    """
    n = DEGENERACY.electron_density(np.array([-np.inf, 0.0]))
    assert n[0] == 0.0
    assert n[1] == pytest.approx(1.0, rel=1e-9)


def test_dn_dpsi_is_the_density_over_the_einstein_ratio() -> None:
    """Filling the band means a given rise in the Fermi level buys less
    density, and the factor it is short by is the same generalized Einstein
    ratio the diffusivity uses. One approximation, used twice.
    """
    n = np.array([1e-3, 1e8, 1e9, 2e10])
    np.testing.assert_allclose(
        DEGENERACY.dn_dpsi(n), n / einstein_ratio(n / DEGENERACY.Nc), rtol=1e-14
    )
    np.testing.assert_allclose(
        DEGENERACY.dp_dpsi(n), n / einstein_ratio(n / DEGENERACY.Nv), rtol=1e-14
    )


def test_dn_dpsi_matches_complex_step_through_the_inversion() -> None:
    """The Poisson diagonal is exactly this derivative, so the closed form and
    the derivative of the inverse have to be the same number.
    """
    for psi in (-10.0, 0.0, 10.0, 20.0, 24.0):
        step = complex(psi, COMPLEX_STEP)
        assert DEGENERACY.dn_dpsi(DEGENERACY.electron_density(psi)) == pytest.approx(
            DEGENERACY.electron_density(step).imag / COMPLEX_STEP, rel=1e-12
        )


def test_dn_dpsi_is_the_density_itself_in_the_boltzmann_limit() -> None:
    """Where the ratio is 1, this has to give back dn/dpsi = n exactly."""
    assert DEGENERACY.dn_dpsi(0.0) == 0.0
    assert DEGENERACY.dp_dpsi(0.0) == 0.0


# --------------------------------------------------------------- the contacts


def test_equilibrium_neutrality_is_exact() -> None:
    """n - p = N to the last bit, because a contact that is not neutral puts a
    space charge sheet at the boundary that nothing in the device asked for.
    """
    N = np.array([-1e10, -1e7, 0.0, 1e7, 1e10, 1e20 / C.n_i()])
    n, p = DEGENERACY.equilibrium_densities(N)
    np.testing.assert_allclose(n - p, N, rtol=1e-15, atol=1e-15)


def test_equilibrium_mass_action_carries_both_degeneracy_factors() -> None:
    """n*p = 1 becomes n*p = gamma_n gamma_p, which is 0.31 at 1e20 rather
    than 1. Exact rather than close, because the minority carrier is taken
    from the product and not from the quadratic formula.
    """
    N = np.array([-1e10, 1e7, 1e10, 1e20 / C.n_i()])
    n, p = DEGENERACY.equilibrium_densities(N)
    product = degeneracy_factor(n / DEGENERACY.Nc) * degeneracy_factor(
        p / DEGENERACY.Nv
    )
    np.testing.assert_allclose(n * p, product, rtol=1e-14)


def test_the_equilibrium_potential_agrees_with_both_carriers() -> None:
    """psi, n and p at a contact node are one state, not three conditions that
    nearly agree. Read off the electrons, it has to satisfy the holes too.
    """
    N = np.array([-1e20 / C.n_i(), -1e10, 1e7, 1e10, 1e20 / C.n_i()])
    n, p = DEGENERACY.equilibrium_densities(N)
    psi = DEGENERACY.equilibrium_psi(N)
    np.testing.assert_allclose(
        DEGENERACY.electron_potential(psi, n), np.log(n), atol=1e-13
    )
    np.testing.assert_allclose(
        DEGENERACY.hole_potential(psi, p), -np.log(p), atol=1e-13
    )


def test_the_equilibrium_contact_reduces_to_boltzmann_at_low_doping() -> None:
    """At 1e13 cm^-3, u = n/Nc is 3.5e-7 and the correction is A1 u = 1.2e-7,
    so the degenerate contact and the Boltzmann one agree to seven digits. A
    device that is nowhere degenerate pays essentially nothing for switching
    statistics on, and the residue it does pay is the correction itself and
    not an error in solving for it.
    """
    N = np.array([-1e3, 0.0, 1e3])
    n, p = DEGENERACY.equilibrium_densities(N)
    n_boltz, p_boltz = equilibrium_densities_scaled(N)
    np.testing.assert_allclose(n, n_boltz, rtol=1e-6)
    np.testing.assert_allclose(p, p_boltz, rtol=1e-6)
    np.testing.assert_allclose(
        DEGENERACY.equilibrium_psi(N), psi_equilibrium_scaled(N), atol=1e-6
    )


def test_the_degenerate_contact_sits_thirty_millivolts_above_boltzmann() -> None:
    """At 1e20 the band is filled, so a given electron density needs a higher
    Fermi level than Boltzmann says. Row 119 predicts 30.5 mV and this is the
    contact actually built from it, not the bare series.
    """
    N = 1e20 / C.n_i()
    shift = (
        (DEGENERACY.equilibrium_psi(N) - psi_equilibrium_scaled(N)) * C.V_T() * 1e3
    )  # [mV]
    assert shift == pytest.approx(30.5, rel=1e-2)


def test_the_equilibrium_contact_is_symmetric_between_the_carriers() -> None:
    """Swapping Nc and Nv and the sign of the doping has to swap n and p. The
    hole branch is a separate code path and this is what says it is a mirror.
    """
    mirrored = Degeneracy(Nc=DEGENERACY.Nv, Nv=DEGENERACY.Nc)
    N = 1e20 / C.n_i()
    n, p = DEGENERACY.equilibrium_densities(N)
    p_mirror, n_mirror = mirrored.equilibrium_densities(-N)
    assert n_mirror == pytest.approx(n, rel=1e-14)
    assert p_mirror == pytest.approx(p, rel=1e-14)
    assert mirrored.equilibrium_psi(-N) == pytest.approx(
        -DEGENERACY.equilibrium_psi(N), rel=1e-14
    )


def test_scalar_input_returns_scalar_shape() -> None:
    """Callers hand these single contact values as well as whole node arrays."""
    assert np.shape(DEGENERACY.equilibrium_psi(1.0)) == ()
    assert np.shape(DEGENERACY.electron_density(0.0)) == ()
    n, p = DEGENERACY.equilibrium_densities(1.0)
    assert np.shape(n) == ()
    assert np.shape(p) == ()
