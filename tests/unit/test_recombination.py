"""Tests for physics/recombination.py.

Shockley-Read-Hall, from docs/01-physics.md:

    R = (n*p - n_i^2) / (tau_p * (n + n1) + tau_n * (p + p1))

R > 0 means net recombination. At equilibrium n*p = n_i^2 and R vanishes
exactly, which is the strongest single check on the algebra: the numerator has
to cancel to the last bit or a device at zero bias generates current out of
nothing.

Everything here is unit agnostic. The same function serves physical cm^-3 and
seconds, or scaled units where n_i^2, n1 and p1 are all measured against C_0.
The two routes are compared directly in one of the tests.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from ddsim.core import constants as C
from ddsim.core.scaling import ScaleFactors
from ddsim.physics.recombination import (
    NoRecombination,
    SRHRecombination,
    scharfetter_lifetime,
    srh_electron_linearization,
    srh_hole_linearization,
    srh_rate,
)

TAU_N = 2.0
"""An arbitrary electron lifetime for the algebra tests [1]."""

TAU_P = 3.0
"""An arbitrary hole lifetime, different from TAU_N so asymmetry shows [1]."""

DENSITY_PAIRS = [(1e-4, 1e4), (1.0, 1.0), (1e6, 1e-6), (1e6, 1e2), (1e8, 1e8)]
"""(n, p) covering deep p-type, intrinsic, deep n-type and high injection [1]."""


def as_float(value: object) -> float:
    """A scalar out of whatever numpy hands back."""
    return float(np.asarray(value))


# ------------------------------------------------------------- the zero of R


@pytest.mark.parametrize("n", [1e-6, 1e-3, 1.0, 1e3, 1e6, 1e10])
def test_rate_is_exactly_zero_at_equilibrium(n: float) -> None:
    """n*p = n_i^2 must give exactly 0.0, not merely a small number.

    A residual generation rate of 1e-30 in a device at zero bias is a current
    with no cause, and it shows up as a floor under the reverse saturation
    current that no amount of mesh refinement removes.
    """
    p = 1.0 / n
    assert srh_rate(n, p, TAU_N, TAU_P) == 0.0


def test_rate_is_positive_above_equilibrium() -> None:
    """np > n_i^2 means net recombination, by the sign convention."""
    assert srh_rate(10.0, 1.0, TAU_N, TAU_P) > 0.0


def test_rate_is_negative_below_equilibrium() -> None:
    """np < n_i^2 means net generation, which is what reverse bias does."""
    assert srh_rate(0.1, 0.1, TAU_N, TAU_P) < 0.0


def test_rate_is_symmetric_under_swapping_the_carriers() -> None:
    """R(n, p, tau_n, tau_p) == R(p, n, tau_p, tau_n).

    Nothing in SRH distinguishes electrons from holes once the lifetimes swap
    with them. If this fails, tau_n and tau_p are attached to the wrong terms
    in the denominator, which is the easiest mistake available here.
    """
    forward = srh_rate(1e4, 1e-2, TAU_N, TAU_P)
    swapped = srh_rate(1e-2, 1e4, TAU_P, TAU_N)
    assert forward == swapped


def test_low_injection_limit_in_n_type() -> None:
    """Pins the pairing that the symmetry test alone cannot see.

    In strongly n-type material the denominator is dominated by tau_p * n, so
    the rate reduces to excess_p / tau_p. That is the physical content of the
    minority carrier lifetime and it is worth asserting directly.
    """
    n = 1e8
    p = 1e-8 + 1e-4  # equilibrium value plus an excess
    np.testing.assert_allclose(
        srh_rate(n, p, TAU_N, TAU_P), 1e-4 / TAU_P, rtol=1e-6
    )


def test_low_injection_limit_in_p_type() -> None:
    """The mirror image: in p-type material the electron lifetime rules."""
    p = 1e8
    n = 1e-8 + 1e-4
    np.testing.assert_allclose(
        srh_rate(n, p, TAU_N, TAU_P), 1e-4 / TAU_N, rtol=1e-6
    )


def test_intrinsic_material_at_equilibrium_has_no_net_rate() -> None:
    assert srh_rate(1.0, 1.0, TAU_N, TAU_P) == 0.0


def test_rate_works_on_arrays() -> None:
    n = np.array([1e-3, 1.0, 1e3])
    p = 1.0 / n
    np.testing.assert_array_equal(srh_rate(n, p, TAU_N, TAU_P), np.zeros(3))


def test_lifetimes_may_vary_per_node() -> None:
    """The Scharfetter lifetime is doping dependent, so tau is an array."""
    n = np.full(3, 1e8)
    p = np.full(3, 1e-8 + 1e-4)
    tau_p = np.array([1.0, 2.0, 4.0])
    np.testing.assert_allclose(
        srh_rate(n, p, TAU_N, tau_p), 1e-4 / tau_p, rtol=1e-6
    )


# ------------------------------------------------------------------- scaling


def test_scaled_and_physical_routes_agree() -> None:
    """R_physical / R_0 must equal R computed from scaled inputs.

    This catches a missing t_0 on the lifetimes, which is otherwise invisible:
    both routes return plausible numbers and they differ by fourteen orders of
    magnitude.
    """
    scale = ScaleFactors.for_silicon()
    n_i = C.n_i()

    n_phys, p_phys = 1e16, 1e6
    tau_n_phys, tau_p_phys = 1e-5, 3e-6

    physical = srh_rate(
        n_phys,
        p_phys,
        tau_n_phys,
        tau_p_phys,
        ni2=n_i * n_i,
        n1=n_i,
        p1=n_i,
    )
    scaled = srh_rate(
        n_phys / scale.C_0,
        p_phys / scale.C_0,
        tau_n_phys / scale.t_0,
        tau_p_phys / scale.t_0,
        ni2=(n_i / scale.C_0) ** 2,
        n1=n_i / scale.C_0,
        p1=n_i / scale.C_0,
    )
    np.testing.assert_allclose(scaled * scale.R_0, physical, rtol=1e-12)


def test_rate_against_a_hand_computed_value() -> None:
    """One fully worked case in physical units, computed a second way.

    n = 1e16, p = 1e12, n_i = 1e10, tau_n = 1e-5, tau_p = 3e-6.
    numerator   = 1e16*1e12 - 1e20 = 9.99999999e27
    denominator = 3e-6*(1e16 + 1e10) + 1e-5*(1e12 + 1e10)
                = 3.000003e10 + 0.00101e10 = 3.001013e10
    R           = 3.3322e17 cm^-3 s^-1
    """
    rate = srh_rate(1e16, 1e12, 1e-5, 3e-6, ni2=1e20, n1=1e10, p1=1e10)
    numerator = 1e16 * 1e12 - 1e20
    denominator = 3e-6 * (1e16 + 1e10) + 1e-5 * (1e12 + 1e10)
    np.testing.assert_allclose(rate, numerator / denominator, rtol=1e-14)
    np.testing.assert_allclose(rate, 3.3322e17, rtol=1e-4)


# ---------------------------------------------------------------- derivatives


def complex_step(function: Callable[[complex], complex], x: float) -> float:
    """df/dx by complex step, exact to machine precision for analytic f."""
    step = 1e-30
    return float(np.imag(function(complex(x, step))) / step)


@pytest.mark.parametrize(("n", "p"), DENSITY_PAIRS)
def test_dR_dn_against_complex_step(n: float, p: float) -> None:
    model = SRHRecombination(tau_n=TAU_N, tau_p=TAU_P)
    reference = complex_step(lambda z: srh_rate(z, p, TAU_N, TAU_P), n)
    np.testing.assert_allclose(as_float(model.d_rate_dn(n, p)), reference, rtol=1e-12)


@pytest.mark.parametrize(("n", "p"), DENSITY_PAIRS)
def test_dR_dp_against_complex_step(n: float, p: float) -> None:
    model = SRHRecombination(tau_n=TAU_N, tau_p=TAU_P)
    reference = complex_step(lambda z: srh_rate(n, z, TAU_N, TAU_P), p)
    np.testing.assert_allclose(as_float(model.d_rate_dp(n, p)), reference, rtol=1e-12)


@pytest.mark.parametrize(("n", "p"), DENSITY_PAIRS)
def test_both_derivatives_are_positive(n: float, p: float) -> None:
    """dR/dn > 0 and dR/dp > 0 at every density.

    This is what keeps the continuity Jacobian an M-matrix once recombination
    joins the diagonal. If either could go negative, a strongly recombining
    node could produce a negative carrier density.
    """
    model = SRHRecombination(tau_n=TAU_N, tau_p=TAU_P)
    assert as_float(model.d_rate_dn(n, p)) > 0.0
    assert as_float(model.d_rate_dp(n, p)) > 0.0


# -------------------------------------------------------------- linearization


@pytest.mark.parametrize(("n", "p"), DENSITY_PAIRS)
def test_electron_linearization_reproduces_the_rate(n: float, p: float) -> None:
    """R = c*n - g exactly at the point it was linearized about.

    The Gummel electron solve replaces R by c*n - g with c and g frozen. That
    substitution is only legitimate if it is exact at the current iterate,
    since otherwise the converged fixed point solves a different equation from
    the one written down.
    """
    c, g = srh_electron_linearization(n, p, TAU_N, TAU_P)
    np.testing.assert_allclose(c * n - g, srh_rate(n, p, TAU_N, TAU_P), rtol=1e-13)


@pytest.mark.parametrize(("n", "p"), DENSITY_PAIRS)
def test_hole_linearization_reproduces_the_rate(n: float, p: float) -> None:
    c, g = srh_hole_linearization(n, p, TAU_N, TAU_P)
    np.testing.assert_allclose(c * p - g, srh_rate(n, p, TAU_N, TAU_P), rtol=1e-13)


@pytest.mark.parametrize(("n", "p"), DENSITY_PAIRS)
def test_linearization_coefficients_are_non_negative(n: float, p: float) -> None:
    """Why the Gummel density solves cannot produce a negative density.

    The continuity matrix is an M-matrix, so its inverse is non-negative. The
    right hand side is built from g and the contact densities, all
    non-negative, so the solution is non-negative too. c >= 0 is what keeps
    the matrix an M-matrix in the first place.
    """
    for c, g in (
        srh_electron_linearization(n, p, TAU_N, TAU_P),
        srh_hole_linearization(n, p, TAU_N, TAU_P),
    ):
        assert as_float(c) >= 0.0
        assert as_float(g) >= 0.0


def test_linearization_slope_is_not_the_exact_derivative() -> None:
    """Guards against someone quietly swapping in dR/dn.

    They agree only where the denominator does not move. The frozen
    denominator is the deliberate choice: it keeps the right hand side
    non-negative, and that is what keeps densities positive.
    """
    n, p = 1e6, 1e2
    model = SRHRecombination(tau_n=TAU_N, tau_p=TAU_P)
    c, _ = srh_electron_linearization(n, p, TAU_N, TAU_P)
    assert as_float(c) != as_float(model.d_rate_dn(n, p))


# ------------------------------------------------------- Scharfetter lifetime


def test_lifetime_is_tau_max_in_undoped_material() -> None:
    assert scharfetter_lifetime(0.0, tau_max=1e-5) == 1e-5


def test_lifetime_is_the_midpoint_at_the_reference_doping() -> None:
    """At N = N_ref the denominator is 2, so tau is halfway to tau_min."""
    tau = scharfetter_lifetime(
        C.N_REF_SRH, tau_max=1e-5, tau_min=1e-7, N_ref=C.N_REF_SRH
    )
    np.testing.assert_allclose(tau, 0.5 * (1e-5 + 1e-7), rtol=1e-14)


def test_lifetime_approaches_tau_min_at_high_doping() -> None:
    """At 1e24 the remaining excess is (tau_max - tau_min)/(N/N_ref), 5e-6 of
    tau_min, so 1e-5 is the tightest honest tolerance here."""
    np.testing.assert_allclose(
        scharfetter_lifetime(1e24, tau_max=1e-5, tau_min=1e-7), 1e-7, rtol=1e-5
    )


def test_lifetime_decreases_with_doping() -> None:
    doping = np.array([0.0, 1e14, 1e16, 1e18, 1e20])
    assert np.all(np.diff(scharfetter_lifetime(doping, tau_max=1e-5)) < 0.0)


def test_lifetime_gamma_sharpens_the_transition() -> None:
    """gamma > 1 makes the rolloff steeper, which is its whole purpose."""
    gentle = scharfetter_lifetime(1e17, tau_max=1e-5, gamma=1.0)
    steep = scharfetter_lifetime(1e17, tau_max=1e-5, gamma=2.0)
    assert steep < gentle


def test_lifetime_at_1e16_matches_the_documented_defaults() -> None:
    """docs/06-constants.md: tau_max 1e-5 s, N_ref 5e16, gamma 1.

    At 1e16 that is 1e-5 / 1.2 = 8.333e-6 s, which is the number the analytic
    saturation current for the Phase 2 diode is built from.
    """
    np.testing.assert_allclose(
        scharfetter_lifetime(1e16, tau_max=C.TAU_N_MAX), 1e-5 / 1.2, rtol=1e-14
    )


def test_negative_doping_raises() -> None:
    """N_total is a total, never a net, so it cannot be negative.

    Passing net doping here is a real and easy mistake, and it would give a
    longer lifetime on the p side than the n side of a symmetric junction.
    """
    with pytest.raises(ValueError, match="total"):
        scharfetter_lifetime(-1e16, tau_max=1e-5)


# -------------------------------------------------------------------- models


def test_srh_model_matches_the_free_function() -> None:
    model = SRHRecombination(tau_n=TAU_N, tau_p=TAU_P)
    n, p = 1e6, 1e2
    np.testing.assert_allclose(
        np.asarray(model.rate(n, p)), srh_rate(n, p, TAU_N, TAU_P), rtol=1e-14
    )


def test_srh_model_carries_its_own_intrinsic_density() -> None:
    """C_0 is not required to be n_i, so n_i^2 cannot be hardcoded as 1.

    docs/02-numerics.md leaves C_0 = max|net doping| open for Phase 5. Under
    that choice scaled n*p at equilibrium is (n_i/C_0)^2 rather than 1.
    """
    model = SRHRecombination(tau_n=TAU_N, tau_p=TAU_P, ni2=4.0, n1=2.0, p1=2.0)
    assert as_float(model.rate(2.0, 2.0)) == 0.0


def test_srh_model_linearizations_match_the_free_functions() -> None:
    model = SRHRecombination(tau_n=TAU_N, tau_p=TAU_P)
    n, p = 1e6, 1e2

    c_model, g_model = model.electron_linearization(n, p)
    c_free, g_free = srh_electron_linearization(n, p, TAU_N, TAU_P)
    np.testing.assert_allclose(np.asarray(c_model), c_free, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(g_model), g_free, rtol=1e-14)

    c_model, g_model = model.hole_linearization(n, p)
    c_free, g_free = srh_hole_linearization(n, p, TAU_N, TAU_P)
    np.testing.assert_allclose(np.asarray(c_model), c_free, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(g_model), g_free, rtol=1e-14)


def test_no_recombination_returns_zeros() -> None:
    model = NoRecombination()
    n = np.array([1e6, 1.0, 1e-6])
    p = np.array([1e-6, 1.0, 1e6])
    electron_c, electron_g = model.electron_linearization(n, p)
    hole_c, hole_g = model.hole_linearization(n, p)
    for values in (
        model.rate(n, p),
        model.d_rate_dn(n, p),
        model.d_rate_dp(n, p),
        electron_c,
        electron_g,
        hole_c,
        hole_g,
    ):
        np.testing.assert_array_equal(np.asarray(values), np.zeros(3))


def test_no_recombination_broadcasts_to_the_input_shape() -> None:
    """A zero model still has to return one value per node.

    Returning a bare 0.0 would broadcast correctly in the residual and then
    silently collapse the Jacobian diagonal contribution to a scalar.
    """
    model = NoRecombination()
    assert np.asarray(model.rate(np.zeros(5), np.zeros(5))).shape == (5,)


def test_a_lifetime_floor_above_the_ceiling_raises() -> None:
    """tau_min above tau_max would make the lifetime rise with doping."""
    with pytest.raises(ValueError, match="tau_max"):
        scharfetter_lifetime(1e16, tau_max=1e-7, tau_min=1e-5)


def test_a_non_positive_reference_doping_raises() -> None:
    with pytest.raises(ValueError, match="N_ref"):
        scharfetter_lifetime(1e16, tau_max=1e-5, N_ref=0.0)
