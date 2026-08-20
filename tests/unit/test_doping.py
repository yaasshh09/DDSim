"""Tests for device/doping.py.

Profiles are callables of position, never arrays. docs/03-architecture.md is
explicit about why: Phase 5 refines the mesh adaptively, so the profile has to
stay re-evaluable on a mesh that does not exist yet.

Sign convention follows the notation table in docs/01-physics.md, where
net_doping is Nd - Na. Donors positive, acceptors negative.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import erfc

from ddsim.device.doping import Erfc, Gaussian, Step, Uniform, abrupt_junction

MICRON = 1e-4
"""One micron [cm]."""


# ------------------------------------------------------------------- uniform


def test_uniform_is_constant_everywhere() -> None:
    profile = Uniform(1e16)
    x = np.linspace(0.0, MICRON, 11)
    np.testing.assert_allclose(profile(x), 1e16)  # [cm^-3]


def test_uniform_accepts_a_scalar_position() -> None:
    assert Uniform(1e16)(0.5 * MICRON) == 1e16  # [cm^-3]


def test_negative_uniform_represents_acceptors() -> None:
    assert Uniform(-1e16)(0.0) == -1e16  # [cm^-3]


# ---------------------------------------------------------------------- step


def test_step_takes_the_left_value_before_the_position() -> None:
    profile = Step(left=-1e16, right=1e16, position=0.5 * MICRON)
    assert profile(0.25 * MICRON) == -1e16  # [cm^-3]


def test_step_takes_the_right_value_after_the_position() -> None:
    profile = Step(left=-1e16, right=1e16, position=0.5 * MICRON)
    assert profile(0.75 * MICRON) == 1e16  # [cm^-3]


def test_step_is_right_continuous_at_the_junction() -> None:
    """A node exactly on the junction belongs to the right side.

    Arbitrary but it has to be decided somewhere, and an abrupt junction is a
    modelling idealisation anyway. Recorded so nobody is surprised.
    """
    profile = Step(left=-1e16, right=1e16, position=0.5 * MICRON)
    assert profile(0.5 * MICRON) == 1e16  # [cm^-3]


def test_step_changes_sign_across_the_junction() -> None:
    profile = Step(left=-1e16, right=1e16, position=0.5 * MICRON)
    x = np.linspace(0.0, MICRON, 101)
    values = profile(x)
    assert np.any(values < 0.0)
    assert np.any(values > 0.0)


# ------------------------------------------------------------------ gaussian


def test_gaussian_peaks_at_its_centre() -> None:
    profile = Gaussian(peak=1e18, centre=0.3 * MICRON, sigma=0.05 * MICRON)
    assert profile(0.3 * MICRON) == pytest.approx(1e18, rel=1e-15)  # [cm^-3]


def test_gaussian_matches_the_analytic_form() -> None:
    peak, centre, sigma = 1e18, 0.3 * MICRON, 0.05 * MICRON
    profile = Gaussian(peak=peak, centre=centre, sigma=sigma)
    x = np.linspace(0.0, MICRON, 21)
    expected = peak * np.exp(-((x - centre) ** 2) / (2.0 * sigma**2))
    np.testing.assert_allclose(profile(x), expected, rtol=1e-14)


def test_gaussian_falls_by_one_e_at_one_sigma() -> None:
    sigma = 0.05 * MICRON
    profile = Gaussian(peak=1e18, centre=0.0, sigma=sigma)
    assert profile(sigma) == pytest.approx(1e18 * math.exp(-0.5), rel=1e-14)


def test_gaussian_rejects_non_positive_sigma() -> None:
    with pytest.raises(ValueError, match="sigma"):
        Gaussian(peak=1e18, centre=0.0, sigma=0.0)


# ---------------------------------------------------------------------- erfc


def test_erfc_matches_the_analytic_form() -> None:
    peak, position, length = 1e19, 0.0, 0.02 * MICRON
    profile = Erfc(peak=peak, position=position, length=length)
    x = np.linspace(0.0, 0.2 * MICRON, 21)
    np.testing.assert_allclose(profile(x), peak * erfc((x - position) / length))


def test_erfc_is_half_the_peak_at_the_position() -> None:
    """erfc(0) = 1, so the surface value is the peak itself."""
    profile = Erfc(peak=1e19, position=0.0, length=0.02 * MICRON)
    assert profile(0.0) == pytest.approx(1e19, rel=1e-14)  # [cm^-3]


def test_erfc_decays_monotonically(  # noqa: D103
) -> None:
    profile = Erfc(peak=1e19, position=0.0, length=0.02 * MICRON)
    x = np.linspace(0.0, 0.2 * MICRON, 51)
    assert np.all(np.diff(profile(x)) < 0.0)


def test_erfc_rejects_non_positive_length() -> None:
    with pytest.raises(ValueError, match="length"):
        Erfc(peak=1e19, position=0.0, length=0.0)


# --------------------------------------------------------------- composition


def test_profiles_compose_by_addition() -> None:
    """A diffused n well on a p substrate, the standard construction."""
    substrate = Uniform(-1e16)
    well = Gaussian(peak=1e18, centre=0.0, sigma=0.05 * MICRON)
    profile = substrate + well

    x = np.array([0.0, 0.5 * MICRON])
    np.testing.assert_allclose(profile(x), substrate(x) + well(x), rtol=1e-15)


def test_composition_is_associative() -> None:
    a, b, c = Uniform(1e15), Uniform(2e15), Uniform(3e15)
    x = np.array([0.0, MICRON])
    np.testing.assert_allclose(((a + b) + c)(x), (a + (b + c))(x), rtol=1e-15)


def test_three_way_composition_sums_all_terms() -> None:
    profile = Uniform(1e15) + Uniform(2e15) + Uniform(3e15)
    assert profile(0.0) == pytest.approx(6e15, rel=1e-15)  # [cm^-3]


def test_profiles_negate() -> None:
    """Turning a donor profile into an acceptor one."""
    profile = -Uniform(1e16)
    assert profile(0.0) == pytest.approx(-1e16, rel=1e-15)  # [cm^-3]


def test_profiles_subtract() -> None:
    profile = Uniform(1e16) - Uniform(4e15)
    assert profile(0.0) == pytest.approx(6e15, rel=1e-15)  # [cm^-3]


def test_composed_profile_compensates_to_zero_where_terms_cancel() -> None:
    """Compensated material, where the asinh form earns its keep."""
    profile = Uniform(1e16) + Uniform(-1e16)
    assert profile(0.0) == 0.0  # [cm^-3]


def test_adding_a_non_profile_raises() -> None:
    with pytest.raises(TypeError):
        Uniform(1e16) + 5.0  # type: ignore[operator]


# --------------------------------------------------------- mesh independence


def test_a_profile_gives_the_same_values_on_any_mesh() -> None:
    """The reason profiles are callables and not arrays.

    Phase 5 refines adaptively, so the same profile has to be re-evaluable on
    a mesh that did not exist when it was defined.
    """
    profile = Uniform(-1e16) + Gaussian(peak=1e18, centre=0.5 * MICRON, sigma=1e-6)

    coarse = np.linspace(0.0, MICRON, 11)
    fine = np.linspace(0.0, MICRON, 1001)
    shared = np.intersect1d(coarse, fine)

    np.testing.assert_allclose(profile(shared), profile(shared), rtol=0.0)
    for position in shared:
        assert profile(position) == pytest.approx(
            float(profile(np.array([position]))[0]), rel=1e-15
        )


# --------------------------------------------------------------- pn junction


def test_abrupt_junction_is_p_type_on_the_left() -> None:
    profile = abrupt_junction(Na=1e16, Nd=1e16, position=0.5 * MICRON)
    assert profile(0.25 * MICRON) == pytest.approx(-1e16, rel=1e-15)  # [cm^-3]


def test_abrupt_junction_is_n_type_on_the_right() -> None:
    profile = abrupt_junction(Na=1e16, Nd=1e16, position=0.5 * MICRON)
    assert profile(0.75 * MICRON) == pytest.approx(1e16, rel=1e-15)  # [cm^-3]


def test_abrupt_junction_takes_magnitudes_not_signed_values() -> None:
    """Na and Nd are concentrations, so both are given positive."""
    profile = abrupt_junction(Na=2e16, Nd=5e17, position=0.5 * MICRON)
    assert profile(0.0) == pytest.approx(-2e16, rel=1e-15)
    assert profile(MICRON) == pytest.approx(5e17, rel=1e-15)


def test_abrupt_junction_rejects_negative_concentrations() -> None:
    with pytest.raises(ValueError, match="Na"):
        abrupt_junction(Na=-1e16, Nd=1e16, position=0.5 * MICRON)


def test_abrupt_junction_rejects_a_negative_donor_concentration() -> None:
    with pytest.raises(ValueError, match="Nd"):
        abrupt_junction(Na=1e16, Nd=-1e16, position=0.5 * MICRON)
