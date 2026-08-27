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

from ddsim.device.doping import (
    Along,
    Coordinates,
    Erfc,
    Gaussian,
    Mirrored,
    Product,
    Step,
    Uniform,
    abrupt_junction,
)

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


# -------------------------------------------------------- two dimensions
#
# Through Phase 4 a profile saw one array of positions and that array was x,
# because nothing built so far varied with depth: a MOS substrate is uniform
# and a diode varies along its length. A MOSFET source is not like that. It is
# an implant, Gaussian in depth and bounded laterally, and it is the first
# profile in this project that genuinely needs both coordinates.
#
# So a profile is now asked for a value at a Coordinates, which carries x and,
# when the mesh has one, y. A bare array is still a position and still means x,
# which is why every test above this line is untouched.


def test_a_bare_position_is_still_the_x_axis() -> None:
    """The 1D calling convention survives, and it means what it always meant."""
    x = np.linspace(0.0, MICRON, 5)

    at = Coordinates.of(x)

    np.testing.assert_array_equal(at.x, x)
    assert at.y is None


def test_coordinates_pass_through_unchanged() -> None:
    at = Coordinates(np.array([0.0, MICRON]), np.array([0.0, 0.0]))

    assert Coordinates.of(at) is at


def test_a_scalar_position_still_works() -> None:
    """Uniform(1e16)(0.5e-4) is how half the tests above call a profile."""
    assert Coordinates.of(0.5 * MICRON).x == pytest.approx(0.5 * MICRON)


def test_coordinates_refuse_axes_of_different_lengths() -> None:
    """x and y are two coordinates of the same set of nodes, so a mismatch is
    two different meshes being mixed and not something to broadcast around."""
    with pytest.raises(ValueError, match="same number of positions"):
        Coordinates(np.zeros(5), np.zeros(4))


def test_asking_for_depth_on_a_line_says_what_is_missing() -> None:
    """A 1D mesh has no depth, so a depth dependent profile on one is a
    modelling mistake rather than something to fill in with zeros."""
    at = Coordinates.of(np.linspace(0.0, MICRON, 5))

    with pytest.raises(ValueError, match="no y coordinate"):
        at.axis("y")


def test_coordinates_refuse_an_axis_that_is_not_x_or_y() -> None:
    at = Coordinates.of(np.linspace(0.0, MICRON, 5))

    with pytest.raises(ValueError, match="x or y"):
        at.axis("z")  # type: ignore[arg-type]


# --------------------------------------------------------------------- along


def test_along_y_reads_the_depth_coordinate() -> None:
    """The whole point of the wrapper: a 1D shape evaluated down the depth."""
    depth = np.linspace(0.0, MICRON, 7)
    at = Coordinates(np.zeros_like(depth), depth)
    shape = Gaussian(peak=1e19, centre=0.0, sigma=0.05 * MICRON)

    np.testing.assert_array_equal(Along(shape, "y")(at), shape(depth))


def test_along_x_is_what_a_profile_already_did() -> None:
    """Wrapping in the axis a profile reads by default changes nothing, which
    is the check that Along re-labels an axis and does nothing else."""
    at = Coordinates(np.linspace(0.0, MICRON, 7), np.linspace(0.0, MICRON, 7))
    shape = Gaussian(peak=1e19, centre=0.5 * MICRON, sigma=0.05 * MICRON)

    np.testing.assert_array_equal(Along(shape, "x")(at), shape(at))


def test_along_y_on_a_line_is_refused() -> None:
    depth = Along(Gaussian(peak=1e19, centre=0.0, sigma=1e-6), "y")

    with pytest.raises(ValueError, match="no y coordinate"):
        depth(np.linspace(0.0, MICRON, 5))


def test_along_refuses_an_axis_it_does_not_have() -> None:
    at = Coordinates(np.zeros(3), np.zeros(3))

    with pytest.raises(ValueError, match="x or y"):
        Along(Uniform(1e16), "z")(at)  # type: ignore[arg-type]


# ------------------------------------------------------------------- product


def test_a_product_is_separable() -> None:
    """A source implant is a lateral window times a vertical Gaussian, and the
    value at a node is the product of the two one dimensional shapes there.
    """
    x = np.array([0.0, 0.5 * MICRON, MICRON, 1.5 * MICRON])
    y = np.array([0.0, 0.1 * MICRON, 0.2 * MICRON, 0.3 * MICRON])
    at = Coordinates(x, y)

    lateral = Step(left=1.0, right=0.0, position=MICRON)
    vertical = Gaussian(peak=1e20, centre=0.0, sigma=0.05 * MICRON)
    implant = Along(lateral, "x") * Along(vertical, "y")

    np.testing.assert_allclose(implant(at), lateral(x) * vertical(y), rtol=0.0)


def test_a_product_multiplies_every_factor() -> None:
    at = Coordinates.of(np.zeros(3))
    profile = Uniform(2.0) * Uniform(3.0) * Uniform(5.0)

    assert isinstance(profile, Product)
    np.testing.assert_allclose(profile(at), 30.0, rtol=0.0)


def test_a_lateral_bound_is_two_steps_multiplied() -> None:
    """The combinators are enough on their own: a window needs no new class.

    Worth asserting rather than assuming, because the alternative is inventing
    a Window profile that Step and Product already express.
    """
    x = np.linspace(0.0, 2.0 * MICRON, 21)
    window = Step(left=0.0, right=1.0, position=0.5 * MICRON) * Step(
        left=1.0, right=0.0, position=1.5 * MICRON
    )

    values = window(x)

    assert np.all(values[x < 0.5 * MICRON] == 0.0)
    assert np.all(values[(x >= 0.5 * MICRON) & (x < 1.5 * MICRON)] == 1.0)
    assert np.all(values[x >= 1.5 * MICRON] == 0.0)


def test_multiplying_by_a_number_scales_the_profile() -> None:
    """A shape and its peak concentration, which is how an implant is written."""
    shape = Gaussian(peak=1.0, centre=0.0, sigma=0.05 * MICRON)
    x = np.linspace(0.0, MICRON, 11)

    np.testing.assert_allclose((shape * 1e20)(x), 1e20 * shape(x), rtol=1e-15)
    np.testing.assert_allclose((1e20 * shape)(x), 1e20 * shape(x), rtol=1e-15)


def test_multiplying_by_something_that_is_neither_raises() -> None:
    with pytest.raises(TypeError, match="DopingProfile or a number"):
        Uniform(1e16) * "half"  # type: ignore[operator]


# ------------------------------------------------------------------ mirrored


def test_mirroring_reflects_about_a_position() -> None:
    """A drain is a source mirrored, which is the only reason this exists."""
    shape = Erfc(peak=0.5, position=0.4 * MICRON, length=0.05 * MICRON)
    x = np.linspace(0.0, 2.0 * MICRON, 21)
    centre = MICRON

    np.testing.assert_array_equal(
        Mirrored(shape, about=centre)(x), shape(2.0 * centre - x)
    )


def test_mirroring_twice_is_the_original() -> None:
    """To rounding on the position rather than exactly.

    Reflecting is a subtraction, so 2c - (2c - x) comes back one ulp of c away
    from x, and an erfc edge is exponentially sensitive to position: an
    absolute 2e-20 cm on a 0.05 um edge shows up as 3e-14 relative in the
    value. That is the shape amplifying a rounding error, not the reflection
    losing anything, and 3e-14 of a doping concentration is nothing.
    """
    shape = Erfc(peak=0.5, position=0.4 * MICRON, length=0.05 * MICRON)
    x = np.linspace(0.0, 2.0 * MICRON, 21)

    there_and_back = Mirrored(Mirrored(shape, about=MICRON), about=MICRON)

    np.testing.assert_allclose(there_and_back(x), shape(x), rtol=1e-13)


def test_mirroring_leaves_the_depth_alone() -> None:
    """A device is reflected across its centre, not turned upside down. So a
    mirrored implant is at the same depth and the other end of the channel."""
    x = np.linspace(0.0, 2.0 * MICRON, 5)
    y = np.linspace(0.0, 0.2 * MICRON, 5)
    at = Coordinates(x, y)

    lateral = Erfc(peak=0.5, position=0.4 * MICRON, length=0.05 * MICRON)
    vertical = Gaussian(peak=1e20, centre=0.0, sigma=0.05 * MICRON)
    implant = Along(lateral, "x") * Along(vertical, "y")

    np.testing.assert_array_equal(
        Mirrored(implant, about=MICRON)(at),
        lateral(2.0 * MICRON - x) * vertical(y),
    )


def test_mirroring_a_bare_position_reflects_it() -> None:
    """A bare array is x everywhere else in this module, and here too."""
    depth = np.linspace(0.0, MICRON, 5)
    shape = Erfc(peak=1.0, position=0.3 * MICRON, length=0.1 * MICRON)

    np.testing.assert_array_equal(
        Along(Mirrored(shape, about=0.5 * MICRON), "y")(
            Coordinates(np.zeros_like(depth), depth)
        ),
        shape(MICRON - depth),
    )
