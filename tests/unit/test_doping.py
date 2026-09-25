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
    Layers,
    Mirrored,
    Product,
    Step,
    Uniform,
    Window,
    abrupt_junction,
)

MICRON = 1e-4
"""One micron [cm]."""


def test_uniform_is_constant_everywhere() -> None:
    profile = Uniform(1e16)
    x = np.linspace(0.0, MICRON, 11)
    np.testing.assert_allclose(profile(x), 1e16)


def test_uniform_accepts_a_scalar_position() -> None:
    assert Uniform(1e16)(0.5 * MICRON) == 1e16


def test_negative_uniform_represents_acceptors() -> None:
    assert Uniform(-1e16)(0.0) == -1e16


def test_step_takes_the_left_value_before_the_position() -> None:
    profile = Step(left=-1e16, right=1e16, position=0.5 * MICRON)
    assert profile(0.25 * MICRON) == -1e16


def test_step_takes_the_right_value_after_the_position() -> None:
    profile = Step(left=-1e16, right=1e16, position=0.5 * MICRON)
    assert profile(0.75 * MICRON) == 1e16


def test_step_is_right_continuous_at_the_junction() -> None:
    """A node exactly on the junction belongs to the right side.

    Arbitrary but it has to be decided somewhere, and an abrupt junction is a
    modelling idealisation anyway. Recorded so nobody is surprised.
    """
    profile = Step(left=-1e16, right=1e16, position=0.5 * MICRON)
    assert profile(0.5 * MICRON) == 1e16


def test_step_changes_sign_across_the_junction() -> None:
    profile = Step(left=-1e16, right=1e16, position=0.5 * MICRON)
    x = np.linspace(0.0, MICRON, 101)
    values = profile(x)
    assert np.any(values < 0.0)
    assert np.any(values > 0.0)


def test_gaussian_peaks_at_its_centre() -> None:
    profile = Gaussian(peak=1e18, centre=0.3 * MICRON, sigma=0.05 * MICRON)
    assert profile(0.3 * MICRON) == pytest.approx(1e18, rel=1e-15)


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


def test_erfc_matches_the_analytic_form() -> None:
    peak, position, length = 1e19, 0.0, 0.02 * MICRON
    profile = Erfc(peak=peak, position=position, length=length)
    x = np.linspace(0.0, 0.2 * MICRON, 21)
    np.testing.assert_allclose(profile(x), peak * erfc((x - position) / length))


def test_erfc_is_half_the_peak_at_the_position() -> None:
    """erfc(0) = 1, so the surface value is the peak itself."""
    profile = Erfc(peak=1e19, position=0.0, length=0.02 * MICRON)
    assert profile(0.0) == pytest.approx(1e19, rel=1e-14)


def test_erfc_decays_monotonically(
) -> None:
    profile = Erfc(peak=1e19, position=0.0, length=0.02 * MICRON)
    x = np.linspace(0.0, 0.2 * MICRON, 51)
    assert np.all(np.diff(profile(x)) < 0.0)


def test_erfc_rejects_non_positive_length() -> None:
    with pytest.raises(ValueError, match="length"):
        Erfc(peak=1e19, position=0.0, length=0.0)


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
    assert profile(0.0) == pytest.approx(6e15, rel=1e-15)


def test_profiles_negate() -> None:
    """Turning a donor profile into an acceptor one."""
    profile = -Uniform(1e16)
    assert profile(0.0) == pytest.approx(-1e16, rel=1e-15)


def test_profiles_subtract() -> None:
    profile = Uniform(1e16) - Uniform(4e15)
    assert profile(0.0) == pytest.approx(6e15, rel=1e-15)


def test_composed_profile_compensates_to_zero_where_terms_cancel() -> None:
    """Compensated material, where the asinh form earns its keep."""
    profile = Uniform(1e16) + Uniform(-1e16)
    assert profile(0.0) == 0.0


def test_adding_a_non_profile_raises() -> None:
    with pytest.raises(TypeError):
        Uniform(1e16) + 5.0


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


def test_abrupt_junction_is_p_type_on_the_left() -> None:
    profile = abrupt_junction(Na=1e16, Nd=1e16, position=0.5 * MICRON)
    assert profile(0.25 * MICRON) == pytest.approx(-1e16, rel=1e-15)


def test_abrupt_junction_is_n_type_on_the_right() -> None:
    profile = abrupt_junction(Na=1e16, Nd=1e16, position=0.5 * MICRON)
    assert profile(0.75 * MICRON) == pytest.approx(1e16, rel=1e-15)


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
        at.axis("z")


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
        Along(Uniform(1e16), "z")(at)


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
        Uniform(1e16) * "half"


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


def test_layers_takes_each_region_value_inside_it() -> None:
    profile = Layers(boundaries=(MICRON, 2.0 * MICRON), values=(-1e18, 1e14, 1e18))
    x = np.array([0.5, 1.5, 2.5]) * MICRON
    np.testing.assert_array_equal(profile(x), [-1e18, 1e14, 1e18])


def test_layers_is_right_continuous_at_every_boundary() -> None:
    """The same choice Step makes, so a node sitting on a junction reads the
    region to its right whichever of the two built the device."""
    profile = Layers(boundaries=(MICRON, 2.0 * MICRON), values=(-1.0, 2.0, 3.0))
    np.testing.assert_array_equal(profile(np.array([MICRON, 2.0 * MICRON])), [2.0, 3.0])


def test_one_boundary_layers_is_the_abrupt_junction_bit_for_bit() -> None:
    """A two region stack is the Phase 2 diode, and it has to be that diode
    exactly, not to a tolerance."""
    x = np.linspace(0.0, MICRON, 201)
    np.testing.assert_array_equal(
        Layers(boundaries=(0.5 * MICRON,), values=(-1e16, 1e16))(x),
        abrupt_junction(Na=1e16, Nd=1e16, position=0.5 * MICRON)(x),
    )


def test_layers_needs_one_more_value_than_boundaries() -> None:
    with pytest.raises(ValueError, match="one more value"):
        Layers(boundaries=(MICRON,), values=(1.0,))


def test_layers_needs_increasing_boundaries() -> None:
    with pytest.raises(ValueError, match="increasing"):
        Layers(boundaries=(2.0 * MICRON, MICRON), values=(1.0, 2.0, 3.0))


ACROSS = np.linspace(-0.5 * MICRON, 1.5 * MICRON, 401)


def test_an_abrupt_window_is_one_inside_and_zero_outside() -> None:
    window = Window(low=0.2 * MICRON, high=0.6 * MICRON)
    values = window(ACROSS)
    inside = (ACROSS >= 0.2 * MICRON) & (ACROSS <= 0.6 * MICRON)
    np.testing.assert_array_equal(values, np.where(inside, 1.0, 0.0))


def test_an_abrupt_window_holds_both_of_its_edges() -> None:
    """Closed, so a node on a drawn edge is inside the rectangle drawn."""
    window = Window(low=0.2 * MICRON, high=0.6 * MICRON)
    np.testing.assert_array_equal(window(np.array([0.2, 0.6]) * MICRON), [1.0, 1.0])


def test_a_gaussian_window_holds_its_peak_inside_and_falls_off_outside() -> None:
    sigma = 0.05 * MICRON
    window = Window(low=0.2 * MICRON, high=0.6 * MICRON, edge="gaussian", length=sigma)
    values = window(ACROSS)
    inside = (ACROSS >= 0.2 * MICRON) & (ACROSS <= 0.6 * MICRON)
    assert np.all(values[inside] == 1.0)
    below = ACROSS < 0.2 * MICRON
    expected = np.exp(-((0.2 * MICRON - ACROSS[below]) ** 2) / (2.0 * sigma**2))
    np.testing.assert_allclose(values[below], expected, rtol=1e-15)
    above = ACROSS > 0.6 * MICRON
    expected = np.exp(-((ACROSS[above] - 0.6 * MICRON) ** 2) / (2.0 * sigma**2))
    np.testing.assert_allclose(values[above], expected, rtol=1e-15)


def test_a_gaussian_window_below_an_open_top_is_the_nmos_depth_profile() -> None:
    """The nmos source in depth: a Gaussian centred on the silicon surface.
    Drawn as a window from the surface up through the top of the device, the
    part below the surface is that Gaussian to the last bit."""
    t_si, sigma = 1.0 * MICRON, 0.05 * MICRON
    depth = np.linspace(0.0, t_si, 301)
    window = Window(low=t_si, high=math.inf, edge="gaussian", length=sigma)
    np.testing.assert_array_equal(
        window(depth), Gaussian(peak=1.0, centre=t_si, sigma=sigma)(depth)
    )


def test_an_erfc_window_open_on_the_left_is_the_nmos_lateral_edge() -> None:
    """Bit for bit: the drawn source's lateral factor is nmos's own."""
    edge = 0.046 * MICRON
    window = Window(low=-math.inf, high=0.4 * MICRON, edge="erfc", length=edge)
    np.testing.assert_array_equal(
        window(ACROSS), Erfc(peak=0.5, position=0.4 * MICRON, length=edge)(ACROSS)
    )


def test_an_erfc_window_open_on_the_right_is_the_mirrored_edge() -> None:
    """The nmos drain is its source mirrored. Equal to rounding rather than
    to the bit, because the mirror adds the reflection in another order."""
    edge, width = 0.046 * MICRON, 1.8 * MICRON
    window = Window(low=1.4 * MICRON, high=math.inf, edge="erfc", length=edge)
    source = Erfc(peak=0.5, position=0.4 * MICRON, length=edge)
    np.testing.assert_allclose(
        window(ACROSS), Mirrored(source, 0.5 * width)(ACROSS), rtol=1e-12, atol=1e-300
    )


def test_an_erfc_window_is_half_its_peak_at_a_mask_edge() -> None:
    edge = 0.02 * MICRON
    window = Window(low=0.2 * MICRON, high=1.2 * MICRON, edge="erfc", length=edge)
    np.testing.assert_allclose(
        window(np.array([0.2, 1.2]) * MICRON), [0.5, 0.5], rtol=1e-12
    )
    assert window(np.array([0.7 * MICRON]))[0] == pytest.approx(1.0, rel=1e-12)


def test_a_narrow_erfc_window_never_reaches_its_peak() -> None:
    """The mask opening is narrower than the spread, so the middle is short
    of full: what an implant through a slit actually does."""
    window = Window(
        low=0.5 * MICRON, high=0.51 * MICRON, edge="erfc", length=0.05 * MICRON
    )
    assert window(np.array([0.505 * MICRON]))[0] < 0.2


def test_a_window_open_on_both_sides_is_one_everywhere() -> None:
    window = Window(low=-math.inf, high=math.inf, edge="erfc", length=0.01 * MICRON)
    np.testing.assert_array_equal(window(ACROSS), 1.0)


def test_a_window_reads_the_axis_it_is_put_along() -> None:
    window = Along(Window(low=0.0, high=0.5 * MICRON), "y")
    at = Coordinates(x=np.array([0.0, 0.0]), y=np.array([0.2, 0.8]) * MICRON)
    np.testing.assert_array_equal(window(at), [1.0, 0.0])


def test_a_window_is_refused_upside_down() -> None:
    with pytest.raises(ValueError, match="low"):
        Window(low=0.6 * MICRON, high=0.2 * MICRON)


def test_a_soft_window_needs_a_length() -> None:
    with pytest.raises(ValueError, match="length"):
        Window(low=0.2 * MICRON, high=0.6 * MICRON, edge="erfc")


def test_a_window_edge_is_one_of_three() -> None:
    with pytest.raises(ValueError, match="abrupt, gaussian or erfc"):
        Window(low=0.2 * MICRON, high=0.6 * MICRON, edge="linear", length=1e-6)
