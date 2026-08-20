"""Tests for core/field.py.

Under de Mari scaling every quantity exists in two numerically plausible forms.
A scaled potential of 40 and a physical potential of 1.034 V are the same thing,
and adding them produces no exception, no NaN and no crash. It produces a wrong
answer that converges. Every raise tested here is a bug that would otherwise be
invisible.
"""

import numpy as np
import pytest

from ddsim.core.field import Field, Location, ScalingState
from ddsim.core.scaling import ScaleFactors


@pytest.fixture
def scale() -> ScaleFactors:
    return ScaleFactors.for_silicon()


def psi_physical(values: list[float] | None = None) -> Field:
    """A physical potential on nodes [V]."""
    return Field(values or [0.0, 0.5, 1.0], "V", ScalingState.PHYSICAL, Location.NODE)


def psi_scaled(values: list[float] | None = None) -> Field:
    """A scaled potential on nodes [1]."""
    return Field(values or [0.0, 19.3, 38.7], "V", ScalingState.SCALED, Location.NODE)


# ---------------------------------------------------------------- construction


def test_field_carries_data_unit_scaling_and_location() -> None:
    f = Field([1.0, 2.0], "cm^-3", ScalingState.PHYSICAL, Location.NODE)
    assert f.unit == "cm^-3"
    assert f.scaling is ScalingState.PHYSICAL
    assert f.location is Location.NODE


def test_data_is_a_plain_numpy_array_for_hot_loops() -> None:
    """Hot loops check state once at function entry then work on .data."""
    f = psi_physical()
    assert isinstance(f.data, np.ndarray)
    assert f.data.dtype == np.float64
    np.testing.assert_array_equal(f.data, [0.0, 0.5, 1.0])


def test_shape_and_len_delegate_to_the_array() -> None:
    f = psi_physical()
    assert f.shape == (3,)
    assert f.size == 3
    assert len(f) == 3


def test_field_accepts_an_optional_name() -> None:
    f = Field([1.0], "V", ScalingState.PHYSICAL, Location.NODE, name="psi")
    assert f.name == "psi"


def test_repr_shows_unit_scaling_and_location() -> None:
    text = repr(psi_scaled())
    assert "V" in text
    assert "SCALED" in text
    assert "NODE" in text


# -------------------------------------------------- addition and subtraction


def test_add_with_matching_metadata_succeeds() -> None:
    result = psi_physical() + psi_physical()
    np.testing.assert_allclose(result.data, [0.0, 1.0, 2.0])
    assert result.unit == "V"
    assert result.scaling is ScalingState.PHYSICAL
    assert result.location is Location.NODE


def test_subtract_with_matching_metadata_succeeds() -> None:
    result = psi_physical() - psi_physical()
    np.testing.assert_allclose(result.data, [0.0, 0.0, 0.0])


def test_add_different_scaling_state_raises() -> None:
    """The headline case. A scaled 38.7 and a physical 1.0 V are the same
    potential, and nothing but this check will notice."""
    with pytest.raises(ValueError, match="scaling"):
        psi_scaled() + psi_physical()


def test_add_different_location_raises() -> None:
    node = psi_physical()
    edge = Field([1.0, 2.0], "V", ScalingState.PHYSICAL, Location.EDGE)
    with pytest.raises(ValueError, match="location"):
        node + edge


def test_add_different_unit_raises() -> None:
    volts = psi_physical()
    density = Field([1.0, 2.0, 3.0], "cm^-3", ScalingState.PHYSICAL, Location.NODE)
    with pytest.raises(ValueError, match="unit"):
        volts + density


def test_subtract_different_scaling_state_raises() -> None:
    with pytest.raises(ValueError, match="scaling"):
        psi_scaled() - psi_physical()


def test_subtract_different_location_raises() -> None:
    edge = Field([1.0, 2.0], "V", ScalingState.PHYSICAL, Location.EDGE)
    with pytest.raises(ValueError, match="location"):
        psi_physical() - edge


def test_subtract_different_unit_raises() -> None:
    density = Field([1.0, 2.0, 3.0], "cm^-3", ScalingState.PHYSICAL, Location.NODE)
    with pytest.raises(ValueError, match="unit"):
        psi_physical() - density


def test_add_a_bare_scalar_raises() -> None:
    """A bare float has no unit, so it cannot be added to a Field."""
    with pytest.raises(TypeError):
        psi_physical() + 1.0  # type: ignore[operator]


def test_add_a_bare_array_raises() -> None:
    with pytest.raises(TypeError):
        psi_physical() + np.array([1.0, 2.0, 3.0])  # type: ignore[operator]


def test_add_mismatched_length_raises() -> None:
    short = Field([1.0], "V", ScalingState.PHYSICAL, Location.NODE)
    with pytest.raises(ValueError, match="length"):
        psi_physical() + short


def test_negation_preserves_metadata() -> None:
    result = -psi_physical()
    np.testing.assert_allclose(result.data, [0.0, -0.5, -1.0])
    assert result.unit == "V"
    assert result.scaling is ScalingState.PHYSICAL


# ------------------------------------------------ multiplication and division


def test_multiply_combines_unit_strings() -> None:
    volts = psi_physical()
    density = Field([2.0, 2.0, 2.0], "cm^-3", ScalingState.PHYSICAL, Location.NODE)
    assert (volts * density).unit == "V*cm^-3"


def test_divide_combines_unit_strings() -> None:
    volts = psi_physical()
    length = Field([2.0, 2.0, 2.0], "cm", ScalingState.PHYSICAL, Location.NODE)
    assert (volts / length).unit == "V/cm"


def test_divide_by_a_compound_unit_parenthesises_it() -> None:
    volts = psi_physical()
    diff = Field([2.0, 2.0, 2.0], "cm^2/s", ScalingState.PHYSICAL, Location.NODE)
    assert (volts / diff).unit == "V/(cm^2/s)"


def test_multiply_by_dimensionless_preserves_the_other_unit() -> None:
    volts = psi_physical()
    ones = Field([2.0, 2.0, 2.0], "1", ScalingState.PHYSICAL, Location.NODE)
    assert (volts * ones).unit == "V"
    assert (ones * volts).unit == "V"


def test_divide_identical_units_gives_dimensionless() -> None:
    nonzero = psi_physical([1.0, 2.0, 4.0])
    assert (nonzero / nonzero).unit == "1"


def test_divide_by_dimensionless_preserves_the_unit() -> None:
    ones = Field([2.0, 2.0, 2.0], "1", ScalingState.PHYSICAL, Location.NODE)
    assert (psi_physical() / ones).unit == "V"


def test_multiply_different_scaling_state_raises() -> None:
    other = Field([1.0, 1.0, 1.0], "cm^-3", ScalingState.SCALED, Location.NODE)
    with pytest.raises(ValueError, match="scaling"):
        psi_physical() * other


def test_multiply_different_location_raises() -> None:
    edge = Field([1.0, 2.0], "cm^-3", ScalingState.PHYSICAL, Location.EDGE)
    with pytest.raises(ValueError, match="location"):
        psi_physical() * edge


def test_divide_different_scaling_state_raises() -> None:
    other = Field([1.0, 1.0, 1.0], "cm", ScalingState.SCALED, Location.NODE)
    with pytest.raises(ValueError, match="scaling"):
        psi_physical() / other


def test_multiply_by_a_python_scalar_preserves_the_unit() -> None:
    """Scaling a field by a pure number is legal. The number has no unit."""
    result = psi_physical() * 2.0
    np.testing.assert_allclose(result.data, [0.0, 1.0, 2.0])
    assert result.unit == "V"


def test_right_multiply_by_a_python_scalar_works() -> None:
    result = 2.0 * psi_physical()
    np.testing.assert_allclose(result.data, [0.0, 1.0, 2.0])


def test_divide_by_a_python_scalar_preserves_the_unit() -> None:
    result = psi_physical() / 2.0
    np.testing.assert_allclose(result.data, [0.0, 0.25, 0.5])
    assert result.unit == "V"


# -------------------------------------------------------- state conversion


def test_to_scaled_uses_the_scale_factors(scale: ScaleFactors) -> None:
    result = psi_physical().to_scaled(scale)
    assert result.scaling is ScalingState.SCALED
    np.testing.assert_allclose(result.data, np.array([0.0, 0.5, 1.0]) / scale.psi_0)


def test_to_physical_uses_the_scale_factors(scale: ScaleFactors) -> None:
    scaled = Field([1.0], "V", ScalingState.SCALED, Location.NODE)
    result = scaled.to_physical(scale)
    assert result.scaling is ScalingState.PHYSICAL
    np.testing.assert_allclose(result.data, [scale.psi_0])


def test_to_scaled_is_not_hardcoded_to_a_single_factor(
    scale: ScaleFactors,
) -> None:
    """Changing C_0 has to change how a density scales. Nothing may cache."""
    density = Field([1e16], "cm^-3", ScalingState.PHYSICAL, Location.NODE)
    other = ScaleFactors.for_silicon(C_0=1e18)
    assert density.to_scaled(scale).data[0] != density.to_scaled(other).data[0]
    np.testing.assert_allclose(density.to_scaled(other).data, [1e16 / 1e18])


def test_round_trip_to_scaled_and_back_preserves_data(scale: ScaleFactors) -> None:
    original = psi_physical([-1.5, 0.0, 3.25])
    result = original.to_scaled(scale).to_physical(scale)
    np.testing.assert_allclose(result.data, original.data, rtol=1e-14, atol=0.0)


def test_to_scaled_on_an_already_scaled_field_raises() -> None:
    """Not a no-op. Calling it means the caller has lost track of state."""
    with pytest.raises(ValueError, match="already"):
        psi_scaled().to_scaled(ScaleFactors.for_silicon())


def test_to_physical_on_an_already_physical_field_raises() -> None:
    with pytest.raises(ValueError, match="already"):
        psi_physical().to_physical(ScaleFactors.for_silicon())


def test_conversion_preserves_unit_and_location(scale: ScaleFactors) -> None:
    edge = Field([1.0, 2.0], "A/cm^2", ScalingState.PHYSICAL, Location.EDGE)
    result = edge.to_scaled(scale)
    assert result.unit == "A/cm^2"
    assert result.location is Location.EDGE


# ------------------------------------------------------------------ immutability


def test_scaling_state_cannot_be_reassigned() -> None:
    f = psi_physical()
    with pytest.raises(AttributeError):
        f.scaling = ScalingState.SCALED  # type: ignore[misc]


def test_unit_cannot_be_reassigned() -> None:
    f = psi_physical()
    with pytest.raises(AttributeError):
        f.unit = "cm^-3"  # type: ignore[misc]


def test_field_does_not_expose_array_protocol() -> None:
    """np.asarray(field) must not silently strip the metadata."""
    assert not hasattr(Field, "__array__")


# ------------------------------------------------------- the definition of done


def test_refuses_to_add_a_scaled_potential_to_a_physical_one() -> None:
    """The Phase 0 definition of done, written literally.

    38.7 scaled and 1.0 V physical are the same potential. Without this check
    the sum is 39.7 of nothing in particular, and the solve converges to it.
    """
    scaled = Field([38.7], "V", ScalingState.SCALED, Location.NODE)
    physical = Field([1.0], "V", ScalingState.PHYSICAL, Location.NODE)
    with pytest.raises(ValueError):
        scaled + physical
