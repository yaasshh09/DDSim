"""Tests for core/scaling.py, the de Mari scale factors.

The single most important test in this file is
test_scaled_poisson_coefficient_is_unity. That group being exactly 1 is the
entire reason de Mari scaling exists. If a scale factor is ever mis-derived,
that test fails and nothing else has to.
"""

import math

import numpy as np
import pytest

from ddsim.core import constants as C
from ddsim.core.scaling import ScaleFactors

UNITS = ["V", "cm^-3", "cm", "cm^2/s", "cm^2/(V s)", "s", "A/cm^2", "cm^-3 s^-1", "1"]


@pytest.fixture
def scale() -> ScaleFactors:
    """Default silicon scale factors at 300 K with C_0 = n_i."""
    return ScaleFactors.for_silicon()


# ------------------------------------------------------- reference quantities


def test_psi_0_equals_thermal_voltage(scale: ScaleFactors) -> None:
    assert scale.psi_0 == C.V_T(300.0)  # [V]


def test_c_0_defaults_to_n_i(scale: ScaleFactors) -> None:
    assert scale.C_0 == C.n_i(300.0)  # [cm^-3]


def test_c_0_is_a_constructor_parameter() -> None:
    """Phase 5 may switch to max|net doping|. It has to be one place."""
    scale = ScaleFactors.for_silicon(C_0=1e18)
    assert scale.C_0 == 1e18  # [cm^-3]


def test_x_0_is_the_debye_length_at_c_0(scale: ScaleFactors) -> None:
    expected = math.sqrt(C.eps_Si() * C.V_T(300.0) / (C.q * C.n_i(300.0)))  # [cm]
    assert scale.x_0 == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize(
    ("doping", "debye_nm"),
    [(1e15, 128.0), (1e16, 40.0), (1e17, 13.0), (1e18, 4.1), (1e20, 0.4)],
)
def test_debye_length_matches_doc_table(doping: float, debye_nm: float) -> None:
    """docs/06-constants.md sanity table, converted from cm to nm.

    The doc quotes two significant figures, so 3 percent is the right
    tolerance. Tightening it further would test our own arithmetic rather
    than agreement with the doc.
    """
    scale = ScaleFactors.for_silicon(C_0=doping)
    assert scale.x_0 * 1e7 == pytest.approx(debye_nm, rel=3e-2)  # [nm]


def test_intrinsic_debye_length_follows_the_doc_formula(scale: ScaleFactors) -> None:
    """docs/06-constants.md used to say 'roughly 24 um'. Its own formula gives
    40.885 um.

    24 um reproduces only as sqrt(eps*V_T/(2*q*n_i)) with n_i = 1.45e10, which
    carries both an extra factor of 2 and the superseded n_i. The formula wins,
    and the doc now says 40.9 um. Pinned here so a future edit that reverts the
    doc cannot quietly take the code with it. See the known deviations table in
    docs/07-decisions.md.
    """
    assert scale.x_0 * 1e4 == pytest.approx(40.885, rel=1e-4)  # [um]


def test_d_0_is_the_larger_carrier_diffusivity(scale: ScaleFactors) -> None:
    assert scale.D_0 == max(C.D_n(300.0), C.D_p(300.0))  # [cm^2/s]


def test_mu_0_equals_d_0_over_psi_0(scale: ScaleFactors) -> None:
    assert scale.mu_0 == pytest.approx(scale.D_0 / scale.psi_0, rel=1e-15)


def test_t_0_equals_x_0_squared_over_d_0(scale: ScaleFactors) -> None:
    assert scale.t_0 == pytest.approx(scale.x_0**2 / scale.D_0, rel=1e-15)


def test_j_0_equals_q_d_0_c_0_over_x_0(scale: ScaleFactors) -> None:
    expected = C.q * scale.D_0 * scale.C_0 / scale.x_0  # [A/cm^2]
    assert scale.J_0 == pytest.approx(expected, rel=1e-15)


def test_r_0_equals_d_0_c_0_over_x_0_squared(scale: ScaleFactors) -> None:
    expected = scale.D_0 * scale.C_0 / scale.x_0**2  # [cm^-3 s^-1]
    assert scale.R_0 == pytest.approx(expected, rel=1e-15)


# ----------------------------------------------------------- the real invariant


@pytest.mark.parametrize("doping", [1e10, 1e14, 1e16, 1e18, 1e20])
def test_scaled_poisson_coefficient_is_unity(doping: float) -> None:
    """lap(psi) = -(p - n + N) only holds if eps*psi_0/(q*C_0*x_0^2) == 1.

    This is the whole point of the scaling. Any error in psi_0, C_0 or x_0
    shows up here and nowhere else until a solve quietly converges wrong.
    """
    scale = ScaleFactors.for_silicon(C_0=doping)
    group = scale.eps * scale.psi_0 / (C.q * scale.C_0 * scale.x_0**2)
    assert group == pytest.approx(1.0, rel=1e-14)


@pytest.mark.parametrize("doping", [1e10, 1e16, 1e20])
def test_scaled_current_coefficient_is_unity(doping: float) -> None:
    """The SG flux is scaled by J_0 = q*D_0*C_0/x_0, so q*D_0*C_0/(x_0*J_0) == 1."""
    scale = ScaleFactors.for_silicon(C_0=doping)
    group = C.q * scale.D_0 * scale.C_0 / (scale.x_0 * scale.J_0)
    assert group == pytest.approx(1.0, rel=1e-14)


def test_scaled_recombination_coefficient_is_unity(scale: ScaleFactors) -> None:
    """div(Jn) = R in scaled form needs J_0/(x_0*R_0) == 1."""
    group = scale.J_0 / (C.q * scale.x_0 * scale.R_0)
    assert group == pytest.approx(1.0, rel=1e-14)


# -------------------------------------------------------------- round tripping


@pytest.mark.parametrize("unit", UNITS)
def test_round_trip_physical_to_scaled_to_physical(
    scale: ScaleFactors, unit: str
) -> None:
    values = np.array([-3.7e5, -1.0, 0.0, 1e-12, 2.5, 8.1e13])
    result = scale.to_physical(scale.to_scaled(values, unit), unit)
    np.testing.assert_allclose(result, values, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("unit", UNITS)
def test_round_trip_scaled_to_physical_to_scaled(
    scale: ScaleFactors, unit: str
) -> None:
    values = np.array([-42.0, -1.0, 0.0, 1e-9, 1.0, 6.02e7])
    result = scale.to_scaled(scale.to_physical(values, unit), unit)
    np.testing.assert_allclose(result, values, rtol=1e-14, atol=0.0)


def test_to_scaled_divides_by_the_factor(scale: ScaleFactors) -> None:
    """A one volt potential is V_T inverse in scaled units, about 38.7."""
    assert scale.to_scaled(1.0, "V") == pytest.approx(1.0 / C.V_T(300.0), rel=1e-15)


def test_to_physical_multiplies_by_the_factor(scale: ScaleFactors) -> None:
    """The scaled potential of 40 from the architecture doc is 1.034 V."""
    assert scale.to_physical(40.0, "V") == pytest.approx(40.0 * C.V_T(300.0), rel=1e-15)


def test_dimensionless_unit_is_the_identity(scale: ScaleFactors) -> None:
    assert scale.factor("1") == 1.0
    assert scale.to_scaled(7.0, "1") == 7.0


def test_scalar_input_returns_a_scalar(scale: ScaleFactors) -> None:
    assert isinstance(scale.to_scaled(1.0, "V"), float)


def test_array_input_returns_an_array(scale: ScaleFactors) -> None:
    result = scale.to_scaled(np.array([1.0, 2.0]), "V")
    assert isinstance(result, np.ndarray)


# ------------------------------------------------------------------ strictness


def test_unknown_unit_raises(scale: ScaleFactors) -> None:
    """Silently passing an unrecognised unit through is the failure this
    whole type exists to prevent."""
    with pytest.raises(KeyError, match="furlong"):
        scale.factor("furlong")


def test_unknown_unit_raises_on_to_scaled(scale: ScaleFactors) -> None:
    with pytest.raises(KeyError):
        scale.to_scaled(1.0, "eV")


def test_scale_factors_are_immutable(scale: ScaleFactors) -> None:
    with pytest.raises((AttributeError, TypeError)):
        scale.psi_0 = 1.0  # type: ignore[misc]


def test_negative_c_0_raises() -> None:
    with pytest.raises(ValueError, match="C_0"):
        ScaleFactors.for_silicon(C_0=-1.0)


# ---------------------------------------------------------------- temperature


def test_scale_factors_at_400k_differ_from_300k() -> None:
    hot = ScaleFactors.for_silicon(T=400.0)
    room = ScaleFactors.for_silicon(T=300.0)
    assert hot.psi_0 > room.psi_0
    assert hot.C_0 > room.C_0


def test_poisson_coefficient_is_unity_at_400k() -> None:
    scale = ScaleFactors.for_silicon(T=400.0)
    group = scale.eps * scale.psi_0 / (C.q * scale.C_0 * scale.x_0**2)
    assert group == pytest.approx(1.0, rel=1e-14)


def test_non_positive_temperature_raises() -> None:
    with pytest.raises(ValueError, match="T"):
        ScaleFactors.for_silicon(T=0.0)


def test_non_positive_permittivity_raises() -> None:
    with pytest.raises(ValueError, match="eps"):
        ScaleFactors.for_silicon(eps=-1.0)


def test_non_positive_diffusivity_raises() -> None:
    with pytest.raises(ValueError, match="D_0"):
        ScaleFactors.for_silicon(D_0=0.0)


def test_direct_construction_rejects_non_positive_temperature() -> None:
    """for_silicon catches this earlier, but the dataclass must guard too."""
    with pytest.raises(ValueError, match="T"):
        ScaleFactors(T=-1.0, C_0=1e10, eps=1e-12, D_0=30.0)
