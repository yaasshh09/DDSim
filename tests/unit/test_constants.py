"""Tests for core/constants.py.

Every number asserted here traces to docs/06-constants.md. Where the doc and a
formula in the doc disagree, the formula wins and the discrepancy is noted in
PROGRESS.md.
"""

import math

import pytest

from ddsim.core import constants as C

# ---------------------------------------------------------------- fundamental


def test_fundamental_constants_match_doc() -> None:
    assert C.q == 1.602176634e-19  # [C]
    assert C.k_B == 1.380649e-23  # [J/K]
    assert C.eps_0 == 8.8541878128e-14  # [F/cm]
    assert C.h == 6.62607015e-34  # [J s]
    assert C.m_0 == 9.1093837015e-31  # [kg]


def test_room_temperature_is_300k() -> None:
    assert C.T_ROOM == 300.0  # [K]


# ------------------------------------------------------------ thermal voltage


def test_v_t_at_300k_matches_doc() -> None:
    assert C.V_T(300.0) == pytest.approx(0.0258520, abs=1e-7)  # [V]


def test_v_t_equals_kt_over_q() -> None:
    T = 350.0
    assert C.V_T(T) == C.k_B * T / C.q  # [V]


def test_v_t_is_linear_in_temperature() -> None:
    assert C.V_T(600.0) == pytest.approx(2.0 * C.V_T(300.0), rel=1e-15)


def test_v_t_default_argument_is_300k() -> None:
    assert C.V_T() == C.V_T(300.0)


# ------------------------------------------------------------------- band gap


def test_eg_varshni_at_300k() -> None:
    expected = 1.1696 - 4.73e-4 * 300.0**2 / (300.0 + 636.0)  # [eV]
    assert C.Eg(300.0) == pytest.approx(expected, rel=1e-15)


def test_eg_at_300k_matches_doc_table_to_rounding() -> None:
    # Doc table says 1.1242 eV. The Varshni formula in the same doc gives
    # 1.124119 eV. The formula is authoritative, so this is a rounding check.
    assert C.Eg(300.0) == pytest.approx(1.1242, abs=2e-4)  # [eV]


def test_eg_at_zero_kelvin_is_varshni_intercept() -> None:
    assert C.Eg(0.0) == pytest.approx(1.1696, rel=1e-15)  # [eV]


def test_eg_decreases_with_temperature() -> None:
    temperatures = [100.0, 200.0, 300.0, 400.0, 500.0]
    gaps = [C.Eg(T) for T in temperatures]
    assert all(a > b for a, b in zip(gaps[:-1], gaps[1:], strict=True))


def test_eg_default_argument_is_300k() -> None:
    assert C.Eg() == C.Eg(300.0)


# ------------------------------------------------- band edge density of states


def test_nc_nv_at_300k_match_doc() -> None:
    assert C.Nc(300.0) == pytest.approx(2.86e19, rel=1e-15)  # [cm^-3]
    assert C.Nv(300.0) == pytest.approx(3.10e19, rel=1e-15)  # [cm^-3]


def test_nc_nv_scale_as_temperature_to_the_three_halves() -> None:
    ratio = (600.0 / 300.0) ** 1.5
    assert C.Nc(600.0) / C.Nc(300.0) == pytest.approx(ratio, rel=1e-14)
    assert C.Nv(600.0) / C.Nv(300.0) == pytest.approx(ratio, rel=1e-14)


def test_band_density_model_is_swappable_without_touching_call_sites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Phase 1 seam: swapping in computed Nc and Nv touches one object."""

    class FakeBandDensity:
        def Nc(self, T: float) -> float:
            return 1.0

        def Nv(self, T: float) -> float:
            return 2.0

    monkeypatch.setattr(C, "BAND_DENSITY", FakeBandDensity())
    assert C.Nc(300.0) == 1.0
    assert C.Nv(300.0) == 2.0


# --------------------------------------------------------- intrinsic density


def test_n_i_at_300k_is_exactly_1e10_by_decision() -> None:
    # docs/06-constants.md picks 1.0e10 over 9.65e9 and 1.45e10 so that
    # textbook worked examples remain valid analytic targets.
    assert C.n_i(300.0) == 1.0e10  # [cm^-3]


def test_n_i_increases_with_temperature() -> None:
    temperatures = [200.0, 300.0, 400.0, 500.0]
    densities = [C.n_i(T) for T in temperatures]
    assert all(a < b for a, b in zip(densities[:-1], densities[1:], strict=True))


def test_n_i_temperature_dependence_obeys_mass_action_shape() -> None:
    """n_i is anchored at 300 K but must still scale like the physics.

    n_i^2 / (Nc * Nv * exp(-Eg / V_T)) has to be independent of temperature.
    It is not equal to 1 here, because n_i is anchored to 1.0e10 rather than
    derived from Nc, Nv and Eg. See the known deviation in PROGRESS.md.
    """

    def group(T: float) -> float:
        return C.n_i(T) ** 2 / (C.Nc(T) * C.Nv(T) * math.exp(-C.Eg(T) / C.V_T(T)))

    reference = group(300.0)
    for T in (250.0, 350.0, 450.0, 600.0):
        assert group(T) == pytest.approx(reference, rel=1e-12)


def test_n_i_default_argument_is_300k() -> None:
    assert C.n_i() == C.n_i(300.0)


# ---------------------------------------------------------------- permittivity


def test_eps_si_is_11_7_times_eps_0() -> None:
    assert C.eps_Si() == pytest.approx(11.7 * C.eps_0, rel=1e-15)  # [F/cm]


def test_eps_ox_is_3_9_times_eps_0() -> None:
    assert C.eps_ox() == pytest.approx(3.9 * C.eps_0, rel=1e-15)  # [F/cm]


# ------------------------------------------------------------------ transport


def test_mobility_constants_match_doc() -> None:
    assert C.mu_n(300.0) == pytest.approx(1417.0, rel=1e-15)  # [cm^2/(V s)]
    assert C.mu_p(300.0) == pytest.approx(470.0, rel=1e-15)  # [cm^2/(V s)]


def test_saturation_velocities_match_doc() -> None:
    assert C.v_sat_n(300.0) == pytest.approx(1.07e7, rel=1e-15)  # [cm/s]
    assert C.v_sat_p(300.0) == pytest.approx(8.3e6, rel=1e-15)  # [cm/s]


def test_einstein_relation_holds_for_both_carriers() -> None:
    T = 350.0
    assert C.D_n(T) / C.mu_n(T) == pytest.approx(C.V_T(T), rel=1e-15)  # [V]
    assert C.D_p(T) / C.mu_p(T) == pytest.approx(C.V_T(T), rel=1e-15)  # [V]


# ------------------------------------------------------------ subthreshold floor


def test_ss_min_equals_v_t_times_ln_10() -> None:
    assert C.SS_min(300.0) == pytest.approx(0.059526, abs=1e-6)  # [V/decade]
    assert C.SS_min(400.0) == pytest.approx(C.V_T(400.0) * math.log(10.0), rel=1e-15)


# ------------------------------------------------------------------ structure


def test_no_temperature_dependent_value_is_a_module_level_constant() -> None:
    """Every temperature dependent quantity must be a function of T.

    A module level float would silently freeze the value at 300 K.
    """
    names = ("V_T", "Eg", "Nc", "Nv", "n_i", "SS_min", "mu_n", "mu_p", "D_n", "D_p")
    for name in names:
        assert callable(getattr(C, name)), f"{name} must be a function of T"
