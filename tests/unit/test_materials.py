from __future__ import annotations
import math
import pytest
from ddsim.core import constants as C
from ddsim.physics.bands import MATERIALS, Semiconductor


def intrinsic(m: Semiconductor, T: float = C.T_ROOM) -> float:
    return math.sqrt(m.Nc * m.Nv) * math.exp(-m.Eg / (2.0 * C.V_T(T)))


def test_sige_with_no_germanium_is_silicon_field_for_field() -> None:
    s, x = Semiconductor.silicon(), Semiconductor.sige(0.0)
    assert (x.eps_r, x.chi, x.Eg, x.Nc, x.Nv) == (s.eps_r, s.chi, s.Eg, s.Nc, s.Nv)


def test_sige_follows_the_ioffe_composition_formulas() -> None:
    s = Semiconductor.silicon()
    for x in (0.1, 0.2, 0.3):
        m = Semiconductor.sige(x)
        assert m.Eg - s.Eg == pytest.approx(-0.41 * x + 0.008 * x * x, rel=1e-12, abs=0)
        assert m.eps_r == pytest.approx(11.7 + 4.5 * x, rel=1e-12, abs=0)
        assert m.chi == s.chi and m.Nc == s.Nc and m.Nv == s.Nv


@pytest.mark.parametrize("x", [-0.01, 0.85, 1.0])
def test_sige_refuses_compositions_outside_the_si_like_range(x: float) -> None:
    with pytest.raises(ValueError):
        Semiconductor.sige(x)


@pytest.mark.parametrize("make,Eg,chi,eps_r,Nc,Nv", [
    (Semiconductor.germanium, 0.661, 4.0, 16.2, 1.0e19, 5.0e18),
    (Semiconductor.gaas, 1.424, 4.07, 12.9, 4.7e17, 9.0e18),
])
def test_the_table_reads_back_the_ioffe_values_at_300_k(make, Eg: float, chi: float, eps_r: float, Nc: float, Nv: float) -> None:
    m = make()
    assert (m.chi, m.eps_r, m.Nc, m.Nv) == (chi, eps_r, Nc, Nv)
    assert m.Eg == pytest.approx(Eg, abs=2e-3)


@pytest.mark.parametrize("make,ni", [(Semiconductor.germanium, 2.0e13), (Semiconductor.gaas, 2.1e6)])
def test_nc_nv_and_eg_imply_the_intrinsic_density_ioffe_prints(make, ni: float) -> None:
    assert intrinsic(make()) == pytest.approx(ni, rel=0.12)


def test_band_densities_scale_as_t_to_the_three_halves() -> None:
    a, b = Semiconductor.germanium(300.0), Semiconductor.germanium(400.0)
    assert b.Nc / a.Nc == pytest.approx((4.0 / 3.0) ** 1.5, rel=1e-12)
    assert b.Nv / a.Nv == pytest.approx((4.0 / 3.0) ** 1.5, rel=1e-12)
    assert b.Eg < a.Eg


def test_every_table_entry_builds_and_silicon_is_the_default() -> None:
    assert set(MATERIALS) >= {"silicon", "germanium", "gaas"}
    for name, make in MATERIALS.items():
        assert make(C.T_ROOM).name == name
    assert MATERIALS["silicon"](C.T_ROOM) == Semiconductor.silicon()
