from __future__ import annotations
import pytest
from ddsim.core import constants as C
from ddsim.device.builder import Material
from ddsim.device.mosfet import nmos
from ddsim.device.transport import TransportModels
from ddsim.physics.mobility import LombardiSurface
from ddsim.physics.statistics import Degeneracy


def test_a_400_K_device_hands_400_K_to_every_model() -> None:
    T = 400.0
    d = nmos(material=Material.silicon(T))
    m = TransportModels.for_device(d, mobility='arora', field_dependent=True, surface=True)
    sc = d.scale
    assert sc.T == T and d.material.n_i == C.n_i(T)
    assert d.degeneracy == Degeneracy.for_silicon(sc.C_0, T)
    assert m.surface is not None
    assert m.surface.electrons == LombardiSurface.electrons(T) and m.surface.holes == LombardiSurface.holes(T)
    assert m.Dn.v_sat == pytest.approx(1.07e7 * (T / 300.0) ** -0.87 * sc.x_0 / sc.D_0, rel=1e-12)
    assert m.Dp.v_sat == pytest.approx(8.3e6 * (T / 300.0) ** -0.52 * sc.x_0 / sc.D_0, rel=1e-12)


def test_saturation_velocity_is_unchanged_at_300_K() -> None:
    assert C.v_sat_n(300.0) == 1.07e7 and C.v_sat_p(300.0) == 8.3e6
