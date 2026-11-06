from __future__ import annotations
import math
import numpy as np
import pytest
from ddsim.core import constants as C
from ddsim.mesh.mesh1d import graded_mesh_1d, uniform_mesh_1d

pytest.importorskip("atomsim")

from ddsim.physics.quantum import bound_states  # noqa: E402

HBAR = C.h / (2 * math.pi)
AIRY = (2.33811, 4.08795, 5.52056)


def test_an_infinite_square_well_has_n_squared_levels() -> None:
    L, m = 1e-6, 0.916
    x = uniform_mesh_1d(length=L, n_nodes=201).x
    s = bound_states(x, np.zeros_like(x), m, n_states=3, n_points=4000)
    for n, e in enumerate(s.energies, start=1):
        want = n**2 * math.pi**2 * HBAR**2 / (2 * m * C.m_0 * (L * 1e-2) ** 2) / C.q
        assert e == pytest.approx(want, rel=1e-5)


def test_a_triangular_well_has_the_airy_levels() -> None:
    F, m = 1e5, 0.916
    x = graded_mesh_1d(length=4e-6, n_nodes=401, refine_at=0.0, h_min=1e-9).x
    s = bound_states(x, F * x, m, n_states=3, n_points=8000)
    scale = (HBAR**2 / (2 * m * C.m_0)) ** (1 / 3) * (C.q * F * 1e2) ** (2 / 3) / C.q
    for e, a in zip(s.energies, AIRY, strict=True):
        assert e == pytest.approx(scale * a, rel=1e-4)


def test_the_wavefunctions_come_back_normalized_on_a_graded_mesh() -> None:
    x = graded_mesh_1d(length=4e-6, n_nodes=401, refine_at=0.0, h_min=1e-9).x
    s = bound_states(x, 1e5 * x, 0.19, n_states=3, n_points=8000)
    for p in s.psi:
        assert np.trapezoid(p**2, x) == pytest.approx(1.0, rel=1e-3)
        assert p[0] == 0.0 and abs(p[-1]) < 1e-6 * np.abs(p).max()
    assert abs(np.trapezoid(s.psi[0] * s.psi[1], x)) < 1e-3
