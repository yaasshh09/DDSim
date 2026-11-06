from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from ddsim.core import constants as C

HBAR = C.h / (2 * math.pi)
BOHR = 4 * math.pi * C.eps_0 * 1e2 * HBAR**2 / (C.m_0 * C.q**2) * 1e2
HARTREE = HBAR**2 / (C.m_0 * (BOHR * 1e-2) ** 2) / C.q


@dataclass(frozen=True)
class Subbands:
    energies: tuple[float, ...]

    psi: npt.NDArray[np.float64]


def bound_states(x: npt.NDArray[np.float64], V: npt.NDArray[np.float64], mass_ratio: float, n_states: int = 3, n_points: int = 4000) -> Subbands:
    from atomsim.numerics.radial_solver import solve_radial
    p = np.asarray(x, dtype=np.float64)
    v = np.asarray(V, dtype=np.float64)
    if p.shape != v.shape or p.ndim != 1:
        raise ValueError(f"x and V must be one array each of the same length, got {p.shape} and {v.shape}")
    if np.any(np.diff(p) <= 0.0):
        raise ValueError("x must increase, since its two ends are the hard walls of the well")
    L = float(p[-1] - p[0])
    s = solve_radial(lambda r: np.interp(r * BOHR, p - p[0], v) / HARTREE, l=0, mu_ratio=mass_ratio, r_max=L / BOHR, n_points=n_points, n_states=n_states)
    r = np.concatenate([[0.0], s.r * BOHR, [L]])
    psi = np.array([np.interp(p - p[0], r, np.concatenate([[0.0], u, [0.0]])) for u in s.u]) / math.sqrt(BOHR)
    return Subbands(energies=tuple(e.value * HARTREE for e in s.energies), psi=psi)
