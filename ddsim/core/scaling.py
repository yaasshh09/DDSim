"""De Mari scaling, the reference quantities that make the system solvable.

Never work in raw physical units. A drift diffusion system in cm, V and cm^-3
spans roughly 25 orders of magnitude and the conditioning destroys any direct
solve. De Mari scaling maps the whole system onto order one numbers.

Reference quantities, from docs/02-numerics.md:

    psi_0 = V_T                                [V]
    C_0   = n_i, or max|net doping|            [cm^-3]
    x_0   = L_D = sqrt(eps * V_T / (q * C_0))  [cm]
    D_0   = max(Dn, Dp)                        [cm^2/s]
    mu_0  = D_0 / V_T                          [cm^2/(V s)]
    t_0   = x_0^2 / D_0                        [s]
    J_0   = q * D_0 * C_0 / x_0                [A/cm^2]
    R_0   = D_0 * C_0 / x_0^2                  [cm^-3 s^-1]

The payoff is that Poisson collapses to

    lap(psi) = -(p - n + N)

with no coefficient at all, and V_T vanishes from the Bernoulli argument in the
Scharfetter-Gummel flux because psi is already measured in units of V_T.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ddsim.core import constants as C


@dataclass(frozen=True)
class ScaleFactors:
    """An immutable set of de Mari reference quantities.

    Immutable on purpose. A scale factor that changes underneath a solve turns
    every field in the device into a silently wrong number.
    """

    T: float
    """Temperature [K]."""

    C_0: float
    """Reference concentration [cm^-3]. n_i by default."""

    eps: float
    """Permittivity of the reference material [F/cm]."""

    D_0: float
    """Reference diffusivity [cm^2/s]."""

    def __post_init__(self) -> None:
        if self.T <= 0.0:
            raise ValueError(f"T must be positive, got {self.T}")
        if self.C_0 <= 0.0:
            raise ValueError(f"C_0 must be positive, got {self.C_0}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}")
        if self.D_0 <= 0.0:
            raise ValueError(f"D_0 must be positive, got {self.D_0}")

    @classmethod
    def for_silicon(
        cls,
        T: float = C.T_ROOM,
        C_0: float | None = None,
        eps: float | None = None,
        D_0: float | None = None,
    ) -> ScaleFactors:
        """Silicon scale factors, C_0 = n_i(T) unless overridden.

        C_0 is a parameter rather than a constant so that switching to
        max|net doping| in a later phase is a single call site change. See the
        choice of C_0 discussion in docs/02-numerics.md.
        """
        return cls(
            T=T,
            C_0=C.n_i(T) if C_0 is None else C_0,
            eps=C.eps_Si() if eps is None else eps,
            D_0=max(C.D_n(T), C.D_p(T)) if D_0 is None else D_0,
        )

    # ------------------------------------------------------ derived quantities

    @property
    def psi_0(self) -> float:
        """Potential scale [V]. Equal to the thermal voltage."""
        return C.V_T(self.T)

    @property
    def x_0(self) -> float:
        """Length scale [cm]. The Debye length at C_0."""
        return math.sqrt(self.eps * self.psi_0 / (C.q * self.C_0))

    @property
    def mu_0(self) -> float:
        """Mobility scale [cm^2/(V s)]. D_0 / V_T by the Einstein relation."""
        return self.D_0 / self.psi_0

    @property
    def t_0(self) -> float:
        """Time scale [s]. The diffusion time across one Debye length."""
        return self.x_0 * self.x_0 / self.D_0

    @property
    def J_0(self) -> float:
        """Current density scale [A/cm^2]."""
        return C.q * self.D_0 * self.C_0 / self.x_0

    @property
    def R_0(self) -> float:
        """Recombination rate scale [cm^-3 s^-1]."""
        return self.D_0 * self.C_0 / (self.x_0 * self.x_0)

    # ------------------------------------------------------------ unit lookup

    def _registry(self) -> dict[str, float]:
        """Maps a unit string to the factor that converts physical to scaled.

        Deliberately explicit and closed. An unrecognised unit raises rather
        than passing through with a factor of 1, because a silent pass through
        is exactly the bug this module exists to prevent.
        """
        return {
            "V": self.psi_0,
            "cm^-3": self.C_0,
            "cm": self.x_0,
            "cm^2/s": self.D_0,
            "cm^2/(V s)": self.mu_0,
            "s": self.t_0,
            "A/cm^2": self.J_0,
            "cm^-3 s^-1": self.R_0,
            "1": 1.0,
        }

    def factor(self, unit: str) -> float:
        """The scale factor for a unit string.

        physical = scaled * factor, scaled = physical / factor.
        """
        registry = self._registry()
        if unit not in registry:
            known = ", ".join(sorted(registry))
            raise KeyError(f"unknown unit {unit!r}. Known units are: {known}")
        return registry[unit]

    def to_scaled(self, value: float | np.ndarray, unit: str) -> float | np.ndarray:
        """Convert a physical value to scaled, dispatching on the unit string."""
        divisor = self.factor(unit)
        if isinstance(value, np.ndarray):
            return value / divisor
        return float(value) / divisor

    def to_physical(self, value: float | np.ndarray, unit: str) -> float | np.ndarray:
        """Convert a scaled value to physical, dispatching on the unit string."""
        multiplier = self.factor(unit)
        if isinstance(value, np.ndarray):
            return value * multiplier
        return float(value) * multiplier
