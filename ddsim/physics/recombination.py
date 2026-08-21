"""Shockley-Read-Hall recombination, with the Scharfetter doping dependence.

Pure functions over arrays, unit agnostic. The same expressions serve physical
units (cm^-3 and seconds) and de Mari scaled units, as long as n_i^2, n1 and p1
are passed in the same system as the densities. The defaults are the scaled
ones for C_0 = n_i, where all three are 1.

From docs/01-physics.md:

    R = (n*p - n_i^2) / (tau_p * (n + n1) + tau_n * (p + p1))

with n1 = p1 = n_i for a midgap trap. R > 0 is net recombination, R < 0 is net
generation, which is what a reverse biased depletion region does.

Two things about this expression are worth stating before writing any solver
code around it.

**The numerator has to cancel exactly at equilibrium.** n*p = n_i^2 must give
0.0, not 1e-30. A residual generation rate at zero bias is a current with no
cause, and it shows up as a floor under the reverse saturation current that no
mesh refinement removes.

**tau_p pairs with n, not with p.** In strongly n-type material the
denominator is dominated by tau_p * n and the rate reduces to excess_p / tau_p.
That is the physical content of the minority carrier lifetime. Swapping them
gives an expression that is still symmetric and still dimensionally correct and
is wrong by the ratio of the two lifetimes.

Linearization for Gummel
------------------------
The electron continuity equation is solved with psi and p held fixed, and it is
only linear in n if R is. Writing the rate as

    R = c * n - g,     c = p / D,   g = n_i^2 / D,   D the denominator

with c and g frozen at the current iterate is exact at that iterate, so the
converged fixed point solves the true equation rather than a linearized
substitute. It has one property the exact tangent dR/dn does not: both c and g
are non-negative at every density. The continuity matrix is an M-matrix, so its
inverse is non-negative, and a non-negative right hand side then guarantees a
non-negative solution. Carrier densities cannot go negative out of this
linearization, which matters because docs/05-pitfalls.md is emphatic that
clamping a negative density is never the right repair.

The exact derivatives are implemented too. They are what the full Newton
Jacobian in Phase 3 needs, and both are strictly positive, so they strengthen
the diagonal rather than threatening it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from ddsim.core import constants as C

Density = float | npt.NDArray[np.float64]
"""A carrier density, scalar or per node. Units are the caller's choice."""

Lifetime = float | npt.NDArray[np.float64]
"""A carrier lifetime, scalar or per node, in the caller's units."""


# ------------------------------------------------------- Scharfetter lifetime


def scharfetter_lifetime(
    N_total: Density,
    *,
    tau_max: float,
    tau_min: float = 0.0,
    N_ref: float = C.N_REF_SRH,
    gamma: float = C.GAMMA_SRH,
) -> npt.NDArray[np.float64]:
    """Doping dependent carrier lifetime [s].

        tau = tau_min + (tau_max - tau_min) / (1 + (N_total / N_ref)^gamma)

    Args:
        N_total: total doping Na + Nd [cm^-3], never a net doping.
        tau_max: lifetime in undoped material [s].
        tau_min: lifetime at very high doping [s].
        N_ref: doping at which tau sits halfway between the two [cm^-3].
        gamma: sharpness of the transition [1].

    Raises on negative N_total. Passing net doping here is an easy mistake and
    it would give a longer lifetime on the p side of a symmetric junction than
    on the n side, which is silently wrong rather than obviously wrong.
    """
    total = np.asarray(N_total, dtype=np.float64)
    if np.any(total < 0.0):
        raise ValueError(
            "N_total is a total doping Na + Nd and cannot be negative. "
            "Pass abs(net_doping) if that is what you have."
        )
    if tau_max < tau_min:
        raise ValueError(f"tau_max={tau_max} is below tau_min={tau_min}")
    if N_ref <= 0.0:
        raise ValueError(f"N_ref must be positive, got {N_ref}")

    return np.asarray(
        tau_min + (tau_max - tau_min) / (1.0 + (total / N_ref) ** gamma)
    )


# ------------------------------------------------------------------- the rate


def _denominator(
    n: Density,
    p: Density,
    tau_n: Lifetime,
    tau_p: Lifetime,
    n1: float,
    p1: float,
) -> Density:
    """tau_p*(n + n1) + tau_n*(p + p1), the SRH denominator.

    Strictly positive for positive densities and lifetimes, so the rate never
    divides by zero. Dtype preserving, so complex step differentiation works
    through it.
    """
    return tau_p * (n + n1) + tau_n * (p + p1)


def srh_rate(
    n: Density,
    p: Density,
    tau_n: Lifetime,
    tau_p: Lifetime,
    ni2: float = 1.0,
    n1: float = 1.0,
    p1: float = 1.0,
) -> Density:
    """Net SRH recombination rate [cm^-3 s^-1], or scaled [1].

    Args:
        n: electron density.
        p: hole density.
        tau_n: electron lifetime.
        tau_p: hole lifetime.
        ni2: n_i^2 in the same units as n*p. 1.0 in scaled units with C_0 = n_i.
        n1: trap level electron density, n_i for a midgap trap.
        p1: trap level hole density, n_i for a midgap trap.

    Positive means net recombination. Dtype preserving, so passing a complex n
    gives a complex rate and complex step differentiation works directly on
    this function, which is how the derivatives below are verified.
    """
    return (n * p - ni2) / _denominator(n, p, tau_n, tau_p, n1, p1)


def d_srh_dn(
    n: Density,
    p: Density,
    tau_n: Lifetime,
    tau_p: Lifetime,
    ni2: float = 1.0,
    n1: float = 1.0,
    p1: float = 1.0,
) -> Density:
    """dR/dn, the exact tangent [1/s] or scaled [1].

        dR/dn = (p*D - (n*p - n_i^2)*tau_p) / D^2

    Strictly positive. Expanding the numerator gives
    tau_p*p*n1 + tau_n*p^2 + tau_n*p*p1 + tau_p*n_i^2, four positive terms, so
    there is no density at which this changes sign.
    """
    denominator = _denominator(n, p, tau_n, tau_p, n1, p1)
    return (p * denominator - (n * p - ni2) * tau_p) / (denominator * denominator)


def d_srh_dp(
    n: Density,
    p: Density,
    tau_n: Lifetime,
    tau_p: Lifetime,
    ni2: float = 1.0,
    n1: float = 1.0,
    p1: float = 1.0,
) -> Density:
    """dR/dp, the exact tangent [1/s] or scaled [1]. Also strictly positive."""
    denominator = _denominator(n, p, tau_n, tau_p, n1, p1)
    return (n * denominator - (n * p - ni2) * tau_n) / (denominator * denominator)


def srh_electron_linearization(
    n: Density,
    p: Density,
    tau_n: Lifetime,
    tau_p: Lifetime,
    ni2: float = 1.0,
    n1: float = 1.0,
    p1: float = 1.0,
) -> tuple[Density, Density]:
    """(c, g) such that R = c*n - g, with the denominator frozen.

    c = p/D is a recombination coefficient [1/s] and g = n_i^2/D is a
    generation rate. Both are non-negative, which is what keeps the electron
    density positive through the Gummel solve. See the module docstring.
    """
    denominator = _denominator(n, p, tau_n, tau_p, n1, p1)
    return p / denominator, ni2 / denominator


def srh_hole_linearization(
    n: Density,
    p: Density,
    tau_n: Lifetime,
    tau_p: Lifetime,
    ni2: float = 1.0,
    n1: float = 1.0,
    p1: float = 1.0,
) -> tuple[Density, Density]:
    """(c, g) such that R = c*p - g, with the denominator frozen."""
    denominator = _denominator(n, p, tau_n, tau_p, n1, p1)
    return n / denominator, ni2 / denominator


# ------------------------------------------------------------------- models


class RecombinationModel(Protocol):
    """What the continuity assembly needs from a recombination model.

    A protocol rather than a base class, so that Auger, or a sum of models,
    can be dropped in later without this module knowing about them.
    """

    def rate(self, n: Density, p: Density) -> Density:
        """Net recombination rate, positive for net recombination."""
        ...

    def d_rate_dn(self, n: Density, p: Density) -> Density:
        """Exact dR/dn, for the Phase 3 Newton Jacobian."""
        ...

    def d_rate_dp(self, n: Density, p: Density) -> Density:
        """Exact dR/dp, for the Phase 3 Newton Jacobian."""
        ...

    def electron_linearization(
        self, n: Density, p: Density
    ) -> tuple[Density, Density]:
        """(c, g) with R = c*n - g, both non-negative."""
        ...

    def hole_linearization(self, n: Density, p: Density) -> tuple[Density, Density]:
        """(c, g) with R = c*p - g, both non-negative."""
        ...


@dataclass(frozen=True)
class SRHRecombination:
    """SRH with fixed lifetimes, which may vary per node.

    Everything is in one unit system, the caller's. The solver builds this in
    scaled units, with lifetimes divided by t_0 and ni2, n1 and p1 measured
    against C_0.
    """

    tau_n: Lifetime
    """Electron lifetime [s], or scaled [1]."""

    tau_p: Lifetime
    """Hole lifetime [s], or scaled [1]."""

    ni2: float = 1.0
    """n_i^2 in the units of n*p [1]. Not hardcoded, because C_0 need not
    be n_i."""

    n1: float = 1.0
    """Trap level electron density [1]. n_i for a midgap trap."""

    p1: float = 1.0
    """Trap level hole density [1]. n_i for a midgap trap."""

    def rate(self, n: Density, p: Density) -> Density:
        """Net recombination rate, positive for net recombination."""
        return srh_rate(n, p, self.tau_n, self.tau_p, self.ni2, self.n1, self.p1)

    def d_rate_dn(self, n: Density, p: Density) -> Density:
        """Exact dR/dn."""
        return d_srh_dn(n, p, self.tau_n, self.tau_p, self.ni2, self.n1, self.p1)

    def d_rate_dp(self, n: Density, p: Density) -> Density:
        """Exact dR/dp."""
        return d_srh_dp(n, p, self.tau_n, self.tau_p, self.ni2, self.n1, self.p1)

    def electron_linearization(
        self, n: Density, p: Density
    ) -> tuple[Density, Density]:
        """(c, g) with R = c*n - g."""
        return srh_electron_linearization(
            n, p, self.tau_n, self.tau_p, self.ni2, self.n1, self.p1
        )

    def hole_linearization(self, n: Density, p: Density) -> tuple[Density, Density]:
        """(c, g) with R = c*p - g."""
        return srh_hole_linearization(
            n, p, self.tau_n, self.tau_p, self.ni2, self.n1, self.p1
        )


@dataclass(frozen=True)
class NoRecombination:
    """R = 0 everywhere.

    Not a convenience. It is the configuration under which Jn and Jp are each
    separately constant across the device, which is the primary correctness
    gate for Phase 2 in phases/PHASE-2.md.
    """

    def _zeros(self, n: Density, p: Density) -> npt.NDArray[np.float64]:
        """Zeros of the shape n and p broadcast to.

        One value per node rather than a bare 0.0, because a scalar would
        broadcast correctly into the residual and then silently collapse the
        Jacobian diagonal contribution.
        """
        return np.zeros(np.broadcast_shapes(np.shape(n), np.shape(p)))

    def rate(self, n: Density, p: Density) -> Density:
        """Zero."""
        return self._zeros(n, p)

    def d_rate_dn(self, n: Density, p: Density) -> Density:
        """Zero."""
        return self._zeros(n, p)

    def d_rate_dp(self, n: Density, p: Density) -> Density:
        """Zero."""
        return self._zeros(n, p)

    def electron_linearization(
        self, n: Density, p: Density
    ) -> tuple[Density, Density]:
        """Zero coefficient and zero generation."""
        return self._zeros(n, p), self._zeros(n, p)

    def hole_linearization(self, n: Density, p: Density) -> tuple[Density, Density]:
        """Zero coefficient and zero generation."""
        return self._zeros(n, p), self._zeros(n, p)
