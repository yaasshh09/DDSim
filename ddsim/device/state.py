"""The solution of a device: psi, n and p, plus whatever produced them.

A Device is a specification and holds no solver state. A DeviceState is what a
solve returns, which is why solving is a function rather than a method.

All three fields are scaled and live on nodes. The quasi-Fermi potentials are
properties rather than stored values, because they are exactly redundant with
the densities:

    phi_n = psi - ln(n)        phi_p = psi + ln(p)

Storing both would let them drift apart, and a phi that disagrees with its own
density is a bug that produces a plausible looking answer. They are needed at
every Gummel cycle, because the nonlinear Poisson solve holds them fixed while
psi moves, and that substitution is what makes the cycle robust.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ddsim.core.field import Field, Location, ScalingState
from ddsim.solve.gummel import GummelResult
from ddsim.solve.newton import NewtonResult


def _quiet_log(density: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """ln of a density [1], without complaining about the zeros.

    A zero density is an insulator node, and the caller replaces the -inf it
    produces with nan. Only the divide is silenced, so a negative density,
    which would be a real failure, still warns on its way to nan.
    """
    with np.errstate(divide="ignore"):
        return np.asarray(np.log(density))


def _undefined_without_carriers(
    level: npt.NDArray[np.float64], density: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """A quasi-Fermi level [1], nan on the nodes that hold no carriers."""
    return np.asarray(np.where(density > 0.0, level, np.nan))


@dataclass(frozen=True)
class DeviceState:
    """A solution. All fields are scaled and live on nodes."""

    psi: Field
    """Electrostatic potential [V], scaled by V_T."""

    n: Field
    """Electron density [cm^-3], scaled by C_0."""

    p: Field
    """Hole density [cm^-3], scaled by C_0."""

    newton: NewtonResult | None = None
    """The Newton history of the last Poisson solve, including its residual
    tail. None if the state was not produced by a Poisson solve."""

    gummel: GummelResult[DeviceState] | None = None
    """The Gummel history, when the state came from a coupled solve. Its
    converged flag is the only honest way to judge a biased solution."""

    @property
    def phi_n(self) -> Field:
        """Electron quasi-Fermi potential [V], scaled. psi - ln(n).

        nan where there are no electrons at all, which is an insulator. The
        level is undefined there rather than large: a quasi-Fermi potential is
        the argument of a Boltzmann factor, and there is no carrier to take the
        factor of. ln(0) offers -inf, which would order against a real level
        and read as a band edge infinitely far away, so it is replaced. nan is
        the value that does not compare and does not travel quietly, which
        matters because device/transport.py hands these to the coupled solve.
        """
        return Field(
            _undefined_without_carriers(
                self.psi.data - _quiet_log(self.n.data), self.n.data
            ),
            "V",
            ScalingState.SCALED,
            Location.NODE,
            name="phi_n",
        )

    @property
    def phi_p(self) -> Field:
        """Hole quasi-Fermi potential [V], scaled. psi + ln(p).

        nan where there are no holes at all. See phi_n.
        """
        return Field(
            _undefined_without_carriers(
                self.psi.data + _quiet_log(self.p.data), self.p.data
            ),
            "V",
            ScalingState.SCALED,
            Location.NODE,
            name="phi_p",
        )

    def __repr__(self) -> str:
        return (
            f"DeviceState {self.psi.size} nodes "
            f"psi [{self.psi.data.min():.3g}, {self.psi.data.max():.3g}] "
            f"n [{self.n.data.min():.3g}, {self.n.data.max():.3g}]"
        )
