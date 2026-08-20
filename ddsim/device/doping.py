"""Doping profiles, as callables of position rather than arrays.

docs/03-architecture.md is explicit about why they are callables: Phase 5
refines the mesh adaptively, so a profile has to stay re-evaluable on a mesh
that does not exist yet. Freezing a profile into an array at construction time
would make adaptive refinement impossible without regenerating the device.

Sign convention from the notation table in docs/01-physics.md: net doping is
Nd - Na, so donors are positive and acceptors negative. Every profile here
returns net doping [cm^-3] for a position [cm].

Profiles compose by addition, which is how real structures are built: a
uniform substrate plus a diffused well plus an implanted source.

    profile = Uniform(-1e16) + Gaussian(peak=1e18, centre=0.0, sigma=5e-6)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.special import erfc as _erfc

Position = float | npt.NDArray[np.float64]


class DopingProfile(ABC):
    """Net doping [cm^-3] as a function of position [cm]."""

    @abstractmethod
    def __call__(self, x: Position) -> npt.NDArray[np.float64]:
        """Net doping Nd - Na [cm^-3] at position x [cm]."""

    def __add__(self, other: DopingProfile) -> DopingProfile:
        if not isinstance(other, DopingProfile):
            raise TypeError(
                f"can only add a DopingProfile to a DopingProfile, "
                f"got {type(other).__name__}. A bare number has no position "
                "dependence, wrap it in Uniform."
            )
        return Sum((self, other))

    def __neg__(self) -> DopingProfile:
        return Scaled(self, -1.0)

    def __sub__(self, other: DopingProfile) -> DopingProfile:
        return self.__add__(-other)


@dataclass(frozen=True)
class Uniform(DopingProfile):
    """Constant doping [cm^-3]. Negative for acceptors."""

    value: float

    def __call__(self, x: Position) -> npt.NDArray[np.float64]:
        """Net doping [cm^-3] at position x [cm]."""
        return np.full_like(np.asarray(x, dtype=np.float64), self.value)


@dataclass(frozen=True)
class Step(DopingProfile):
    """An abrupt change from one constant level to another.

    Right continuous: a node sitting exactly on the junction takes the right
    value. Arbitrary, but it has to be decided somewhere.
    """

    left: float
    """Net doping below the junction [cm^-3]."""

    right: float
    """Net doping at and above the junction [cm^-3]."""

    position: float
    """Junction position [cm]."""

    def __call__(self, x: Position) -> npt.NDArray[np.float64]:
        """Net doping [cm^-3] at position x [cm]."""
        values = np.asarray(x, dtype=np.float64)
        return np.where(values < self.position, self.left, self.right)


@dataclass(frozen=True)
class Gaussian(DopingProfile):
    """An implanted profile, peak * exp(-(x - centre)^2 / (2 sigma^2))."""

    peak: float
    """Peak concentration [cm^-3]."""

    centre: float
    """Position of the peak [cm]."""

    sigma: float
    """Standard deviation [cm]."""

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")

    def __call__(self, x: Position) -> npt.NDArray[np.float64]:
        """Net doping [cm^-3] at position x [cm]."""
        values = np.asarray(x, dtype=np.float64)
        offset = values - self.centre
        return np.asarray(
            self.peak * np.exp(-(offset**2) / (2.0 * self.sigma**2))
        )


@dataclass(frozen=True)
class Erfc(DopingProfile):
    """A diffused profile, peak * erfc((x - position) / length).

    length is 2*sqrt(D*t) for a constant source diffusion.
    """

    peak: float
    """Surface concentration [cm^-3]."""

    position: float
    """Where the profile starts [cm]."""

    length: float
    """Characteristic diffusion length [cm]."""

    def __post_init__(self) -> None:
        if self.length <= 0.0:
            raise ValueError(f"length must be positive, got {self.length}")

    def __call__(self, x: Position) -> npt.NDArray[np.float64]:
        """Net doping [cm^-3] at position x [cm]."""
        values = np.asarray(x, dtype=np.float64)
        return np.asarray(self.peak * _erfc((values - self.position) / self.length))


@dataclass(frozen=True)
class Sum(DopingProfile):
    """Several profiles added together. Built by the + operator."""

    terms: tuple[DopingProfile, ...]

    def __call__(self, x: Position) -> npt.NDArray[np.float64]:
        """Net doping [cm^-3] at position x [cm]."""
        total = np.zeros_like(np.asarray(x, dtype=np.float64))
        for term in self.terms:
            total = total + term(x)
        return total


@dataclass(frozen=True)
class Scaled(DopingProfile):
    """A profile multiplied by a constant. Built by the unary minus operator."""

    profile: DopingProfile
    factor: float

    def __call__(self, x: Position) -> npt.NDArray[np.float64]:
        """Net doping [cm^-3] at position x [cm]."""
        return np.asarray(self.factor * self.profile(x))


def abrupt_junction(Na: float, Nd: float, position: float) -> DopingProfile:
    """An abrupt PN junction, p-type on the left and n-type on the right.

    Args:
        Na: acceptor concentration on the left [cm^-3], given positive.
        Nd: donor concentration on the right [cm^-3], given positive.
        position: junction position [cm].

    Na and Nd are concentrations, so both are passed as positive magnitudes.
    The sign convention is applied here, once.
    """
    if Na <= 0.0:
        raise ValueError(f"Na must be positive, got {Na}")
    if Nd <= 0.0:
        raise ValueError(f"Nd must be positive, got {Nd}")
    return Step(left=-Na, right=Nd, position=position)
