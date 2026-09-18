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

They compose by multiplication too, and that is what makes a source implant
expressible without a new class for it. An implant is separable: a lateral
window times a vertical Gaussian. Along says which axis a one dimensional
shape reads, and the product of the two is the implant.

    source = Along(window, "x") * Along(depth, "y") * 1e20

A profile is asked for a value at a Coordinates, which carries x and, when the
mesh has one, y. A bare array is still a position and still means x, so every
device built before Phase 5 calls a profile exactly as it always did and gets
the same array back, bit for bit.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.special import erfc as _erfc

Axis = Literal["x", "y"]
"""Which coordinate a one dimensional shape reads."""


@dataclass(frozen=True)
class Coordinates:
    """Where a profile is being asked for a value [cm].

    One array per axis, both indexed by node, because x and y are two
    coordinates of the same set of nodes rather than two independent sweeps.
    A 1D mesh is a line along x and has no y at all, which is None here and
    not an array of zeros: a depth profile on a line is a modelling mistake,
    and zeros would hide it by reading the peak everywhere.
    """

    x: npt.NDArray[np.float64]
    """Position along the device [cm]."""

    y: npt.NDArray[np.float64] | None = None
    """Depth into the device [cm], or None on a mesh that has no depth."""

    def __post_init__(self) -> None:
        if self.y is not None and self.y.size != self.x.size:
            raise ValueError(
                f"x and y must cover the same number of positions, got "
                f"{self.x.size} and {self.y.size}. They are two coordinates "
                "of one set of nodes, so a mismatch is two meshes mixed."
            )

    @classmethod
    def of(cls, at: Position) -> Coordinates:
        """Whatever a caller passed, as Coordinates.

        A bare number or array is a position along x, which is the calling
        convention every profile had before there was a second axis.
        """
        if isinstance(at, Coordinates):
            return at
        return cls(np.asarray(at, dtype=np.float64))

    def axis(self, name: Axis) -> npt.NDArray[np.float64]:
        """The named coordinate [cm]."""
        if name == "x":
            return self.x
        if name == "y":
            if self.y is None:
                raise ValueError(
                    "this profile reads the depth of the device, but it was "
                    "evaluated somewhere with no y coordinate. A 1D mesh is a "
                    "line along x."
                )
            return self.y
        raise ValueError(f"an axis is x or y, got {name!r}")


Position = float | npt.NDArray[np.float64] | Coordinates


class DopingProfile(ABC):
    """Net doping [cm^-3] as a function of position [cm]."""

    @abstractmethod
    def __call__(self, at: Position) -> npt.NDArray[np.float64]:
        """Net doping Nd - Na [cm^-3] at the positions `at` [cm]."""

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

    def __mul__(self, other: DopingProfile | float) -> DopingProfile:
        """A product of profiles, or a profile scaled by a number.

        Both readings of * are wanted and neither is ambiguous. A separable
        implant is a product of shapes, and a shape times its peak
        concentration is that shape scaled.
        """
        if isinstance(other, DopingProfile):
            return Product((self, other))
        if isinstance(other, int | float):
            return Scaled(self, float(other))
        raise TypeError(
            f"can only multiply a DopingProfile by a DopingProfile or a "
            f"number, got {type(other).__name__}."
        )

    __rmul__ = __mul__


@dataclass(frozen=True)
class Uniform(DopingProfile):
    """Constant doping [cm^-3]. Negative for acceptors."""

    value: float

    def __call__(self, at: Position) -> npt.NDArray[np.float64]:
        """Net doping [cm^-3] at the positions `at` [cm]."""
        return np.full_like(Coordinates.of(at).x, self.value)


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

    def __call__(self, at: Position) -> npt.NDArray[np.float64]:
        """Net doping [cm^-3] at the positions `at` [cm]."""
        values = Coordinates.of(at).x
        return np.where(values < self.position, self.left, self.right)


@dataclass(frozen=True)
class Layers(DopingProfile):
    """Constant doping in each of several regions laid end to end along x.

    Step generalised to any number of junctions, for a 1D stack of doped
    regions. Each value is picked rather than built by adding steps, so a
    region holds exactly the number it was given: a sum of steps would leave
    a region at left + (right - left), which is not always right to the last
    bit. Right continuous at every boundary, the same choice Step makes.
    """

    boundaries: tuple[float, ...]
    """Where each region ends and the next begins [cm], increasing."""

    values: tuple[float, ...]
    """Net doping in each region [cm^-3], one more than there are boundaries."""

    def __post_init__(self) -> None:
        if len(self.values) != len(self.boundaries) + 1:
            raise ValueError(
                f"layers need one more value than boundaries, got "
                f"{len(self.values)} values and {len(self.boundaries)} boundaries"
            )
        if any(np.diff(self.boundaries) <= 0.0):
            raise ValueError(
                f"layer boundaries must be increasing, got {self.boundaries}"
            )

    def __call__(self, at: Position) -> npt.NDArray[np.float64]:
        """Net doping [cm^-3] at the positions `at` [cm]."""
        region = np.searchsorted(self.boundaries, Coordinates.of(at).x, side="right")
        return np.asarray(self.values, dtype=np.float64)[region]


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

    def __call__(self, at: Position) -> npt.NDArray[np.float64]:
        """Net doping [cm^-3] at the positions `at` [cm]."""
        offset = Coordinates.of(at).x - self.centre
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

    def __call__(self, at: Position) -> npt.NDArray[np.float64]:
        """Net doping [cm^-3] at the positions `at` [cm]."""
        values = Coordinates.of(at).x
        return np.asarray(self.peak * _erfc((values - self.position) / self.length))


WindowEdge = Literal["abrupt", "gaussian", "erfc"]
"""How a window falls off outside its edges."""


@dataclass(frozen=True)
class Window(DopingProfile):
    """One inside [low, high], and falling off outside it, along x.

    One axis of a drawn implant's rectangle. Three edges:

    - "abrupt": one inside, both edges included, zero outside.
    - "gaussian": one inside, exp(-d^2 / (2 length^2)) at a distance d
      outside. The depth of an implant: the peak holds over the drawn
      rectangle and the straggle carries it past.
    - "erfc": 0.5 (erfc((x - high)/length) - erfc((x - low)/length)), which
      is the window blurred by a Gaussian, half the peak at each edge. The
      side of an implant, spreading under a mask edge. Not flat inside: a
      window narrower than a few lengths never reaches its peak, which is
      what a slit in a mask does.

    An infinite edge has nothing to fall off from, and it is how a drawn
    rectangle flush with the device boundary is carried, since the implant
    does not stop where the drawing does. A window open on one side is then
    one term, written the way nmos writes its source, so a drawn MOSFET's
    lateral factor is nmos's own to the last bit.
    """

    low: float
    """Lower edge [cm], or -inf."""

    high: float
    """Upper edge [cm], or +inf."""

    edge: WindowEdge = "abrupt"
    """How the window falls off outside."""

    length: float = 0.0
    """The fall off [cm]: the Gaussian's sigma, or the erfc's length. Unused
    by an abrupt window."""

    def __post_init__(self) -> None:
        if self.edge not in ("abrupt", "gaussian", "erfc"):
            raise ValueError(
                f"a window edge is abrupt, gaussian or erfc, got {self.edge!r}"
            )
        if not self.low <= self.high:
            raise ValueError(
                f"a window needs low <= high, got low={self.low:g} and "
                f"high={self.high:g}"
            )
        if self.edge != "abrupt" and not self.length > 0.0:
            raise ValueError(
                f"a {self.edge} window needs a positive length, got {self.length:g}"
            )

    def __call__(self, at: Position) -> npt.NDArray[np.float64]:
        """The window's value [1] at the positions `at` [cm]."""
        x = Coordinates.of(at).x
        if self.edge == "abrupt":
            return np.where((x >= self.low) & (x <= self.high), 1.0, 0.0)
        if self.edge == "gaussian":
            outside = np.maximum(np.maximum(self.low - x, x - self.high), 0.0)
            return np.asarray(np.exp(-(outside**2) / (2.0 * self.length**2)))
        # Open on the right, the general form would be 0.5 (2 - erfc), which
        # cancels to nothing in the tail. Open on the left needs no case: its
        # second term is erfc(+inf) = 0 exactly, which leaves nmos's source.
        if math.isinf(self.high) and not math.isinf(self.low):
            return np.asarray(0.5 * _erfc((self.low - x) / self.length))
        return np.asarray(
            0.5
            * (
                _erfc((x - self.high) / self.length)
                - _erfc((x - self.low) / self.length)
            )
        )


@dataclass(frozen=True)
class Sum(DopingProfile):
    """Several profiles added together. Built by the + operator."""

    terms: tuple[DopingProfile, ...]

    def __call__(self, at: Position) -> npt.NDArray[np.float64]:
        """Net doping [cm^-3] at the positions `at` [cm]."""
        total = np.zeros_like(Coordinates.of(at).x)
        for term in self.terms:
            total = total + term(at)
        return total


@dataclass(frozen=True)
class Scaled(DopingProfile):
    """A profile multiplied by a constant.

    Built by the unary minus operator, and by multiplying a shape by its peak
    concentration.
    """

    profile: DopingProfile
    factor: float

    def __call__(self, at: Position) -> npt.NDArray[np.float64]:
        """Net doping [cm^-3] at the positions `at` [cm]."""
        return np.asarray(self.factor * self.profile(at))


@dataclass(frozen=True)
class Product(DopingProfile):
    """Several profiles multiplied together. Built by the * operator.

    The user is a separable implant: a lateral window times a vertical
    Gaussian is a source, and neither factor needs to know about the other.
    """

    factors: tuple[DopingProfile, ...]

    def __call__(self, at: Position) -> npt.NDArray[np.float64]:
        """Net doping [cm^-3] at the positions `at` [cm]."""
        values = self.factors[0](at)
        for factor in self.factors[1:]:
            values = values * factor(at)
        return np.asarray(values)


@dataclass(frozen=True)
class Mirrored(DopingProfile):
    """A profile reflected about a position along x.

    A drain is a source mirrored, and saying it that way is both shorter than
    writing the implant out twice and more accurate: the two are the same
    implant through the same mask, so anything that changes one changes the
    other.

    Only x is reflected. A device is turned end for end about its centre, not
    upside down, so a mirrored implant sits at the same depth.
    """

    profile: DopingProfile
    """The profile being reflected."""

    about: float
    """The position reflected about [cm], usually the centre of the device."""

    def __call__(self, at: Position) -> npt.NDArray[np.float64]:
        """Net doping [cm^-3] at the positions `at` [cm]."""
        here = Coordinates.of(at)
        return self.profile(Coordinates(2.0 * self.about - here.x, here.y))


@dataclass(frozen=True)
class Along(DopingProfile):
    """A one dimensional shape, read along a named axis.

    Every profile above reads x, because that is all a profile was ever handed
    before there was a second axis. Along re-labels which coordinate the shape
    is a function of and does nothing else, so `Along(shape, "x")` is `shape`.
    """

    profile: DopingProfile
    """The shape."""

    axis: Axis
    """Which coordinate it is a function of."""

    def __call__(self, at: Position) -> npt.NDArray[np.float64]:
        """Net doping [cm^-3] at the positions `at` [cm]."""
        return self.profile(Coordinates.of(at).axis(self.axis))


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
