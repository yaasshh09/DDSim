"""The Field type, the defence against silently mixing scaled and physical data.

Every array of physical numbers in this codebase carries three pieces of
metadata: a unit string, a scaling state, and a mesh location. Arithmetic that
mixes any of them raises rather than coercing.

Why this matters more here than anywhere else: de Mari scaling means every
quantity exists in two numerically plausible forms. A scaled potential of 38.7
and a physical potential of 1.0 V are the same thing. Adding them produces no
exception, no NaN and no crash. It produces a wrong answer that converges
cleanly. Runtime type checking is the only thing that catches it.

There are deliberately no convenience coercions. No __array__, no in place
operators, no implicit float promotion. Every one of those would be a hole
through which an unlabelled array re-enters the system.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ddsim.core.scaling import ScaleFactors


class ScalingState(Enum):
    """Whether the numbers are in physical units or de Mari scaled units."""

    PHYSICAL = "physical"
    SCALED = "scaled"


class Location(Enum):
    """Where on the mesh the quantity lives.

    Node quantities (psi, n, p) and edge quantities (current density, field)
    are not interchangeable, and a length coincidence must never let them be
    combined by accident.
    """

    NODE = "node"
    EDGE = "edge"
    CELL = "cell"


def _combine_multiply(left: str, right: str) -> str:
    """Unit string for a product. Not dimensional analysis, just bookkeeping."""
    if left == "1":
        return right
    if right == "1":
        return left
    return f"{left}*{right}"


def _combine_divide(numerator: str, denominator: str) -> str:
    """Unit string for a quotient. Not dimensional analysis, just bookkeeping."""
    if numerator == denominator:
        return "1"
    if denominator == "1":
        return numerator
    if "/" in denominator or "*" in denominator:
        return f"{numerator}/({denominator})"
    return f"{numerator}/{denominator}"


@dataclass(frozen=True, eq=False)
class Field:
    """A numpy array that knows its unit, its scaling state and where it lives.

    Attributes are frozen. A field whose scaling state can be reassigned in
    place is no protection at all.

    Equality is identity. Generated structural equality would compare the
    underlying arrays and return an array, which is worse than useless in an
    assertion.
    """

    data: np.ndarray
    """The raw values. Units are given by `unit`, interpretation by `scaling`."""

    unit: str
    """Symbolic unit string, for example "cm^-3", "V", "A/cm^2", or "1"."""

    scaling: ScalingState
    """PHYSICAL or SCALED. Never mixed, never coerced."""

    location: Location
    """NODE, EDGE or CELL."""

    name: str | None = dataclass_field(default=None)
    """Optional label, for error messages and plots."""

    # Make numpy defer instead of silently broadcasting a Field into an array.
    __array_ufunc__ = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", np.asarray(self.data, dtype=np.float64))

    # ------------------------------------------------------------- properties

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying array."""
        return self.data.shape

    @property
    def size(self) -> int:
        """Number of entries."""
        return int(self.data.size)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __repr__(self) -> str:
        label = f" {self.name!r}" if self.name is not None else ""
        return (
            f"Field{label} [{self.unit}] {self.scaling.name} "
            f"{self.location.name} n={self.size}"
        )

    # --------------------------------------------------------------- guards

    def _require_field(self, other: object, operation: str) -> Field:
        if not isinstance(other, Field):
            raise TypeError(
                f"cannot {operation} a Field and {type(other).__name__}. "
                "A bare number or array carries no unit, no scaling state and "
                "no mesh location, so the result would be unverifiable. Wrap "
                "it in a Field, or multiply by a scalar if it is dimensionless."
            )
        return other

    def _check_same_state(self, other: Field, operation: str) -> None:
        """Scaling and location must match for every binary operation."""
        if self.scaling is not other.scaling:
            raise ValueError(
                f"cannot {operation} fields with different scaling states: "
                f"{self.scaling.name} and {other.scaling.name}. These are the "
                "same physical quantity in two different unit systems. Convert "
                "one with to_scaled or to_physical first."
            )
        if self.location is not other.location:
            raise ValueError(
                f"cannot {operation} fields at different mesh locations: "
                f"{self.location.name} and {other.location.name}."
            )

    def _check_additive(self, other: Field, operation: str) -> None:
        """Addition and subtraction additionally require matching units."""
        self._check_same_state(other, operation)
        if self.unit != other.unit:
            raise ValueError(
                f"cannot {operation} fields with different units: "
                f"[{self.unit}] and [{other.unit}]."
            )
        if self.shape != other.shape:
            raise ValueError(
                f"cannot {operation} fields of different length: "
                f"{self.shape} and {other.shape}."
            )

    def _like(self, data: np.ndarray, unit: str | None = None) -> Field:
        """A new Field with the same metadata and new values."""
        return Field(
            data,
            self.unit if unit is None else unit,
            self.scaling,
            self.location,
            self.name,
        )

    # ------------------------------------------------------------ arithmetic

    def __add__(self, other: Field) -> Field:
        other = self._require_field(other, "add")
        self._check_additive(other, "add")
        return self._like(self.data + other.data)

    def __sub__(self, other: Field) -> Field:
        other = self._require_field(other, "subtract")
        self._check_additive(other, "subtract")
        return self._like(self.data - other.data)

    def __neg__(self) -> Field:
        return self._like(-self.data)

    def __mul__(self, other: Field | float) -> Field:
        if isinstance(other, int | float):
            return self._like(self.data * other)
        other = self._require_field(other, "multiply")
        self._check_same_state(other, "multiply")
        return self._like(
            self.data * other.data, _combine_multiply(self.unit, other.unit)
        )

    def __rmul__(self, other: float) -> Field:
        return self.__mul__(other)

    def __truediv__(self, other: Field | float) -> Field:
        if isinstance(other, int | float):
            return self._like(self.data / other)
        other = self._require_field(other, "divide")
        self._check_same_state(other, "divide")
        return self._like(
            self.data / other.data, _combine_divide(self.unit, other.unit)
        )

    # ------------------------------------------------------ state conversion

    def to_scaled(self, scale: ScaleFactors) -> Field:
        """Convert to de Mari scaled units.

        Raises if the field is already scaled. This is not a no-op, because
        calling it on a scaled field means the caller has lost track of state,
        and that is the bug worth surfacing.
        """
        if self.scaling is ScalingState.SCALED:
            raise ValueError(
                f"field [{self.unit}] is already SCALED. Calling to_scaled "
                "again would divide by the scale factor a second time."
            )
        return Field(
            self.data / scale.factor(self.unit),
            self.unit,
            ScalingState.SCALED,
            self.location,
            self.name,
        )

    def to_physical(self, scale: ScaleFactors) -> Field:
        """Convert to physical units. Raises if the field is already physical."""
        if self.scaling is ScalingState.PHYSICAL:
            raise ValueError(
                f"field [{self.unit}] is already PHYSICAL. Calling to_physical "
                "again would multiply by the scale factor a second time."
            )
        return Field(
            self.data * scale.factor(self.unit),
            self.unit,
            ScalingState.PHYSICAL,
            self.location,
            self.name,
        )
