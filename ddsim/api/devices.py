"""The device registry the browser builds a form from.

phases/PHASE-7.md asks for device definition in the browser with nothing
hardcoded per device. That is kept honest by reading the device constructors'
own signatures rather than restating them here: a knob added to nmos() is
offered by the API the moment it exists, and a default changed in
device/pn_diode.py is the default the browser shows. A second copy of either
would go stale and the browser would quietly be simulating something else.

Only parameters that a JSON number or boolean can express are offered. The
material argument is the one exclusion, and it is excluded in both directions:
not offered, and refused if sent. A device built on a material description the
browser invented is not a device this project ever validated.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ddsim.device.builder import Device
from ddsim.device.mos_cap import mos_cap
from ddsim.device.mosfet import nmos
from ddsim.device.pn_diode import pn_diode

DEVICE_KINDS: dict[str, Callable[..., Device]] = {
    "pn_diode": pn_diode,
    "mos_cap": mos_cap,
    "nmos": nmos,
}
"""Every device the API will build, by the name the client sends."""

_EXPRESSIBLE = ("float", "int", "bool")
"""Annotations a JSON value can carry. Read as written text rather than
evaluated, which is why device modules declaring `from __future__ import
annotations` are a help here rather than an obstacle."""


@dataclass(frozen=True)
class DeviceParameter:
    """One knob on one device, as the browser needs to render it."""

    name: str
    """The constructor's own argument name."""

    default: float | int | bool
    """The constructor's own default. There is no second set of defaults."""

    type: str
    """One of float, int, bool."""


def _builder(kind: str) -> Callable[..., Device]:
    if kind not in DEVICE_KINDS:
        known = ", ".join(sorted(DEVICE_KINDS))
        raise ValueError(f"unknown device kind {kind!r}. Known kinds: {known}")
    return DEVICE_KINDS[kind]


def device_parameters(kind: str) -> tuple[DeviceParameter, ...]:
    """Every settable knob on a device, in the order the constructor declares.

    Args:
        kind: a key of DEVICE_KINDS.

    Units are not carried here. They live in the constructor docstrings, which
    is the single place this project writes them down.
    """
    signature = inspect.signature(_builder(kind))
    return tuple(
        DeviceParameter(
            name=name,
            default=parameter.default,
            type=str(parameter.annotation),
        )
        for name, parameter in signature.parameters.items()
        if str(parameter.annotation) in _EXPRESSIBLE
    )


def build_from_spec(kind: str, parameters: dict[str, Any]) -> Device:
    """Build a device from a name and a dict of constructor arguments.

    Args:
        kind: a key of DEVICE_KINDS.
        parameters: argument names and values. Anything left out keeps the
            constructor's default.

    Raises ValueError for a name the device does not have and TypeError for a
    value of the wrong kind. Both refuse rather than fall back: a silently
    dropped typo hands back the default device, and a plot of the wrong device
    that nobody was warned about is the worst outcome this layer can produce.
    """
    offered = {p.name: p for p in device_parameters(kind)}
    accepted: dict[str, float | int | bool] = {}

    for name, value in parameters.items():
        if name not in offered:
            known = ", ".join(offered)
            raise ValueError(
                f"{kind} has no settable parameter {name!r}. Settable: {known}"
            )
        accepted[name] = _checked(kind, offered[name], value)

    return _builder(kind)(**accepted)


def _checked(kind: str, parameter: DeviceParameter, value: Any) -> float | int | bool:
    """One value against one declared type.

    bool is a subclass of int in Python, so isinstance alone lets True through
    as a node count, which builds a one node mesh and fails decades away from
    the field that caused it. The checks are on the exact type for that reason.
    """
    if parameter.type == "bool":
        if type(value) is not bool:
            raise TypeError(
                f"{kind}.{parameter.name} is a boolean, got {type(value).__name__}"
            )
        return value
    if parameter.type == "int":
        if type(value) is not int:
            raise TypeError(
                f"{kind}.{parameter.name} is an integer, got {type(value).__name__}"
            )
        return value
    # float. A whole number arrives from JSON as an int and is perfectly good
    # here, so it is widened rather than refused.
    if type(value) is int:
        return float(value)
    if type(value) is not float:
        raise TypeError(
            f"{kind}.{parameter.name} is a number, got {type(value).__name__}"
        )
    return value
