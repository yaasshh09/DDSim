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
from enum import Enum
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

_EXPRESSIBLE = ("float", "int", "bool", "str")
"""Annotations a JSON value can carry. Read as written text rather than
evaluated, which is why device modules declaring `from __future__ import
annotations` are a help here rather than an obstacle.

No device constructor takes a string today. The models a sweep runs do, which
is where `str` is earning its place. See ddsim/api/sweeps.py."""


@dataclass(frozen=True)
class Parameter:
    """One settable knob, as the browser needs to render it.

    Used for a device constructor's arguments and for a sweep's, which are the
    same thing from a form's point of view: a name, a type and the default the
    function itself declares.
    """

    name: str
    """The function's own argument name."""

    default: float | int | bool | str
    """The function's own default. There is no second set of defaults."""

    type: str
    """One of float, int, bool, str."""

    choices: tuple[str, ...] = ()
    """The values a string knob accepts, where they are a closed set the
    project writes down somewhere. Empty otherwise, and empty for every
    numeric knob. A hint for rendering a form rather than a second validator:
    what a name means is decided by the code that consumes it, and that code
    is the one that refuses a name it does not know.
    """


def _builder(kind: str) -> Callable[..., Device]:
    if kind not in DEVICE_KINDS:
        known = ", ".join(sorted(DEVICE_KINDS))
        raise ValueError(f"unknown device kind {kind!r}. Known kinds: {known}")
    return DEVICE_KINDS[kind]


def parameters_of(
    function: Callable[..., Any], choices: dict[str, tuple[str, ...]] | None = None
) -> tuple[Parameter, ...]:
    """Every knob a JSON value can set on a function, in declaration order.

    Args:
        function: any function the API calls for the browser.
        choices: the closed sets some of its string arguments accept, by
            argument name. Carried through to the Parameter and not checked
            here.

    Units are not carried. They live in the docstrings, which is the single
    place this project writes them down.
    """
    named = choices or {}
    offered: list[Parameter] = []
    for name, parameter in inspect.signature(function).parameters.items():
        # An argument with no default is part of the request rather than a
        # knob on a form: there is nothing to render beside it and nothing to
        # fall back to if it is left out. The caller passes those itself.
        if parameter.default is inspect.Parameter.empty:
            continue
        # An enumerated argument is a closed set of names, and both the names
        # and the default are on the enum itself. Crossing the wire as the
        # string the enum already uses as its value means the browser never
        # has to know the type exists.
        if isinstance(parameter.default, Enum):
            offered.append(
                Parameter(
                    name=name,
                    default=parameter.default.value,
                    type="str",
                    choices=tuple(
                        str(member.value) for member in type(parameter.default)
                    ),
                )
            )
        elif str(parameter.annotation) in _EXPRESSIBLE:
            offered.append(
                Parameter(
                    name=name,
                    default=parameter.default,
                    type=str(parameter.annotation),
                    choices=named.get(name, ()),
                )
            )
    return tuple(offered)


def enum_arguments(function: Callable[..., Any]) -> dict[str, type[Enum]]:
    """The arguments of `function` whose value is one of a closed set.

    The caller turns the name it was sent back into the member before the
    call. Kept next to parameters_of because the two have to agree about
    which arguments those are, and the answer is read from the signature in
    both.
    """
    return {
        name: type(parameter.default)
        for name, parameter in inspect.signature(function).parameters.items()
        if isinstance(parameter.default, Enum)
    }


def as_enum(what: str, name: str, kind: type[Enum], value: Any) -> Enum:
    """One name back into its member, or a refusal that lists the names.

    Enum's own ValueError does not say which argument it was about, and on a
    form with two of them that is the only thing the reader needs.
    """
    try:
        return kind(value)
    except ValueError as error:
        known = ", ".join(str(member.value) for member in kind)
        raise ValueError(
            f"{what}.{name} has no setting {value!r}. Settings: {known}"
        ) from error


def device_parameters(kind: str) -> tuple[Parameter, ...]:
    """Every settable knob on a device, in the order the constructor declares.

    Args:
        kind: a key of DEVICE_KINDS.
    """
    return parameters_of(_builder(kind))


def checked_arguments(
    what: str,
    offered: dict[str, Parameter],
    sent: dict[str, Any],
) -> dict[str, float | int | bool | str]:
    """Every value in `sent` against the knob of the same name in `offered`.

    Args:
        what: the thing being configured, for the refusals. A device kind or a
            sweep kind.
        offered: the knobs there are, by name.
        sent: what the client asked for.

    Raises ValueError for a name that is not a knob and TypeError for a value
    of the wrong kind. Both refuse rather than fall back: a silently dropped
    typo hands back the default, and a plot of something nobody asked for that
    nobody was warned about is the worst outcome this layer can produce.
    """
    accepted: dict[str, float | int | bool | str] = {}
    for name, value in sent.items():
        if name not in offered:
            known = ", ".join(offered)
            raise ValueError(
                f"{what} has no settable parameter {name!r}. Settable: {known}"
            )
        accepted[name] = _checked(what, offered[name], value)
    return accepted


def build_from_spec(kind: str, parameters: dict[str, Any]) -> Device:
    """Build a device from a name and a dict of constructor arguments.

    Args:
        kind: a key of DEVICE_KINDS.
        parameters: argument names and values. Anything left out keeps the
            constructor's default.

    Raises ValueError for a name the device does not have and TypeError for a
    value of the wrong kind. See checked_arguments for why neither falls back.
    """
    offered = {p.name: p for p in device_parameters(kind)}
    return _builder(kind)(**checked_arguments(kind, offered, parameters))


def _checked(
    kind: str, parameter: Parameter, value: Any
) -> float | int | bool | str:
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
    if parameter.type == "str":
        if type(value) is not str:
            raise TypeError(
                f"{kind}.{parameter.name} is a name, got {type(value).__name__}"
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
