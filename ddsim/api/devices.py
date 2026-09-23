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
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from enum import Enum
from functools import cache
from typing import Any

from ddsim.device.builder import Device
from ddsim.device.drawing import NODE_BUDGET, Block, Electrode, Implant, drawing
from ddsim.device.mos_cap import mos_cap
from ddsim.device.mosfet import nmos
from ddsim.device.pn_diode import pn_diode
from ddsim.device.stack import Region, stack

DEVICE_KINDS: dict[str, Callable[..., Device]] = {
    "pn_diode": pn_diode,
    "mos_cap": mos_cap,
    "nmos": nmos,
    "stack": stack,
    "drawing": drawing,
}
"""Every device the API will build, by the name the client sends."""

_EXPRESSIBLE = ("float", "int", "bool", "str")
"""Annotations a JSON value can carry. Read as written text rather than
evaluated, which is why device modules declaring `from __future__ import
annotations` are a help here rather than an obstacle.

No device constructor takes a string today. The models a sweep runs do, which
is where `str` is earning its place. See ddsim/api/sweeps.py."""


@dataclass(frozen=True)
class Preset:
    """A coarse mesh offered beside a device's converged one.

    Knob settings and nothing else, so a preset cannot change what is solved,
    only how finely. The converged mesh needs no entry here: it is the
    constructor's own defaults, which is what clearing the preset goes back
    to.
    """

    parameters: dict[str, float | int]
    """What to set, by knob name. Every name is one the device has, and a test
    holds that."""

    note: str
    """What choosing it costs, with the numbers it was measured with. The page
    shows this next to the device, because a student reading a current off a
    coarse mesh should know how far off it is."""


COARSE: dict[str, Preset] = {
    "mos_cap": Preset(
        parameters={"n_silicon": 41, "n_oxide": 3, "h_min": 2e-7},
        note=(
            "A coarse mesh: 129 nodes against the converged 375. Measured "
            "2026-09-17 over a -2 V to 2 V C-V, the capacitance reads 0.556 "
            "percent high in accumulation and 0.714 percent high in "
            "depletion, and the sweep solves 1.8 times faster. The shape of "
            "the curve is the same; the numbers on it are not the validated "
            "ones."
        ),
    ),
    "nmos": Preset(
        parameters={
            "n_contact": 4,
            "n_sd": 10,
            "n_channel": 12,
            "n_silicon": 29,
            "n_oxide": 4,
            "h_min_x": 5e-7,
            "h_min_y": 1e-7,
        },
        note=(
            "A coarse mesh: 1504 nodes against the converged 8379. Measured "
            "2026-09-17 over a 0 V to 1.2 V transfer at 50 mV drain, the "
            "drain current at 1.2 V reads 0.621 percent high and the "
            "extrapolated threshold moves 0.7 mV, for a sweep that takes "
            "5.1 s instead of 20.4 s. Good enough to watch a MOSFET switch, "
            "not the mesh any number in the README was taken on."
        ),
    ),
    "drawing": Preset(
        parameters={"nx": 39, "ny": 35, "h_min_x": 5e-7, "h_min_y": 1e-7},
        note=(
            "A coarse mesh: 1365 nodes against the converged 8379. Measured "
            "2026-09-18 on the default drawing, the benchmark nmos, over a 0 V "
            "to 1.2 V transfer at 50 mV drain: the drain current reads 0.75 "
            "percent high at 1.2 V and up to 3.9 percent high between 0.5 and "
            "0.7 V, and the extrapolated threshold moves 1.5 mV, for a sweep "
            "that takes 4.0 s instead of 18.9 s. A drawing of your own was not "
            "measured and can be further off."
        ),
    ),
}
"""The coarse mesh on offer per device, for the devices the page does not
solve live.

The diode has none. Measured on the same day, its converged mesh solves a
seven point I-V in 0.2 s, so a second mesh there would cost a student an
explanation and save them nothing. See phases/PHASE-7.md Stage 2.
"""


def node_count(device: Device) -> int:
    """How many nodes a built device's mesh has [1].

    Measured on the mesh rather than added up from the node knobs, because a
    2D mesh is a tensor product of several of them and a 1D one has a single
    knob that is not spelled the same way.
    """
    return device.mesh.n_nodes


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

    explanation: str = ""
    """The function's own Args: line for this argument, continuation lines
    joined. What the page shows beside the knob."""

    unit: str = ""
    """The first [bracketed] unit in the explanation, without the brackets.
    "1" for a dimensionless number. Empty for a switch or a name."""

    low: float | None = None
    """The bottom of the range the explanation declares, or None where it
    declares none. A slider needs two ends, and where they belong is a claim
    about the device rather than about the page, so it is written where the
    knob is and read from there. None means no slider, not a guessed pair."""

    high: float | None = None
    """The top of that range, or None."""

    axis: str = "linear"
    """How a slider should space the range, "linear" or "log". A doping that
    runs over five decades is unusable on a linear slider: every setting below
    1e18 sits inside the last tenth of the travel. The same argument as the log
    axis on the plots, and the same answer: this is a position on a screen."""


def _builder(kind: str) -> Callable[..., Device]:
    if kind not in DEVICE_KINDS:
        known = ", ".join(sorted(DEVICE_KINDS))
        raise ValueError(f"unknown device kind {kind!r}. Known kinds: {known}")
    return DEVICE_KINDS[kind]


_ARG_LINE = re.compile(r"^    (\w+): (.*)$")
_UNIT = re.compile(r"\[([^\]]+)\]")

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_RANGE = re.compile(rf"Range\s+({_NUMBER})\s+to\s+({_NUMBER})(,\s*log)?")
"""`Range 1e14 to 1e19, log.` in an Args: line, spelled out rather than
inferred from the numbers. Docstrings here are full of pairs of numbers that
are not ranges, and a parser that took any of them would put a slider on the
wrong span without anyone noticing."""


def argument_docs(function: Callable[..., Any]) -> dict[str, str]:
    """Each argument's description from the function's Args: block.

    Args:
        function: any documented function.

    Reads the Google style block this project writes everywhere: an `Args:`
    line, then one entry per argument indented four spaces, with continuation
    lines indented eight. The block ends at the first line that is not
    indented.
    """
    docs: dict[str, str] = {}
    current: str | None = None
    inside = False
    for line in (inspect.getdoc(function) or "").splitlines():
        if line.strip() == "Args:":
            inside = True
            continue
        if not inside:
            continue
        if line and not line.startswith(" "):
            break
        entry = _ARG_LINE.match(line)
        if entry:
            current = entry.group(1)
            docs[current] = entry.group(2).strip()
        elif current is not None and line.startswith("        "):
            docs[current] += " " + line.strip()
    return docs


def _unit_of(explanation: str) -> str:
    found = _UNIT.search(explanation)
    return found.group(1) if found else ""


def _range_of(explanation: str) -> tuple[float | None, float | None, str]:
    """The declared slider range, as (low, high, axis).

    (None, None, "linear") where the explanation declares none, which is what
    every knob the page renders as a text box looks like.
    """
    found = _RANGE.search(explanation)
    if found is None:
        return None, None, "linear"
    return (
        float(found.group(1)),
        float(found.group(2)),
        "log" if found.group(3) else "linear",
    )


@cache
def device_dimension(kind: str) -> int:
    """How many axes the device this kind builds actually has, 1 or 2.

    Read from the mesh the constructor returns rather than from a list here,
    for the reason every other fact in this module is read from the code that
    owns it. It decides which devices the page will solve live: a 2D solve is
    seconds to minutes, so phases/PHASE-7.md keeps the solve button on those.

    Args:
        kind: a key of DEVICE_KINDS.

    Cached, because the answer cannot change while the process runs and
    building an nmos to ask is a mesh and a doping profile.
    """
    return 2 if hasattr(_builder(kind)().mesh, "ny") else 1


@cache
def contact_names(kind: str) -> tuple[str, ...]:
    """The terminals the device this kind builds by default has, in order.

    Args:
        kind: a key of DEVICE_KINDS.

    Read from the built device, like device_dimension and for the same
    reason. A drawn device names its own electrodes, so for a drawing these
    are the default drawing's and the page reads the rows it has instead.
    """
    return tuple(contact.name for contact in _builder(kind)().contacts)


def parameters_of(
    function: Callable[..., Any], choices: dict[str, tuple[str, ...]] | None = None
) -> tuple[Parameter, ...]:
    """Every knob a JSON value can set on a function, in declaration order.

    Args:
        function: any function the API calls for the browser.
        choices: the closed sets some of its string arguments accept, by
            argument name. Carried through to the Parameter and not checked
            here.

    The explanation and unit are read from the function's own docstring,
    which is the single place this project writes them down.
    """
    named = choices or {}
    docs = argument_docs(function)
    offered: list[Parameter] = []

    def described(name: str, **rest: Any) -> Parameter:
        """One Parameter with everything the docstring says about it."""
        explanation = docs.get(name, "")
        low, high, axis = _range_of(explanation)
        return Parameter(
            name=name,
            explanation=explanation,
            unit=_unit_of(explanation),
            low=low,
            high=high,
            axis=axis,
            **rest,
        )

    for name, parameter in inspect.signature(function).parameters.items():
        if parameter.default is inspect.Parameter.empty:
            continue
        if isinstance(parameter.default, Enum):
            offered.append(
                described(
                    name,
                    default=parameter.default.value,
                    type="str",
                    choices=tuple(
                        str(member.value) for member in type(parameter.default)
                    ),
                )
            )
        elif str(parameter.annotation) in _EXPRESSIBLE:
            offered.append(
                described(
                    name,
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


RECORDS: dict[str, type] = {
    "regions": Region,
    "blocks": Block,
    "implants": Implant,
    "electrodes": Electrode,
}
"""The arguments a device takes as a list of records rather than one number,
by argument name, with the record each entry is. A stack is built from
regions; a drawing from blocks, implants and electrodes. None of them is a
knob: the page draws each as rows."""

DRAWING_PARTS = ("blocks", "implants", "electrodes")
"""The record lists a drawn device is made of, in the order the page shows."""


def record_defaults(kind: str, name: str) -> list[dict[str, Any]] | None:
    """The records a device starts from under one argument, as the page sends
    them, or None for a device without that argument.

    Args:
        kind: a key of DEVICE_KINDS.
        name: a key of RECORDS.

    Read from the constructor's own default, like every other default here.
    """
    argument = inspect.signature(_builder(kind)).parameters.get(name)
    if argument is None:
        return None
    return [asdict(record) for record in argument.default]


def region_defaults(kind: str) -> list[dict[str, Any]] | None:
    """The regions a stack device starts from, or None. See record_defaults."""
    return record_defaults(kind, "regions")


def drawing_defaults(kind: str) -> dict[str, list[dict[str, Any]]] | None:
    """The blocks, implants and electrodes a drawn device starts from, or None
    for a device that is not drawn."""
    if record_defaults(kind, "blocks") is None:
        return None
    return {name: record_defaults(kind, name) or [] for name in DRAWING_PARTS}


def records_from_json(name: str, sent: Any) -> tuple[Any, ...]:
    """Records as the page or a saved device file sends them, checked.

    Args:
        name: a key of RECORDS, which says what each entry is.
        sent: a list of objects, each with exactly the fields of that record.

    A saved file is a student's own, edited by hand as often as not, so what
    is wrong with it is named by entry number and field. Whether the records
    make a device the models cover is the device's own judgement, not this.
    """
    record = RECORDS[name]
    what = name[:-1]
    expected = {field.name: field.type for field in fields(record)}
    if not isinstance(sent, list):
        raise TypeError(f"{name} is a list of {name}, got {type(sent).__name__}")
    records = []
    for number, entry in enumerate(sent, start=1):
        if not isinstance(entry, dict):
            raise TypeError(
                f"{what} {number} is {type(entry).__name__}, not an object with "
                f"{', '.join(expected)}"
            )
        missing = [field for field in expected if field not in entry]
        extra = [field for field in entry if field not in expected]
        if missing or extra:
            raise ValueError(
                f"{what} {number} has the fields {', '.join(expected)}; "
                f"missing {missing}, not a field {extra}"
            )
        for field, annotation in expected.items():
            value = entry[field]
            fits = (
                type(value) is str
                if annotation == "str"
                else type(value) in (int, float)
            )
            if not fits:
                raise TypeError(
                    f"{what} {number}: {field} is a "
                    f"{'name' if annotation == 'str' else 'number'}, got "
                    f"{type(value).__name__}"
                )
        records.append(
            record(
                **{
                    field: entry[field]
                    if annotation == "str"
                    else float(entry[field])
                    for field, annotation in expected.items()
                }
            )
        )
    return tuple(records)


def regions_from_json(sent: Any) -> tuple[Region, ...]:
    """A stack's regions, checked. See records_from_json."""
    return records_from_json("regions", sent)


def build_from_spec(kind: str, parameters: dict[str, Any]) -> Device:
    """Build a device from a name and a dict of constructor arguments.

    Args:
        kind: a key of DEVICE_KINDS.
        parameters: argument names and values. Anything left out keeps the
            constructor's default. A stack's regions and a drawing's blocks,
            implants and electrodes come as lists of objects, see
            records_from_json.

    Raises ValueError for a name the device does not have, or for a mesh over
    NODE_BUDGET, and TypeError for a value of the wrong kind. See
    checked_arguments for why neither falls back.
    """
    offered = {p.name: p for p in device_parameters(kind)}
    knobs = dict(parameters)
    structured: dict[str, Any] = {}
    for name in RECORDS:
        if name in knobs:
            if record_defaults(kind, name) is None:
                raise ValueError(f"{kind} is not built from {name}")
            structured[name] = records_from_json(name, knobs.pop(name))
    accepted = checked_arguments(kind, offered, knobs)
    # Every integer knob is a node count. One over the budget cannot make a
    # mesh under it, so it is refused before a typo gets to allocate one.
    # ponytail: several knobs each under the budget can still multiply into
    # a mesh too big to build; bound the product per device if that happens.
    for name, value in accepted.items():
        if type(value) is int and value > NODE_BUDGET:
            raise ValueError(
                f"{kind}.{name} = {value} is over the node budget of {NODE_BUDGET}"
            )
    device = _builder(kind)(**accepted, **structured)
    if node_count(device) > NODE_BUDGET:
        raise ValueError(
            f"this {kind} mesh is {node_count(device)} nodes, over the budget of "
            f"{NODE_BUDGET}. Fewer nodes along one axis, or the coarse mesh."
        )
    return device


def _checked(kind: str, parameter: Parameter, value: Any) -> float | int | bool | str:
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
    if type(value) is int:
        return float(value)
    if type(value) is not float:
        raise TypeError(
            f"{kind}.{parameter.name} is a number, got {type(value).__name__}"
        )
    return value
