"""Turning a frame into a message, phases/PHASE-7.md.

A telemetry frame is a handful of scalars and crosses as JSON text. A field is
one number per node, which on a 200 by 100 device is 20000 of them per array,
and JSON costs roughly ten times the bytes and parses slowly enough to be
visible. Those cross as float32 behind a small JSON header, the same
architecture AtomSIM uses.

The binary layout, which the client reads and nothing else does:

    uint32 little endian   length of the header in bytes
    utf-8 JSON             the header, space padded so the payload starts
                           on a multiple of 4
    float32 little endian  every array end to end, in the header's order

The header names each array, its unit and its length, so the client slices the
payload from the header rather than from a layout of its own. Anything it
needs to draw is in there.

Two decisions worth stating.

**Non finite numbers cross as null.** JSON has no infinity and no NaN.
json.dumps writes both as bare words that JSON.parse refuses, so an encoder
that passed them through would break the stream on the one frame that
mattered: a diverged Gummel update is not finite by construction, and that is
the frame a reader is watching for.

**A frame this module does not know is refused.** A new frame type added to a
solver and not added here would otherwise vanish, and a plot silently missing
points is worse than an error.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ddsim.core.field import Field
from ddsim.device.builder import Device
from ddsim.device.state import DeviceState
from ddsim.device.transport import TransportModels
from ddsim.extract.bands import band_edges
from ddsim.extract.cv import CVCurve, CVFrame, CVPoint
from ddsim.extract.iv import IVCurve, IVFrame, IVPoint, node_current_density
from ddsim.solve.continuation import ContinuationEvent
from ddsim.solve.gummel import GummelIteration
from ddsim.solve.newton import NewtonIteration

WIRE_DTYPE = "<f4"
"""float32, little endian. Half the bytes of a double for a plot that is
drawn a few hundred pixels wide, which is the whole argument. Nothing is
computed from these numbers on the far side."""


@dataclass(frozen=True)
class FieldFrame:
    """Every array the client needs to draw one solved state.

    Built by field_frame rather than by the solver, from a state that has
    already converged. Nothing inside a solve ever holds one of these, which
    is what keeps the scalars only rule on the telemetry frames intact.
    """

    index: int
    """Which sweep point this is the state at, from zero."""

    voltage: float
    """The bias that point was solved at [V]."""

    shape: tuple[int, ...]
    """(n,) in 1D, or (ny, nx) in 2D, which is the image a field reshapes to.

    The 2D node order is the one mesh2d builds: x fastest, so row j of the
    image is the row of nodes at y[j].
    """

    arrays: tuple[tuple[str, str, npt.NDArray[np.float64]], ...]
    """(name, unit, values) for each array, in the order they are sent."""


@dataclass(frozen=True)
class Status:
    """How a job ended, which the solver has no opinion about.

    Sent by the layer running the job rather than by the sweep, because
    cancelled and failed are facts about the job and not about the physics.
    """

    status: str
    """One of the JobStatus values."""

    message: str = ""
    """Why it failed or stalled, or empty."""

    dropped: int = 0
    """Frames discarded because the reader was behind. Reported rather than
    swallowed: a stream that quietly loses points is a plot that lies."""


def _finite(value: float | None) -> float | None:
    """A number JSON can carry, or None. See the module docstring."""
    if value is None or not math.isfinite(value):
        return None
    return value


def _finite_by_family(
    split: dict[str, float] | None,
) -> dict[str, float | None] | None:
    """A per family split JSON can carry, family by family. See _finite."""
    if split is None:
        return None
    return {family: _finite(value) for family, value in split.items()}


def _body(frame: object) -> dict[str, Any]:
    """One frame as the object the client switches on, by its own type."""
    if isinstance(frame, NewtonIteration):
        return {
            "type": "newton",
            "iteration": frame.iteration,
            "residual": _finite(frame.residual),
            "update": _finite(frame.update),
            "damping": _finite(frame.damping),
            "limited": frame.limited,
            "residual_by_family": _finite_by_family(frame.residual_by_family),
            "update_by_family": _finite_by_family(frame.update_by_family),
        }
    if isinstance(frame, GummelIteration):
        return {
            "type": "gummel",
            "iteration": frame.iteration,
            "update": _finite(frame.update),
        }
    if isinstance(frame, ContinuationEvent):
        return {
            "type": "continuation",
            "parameter": _finite(frame.parameter),
            "step": _finite(frame.step),
            "converged": frame.converged,
            "message": frame.message,
        }
    if isinstance(frame, IVFrame):
        return {
            "type": "point",
            "index": frame.index,
            "voltage": _finite(frame.voltage),
            "current": _finite(frame.current),
        }
    if isinstance(frame, CVFrame):
        return {
            "type": "cv_point",
            "index": frame.index,
            "gate_voltage": _finite(frame.gate_voltage),
            "capacitance": _finite(frame.capacitance),
            "charge": _finite(frame.charge),
        }
    if isinstance(frame, Status):
        return {
            "type": "status",
            "status": frame.status,
            "message": frame.message,
            "dropped": frame.dropped,
        }
    raise TypeError(
        f"a {type(frame).__name__} cannot be sent. Every frame type the "
        "solvers report has to be named in api/frames.py, or it would be "
        "dropped from the stream with nothing to say so."
    )


def encode(frame: object) -> str | bytes:
    """One frame as a websocket message: text for scalars, bytes for fields.

    Args:
        frame: anything a solver or a job reports.

    Raises TypeError for a frame type this module does not know.
    """
    if isinstance(frame, FieldFrame):
        header = json.dumps(
            {
                "type": "fields",
                "index": frame.index,
                "voltage": _finite(frame.voltage),
                "shape": list(frame.shape),
                "arrays": [
                    {"name": name, "unit": unit, "length": int(values.size)}
                    for name, unit, values in frame.arrays
                ],
            }
        ).encode("utf-8")
        header += b" " * (-(4 + len(header)) % 4)
        payload = b"".join(
            np.ascontiguousarray(values, dtype=WIRE_DTYPE).tobytes()
            for _, _, values in frame.arrays
        )
        return struct.pack("<I", len(header)) + header + payload

    return json.dumps(_body(frame), allow_nan=False)


def decode_fields(message: bytes) -> dict[str, npt.NDArray[np.float64]]:
    """A field message back into its arrays, by name.

    The client does this in JavaScript. This is here so that the tests read
    the bytes the same way the browser does rather than trusting the writer,
    and so that the layout has exactly one description in Python.
    """
    (length,) = struct.unpack_from("<I", message, 0)
    header = json.loads(message[4 : 4 + length])
    payload = np.frombuffer(message[4 + length :], dtype=WIRE_DTYPE)

    arrays: dict[str, npt.NDArray[np.float64]] = {}
    at = 0
    for array in header["arrays"]:
        size = int(array["length"])
        arrays[array["name"]] = payload[at : at + size].astype(np.float64)
        at += size
    return arrays


def curve_body(curve: IVCurve | CVCurve) -> dict[str, Any]:
    """A finished curve as the object the client plots.

    Scalars again, and JSON rather than float32: a sweep is tens of points,
    not tens of thousands, and a double written by json.dumps round trips
    exactly. That exactness is an acceptance criterion, since what the browser
    draws has to be what pytest gets from the same call.

    The converged states are not in here. They are what the field messages
    carry, one point at a time and only when asked for.
    """
    if isinstance(curve, CVCurve):
        return {
            "kind": "cv",
            "contact": curve.contact,
            "response": curve.response.value,
            "complete": curve.complete,
            "message": curve.message,
            "points": [
                {
                    "index": index,
                    "voltage": point.gate_voltage,
                    "capacitance": point.capacitance,
                    "charge": point.charge,
                }
                for index, point in enumerate(curve.points)
            ],
        }
    return {
        "kind": "iv",
        "contact": curve.contact,
        "measured_at": curve.measured_at,
        "complete": curve.complete,
        "message": curve.message,
        "points": [
            {"index": index, "voltage": point.voltage, "current": point.current}
            for index, point in enumerate(curve.points)
        ],
    }


def point_voltage(point: IVPoint | CVPoint) -> float:
    """The bias one point was solved at [V], whichever curve it came from.

    An I-V point calls it voltage and a C-V point calls it gate_voltage,
    which is the name each curve uses on its own axis. The wire calls it
    voltage for both, because a plot has one x axis.
    """
    if isinstance(point, CVPoint):
        return point.gate_voltage
    return point.voltage


def _physical(field: Field, device: Device) -> npt.NDArray[np.float64]:
    return np.asarray(field.to_physical(device.scale).data, dtype=np.float64)


def field_frame(
    device: Device,
    state: DeviceState,
    index: int,
    voltage: float,
    models: TransportModels | None = None,
) -> FieldFrame:
    """The arrays for one converged state, in physical units.

    Args:
        device: the device, for its mesh and its scale factors.
        state: a converged solution. Its fields are scaled, as every
            DeviceState's are, and they are converted here.
        index: which sweep point this is, from zero.
        voltage: the bias it was solved at [V].
        models: the transport models the state was solved with; when given,
            the node current density Jx, Jy [A/cm^2] travels too.

    The band edges and quasi-Fermi levels Ec, Ev, Efn, Efp [eV] always
    travel, from extract/bands.py, NaN at oxide nodes.

    The mesh axes travel with the fields because the client cannot know them
    otherwise, and because the mesh is graded: a plot against node number
    rather than against position would compress the junction into nothing,
    which is the one place anyone is looking.
    """
    mesh: Any = device.mesh
    axes: tuple[tuple[str, str, npt.NDArray[np.float64]], ...]
    if hasattr(mesh, "x_axis"):
        shape: tuple[int, ...] = (mesh.ny, mesh.nx)
        axes = (
            ("x", "cm", np.asarray(mesh.x_axis.x, dtype=np.float64)),
            ("y", "cm", np.asarray(mesh.y_axis.x, dtype=np.float64)),
        )
    else:
        shape = (mesh.n_nodes,)
        axes = (("x", "cm", np.asarray(mesh.x, dtype=np.float64)),)

    bands = band_edges(device, state)
    arrays: list[tuple[str, str, npt.NDArray[np.float64]]] = [
        *axes,
        ("psi", "V", _physical(state.psi, device)),
        ("n", "cm^-3", _physical(state.n, device)),
        ("p", "cm^-3", _physical(state.p, device)),
        ("Ec", "eV", bands.Ec),
        ("Ev", "eV", bands.Ev),
        ("Efn", "eV", bands.Efn),
        ("Efp", "eV", bands.Efp),
    ]
    if models is not None:
        Jx, Jy = node_current_density(device, state, models)
        if len({c.voltage for c in device.semiconductor_contacts}) <= 1:
            Jx, Jy = np.zeros_like(Jx), np.zeros_like(Jy)
        arrays += [("Jx", "A/cm^2", Jx), ("Jy", "A/cm^2", Jy)]

    return FieldFrame(index=index, voltage=voltage, shape=shape, arrays=tuple(arrays))
