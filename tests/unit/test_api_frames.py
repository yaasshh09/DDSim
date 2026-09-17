"""What crosses the socket, ddsim/api/frames.py.

Two kinds of message, for the reason phases/PHASE-7.md gives. A telemetry
frame is a handful of scalars and goes as JSON text. A field is 20000 points
per array on a 200 by 100 device, where JSON costs roughly ten times the bytes
and parses slowly enough to be visible, so it goes as float32 behind a small
JSON header.

Two things here are less obvious than the byte layout and both are about
honesty rather than speed. JSON has no infinity, and a diverged update is
exactly the number a reader most wants; and every array crosses in physical
units, because a potential scaled by V_T is a number 38.7 times too large and
nothing on the wire says so.
"""

from __future__ import annotations

import json
import math
import struct

import numpy as np
import pytest

from ddsim.api.frames import FieldFrame, decode_fields, encode, field_frame
from ddsim.device.equilibrium import solve_equilibrium
from ddsim.device.mos_cap import mos_cap
from ddsim.device.mosfet import nmos
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import TransportModels, solve_bias_ramped
from ddsim.extract.bands import band_edges
from ddsim.extract.cv import CVFrame
from ddsim.extract.iv import IVFrame
from ddsim.solve.continuation import ContinuationEvent
from ddsim.solve.gummel import GummelIteration
from ddsim.solve.newton import NewtonIteration

COARSE_FET = {
    "n_contact": 4,
    "n_sd": 10,
    "n_channel": 12,
    "n_silicon": 29,
    "n_oxide": 4,
    "h_min_x": 5e-7,
    "h_min_y": 1e-7,
    "drain_voltage": 0.05,
}
"""The coarse MOSFET the unit tests share."""


def diode():
    return pn_diode(n_nodes=61, h_min=5e-7)


def as_json(frame) -> dict:
    message = encode(frame)
    assert isinstance(message, str)
    return json.loads(message)


def header_of(message: bytes) -> dict:
    (length,) = struct.unpack_from("<I", message, 0)
    return json.loads(message[4 : 4 + length])


# ------------------------------------------------------------ telemetry text


def test_a_newton_iteration_crosses_as_the_numbers_it_carries() -> None:
    body = as_json(
        NewtonIteration(
            iteration=3, residual=1.5e-7, update=2.0e-4, damping=0.5, limited=True
        )
    )

    assert body == {
        "type": "newton",
        "iteration": 3,
        "residual": 1.5e-7,
        "update": 2.0e-4,
        "damping": 0.5,
        "limited": True,
        "residual_by_family": None,
        "update_by_family": None,
    }


def test_a_newton_iteration_carries_its_split_by_equation_family() -> None:
    """So the page can draw psi, n and p apart and name the one that stalled.
    A family that went non finite crosses as null like any other number."""
    body = as_json(
        NewtonIteration(
            iteration=2,
            residual=math.inf,
            update=1e-3,
            damping=1.0,
            limited=False,
            residual_by_family={"psi": 1e-9, "n": math.inf, "p": 3e-7},
            update_by_family={"psi": 1e-3, "n": 2e-4, "p": 5e-5},
        )
    )

    assert body["residual_by_family"] == {"psi": 1e-9, "n": None, "p": 3e-7}
    assert body["update_by_family"] == {"psi": 1e-3, "n": 2e-4, "p": 5e-5}


def test_the_first_newton_iteration_says_there_was_no_step() -> None:
    """None rather than zero. A zero on the update plot is a step that was
    taken and went nowhere, which is not what iteration 0 is."""
    body = as_json(
        NewtonIteration(
            iteration=0, residual=8.0, update=None, damping=None, limited=False
        )
    )

    assert body["update"] is None
    assert body["damping"] is None


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_a_number_that_is_not_finite_crosses_as_null(value) -> None:
    """A diverged Gummel update is not finite, and that is precisely the frame
    a reader needs. JSON has no Infinity and no NaN: json.dumps writes both as
    bare words that JSON.parse refuses, so a naive encoder breaks the stream
    on the one iteration that mattered."""
    message = encode(GummelIteration(iteration=4, update=value))

    assert "Infinity" not in message
    assert "NaN" not in message
    assert json.loads(message)["update"] is None


def test_a_gummel_cycle_crosses_as_a_cycle() -> None:
    assert as_json(GummelIteration(iteration=2, update=3.0e-3)) == {
        "type": "gummel",
        "iteration": 2,
        "update": 3.0e-3,
    }


def test_a_continuation_attempt_carries_whether_it_was_accepted() -> None:
    body = as_json(
        ContinuationEvent(parameter=0.35, step=0.05, converged=False, message="halving")
    )

    assert body == {
        "type": "continuation",
        "parameter": 0.35,
        "step": 0.05,
        "converged": False,
        "message": "halving",
    }


def test_a_finished_current_point_crosses_as_a_point() -> None:
    assert as_json(IVFrame(index=2, voltage=0.3, current=1.25e-4)) == {
        "type": "point",
        "index": 2,
        "voltage": 0.3,
        "current": 1.25e-4,
    }


def test_a_finished_capacitance_point_crosses_as_its_own_kind() -> None:
    """A capacitance is not a current and a plot that treated the two as one
    stream would draw a C-V on the I-V axes."""
    body = as_json(
        CVFrame(index=0, gate_voltage=-1.0, capacitance=3.4e-8, charge=1e-8)
    )

    assert body["type"] == "cv_point"
    assert body["capacitance"] == 3.4e-8


def test_something_that_is_not_a_frame_is_refused() -> None:
    """Refused rather than skipped. A frame type added to the solver and not
    added here would otherwise vanish, and the plot would be missing points
    with nothing to say it was."""
    with pytest.raises(TypeError, match="cannot be sent"):
        encode({"iteration": 1})


# --------------------------------------------------------------- field bytes


def test_a_field_frame_is_a_json_header_followed_by_float32() -> None:
    device = diode()
    state = solve_equilibrium(device)

    message = encode(field_frame(device, state, index=0, voltage=0.0))

    assert isinstance(message, bytes)
    header = header_of(message)
    (length,) = struct.unpack_from("<I", message, 0)
    payload = message[4 + length :]

    assert header["type"] == "fields"
    total = sum(array["length"] for array in header["arrays"])
    assert len(payload) == 4 * total


@pytest.mark.parametrize("index", [0, 10, 100, 1000])
def test_the_float32_payload_starts_on_a_four_byte_boundary(index) -> None:
    """The browser reads the payload as a Float32Array view, and that throws
    unless the offset is a multiple of 4. numpy's frombuffer does not care, so
    decode_fields passed while the page drew nothing. Four index widths make
    four consecutive header lengths, which covers every residue."""
    frame = FieldFrame(
        index=index,
        voltage=0.5,
        shape=(3,),
        arrays=(("psi", "V", np.array([0.1, 0.2, 0.3])),),
    )

    message = encode(frame)
    (length,) = struct.unpack_from("<I", message, 0)

    assert (4 + length) % 4 == 0
    assert header_of(message)["index"] == index
    np.testing.assert_allclose(
        decode_fields(message)["psi"], [0.1, 0.2, 0.3], rtol=1e-6
    )


def test_the_arrays_arrive_in_the_order_the_header_lists_them() -> None:
    """The header is the only thing telling the client where one array ends,
    so a payload in a different order is silently the wrong picture."""
    device = diode()
    state = solve_equilibrium(device)

    arrays = decode_fields(encode(field_frame(device, state, index=0, voltage=0.0)))

    assert list(arrays) == ["x", "psi", "n", "p", "Ec", "Ev", "Efn", "Efp"]
    np.testing.assert_allclose(arrays["x"], device.mesh.x, rtol=1e-6)


def test_the_fields_cross_in_physical_units() -> None:
    """psi on a DeviceState is scaled by V_T. Sent as it sits, a 0.9 V
    built in potential reads as 34.8 and the band diagram is nonsense."""
    device = diode()
    state = solve_equilibrium(device)

    arrays = decode_fields(encode(field_frame(device, state, index=0, voltage=0.0)))

    np.testing.assert_allclose(
        arrays["psi"], state.psi.to_physical(device.scale).data, rtol=1e-6
    )
    assert np.max(np.abs(arrays["psi"])) < 5.0
    assert np.max(arrays["n"]) > 1e15


def test_a_one_dimensional_device_says_it_has_one_axis() -> None:
    device = diode()
    state = solve_equilibrium(device)

    header = header_of(encode(field_frame(device, state, index=0, voltage=0.0)))

    assert header["shape"] == [device.mesh.n_nodes]


def test_a_two_dimensional_device_carries_both_axes_and_its_shape() -> None:
    """The mesh is a tensor product, so a field is an image and the client can
    contour it without being sent a triangulation. The node order is the one
    mesh2d builds: x fastest, which makes the image (ny, nx)."""
    device = nmos(**COARSE_FET)
    state = solve_bias_ramped(device)

    message = encode(field_frame(device, state, index=0, voltage=0.0))
    header = header_of(message)
    arrays = decode_fields(message)

    assert header["shape"] == [device.mesh.ny, device.mesh.nx]
    assert list(arrays) == ["x", "y", "psi", "n", "p", "Ec", "Ev", "Efn", "Efp"]
    assert arrays["psi"].size == device.mesh.ny * device.mesh.nx
    np.testing.assert_allclose(arrays["x"], device.mesh.x_axis.x, rtol=1e-6)
    np.testing.assert_allclose(arrays["y"], device.mesh.y_axis.x, rtol=1e-6)


def test_a_field_frame_says_which_bias_it_is_of() -> None:
    """A profile plot with no bias on it is a profile of nothing in
    particular, and on a sweep there are as many as there are points."""
    device = mos_cap(gate_voltage=-1.0)
    state = solve_equilibrium(device)

    header = header_of(encode(field_frame(device, state, index=3, voltage=-1.0)))

    assert header["index"] == 3
    assert header["voltage"] == -1.0


def test_the_units_of_every_array_travel_with_it() -> None:
    """The client labels an axis from this rather than from a list of its own,
    which is the same argument the device registry makes about defaults."""
    device = diode()
    state = solve_equilibrium(device)

    header = header_of(encode(field_frame(device, state, index=0, voltage=0.0)))
    units = {array["name"]: array["unit"] for array in header["arrays"]}

    assert units == {
        "x": "cm",
        "psi": "V",
        "n": "cm^-3",
        "p": "cm^-3",
        "Ec": "eV",
        "Ev": "eV",
        "Efn": "eV",
        "Efp": "eV",
    }


def test_a_transport_frame_carries_the_node_currents() -> None:
    device = nmos(**COARSE_FET)
    state = solve_bias_ramped(device)
    models = TransportModels.for_device(device)

    arrays = decode_fields(
        encode(field_frame(device, state, index=0, voltage=0.0, models=models))
    )

    assert list(arrays)[-2:] == ["Jx", "Jy"]
    assert arrays["Jx"].size == device.mesh.nx * device.mesh.ny


def test_the_band_edges_cross_in_ev() -> None:
    device = diode()
    state = solve_equilibrium(device)

    arrays = decode_fields(encode(field_frame(device, state, index=0, voltage=0.0)))

    np.testing.assert_allclose(arrays["Ec"], band_edges(device, state).Ec, rtol=1e-6)
