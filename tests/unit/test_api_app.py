"""The HTTP and WebSocket surface, ddsim/api/app.py.

Contract tests with no browser in them, which is what phases/PHASE-7.md asks
for first. Three things here are acceptance criteria rather than conveniences
and they are marked where they appear: telemetry arrives before the job is
finished, a cancel actually stops the solve, and what the browser gets is bit
for bit what pytest gets from the same call.

The devices are coarse on purpose. What is under test is the layer, and a fine
mesh would only make a failure slower to reach.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ddsim.api.app import create_app
from ddsim.api.frames import decode_fields
from ddsim.api.jobs import JobRegistry, JobStatus
from ddsim.device.mos_cap import mos_cap
from ddsim.device.mosfet import nmos
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import TransportModels
from ddsim.extract.cv import cv_sweep
from ddsim.extract.iv import gate_sweep, iv_sweep

DIODE = {"n_nodes": 61, "h_min": 5e-7}
"""A coarse diode, the same one the other api tests use."""

FET = {
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

LONG = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
"""Enough bias points that the job is still running when the first frame of
its stream arrives, which is what the live telemetry tests assert."""


@pytest.fixture
def client():
    with TestClient(create_app()) as client:
        yield client


def diode_request(voltages=(0.0, 0.2), **sweep: Any) -> dict:
    return {
        "device": {"kind": "pn_diode", "parameters": DIODE},
        "sweep": {
            "kind": "iv",
            "contact": "anode",
            "voltages": list(voltages),
            **sweep,
        },
    }


def submit(client, request: dict) -> str:
    response = client.post("/api/jobs", json=request)
    assert response.status_code == 200, response.text
    return response.json()["id"]


def drain(client, job_id: str) -> list[Any]:
    """Every frame of a stream, as the client would read them."""
    frames: list[Any] = []
    with client.websocket_connect(f"/api/jobs/{job_id}/stream") as socket:
        while True:
            frame = json.loads(socket.receive_text())
            frames.append(frame)
            if frame["type"] == "status":
                return frames


# --------------------------------------------------------------- the schema


def test_the_schema_offers_the_devices_the_registry_knows(client) -> None:
    body = client.get("/api/schema").json()

    assert set(body["devices"]) == {"pn_diode", "mos_cap", "nmos"}
    assert set(body["sweeps"]) == {"iv", "transfer", "cv"}


def test_the_schema_carries_the_constructors_own_defaults(client) -> None:
    """The browser builds its form from this, so a default here that is not
    the constructor's is a browser simulating something else."""
    body = client.get("/api/schema").json()
    knobs = {p["name"]: p for p in body["devices"]["pn_diode"]}

    assert knobs["Na"]["default"] == pytest.approx(1e16)
    assert knobs["n_nodes"]["type"] == "int"


def test_the_schema_carries_the_model_flags_and_their_choices(client) -> None:
    """The flags decide whether a MOSFET has velocity saturation in it, so
    they belong on the form rather than in a default nobody sees."""
    body = client.get("/api/schema").json()
    flags = {p["name"]: p for p in body["models"]}

    assert flags["mobility"]["choices"] == ["constant", "arora"]
    assert flags["field_dependent"]["default"] is False


# -------------------------------------------------------- refusing a request


def test_an_unknown_device_is_refused_with_the_known_ones_named(client) -> None:
    response = client.post(
        "/api/jobs",
        json={
            "device": {"kind": "bjt", "parameters": {}},
            "sweep": {"kind": "iv", "contact": "anode", "voltages": [0.1]},
        },
    )

    assert response.status_code == 400
    assert "pn_diode" in response.json()["detail"]


def test_a_parameter_of_the_wrong_type_is_refused(client) -> None:
    """61.5 nodes is not a mesh. Refused here rather than a mesh builder
    failing decades away from the field that caused it."""
    response = client.post(
        "/api/jobs",
        json={
            "device": {"kind": "pn_diode", "parameters": {"n_nodes": 61.5}},
            "sweep": {"kind": "iv", "contact": "anode", "voltages": [0.1]},
        },
    )

    assert response.status_code == 400
    assert "n_nodes" in response.json()["detail"]


def test_a_contact_the_device_does_not_have_is_refused(client) -> None:
    response = client.post(
        "/api/jobs",
        json={
            "device": {"kind": "pn_diode", "parameters": DIODE},
            "sweep": {"kind": "iv", "contact": "gate", "voltages": [0.1]},
        },
    )

    assert response.status_code == 400
    assert "gate" in response.json()["detail"]


def test_a_request_missing_its_sweep_is_refused_by_the_schema(client) -> None:
    """Pydantic's own 422 rather than a 500 from a KeyError."""
    response = client.post("/api/jobs", json={"device": {"kind": "pn_diode"}})

    assert response.status_code == 422


def test_an_unknown_job_is_not_found(client) -> None:
    assert client.get("/api/jobs/nosuchjob").status_code == 404


# ------------------------------------------------------------- the telemetry


def test_a_stream_carries_solver_frames_and_then_a_status(client) -> None:
    job = submit(client, diode_request())
    frames = drain(client, job)

    kinds = {frame["type"] for frame in frames}
    assert "gummel" in kinds
    assert "point" in kinds
    assert frames[-1]["type"] == "status"
    assert frames[-1]["status"] == JobStatus.DONE.value


def test_telemetry_arrives_before_the_job_is_finished(client) -> None:
    """An acceptance criterion of phases/PHASE-7.md. The residual plot has to
    move while the solve runs, and a page that renders after it finishes is a
    spinner with extra steps."""
    job = submit(client, diode_request(voltages=LONG))

    with client.websocket_connect(f"/api/jobs/{job}/stream") as socket:
        first = json.loads(socket.receive_text())
        still_going = client.get(f"/api/jobs/{job}").json()["status"]

    assert first["type"] != "status"
    assert still_going == JobStatus.RUNNING.value


def test_a_stalled_sweep_is_reported_as_the_measurement_it_is(client) -> None:
    """Not as a failure. phases/PHASE-2.md asks for the bias where Gummel
    gives up, and the curve carries it with complete false."""
    job = submit(
        client,
        diode_request(
            voltages=(0.2, 0.4, 5.0),
            settings={"step": 0.2, "max_iterations": 8},
        ),
    )
    frames = drain(client, job)

    assert frames[-1]["status"] == JobStatus.DONE.value
    curve = client.get(f"/api/jobs/{job}/result").json()
    assert curve["complete"] is False
    assert curve["message"]


def test_a_device_that_cannot_be_built_is_refused_before_it_is_a_job(
    client,
) -> None:
    """Five nodes on a graded mesh is not a mesh. The refusal is the mesh
    builder's own, and it arrives as a 400 rather than as a job that starts
    and dies, which is the whole point of validating before submitting."""
    response = client.post(
        "/api/jobs",
        json={
            "device": {"kind": "pn_diode", "parameters": {"n_nodes": 5}},
            "sweep": {"kind": "iv", "contact": "anode", "voltages": [0.1]},
        },
    )

    assert response.status_code == 400
    assert "max_ratio" in response.json()["detail"]


def test_a_failing_solve_says_why_rather_than_going_quiet(client) -> None:
    """A sweep begun where nothing converges. Every point is continued from
    the first one, so there is nothing to continue from and the sweep raises
    rather than returning an empty curve. The browser shows the reason;
    phases/PHASE-7.md forbids reporting a success that never came."""
    job = submit(
        client,
        diode_request(
            voltages=(5.2,), settings={"start": 5.0, "max_iterations": 8}
        ),
    )
    frames = drain(client, job)

    assert frames[-1]["status"] == JobStatus.FAILED.value
    assert "could not be started" in frames[-1]["message"]


# -------------------------------------------------------------- cancellation


def test_cancelling_stops_the_solve(client) -> None:
    """An acceptance criterion. The job reaches cancelled rather than running
    to the end of the sweep with nobody watching."""
    job = submit(client, diode_request(voltages=LONG))

    with client.websocket_connect(f"/api/jobs/{job}/stream") as socket:
        json.loads(socket.receive_text())
        assert client.post(f"/api/jobs/{job}/cancel").json()["cancelled"]
        while True:
            frame = json.loads(socket.receive_text())
            if frame["type"] == "status":
                break

    assert frame["status"] == JobStatus.CANCELLED.value
    assert client.get(f"/api/jobs/{job}").json()["status"] == (
        JobStatus.CANCELLED.value
    )


def test_cancelling_a_finished_job_says_it_changed_nothing(client) -> None:
    """The click and the last bias point can land in either order, and
    telling the browser a finished sweep was cancelled would throw away a
    result that exists."""
    job = submit(client, diode_request(voltages=(0.1,)))
    drain(client, job)

    assert client.post(f"/api/jobs/{job}/cancel").json()["cancelled"] is False


# ---------------------------------------------------------------- the curve


def test_the_result_is_available_after_the_stream_has_closed(client) -> None:
    """The frame queue may drop its oldest frame, so the curve is fetched
    rather than only streamed. A browser that connected late still gets it."""
    job = submit(client, diode_request())
    drain(client, job)

    curve = client.get(f"/api/jobs/{job}/result").json()

    assert curve["kind"] == "iv"
    assert [point["voltage"] for point in curve["points"]] == [0.0, 0.2]


def test_a_result_asked_for_too_early_is_refused(client) -> None:
    """Rather than an empty curve, which reads like a device with no current
    in it."""
    job = submit(client, diode_request(voltages=LONG))

    assert client.get(f"/api/jobs/{job}/result").status_code == 409


@pytest.mark.parametrize(
    ("request_body", "direct"),
    [
        (
            diode_request(voltages=(0.0, 0.2)),
            lambda: iv_sweep(
                pn_diode(**DIODE),
                "anode",
                [0.0, 0.2],
                models=TransportModels.for_device(pn_diode(**DIODE)),
            ),
        ),
        (
            {
                "device": {"kind": "mos_cap", "parameters": {}},
                "sweep": {
                    "kind": "cv",
                    "contact": "gate",
                    "voltages": [-1.0, 0.0, 1.0],
                },
            },
            lambda: cv_sweep(mos_cap(), "gate", [-1.0, 0.0, 1.0]),
        ),
        (
            {
                "device": {"kind": "nmos", "parameters": FET},
                "sweep": {
                    "kind": "transfer",
                    "contact": "gate",
                    "voltages": [0.2, 0.4],
                    "settings": {"step": 0.2},
                },
            },
            lambda: gate_sweep(
                nmos(**FET),
                [0.2, 0.4],
                step=0.2,
                models=TransportModels.for_device(nmos(**FET)),
            ),
        ),
    ],
    ids=["diode", "capacitor", "mosfet"],
)
def test_the_browser_gets_bit_for_bit_what_pytest_gets(
    client, request_body, direct
) -> None:
    """The acceptance criterion for all three device classes. JSON carries a
    double exactly, so this is equality and not a tolerance."""
    job = submit(client, request_body)
    drain(client, job)
    served = client.get(f"/api/jobs/{job}/result").json()

    expected = direct()
    is_cv = served["kind"] == "cv"
    values = "capacitance" if is_cv else "current"

    assert [point["voltage"] for point in served["points"]] == list(
        expected.gate_voltage if is_cv else expected.voltage
    )
    assert [point[values] for point in served["points"]] == list(
        expected.capacitance if is_cv else expected.current
    )


# --------------------------------------------------------------- the fields


def test_the_fields_of_a_point_come_back_as_float32_behind_a_header(
    client,
) -> None:
    """Over HTTP rather than the socket. A field is asked for when someone
    drags to a point, which can be long after the solve finished, and asking
    for one must never be able to hold up a solver."""
    job = submit(client, diode_request())
    drain(client, job)

    response = client.get(f"/api/jobs/{job}/fields/1")

    assert response.status_code == 200
    arrays = decode_fields(response.content)
    assert list(arrays) == [
        "x", "psi", "n", "p", "Ec", "Ev", "Efn", "Efp", "Jx", "Jy"
    ]
    assert np.max(arrays["n"]) > 1e15


def test_the_fields_of_a_capacitance_point_carry_its_gate_bias(client) -> None:
    """A C-V point calls its bias gate_voltage and an I-V point calls it
    voltage. The wire calls both voltage, so this is the branch that would
    otherwise label every capacitance profile with the wrong bias."""
    job = submit(
        client,
        {
            "device": {"kind": "mos_cap", "parameters": {}},
            "sweep": {"kind": "cv", "contact": "gate", "voltages": [-1.0, 1.0]},
        },
    )
    drain(client, job)

    response = client.get(f"/api/jobs/{job}/fields/1")
    header = json.loads(
        response.content[4 : 4 + int.from_bytes(response.content[:4], "little")]
    )

    assert header["voltage"] == 1.0
    assert header["index"] == 1


def test_the_fields_of_a_point_that_was_never_solved_are_not_found(
    client,
) -> None:
    job = submit(client, diode_request())
    drain(client, job)

    assert client.get(f"/api/jobs/{job}/fields/9").status_code == 404


def test_the_fields_of_a_running_job_are_refused(client) -> None:
    """There is no converged state at a point that has not been reached, and
    a plot of the guess would look like an answer."""
    job = submit(client, diode_request(voltages=LONG))

    assert client.get(f"/api/jobs/{job}/fields/0").status_code == 409


# ----------------------------------------------------------------- the page


def test_the_page_is_served_from_the_root(client) -> None:
    """One command and a working page, which phases/PHASE-7.md asks for. No
    build step, so the page is a file this repo already contains."""
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_the_page_is_revalidated_rather_than_cached(client) -> None:
    """The client and the wire format ship together. A browser that keeps an
    old page after the package changes draws with yesterday's reader, and the
    first time I opened this page it did exactly that."""
    response = client.get("/")

    assert response.headers["cache-control"] == "no-cache"


def test_shutting_the_app_down_cancels_what_is_still_solving() -> None:
    """Ctrl+C on `ddsim serve` mid sweep, or a test that walks away from a
    long job. Either way the solver thread must not outlive the app."""
    registry = JobRegistry()
    with TestClient(create_app(registry)) as client:
        job = submit(client, diode_request(voltages=LONG))

    assert registry.status(job) is JobStatus.CANCELLED


def test_client_scripts_are_served_and_revalidated(client) -> None:
    """The page loads its script from /static. Revalidated for the same
    reason as the page: the script and the wire format ship together."""
    response = client.get("/static/js/app.js")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert response.headers["cache-control"] == "no-cache"


def test_the_page_loads_its_script_rather_than_inlining_it(client) -> None:
    page = client.get("/").text

    assert '<script src="/static/js/app.js"></script>' in page
