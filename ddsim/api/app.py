"""The HTTP and WebSocket surface, phases/PHASE-7.md.

One local process, no database, no accounts, bound to localhost by whoever
starts it. An instrument rather than a service, and the README says so.

Three routes carry the work and the split between them is the whole design.

**POST /api/jobs** validates everything it can without solving and answers a
bad request with a refusal rather than a job that starts and dies. What it
returns is a job id, immediately, because a MOSFET sweep is minutes long.

**WS /api/jobs/{id}/stream** carries telemetry as it happens: Newton
iterations, continuation attempts, finished sweep points, and a final status.
Text only. It never carries a field, so a slow reader cannot make a solver
wait, and a client is free to connect late, disconnect and come back.

**GET /api/jobs/{id}/fields/{index}** carries one converged state as float32.
Over HTTP rather than the socket because it is a question with an answer:
someone dragged to a bias point and wants the profile there, which can happen
long after the solve finished.

Nothing here computes a physical quantity. It builds a device, calls the
sweep the CLI would call, and forwards what comes back.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

from ddsim.api.devices import (
    COARSE,
    DEVICE_KINDS,
    build_from_spec,
    device_dimension,
    device_parameters,
    region_defaults,
)
from ddsim.api.frames import (
    Status,
    curve_body,
    encode,
    field_frame,
    point_voltage,
)
from ddsim.api.jobs import JobRegistry, JobStatus, Send
from ddsim.api.learn import (
    KNOB_TOPICS,
    PLOT_TOPICS,
    STATUS_TOPICS,
    lesson_names,
    load_lesson,
    load_topic,
    topic_names,
)
from ddsim.api.sweeps import (
    SWEEP_KINDS,
    check_request,
    model_parameters,
    run_sweep,
    sweep_parameters,
)
from ddsim.device.builder import Device
from ddsim.device.transport import TransportModels
from ddsim.extract.cv import CVCurve
from ddsim.extract.iv import IVCurve

SHUTDOWN_TIMEOUT = 30.0
"""How long shutdown waits for running solves to reach their next frame [s].
One Newton iteration on the finest MOSFET mesh is well inside this."""

PAGE = Path(__file__).parent / "static" / "index.html"
"""The client page. Its scripts sit next to it under static/js, with no build
step, which is what phases/PHASE-7.md means by someone runs one command and
has a working page."""


class _Revalidated(StaticFiles):
    """Static files the browser must check before reusing. The page, its
    scripts and the wire format change together, and a cached script is
    yesterday's reader."""

    async def get_response(self, path: str, scope: Any) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


_QUIET_POLL = 0.25
"""Seconds to wait for a frame before looking at the job's status [s].

Only needed for the case where the stream has already been read to its end by
someone else, which is what a reload looks like: the end marker is taken once
and a second reader would otherwise wait for a frame that cannot come.
"""

_TERMINAL = (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)

_ENDED = object()
_NOTHING_YET = object()


class DeviceSpec(BaseModel):
    """Which device, and the constructor arguments to build it with."""

    kind: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class SweepSpec(BaseModel):
    """Which sweep, over which terminal, at which biases."""

    kind: str
    contact: str
    voltages: list[float]
    settings: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, Any] = Field(default_factory=dict)
    measure_at: str | None = None


class JobRequest(BaseModel):
    """One press of solve."""

    device: DeviceSpec
    sweep: SweepSpec


@dataclass(frozen=True)
class Finished:
    """What a job hands back: the curve, and the device it was taken on.

    The device travels with the curve because a field message needs its mesh
    and its scale factors, and the curve's points carry only the states.
    """

    device: Device
    curve: IVCurve | CVCurve
    models: TransportModels | None
    """The transport models the curve was solved with, so the node currents
    of a field frame come from the same mobility and recombination. None for
    a C-V."""


def create_app(registry: JobRegistry | None = None) -> FastAPI:
    """The application, with its own job registry unless one is supplied.

    Args:
        registry: an existing registry, which the tests use to reach in. A
            fresh one per app otherwise, since one process is one instrument.
    """
    jobs = registry if registry is not None else JobRegistry()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        # Nothing solves after the app is gone. See JobRegistry.close.
        jobs.close(timeout=SHUTDOWN_TIMEOUT)

    app = FastAPI(title="DDSim", lifespan=lifespan)
    app.mount("/static", _Revalidated(directory=PAGE.parent), name="static")

    @app.get("/api/schema")
    def schema() -> dict[str, Any]:
        """Every knob the browser can render, read from the code that has them.

        There is no second copy of a default anywhere in the client. A knob
        added to nmos() appears on the form the moment it exists.
        """
        return {
            "devices": {
                kind: [_knob(p) for p in device_parameters(kind)]
                for kind in DEVICE_KINDS
            },
            # Which devices the page will solve live. Read from the mesh each
            # constructor builds, so a device that grew a second axis stops
            # being dragged rather than solving for minutes on every drag.
            "dimensions": {kind: device_dimension(kind) for kind in DEVICE_KINDS},
            # The regions each stack device starts from, drawn by the page as
            # rows rather than boxes. Only the devices built from regions.
            "regions": {
                kind: regions
                for kind in DEVICE_KINDS
                if (regions := region_defaults(kind)) is not None
            },
            # The coarse mesh on offer per device, with what it costs. Only
            # the devices that have one appear.
            "presets": {
                kind: {"parameters": dict(preset.parameters), "note": preset.note}
                for kind, preset in COARSE.items()
            },
            "sweeps": {
                kind: [_knob(p) for p in sweep_parameters(kind)] for kind in SWEEP_KINDS
            },
            "models": [_knob(p) for p in model_parameters()],
            "plots": dict(PLOT_TOPICS),
            "statuses": dict(STATUS_TOPICS),
        }

    @app.get("/api/learn")
    def topics() -> list[dict[str, str]]:
        """Every explanation, by name, title and one line."""
        return [
            {"name": t.name, "title": t.title, "summary": t.summary}
            for t in (load_topic(name) for name in topic_names())
        ]

    @app.get("/api/learn/{name}")
    def topic(name: str) -> dict[str, Any]:
        """One explanation, in its two layers, with the knobs it explains."""
        found = _found(lambda: load_topic(name))
        return {
            "name": found.name,
            "title": found.title,
            "summary": found.summary,
            "docs": list(found.docs),
            "plain": found.plain,
            "depth": found.depth,
            "knobs": sorted(k for k, t in KNOB_TOPICS.items() if t == found.name),
        }

    @app.get("/api/lessons")
    def lessons() -> list[dict[str, str]]:
        """Every guided experiment, by name, title and one line."""
        return [
            {"name": found.name, "title": found.title, "summary": found.summary}
            for found in (load_lesson(name) for name in lesson_names())
        ]

    @app.get("/api/lessons/{name}")
    def lesson(name: str) -> dict[str, Any]:
        """One guided experiment, with every request it sets up whole."""
        found = _found(lambda: load_lesson(name))
        return {
            "name": found.name,
            "title": found.title,
            "summary": found.summary,
            "claims": list(found.claims),
            "request": found.request,
            "mesh_note": found.mesh_note,
            "steps": [
                {"title": s.title, "text": s.text, "request": s.request}
                for s in found.steps
            ],
            "look_for": found.look_for,
            "explanation": found.explanation,
        }

    @app.post("/api/jobs")
    def submit(request: JobRequest) -> dict[str, str]:
        """Start a sweep and hand back its id at once.

        Everything that can be judged without solving is judged here, so a
        typo is a refusal naming the knob rather than a job that fails a
        second later with the same message somewhere less visible.
        """
        device = _checked(
            lambda: build_from_spec(request.device.kind, request.device.parameters)
        )
        sweep = request.sweep
        _checked(
            lambda: check_request(
                sweep.kind,
                device,
                sweep.contact,
                sweep.settings,
                sweep.models,
                sweep.measure_at,
            )
        )

        def work(send: Send) -> Finished:
            curve, models = run_sweep(
                sweep.kind,
                device,
                sweep.contact,
                sweep.voltages,
                settings=sweep.settings,
                models=sweep.models or None,
                measure_at=sweep.measure_at,
                on_frame=send,
            )
            return Finished(device=device, curve=curve, models=models)

        return {"id": jobs.submit(work).id}

    @app.get("/api/jobs/{job_id}")
    def status(job_id: str) -> dict[str, Any]:
        """Where the job is, and how much telemetry it lost on the way."""
        return {
            "id": job_id,
            "status": _found(lambda: jobs.status(job_id)).value,
            "message": jobs.message(job_id),
            "dropped": jobs.dropped(job_id),
        }

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict[str, bool]:
        """Ask the solve to stop at its next reported iteration.

        Reports whether anything was actually stopped. A sweep that had
        already finished is left alone, because telling the browser a finished
        curve was cancelled would throw away a result that exists.
        """
        return {"cancelled": _found(lambda: jobs.cancel(job_id))}

    @app.get("/api/jobs/{job_id}/result")
    def result(job_id: str) -> dict[str, Any]:
        """The finished curve, fetched rather than streamed.

        The frame queue may drop its oldest frame, so the curve a browser
        plots point by point is not the authority on what the sweep found.
        This is.
        """
        return curve_body(_finished(jobs, job_id).curve)

    @app.get("/api/jobs/{job_id}/fields/{index}")
    def fields(job_id: str, index: int) -> Response:
        """The fields at one solved bias point, as float32 with a header."""
        done = _finished(jobs, job_id)
        points = done.curve.points
        if not 0 <= index < len(points):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"this sweep reached {len(points)} points, so there is no "
                    f"point {index} to show a field at"
                ),
            )
        point = points[index]
        message = encode(
            field_frame(
                done.device,
                point.state,
                index=index,
                voltage=point_voltage(point),
                models=done.models,
            )
        )
        assert isinstance(message, bytes)
        return Response(content=message, media_type="application/octet-stream")

    @app.websocket("/api/jobs/{job_id}/stream")
    async def stream(socket: WebSocket, job_id: str) -> None:
        """Every frame of one job, then how it ended.

        The reads happen on a worker thread. A frame arrives from a queue the
        solver is filling, and blocking the event loop on that would stop this
        process serving anything else while a MOSFET converges.
        """
        await socket.accept()
        try:
            jobs.status(job_id)
        except KeyError as missing:
            await socket.close(code=1008, reason=str(missing))
            return

        while True:
            frame = await anyio.to_thread.run_sync(_poll, jobs, job_id)
            if frame is _ENDED:
                break
            if frame is _NOTHING_YET:
                # Nothing waiting and the job is over: someone else read this
                # stream to its end. There is nothing more coming.
                if jobs.status(job_id) in _TERMINAL:
                    break
                continue
            await _send(socket, encode(frame))

        await _send(
            socket,
            encode(
                Status(
                    status=jobs.status(job_id).value,
                    message=jobs.message(job_id),
                    dropped=jobs.dropped(job_id),
                )
            ),
        )

    @app.get("/")
    def page() -> FileResponse:
        """The client."""
        # The page and the wire format change together, so never a stale page.
        return FileResponse(
            PAGE, media_type="text/html", headers={"Cache-Control": "no-cache"}
        )

    return app


def _knob(parameter: Any) -> dict[str, Any]:
    """One settable knob as the form needs it."""
    return {
        "name": parameter.name,
        "default": parameter.default,
        "type": parameter.type,
        "choices": list(parameter.choices),
        "explanation": parameter.explanation,
        "unit": parameter.unit,
        "topic": KNOB_TOPICS.get(parameter.name, ""),
        # Null where the knob declares no range. The page reads that as no
        # slider rather than filling in ends of its own.
        "low": parameter.low,
        "high": parameter.high,
        "axis": parameter.axis,
    }


def _checked(call: Any) -> Any:
    """Run a validation and answer its refusal with a 400.

    The messages from api/devices.py and api/sweeps.py name the knob and list
    what was available, which is exactly what the person filling in the form
    needs. They are forwarded rather than replaced.
    """
    try:
        return call()
    except (ValueError, TypeError, KeyError) as refusal:
        raise HTTPException(status_code=400, detail=str(refusal)) from refusal


def _found(call: Any) -> Any:
    """Run a lookup and answer an unknown job with a 404."""
    try:
        return call()
    except KeyError as missing:
        raise HTTPException(status_code=404, detail=str(missing)) from missing


def _finished(jobs: JobRegistry, job_id: str) -> Finished:
    """The result of a job that has one, or a refusal saying why not.

    409 rather than an empty curve. A sweep still running has no answer yet,
    and a curve with no points in it reads like a device with no current.
    """
    done = _found(lambda: jobs.result(job_id))
    if done is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"job {job_id} is {jobs.status(job_id).value} and has no "
                f"curve to show. {jobs.message(job_id)}".strip()
            ),
        )
    assert isinstance(done, Finished)
    return done


def _poll(jobs: JobRegistry, job_id: str) -> Any:
    """The next frame, or a marker for ended and for nothing yet.

    A fresh iterator per call, which costs a dictionary lookup and keeps this
    function free of state. StopIteration is turned into a marker because it
    cannot cross back into a coroutine.
    """
    try:
        return next(jobs.frames(job_id, timeout=_QUIET_POLL))
    except StopIteration:
        return _ENDED
    except TimeoutError:
        return _NOTHING_YET


async def _send(socket: WebSocket, message: str | bytes) -> None:
    if isinstance(message, str):
        await socket.send_text(message)
    else:
        await socket.send_bytes(message)
