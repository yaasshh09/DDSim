"""Jobs, the unit of work the browser submits.

phases/PHASE-7.md: a solve is a job, not a request. A MOSFET sweep is minutes
of wall clock across dozens of bias points, each one a continuation ladder of
Newton solves, and request and response cannot express that. So the work runs
on a worker thread, its telemetry goes into a queue the reader drains while it
runs, and it can be cancelled.

Nothing here knows what a frame is. The work is handed a send and whatever it
sends is what a reader gets, which keeps this module testable without a device
and leaves the shape of the telemetry to the layer above.

Cancellation rides on send. A solve is a tight numerical loop that never looks
up except to report an iteration, so the report is the only place a stop can be
noticed without threading a flag through every signature in the solver. send
raises CancelledError once the job is cancelled, the work is free to catch it and
put a partial result down tidily, and the thread ends. A solve that has stopped
reporting cannot be cancelled until it reports again. That is a real limit and
it belongs in the README rather than hidden here.

The queue is bounded and drops its oldest frame when full. The solver is never
made to wait on a slow reader, which is phases/PHASE-7.md's rule for field
frames and costs nothing to apply to all of them. Dropped frames are counted,
because a stream that quietly loses points is a plot that quietly lies.
"""

from __future__ import annotations

import queue
import threading
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

DEFAULT_QUEUE_SIZE = 4096
"""Frames held for a reader that has not arrived yet. A Newton solve emits one
frame per iteration and a bias point is tens of those, so this is minutes of
telemetry rather than seconds."""


class CancelledError(Exception):
    """Raised inside the work, at its next frame, once the job is cancelled.

    An exception rather than a return code, so that a solve deep inside a
    continuation ladder unwinds the whole stack in one go, and so that work
    holding a partial result can catch it, put the result down and re-raise.
    """


class JobStatus(Enum):
    """Where a job is. The last three are terminal."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL = (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)

_END = object()
"""Sentinel closing a frame stream. Every job puts exactly one, whichever way
it ended, so a reader always stops."""

Send = Callable[[Any], None]
Work = Callable[[Send], Any]
"""The work a job runs. It is handed a send and reports through it."""


@dataclass
class Job:
    """One submitted piece of work and the state a reader needs."""

    id: str
    """Opaque and unique. The client's handle on the job."""

    status: JobStatus = JobStatus.PENDING

    message: str = ""
    """Why it failed, when it did."""

    dropped: int = 0
    """Frames discarded because the reader was behind. Reported rather than
    swallowed."""

    frames: queue.Queue[Any] = field(default_factory=queue.Queue)
    cancelling: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)


class JobRegistry:
    """Every job this process is running, by id.

    One process, one registry, no database and no accounts. phases/PHASE-7.md
    is explicit that this is an instrument and not a service.
    """

    def __init__(self, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        """Args:
        queue_size: frames held per job before the oldest is dropped.
        """
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue_size = queue_size

    def submit(self, work: Work) -> Job:
        """Start the work on a worker thread and return its job at once.

        The return is immediate by design. The browser needs its job id while
        the sweep is still running, not after it.
        """
        # One slot past the frame capacity, reserved for the end marker, so
        # that a stream nobody is draining still closes. Dropping a frame to
        # make room for the marker would lose the last frame before the end,
        # which is the one a reader most wants.
        job = Job(
            id=uuid.uuid4().hex, frames=queue.Queue(maxsize=self._queue_size + 1)
        )
        with self._lock:
            self._jobs[job.id] = job

        thread = threading.Thread(
            target=self._run,
            args=(job, work),
            name=f"ddsim-job-{job.id[:8]}",
            daemon=True,
        )
        thread.start()
        return job

    def _run(self, job: Job, work: Work) -> None:
        job.status = JobStatus.RUNNING
        try:
            work(lambda frame: self._send(job, frame))
        except CancelledError:
            job.status = JobStatus.CANCELLED
        except Exception as error:  # noqa: BLE001
            # A failed solve is a result to report, not a crash to lose. The
            # browser shows the reason rather than a socket that went quiet.
            job.status = JobStatus.FAILED
            job.message = f"{type(error).__name__}: {error}"
        else:
            job.status = JobStatus.DONE
        finally:
            # Never a blocking put. The queue is full exactly when nobody is
            # draining it, which is the case where a blocking put here would
            # hang the worker forever and the job would never reach a terminal
            # state. The reserved slot is what makes this fit.
            job.frames.put_nowait(_END)
            job.finished.set()

    def _send(self, job: Job, frame: Any) -> None:
        """Queue one frame, dropping the oldest rather than waiting.

        A solver made to wait on a slow reader is a solve whose wall clock
        depends on the browser, which phases/PHASE-7.md rules out. Only the
        worker thread puts, so comparing against the capacity here cannot race
        with another writer, and a reader draining in between can only make
        room that this drop did not need.
        """
        if job.cancelling.is_set():
            raise CancelledError(f"job {job.id} was cancelled")
        if job.frames.qsize() >= self._queue_size:
            try:
                job.frames.get_nowait()
                job.dropped += 1
            except queue.Empty:  # pragma: no cover - the reader just drained it
                pass
        job.frames.put_nowait(frame)

    def _job(self, job_id: str) -> Job:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"no job {job_id!r}")
            return self._jobs[job_id]

    def status(self, job_id: str) -> JobStatus:
        """Where the job is now."""
        return self._job(job_id).status

    def message(self, job_id: str) -> str:
        """Why the job failed, or empty."""
        return self._job(job_id).message

    def dropped(self, job_id: str) -> int:
        """How many frames were discarded because the reader was behind."""
        return self._job(job_id).dropped

    def cancel(self, job_id: str) -> bool:
        """Ask the job to stop at its next frame.

        Returns whether anything was actually asked to stop. A job that has
        already finished is left alone and reported as such: the click and the
        last bias point can land in either order, and telling the browser that
        a finished sweep was cancelled would throw away a result that exists.
        """
        job = self._job(job_id)
        if job.status in _TERMINAL:
            return False
        job.cancelling.set()
        return True

    def frames(self, job_id: str, timeout: float | None = None) -> Iterator[Any]:
        """Yield frames as they are produced, ending when the job does.

        Args:
            job_id: the job to read.
            timeout: seconds to wait for any one frame, or None to wait as
                long as it takes. A Newton solve on a fine mesh can be quiet
                for a while, so a timeout here is a test convenience rather
                than a health check.
        """
        job = self._job(job_id)
        while True:
            try:
                frame = job.frames.get(timeout=timeout)
            except queue.Empty as error:
                raise TimeoutError(
                    f"job {job_id} sent no frame within {timeout} s"
                ) from error
            if frame is _END:
                return
            yield frame

    def wait(self, job_id: str, timeout: float | None = None) -> JobStatus:
        """Block until the job reaches a terminal status, and return it."""
        job = self._job(job_id)
        if not job.finished.wait(timeout=timeout):
            raise TimeoutError(f"job {job_id} did not finish within {timeout} s")
        return job.status
