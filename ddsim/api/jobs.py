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
import time
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


class BusyError(Exception):
    """Raised by submit when as many jobs are running as the registry allows."""


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

    result: Any = None
    """Whatever the work returned, once it finished on its own.

    Held here rather than sent as a frame because the frame queue is allowed
    to drop its oldest entry, and a sweep's finished curve is the one thing a
    reader cannot afford to lose. Set only on the way to DONE: a failed or
    cancelled job stopped somewhere nobody chose, and its partial state is in
    the frames the reader already has.
    """

    started_at: float = 0.0
    """When it was submitted, on the registry's clock [s]."""

    ended_at: float | None = None
    """When it reached a terminal status, on the registry's clock [s]."""

    frames: queue.Queue[Any] = field(default_factory=queue.Queue)
    cancelling: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)


class JobRegistry:
    """Every job this process is running, by id.

    One process, one registry, no database and no accounts. phases/PHASE-7.md
    is explicit that this is an instrument and not a service. The three limits
    exist for when it is served publicly anyway, and each is off when None.
    """

    def __init__(
        self,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        max_running: int | None = None,
        keep_for: float | None = None,
        time_limit: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Args:
        queue_size: frames held per job before the oldest is dropped.
        max_running: jobs allowed to run at once. submit raises BusyError past it.
        keep_for: how long a finished job stays readable [s]. Older ones are
            dropped at the next submit.
        time_limit: wall clock a job may run before it is cancelled [s].
        clock: the time source [s], replaceable so tests need not sleep.
        """
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue_size = queue_size
        self._max_running = max_running
        self._keep_for = keep_for
        self._time_limit = time_limit
        self._clock = clock

    def submit(self, work: Work) -> Job:
        """Start the work on a worker thread and return its job at once.

        The return is immediate by design. The browser needs its job id while
        the sweep is still running, not after it.
        """
        job = Job(
            id=uuid.uuid4().hex,
            started_at=self._clock(),
            frames=queue.Queue(maxsize=self._queue_size + 1),
        )
        with self._lock:
            self._forget_old(job.started_at)
            running = sum(j.status not in _TERMINAL for j in self._jobs.values())
            if self._max_running is not None and running >= self._max_running:
                raise BusyError(
                    f"the server is already running {running} solves, which is "
                    "as many as it takes at once. Try again in a minute."
                )
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
            produced = work(lambda frame: self._send(job, frame))
        except CancelledError:
            job.status = JobStatus.CANCELLED
        except Exception as error:  # noqa: BLE001
            job.status = JobStatus.FAILED
            job.message = f"{type(error).__name__}: {error}"
        else:
            job.result = produced
            job.status = JobStatus.DONE
        finally:
            job.ended_at = self._clock()
            job.frames.put_nowait(_END)
            job.finished.set()

    def _forget_old(self, now: float) -> None:
        """Drop finished jobs older than keep_for. Called with the lock held."""
        if self._keep_for is None:
            return
        for job_id, job in list(self._jobs.items()):
            if job.ended_at is not None and now - job.ended_at > self._keep_for:
                del self._jobs[job_id]

    def _send(self, job: Job, frame: Any) -> None:
        """Queue one frame, dropping the oldest rather than waiting.

        A solver made to wait on a slow reader is a solve whose wall clock
        depends on the browser, which phases/PHASE-7.md rules out. Only the
        worker thread puts, so comparing against the capacity here cannot race
        with another writer, and a reader draining in between can only make
        room that this drop did not need.
        """
        if (
            self._time_limit is not None
            and self._clock() - job.started_at > self._time_limit
        ):
            job.message = (
                f"stopped after {self._time_limit:.0f} s, the longest one solve "
                "may run on this server. A coarser mesh or fewer bias points "
                "will finish sooner."
            )
            job.cancelling.set()
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

    def result(self, job_id: str) -> Any:
        """What the work returned, or None if it has not finished cleanly.

        None is also what work returning nothing gives back. The status is
        what distinguishes the two, and a caller asks after `wait`.
        """
        return self._job(job_id).result

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

    def close(self, timeout: float | None = None) -> None:
        """Cancel every job still running and wait until each has stopped.

        Called when the app shuts down. A solver thread that outlives the
        interpreter dies inside numpy, and Python then exits 120 over a run
        that was otherwise clean.

        Args:
            timeout: seconds to wait for all of them together [s], or None.

        Raises TimeoutError when a job has not stopped in time. Cancellation
        lands at a job's next frame, so work that never reports cannot be
        stopped, and a shutdown that hung on it would be worse than saying so.
        """
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.status not in _TERMINAL:
                job.cancelling.set()

        deadline = None if timeout is None else time.monotonic() + timeout
        for job in jobs:
            left = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not job.finished.wait(timeout=left):
                raise TimeoutError(f"job {job.id} did not stop within {timeout} s")

    def wait(self, job_id: str, timeout: float | None = None) -> JobStatus:
        """Block until the job reaches a terminal status, and return it."""
        job = self._job(job_id)
        if not job.finished.wait(timeout=timeout):
            raise TimeoutError(f"job {job_id} did not finish within {timeout} s")
        return job.status
