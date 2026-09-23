"""Tests for api/jobs.py, the job runner underneath the HTTP layer.

phases/PHASE-7.md: a solve is a job, not a request. A MOSFET sweep is minutes
of Newton solves, so it is submitted, streamed while it runs, and cancellable.
This module is that machinery with no HTTP and no solver in it, which is why
the work here is fake: threading and cancellation are what is under test, and
a real device would only make the test slow and the failure ambiguous.
"""

from __future__ import annotations

import threading

import pytest

from ddsim.api.jobs import CancelledError, JobRegistry, JobStatus


def test_a_job_runs_and_finishes() -> None:
    jobs = JobRegistry()

    job = jobs.submit(lambda send: send("only frame"))
    jobs.wait(job.id, timeout=5.0)

    assert jobs.status(job.id) is JobStatus.DONE


def test_the_frames_arrive_in_the_order_they_were_sent() -> None:
    jobs = JobRegistry()

    job = jobs.submit(lambda send: [send(n) for n in range(5)])

    assert list(jobs.frames(job.id, timeout=5.0)) == [0, 1, 2, 3, 4]


def test_the_stream_ends_when_the_work_does() -> None:
    """A reader draining frames has to be able to stop. Without an end the
    websocket holds a socket open on a job that finished minutes ago."""
    jobs = JobRegistry()

    job = jobs.submit(lambda send: send("done"))
    drained = list(jobs.frames(job.id, timeout=5.0))

    assert drained == ["done"]
    assert jobs.status(job.id) is JobStatus.DONE


def test_submitting_does_not_wait_for_the_work() -> None:
    """The whole reason a solve is a job. If submit blocked, the browser would
    get its job id when the sweep was already over."""
    release = threading.Event()
    jobs = JobRegistry()

    job = jobs.submit(lambda send: release.wait(timeout=5.0))

    assert jobs.status(job.id) in (JobStatus.PENDING, JobStatus.RUNNING)
    release.set()
    jobs.wait(job.id, timeout=5.0)


def test_work_that_raises_leaves_the_job_failed_with_the_reason() -> None:
    """phases/PHASE-7.md: a failed solve never reports a success it did not
    get, and it says what went wrong rather than spinning."""

    def explode(send: object) -> None:
        raise RuntimeError("the Jacobian is singular")

    jobs = JobRegistry()
    job = jobs.submit(explode)
    jobs.wait(job.id, timeout=5.0)

    assert jobs.status(job.id) is JobStatus.FAILED
    assert "singular" in jobs.message(job.id)


def test_cancelling_stops_the_work_at_its_next_frame() -> None:
    """Cancellation rides on the telemetry callback, which is the only place
    a solve looks up from the arithmetic. Sending a frame is what notices."""
    started = threading.Event()
    frames_sent = []

    def forever(send) -> None:
        started.set()
        for n in range(1_000_000):
            send(n)
            frames_sent.append(n)

    jobs = JobRegistry()
    job = jobs.submit(forever)
    started.wait(timeout=5.0)

    assert jobs.cancel(job.id) is True
    jobs.wait(job.id, timeout=5.0)
    assert jobs.status(job.id) is JobStatus.CANCELLED
    assert len(frames_sent) < 1_000_000


def test_a_cancelled_job_ends_its_stream() -> None:
    """Otherwise a cancelled sweep leaves the browser waiting forever."""
    started = threading.Event()

    def forever(send) -> None:
        started.set()
        while True:
            send("tick")

    jobs = JobRegistry()
    job = jobs.submit(forever)
    started.wait(timeout=5.0)
    jobs.cancel(job.id)

    drained = list(jobs.frames(job.id, timeout=5.0))

    assert drained[-1] == "tick"
    assert jobs.status(job.id) is JobStatus.CANCELLED


def test_cancelling_a_finished_job_changes_nothing() -> None:
    """The browser cannot know the sweep ended a moment before the click."""
    jobs = JobRegistry()

    job = jobs.submit(lambda send: send("one"))
    jobs.wait(job.id, timeout=5.0)

    assert jobs.cancel(job.id) is False
    assert jobs.status(job.id) is JobStatus.DONE


def test_the_work_can_see_that_it_was_cancelled() -> None:
    """A solve holding a partial result should be able to put it down tidily
    rather than be stopped mid write. CancelledError is a real exception it can
    catch, not a thread kill."""
    caught: list[str] = []

    def tidy(send) -> None:
        try:
            while True:
                send("tick")
        except CancelledError:
            caught.append("cleaned up")
            raise

    started = threading.Event()
    jobs = JobRegistry()
    job = jobs.submit(lambda send: (started.set(), tidy(send)))
    started.wait(timeout=5.0)
    jobs.cancel(job.id)
    jobs.wait(job.id, timeout=5.0)

    assert caught == ["cleaned up"]


def test_two_jobs_do_not_share_a_stream() -> None:
    jobs = JobRegistry()

    first = jobs.submit(lambda send: send("first"))
    second = jobs.submit(lambda send: send("second"))

    assert list(jobs.frames(first.id, timeout=5.0)) == ["first"]
    assert list(jobs.frames(second.id, timeout=5.0)) == ["second"]


def test_an_unknown_job_is_refused() -> None:
    jobs = JobRegistry()

    with pytest.raises(KeyError):
        jobs.status("no-such-job")


def test_every_job_gets_its_own_id() -> None:
    jobs = JobRegistry()

    ids = {jobs.submit(lambda send: None).id for _ in range(10)}

    assert len(ids) == 10


def test_a_full_queue_drops_its_oldest_frame_and_counts_the_loss() -> None:
    """phases/PHASE-7.md: the server drops frames rather than making the
    solver wait. Counting them is the difference between a plot that is
    coarse and a plot that is wrong without saying so."""
    jobs = JobRegistry(queue_size=4)

    job = jobs.submit(lambda send: [send(n) for n in range(10)])
    jobs.wait(job.id, timeout=5.0)

    assert jobs.dropped(job.id) == 6
    assert list(jobs.frames(job.id, timeout=5.0)) == [6, 7, 8, 9]


def test_a_job_finishes_even_when_nobody_drains_its_queue() -> None:
    """The queue fills exactly when no reader is attached, which is also when
    a blocking put would hang the worker forever. A job that never reaches a
    terminal state is a browser waiting on a result that will never come, and
    a process that cannot be shut down tidily."""
    jobs = JobRegistry(queue_size=2)

    job = jobs.submit(lambda send: [send(n) for n in range(50)])

    assert jobs.wait(job.id, timeout=5.0) is JobStatus.DONE


def test_reading_frames_gives_up_rather_than_waiting_forever() -> None:
    """A websocket handler needs to be able to bound its wait. Without this
    a job whose worker is wedged holds a connection open indefinitely."""
    jobs = JobRegistry()
    job = jobs.submit(lambda send: threading.Event().wait(timeout=5.0))

    with pytest.raises(TimeoutError):
        list(jobs.frames(job.id, timeout=0.05))


def test_waiting_for_a_job_gives_up_rather_than_hanging() -> None:
    """Same argument on the other call, and it is the one the tests here
    lean on, so a silent hang would look like a slow suite."""
    jobs = JobRegistry()
    job = jobs.submit(lambda send: threading.Event().wait(timeout=5.0))

    with pytest.raises(TimeoutError):
        jobs.wait(job.id, timeout=0.05)


def test_a_finished_job_keeps_what_the_work_returned() -> None:
    """A sweep's curve is the one thing a reader cannot afford to lose, and
    the frame queue is allowed to drop its oldest frame. So the return value
    is held on the job rather than sent, and a browser that connects after
    the last point still gets the curve."""
    jobs = JobRegistry()

    job = jobs.submit(lambda send: "the curve")
    jobs.wait(job.id, timeout=5.0)

    assert jobs.result(job.id) == "the curve"


def test_a_job_that_is_still_running_has_no_result_yet() -> None:
    """None rather than a wait. The caller asks once the status is terminal,
    and a None that meant either not yet or nothing returned would be the
    same ambiguity a spinner is."""
    jobs = JobRegistry()
    holding = threading.Event()

    job = jobs.submit(lambda send: holding.wait(timeout=5.0))
    try:
        assert jobs.result(job.id) is None
    finally:
        holding.set()
    jobs.wait(job.id, timeout=5.0)


def test_a_failed_job_has_no_result() -> None:
    """Whatever the work built before it raised is not a result, and handing
    back a half filled curve as though it were one is the worst outcome
    available here."""
    jobs = JobRegistry()

    job = jobs.submit(lambda send: 1 / 0)
    jobs.wait(job.id, timeout=5.0)

    assert jobs.status(job.id) is JobStatus.FAILED
    assert jobs.result(job.id) is None


def test_a_cancelled_job_keeps_nothing_either() -> None:
    """A cancelled sweep stopped somewhere nobody chose. Its partial curve is
    in the frames the reader already has.

    The work waits so that the cancel lands while it is still inside. Without
    that, this work finishes before the cancel arrives and the job is DONE
    with a result, which is the right answer to a different question.
    """
    jobs = JobRegistry()
    cancelled = threading.Event()

    def work(send):
        send("one")
        cancelled.wait(timeout=5.0)
        send("two")
        return "finished"

    job = jobs.submit(work)
    next(jobs.frames(job.id, timeout=5.0))
    assert jobs.cancel(job.id)
    cancelled.set()

    assert jobs.wait(job.id, timeout=5.0) is JobStatus.CANCELLED
    assert jobs.result(job.id) is None


def test_closing_the_registry_stops_every_running_job() -> None:
    """A solver thread still running when the interpreter exits dies inside
    numpy, and Python reports exit code 120 over a test run that passed. So
    close cancels what is running and does not return until it has stopped."""
    jobs = JobRegistry()

    def forever(send) -> None:
        while True:
            send("iteration")

    running = [jobs.submit(forever) for _ in range(3)]
    finished = jobs.submit(lambda send: send("done"))
    jobs.wait(finished.id, timeout=5.0)

    jobs.close(timeout=5.0)

    assert all(jobs.status(job.id) is JobStatus.CANCELLED for job in running)
    assert jobs.status(finished.id) is JobStatus.DONE


def test_closing_gives_up_on_work_that_never_reports() -> None:
    """Cancellation lands at the next frame, so work that never sends one
    cannot be stopped. close says so rather than hanging the shutdown."""
    release = threading.Event()
    jobs = JobRegistry()
    jobs.submit(lambda send: release.wait(timeout=5.0))

    with pytest.raises(TimeoutError):
        jobs.close(timeout=0.1)
    release.set()
