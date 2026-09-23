"""The command line, ddsim/cli.py.

phases/PHASE-7.md asks that someone who cloned the repo runs one command and
has a working page. This is that command. What is tested here is the argument
handling and the binding, not uvicorn: the server is injected so that a test
never opens a socket, which keeps this file fast and keeps a failure here
unambiguous.
"""

from __future__ import annotations

import pytest

from ddsim.cli import main


def captured():
    """A stand in for uvicorn.run that records what it was asked to serve."""
    calls: list[dict] = []

    def run(app, **options):
        calls.append({"app": app, **options})

    return calls, run


def test_serve_binds_to_loopback_by_default() -> None:
    """phases/PHASE-7.md: single user, local, no authentication. A default of
    0.0.0.0 would put an unauthenticated solver on the network."""
    calls, run = captured()

    assert main(["serve"], run=run) == 0
    assert calls[0]["host"] == "127.0.0.1"


def test_serve_takes_the_host_and_port_it_is_given() -> None:
    calls, run = captured()

    main(["serve", "--host", "127.0.0.2", "--port", "9123"], run=run)

    assert calls[0]["host"] == "127.0.0.2"
    assert calls[0]["port"] == 9123


def test_serving_off_loopback_says_what_it_is_doing(capsys) -> None:
    """Not refused, because there are reasons to do it deliberately, but
    never silent. There is no authentication in front of this."""
    calls, run = captured()

    main(["serve", "--host", "0.0.0.0"], run=run)

    assert calls[0]["host"] == "0.0.0.0"
    assert "no authentication" in capsys.readouterr().err


def test_serve_hands_over_an_application() -> None:
    """Built here rather than named as an import string, so that the command
    works from a clone with nothing else configured."""
    calls, run = captured()

    main(["serve"], run=run)

    assert calls[0]["app"] is not None


def test_a_command_that_does_not_exist_is_refused() -> None:
    with pytest.raises(SystemExit) as raised:
        main(["simulate"])

    assert raised.value.code == 2


def test_no_command_at_all_is_refused() -> None:
    """A bare `ddsim` should say what it can do rather than do nothing."""
    with pytest.raises(SystemExit) as raised:
        main([])

    assert raised.value.code == 2


def test_the_real_server_can_upgrade_to_a_websocket() -> None:
    """The residual reaches the page over a websocket, and bare uvicorn has no
    websocket library of its own. Without one it answers every upgrade with a
    404 and the page sits at "solving" forever. TestClient never goes through
    uvicorn, so every route test passed while the first real browser failed.
    This asks uvicorn directly which protocol it would serve."""
    import uvicorn

    from ddsim.api.app import create_app

    config = uvicorn.Config(create_app(), ws="auto")
    config.load()

    assert config.ws_protocol_class is not None


def test_serving_off_loopback_limits_the_jobs() -> None:
    """Off loopback, anyone can submit, so the registry caps how many solves
    run at once. On loopback it is one person's instrument and stays open."""
    import threading

    from ddsim.api.jobs import BusyError

    release = threading.Event()
    calls, run = captured()
    main(["serve", "--host", "0.0.0.0"], run=run)
    jobs = calls[0]["app"].state.jobs

    held = [jobs.submit(lambda send: release.wait(timeout=5.0)) for _ in range(2)]
    with pytest.raises(BusyError):
        jobs.submit(lambda send: None)
    release.set()
    for job in held:
        jobs.wait(job.id, timeout=5.0)


def test_serving_on_loopback_leaves_the_jobs_unlimited() -> None:
    import threading

    release = threading.Event()
    calls, run = captured()
    main(["serve"], run=run)
    jobs = calls[0]["app"].state.jobs

    held = [jobs.submit(lambda send: release.wait(timeout=5.0)) for _ in range(5)]
    release.set()
    for job in held:
        jobs.wait(job.id, timeout=5.0)
