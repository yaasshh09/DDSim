"""One headless browser against the real server, phases/PHASE-7.md.

Every route has a contract test through TestClient, and the first time a real
browser opened the page it found five bugs none of them could reach: uvicorn
with no websocket library, a float32 payload the browser refused to read, and
three more that only a drawn page shows. TestClient skips uvicorn and the
Python field reader is not the JavaScript one. This test skips neither.

It is a smoke test and nothing more. It proves the pieces meet: a solve is
submitted from the page, telemetry reaches the page while the solve is still
running, the solve finishes, and a profile is read and drawn with no error on
the page. Whether the numbers are right is every other test's job.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator

import pytest
import uvicorn
from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PageTimeout

from ddsim.api.app import create_app

STARTUP_TIMEOUT = 10.0
"""Seconds to wait for uvicorn to start listening [s]."""

SOLVE_TIMEOUT_MS = 120_000
"""How long a diode sweep may take on a slow CI runner [ms]."""


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture
def server() -> Iterator[str]:
    """The real uvicorn, on its own thread, with the app `ddsim serve` builds."""
    port = free_port()
    config = uvicorn.Config(
        create_app(), host="127.0.0.1", port=port, log_level="warning"
    )
    running = uvicorn.Server(config)
    thread = threading.Thread(target=running.run, daemon=True)
    thread.start()

    waited = threading.Event()
    for _ in range(int(STARTUP_TIMEOUT / 0.05)):
        if running.started:
            break
        waited.wait(0.05)
    assert running.started, "uvicorn did not start"

    yield f"http://127.0.0.1:{port}"

    running.should_exit = True
    thread.join(timeout=STARTUP_TIMEOUT)


def wait_until(page: Page, condition: str, errors: list[str]) -> None:
    """Wait for a condition on the page, and on a timeout say what the page
    threw. A bare timeout on a missing profile reads like a slow solve, when
    the real cause is a RangeError the page raised in the first second."""
    try:
        page.wait_for_function(condition, timeout=SOLVE_TIMEOUT_MS)
    except PageTimeout:
        pytest.fail(f"never true: {condition}; page errors: {errors}")


def test_a_diode_solve_streams_finishes_and_draws_in_a_real_browser(server) -> None:
    """The acceptance criterion: a telemetry frame arrives before completion.
    The rest is what the first real browser session found broken."""
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        try:
            page = browser.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))

            page.goto(server)
            page.wait_for_function("el('state').textContent === 'ready'")
            page.fill("#voltages", "0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6")
            page.click("#solve")

            wait_until(
                page,
                "state.residual.length > 0 && el('state').textContent === 'solving'",
                errors,
            )
            wait_until(
                page,
                "el('state').textContent === 'done' && state.fields !== null",
                errors,
            )

            assert page.evaluate("state.points.length") == 7
            assert page.evaluate("state.fields.arrays.psi.length") > 0
            assert errors == []
        finally:
            browser.close()
