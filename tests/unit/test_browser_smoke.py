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

import json
import socket
import threading
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
import uvicorn
from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PageTimeout

from ddsim.api.app import create_app
from ddsim.api.jobs import JobRegistry, JobStatus

STARTUP_TIMEOUT = 10.0
"""Seconds to wait for uvicorn to start listening [s]."""

SOLVE_TIMEOUT_MS = 120_000
"""How long a diode sweep may take on a slow CI runner [ms]."""


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@dataclass(frozen=True)
class Served:
    """A running server and the registry its jobs land in.

    The registry is here because the live slider criterion is about jobs and
    not about pixels: what has to be true is that five drags leave one solve
    running rather than five, and only the server can answer that.
    """

    url: str
    jobs: JobRegistry


@pytest.fixture
def served() -> Iterator[Served]:
    """The real uvicorn, on its own thread, with the app `ddsim serve` builds."""
    port = free_port()
    registry = JobRegistry()
    config = uvicorn.Config(
        create_app(registry), host="127.0.0.1", port=port, log_level="warning"
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

    yield Served(url=f"http://127.0.0.1:{port}", jobs=registry)

    running.should_exit = True
    thread.join(timeout=STARTUP_TIMEOUT)


@pytest.fixture
def server(served: Served) -> str:
    """Just the address, which is all most of these tests want."""
    return served.url


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
            hosts: set[str] = set()
            page.on("request", lambda request: hosts.add(request.url.split("/")[2]))

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
                "el('state').textContent.startsWith('done') && state.fields !== null",
                errors,
            )

            assert page.evaluate("state.points.length") == 7
            assert page.evaluate("state.fields.arrays.psi.length") > 0
            assert errors == []
            assert hosts == {server.split("/")[2]}, f"requests left: {hosts}"
            assert page.evaluate("typeof marked.parse") == "function"
            assert page.evaluate("typeof renderMathInElement") == "function"
        finally:
            browser.close()


def test_a_knob_explains_itself_with_rendered_maths(server) -> None:
    """Part two: every knob one click from its explanation, and the depth
    layer's equations rendered rather than shown as LaTeX source."""
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        try:
            page = browser.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))

            page.goto(server)
            wait_until(page, "el('state').textContent === 'ready'", errors)
            page.click('label:has([data-name="Na"]) .explain')

            wait_until(page, "el('drawer').classList.contains('open')", errors)
            wait_until(
                page, "document.querySelector('#drawer-depth .katex') !== null", errors
            )

            assert "cm^-3" in page.inner_text("#drawer-knob")
            assert page.inner_text("#drawer-title")
            assert "$$" not in page.inner_text("#drawer-depth")
            # marked reads \, as a markdown escape and drops the backslash,
            # so KaTeX would draw a comma. Its annotation keeps what it got.
            tex = page.evaluate(
                "[...document.querySelectorAll('#drawer-depth annotation')]"
                ".map((a) => a.textContent).join(' ')"
            )
            assert r"\," in tex, tex
            assert errors == []
        finally:
            browser.close()


def test_the_band_view_draws_after_a_diode_solve(server) -> None:
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        try:
            page = browser.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(server)
            wait_until(page, "el('state').textContent === 'ready'", errors)
            page.fill("#voltages", "0, 0.3")
            page.click("#solve")
            wait_until(
                page,
                "el('state').textContent.startsWith('done') && state.fields !== null",
                errors,
            )

            page.check("#bands")

            assert page.evaluate("state.fields.arrays.Ec.length") > 0
            assert page.is_visible("#legend-bands")
            assert errors == []
        finally:
            browser.close()


COARSE_FET = {
    "n_contact": "4",
    "n_sd": "10",
    "n_channel": "12",
    "n_silicon": "29",
    "n_oxide": "4",
    "h_min_x": "5e-7",
    "h_min_y": "1e-7",
    "drain_voltage": "0.05",
}


def test_streamlines_trace_through_a_mosfet(server) -> None:
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        try:
            page = browser.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(server)
            wait_until(page, "el('state').textContent === 'ready'", errors)
            page.select_option("#device-kind", "nmos")
            page.select_option("#sweep-kind", "transfer")
            for name, value in COARSE_FET.items():
                page.fill(f'[data-name="{name}"]', value)
            page.fill("#measure-at", "drain")
            page.fill("#voltages", "1.0")
            page.click("#solve")
            wait_until(
                page,
                "el('state').textContent.startsWith('done') && state.fields !== null",
                errors,
            )

            count = page.evaluate(
                "traceStreamlines(state.fields, 12, 6).filter(l => l.length > 5).length"
            )

            assert count > 0

            # Before anyone drags a cutline, that canvas has never been sized
            # by fit(), so a width bound stretches its default 300 by 200 shape
            # and leaves the panel hundreds of pixels tall with nothing in it.
            cutline_height = page.evaluate(
                "el('cutline').getBoundingClientRect().height"
            )

            assert cutline_height == pytest.approx(200, abs=1)
            assert errors == []
        finally:
            browser.close()


def test_an_equation_wider_than_the_drawer_can_be_reached(server) -> None:
    """A display equation that does not fit the drawer has to scroll. The
    Scharfetter-Gummel topic has one 586 px wide in a 427 px drawer, and
    without this it was simply cut off at the edge with no way to see it."""
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        try:
            page = browser.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(server)
            wait_until(page, "el('state').textContent === 'ready'", errors)
            page.evaluate("explain('scharfetter-gummel')")
            wait_until(
                page, "el('drawer-depth').querySelector('.katex') !== null", errors
            )

            clipped = page.evaluate(
                """[...el('drawer-depth').querySelectorAll('.katex-display')]
                   .filter(block => block.scrollWidth > block.clientWidth + 1
                       && getComputedStyle(block).overflowX === 'visible').length"""
            )

            assert clipped == 0
            assert errors == []
        finally:
            browser.close()


# ------------------------------------------------------ stage 2, the sandbox


def solved(page: Page, errors: list[str]) -> None:
    """Press solve and wait for the curve to arrive."""
    page.click("#solve")
    wait_until(page, "el('state').textContent.startsWith('done')", errors)


def until(page: Page, ready, what: str, seconds: float = 30.0) -> None:
    """Wait for something on the server side to become true.

    The page's own status text says what the browser last heard, which is not
    the same question as what the job is doing, and a slider test that read
    the text would pass by racing past a solve that had already ended.

    The wait goes through the page because Playwright's sync API only runs
    its event handlers while a call into it is in progress; a plain sleep
    would leave every response the page received unheard.
    """
    for _ in range(int(seconds / 0.02)):
        if ready():
            return
        page.wait_for_timeout(20)
    pytest.fail(f"waited {seconds} s for {what}")


def test_a_finished_run_stays_on_the_plot_as_the_numbers_it_was_drawn_with(
    server,
) -> None:
    """The acceptance criterion: an overlay is the earlier run's numbers,
    never recomputed. So the test reads the first curve off the page, solves a
    different device, and demands the kept copy be the same array to the last
    digit rather than something solved again on the new knobs."""
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        try:
            page = browser.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(server)
            wait_until(page, "el('state').textContent === 'ready'", errors)

            page.fill("#voltages", "0, 0.2, 0.4")
            solved(page, errors)
            first = page.evaluate("state.points.map((p) => [p.voltage, p.value])")

            page.fill('[data-name="Na"]', "2e17")
            solved(page, errors)
            second = page.evaluate("state.points.map((p) => [p.voltage, p.value])")

            kept = page.evaluate(
                "state.runs.map((r) => r.points.map((p) => [p.voltage, p.value]))"
            )
            labels = page.evaluate("state.runs.map((r) => r.label)")

            assert kept == [first], "the overlay is not the first run's own numbers"
            assert second != first, "the second solve did not move the curve"
            # Labelled by what differs from the run before it.
            assert "Na" in labels[0], labels

            page.click("#clear-runs")
            assert page.evaluate("state.runs.length") == 0
            assert errors == []
        finally:
            browser.close()


def test_five_slider_moves_in_a_second_leave_one_solve_running(served) -> None:
    """The acceptance criterion, in two halves. Moving faster than the page
    submits collapses to a single job, and a move that lands while a solve is
    running cancels it rather than queueing behind it. Either way the process
    is idle once the last one finishes."""
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        try:
            page = browser.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            submitted: list[str] = []

            def note_job(response) -> None:
                if response.request.method == "POST" and response.url.endswith(
                    "/api/jobs"
                ):
                    submitted.append(response.json()["id"])

            page.on("response", note_job)
            page.goto(served.url)
            wait_until(page, "el('state').textContent === 'ready'", errors)

            # Five moves inside a second, faster than the page submits.
            page.evaluate(
                """(async () => {
                  const slider = document.querySelector('[data-slider="Na"]');
                  for (const at of [0.2, 0.3, 0.4, 0.5, 0.6]) {
                    slider.value = String(
                      Number(slider.min) + at * (slider.max - slider.min)
                    );
                    slider.dispatchEvent(new Event("input", { bubbles: true }));
                    await new Promise((done) => setTimeout(done, 50));
                  }
                })()"""
            )
            wait_until(page, "el('state').textContent.startsWith('done')", errors)

            assert len(submitted) == 1, f"five moves submitted {len(submitted)} jobs"
            assert served.jobs.status(submitted[0]) is JobStatus.DONE

            # And a move that lands while a solve is running replaces it. The
            # long voltage list is what gives the move something to interrupt,
            # and the waits are on the job rather than on the page, so this
            # cannot pass by racing past a solve that already finished.
            page.fill("#voltages", ", ".join(str(v / 50) for v in range(40)))
            move = """(() => {
                  const slider = document.querySelector('[data-slider="Na"]');
                  slider.value = String(
                    Number(slider.min) + AT * (slider.max - slider.min)
                  );
                  slider.dispatchEvent(new Event("input", { bubbles: true }));
                })()"""

            page.evaluate(move.replace("AT", "0.7"))
            until(page, lambda: len(submitted) == 2, "a second job to be submitted")
            until(
                page,
                lambda: served.jobs.status(submitted[1]) is JobStatus.RUNNING,
                "the second job to start solving",
            )

            page.evaluate(move.replace("AT", "0.8"))
            until(page, lambda: len(submitted) == 3, "a third job to be submitted")
            wait_until(page, "el('state').textContent.startsWith('done')", errors)

            assert served.jobs.status(submitted[1]) is JobStatus.CANCELLED
            assert served.jobs.status(submitted[2]) is JobStatus.DONE
            assert errors == []
        finally:
            # Before the close, or a response still in flight reaches a
            # handler whose page is already gone.
            page.remove_listener("response", note_job)
            browser.close()


def test_a_two_dimensional_device_offers_a_coarse_mesh_and_no_sliders(server) -> None:
    """phases/PHASE-7.md: live sliders on the 1D devices only, and the page
    says why. The 2D ones get the coarse mesh instead, with the note on what
    choosing it costs."""
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        try:
            page = browser.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(server)
            wait_until(page, "el('state').textContent === 'ready'", errors)

            sliders = "document.querySelectorAll('[data-slider]').length"
            assert page.evaluate(sliders) > 0
            # The diode has no coarse mesh, so no buttons offering one.
            assert page.is_hidden("#mesh-coarse")

            page.select_option("#device-kind", "nmos")
            assert page.evaluate(sliders) == 0
            assert page.is_visible("#mesh-coarse")

            page.click("#mesh-coarse")
            assert page.input_value('[data-name="n_silicon"]') == "29"
            assert "percent" in page.inner_text("#mesh-note")

            page.click("#mesh-converged")
            assert page.input_value('[data-name="n_silicon"]') == "101"
            assert errors == []
        finally:
            browser.close()


def test_a_lesson_sets_up_its_steps_and_leaves_the_device_behind(server) -> None:
    """phases/PHASE-7.md Stage 3: a lesson sets up its device, tells the
    student what to change, and can be left at any point with the device kept
    as a sandbox. The step is solved here too, since a request the page builds
    from a lesson and the server then refuses is exactly the kind of break
    that only a real page finds."""
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        try:
            page = browser.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(server)
            wait_until(page, "el('state').textContent === 'ready'", errors)
            wait_until(page, "el('lesson').options.length === 6", errors)

            page.select_option("#lesson", "02-bias")
            wait_until(page, "!el('lesson-panel').hidden", errors)
            assert page.input_value("#device-kind") == "pn_diode"
            assert page.input_value("#voltages").startswith("0, 0.05, 0.1")
            # The explanation's maths is rendered, not left as TeX.
            rendered = "document.querySelector('#lesson-saw .katex') !== null"
            assert page.evaluate(rendered)

            page.click("#lesson-steps li:has-text('Reverse bias') button")
            assert float(page.input_value('[data-name="length"]')) == 4e-4
            assert page.input_value("#voltages") == "0, -0.5, -1, -2"
            # The slider was rebuilt at the lesson's value, to within the one
            # step it snaps to, rather than at the default of -4 decades.
            slider = float(page.input_value('[data-slider="length"]'))
            assert abs(slider - (-3.3979)) <= 0.01

            solved(page, errors)
            assert page.evaluate("state.points.length") == 4

            page.click("#lesson-leave")
            assert page.is_hidden("#lesson-panel")
            assert float(page.input_value('[data-name="length"]')) == 4e-4

            page.select_option("#lesson", "04-mosfet")
            wait_until(page, "el('device-kind').value === 'nmos'", errors)
            assert page.input_value('[data-name="n_silicon"]') == "29"
            assert "coarse mesh" in page.inner_text("#mesh-note")
            assert errors == []
        finally:
            browser.close()


def set_region(page: Page, row: int, dopant: str, length: str, doping: str) -> None:
    """Fill in one row of the region editor, counted from 1."""
    rows = f"#regions > div:nth-child({row})"
    page.select_option(f"{rows} [data-region=dopant]", dopant)
    page.fill(f"{rows} [data-region=length]", length)
    page.fill(f"{rows} [data-region=concentration]", doping)


def test_a_student_builds_a_stack_solves_it_saves_it_and_loads_it(
    server, tmp_path
) -> None:
    """phases/PHASE-7.md Stage 4 from the page: regions stacked left to
    right, a refusal that says why, and a device that goes to a file and
    comes back from one, which is how students hand each other a device."""
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        try:
            page = browser.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(server)
            wait_until(page, "el('state').textContent === 'ready'", errors)

            assert page.is_hidden("#stack")
            page.select_option("#device-kind", "stack")
            assert page.is_visible("#stack")
            assert page.locator("#regions > div").count() == 2
            assert page.input_value("#contact") == "left"
            assert "not validated" in page.inner_text("#stack-note")

            page.click("#region-add")
            assert page.locator("#regions > div").count() == 3
            set_region(page, 1, "p", "2e-5", "1e18")
            set_region(page, 2, "n", "1e-4", "1e14")
            set_region(page, 3, "n", "2e-5", "1e18")
            page.fill("#voltages", "0, 0.2")
            solved(page, errors)
            assert page.evaluate("state.points.length") == 2
            # The result is labelled as the validated solver on a structure
            # nobody validated, and a benchmark device's is not.
            assert page.is_visible("#built-note")

            # A second run with the base lightly doped the other way. Its
            # overlay label names the regions rather than printing objects.
            set_region(page, 2, "p", "1e-4", "1e14")
            solved(page, errors)
            label = page.inner_text("#runs-note")
            assert "regions" in label and "object" not in label

            set_region(page, 2, "n", "1e-4", "1e21")
            page.click("#solve")
            wait_until(page, "el('state').textContent === 'refused'", errors)
            refusal = page.inner_text("#message")
            assert "region 2" in refusal and "docs/01-physics.md" in refusal
            set_region(page, 2, "n", "1e-4", "1e14")

            with page.expect_download() as saving:
                page.click("#device-save")
            saved = json.loads(saving.value.path().read_text(encoding="utf-8"))
            assert saved["kind"] == "stack"
            assert saved["parameters"]["regions"][1] == {
                "dopant": "n",
                "length": 1e-4,
                "concentration": 1e14,
            }

            page.select_option("#device-kind", "pn_diode")
            assert page.is_hidden("#stack")
            handed_over = tmp_path / "pin.json"
            handed_over.write_text(json.dumps(saved), encoding="utf-8")
            page.set_input_files("#device-load", str(handed_over))
            wait_until(page, "el('device-kind').value === 'stack'", errors)
            assert page.locator("#regions > div").count() == 3
            base = "#regions > div:nth-child(2) [data-region=concentration]"
            assert page.input_value(base) == "1e+14"

            page.click("#regions > div:nth-child(3) [data-region=remove]")
            assert page.locator("#regions > div").count() == 2

            broken = tmp_path / "broken.json"
            broken.write_text("{not json", encoding="utf-8")
            page.set_input_files("#device-load", str(broken))
            wait_until(page, "el('message').textContent.includes('JSON')", errors)
            assert errors == []
        finally:
            browser.close()
