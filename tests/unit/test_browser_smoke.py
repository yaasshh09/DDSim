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
import logging
import math
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
    # A page that leaves mid solve is a normal thing for a page to do, and the
    # server has to take it quietly. uvicorn logs an escaped exception here and
    # the page never sees it, so this is the only place it can be caught.
    server_errors: list[logging.LogRecord] = []
    catcher = logging.Handler(level=logging.ERROR)
    catcher.emit = server_errors.append  # type: ignore[method-assign]
    logging.getLogger("uvicorn.error").addHandler(catcher)
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
    logging.getLogger("uvicorn.error").removeHandler(catcher)
    assert [record.getMessage() for record in server_errors] == []


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
            page.select_option("#measure-at", "drain")
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


def test_a_knob_cannot_be_pushed_into_a_device_that_does_not_build(server) -> None:
    """Every slider stays inside its own range, but two knobs together can
    still ask for a device that does not exist: a diode shorter than where its
    junction sits, or more mesh cells than fit at the spacing asked for. The
    user hit both as red refusals. The knob now stops at the last value that
    builds, says why in a quiet note, and the solve goes ahead."""
    drag = """([name, at]) => {
      const slider = document.querySelector('[data-slider="' + name + '"]');
      slider.value = String(Number(slider.min) + at * (slider.max - slider.min));
      slider.dispatchEvent(new Event("input", { bubbles: true }));
    }"""
    box = """(name) => Number(document.querySelector(
      '#device-knobs [data-name="' + name + '"]').value)"""
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        try:
            page = browser.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(server)
            wait_until(page, "el('state').textContent === 'ready'", errors)

            def solved_again(move) -> None:
                """Make a move, then wait for the solve it starts, and not
                for the done a previous solve left on the page."""
                before = page.evaluate("state.job")
                move()
                wait_until(
                    page,
                    f"state.job !== {json.dumps(before)} && "
                    "el('state').textContent.startsWith('done')",
                    errors,
                )

            # The length slider's bottom end is 0.1 um, under the 0.5 um
            # junction. It stops just past the junction instead.
            solved_again(lambda: page.evaluate(drag, ["length", 0.0]))
            assert page.evaluate(box, "length") > page.evaluate(box, "junction")
            assert "stops at" in page.inner_text("#knob-note")

            # Every node the slider has, at the coarsest spacing it has: 1000
            # cells of 10 nm is 10 um in a 1 um device. h_min stops short.
            solved_again(lambda: page.evaluate(drag, ["length", 0.5]))
            solved_again(lambda: page.evaluate(drag, ["n_nodes", 1.0]))
            solved_again(lambda: page.evaluate(drag, ["h_min", 1.0]))
            assert page.evaluate(box, "h_min") < 1e-6
            assert "h_min" in page.inner_text("#knob-note")

            # A number typed past a slider's end is held to that end.
            page.fill('#device-knobs [data-name="Na"]', "1e25")
            solved_again(
                lambda: page.dispatch_event('#device-knobs [data-name="Na"]', "change")
            )
            assert page.evaluate(box, "Na") == 1e19

            # The terminal is picked from the device's own, never typed.
            options = "Array.from(el('contact').options, (o) => o.value)"
            assert page.evaluate(options) == ["anode", "cathode"]

            # And a voltage past the anode's declared range is held to it.
            page.fill("#voltages", "0, 0.5, 3")
            solved_again(lambda: page.click("#solve"))
            assert page.input_value("#voltages") == "0, 0.5, 1"
            assert "held to" in page.inner_text("#voltage-note")
            assert errors == []
        finally:
            browser.close()


def test_a_two_dimensional_device_offers_a_coarse_mesh(server) -> None:
    """phases/PHASE-7.md: live sliders on the 1D devices only, and the page
    says why. The 2D ones get the coarse mesh instead, with the note on what
    choosing it costs. Their sliders set a knob and solve nothing, see the
    negative log slider test."""
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

            assert page.is_visible("#bands")
            page.select_option("#device-kind", "nmos")
            assert page.evaluate(sliders) > 0
            assert page.is_visible("#mesh-coarse")
            # The band view draws a 1D profile. A 2D device gets bands from
            # the cutline, so the checkbox would do nothing there.
            assert page.is_hidden("#bands")
            # An iv sweep takes ohmic contacts only, and a 2D device starts on
            # its gate, so leaving the diode's iv would make the first solve a
            # refusal. Going back to a 1D device goes back to iv.
            assert page.input_value("#sweep-kind") == "transfer"
            page.select_option("#device-kind", "pn_diode")
            assert page.input_value("#sweep-kind") == "iv"
            assert page.is_visible("#bands")
            page.select_option("#device-kind", "nmos")

            page.click("#mesh-coarse")
            assert page.input_value('[data-name="n_silicon"]') == "29"
            assert "percent" in page.inner_text("#mesh-note")

            page.click("#mesh-converged")
            assert page.input_value('[data-name="n_silicon"]') == "101"
            assert errors == []
        finally:
            browser.close()


def test_a_negative_log_slider_runs_in_decades_and_a_2d_one_does_not_solve(
    server,
) -> None:
    """A p-type body is a negative doping, and log10 of it is NaN, which
    parked the slider at its middle. It runs in decades of the size instead.
    A 2D device takes seconds to minutes a solve, so its slider only sets the
    knob and the solve button stays the way to run it."""
    drag = """(at) => {
      const slider = document.querySelector('[data-slider="substrate_doping"]');
      slider.value = String(Number(slider.min) + at * (slider.max - slider.min));
      slider.dispatchEvent(new Event("input", { bubbles: true }));
    }"""
    box = """() => Number(document.querySelector(
      '#device-knobs [data-name="substrate_doping"]').value)"""
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        try:
            page = browser.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(server)
            wait_until(page, "el('state').textContent === 'ready'", errors)
            page.select_option("#device-kind", "mos_cap")

            slider = '[data-slider="substrate_doping"]'
            assert page.get_attribute(slider, "min") == "-19"
            assert page.get_attribute(slider, "max") == "-14"
            assert page.input_value(slider) == "-16"

            before = page.evaluate("state.job")
            page.evaluate(drag, 0.0)
            assert page.evaluate(box) == -1e19
            page.evaluate(drag, 1.0)
            assert page.evaluate(box) == -1e14
            page.evaluate(drag, 0.5)
            assert math.isclose(page.evaluate(box), -(10**16.5), rel_tol=2e-2)

            page.wait_for_timeout(1000)
            assert page.evaluate("state.job") == before
            assert page.inner_text("#state") == "ready"
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
            # Each run in the rail carries a picture of its own curve: the
            # one on screen and the one kept from the first solve. This sweep
            # stops after 0 V, and a one-point run still gets its dot.
            # "done" lands before the final curve is fetched, so this waits.
            wait_until(
                page,
                "document.querySelectorAll('#runs-note svg polyline').length === 2",
                errors,
            )
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


# ------------------------------------------------------ stage 5, drawing in 2D


def electrode(page: Page, row: int, field: str, value: str) -> None:
    """Type one field of one electrode row, as a student would."""
    box = page.locator("#drawing-electrodes > div").nth(row)
    box.locator(f"[data-part={field}]").fill(value)
    box.locator(f"[data-part={field}]").dispatch_event("change")


def test_a_student_draws_a_device_is_refused_solves_it_and_saves_it(
    server, tmp_path
) -> None:
    """The whole Stage 5 path in a real browser: the drawing opens as the
    benchmark nmos, a gate dragged onto silicon is refused as a Schottky
    contact, the coarse mesh solves, the result is labelled as an unvalidated
    structure, and the saved file loads back as the same drawing."""
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        try:
            page = browser.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(server)
            wait_until(page, "el('state').textContent === 'ready'", errors)

            assert page.is_hidden("#drawing")
            page.select_option("#device-kind", "drawing")
            assert page.is_visible("#drawing")
            assert page.locator("#drawing-blocks > div").count() == 2
            assert page.locator("#drawing-implants > div").count() == 3
            assert page.locator("#drawing-electrodes > div").count() == 4
            assert page.input_value("#contact") == "gate"
            assert "20000" in page.inner_text("#drawing-note")

            # A drag along the bottom edge with the gate tool adds a gate
            # there, snapped onto y = 0, which is silicon.
            page.select_option("#drawing-tool", "gate")
            view = page.locator("#drawing-view").bounding_box()
            bottom = view["y"] + view["height"] - 2
            page.mouse.move(view["x"] + 0.3 * view["width"], bottom)
            page.mouse.down()
            page.mouse.move(view["x"] + 0.7 * view["width"], bottom)
            page.mouse.up()
            assert page.locator("#drawing-electrodes > div").count() == 5
            added = page.locator("#drawing-electrodes > div").nth(4)
            assert added.locator("[data-part=y0]").input_value() == "0"
            assert added.locator("[data-part=kind]").input_value() == "gate"

            page.click("#mesh-coarse")
            page.select_option("#sweep-kind", "transfer")
            page.fill("#voltages", "0.6, 1.2")
            page.click("#solve")
            wait_until(page, "el('state').textContent === 'refused'", errors)
            assert "Schottky" in page.inner_text("#message")

            added.locator("button").click()
            assert page.locator("#drawing-electrodes > div").count() == 4
            electrode(page, 1, "voltage", "0.05")
            solved(page, errors)
            assert page.evaluate("state.points.length") == 2
            assert page.evaluate("state.points[1].value > state.points[0].value")
            assert page.is_visible("#built-note")

            with page.expect_download() as saving:
                page.click("#device-save")
            saved = json.loads(saving.value.path().read_text(encoding="utf-8"))
            assert saved["kind"] == "drawing"
            assert saved["parameters"]["electrodes"][1]["voltage"] == 0.05
            assert saved["parameters"]["nx"] == 39

            page.select_option("#device-kind", "nmos")
            assert page.is_hidden("#drawing")
            handed_over = tmp_path / "drawn.json"
            handed_over.write_text(json.dumps(saved), encoding="utf-8")
            page.set_input_files("#device-load", str(handed_over))
            wait_until(page, "el('device-kind').value === 'drawing'", errors)
            assert page.locator("#drawing-electrodes > div").count() == 4
            second = page.locator("#drawing-electrodes > div").nth(1)
            assert second.locator("[data-part=voltage]").input_value() == "0.05"
            assert page.inner_text("#message") == ""
            assert errors == []
        finally:
            browser.close()


def test_a_redrawn_canvas_keeps_its_height_on_a_scaled_screen(server) -> None:
    """fit() used to read back the height it had just multiplied by the
    device pixel ratio, so every redraw grew the canvas by that ratio.

    The height it should hold at is read off the element rather than written
    out here. What this test is about is that redrawing changes nothing, and
    a literal pinned the editor's canvas to one size as a side effect.
    """
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        try:
            page = browser.new_page(device_scale_factor=2)
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(server)
            wait_until(page, "el('state').textContent === 'ready'", errors)
            page.select_option("#device-kind", "drawing")
            measured = page.evaluate(
                """() => {
                  const view = el('drawing-view');
                  // The css height, not the attribute: fit() writes the
                  // attribute, so reading it back here would be asking the
                  // code under test what it should have done. The stylesheet
                  // is an independent declaration of the same number, and
                  // the two being equal is itself part of the contract.
                  const declared = parseFloat(getComputedStyle(view).height);
                  const ratio = window.devicePixelRatio || 1;
                  const drawn = [1, 2, 3].map(() => {
                    drawPreview();
                    return view.height;
                  });
                  return { drawn: drawn, want: declared * ratio };
                }"""
            )

            assert measured["want"] > 0
            assert measured["drawn"] == [measured["want"]] * 3
            assert errors == []
        finally:
            browser.close()


def test_the_potential_image_puts_each_node_where_the_cutline_reads_it(
    server,
) -> None:
    """The cutline and the streamlines put node i at i / (nx - 1) of the
    width. The image used to stretch nx pixels across it, which put node i at
    (i + 0.5) / nx, so a cutline landed up to half a cell from what the
    picture showed under the cursor. A ramp in i, read back at each node's
    pixel, has to be that node's own colour.

    The colour expected at each node is asked of the page's own ramp rather
    than written out again here. What this test is about is which node lands
    under which pixel, and a second copy of the colour formula only made a
    change of palette look like a change of placement."""
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        try:
            page = browser.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(server)
            wait_until(page, "el('state').textContent === 'ready'", errors)

            read = page.evaluate(
                """() => {
                  const nx = 5, ny = 3;
                  const psi = new Float64Array(nx * ny);
                  for (let j = 0; j < ny; j++)
                    for (let i = 0; i < nx; i++) psi[j * nx + i] = i;
                  const box = fit(el('profile'));
                  drawImage(box, { shape: [ny, nx], arrays: { psi: psi } });
                  const ratio = window.devicePixelRatio || 1;
                  return [1, 2, 3].map((i) => {
                    const x = Math.round((i / (nx - 1)) * box.width * ratio);
                    const y = Math.round(0.5 * box.height * ratio);
                    return {
                      drawn: Array.from(
                        box.pen.getImageData(x, y, 1, 1).data.slice(0, 3)),
                      wanted: ramp(i / (nx - 1)),
                    };
                  });
                }"""
            )

            for i, node in zip([1, 2, 3], read, strict=True):
                assert node["drawn"] == pytest.approx(node["wanted"], abs=4), (
                    i,
                    node,
                )
            assert errors == []
        finally:
            browser.close()


def test_a_cutline_dragged_on_a_mosfet_reads_the_node_values(server) -> None:
    """The cutline was only ever checked for its panel height. A drag down
    the middle of a solved nmos has to draw the bands along it, and sampling
    straight down a mesh column at its own nodes has to give back exactly the
    node values the server sent, over the column's real length."""
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        try:
            page = browser.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(server)
            wait_until(page, "el('state').textContent === 'ready'", errors)
            page.select_option("#device-kind", "nmos")
            for name, value in COARSE_FET.items():
                page.fill(f'[data-name="{name}"]', value)
            page.fill("#voltages", "1.0")
            page.click("#solve")
            wait_until(
                page,
                "el('state').textContent.startsWith('done') && state.fields !== null",
                errors,
            )

            page.locator("#profile").scroll_into_view_if_needed()
            profile = page.locator("#profile").bounding_box()
            assert profile is not None
            middle = profile["x"] + 0.5 * profile["width"]
            page.mouse.move(middle, profile["y"] + 2)
            page.mouse.down()
            page.mouse.move(middle, profile["y"] + profile["height"] - 2)
            page.mouse.up()

            assert page.inner_text("#cutline-note") == "bands along the line you drew"
            cut = page.evaluate("state.cutline")
            nx = page.evaluate("state.fields.shape[1]")
            assert cut["from"]["i"] == pytest.approx((nx - 1) / 2, abs=0.5)
            assert cut["from"]["j"] > cut["to"]["j"]

            column = page.evaluate(
                """() => {
                  const f = state.fields, ny = f.shape[0], nx = f.shape[1];
                  const i = Math.floor(nx / 2);
                  const along = sampleAlong(f, 'Ec', {i: i, j: 0},
                    {i: i, j: ny - 1}, ny);
                  const nodes = [];
                  for (let j = 0; j < ny; j++) nodes.push(f.arrays.Ec[j * nx + i]);
                  return { sampled: Array.from(along.values), nodes: nodes,
                    length: along.s[ny - 1],
                    span: f.arrays.y[ny - 1] - f.arrays.y[0] };
                }"""
            )
            pairs = zip(column["sampled"], column["nodes"], strict=True)
            for sampled, node in pairs:
                if math.isnan(node):  # an oxide node, where the line breaks
                    assert math.isnan(sampled)
                else:
                    assert sampled == pytest.approx(node, rel=1e-12, abs=1e-12)
            assert column["length"] == pytest.approx(column["span"], rel=1e-12)
            assert errors == []
        finally:
            browser.close()


def test_the_last_explanation_asked_for_is_the_one_shown(server) -> None:
    """Two clicks in quick succession: if the first topic's answer comes
    back after the second's, the drawer must still show the second."""
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        try:
            page = browser.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(server)
            wait_until(page, "el('state').textContent === 'ready'", errors)
            wanted = page.evaluate(
                "fetch('/api/learn/band-diagram').then((r) => r.json())"
                ".then((t) => t.title)"
            )

            held: list = []
            page.route("**/api/learn/sweep-kinds", lambda route: held.append(route))
            page.click('[data-topic-id="sweep-kind"] > .explain')
            page.wait_for_timeout(300)
            assert len(held) == 1
            page.click('[data-topic-id="bands-view"] > .explain')
            wait_until(
                page,
                f"el('drawer-title').textContent === {json.dumps(wanted)}",
                errors,
            )

            held[0].continue_()
            page.wait_for_timeout(500)

            assert page.inner_text("#drawer-title") == wanted
            assert errors == []
        finally:
            browser.close()


def test_blocks_and_implants_can_be_added_by_dragging(server) -> None:
    """Only a gate had ever been dragged in. An oxide block dragged edge to
    edge snaps to the device's own width, and an n implant dragged inside it
    arrives as an n row at the starting concentration."""
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        try:
            page = browser.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(server)
            wait_until(page, "el('state').textContent === 'ready'", errors)
            page.select_option("#device-kind", "drawing")
            view = page.locator("#drawing-view").bounding_box()
            assert view is not None

            def drag(tool: str, x0: float, y0: float, x1: float, y1: float) -> None:
                page.select_option("#drawing-tool", tool)
                page.mouse.move(
                    view["x"] + x0 * view["width"], view["y"] + y0 * view["height"]
                )
                page.mouse.down()
                page.mouse.move(
                    view["x"] + x1 * view["width"], view["y"] + y1 * view["height"]
                )
                page.mouse.up()

            right = page.evaluate("extent(drawingSoFar()).width")
            drag("oxide", 0.002, 0.1, 0.998, 0.3)
            blocks = page.locator("#drawing-blocks > div")
            assert blocks.count() == 3
            oxide = blocks.nth(2)
            assert oxide.locator("[data-part=material]").input_value() == "oxide"
            assert float(oxide.locator("[data-part=x0]").input_value()) == 0
            assert float(oxide.locator("[data-part=x1]").input_value()) == right

            drag("n", 0.4, 0.6, 0.6, 0.8)
            implants = page.locator("#drawing-implants > div")
            assert implants.count() == 4
            added = implants.nth(3)
            assert added.locator("[data-part=dopant]").input_value() == "n"
            concentration = added.locator("[data-part=concentration]").input_value()
            assert float(concentration) == 1e18
            x0 = float(added.locator("[data-part=x0]").input_value())
            x1 = float(added.locator("[data-part=x1]").input_value())
            assert 0 < x0 < x1 < right
            assert errors == []
        finally:
            browser.close()
