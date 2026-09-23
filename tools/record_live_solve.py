"""Record the page solving a MOSFET Id-Vg live, for the README.

phases/PHASE-7.md asks for a short screen capture of a MOSFET transfer curve
solving in the browser: the residual falling per iteration and the curve
drawing point by point. This starts the same server `ddsim serve` does,
drives the real page with Playwright, records the tab, and turns the
recording into docs/images/mosfet_live.gif with ffmpeg.

The capture shows the solver's own numbers. Nothing here touches physics.

Usage:
    .venv/Scripts/python.exe tools/record_live_solve.py
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import uvicorn
from playwright.sync_api import ViewportSize, sync_playwright

from ddsim.api.app import create_app
from ddsim.api.jobs import JobRegistry

OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "images" / "mosfet_live.gif"
VIEWPORT: ViewportSize = {"width": 1280, "height": 800}
GATE_VOLTAGES = "0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2"
"""Gate sweep [V]."""

DRAIN_VOLTAGE = "0.05"
"""Drain bias [V]. The page starts the nmos at 0 V, where Id is zero and the
curve would be roundoff."""

SOLVE_TIMEOUT_MS = 600_000
"""A coarse nmos sweep takes tens of seconds; this is room for a slow machine [ms]."""

GIF_FPS = 8
GIF_WIDTH = 800
"""Output frame rate [1/s] and width [px]. Kept small so the README loads fast."""


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def start_server() -> tuple[uvicorn.Server, threading.Thread, str]:
    port = free_port()
    config = uvicorn.Config(
        create_app(JobRegistry()), host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    if not server.started:
        sys.exit("uvicorn did not start")
    return server, thread, f"http://127.0.0.1:{port}"


def record(url: str, video_dir: Path) -> Path:
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        context = browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(video_dir),
            record_video_size=VIEWPORT,
        )
        page = context.new_page()
        page.goto(url)
        page.wait_for_function("el('state').textContent === 'ready'")
        page.select_option("#device-kind", "nmos")
        page.click("#mesh-coarse")
        page.fill('[data-name="drain_voltage"]', DRAIN_VOLTAGE)
        page.select_option("#sweep-kind", "transfer")
        page.fill("#voltages", GATE_VOLTAGES)
        page.wait_for_timeout(800)
        page.click("#solve")
        page.wait_for_function(
            "el('state').textContent.startsWith('done')", timeout=SOLVE_TIMEOUT_MS
        )
        page.wait_for_timeout(2500)
        video = page.video
        context.close()
        browser.close()
        assert video is not None
        return Path(video.path())


def to_gif(video: Path, gif: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        sys.exit("ffmpeg is not on PATH; the webm is at " + str(video))
    scale = f"fps={GIF_FPS},scale={GIF_WIDTH}:-1:flags=lanczos"
    palette = "split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"{scale},{palette}",
            str(gif),
        ],
        check=True,
    )


def main() -> None:
    server, thread, url = start_server()
    try:
        with tempfile.TemporaryDirectory() as scratch:
            video = record(url, Path(scratch))
            to_gif(video, OUTPUT)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
