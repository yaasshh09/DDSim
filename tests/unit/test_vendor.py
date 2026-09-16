"""The libraries the page renders explanations with, vendored.

phases/PHASE-7.md part two: nothing loads from a CDN, the page works with no
network. So marked and KaTeX live in the repo, and this file holds them to the
manifest tools/vendor_client_libs.py wrote: every file present, every hash
matching, every licence shipped. A hand edit to a vendored file fails here.
"""

from __future__ import annotations

import hashlib
import json

from ddsim.api.app import PAGE

VENDOR = PAGE.parent / "vendor"
MANIFEST = json.loads((VENDOR / "VENDOR.json").read_text(encoding="utf-8"))


def test_both_libraries_are_vendored() -> None:
    names = {package["name"] for package in MANIFEST["packages"]}

    assert names == {"katex", "marked"}


def test_every_vendored_file_matches_its_recorded_hash() -> None:
    for package in MANIFEST["packages"]:
        for relative, expected in package["files"].items():
            data = (VENDOR / relative).read_bytes()
            assert hashlib.sha256(data).hexdigest() == expected, relative


def test_every_library_ships_its_licence() -> None:
    for package in MANIFEST["packages"]:
        assert (VENDOR / package["licence"]).read_text(encoding="utf-8").strip()


def test_the_page_names_no_remote_host() -> None:
    page = PAGE.read_text(encoding="utf-8")

    assert "http://" not in page
    assert "https://" not in page
