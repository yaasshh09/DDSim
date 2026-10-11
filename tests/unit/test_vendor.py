from __future__ import annotations
import hashlib
import json
import re
from ddsim.api.app import PAGE

VENDOR = PAGE.parent/'vendor'
MANIFEST = json.loads((VENDOR / "VENDOR.json").read_text(encoding='utf-8'))

def test_both_libraries_are_vendored() -> None:
    bar = {b['name'] for b in MANIFEST['packages']}
    assert bar == {"katex", "marked", 'fonts'}

def test_no_vendored_stylesheet_reaches_out_to_a_remote_host() -> None:
    for res in VENDOR.rglob('*.css'):
        k = res.read_text(encoding='utf-8')
        assert 'http://' not in k, res
        assert "https://" not in k, res


def test_every_vendored_file_matches_its_recorded_hash() -> None:
    for item in MANIFEST['packages']:
        for m, e in item["files"].items():
            b = (VENDOR / m).read_bytes()
            assert hashlib.sha256(b).hexdigest() == e, m
def test_every_library_ships_its_licence() -> None:
    for yy in MANIFEST["packages"]:
        assert (VENDOR / yy["licence"]).read_text(encoding="utf-8").strip()


def test_the_page_loads_nothing_from_a_remote_host() -> None:
    g = PAGE.read_text(encoding='utf-8')
    tags = re.findall(r"<(?:script|link|img|iframe|source|video|audio)\b[^>]*>", g)
    assert tags
    for t in tags:
        assert not re.search(r'(?:src|href)\s*=\s*"(?:https?:)?//', t), t
    assert not re.search(r"url\(\s*['\"]?(?:https?:)?//", g)


def test_every_link_off_the_page_opens_in_a_new_tab() -> None:
    g = PAGE.read_text(encoding='utf-8')
    for t in re.findall(r'<a\b[^>]*href="https?://[^"]*"[^>]*>', g):
        assert 'target="_blank"' in t and 'rel="noopener"' in t, t
