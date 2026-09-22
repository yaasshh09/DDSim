"""Fetch marked, KaTeX and the two design fonts into ddsim/api/static/vendor.

Run once, and again only to upgrade. For the npm packages it reads each one's
latest version from the registry, downloads the tarball, checks it against the
registry's sha512 integrity string and extracts only the files the page needs.
For the fonts it asks Google for the css2 response and rewrites every url to a
local file, because a stylesheet that reaches fonts.gstatic.com is a CDN by
another name. Either way it writes VENDOR.json with a sha256 per file so
tests/unit/test_vendor.py can hold the tree to it.

    .venv/Scripts/python tools/vendor_client_libs.py
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import pathlib
import re
import tarfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDOR = ROOT / "ddsim" / "api" / "static" / "vendor"

WANTED = {
    "katex": {
        "package/dist/katex.min.js": "katex/katex.min.js",
        "package/dist/katex.min.css": "katex/katex.min.css",
        "package/dist/contrib/auto-render.min.js": "katex/contrib/auto-render.min.js",
        "package/LICENSE": "katex/LICENSE",
    },
    "marked": {
        "package/lib/marked.umd.js": "marked/marked.umd.js",
        "package/LICENSE": "marked/LICENSE",
    },
}
"""Tarball path to vendored path, per package. KaTeX's woff2 fonts are added
by pattern below, since their names carry version specific hashes."""

FONTS_CSS = (
    "https://fonts.googleapis.com/css2"
    "?family=Schibsted+Grotesk:wght@400;500;600;700&display=swap"
)
"""The one family docs/08-design.md names, at the weights the page uses.

One rather than two. The second face was a monospace, and the only job it had
was keeping a number from reshuffling when a slider changed it. Tabular
figures do that inside a proportional face, so the terminal look was paying
for nothing.
"""

FONT_LICENCES = {
    "SCHIBSTED-GROTESK-OFL.txt": (
        "https://raw.githubusercontent.com/google/fonts/main/ofl/"
        "schibstedgrotesk/OFL.txt"
    ),
}
"""SIL OFL 1.1, and the page ships the text with the font."""

BROWSER = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
"""Google serves woff1 and one undivided face to a client it does not know.
The per subset woff2 blocks this script rewrites only arrive with this."""

FONTS_HEADER = """/* Schibsted Grotesk, every subset Google serves for it.
   Generated from the Google Fonts css2 response with the urls rewritten to
   local files, because phases/PHASE-7.md says the page works with no
   network. Regenerate with tools/vendor_client_libs.py. */

"""


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": BROWSER})
    with urllib.request.urlopen(request, timeout=60) as response:
        return bytes(response.read())


def fonts() -> dict[str, str]:
    """Download every woff2 the css2 response names, write a local fonts.css
    pointing at them, and return the vendored path to sha256 map."""
    into = VENDOR / "fonts"
    into.mkdir(parents=True, exist_ok=True)
    source = fetch(FONTS_CSS).decode("utf-8")

    faces = re.findall(r"/\*\s*([a-z-]+)\s*\*/\s*(@font-face\s*\{.*?\})", source, re.S)
    assert faces, "Google returned no per subset faces, check the user agent"

    local: dict[str, str] = {}
    written: dict[str, str] = {}
    rewritten: list[str] = []
    for subset, face in faces:
        family = re.search(r"font-family:\s*'([^']+)'", face)
        url = re.search(r"url\((https://[^)]+)\)", face)
        assert family is not None and url is not None, face
        if url.group(1) not in local:
            slug = f"{family.group(1).lower().replace(' ', '-')}-{subset}.woff2"
            data = fetch(url.group(1))
            (into / slug).write_bytes(data)
            local[url.group(1)] = slug
            written[f"fonts/{slug}"] = hashlib.sha256(data).hexdigest()
        rewritten.append(
            face.replace(url.group(1), f"/static/vendor/fonts/{local[url.group(1)]}")
        )

    stylesheet = (FONTS_HEADER + "\n\n".join(rewritten) + "\n").encode("utf-8")
    (into / "fonts.css").write_bytes(stylesheet)
    written["fonts/fonts.css"] = hashlib.sha256(stylesheet).hexdigest()

    for name, url in FONT_LICENCES.items():
        data = fetch(url)
        (into / name).write_bytes(data)
        written[f"fonts/{name}"] = hashlib.sha256(data).hexdigest()

    return dict(sorted(written.items()))


def main() -> None:
    packages = []
    for name, files in WANTED.items():
        meta = json.loads(fetch(f"https://registry.npmjs.org/{name}/latest"))
        tarball = fetch(meta["dist"]["tarball"])
        algorithm, digest = meta["dist"]["integrity"].split("-", 1)
        assert algorithm == "sha512", algorithm
        actual = base64.b64encode(hashlib.sha512(tarball).digest()).decode()
        assert actual == digest, f"{name} tarball does not match its integrity"

        written: dict[str, str] = {}
        with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as archive:
            wanted = dict(files)
            if name == "katex":
                for member in archive.getnames():
                    font = member.startswith("package/dist/fonts/")
                    if font and member.endswith(".woff2"):
                        wanted[member] = "katex/fonts/" + member.rsplit("/", 1)[1]
            for source, target in wanted.items():
                extracted = archive.extractfile(source)
                assert extracted is not None, f"{name} has no {source}"
                data = extracted.read()
                path = VENDOR / target
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                written[target] = hashlib.sha256(data).hexdigest()

        licence = next(target for target in files.values() if "LICENSE" in target)
        packages.append(
            {
                "name": name,
                "version": meta["version"],
                "licence": licence,
                "files": dict(sorted(written.items())),
            }
        )

    packages.append(
        {
            "name": "fonts",
            "version": FONTS_CSS.split("?", 1)[1],
            "licence": "fonts/SCHIBSTED-GROTESK-OFL.txt",
            "files": fonts(),
        }
    )

    (VENDOR / "VENDOR.json").write_bytes(
        (json.dumps({"packages": packages}, indent=2) + "\n").encode("utf-8")
    )


if __name__ == "__main__":
    main()
