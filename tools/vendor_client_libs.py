"""Fetch marked and KaTeX into ddsim/api/static/vendor, verified.

Run once, and again only to upgrade. It reads each package's latest version
from the npm registry, downloads the tarball, checks it against the registry's
sha512 integrity string, extracts only the files the page needs, and writes
VENDOR.json with a sha256 per file so tests/unit/test_vendor.py can hold the
tree to it.

    .venv/Scripts/python tools/vendor_client_libs.py
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import pathlib
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


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return bytes(response.read())


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

    (VENDOR / "VENDOR.json").write_text(
        json.dumps({"packages": packages}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
