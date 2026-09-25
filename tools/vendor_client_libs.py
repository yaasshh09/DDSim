"""Put marked, KaTeX and the page's typeface into ddsim/api/static/vendor.

Run once, and again only to upgrade. For the npm packages it reads each one's
latest version from the registry, downloads the tarball, checks it against the
registry's sha512 integrity string and extracts only the files the page needs.

The typeface is not downloaded at all. The release is dropped in fonts/ and
this subsets the two weights the page uses down to the characters it draws,
which is the whole reason the vendored tree is 33 KB rather than 3 MB.

Either way it writes VENDOR.json with a sha256 per file so
tests/unit/test_vendor.py can hold the tree to it.

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

FONT_SOURCE = ROOT / "fonts"
"""Where the family is dropped before vendoring. That folder holds the whole
Poppins release, nine weights and their italics, which is more than three
megabytes and seventeen faces the page never asks for."""

FONT_WEIGHTS = {"Poppins-Medium": 500, "Poppins-SemiBold": 600}
"""The two weights the page is set in. Medium carries everything that is read,
SemiBold everything that is a heading or a value worth finding."""

FONT_SUBSET = (
    "U+0020-007E,U+00A0-00FF,U+00B7,U+2018-201D,U+2022,U+2026,U+2013-2014,"
    "U+00D7,U+2212,U+00B5,U+03BC,U+03B2,U+03A9,U+25B8,U+25BE,U+2713,U+2192"
)
"""Latin, the punctuation the labels use, and the few greek letters and arrows
the page draws. Poppins also ships Devanagari, which is most of the file and
none of this page: subsetting takes each weight from 156 KB to under 17 KB."""

FACE = """@font-face {{
  font-family: 'Poppins';
  font-style: normal;
  font-weight: {weight};
  font-display: swap;
  src: url(/static/vendor/fonts/{slug}) format('truetype');
}}"""

FONTS_HEADER = """/* Poppins, subset to the characters this page draws.

   Vendored from the release in fonts/ rather than fetched, because
   phases/PHASE-7.md says the page works with no network. Regenerate with
   tools/vendor_client_libs.py, which needs that folder present.

   Poppins has no tabular figure feature and its digits are proportional: a
   one is 350 units against a zero's 647. Every number the page shows sits in
   a box of its own width and #state is pinned, so a value changing its digits
   moves nothing around it. See docs/08-design.md. */

"""


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return bytes(response.read())


def fonts() -> dict[str, str]:
    """Subset each wanted weight into the vendor tree, write a local fonts.css
    naming them, and return the vendored path to sha256 map."""
    from fontTools import subset

    into = VENDOR / "fonts"
    into.mkdir(parents=True, exist_ok=True)
    for stale in into.iterdir():
        stale.unlink()

    written: dict[str, str] = {}
    faces: list[str] = []
    for name, weight in FONT_WEIGHTS.items():
        source = FONT_SOURCE / f"{name}.ttf"
        assert source.exists(), f"{source} is missing, drop the release in fonts/"
        slug = f"{name.lower()}-latin.ttf"
        subset.main(
            [
                str(source),
                "--unicodes=" + FONT_SUBSET,
                "--layout-features=*",
                "--output-file=" + str(into / slug),
            ]
        )
        data = (into / slug).read_bytes()
        written[f"fonts/{slug}"] = hashlib.sha256(data).hexdigest()
        faces.append(FACE.format(weight=weight, slug=slug))

    stylesheet = (FONTS_HEADER + "\n\n".join(faces) + "\n").encode("utf-8")
    (into / "fonts.css").write_bytes(stylesheet)
    written["fonts/fonts.css"] = hashlib.sha256(stylesheet).hexdigest()

    licence = (FONT_SOURCE / "OFL.txt").read_bytes()
    (into / "POPPINS-OFL.txt").write_bytes(licence)
    written["fonts/POPPINS-OFL.txt"] = hashlib.sha256(licence).hexdigest()

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
            "version": "Poppins " + ", ".join(sorted(FONT_WEIGHTS)),
            "licence": "fonts/POPPINS-OFL.txt",
            "files": fonts(),
        }
    )

    (VENDOR / "VENDOR.json").write_bytes(
        (json.dumps({"packages": packages}, indent=2) + "\n").encode("utf-8")
    )


if __name__ == "__main__":
    main()
