"""The explanations behind the page, phases/PHASE-7.md part two.

Each topic is one markdown file in static/learn, in two layers: "In plain
words" for a first course in semiconductors, "In more depth" with the
equations. A short header names the title, a one line summary and the headings
in docs/ the depth layer condenses.

The three maps say which topic explains which part of the page. They are the
only place that decision is written down, and tests/unit/test_learn.py holds
them complete: a knob, marked element or status with no topic fails a test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LEARN = Path(__file__).parent / "static" / "learn"
"""Where the topic files live."""

_PLAIN = "## In plain words"
_DEPTH = "## In more depth"


@dataclass(frozen=True)
class Topic:
    """One explanation, split into the parts the page renders separately."""

    name: str
    """The file name without .md, which is also the URL segment."""

    title: str
    summary: str

    docs: tuple[str, ...]
    """References into docs/, each `file.md#Heading text`."""

    plain: str
    """The plain layer, markdown, without its heading."""

    depth: str
    """The depth layer, markdown with $inline$ and $$display$$ maths."""


def topic_names() -> tuple[str, ...]:
    """Every topic, sorted."""
    return tuple(sorted(path.stem for path in LEARN.glob("*.md")))


def load_topic(name: str) -> Topic:
    """Read and check one topic.

    Args:
        name: a value topic_names() returned. Anything else is a KeyError,
            which is also what keeps a path in the name from reaching the
            filesystem.

    Raises ValueError naming what is missing from a malformed file.
    """
    if name not in topic_names():
        raise KeyError(f"no topic {name!r}")
    text = (LEARN / f"{name}.md").read_text(encoding="utf-8")

    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise ValueError(f"{name}: no --- header")
    header, body = text[4:].split("\n---\n", 1)
    fields: dict[str, str] = {}
    for line in header.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    for key in ("title", "summary", "docs"):
        if not fields.get(key):
            raise ValueError(f"{name}: header has no {key}")

    if _PLAIN not in body:
        raise ValueError(f"{name}: no '{_PLAIN}' section")
    if _DEPTH not in body:
        raise ValueError(f"{name}: no '{_DEPTH}' section")
    plain, depth = body.split(_PLAIN, 1)[1].split(_DEPTH, 1)

    return Topic(
        name=name,
        title=fields["title"],
        summary=fields["summary"],
        docs=tuple(part.strip() for part in fields["docs"].split(";") if part.strip()),
        plain=plain.strip(),
        depth=depth.strip(),
    )


KNOB_TOPICS: dict[str, str] = {
    # pn diode
    "Na": "doping",
    "Nd": "doping",
    "length": "pn-diode",
    "junction": "pn-diode",
    "n_nodes": "mesh",
    "h_min": "mesh",
    "anode_voltage": "contacts-and-bias",
    "cathode_voltage": "contacts-and-bias",
    # MOS capacitor
    "substrate_doping": "doping",
    "t_ox": "mos-capacitor",
    "t_si": "mos-capacitor",
    "width": "mos-capacitor",
    "nx": "mesh",
    "n_silicon": "mesh",
    "n_oxide": "mesh",
    "gate_voltage": "contacts-and-bias",
    "body_voltage": "contacts-and-bias",
    "work_function": "mos-capacitor",
    # nMOSFET
    "L_gate": "mosfet",
    "sd_length": "mosfet",
    "contact_length": "mosfet",
    "sd_peak": "doping",
    "x_j": "doping",
    "lateral_diffusion": "doping",
    "n_contact": "mesh",
    "n_sd": "mesh",
    "n_channel": "mesh",
    "h_min_x": "mesh",
    "h_min_y": "mesh",
    "drain_voltage": "contacts-and-bias",
    "source_voltage": "contacts-and-bias",
    "degenerate": "fermi-dirac-statistics",
    # sweeps
    "step": "continuation",
    "start": "continuation",
    "max_iterations": "convergence",
    "update_tol": "convergence",
    "response": "cv-sweep",
    # models
    "mobility": "mobility",
    "auger": "recombination",
    "field_dependent": "velocity-saturation",
    "surface": "surface-scattering",
}
"""Which topic explains each knob, by argument name. A name shared by two
devices means the same thing on both, which is why this is keyed by name."""

PLOT_TOPICS: dict[str, str] = {
    "residual-plot": "residual-plot",
    "legend-psi-residual": "newton",
    "legend-n-residual": "newton",
    "legend-p-residual": "newton",
    "legend-gummel-update": "gummel",
    "legend-rejected-step": "continuation",
    "curve-plot": "curve-plot",
    "profile-plot": "profile-plot",
    "legend-psi": "potential",
    "legend-n": "carrier-densities",
    "legend-p": "carrier-densities",
    "bands-view": "band-diagram",
    "legend-quasi-fermi": "quasi-fermi-levels",
    "cutline": "cutline",
    "streamlines": "current-flow",
    "sweep-kind": "sweep-kinds",
    "device-kind": "devices",
}
"""Which topic explains each marked element on the page, by its data-topic-id."""

STATUS_TOPICS: dict[str, str] = {
    "done": "convergence",
    "failed": "failed",
    "cancelled": "cancelled",
    "stalled": "stalled",
    "dropped": "dropped-frames",
}
"""Which topic explains each way a job can end, and dropped telemetry."""
