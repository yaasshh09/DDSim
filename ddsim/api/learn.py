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

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ddsim.api.devices import COARSE

LEARN = Path(__file__).parent / "static" / "learn"
"""Where the topic files live."""

LESSONS = Path(__file__).parent / "lessons"
"""Where the guided experiments live, one markdown file each."""

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
    fields, body = _split_header(name, text, ("title", "summary", "docs"))

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


def _split_header(
    name: str, text: str, required: tuple[str, ...]
) -> tuple[dict[str, str], str]:
    """The `key: value` lines between two `---` lines, and the rest.

    Raises ValueError naming the file and the first required key it lacks.
    """
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise ValueError(f"{name}: no --- header")
    header, body = text[4:].split("\n---\n", 1)
    fields: dict[str, str] = {}
    for line in header.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    for key in required:
        if not fields.get(key):
            raise ValueError(f"{name}: header has no {key}")
    return fields, body


# ---------------------------------------------------------------- lessons

_STEPS = "## Steps"
_LOOK = "## What to look for"
_SAW = "## What you saw"
_SET = "set:"


@dataclass(frozen=True)
class Step:
    """One thing the student is asked to do."""

    title: str

    text: str
    """What to change and what to watch, markdown."""

    request: dict[str, Any] | None
    """The whole job this step sets up, the lesson's start with the step's own
    changes on top, or None for a step that changes nothing. Whole rather than
    a difference so the page and the claim tests read the same request, and
    neither has to know how the two were merged."""


@dataclass(frozen=True)
class Lesson:
    """A guided experiment, phases/PHASE-7.md Stage 3."""

    name: str
    title: str
    summary: str

    claims: tuple[str, ...]
    """What the lesson teaches, by name. Each one is a test of the same name
    in tests/analytic/test_lesson_claims.py, run on this lesson's own
    requests, and a claim with no test fails a unit test."""

    request: dict[str, Any]
    """The job the lesson starts from, as POST /api/jobs takes it."""

    mesh_note: str
    """What the coarse mesh costs, where the lesson starts on one. Empty on
    the converged mesh."""

    steps: tuple[Step, ...]
    look_for: str
    explanation: str
    """What the student saw and why, markdown with $maths$."""

    def step(self, title: str) -> Step:
        """The step with this title. How a claim test names the request it
        runs, so a reordered lesson still checks the right one."""
        for step in self.steps:
            if step.title == title:
                return step
        raise KeyError(f"{self.name} has no step {title!r}")


def lesson_names() -> tuple[str, ...]:
    """Every lesson, in the order the page offers them."""
    return tuple(sorted(path.stem for path in LESSONS.glob("*.md")))


def load_lesson(name: str) -> Lesson:
    """Read and check one lesson.

    Args:
        name: a value lesson_names() returned. Anything else is a KeyError,
            which keeps a path in the name away from the filesystem.
    """
    if name not in lesson_names():
        raise KeyError(f"no lesson {name!r}")
    return parse_lesson(name, (LESSONS / f"{name}.md").read_text(encoding="utf-8"))


def parse_lesson(name: str, text: str) -> Lesson:
    """One lesson from its text.

    Args:
        name: what to call it in a refusal.
        text: the file. A header with title, summary, claims (separated by
            semicolons), device and sweep (one line of JSON each, as POST
            /api/jobs takes them) and optionally `mesh: coarse`. Then the
            three sections, the steps as `###` headings under the first. A
            step may open with a `set:` line of JSON, `{"device": {...},
            "sweep": {...}}`, laid over the lesson's start.

    Raises ValueError naming the lesson and what is wrong with it.
    """
    fields, body = _split_header(
        name, text, ("title", "summary", "claims", "device", "sweep")
    )
    device = _json(name, "device", fields["device"])
    start = {"device": device, "sweep": _json(name, "sweep", fields["sweep"])}
    device.setdefault("parameters", {})

    mesh_note = ""
    if fields.get("mesh") == "coarse":
        kind = device.get("kind")
        if kind not in COARSE:
            raise ValueError(f"{name}: {kind} has no coarse mesh")
        preset = COARSE[kind]
        # The lesson's own settings win, so a lesson can move one mesh knob
        # and keep the rest of the preset.
        device["parameters"] = {**preset.parameters, **device["parameters"]}
        # The preset's note was measured on the device's defaults. A lesson
        # that changes the device measures its own and says so here.
        mesh_note = fields.get("mesh_note") or preset.note

    for section in (_STEPS, _LOOK, _SAW):
        if section not in body:
            raise ValueError(f"{name}: no '{section}' section")
    steps, rest = body.split(_STEPS, 1)[1].split(_LOOK, 1)
    look_for, explanation = rest.split(_SAW, 1)

    return Lesson(
        name=name,
        title=fields["title"],
        summary=fields["summary"],
        claims=tuple(c.strip() for c in fields["claims"].split(";") if c.strip()),
        request=start,
        mesh_note=mesh_note,
        steps=tuple(
            _step(name, start, chunk) for chunk in steps.split("### ")[1:]
        ),
        look_for=look_for.strip(),
        explanation=explanation.strip(),
    )


def _step(name: str, start: dict[str, Any], chunk: str) -> Step:
    """One `###` step: its title, an optional set line, then its text."""
    title, _, text = chunk.partition("\n")
    title = title.strip()
    text = text.strip()
    if not text.startswith(_SET):
        return Step(title=title, text=text, request=None)

    line, _, text = text.partition("\n")
    changes = _json(name, title, line[len(_SET) :])
    request = copy.deepcopy(start)
    request["device"]["parameters"].update(changes.get("device", {}))
    for key, value in changes.get("sweep", {}).items():
        # A dict of knobs is merged into, so a step that sets one setting
        # keeps the lesson's others. Anything else is replaced whole.
        if isinstance(value, dict):
            request["sweep"].setdefault(key, {}).update(value)
        else:
            request["sweep"][key] = value
    return Step(title=title, text=text.strip(), request=request)


def _json(name: str, what: str, text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name}: {what} is not JSON: {error}") from error


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
    "compare-runs": "curve-plot",
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
