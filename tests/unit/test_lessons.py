"""The guided experiments, phases/PHASE-7.md part two, Stage 3.

A lesson is a file under ddsim/api/lessons: the device and sweep to start
from, the steps, what to look for and what it all meant. What it teaches is
held by tests/analytic/test_lesson_claims.py, on the lesson's own device. This
file holds the rest: every lesson parses, every claim it names has a check,
and every request it would hand the page is one the API accepts.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
from fastapi.testclient import TestClient

from ddsim.api.app import create_app
from ddsim.api.devices import COARSE, build_from_spec
from ddsim.api.learn import LESSONS, lesson_names, load_lesson, parse_lesson
from ddsim.api.sweeps import check_request

CLAIMS = pathlib.Path(__file__).parents[1] / "analytic" / "test_lesson_claims.py"


@pytest.fixture
def client():
    with TestClient(create_app()) as client:
        yield client


def checks() -> set[str]:
    """Every test function name in the claims file, without the test_."""
    tree = ast.parse(CLAIMS.read_text(encoding="utf-8"))
    return {
        node.name.removeprefix("test_")
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }


LESSON = """---
title: A lesson
summary: One line.
claims: first_claim; second_claim
device: {"kind": "pn_diode", "parameters": {"Na": 1e17, "length": 2e-4}}
sweep: {"kind":"iv","contact":"anode","voltages":[0.0,0.1],"settings":{"step":0.05}}
---
## Steps

### Solve it

Press solve.

### Push it

set: {"device":{"Na":1e18},"sweep":{"voltages":[0.0,-1.0],"settings":{"start":0.1}}}

Now press solve again.

## What to look for

The curve.

## What you saw

Physics.
"""


def test_a_lesson_splits_into_its_parts() -> None:
    lesson = parse_lesson("example", LESSON)

    assert lesson.title == "A lesson"
    assert lesson.claims == ("first_claim", "second_claim")
    assert [step.title for step in lesson.steps] == ["Solve it", "Push it"]
    assert lesson.steps[0].text == "Press solve."
    assert lesson.look_for == "The curve."
    assert lesson.explanation == "Physics."


def test_a_step_without_a_set_line_changes_nothing() -> None:
    lesson = parse_lesson("example", LESSON)

    assert lesson.steps[0].request is None


def test_a_step_is_the_start_with_its_own_changes_on_top() -> None:
    lesson = parse_lesson("example", LESSON)
    pushed = lesson.steps[1].request

    assert pushed is not None
    # The knob it names moves and the one it does not keep the lesson's value.
    assert pushed["device"]["parameters"] == {"Na": 1e18, "length": 2e-4}
    # A whole field is replaced, a dict of settings is merged into.
    assert pushed["sweep"]["voltages"] == [0.0, -1.0]
    assert pushed["sweep"]["settings"] == {"step": 0.05, "start": 0.1}
    assert pushed["sweep"]["contact"] == "anode"
    # And the start itself is untouched by it.
    assert lesson.request["device"]["parameters"] == {"Na": 1e17, "length": 2e-4}


def test_the_text_after_a_set_line_is_the_step() -> None:
    lesson = parse_lesson("example", LESSON)

    assert lesson.steps[1].text == "Now press solve again."


def test_a_step_finds_its_request_by_title() -> None:
    lesson = parse_lesson("example", LESSON)

    assert lesson.step("Push it") is lesson.steps[1]
    with pytest.raises(KeyError, match="Nothing"):
        lesson.step("Nothing")


def test_a_coarse_lesson_starts_on_the_coarse_mesh_it_names() -> None:
    text = LESSON.replace(
        'device: {"kind": "pn_diode", "parameters": {"Na": 1e17, "length": 2e-4}}',
        'device: {"kind": "nmos", "parameters": {"n_sd": 11}}\nmesh: coarse',
    )
    lesson = parse_lesson("example", text)
    parameters = lesson.request["device"]["parameters"]

    for name, value in COARSE["nmos"].parameters.items():
        if name != "n_sd":
            assert parameters[name] == value
    # What the lesson writes itself wins over the preset.
    assert parameters["n_sd"] == 11
    assert lesson.mesh_note == COARSE["nmos"].note


def test_a_lesson_on_its_own_device_says_what_its_own_coarse_mesh_costs() -> None:
    text = LESSON.replace(
        'device: {"kind": "pn_diode", "parameters": {"Na": 1e17, "length": 2e-4}}',
        'device: {"kind": "nmos", "parameters": {}}\nmesh: coarse\n'
        "mesh_note: Measured on this lesson's device.",
    )

    assert parse_lesson("example", text).mesh_note == "Measured on this lesson's device."


def test_a_lesson_on_the_converged_mesh_carries_no_note() -> None:
    assert parse_lesson("example", LESSON).mesh_note == ""


def test_a_coarse_mesh_a_device_does_not_have_is_refused() -> None:
    text = LESSON.replace("summary: One line.", "summary: One line.\nmesh: coarse")

    with pytest.raises(ValueError, match="pn_diode has no coarse mesh"):
        parse_lesson("example", text)


@pytest.mark.parametrize(
    ("broken", "complaint"),
    [
        (LESSON.replace("claims: first_claim; second_claim\n", ""), "claims"),
        (LESSON.replace("## What you saw", "## Something"), "What you saw"),
        (LESSON.replace("## Steps", "## Stairs"), "Steps"),
        (LESSON.replace('"voltages":[0.0,0.1]', '"voltages":[0.0,'), "sweep"),
        (LESSON.replace("set: {", "set: {{"), "Push it"),
        (LESSON.replace("---\ntitle", "title"), "header"),
    ],
)
def test_a_malformed_lesson_says_what_is_wrong(broken, complaint) -> None:
    with pytest.raises(ValueError, match=complaint):
        parse_lesson("example", broken)


def test_an_unknown_lesson_is_a_key_error() -> None:
    with pytest.raises(KeyError):
        load_lesson("../app")


# ------------------------------------------------------ the shipped lessons


def test_there_are_the_five_lessons_the_phase_asks_for() -> None:
    assert len(lesson_names()) == 5


@pytest.mark.parametrize("name", lesson_names())
def test_every_lesson_is_complete(name) -> None:
    lesson = load_lesson(name)

    assert lesson.title and lesson.summary
    assert lesson.claims, "a lesson with no claim teaches nothing a test holds"
    assert lesson.steps
    assert len(lesson.explanation.split()) >= 60
    assert lesson.look_for.strip()


@pytest.mark.parametrize("name", lesson_names())
def test_every_claim_a_lesson_names_has_a_check(name) -> None:
    missing = set(load_lesson(name).claims) - checks()

    assert not missing, f"{name} claims {sorted(missing)} and nothing tests it"


def test_every_check_belongs_to_a_lesson() -> None:
    claimed = {claim for name in lesson_names() for claim in load_lesson(name).claims}

    assert checks() <= claimed, sorted(checks() - claimed)


@pytest.mark.parametrize("name", lesson_names())
def test_every_request_a_lesson_makes_is_one_the_api_takes(name) -> None:
    lesson = load_lesson(name)
    requests = [lesson.request] + [s.request for s in lesson.steps if s.request]

    for request in requests:
        device = build_from_spec(
            request["device"]["kind"], request["device"]["parameters"]
        )
        sweep = request["sweep"]
        check_request(
            sweep["kind"],
            device,
            sweep["contact"],
            sweep.get("settings"),
            sweep.get("models"),
            sweep.get("measure_at"),
        )


@pytest.mark.parametrize("path", sorted(LESSONS.glob("*.md")), ids=lambda p: p.name)
def test_no_lesson_uses_an_em_dash(path) -> None:
    assert "\u2014" not in path.read_text(encoding="utf-8")


# ------------------------------------------------------------------ routes


def test_the_lessons_are_listed(client) -> None:
    listed = client.get("/api/lessons").json()

    assert [entry["name"] for entry in listed] == list(lesson_names())
    assert all(entry["title"] and entry["summary"] for entry in listed)


def test_a_lesson_is_served_whole(client) -> None:
    name = lesson_names()[0]
    body = client.get(f"/api/lessons/{name}").json()
    lesson = load_lesson(name)

    assert body["request"] == lesson.request
    assert [step["title"] for step in body["steps"]] == [s.title for s in lesson.steps]
    assert body["claims"] == list(lesson.claims)
    assert body["explanation"] == lesson.explanation


def test_an_unknown_lesson_is_a_404(client) -> None:
    assert client.get("/api/lessons/nothing").status_code == 404
