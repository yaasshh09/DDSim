"""The explanations behind every part of the page, phases/PHASE-7.md part two.

A topic is a markdown file in two layers: a plain paragraph a first course in
semiconductors can follow, then the equations. Everything here keeps them
honest and complete: every knob, plot and status points at a topic that exists,
every reference into docs/ names a heading that exists, and no topic ships
half written.
"""

from __future__ import annotations

import pathlib
import re

import pytest
from fastapi.testclient import TestClient

from ddsim.api.app import create_app
from ddsim.api.devices import DEVICE_KINDS, device_parameters
from ddsim.api.learn import (
    KNOB_TOPICS,
    LEARN,
    PLOT_TOPICS,
    STATUS_TOPICS,
    load_topic,
    topic_names,
)
from ddsim.api.sweeps import SWEEP_KINDS, model_parameters, sweep_parameters

DOCS = pathlib.Path(__file__).parents[2] / "docs"


@pytest.fixture
def client():
    with TestClient(create_app()) as client:
        yield client


@pytest.mark.parametrize("name", topic_names())
def test_every_topic_has_both_layers(name) -> None:
    topic = load_topic(name)

    assert topic.title and topic.summary
    assert len(topic.plain.split()) >= 40, "the plain layer is a paragraph, not a line"
    assert topic.depth.strip()


@pytest.mark.parametrize("name", topic_names())
def test_every_docs_reference_names_a_heading_that_exists(name) -> None:
    for reference in load_topic(name).docs:
        file, _, heading = reference.partition("#")
        text = (DOCS / file).read_text(encoding="utf-8")
        headings = {
            line.lstrip("#").strip()
            for line in text.splitlines()
            if line.startswith("#")
        }
        assert heading in headings, f"{name}: docs/{file} has no heading {heading!r}"


@pytest.mark.parametrize("path", sorted(LEARN.glob("*.md")), ids=lambda p: p.name)
def test_no_topic_uses_an_em_dash(path) -> None:
    assert "—" not in path.read_text(encoding="utf-8")


def all_knob_names() -> set[str]:
    names = {p.name for kind in DEVICE_KINDS for p in device_parameters(kind)}
    names |= {p.name for kind in SWEEP_KINDS for p in sweep_parameters(kind)}
    names |= {p.name for p in model_parameters()}
    return names


def test_every_knob_points_at_a_topic() -> None:
    missing = all_knob_names() - set(KNOB_TOPICS)

    assert not missing, sorted(missing)


def test_every_mapping_points_at_a_topic_that_exists() -> None:
    known = set(topic_names())
    for mapping in (KNOB_TOPICS, PLOT_TOPICS, STATUS_TOPICS):
        missing = set(mapping.values()) - known
        assert not missing, sorted(missing)


def test_every_marked_element_in_the_page_has_a_topic() -> None:
    page = (LEARN.parent / "index.html").read_text(encoding="utf-8")
    ids = set(re.findall(r'data-topic-id="([^"]+)"', page))

    assert ids, "the page marks no explainable element"
    assert ids == set(PLOT_TOPICS), sorted(ids ^ set(PLOT_TOPICS))


def test_a_malformed_topic_is_refused(tmp_path, monkeypatch) -> None:
    import ddsim.api.learn as learn

    (tmp_path / "broken.md").write_text(
        "---\ntitle: x\nsummary: y\ndocs: z\n---\nno layers\n", encoding="utf-8"
    )
    monkeypatch.setattr(learn, "LEARN", tmp_path)

    with pytest.raises(ValueError, match="In plain words"):
        learn.load_topic("broken")


def test_the_topic_list_is_served(client) -> None:
    listed = client.get("/api/learn").json()

    assert {entry["name"] for entry in listed} == set(topic_names())


def test_one_topic_is_served_with_the_knobs_it_explains(client) -> None:
    body = client.get("/api/learn/potential").json()

    assert body["title"]
    assert "In plain words" not in body["plain"]
    assert body["knobs"] == sorted(
        k for k, t in KNOB_TOPICS.items() if t == "potential"
    )


@pytest.mark.parametrize("name", ["nothing", "..%2Fapp", "..%2Findex"])
def test_an_unknown_topic_is_a_404(client, name) -> None:
    assert client.get(f"/api/learn/{name}").status_code == 404
