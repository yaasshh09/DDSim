"""Every knob the page offers explains itself, phases/PHASE-7.md part two.

The explanation is the Args: line of the function the knob belongs to, so
there is one place a knob is described and it is next to the code that reads
it. A knob added without a docstring line never reaches a student unexplained,
because this file fails first.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ddsim.api.app import create_app
from ddsim.api.devices import DEVICE_KINDS, argument_docs, device_parameters
from ddsim.api.sweeps import SWEEP_KINDS, model_parameters, sweep_parameters


def every_knob():
    for kind in DEVICE_KINDS:
        for parameter in device_parameters(kind):
            yield f"device {kind}", parameter
    for kind in SWEEP_KINDS:
        for parameter in sweep_parameters(kind):
            yield f"sweep {kind}", parameter
    for parameter in model_parameters():
        yield "models", parameter


KNOBS = list(every_knob())
IDS = [f"{owner} {parameter.name}" for owner, parameter in KNOBS]


@pytest.fixture
def client():
    with TestClient(create_app()) as client:
        yield client


@pytest.mark.parametrize(("owner", "parameter"), KNOBS, ids=IDS)
def test_every_knob_has_an_explanation(owner, parameter) -> None:
    assert parameter.explanation.strip(), f"{owner} {parameter.name} has no Args: line"


@pytest.mark.parametrize(("owner", "parameter"), KNOBS, ids=IDS)
def test_every_numeric_knob_has_a_unit(owner, parameter) -> None:
    if parameter.type in ("float", "int"):
        assert parameter.unit, (
            f"{owner} {parameter.name}: no [unit] in {parameter.explanation!r}"
        )


def test_the_parser_reads_continuation_lines_and_the_unit() -> None:
    def sample(depth: float = 1.0, flag: bool = False) -> None:
        """Nothing.

        Args:
            depth: how far down [cm], measured from
                the top surface.
            flag: a switch.
        """

    assert argument_docs(sample) == {
        "depth": "how far down [cm], measured from the top surface.",
        "flag": "a switch.",
    }


def test_a_knob_crosses_the_schema_with_its_explanation(client) -> None:
    knobs = client.get("/api/schema").json()["devices"]["pn_diode"]
    na = next(knob for knob in knobs if knob["name"] == "Na")

    assert na["unit"] == "cm^-3"
    assert "acceptor" in na["explanation"]
