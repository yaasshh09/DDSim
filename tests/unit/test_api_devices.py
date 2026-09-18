"""Tests for api/devices.py, the device registry the browser builds from.

phases/PHASE-7.md wants the browser to define a device with the same config
the CLI takes and nothing hardcoded per device. The way this package keeps
that promise is by reading the device constructors' own signatures, so a
parameter added to nmos() shows up in the browser without anyone editing the
API. These tests are the contract that says so, and they are deliberately
written against the signatures rather than against a copy of them.

No HTTP here. This is the layer underneath it.
"""

from __future__ import annotations

import pytest

from ddsim.api.devices import (
    COARSE,
    DEVICE_KINDS,
    build_from_spec,
    device_dimension,
    device_parameters,
    node_count,
    parameters_of,
    region_defaults,
)
from ddsim.api.sweeps import run_sweep


def test_the_device_classes_are_offered() -> None:
    """The three phases/PHASE-7.md names in its acceptance criteria, and the
    1D stack of its Stage 4."""
    assert set(DEVICE_KINDS) == {"pn_diode", "mos_cap", "nmos", "stack"}


def test_the_parameters_come_from_the_constructor_signature() -> None:
    """Read, not copied. A schema written out by hand here would go stale the
    first time a device grows a knob, and the browser would stop offering it."""
    names = {parameter.name for parameter in device_parameters("pn_diode")}

    assert "Na" in names
    assert "junction" in names
    assert "anode_voltage" in names


def test_a_parameter_carries_its_default_and_its_type() -> None:
    """The browser renders a form from this and nothing else."""
    by_name = {p.name: p for p in device_parameters("pn_diode")}

    assert by_name["Na"].default == 1e16
    assert by_name["Na"].type == "float"
    assert by_name["n_nodes"].default == 201
    assert by_name["n_nodes"].type == "int"


def test_a_boolean_parameter_is_offered_as_a_boolean() -> None:
    """nmos has one. Rendering it as a number would put 0 and 1 in a form."""
    by_name = {p.name: p for p in device_parameters("nmos")}

    assert by_name["degenerate"].type == "bool"
    assert by_name["degenerate"].default is True


def test_the_material_argument_is_not_offered() -> None:
    """It is an object, not a number, so there is no honest form field for it.
    Offering it would mean accepting one over the wire, and a device built on
    a material the browser invented is not a device this project validated."""
    for kind in DEVICE_KINDS:
        assert "material" not in {p.name for p in device_parameters(kind)}


def test_an_unknown_kind_names_the_ones_that_exist() -> None:
    with pytest.raises(ValueError, match="pn_diode"):
        device_parameters("transistor")


def test_building_with_no_parameters_gives_the_constructor_default() -> None:
    """The defaults in the signature are the ones this project validated, so
    the API must not carry a second set of its own."""
    device = build_from_spec("pn_diode", {})

    assert device.mesh.n_nodes == 201


def test_a_parameter_reaches_the_constructor() -> None:
    """The test that says the spec is wired to the builder at all."""
    device = build_from_spec("pn_diode", {"n_nodes": 51})

    assert device.mesh.n_nodes == 51


def test_an_unknown_parameter_is_refused_rather_than_ignored() -> None:
    """A typo that is silently dropped gives the user the default device and
    a plot they will believe. Refusing is the only safe reading."""
    with pytest.raises(ValueError, match="n_node"):
        build_from_spec("pn_diode", {"n_node": 51})


def test_a_parameter_of_the_wrong_type_is_refused() -> None:
    with pytest.raises(TypeError, match="Na"):
        build_from_spec("pn_diode", {"Na": "1e16"})


def test_a_whole_number_is_accepted_where_a_float_is_wanted() -> None:
    """JSON has one number type, and 1 arrives as an int. Refusing it would
    make a form field reject a perfectly good entry."""
    device = build_from_spec("pn_diode", {"anode_voltage": 1})

    assert device.contacts[0].voltage == pytest.approx(1.0)


def test_a_boolean_is_refused_where_an_integer_is_wanted() -> None:
    """bool is a subclass of int in Python, so an isinstance check alone lets
    True through as a node count. It is a mesh of one node, and the failure
    surfaces decades away from the field that caused it."""
    with pytest.raises(TypeError, match="n_nodes"):
        build_from_spec("pn_diode", {"n_nodes": True})


def test_an_integer_is_refused_where_a_boolean_is_wanted() -> None:
    """The same confusion from the other side."""
    with pytest.raises(TypeError, match="degenerate"):
        build_from_spec("nmos", {"degenerate": 1})


def test_the_material_argument_cannot_be_passed_either() -> None:
    """Not offering it is not the same as refusing it."""
    with pytest.raises(ValueError, match="material"):
        build_from_spec("pn_diode", {"material": "silicon"})


def test_a_boolean_reaches_the_constructor() -> None:
    """The refusals above say what a bad value does. This says the good one
    is not refused along with it."""
    device = build_from_spec("nmos", {"degenerate": False})

    assert device.degeneracy is None


def test_a_float_reaches_the_constructor_unchanged() -> None:
    """Doping is the parameter most worth checking arrives intact, since
    every result in the project scales with it."""
    device = build_from_spec("pn_diode", {"Na": 2.5e16})

    assert device.net_doping.data[0] == pytest.approx(-2.5e16)


def test_an_argument_with_no_default_is_not_a_knob() -> None:
    """A knob is rendered with its default beside it, and an argument with no
    default has none: offering it would put the text <class 'inspect._empty'>
    into a form field.

    Tested on a function written here rather than on a sweep, because today
    the only case is iv_sweep's contact and api/sweeps.py excludes the
    terminal names for its own reason. Two rules cover the same argument by
    coincidence, and a test that leaned on that would be checking neither.
    """

    def example(required: float, optional: float = 1.0, named: str = "x") -> None:
        """A stand in for any function the API might offer knobs from."""

    assert [p.name for p in parameters_of(example)] == ["optional", "named"]


def test_a_coarse_preset_exists_for_every_device_not_solved_live() -> None:
    """The 2D ones. phases/PHASE-7.md keeps the solve button on those because
    a 2D solve is seconds to minutes, and a coarse mesh is what makes the wait
    bearable while a student is still finding their way around.

    The diode has none on purpose. Measured 2026-09-17, its converged mesh
    solves a seven point I-V in 0.2 s, so there is nothing to save and a
    second mesh to explain instead.
    """
    live = {kind for kind in DEVICE_KINDS if device_dimension(kind) == 1}

    assert set(COARSE) == set(DEVICE_KINDS) - live


@pytest.mark.parametrize("kind", sorted(COARSE))
def test_a_coarse_preset_only_names_knobs_its_device_has(kind) -> None:
    """A preset is knob settings, not a second constructor. A name that has
    gone from the device would be a refusal on the first click."""
    known = {parameter.name for parameter in device_parameters(kind)}

    assert set(COARSE[kind].parameters) <= known


@pytest.mark.parametrize("kind", sorted(COARSE))
def test_a_coarse_preset_builds_and_is_coarser(kind) -> None:
    """Coarser is the whole claim, so it is measured on the mesh the preset
    actually builds rather than read off the node knobs, which are one axis
    each on a device whose mesh is a tensor product of several."""
    coarse = build_from_spec(kind, dict(COARSE[kind].parameters))
    converged = build_from_spec(kind, {})

    assert node_count(coarse) < node_count(converged)


@pytest.mark.parametrize("kind", sorted(COARSE))
def test_a_coarse_preset_says_what_changes(kind) -> None:
    """phases/PHASE-7.md: marked as coarse, with a note on what changes. A
    preset that only said 'faster' would be asking a student to trust a number
    nobody measured."""
    note = COARSE[kind].note

    assert "percent" in note, f"{kind}: {note!r} names no measured difference"
    assert str(node_count(build_from_spec(kind, {}))) in note, (
        f"{kind}: {note!r} does not say what it is coarse against"
    )


# ------------------------------------------------------------- the 1D stack

PIN = [
    {"dopant": "p", "length": 2e-5, "concentration": 1e18},
    {"dopant": "n", "length": 1e-4, "concentration": 1e14},
    {"dopant": "n", "length": 2e-5, "concentration": 1e18},
]
"""A pin diode as the page sends one, and as a saved device file holds it."""


def test_a_stack_is_built_from_the_regions_the_page_sends() -> None:
    device = build_from_spec("stack", {"regions": PIN, "n_nodes": 301})
    assert device.mesh.n_nodes == 301
    assert device.mesh.x[-1] == pytest.approx(1.4e-4, rel=1e-14)


def test_the_default_regions_are_offered_as_the_page_sends_them() -> None:
    """The form is filled from these, and building them back is the default
    stack, so there is no second copy of the default anywhere."""
    regions = region_defaults("stack")
    assert regions == [
        {"dopant": "p", "length": 5e-5, "concentration": 1e16},
        {"dopant": "n", "length": 5e-5, "concentration": 1e16},
    ]
    assert (
        build_from_spec("stack", {"regions": regions}).mesh.x
        == build_from_spec("stack", {}).mesh.x
    ).all()


def test_a_device_without_regions_offers_none() -> None:
    assert region_defaults("pn_diode") is None


def test_regions_are_not_a_knob_on_the_form() -> None:
    """A list of regions is not one number, so it is not rendered as a box.
    The page draws it as rows, from region_defaults."""
    assert "regions" not in {p.name for p in device_parameters("stack")}


def test_the_stack_is_solved_live() -> None:
    assert device_dimension("stack") == 1


@pytest.mark.parametrize(
    ("regions", "complaint"),
    [
        ("pn", "list of regions"),
        (["p"], "region 1 is"),
        ([{"dopant": "p", "length": 1e-4}], "region 1.*concentration"),
        (
            [{"dopant": "p", "length": 1e-4, "concentration": 1e16, "x": 1}],
            "region 1.*'x'",
        ),
        ([{"dopant": "p", "length": "1e-4", "concentration": 1e16}], "length"),
        ([{"dopant": "p", "length": 1e-4, "concentration": True}], "concentration"),
        ([{"dopant": 1, "length": 1e-4, "concentration": 1e16}], "dopant"),
    ],
    ids=["not a list", "not an object", "missing", "extra", "string", "bool", "number"],
)
def test_a_malformed_region_is_refused_naming_it(regions, complaint) -> None:
    """A saved file is a student's own, edited by hand as often as not. What
    is wrong with it is said by region number and field name."""
    with pytest.raises((TypeError, ValueError), match=complaint):
        build_from_spec("stack", {"regions": regions})


def test_regions_are_refused_on_a_device_that_has_none() -> None:
    with pytest.raises(ValueError, match="regions"):
        build_from_spec("pn_diode", {"regions": PIN})


def test_the_phase_2_diode_drawn_as_a_stack_has_the_same_i_v() -> None:
    """phases/PHASE-7.md part two: a 1D stack drawn as the Phase 2 diode
    reproduces pn_diode's I-V. Bit for bit, through the same request path the
    page uses, so no tolerance needs recording."""
    voltages = [0.0, 0.2, 0.4, -0.5]
    diode, _ = run_sweep("iv", build_from_spec("pn_diode", {}), "anode", voltages)
    drawn, _ = run_sweep(
        "iv",
        build_from_spec("stack", {"regions": region_defaults("stack")}),
        "left",
        voltages,
    )
    assert diode.complete and drawn.complete
    assert [p.current for p in drawn.points] == [p.current for p in diode.points]
