"""Tests for device/drawing.py, the 2D builder of phases/PHASE-7.md Stage 5.

A drawing is rectangles of silicon and oxide, rectangles of doping, and
electrodes along straight segments. The first thing it has to do is be the
benchmark devices when it is drawn as them: the doping to the last bit where
the arithmetic allows, and the solves to a recorded tolerance, which is
tests/analytic/test_drawn_devices.py. The rest is the guard rails, each with a
drawing that trips it and the reason it has to name.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ddsim.device.doping import Coordinates
from ddsim.device.drawing import (
    MOS_CAP_DRAWING,
    NMOS_DRAWING,
    NODE_BUDGET,
    Block,
    Electrode,
    drawing,
)
from ddsim.device.mosfet import nmos
from ddsim.device.regions import OXIDE, SILICON
from ddsim.discretize.boundary import GateContact, OhmicPlate

NM = 1e-7
"""One nanometre [cm]."""

MICRON = 1e-4
"""One micron [cm]."""

CAP_WIDTH = 1e-5
"""mos_cap's width [cm], 0.1 um."""

CAP_SURFACE = 2e-4
"""mos_cap's silicon thickness [cm], where its oxide starts."""


def drawn_nmos(**changes):
    blocks, implants, electrodes = NMOS_DRAWING
    return drawing(
        blocks=changes.pop("blocks", blocks),
        implants=changes.pop("implants", implants),
        electrodes=changes.pop("electrodes", electrodes),
        **changes,
    )


def drawn_cap(**changes):
    blocks, implants, electrodes = MOS_CAP_DRAWING
    return drawing(
        blocks=changes.pop("blocks", blocks),
        implants=changes.pop("implants", implants),
        electrodes=changes.pop("electrodes", electrodes),
        **changes,
    )


def cell_centres(mesh):
    x, y = mesh.x_axis.x, mesh.y_axis.x
    return 0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1])


# ------------------------------------------------------------ what is drawn


def test_the_default_drawing_is_the_benchmark_nmos() -> None:
    blocks, implants, electrodes = NMOS_DRAWING
    device = drawing()
    assert {e.name for e in electrodes} == {c.name for c in device.contacts}
    assert device.degenerate


def test_every_drawn_edge_is_a_mesh_line() -> None:
    blocks, implants, electrodes = NMOS_DRAWING
    mesh = drawn_nmos().mesh
    for thing in blocks + implants + electrodes:
        assert thing.x0 in mesh.x_axis.x and thing.x1 in mesh.x_axis.x
        assert thing.y0 in mesh.y_axis.x and thing.y1 in mesh.y_axis.x


def test_the_mesh_is_graded_at_the_mask_edges_and_the_surface() -> None:
    """Where nmos grades its own mesh: h_min at the two mask edges in x and
    at the silicon surface in y, to the rounding a pinned line costs."""
    device = drawn_nmos()
    x, y = device.mesh.x_axis, device.mesh.y_axis
    for edge in (0.4 * MICRON, 1.8 * MICRON - 0.4 * MICRON):
        node = int(np.flatnonzero(x.x == edge)[0])
        np.testing.assert_allclose(x.h[node - 1 : node + 1], 2 * NM, rtol=0.25)
    surface = int(np.flatnonzero(y.x == 1.0 * MICRON)[0])
    np.testing.assert_allclose(y.h[surface - 1 : surface + 1], 6.25e-9, rtol=0.25)


def test_the_materials_are_painted_on_the_cells() -> None:
    device = drawn_cap()
    cells = device.regions.cell_material
    surface = int(np.flatnonzero(device.mesh.y_axis.x == CAP_SURFACE)[0])
    assert np.all(cells[:surface] == SILICON)
    assert np.all(cells[surface:] == OXIDE)


def test_a_later_block_paints_over_an_earlier_one() -> None:
    """Order is the drawing order, so a trench is oxide drawn over silicon."""
    blocks, implants, electrodes = MOS_CAP_DRAWING
    trench = Block(
        "oxide",
        0.4 * CAP_WIDTH,
        0.6 * CAP_WIDTH,
        CAP_SURFACE - 0.5 * MICRON,
        CAP_SURFACE,
    )
    device = drawn_cap(blocks=blocks + (trench,), nx=41)
    centre_x, centre_y = cell_centres(device.mesh)
    inside = np.outer(
        (centre_y > trench.y0) & (centre_y < trench.y1),
        (centre_x > trench.x0) & (centre_x < trench.x1),
    )
    assert inside.any()
    assert np.all(device.regions.cell_material[inside] == OXIDE)
    below = ~inside & (centre_y < CAP_SURFACE)[:, None]
    assert np.all(device.regions.cell_material[below] == SILICON)


def test_the_drawn_nmos_doping_is_nmos_doping() -> None:
    """nmos's own profile, evaluated on the drawn mesh, at every node with
    silicon in it. The source half is bit for bit: the drawn implant is the
    same product of the same terms. The drain is equal to rounding, because
    nmos reflects its source and the drawing writes the drain out. Across
    the drain junction the net doping is the difference of two terms near
    1e17, so that rounding is judged against the body doping: measured, it
    is 976 cm^-3 at worst, 1e-14 of the terms."""
    device = drawn_nmos()
    mesh = device.mesh
    silicon = device.regions.semiconductor_volume > 0.0
    at = Coordinates(mesh.node_x, mesh.node_y)
    expected = nmos().doping(at)
    drawn = device.doping(at)
    np.testing.assert_allclose(
        drawn[silicon], expected[silicon], rtol=1e-13, atol=1e-13 * 1e17
    )
    source_half = silicon & (mesh.node_x < 0.9 * MICRON)
    np.testing.assert_array_equal(drawn[source_half], expected[source_half])


def test_electrodes_become_the_contacts_they_name() -> None:
    device = drawn_nmos()
    contacts = {c.name: c for c in device.contacts}
    assert isinstance(contacts["gate"], GateContact)
    assert isinstance(contacts["source"], OhmicPlate)
    mesh = device.mesh
    assert np.all(mesh.node_y[list(contacts["body"].nodes)] == 0.0)
    gate = next(e for e in NMOS_DRAWING[2] if e.name == "gate")
    assert np.all(mesh.node_y[list(contacts["gate"].nodes)] == gate.y0)
    gate_x = mesh.node_x[list(contacts["gate"].nodes)]
    assert gate_x.min() == gate.x0
    assert gate_x.max() == gate.x1


def test_an_electrode_carries_its_own_bias_and_work_function() -> None:
    blocks, implants, electrodes = NMOS_DRAWING
    changed = tuple(
        replace(e, voltage=0.7, work_function=4.5) if e.name == "gate" else e
        for e in electrodes
    )
    contacts = drawn_nmos(electrodes=changed).contacts
    gate = next(c for c in contacts if c.name == "gate")
    assert gate.voltage == 0.7
    assert gate.work_function == 4.5


def test_a_drawn_device_takes_a_bias_by_electrode_name() -> None:
    biased = drawn_nmos().with_bias(drain=0.05)
    assert next(c for c in biased.contacts if c.name == "drain").voltage == 0.05


# -------------------------------------------------------------- guard rails


def test_a_silicon_island_no_ohmic_contact_touches_is_refused() -> None:
    """A silicon block buried in a thick oxide, with nothing on it."""
    blocks, implants, electrodes = MOS_CAP_DRAWING
    top = CAP_SURFACE + 0.1 * MICRON
    thick = tuple(replace(b, y1=top) if b.material == "oxide" else b for b in blocks)
    island = Block(
        "silicon",
        0.3 * CAP_WIDTH,
        0.7 * CAP_WIDTH,
        CAP_SURFACE + 0.03 * MICRON,
        CAP_SURFACE + 0.06 * MICRON,
    )
    raised = tuple(
        replace(e, y0=top, y1=top) if e.name == "gate" else e for e in electrodes
    )
    with pytest.raises(ValueError, match="floats"):
        drawn_cap(blocks=thick + (island,), electrodes=raised, nx=41)


def test_a_gate_on_silicon_is_refused_as_a_schottky_contact() -> None:
    blocks, implants, electrodes = MOS_CAP_DRAWING
    on_silicon = Electrode("back", "gate", 0.2 * CAP_WIDTH, 0.8 * CAP_WIDTH, 0.0, 0.0)
    side = Electrode("body", "ohmic", 0.0, 0.0, 0.0, CAP_SURFACE)
    gate = next(e for e in electrodes if e.name == "gate")
    with pytest.raises(ValueError, match="Schottky"):
        drawn_cap(electrodes=(side, gate, on_silicon))


def test_an_ohmic_contact_on_oxide_is_refused() -> None:
    blocks, implants, electrodes = MOS_CAP_DRAWING
    on_oxide = tuple(
        replace(e, kind="ohmic") if e.name == "gate" else e for e in electrodes
    )
    with pytest.raises(ValueError, match="no silicon under it"):
        drawn_cap(electrodes=on_oxide)


def test_two_edges_closer_than_the_mesh_resolves_are_refused() -> None:
    """A contact edge a fraction of a nanometre from a mask edge."""
    blocks, implants, electrodes = NMOS_DRAWING
    near = tuple(
        replace(e, x1=0.4 * MICRON - 0.1 * NM) if e.name == "source" else e
        for e in electrodes
    )
    with pytest.raises(ValueError, match="closer than h_min_x"):
        drawn_nmos(electrodes=near)


def test_a_feature_thinner_than_the_mesh_resolves_is_refused() -> None:
    """An oxide 3 nm thick on a mesh that puts 2 nm cells at the interface:
    one cell, so no node inside it to carry the field across."""
    blocks, implants, electrodes = MOS_CAP_DRAWING
    top = CAP_SURFACE + 3 * NM
    thin = tuple(replace(b, y1=top) if b.material == "oxide" else b for b in blocks)
    raised = tuple(
        replace(e, y0=top, y1=top) if e.name == "gate" else e for e in electrodes
    )
    with pytest.raises(ValueError, match="smaller than the mesh resolves"):
        drawn_cap(blocks=thin, electrodes=raised, h_min_y=2 * NM)


def test_a_mesh_over_the_node_budget_is_refused() -> None:
    with pytest.raises(ValueError, match=f"budget of {NODE_BUDGET}"):
        drawn_nmos(nx=200, ny=200)


def test_a_gap_in_the_drawing_is_refused() -> None:
    """Nothing drawn is vacuum, and this solver has no material for it."""
    blocks, implants, electrodes = MOS_CAP_DRAWING
    short = tuple(
        replace(b, x1=0.5 * CAP_WIDTH) if b.material == "oxide" else b for b in blocks
    )
    with pytest.raises(ValueError, match="nothing is drawn"):
        drawn_cap(blocks=short)


def test_a_drawing_with_no_silicon_is_refused() -> None:
    oxide = (Block("oxide", 0.0, MICRON, 0.0, MICRON),)
    gate = (Electrode("gate", "gate", 0.0, MICRON, MICRON, MICRON),)
    with pytest.raises(ValueError, match="no silicon in it"):
        drawing(blocks=oxide, implants=(), electrodes=gate, nx=11, ny=11)


def test_a_doping_outside_the_range_is_refused() -> None:
    blocks, implants, electrodes = MOS_CAP_DRAWING
    heavy = (replace(implants[0], concentration=1e21),)
    with pytest.raises(ValueError, match="outside"):
        drawn_cap(implants=heavy)


def test_boltzmann_statistics_narrow_the_doping_range() -> None:
    """1e20 is inside the range with Fermi-Dirac on and outside it off, for
    the reason docs/01-physics.md gives."""
    with pytest.raises(ValueError, match="Fermi-Dirac"):
        drawn_nmos(degenerate=False)


def test_the_drawing_starts_at_the_origin() -> None:
    blocks, implants, electrodes = MOS_CAP_DRAWING
    moved = tuple(replace(b, x0=b.x0 + MICRON, x1=b.x1 + MICRON) for b in blocks)
    with pytest.raises(ValueError, match="origin"):
        drawn_cap(blocks=moved)


def test_something_drawn_outside_the_blocks_is_refused() -> None:
    blocks, implants, electrodes = MOS_CAP_DRAWING
    beyond = (replace(implants[0], x1=5 * MICRON),)
    with pytest.raises(ValueError, match="outside the drawing"):
        drawn_cap(implants=beyond)


def test_electrodes_that_share_a_node_are_refused() -> None:
    blocks, implants, electrodes = MOS_CAP_DRAWING
    second = Electrode("body2", "ohmic", 0.0, 0.5 * CAP_WIDTH, 0.0, 0.0)
    with pytest.raises(ValueError, match="share"):
        drawn_cap(electrodes=electrodes + (second,))


def test_an_electrode_is_a_straight_line() -> None:
    blocks, implants, electrodes = MOS_CAP_DRAWING
    slanted = (
        Electrode("body", "ohmic", 0.0, 0.5 * CAP_WIDTH, 0.0, 0.1 * MICRON),
        next(e for e in electrodes if e.name == "gate"),
    )
    with pytest.raises(ValueError, match="straight line"):
        drawn_cap(electrodes=slanted)


@pytest.mark.parametrize(
    ("record", "match"),
    [
        (Block("glass", 0.0, MICRON, 0.0, MICRON), "silicon or oxide"),
        (Block("silicon", MICRON, 0.0, 0.0, MICRON), "positive"),
    ],
)
def test_a_block_names_what_is_wrong_with_it(record, match) -> None:
    with pytest.raises(ValueError, match=match):
        drawing(blocks=(record,), implants=(), electrodes=(), nx=11, ny=11)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"dopant": "x"}, "'n' or 'p'"),
        ({"profile": "linear"}, "uniform or gaussian"),
        ({"profile": "gaussian"}, "straggle and lateral"),
    ],
)
def test_an_implant_names_what_is_wrong_with_it(changes, match) -> None:
    blocks, implants, electrodes = MOS_CAP_DRAWING
    with pytest.raises(ValueError, match=match):
        drawn_cap(implants=(replace(implants[0], **changes),))


def test_an_electrode_is_ohmic_or_a_gate() -> None:
    blocks, implants, electrodes = MOS_CAP_DRAWING
    wrong = tuple(
        replace(e, kind="schottky") if e.name == "body" else e for e in electrodes
    )
    with pytest.raises(ValueError, match="ohmic or gate"):
        drawn_cap(electrodes=wrong)
