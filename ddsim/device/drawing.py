"""Devices drawn from rectangles, the 2D builder of phases/PHASE-7.md Stage 5.

Three kinds of thing are drawn, each an axis aligned rectangle or segment in
cm, with x across the device and y up it, the gate side on top:

- blocks of silicon or oxide, painted in order, a later block over an earlier
  one, so a trench is oxide drawn over silicon. The device is their bounding
  box, which starts at the origin, and every part of it has to be painted.
- implants, rectangles of doping, uniform or Gaussian, added up with the sign
  of their dopant. Doping drawn over oxide changes nothing: an oxide node has
  no charge volume to put it in.
- electrodes, straight segments: ohmic on silicon, or a gate on oxide with a
  work function.

Nothing new is solved. The mesh is a tensor product of two graded axes, the
region map is device/regions.py's, the contacts are the plates and gates the
benchmark devices already use. Drawn as the benchmark MOS capacitor or MOSFET
it is that device's doping on a different mesh, see NMOS_DRAWING.

Rectangles only
---------------
The 2D mesh is a tensor product, so every mesh line runs the full width or
height of the device. A slanted or curved edge would cut cells in two, and a
cell that is half oxide has no single permittivity. So every edge drawn is a
mesh line, and nothing else can be drawn.

The mesh
--------
Each axis is graded_mesh_1d_through: a node line on every edge drawn, graded
to h_min at every doping edge and every Si/SiO2 interface. The interfaces are
where an inversion layer sits, which is why nmos and mos_cap grade their
silicon to the surface. An implant edge on the device boundary is not graded
towards, since the implant does not stop there; it continues past the drawing
and its profile is carried as open on that side.

Guard rails
-----------
Each refusal names what it is about. A silicon island no ohmic contact
touches floats, since nothing sets its Fermi level. A gate on silicon is a
Schottky contact, which this solver does not model. The island rule holds for
each doping type too: a p or n region no ohmic contact touches is a floating
body, which the solve cannot pin, so an SOI film needs a body tie. An ohmic
contact on oxide has no carriers to pin. Two edges closer than h_min, or a
rectangle with fewer than two node lines inside it, is a feature smaller than
the mesh resolves. A mesh over NODE_BUDGET is refused before it is built.

Surface mobility reads the field normal to a horizontal interface, which a
drawing with a vertical Si/SiO2 wall also has. That refusal belongs to the
model and lives in device/transport.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from ddsim.core import constants as C
from ddsim.device.builder import Device, Material, build_device
from ddsim.device.doping import Along, Coordinates, DopingProfile, Sum, Window
from ddsim.device.mosfet import implant_lengths
from ddsim.device.regions import OXIDE, SILICON, region_map
from ddsim.device.stack import DOPING_RANGE, NODES_INSIDE
from ddsim.discretize.boundary import Contact, GateContact, OhmicPlate
from ddsim.mesh.mesh1d import graded_mesh_1d_through
from ddsim.mesh.mesh2d import Mesh2D, tensor_mesh_2d

MATERIALS = {"silicon": SILICON, "oxide": OXIDE}
"""The materials a block can be, by the name the page sends."""

DEGENERATE_DOPING_TOP = 1e20
"""The heaviest doping allowed with Fermi-Dirac statistics on [cm^-3].

The nmos source and drain peak, where n/Nc is 3.5, inside the range where
the Joyce-Dixon inversion docs/01-physics.md uses holds."""

NODE_BUDGET = 20000
"""The most nodes a drawn mesh may have [1].

The benchmark nmos is 8379 nodes and its transfer curve takes about 20 s, so
this is a minute or so for a curve, not an hour. The page states it."""


@dataclass(frozen=True)
class Block:
    """A rectangle of one material."""

    material: str
    """"silicon" or "oxide"."""

    x0: float
    """Left edge [cm]."""

    x1: float
    """Right edge [cm]."""

    y0: float
    """Bottom edge [cm]."""

    y1: float
    """Top edge [cm]."""


@dataclass(frozen=True)
class Implant:
    """A rectangle of doping, added to whatever else is drawn there."""

    dopant: str
    """"n" for donors, "p" for acceptors."""

    concentration: float
    """Peak concentration [cm^-3], given positive."""

    x0: float
    """Left edge [cm]."""

    x1: float
    """Right edge [cm]."""

    y0: float
    """Bottom edge [cm]."""

    y1: float
    """Top edge [cm]."""

    profile: str = "uniform"
    """"uniform", the peak inside and nothing outside, or "gaussian", the shape
    of an implant: the peak holds inside the rectangle, falls below and above
    it as a Gaussian of `straggle`, and spreads past its left and right edges
    as an erfc of `lateral`, half the peak at the mask edge."""

    straggle: float = 0.0
    """The vertical sigma of a Gaussian implant [cm]."""

    lateral: float = 0.0
    """The erfc length of a Gaussian implant's sides [cm]."""


@dataclass(frozen=True)
class Electrode:
    """A contact along a straight segment."""

    name: str
    """Terminal name, which a sweep and a bias refer to it by."""

    kind: str
    """"ohmic", on silicon, or "gate", on oxide."""

    x0: float
    """Start of the segment along x [cm]."""

    x1: float
    """End of the segment along x [cm], x0 for a vertical one."""

    y0: float
    """Start of the segment along y [cm]."""

    y1: float
    """End of the segment along y [cm], y0 for a horizontal one."""

    voltage: float = 0.0
    """Applied bias [V]."""

    work_function: float = C.PHI_M_N_POLY
    """Work function of a gate [eV]. n+ poly by default. Unused on ohmic."""


Drawing = tuple[tuple[Block, ...], tuple[Implant, ...], tuple[Electrode, ...]]


def _nmos_drawing() -> Drawing:
    """nmos's defaults, drawn from rectangles.

    The oxide spans the top, the gate sits on it over the channel, and the
    source and drain implants are drawn as their mask openings: rectangles
    from the silicon surface up through the oxide, open on the outer side
    because they are flush with the device boundary. Their profile is then
    nmos's own product, an erfc across the mask edge times a Gaussian below
    the surface, with nmos's own sigma and erfc length.
    """
    L_gate, sd, contact, Na, peak = 1e-4, 4e-5, 2e-5, 1e17, 1e20
    t_ox, t_si = 2e-6, 1e-4
    width, top = 2.0 * sd + L_gate, t_si + t_ox
    sigma, edge = implant_lengths(1.5e-5, 1e-5, peak, Na)
    return (
        (
            Block("silicon", 0.0, width, 0.0, t_si),
            Block("oxide", 0.0, width, t_si, top),
        ),
        (
            Implant("p", Na, 0.0, width, 0.0, t_si),
            Implant("n", peak, 0.0, sd, t_si, top, "gaussian", sigma, edge),
            Implant("n", peak, width - sd, width, t_si, top, "gaussian", sigma, edge),
        ),
        (
            Electrode("source", "ohmic", 0.0, contact, t_si, t_si),
            Electrode("drain", "ohmic", width - contact, width, t_si, t_si),
            Electrode("gate", "gate", sd, width - sd, top, top),
            Electrode("body", "ohmic", 0.0, width, 0.0, 0.0),
        ),
    )


def _mos_cap_drawing() -> Drawing:
    """mos_cap's defaults, drawn from rectangles."""
    Na, t_ox, t_si, width = 1e16, 1e-6, 2e-4, 1e-5
    top = t_si + t_ox
    return (
        (
            Block("silicon", 0.0, width, 0.0, t_si),
            Block("oxide", 0.0, width, t_si, top),
        ),
        (Implant("p", Na, 0.0, width, 0.0, t_si),),
        (
            Electrode("body", "ohmic", 0.0, width, 0.0, 0.0),
            Electrode("gate", "gate", 0.0, width, top, top),
        ),
    )


NMOS_DRAWING = _nmos_drawing()
"""The benchmark nmos, drawn. The default drawing."""

MOS_CAP_DRAWING = _mos_cap_drawing()
"""The benchmark MOS capacitor, drawn."""


def _box(what: str, x0: float, x1: float, y0: float, y1: float) -> str:
    return f"{what} ({x0:g} to {x1:g} cm across, {y0:g} to {y1:g} cm up)"


def _check_records(
    blocks: tuple[Block, ...],
    implants: tuple[Implant, ...],
    electrodes: tuple[Electrode, ...],
    degenerate: bool,
) -> None:
    """Refuse anything drawn that is not one of the things that can be."""
    for number, block in enumerate(blocks, start=1):
        if block.material not in MATERIALS:
            raise ValueError(
                f"block {number}: a block is silicon or oxide, got {block.material!r}"
            )
        if not (block.x0 < block.x1 and block.y0 < block.y1):
            raise ValueError(
                f"block {number}: a block needs a positive width and height, "
                "x0 < x1 and y0 < y1, got "
                + _box("it", block.x0, block.x1, block.y0, block.y1)
            )

    low = DOPING_RANGE[0]
    high = DEGENERATE_DOPING_TOP if degenerate else DOPING_RANGE[1]
    for number, implant in enumerate(implants, start=1):
        if implant.dopant not in ("n", "p"):
            raise ValueError(
                f"implant {number}: the dopant is 'n' or 'p', got {implant.dopant!r}"
            )
        if implant.profile not in ("uniform", "gaussian"):
            raise ValueError(
                f"implant {number}: a profile is uniform or gaussian, got "
                f"{implant.profile!r}"
            )
        if implant.profile == "gaussian" and not (
            implant.straggle > 0.0 and implant.lateral > 0.0
        ):
            raise ValueError(
                f"implant {number}: a gaussian implant needs a positive straggle "
                f"and lateral, got straggle={implant.straggle:g} and "
                f"lateral={implant.lateral:g} cm"
            )
        if not (implant.x0 < implant.x1 and implant.y0 < implant.y1):
            raise ValueError(
                f"implant {number}: an implant needs a positive width and "
                "height, x0 < x1 and y0 < y1"
            )
        if not low <= implant.concentration <= high:
            why = (
                ""
                if degenerate or implant.concentration < low
                else f" Above {high:g}, Boltzmann statistics put the Fermi level "
                "in the wrong place. Turn on degenerate (Fermi-Dirac "
                "statistics), the way nmos does."
            )
            raise ValueError(
                f"implant {number}: a concentration of {implant.concentration:g} "
                f"cm^-3 is outside {low:g} to {high:g} cm^-3, the range the "
                f"models here are built for (see docs/01-physics.md).{why}"
            )

    for electrode in electrodes:
        if electrode.kind not in ("ohmic", "gate"):
            raise ValueError(
                f"electrode {electrode.name!r}: an electrode is ohmic or gate, got "
                f"{electrode.kind!r}. A metal on silicon that is not ohmic is a "
                "Schottky contact, which this solver does not model."
            )
        across = electrode.x1 - electrode.x0
        up = electrode.y1 - electrode.y0
        if not ((across > 0.0 and up == 0.0) or (up > 0.0 and across == 0.0)):
            raise ValueError(
                f"electrode {electrode.name!r}: an electrode is a straight line "
                "along x or along y, with x0 < x1 and y0 == y1 or the other way "
                "round, got "
                + _box("it", electrode.x0, electrode.x1, electrode.y0, electrode.y1)
            )


def _lines(
    things: list[tuple[str, float, float]], h_min: float, axis: str
) -> list[float]:
    """Every edge along one axis, refusing two closer than h_min.

    Args:
        things: a description and the two edges of everything drawn.
        h_min: the finest spacing the mesh is asked for on this axis [cm].
        axis: "x" or "y", for the refusal.
    """
    owners: dict[float, list[str]] = {}
    for what, low, high in things:
        owners.setdefault(low, []).append(what)
        owners.setdefault(high, []).append(what)
    lines = sorted(owners)
    for a, b in zip(lines[:-1], lines[1:], strict=True):
        if b - a < h_min:
            raise ValueError(
                f"{', '.join(owners[a])} and {', '.join(owners[b])} put mesh "
                f"lines {b - a:g} cm apart, at {axis} = {a:g} and {b:g} cm, "
                f"closer than h_min_{axis}={h_min:g} cm. That is a feature "
                "smaller than the mesh resolves. Line the edges up, or move them "
                "at least h_min apart."
            )
    return lines


def _paint(
    blocks: tuple[Block, ...],
    x_edges: npt.NDArray[np.float64],
    y_edges: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    """The material of every cell between the given lines, -1 where nothing is.

    Read at the cell centre, which is inside exactly the blocks the cell is,
    because every block edge is one of the lines.
    """
    centre_x = 0.5 * (x_edges[1:] + x_edges[:-1])
    centre_y = 0.5 * (y_edges[1:] + y_edges[:-1])
    cells = np.full((centre_y.size, centre_x.size), -1, dtype=np.int64)
    for block in blocks:
        inside = np.outer(
            (centre_y > block.y0) & (centre_y < block.y1),
            (centre_x > block.x0) & (centre_x < block.x1),
        )
        cells[inside] = MATERIALS[block.material]
    return cells


def _interfaces(
    cells: npt.NDArray[np.int64],
    x_edges: npt.NDArray[np.float64],
    y_edges: npt.NDArray[np.float64],
) -> tuple[set[float], set[float]]:
    """Where silicon meets oxide: the x of every vertical wall, the y of every
    horizontal one."""
    across = cells[:, 1:] != cells[:, :-1]
    up = cells[1:, :] != cells[:-1, :]
    walls = {float(x_edges[i + 1]) for i in np.flatnonzero(across.any(axis=0))}
    floors = {float(y_edges[j + 1]) for j in np.flatnonzero(up.any(axis=1))}
    return walls, floors


def _implant_profile(implant: Implant, width: float, height: float) -> DopingProfile:
    """One implant as a doping profile, open on any side flush with the boundary."""
    left = -math.inf if implant.x0 <= 0.0 else implant.x0
    right = math.inf if implant.x1 >= width else implant.x1
    bottom = -math.inf if implant.y0 <= 0.0 else implant.y0
    top = math.inf if implant.y1 >= height else implant.y1
    if implant.profile == "uniform":
        across = Window(left, right)
        up = Window(bottom, top)
    else:
        across = Window(left, right, "erfc", implant.lateral)
        up = Window(bottom, top, "gaussian", implant.straggle)
    sign = 1.0 if implant.dopant == "n" else -1.0
    return Along(across, "x") * Along(up, "y") * (sign * implant.concentration)


def _electrode_nodes(mesh: Mesh2D, electrode: Electrode) -> tuple[int, ...]:
    """The mesh nodes on an electrode's segment, which is on mesh lines."""
    on = (
        (mesh.node_x >= electrode.x0)
        & (mesh.node_x <= electrode.x1)
        & (mesh.node_y >= electrode.y0)
        & (mesh.node_y <= electrode.y1)
    )
    return tuple(int(node) for node in np.flatnonzero(on))


def drawing(
    blocks: tuple[Block, ...] = NMOS_DRAWING[0],
    implants: tuple[Implant, ...] = NMOS_DRAWING[1],
    electrodes: tuple[Electrode, ...] = NMOS_DRAWING[2],
    nx: int = 63,
    ny: int = 133,
    h_min_x: float = 2e-7,
    h_min_y: float = 6.25e-9,
    degenerate: bool = True,
    material: Material | None = None,
) -> Device:
    """A 2D device drawn from rectangles of material and doping.

    Args:
        blocks: rectangles of silicon and oxide, painted in order. Together
            they cover a rectangle whose lower left corner is the origin.
        implants: rectangles of doping, added up.
        electrodes: ohmic contacts on silicon and gates on oxide, each along
            a straight segment.
        nx: mesh points across the device [1]. Every edge you draw needs one,
            so a busy drawing needs more. Range 38 to 200.
        ny: mesh points up the device [1]. Range 22 to 400.
        h_min_x: the smallest column spacing, at every doping edge and every
            vertical silicon to oxide wall [cm]. Two edges closer than this
            get turned down. Range 4.1e-8 to 4.7e-6, log.
        h_min_y: the smallest row spacing, at every doping edge and every
            flat silicon to oxide interface [cm]. On a MOSFET the inversion
            layer is only a few nanometres thick, and this is what has to
            catch it. Range 1e-9 to 7.7e-7, log.
        degenerate: use Fermi-Dirac statistics instead of the simpler
            Boltzmann ones. On by default, because the starting drawing is
            the benchmark nmos, doped to 1e20 cm^-3 in its source and drain.
            With it off, any doping above 1e19 gets turned down.
        material: defaults to silicon at 300 K.

    The mesh ranges are the default drawing's, the benchmark nmos. Each end
    is the last value that builds on the converged or the coarse mesh, and
    each was solved over the 0 V to 1.5 V transfer with the full mobility
    stack, 2026-09-23. Every sweep completed. A drawing of your own can
    refuse sooner, and the page stops the knob where it does.
    """
    _check_records(blocks, implants, electrodes, degenerate)
    if not blocks:
        raise ValueError("nothing is drawn: a device needs at least one block")

    origin_x = min(block.x0 for block in blocks)
    origin_y = min(block.y0 for block in blocks)
    if origin_x != 0.0 or origin_y != 0.0:
        raise ValueError(
            f"the drawing starts at x={origin_x:g}, y={origin_y:g} cm; draw it "
            "from the origin, x = 0 and y = 0 at its lower left corner"
        )
    width = max(block.x1 for block in blocks)
    height = max(block.y1 for block in blocks)

    drawn: list[tuple[str, Block | Implant | Electrode]] = []
    drawn += [(f"block {n}", b) for n, b in enumerate(blocks, start=1)]
    drawn += [(f"implant {n}", i) for n, i in enumerate(implants, start=1)]
    drawn += [(f"electrode {e.name!r}", e) for e in electrodes]
    for what, thing in drawn:
        inside = (
            0.0 <= thing.x0
            and thing.x1 <= width
            and 0.0 <= thing.y0
            and thing.y1 <= height
        )
        if not inside:
            raise ValueError(
                f"{_box(what, thing.x0, thing.x1, thing.y0, thing.y1)} reaches "
                f"outside the drawing, which is the blocks' extent: 0 to "
                f"{width:g} cm across and 0 to {height:g} cm up"
            )

    if nx * ny > NODE_BUDGET:
        raise ValueError(
            f"a mesh of {nx} by {ny} is {nx * ny} nodes, over the budget of "
            f"{NODE_BUDGET}. A 2D solve slows faster than its node count grows; "
            "the benchmark nmos is 8379 nodes and about 20 s for a transfer curve."
        )

    x_lines = _lines([(w, t.x0, t.x1) for w, t in drawn], h_min_x, "x")
    y_lines = _lines([(w, t.y0, t.y1) for w, t in drawn], h_min_y, "y")

    coarse = _paint(blocks, np.array(x_lines), np.array(y_lines))
    if (coarse < 0).any():
        j, i = (int(k[0]) for k in np.nonzero(coarse < 0))
        raise ValueError(
            f"nothing is drawn between x = {x_lines[i]:g} and {x_lines[i + 1]:g} "
            f"cm, y = {y_lines[j]:g} and {y_lines[j + 1]:g} cm. An undrawn gap "
            "is vacuum, which this solver has no material for. Cover it with a "
            "block."
        )
    if not (coarse == SILICON).any():
        raise ValueError("the drawing has no silicon in it, so nothing to simulate")

    walls, floors = _interfaces(coarse, np.array(x_lines), np.array(y_lines))
    x_points = walls | {i.x0 for i in implants if i.x0 > 0.0}
    x_points |= {i.x1 for i in implants if i.x1 < width}
    y_points = floors | {i.y0 for i in implants if i.y0 > 0.0}
    y_points |= {i.y1 for i in implants if i.y1 < height}
    mesh = tensor_mesh_2d(
        graded_mesh_1d_through(
            width, nx, tuple(x_lines), tuple(sorted(x_points)), h_min_x
        ),
        graded_mesh_1d_through(
            height, ny, tuple(y_lines), tuple(sorted(y_points)), h_min_y
        ),
    )

    columns, rows = mesh.x_axis.x, mesh.y_axis.x
    for what, thing in drawn[: len(blocks) + len(implants)]:
        inside_x = int(np.count_nonzero((columns > thing.x0) & (columns < thing.x1)))
        inside_y = int(np.count_nonzero((rows > thing.y0) & (rows < thing.y1)))
        if thing.x0 == 0.0 and thing.x1 == width:
            inside_x = NODES_INSIDE
        if thing.y0 == 0.0 and thing.y1 == height:
            inside_y = NODES_INSIDE
        if min(inside_x, inside_y) < NODES_INSIDE:
            raise ValueError(
                f"{_box(what, thing.x0, thing.x1, thing.y0, thing.y1)} is smaller "
                f"than the mesh resolves: it holds {inside_x} node columns and "
                f"{inside_y} node rows inside it and needs at least "
                f"{NODES_INSIDE} of each. Make it bigger, or refine the mesh "
                "with more nodes or a smaller h_min."
            )

    cells = _paint(blocks, columns, rows)
    regions = region_map(mesh, cells)
    silicon = regions.semiconductor_volume > 0.0

    contacts: list[Contact] = []
    taken: dict[int, str] = {}
    for electrode in electrodes:
        nodes = _electrode_nodes(mesh, electrode)
        where = _box(
            f"electrode {electrode.name!r}",
            electrode.x0,
            electrode.x1,
            electrode.y0,
            electrode.y1,
        )
        if electrode.kind == "ohmic":
            if not silicon[list(nodes)].all():
                raise ValueError(
                    f"{where} is ohmic, and part of it sits on oxide with no "
                    "silicon under it, so no carrier density to pin there. Keep "
                    "it on silicon, or make it a gate."
                )
            contacts.append(
                OhmicPlate(name=electrode.name, nodes=nodes, voltage=electrode.voltage)
            )
        else:
            if silicon[list(nodes)].any():
                raise ValueError(
                    f"{where} is a gate touching silicon. A metal on silicon is "
                    "a Schottky contact, which this solver does not model: a "
                    "gate sits on oxide. Put oxide under it, or make it ohmic."
                )
            contacts.append(
                GateContact(
                    name=electrode.name,
                    nodes=nodes,
                    voltage=electrode.voltage,
                    work_function=electrode.work_function,
                )
            )
        for node in nodes:
            if node in taken:
                raise ValueError(
                    f"electrodes {taken[node]!r} and {electrode.name!r} share "
                    f"the node at x={mesh.node_x[node]:g}, y={mesh.node_y[node]:g}"
                    " cm. One node cannot hold two contacts."
                )
            taken[node] = electrode.name

    ohmic = {
        node
        for contact in contacts
        if isinstance(contact, OhmicPlate)
        for node in contact.nodes
    }
    islands, count = ndimage.label(cells == SILICON)
    for island in range(1, count + 1):
        cell_rows, cell_columns = np.nonzero(islands == island)
        corners = {
            mesh.node_at(int(i) + di, int(j) + dj)
            for j, i in zip(cell_rows, cell_columns, strict=True)
            for dj in (0, 1)
            for di in (0, 1)
        }
        if not corners & ohmic:
            x = 0.5 * (columns[cell_columns[0]] + columns[cell_columns[0] + 1])
            y = 0.5 * (rows[cell_rows[0]] + rows[cell_rows[0] + 1])
            raise ValueError(
                f"the silicon around x={x:g}, y={y:g} cm floats: no ohmic "
                "contact touches it, so nothing sets its Fermi level and its "
                "charge is whatever the solver starts from. Put an ohmic "
                "electrode on it."
            )

    doping = Sum(tuple(_implant_profile(i, width, height) for i in implants))

    net = doping(Coordinates(mesh.node_x, mesh.node_y)).reshape(mesh.ny, mesh.nx)
    on_silicon = silicon.reshape(mesh.ny, mesh.nx)
    for kind, carriers, of_type in (
        ("p", "holes", net < 0.0),
        ("n", "electrons", net > 0.0),
    ):
        regions_of_type, count = ndimage.label(on_silicon & of_type)
        for region in range(1, count + 1):
            members = set(np.flatnonzero(regions_of_type.ravel() == region).tolist())
            if not members & ohmic:
                node = min(members)
                raise ValueError(
                    f"the {kind} silicon around x={mesh.node_x[node]:g}, "
                    f"y={mesh.node_y[node]:g} cm floats: no ohmic contact "
                    f"touches it, so its {carriers} reach a contact only "
                    "through a junction. That leakage is too small for the "
                    "solver to pin the region's potential, and the solve "
                    "stalls. This solver can't handle a floating body, so tie "
                    f"it down with an ohmic electrode on the {kind} region."
                )

    return build_device(
        mesh=mesh,
        doping=doping,
        contacts=tuple(contacts),
        material=material,
        regions=regions,
        degenerate=degenerate,
    )
