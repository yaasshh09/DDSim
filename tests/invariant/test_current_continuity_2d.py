"""Current continuity in 2D, the last open acceptance criterion of Phase 4.

phases/PHASE-4.md asks for four things this file measures: that the 2D path
reproduces the 1D answer, that no carrier density goes negative anywhere in
2D, that current continuity holds, and, by implication of the first, that a
device uniform in y is the 1D device.

What the invariant becomes in two dimensions
--------------------------------------------
In 1D it is that Jn + Jp is the same number on every edge, because there is
only one path from one contact to the other. In 2D that is false and should
be: the current spreads across the height of the device, so the density on any
one edge says nothing on its own. What is conserved is the integral.

    the total current crossing every vertical cut is the same

A cut between column i and column i+1 is the set of horizontal edges at that
column, one per row, and the current through it is the sum of (Jn + Jp) times
the face each of those edges offers a carrier. That is the statement the
discrete divergence theorem makes, and it is exact for box integration in a
source free steady state, which is why the tolerance below is at the
arithmetic floor rather than at anything physical.

The dielectric cap
------------------
The second half of the file puts an oxide layer on top of the same silicon and
solves transport through it. This is the combination that has never existed
before: an insulator and a continuity equation in one device. A MOSFET is
exactly this stack with a gate on it, so the trap here is the one Phase 5 will
walk into first, and it is worth catching with a device that has no gate and
no short channel physics in it to argue about.

The trap is that a zero charge volume in the oxide does nothing to the carrier
flux. The volume multiplies what the equations integrate, and a flux is not
that. Leave the dual face alone and the edge running from the interface up
into the oxide carries electrons out of the silicon and into a node whose
density is pinned at zero, which is a perfect sink bolted to the surface of
the device. See EdgeGeometry.carrier_face and device/regions.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddsim.device.builder import build_device
from ddsim.device.doping import abrupt_junction
from ddsim.device.regions import stacked_regions
from ddsim.device.transport import TransportModels, solve_bias_newton
from ddsim.discretize.boundary import OhmicPlate
from ddsim.extract.iv import (
    current_densities,
    edge_current_face,
    terminal_currents,
)
from ddsim.mesh.mesh1d import graded_mesh_1d, stacked_mesh_1d, uniform_mesh_1d
from ddsim.mesh.mesh2d import tensor_mesh_2d
from ddsim.physics.recombination import NoRecombination

MICRON = 1e-4
"""One micron [cm]."""

LENGTH = 12 * MICRON
"""Device length along x [cm], the same diode the 1D file uses."""

JUNCTION = 6 * MICRON
"""Metallurgical junction position [cm]."""

NX = 81
"""Nodes along x. Graded to the junction, as in 1D."""

HEIGHT = 2 * MICRON
"""Silicon height along y [cm]."""

NY = 5
"""Nodes through the silicon, including the surface."""

T_OX = 0.1 * MICRON
"""Oxide thickness of the capped device [cm], 10 nm."""

N_OX = 4
"""Nodes through the oxide, including the interface."""

DOPING = 1e16
"""Doping either side of the junction [cm^-3]."""

BIAS = 0.5
"""Forward bias [V]. Above the cancellation floor described in the 1D file."""


def x_axis():
    """The x mesh, graded to the junction. Shared by every device here."""
    return graded_mesh_1d(
        length=LENGTH, n_nodes=NX, refine_at=JUNCTION, h_min=5e-7
    )


def diode_2d(voltage: float = BIAS):
    """A pn diode on a 2D mesh, uniform in y, contacted by two plates.

    The contacts are plates over the whole left and right edges. A point
    contact on one node of an edge is a different device: it would leave the
    rest of that edge reflecting, and the answer would not be the 1D one.
    """
    mesh = tensor_mesh_2d(x_axis(), uniform_mesh_1d(length=HEIGHT, n_nodes=NY))

    return build_device(
        mesh=mesh,
        doping=abrupt_junction(Na=DOPING, Nd=DOPING, position=JUNCTION),
        contacts=(
            OhmicPlate(
                name="anode",
                nodes=tuple(mesh.node_at(0, j) for j in range(mesh.ny)),
                voltage=voltage,
            ),
            OhmicPlate(
                name="cathode",
                nodes=tuple(mesh.node_at(mesh.nx - 1, j) for j in range(mesh.ny)),
                voltage=0.0,
            ),
        ),
    )


def capped_diode_2d(voltage: float = BIAS):
    """The same silicon with an oxide layer stacked on top of it.

    No gate. The oxide is a passivation layer here and nothing more, which is
    deliberate: it puts an insulator and a transport solve in one device
    without bringing in any of the physics Phase 5 is about.

    The contacts stop at the interface. Metal on the side of an oxide layer
    would be a second gate.
    """
    silicon = uniform_mesh_1d(length=HEIGHT, n_nodes=NY)
    oxide = uniform_mesh_1d(length=T_OX, n_nodes=N_OX)
    mesh = tensor_mesh_2d(x_axis(), stacked_mesh_1d(silicon, oxide))
    regions = stacked_regions(mesh, interface_y=HEIGHT)

    return build_device(
        mesh=mesh,
        doping=abrupt_junction(Na=DOPING, Nd=DOPING, position=JUNCTION),
        contacts=(
            OhmicPlate(
                name="anode",
                nodes=tuple(mesh.node_at(0, j) for j in range(NY)),
                voltage=voltage,
            ),
            OhmicPlate(
                name="cathode",
                nodes=tuple(mesh.node_at(mesh.nx - 1, j) for j in range(NY)),
                voltage=0.0,
            ),
        ),
        regions=regions,
    )


def solved(device, recombination=None):
    """A converged coupled solve, with the models it was solved with."""
    models = TransportModels.for_device(device, recombination=recombination)
    state = solve_bias_newton(device, models=models)
    assert state.newton is not None and state.newton.converged, (
        f"the 2D solve did not converge: {state.newton}"
    )
    return state, models


def cut_currents(device, state, models) -> np.ndarray:
    """Total current per unit depth through every vertical cut [A/cm].

    One entry per column of horizontal edges. Each is the sum over the rows of
    the current density on the edge times the face that edge offers a carrier,
    which is the discrete surface integral of J over a plane through the
    device.

    The face is the semiconductor face and not the whole dual face. On the
    capped device below they differ: the edge lying along the interface has
    silicon under it and oxide over it, and only the silicon half of its face
    carries anything. Weighting by the whole face there counts a piece of
    dielectric as if it conducted, and the sum stops being conserved.
    """
    mesh = device.mesh
    Jn, Jp = current_densities(device, state, models)
    total = Jn.data + Jp.data
    face = edge_current_face(device)

    per_cut = []
    for i in range(mesh.nx - 1):
        edges = [j * (mesh.nx - 1) + i for j in range(mesh.ny)]
        per_cut.append(float(np.sum(total[edges] * face[edges])))
    return np.asarray(per_cut)


def spread(values: np.ndarray) -> float:
    """Largest deviation from the mean, relative to the mean [1]."""
    return float(np.max(np.abs(values - values.mean())) / abs(values.mean()))


@pytest.fixture(scope="module")
def plain():
    """The all silicon 2D diode, solved once for the whole module."""
    device = diode_2d()
    state, models = solved(device, NoRecombination())
    return device, state, models


@pytest.fixture(scope="module")
def capped():
    """The same diode with a dielectric cap, solved once."""
    device = capped_diode_2d()
    state, models = solved(device, NoRecombination())
    return device, state, models


# ------------------------------------------------------------- the gate


def test_the_total_current_through_every_cut_is_the_same(plain) -> None:
    """The Phase 4 form of the Phase 2 gate.

    Box integration makes the discrete divergence of the discrete current
    exactly zero in a source free steady state, in any dimension. Summing that
    over every dual cell to the left of a plane telescopes to the flux through
    the plane, so the number below cannot depend on which plane it is, however
    the mesh is graded and however the current is distributed over the height.
    """
    device, state, models = plain
    cuts = cut_currents(device, state, models)

    deviation = spread(cuts)
    assert deviation < 1e-6, f"the cut current varies by {deviation:.2e}"


def test_the_terminal_currents_sum_to_zero(plain) -> None:
    """Kirchhoff, on plates rather than points.

    A plate terminal is the sum of the residual over every node it covers.
    Getting that wrong by reading one node of the plate would leave most of
    the current unaccounted for and this would fail by order one.
    """
    device, state, models = plain
    currents = terminal_currents(device, state, models)

    largest = max(abs(value) for value in currents.values())
    assert abs(sum(currents.values())) < 1e-8 * largest


def test_no_density_is_negative_anywhere(plain) -> None:
    """A Phase 4 acceptance criterion in its own words.

    docs/04-validation.md: a negative density means the maximum principle is
    broken. Nothing is clamped anywhere, so this is a property of the scheme
    and not of a guard.
    """
    _, state, _ = plain

    assert np.all(state.n.data > 0.0)
    assert np.all(state.p.data > 0.0)


def test_the_current_runs_from_the_anode(plain) -> None:
    """The first thing to check when a sign is suspect.

    docs/05-pitfalls.md: reversing the Bernoulli asymmetry converges cleanly
    to a physically wrong answer with the current running backwards.
    """
    device, state, models = plain
    currents = terminal_currents(device, state, models)

    assert currents["anode"] > 0.0
    assert currents["cathode"] < 0.0


# ------------------------------------------------- the reduction to 1D


def test_the_solution_is_the_same_on_every_row(plain) -> None:
    """A device uniform in y solves to a state uniform in y.

    Nothing here forces that. The rows are coupled by vertical edges that are
    free to carry flux, and they carry none only because the state either side
    of them is equal. If the dual face of a vertical edge came from the wrong
    axis, or the plates covered the wrong nodes, the rows would differ.
    """
    device, state, _ = plain
    nx, ny = device.mesh.nx, device.mesh.ny

    for name, field in (("psi", state.psi), ("n", state.n), ("p", state.p)):
        rows = field.data.reshape(ny, nx)
        for j in range(1, ny):
            np.testing.assert_allclose(
                rows[j], rows[0], rtol=1e-12, err_msg=f"{name} row {j}"
            )


def test_the_2d_answer_is_the_1d_answer(plain) -> None:
    """phases/PHASE-4.md, in its own words, at 1e-10.

    The 1D problem is not an approximation to a device uniform in y. It is the
    same problem with one coordinate suppressed, so the answer has to be the
    same numbers and not merely close ones.
    """
    device, state, _ = plain

    one_d = build_device(
        mesh=x_axis(),
        doping=abrupt_junction(Na=DOPING, Nd=DOPING, position=JUNCTION),
        contacts=(
            OhmicPlate(name="anode", nodes=(0,), voltage=BIAS),
            OhmicPlate(name="cathode", nodes=(NX - 1,), voltage=0.0),
        ),
    )
    reference, _ = solved(one_d, NoRecombination())

    rows = state.psi.data.reshape(device.mesh.ny, device.mesh.nx)
    np.testing.assert_allclose(rows[0], reference.psi.data, rtol=1e-10)

    rows = state.n.data.reshape(device.mesh.ny, device.mesh.nx)
    np.testing.assert_allclose(rows[0], reference.n.data, rtol=1e-10)

    rows = state.p.data.reshape(device.mesh.ny, device.mesh.nx)
    np.testing.assert_allclose(rows[0], reference.p.data, rtol=1e-10)


def test_the_terminal_current_is_the_1d_one_times_the_height(plain) -> None:
    """The one number a 2D device reports that a 1D one cannot.

    A 1D current density is per unit area and a 2D one is per unit depth, so
    the two differ by the height of the device and by nothing else. This is
    the check that the dual faces are areas and not lengths, or the other way
    round, which no comparison of psi would notice.
    """
    device, state, models = plain
    cuts = cut_currents(device, state, models)

    one_d = build_device(
        mesh=x_axis(),
        doping=abrupt_junction(Na=DOPING, Nd=DOPING, position=JUNCTION),
        contacts=(
            OhmicPlate(name="anode", nodes=(0,), voltage=BIAS),
            OhmicPlate(name="cathode", nodes=(NX - 1,), voltage=0.0),
        ),
    )
    reference, reference_models = solved(one_d, NoRecombination())
    expected = terminal_currents(one_d, reference, reference_models)["anode"]

    assert cuts.mean() == pytest.approx(expected * HEIGHT, rel=1e-8)


# ------------------------------------------------- transport under an oxide


def test_no_carrier_crosses_into_the_dielectric(capped) -> None:
    """Exactly zero on every edge the oxide touches, not merely small.

    Two families of edge, and both matter. The vertical edges leaving the
    interface are the ones that drain an inversion layer, and the horizontal
    edges inside the oxide are the ones that would let it conduct along the
    surface. The face is zero on both, so the flux is zero by construction
    rather than because the densities above happen to be.
    """
    device, state, models = capped
    Jn, Jp = current_densities(device, state, models)

    blocked = np.flatnonzero(device.regions.semiconductor_face == 0.0)
    assert blocked.size > 0

    np.testing.assert_array_equal(Jn.data[blocked], 0.0)
    np.testing.assert_array_equal(Jp.data[blocked], 0.0)


def test_the_capped_device_still_conserves_current(capped) -> None:
    """The invariant, with an insulator sitting on the device.

    This is the one that fails if the oxide is allowed to take carriers. The
    leak is largest where the density at the surface is largest, so the cut
    currents fall off across the device instead of staying flat, and the
    shape of the failure points straight at the interface.
    """
    device, state, models = capped
    cuts = cut_currents(device, state, models)

    deviation = spread(cuts)
    assert deviation < 1e-6, f"the cut current varies by {deviation:.2e}"


def test_the_capped_terminal_currents_sum_to_zero(capped) -> None:
    """Where a leak into the oxide would show up as broken bookkeeping.

    Summing either continuity residual over every node telescopes the
    divergence away, so the terminal currents sum to zero only if every node
    that is not a terminal has a residual of zero. An oxide node is not a
    terminal and its rows are pinned, so any current the oxide absorbs is
    current that leaves the terminals without arriving anywhere they can see.
    """
    device, state, models = capped
    currents = terminal_currents(device, state, models)

    largest = max(abs(value) for value in currents.values())
    assert abs(sum(currents.values())) < 1e-8 * largest


def test_the_oxide_holds_no_carriers(capped) -> None:
    """Zero, because there is nothing there to hold one.

    Not a small number produced by a Boltzmann factor of a potential no
    carrier is sitting in. See docs/07-decisions.md.
    """
    device, state, _ = capped
    oxide = list(device.carrier_free_nodes)
    assert oxide

    np.testing.assert_array_equal(state.n.data[oxide], 0.0)
    np.testing.assert_array_equal(state.p.data[oxide], 0.0)
