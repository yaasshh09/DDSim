"""The NMOS builder: what it constructs, and what it refuses.

The physics is in tests/analytic/test_mosfet_equilibrium.py. This file is about
the structure being the structure that was asked for, and it exists for the
same reason tests/unit/test_mos_cap.py does: almost every way of getting a
device geometry wrong still converges. A channel a hundred nanometres shorter
than requested solves perfectly and reports a threshold voltage that is simply
wrong, and there is nothing in the solution to say so. On a project whose
headline result is threshold voltage against gate length, that is the error
that would invalidate everything.

Two lengths, and they are not the same length
---------------------------------------------
`L_gate` is the gate electrode, which is what a process engineer draws and what
the sweep in phases/PHASE-5.md is against. The metallurgical channel is shorter,
because the source and drain diffuse sideways under the mask edge by
`lateral_diffusion`, so the junctions sit inside the gate span. Both are
checked here, separately, against closed forms.

Reading the junctions off the profile, not off the mesh
-------------------------------------------------------
A doping profile is a callable, so the junction positions are found by solving
the profile on a fine independent line rather than by hunting for a sign change
between mesh nodes. That measures what the builder constructed instead of what
the mesh happened to resolve, and it keeps these tests from failing for the
unrelated reason that a cell near the junction got coarser.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from ddsim.core import constants as C
from ddsim.device.doping import Coordinates
from ddsim.device.mosfet import BODY, DRAIN, GATE, SOURCE, nmos
from ddsim.device.regions import OXIDE, SILICON
from ddsim.discretize.boundary import GateContact, OhmicPlate

NA = 1e17
"""Substrate acceptor concentration [cm^-3]. Net doping is -NA."""

SD_PEAK = 1e20
"""Source and drain surface concentration [cm^-3]."""

L_GATE = 1e-4
"""Gate length [cm], 1 um."""

SD_LENGTH = 4e-5
"""Source and drain mask length, per side [cm], 0.4 um."""

CONTACT_LENGTH = 2e-5
"""Length of each source and drain contact plate [cm], 0.2 um."""

X_J = 1.5e-5
"""Junction depth [cm], 150 nm."""

LATERAL = 1e-5
"""Lateral diffusion under each gate edge [cm], 100 nm."""

T_OX = 2e-6
"""Oxide thickness [cm], 20 nm."""

T_SI = 1e-4
"""Silicon thickness [cm], 1 um."""

H_MIN_X = 2e-7
"""Column spacing asked for at each junction [cm], 2 nm."""

WIDTH = 2.0 * SD_LENGTH + L_GATE
"""Total device length [cm]."""


@pytest.fixture(scope="module")
def fet():
    return nmos(
        L_gate=L_GATE,
        sd_length=SD_LENGTH,
        contact_length=CONTACT_LENGTH,
        substrate_doping=-NA,
        sd_peak=SD_PEAK,
        x_j=X_J,
        lateral_diffusion=LATERAL,
        t_ox=T_OX,
        t_si=T_SI,
    )


def terminal(device, name):
    """The contact called `name`."""
    return next(c for c in device.contacts if c.name == name)


def net_doping_at(device, x: float, y: float) -> float:
    """The profile itself at one point [cm^-3], off the mesh entirely."""
    at = Coordinates(np.array([x]), np.array([y]))
    return float(device.doping(at)[0])


# ------------------------------------------------------------------ terminals


def test_the_device_has_four_terminals(fet):
    assert {c.name for c in fet.contacts} == {SOURCE, DRAIN, GATE, BODY}


def test_the_gate_is_a_gate_and_the_rest_are_plates(fet):
    assert isinstance(terminal(fet, GATE), GateContact)
    for name in (SOURCE, DRAIN, BODY):
        assert isinstance(terminal(fet, name), OhmicPlate)


def test_the_gate_sits_on_the_top_of_the_oxide(fet):
    """Not on the interface. The gate is metal on the far side of the oxide,
    and putting it on the silicon surface removes the oxide from the device
    while leaving every other number looking plausible."""
    top = fet.mesh.y_axis.x[-1]
    for node in terminal(fet, GATE).nodes:
        assert fet.mesh.node_y[node] == top


def test_the_gate_spans_the_channel_and_nothing_else(fet):
    """The whole of it and no more. A gate that overhangs the source is a
    different device with a different overlap capacitance, and one that falls
    short leaves a stretch of channel no gate controls."""
    x = np.sort(fet.mesh.node_x[list(terminal(fet, GATE).nodes)])

    assert x[0] == pytest.approx(SD_LENGTH, rel=1e-12)
    assert x[-1] == pytest.approx(SD_LENGTH + L_GATE, rel=1e-12)

    columns = np.sort(np.unique(fet.mesh.node_x))
    inside = columns[(columns >= x[0]) & (columns <= x[-1])]
    np.testing.assert_allclose(x, inside, rtol=1e-12)


def test_the_source_and_drain_plates_sit_on_the_silicon_surface(fet):
    """At the interface row, which is a semiconductor row: half of its dual
    cell is silicon, and it is where the channel forms."""
    for name in (SOURCE, DRAIN):
        for node in terminal(fet, name).nodes:
            assert fet.mesh.node_y[node] == pytest.approx(T_SI, rel=1e-12)


def test_the_source_and_drain_contacts_stop_short_of_the_junction(fet):
    """A contact pins psi, n and p. Running one up to the metallurgical
    junction pins the junction itself, and the built in potential stops being
    something the solver works out."""
    source_x = fet.mesh.node_x[list(terminal(fet, SOURCE).nodes)]
    drain_x = fet.mesh.node_x[list(terminal(fet, DRAIN).nodes)]

    assert source_x.min() == 0.0
    assert source_x.max() == pytest.approx(CONTACT_LENGTH, rel=1e-12)
    assert drain_x.min() == pytest.approx(WIDTH - CONTACT_LENGTH, rel=1e-12)
    assert drain_x.max() == pytest.approx(WIDTH, rel=1e-12)


def test_the_body_contact_covers_the_whole_bottom(fet):
    """A point contact would leave the rest of that edge reflecting, which is
    a different device. Same argument as the MOS capacitor."""
    nodes = terminal(fet, BODY).nodes

    assert len(nodes) == fet.mesh.nx
    assert np.all(fet.mesh.node_y[list(nodes)] == 0.0)


# ----------------------------------------------------------------- the stack


def test_the_layers_have_the_thicknesses_asked_for(fet):
    y = fet.mesh.y_axis.x
    assert y[-1] == pytest.approx(T_SI + T_OX, rel=1e-14)
    assert np.count_nonzero(y == T_SI) == 1


def test_the_cells_below_the_interface_are_silicon_and_above_are_oxide(fet):
    assert fet.regions is not None
    interface = int(np.flatnonzero(fet.mesh.y_axis.x == T_SI)[0])
    material = fet.regions.cell_material

    assert np.all(material[:interface] == SILICON)
    assert np.all(material[interface:] == OXIDE)


def test_the_silicon_is_graded_to_the_surface(fet):
    """The inversion layer is a couple of nanometres thick and the substrate
    is a micron, so one spacing cannot serve both."""
    interface = int(np.flatnonzero(fet.mesh.y_axis.x == T_SI)[0])
    h = np.diff(fet.mesh.y_axis.x[: interface + 1])

    assert h[-1] < h[0] / 100.0


def test_the_mesh_is_finest_at_the_junctions(fet):
    """Where the doping turns over. docs/02-numerics.md wants the local Debye
    length resolved there, and it is smallest at the most heavily doped end of
    the junction."""
    x = np.sort(np.unique(fet.mesh.node_x))
    h = np.diff(x)
    finest = x[np.argmin(h)]

    assert finest == pytest.approx(SD_LENGTH, abs=2.0 * float(np.min(h)))


def test_no_doping_survives_inside_the_oxide(fet):
    """The zero charge volume already makes this true in the equations. This
    is for everything that reads the doping for another reason."""
    assert fet.regions is not None
    oxide = np.zeros(fet.mesh.n_nodes, dtype=bool)
    oxide[fet.regions.oxide_nodes] = True

    np.testing.assert_array_equal(fet.net_doping.data[oxide], 0.0)


# ------------------------------------------------------------------- doping


def test_the_source_surface_reaches_the_concentration_asked_for(fet):
    """At the outer edge, where neither the depth falloff nor the lateral one
    has taken anything off yet."""
    assert net_doping_at(fet, 0.0, T_SI) == pytest.approx(SD_PEAK - NA, rel=1e-9)


def test_the_channel_is_the_substrate(fet):
    """Halfway between the junctions, at the surface. Nothing has diffused
    that far, and if it had, the threshold voltage would be set by the tail of
    an implant rather than by the substrate doping."""
    middle = net_doping_at(fet, 0.5 * WIDTH, T_SI)

    assert middle == pytest.approx(-NA, rel=1e-6)


def test_the_substrate_is_the_substrate_under_the_source(fet):
    """Deep below the source, past the junction, it is body again."""
    assert net_doping_at(fet, 0.0, 0.0) == pytest.approx(-NA, rel=1e-12)


def test_the_junction_sits_at_the_depth_asked_for(fet):
    """The metallurgical junction, where net doping crosses zero, measured
    straight off the profile.

    Closed form: the implant is Gaussian in depth from the surface, so it
    equals the substrate doping at x_j by construction of sigma. Getting sigma
    wrong moves the junction, and a junction deeper than asked for lowers the
    threshold and worsens the roll-off, both of which would read as physics.
    """
    depth = brentq(
        lambda y: net_doping_at(fet, 0.0, y), T_SI - 2.0 * X_J, T_SI, xtol=1e-16
    )

    assert T_SI - depth == pytest.approx(X_J, rel=1e-9)


def test_the_junctions_encroach_under_the_gate_by_the_lateral_diffusion(fet):
    """So the metallurgical channel is shorter than the gate, which is the
    whole reason both lengths are named separately.

    The source edge of the gate is at SD_LENGTH and the junction is inside it
    by LATERAL, at the surface where the implant is strongest.
    """
    junction = brentq(
        lambda x: net_doping_at(fet, x, T_SI),
        SD_LENGTH,
        0.5 * WIDTH,
        xtol=1e-16,
    )

    assert junction - SD_LENGTH == pytest.approx(LATERAL, rel=1e-9)


def test_the_metallurgical_channel_is_shorter_than_the_gate(fet):
    """Stated as the length itself, because that is the number the threshold
    voltage actually responds to."""
    source_side = brentq(
        lambda x: net_doping_at(fet, x, T_SI), SD_LENGTH, 0.5 * WIDTH, xtol=1e-16
    )
    drain_side = brentq(
        lambda x: net_doping_at(fet, x, T_SI),
        0.5 * WIDTH,
        SD_LENGTH + L_GATE,
        xtol=1e-16,
    )

    assert drain_side - source_side == pytest.approx(
        L_GATE - 2.0 * LATERAL, rel=1e-9
    )


def test_the_drain_is_the_source_mirrored(fet):
    """Node for node, on a mesh that is itself symmetric. If either the
    profile or the mesh lost its symmetry, a device with the source and the
    drain at the same bias would carry a current."""
    doping = fet.net_doping.data
    x = fet.mesh.node_x
    mirror = np.empty(fet.mesh.n_nodes, dtype=np.int64)
    columns = np.sort(np.unique(x))

    for i in range(fet.mesh.nx):
        assert columns[i] == pytest.approx(
            WIDTH - columns[fet.mesh.nx - 1 - i], abs=1e-16
        )
        for j in range(fet.mesh.ny):
            mirror[fet.mesh.node_at(i, j)] = fet.mesh.node_at(
                fet.mesh.nx - 1 - i, j
            )

    np.testing.assert_allclose(doping, doping[mirror], rtol=1e-12)


# ------------------------------------------------------------------ refusals


def test_a_gate_shorter_than_the_lateral_diffusion_is_refused():
    """The two junctions would have met and there would be no channel at all.
    A solver handed that geometry returns a short circuit, not an error."""
    with pytest.raises(ValueError, match="no channel"):
        nmos(L_gate=1e-5, lateral_diffusion=1e-5)


def test_a_contact_reaching_the_gate_edge_is_refused():
    with pytest.raises(ValueError, match="contact_length"):
        nmos(sd_length=4e-5, contact_length=4e-5)


def test_a_junction_deeper_than_the_silicon_is_refused():
    with pytest.raises(ValueError, match="x_j"):
        nmos(x_j=2e-4, t_si=1e-4)


def test_a_source_lighter_than_the_substrate_is_refused():
    """There would be no junction to find, and sigma would come out imaginary
    rather than the geometry being reported as impossible."""
    with pytest.raises(ValueError, match="sd_peak"):
        nmos(sd_peak=1e16, substrate_doping=-1e17)


def test_an_n_type_substrate_is_refused():
    """This builds an NMOS. An n-type body is a different device and the sign
    conventions below would all be backwards in it."""
    with pytest.raises(ValueError, match="p-type"):
        nmos(substrate_doping=1e17)


def test_a_negative_gate_length_is_refused():
    with pytest.raises(ValueError, match="L_gate"):
        nmos(L_gate=-1e-4)


def test_a_gate_work_function_can_be_chosen():
    """n+ poly is the default because it is the ordinary NMOS gate, but the
    threshold moves with it and the choice belongs to the caller."""
    fet = nmos(work_function=C.PHI_M_MIDGAP)

    assert terminal(fet, GATE).work_function == C.PHI_M_MIDGAP


def test_a_mesh_segment_with_one_node_is_refused():
    """A segment with a single node has no extent to carry a field across,
    and stacking one produces a mesh with a repeated position."""
    with pytest.raises(ValueError, match="at least 2 nodes"):
        nmos(n_channel=1)


def test_a_gate_too_short_to_grade_is_meshed_uniformly():
    """h_min_x is a refinement target, not a floor.

    Grading exists to spend nodes near the junction and save them far from
    it. Once the gate is short enough that the columns it has already sit
    below h_min_x when spread evenly, there is nothing left to save and no
    coarse end to grade away from, so the segment is uniform. Refusing that
    request would mean the 50 nm end of the sweep phases/PHASE-5.md is graded
    on could not be built at all.
    """
    short = nmos(
        L_gate=5e-6, lateral_diffusion=1e-6, x_j=2.5e-6, h_min_x=H_MIN_X
    )

    x = np.sort(np.unique(short.mesh.node_x))
    inside = (x >= SD_LENGTH) & (x <= SD_LENGTH + 5e-6)
    h = np.diff(x[inside])

    assert h.max() <= H_MIN_X
    np.testing.assert_allclose(h, h[0], rtol=1e-12)
