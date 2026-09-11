"""An n-channel MOSFET, the Phase 5 device.

Four terminals on two materials: a p-type body with an n+ source and drain
implanted into its surface, an oxide over the whole of that surface, and a gate
electrode on the oxide over the channel alone.

The point of phases/PHASE-5.md is that threshold roll-off, DIBL and velocity
saturation come out of Poisson and two continuity equations on this geometry
and nothing else. So nothing here is fitted and nothing is a model of a short
channel effect. Everything below is either a length, a concentration, or a
consequence of one of those.

The two channel lengths
-----------------------
`L_gate` is the electrode, which is what the sweep is against and what a data
sheet quotes. The metallurgical channel is shorter, by `lateral_diffusion` at
each end, because an implant spreads sideways under the mask edge. That gap is
not a correction bolted on afterwards: it is in the doping profile, so the
solver sees the shorter channel without being told about it, and the roll-off
that follows is a result rather than an input.

Where the terminals are
-----------------------
The oxide spans the whole width, so the Si/SiO2 interface is one flat node
line, which is what device/regions.py requires and what keeps the mesh a clean
tensor product.

The gate covers the channel span exactly. The oxide above the source and drain
has no electrode over it and so takes the natural condition, dpsi/dn = 0. That
is the ideal structure of docs/01-physics.md: no gate overlap, so no overlap
capacitance and no gate induced drain leakage.

Source and drain are plates on the silicon surface, at the interface row,
covering the outer part of each implant. They stop short of the mask edge
deliberately. A contact pins psi, n and p, so running one all the way to the
junction would pin the junction, and the built in potential would stop being
something the solver works out for itself.

The body is a plate over the whole bottom edge, for the reason the MOS
capacitor gives: a point contact leaves the rest of that boundary reflecting,
which is a different device.

What is not here
----------------
No fixed interface charge, no poly depletion, no halo or pocket implant, and no
lightly doped drain. Every one of those exists to move the threshold voltage or
to soften a field, which is to say every one of them would be tuning the answer
this phase is supposed to derive.
"""

from __future__ import annotations

import math

from scipy.special import erfcinv as _erfcinv

from ddsim.core import constants as C
from ddsim.device.builder import Device, Material, build_device
from ddsim.device.doping import Along, Erfc, Gaussian, Mirrored, Uniform
from ddsim.device.regions import stacked_regions
from ddsim.discretize.boundary import GateContact, OhmicPlate
from ddsim.mesh.mesh1d import (
    Mesh1D,
    graded_mesh_1d,
    stacked_mesh_1d,
    uniform_mesh_1d,
)
from ddsim.mesh.mesh2d import tensor_mesh_2d

SOURCE = "source"
"""Terminal name of the source."""

DRAIN = "drain"
"""Terminal name of the drain."""

GATE = "gate"
"""Terminal name of the gate."""

BODY = "body"
"""Terminal name of the substrate contact."""


def _junction_mesh(
    length: float, n_nodes: int, refine_at: float, h_min: float
) -> Mesh1D:
    """Columns graded toward a junction, or uniform if they are already finer.

    `h_min` is what the junction wants resolved, not a floor on cell size.
    Grading trades a coarse far end for a fine near one, so it has something
    to trade only while spreading the nodes evenly would leave them coarser
    than `h_min`. Once the segment is short enough that it would not, every
    cell is already inside the target and the even spacing is the answer.

    This is what lets the short end of the gate length sweep exist: at a 50 nm
    gate the half channel holds its columns at 1.7 nm with h_min asking for 2.
    """
    if h_min * (n_nodes - 1) >= length:
        return uniform_mesh_1d(length=length, n_nodes=n_nodes)
    return graded_mesh_1d(
        length=length, n_nodes=n_nodes, refine_at=refine_at, h_min=h_min
    )


def nmos(
    L_gate: float = 1e-4,
    sd_length: float = 4e-5,
    contact_length: float = 2e-5,
    substrate_doping: float = -1e17,
    sd_peak: float = 1e20,
    x_j: float = 1.5e-5,
    lateral_diffusion: float = 1e-5,
    t_ox: float = 2e-6,
    t_si: float = 1e-4,
    n_contact: int = 6,
    n_sd: int = 12,
    n_channel: int = 16,
    n_silicon: int = 101,
    n_oxide: int = 33,
    h_min_x: float = 2e-7,
    h_min_y: float = 6.25e-9,
    gate_voltage: float = 0.0,
    drain_voltage: float = 0.0,
    source_voltage: float = 0.0,
    body_voltage: float = 0.0,
    work_function: float = C.PHI_M_N_POLY,
    material: Material | None = None,
    degenerate: bool = True,
) -> Device:
    """An n-channel MOSFET on a p-type substrate.

    Args:
        L_gate: gate length [cm], the electrode span. 1e-4 is 1 um.
        sd_length: length of each source and drain region [cm], from the outer
            boundary to the gate mask edge.
        contact_length: length of each source and drain contact plate [cm],
            measured in from the outer boundary. Less than sd_length.
        substrate_doping: net doping of the body [cm^-3], negative for p-type.
        sd_peak: source and drain surface concentration [cm^-3].
        x_j: junction depth [cm], where the implant meets the substrate doping
            directly below the outer part of the source.
        lateral_diffusion: how far the junction reaches under the gate mask
            edge at the surface [cm]. The metallurgical channel is L_gate less
            twice this.
        t_ox: oxide thickness [cm]. 2e-6 is 20 nm.
        t_si: silicon thickness [cm], several times the depletion width.
        n_contact: columns under each contact plate.
        n_sd: columns from a contact edge to the gate mask edge.
        n_channel: columns in each half of the channel.
        n_silicon: rows through the silicon, including the interface.
        n_oxide: rows through the oxide, including the interface.
        h_min_x: column spacing at each junction [cm].
        h_min_y: row spacing at the silicon surface [cm]. This is the one
            spacing the drain current is really sensitive to, because the
            inversion layer is the only structure on the device a mesh can
            miss. Halve it and n_oxide together or the Si/SiO2 seam opens up,
            see docs/05-pitfalls.md. The default is the rung of
            tests/convergence/test_mosfet_mesh_convergence.py where the drain
            current stops moving by more than a tenth of what benchmark 6
            asserts.
        gate_voltage: bias on the gate [V].
        drain_voltage: bias on the drain [V].
        source_voltage: bias on the source [V].
        body_voltage: bias on the substrate contact [V].
        work_function: work function of the gate electrode [eV]. n+ poly by
            default, the ordinary NMOS gate.
        material: defaults to silicon at 300 K.

    The implant is separable and both halves of it are closed form, which is
    what lets a test measure the junction depth and the lateral encroachment
    against the numbers asked for rather than against a previous run:

        N(x, y) = sd_peak * lateral(x) * exp(-(t_si - y)^2 / (2 sigma^2))

    with `lateral` an erfc edge at the mask. sigma comes from x_j and the two
    concentrations, and the erfc length from lateral_diffusion and the same
    two, so both junctions land exactly where they were asked to.
    """
    for name, value in (
        ("L_gate", L_gate),
        ("sd_length", sd_length),
        ("contact_length", contact_length),
        ("x_j", x_j),
        ("lateral_diffusion", lateral_diffusion),
        ("t_ox", t_ox),
        ("t_si", t_si),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}")

    if substrate_doping >= 0.0:
        raise ValueError(
            f"nmos builds an n-channel device, so the body must be p-type and "
            f"substrate_doping negative, got {substrate_doping:g}. The sign "
            "convention is net doping, Nd - Na, everywhere in this codebase."
        )

    Na = -substrate_doping
    if sd_peak <= Na:
        raise ValueError(
            f"sd_peak must exceed the substrate doping, got sd_peak="
            f"{sd_peak:g} against {Na:g} cm^-3. A source no heavier than the "
            "body it sits in makes no junction, and there would be no depth "
            "at which to put one."
        )

    if x_j >= t_si:
        raise ValueError(
            f"x_j must be less than t_si, got x_j={x_j:g} cm in {t_si:g} cm "
            "of silicon. The source would reach the body contact."
        )

    if 2.0 * lateral_diffusion >= L_gate:
        raise ValueError(
            f"no channel is left: the source and drain each reach "
            f"{lateral_diffusion:g} cm under a gate {L_gate:g} cm long, so "
            "the two junctions meet or cross. That geometry is a short "
            "circuit, and a solver handed it returns one rather than an error."
        )

    if contact_length >= sd_length:
        raise ValueError(
            f"contact_length must be less than sd_length, got "
            f"{contact_length:g} against {sd_length:g} cm. A contact reaching "
            "the mask edge pins the junction itself, and the built in "
            "potential stops being something the solve works out."
        )

    if n_contact < 2 or n_sd < 2 or n_channel < 2 or n_silicon < 2 or n_oxide < 2:
        raise ValueError(
            "every mesh segment needs at least 2 nodes, got n_contact="
            f"{n_contact}, n_sd={n_sd}, n_channel={n_channel}, n_silicon="
            f"{n_silicon}, n_oxide={n_oxide}. A segment with one node has no "
            "extent to carry a field across."
        )

    width = 2.0 * sd_length + L_gate
    outer = sd_length - contact_length
    half_gate = 0.5 * L_gate

    # Six segments, symmetric about the centre, with node lines landing on
    # every boundary that anything is measured against: the two contact edges
    # and the two gate mask edges. The two graded pairs are mirror images, so
    # the columns come out symmetric and a device with the source and drain at
    # the same bias has nothing to break its symmetry.
    x_axis = stacked_mesh_1d(
        uniform_mesh_1d(length=contact_length, n_nodes=n_contact),
        _junction_mesh(outer, n_sd, refine_at=outer, h_min=h_min_x),
        _junction_mesh(half_gate, n_channel, refine_at=0.0, h_min=h_min_x),
        _junction_mesh(half_gate, n_channel, refine_at=half_gate, h_min=h_min_x),
        _junction_mesh(outer, n_sd, refine_at=0.0, h_min=h_min_x),
        uniform_mesh_1d(length=contact_length, n_nodes=n_contact),
    )

    # The silicon is graded to the surface, where the inversion layer is a
    # couple of nanometres thick. The oxide holds no charge, so its potential
    # is a straight line and uniform cells resolve it exactly.
    y_axis = stacked_mesh_1d(
        graded_mesh_1d(
            length=t_si, n_nodes=n_silicon, refine_at=t_si, h_min=h_min_y
        ),
        uniform_mesh_1d(length=t_ox, n_nodes=n_oxide),
    )

    mesh = tensor_mesh_2d(x_axis, y_axis)
    regions = stacked_regions(mesh, interface_y=t_si)

    # Column indices of every boundary, by counting nodes rather than by
    # searching positions. Stacking shares one node at each join, so segment k
    # ends at sum of the counts so far less one per join.
    i_contact_end = n_contact - 1
    i_gate_start = n_contact + n_sd - 2
    i_gate_end = i_gate_start + 2 * (n_channel - 1)
    i_drain_start = i_gate_end + n_sd - 1

    surface = n_silicon - 1
    top = mesh.ny - 1

    contacts = (
        OhmicPlate(
            name=SOURCE,
            nodes=tuple(
                mesh.node_at(i, surface) for i in range(i_contact_end + 1)
            ),
            voltage=source_voltage,
        ),
        OhmicPlate(
            name=DRAIN,
            nodes=tuple(
                mesh.node_at(i, surface) for i in range(i_drain_start, mesh.nx)
            ),
            voltage=drain_voltage,
        ),
        GateContact(
            name=GATE,
            nodes=tuple(
                mesh.node_at(i, top) for i in range(i_gate_start, i_gate_end + 1)
            ),
            voltage=gate_voltage,
            work_function=work_function,
        ),
        OhmicPlate(
            name=BODY,
            nodes=tuple(mesh.node_at(i, 0) for i in range(mesh.nx)),
            voltage=body_voltage,
        ),
    )

    # The implant, in closed form both ways. sigma is set by where the depth
    # profile is to cross the substrate doping, and the erfc length by where
    # the lateral one is, so x_j and lateral_diffusion are what they say.
    sigma = x_j / math.sqrt(2.0 * math.log(sd_peak / Na))
    edge = lateral_diffusion / float(_erfcinv(2.0 * Na / sd_peak))

    source = (
        Along(Erfc(peak=0.5, position=sd_length, length=edge), "x")
        * Along(Gaussian(peak=1.0, centre=t_si, sigma=sigma), "y")
        * sd_peak
    )
    doping = Uniform(substrate_doping) + source + Mirrored(source, 0.5 * width)

    return build_device(
        mesh=mesh,
        doping=doping,
        contacts=contacts,
        material=material,
        regions=regions,
        degenerate=degenerate,
    )
