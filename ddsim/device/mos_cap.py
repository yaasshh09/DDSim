"""A MOS capacitor, the Phase 4 test device.

Metal on oxide on silicon on a body contact. Two materials, one gate and one
plate, and the whole thing solves at equilibrium: no current flows through an
ideal insulator, so the semiconductor stays in equilibrium with its body
contact at every gate bias. phi_n and phi_p are flat at the body potential and
Poisson closes on psi alone. That is why this device needs nothing beyond what
Phase 1 already had, once the mesh has two dimensions and the mesh has two
materials in it.

The y mesh is a stack, not one graded axis
------------------------------------------
The two layers want different meshes, and one axis cannot give them both.

The silicon is graded hard to the surface. In inversion the electrons sit
within a couple of nanometres of the interface, so the surface spacing has to
be a fraction of that, while the substrate has to be several times the maximum
depletion width so the body contact is not holding the depletion region open.
Those two demands are three orders of magnitude apart, which is what grading is
for.

The oxide holds no charge, so its potential is exactly linear and a handful of
uniform cells resolves it to machine precision. Grading it would spend nodes
resolving a straight line.

Stacking the two also makes the interface a node by construction, which
device/regions.py requires: a cell that is half oxide and half silicon has no
single permittivity, and rounding the interface to the nearer node line would
move the oxide thickness by up to half a cell without saying so. t_ox is what
the accumulation capacitance is measured against, so that is not a rounding
error anyone would find later.

The x mesh is as coarse as it can be
------------------------------------
Nothing varies across the device, so three columns is enough and the answer
must not depend on how many there are. That is a test rather than an
assumption, and it is a real check of the 2D assembly: if the horizontal faces
carried the wrong area or the wrong permittivity, the columns would disagree.

What is not here
----------------
No fixed interface charge Q_f, which would shift flatband by -Q_f/C_ox, and no
poly depletion. This is the ideal capacitor of docs/01-physics.md.
"""

from __future__ import annotations

from ddsim.core import constants as C
from ddsim.device.builder import Device, Material, build_device
from ddsim.device.doping import Uniform
from ddsim.device.regions import stacked_regions
from ddsim.discretize.boundary import GateContact, OhmicPlate
from ddsim.mesh.mesh1d import graded_mesh_1d, stacked_mesh_1d, uniform_mesh_1d
from ddsim.mesh.mesh2d import tensor_mesh_2d

GATE = "gate"
"""Terminal name of the gate."""

BODY = "body"
"""Terminal name of the substrate contact."""


def mos_cap(
    substrate_doping: float = -1e16,
    t_ox: float = 1e-6,
    t_si: float = 2e-4,
    width: float = 1e-5,
    nx: int = 3,
    n_silicon: int = 121,
    n_oxide: int = 5,
    h_min: float = 5e-8,
    gate_voltage: float = 0.0,
    body_voltage: float = 0.0,
    work_function: float = C.PHI_M_N_POLY,
    material: Material | None = None,
) -> Device:
    """An ideal MOS capacitor, silicon at the bottom and gate metal on top.

    Args:
        substrate_doping: net doping of the substrate [cm^-3], negative for
            p-type. The sign convention is net doping everywhere in this
            codebase, so -1e16 is the ordinary NMOS body.
        t_ox: oxide thickness [cm]. 1e-6 is 10 nm.
        t_si: silicon thickness [cm]. Several times the maximum depletion
            width, or the body contact holds the depletion region open.
        width: device width [cm]. Nothing depends on it; the capacitance is
            per unit area.
        nx: node count across the device, at least 2.
        n_silicon: node count through the silicon, including the interface.
        n_oxide: node count through the oxide, including the interface.
        h_min: mesh spacing at the silicon surface [cm]. 5e-8 is 0.5 nm, which
            resolves an inversion layer.
        gate_voltage: bias on the gate [V].
        body_voltage: bias on the substrate contact [V].
        work_function: work function of the gate metal [eV]. Defaults to n+
            poly, the ordinary NMOS gate.
        material: defaults to silicon at 300 K.

    The body contact is a plate over the whole bottom edge rather than a point.
    A point would leave the rest of that boundary reflecting, which is a
    different device: its bottom edge would impose dpsi/dy = 0 everywhere
    except at one node, and it does not have the same solution.
    """
    if t_ox <= 0.0:
        raise ValueError(f"t_ox must be positive, got {t_ox}")
    if t_si <= 0.0:
        raise ValueError(f"t_si must be positive, got {t_si}")
    if n_silicon < 2 or n_oxide < 2:
        raise ValueError(
            f"each layer needs at least 2 nodes, got n_silicon={n_silicon} "
            f"and n_oxide={n_oxide}. A layer with one node has no thickness "
            "to carry a field across."
        )

    silicon = graded_mesh_1d(
        length=t_si, n_nodes=n_silicon, refine_at=t_si, h_min=h_min
    )
    oxide = uniform_mesh_1d(length=t_ox, n_nodes=n_oxide)
    mesh = tensor_mesh_2d(
        uniform_mesh_1d(length=width, n_nodes=nx),
        stacked_mesh_1d(silicon, oxide),
    )
    regions = stacked_regions(mesh, interface_y=t_si)

    contacts = (
        OhmicPlate(
            name=BODY,
            nodes=tuple(mesh.node_at(i, 0) for i in range(mesh.nx)),
            voltage=body_voltage,
        ),
        GateContact(
            name=GATE,
            nodes=tuple(mesh.node_at(i, mesh.ny - 1) for i in range(mesh.nx)),
            voltage=gate_voltage,
            work_function=work_function,
        ),
    )

    return build_device(
        mesh=mesh,
        doping=Uniform(substrate_doping),
        contacts=contacts,
        material=material,
        regions=regions,
    )
