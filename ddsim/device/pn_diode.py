"""An abrupt PN diode, the Phase 1 test device.

Defaults are the 1e16 / 1e16 junction that docs/04-validation.md names as the
clean analytic target, where V_bi = V_T*ln(Na*Nd/n_i^2) is a good
approximation. Above 1e18 degeneracy starts to matter and that expression
degrades.

The mesh is graded to the junction, because the depletion region is where all
the structure is. At 1e16 the Debye length is 41 nm and docs/02-numerics.md
wants the spacing below half of that.
"""

from __future__ import annotations

from ddsim.device.builder import Device, Material, build_device
from ddsim.device.doping import abrupt_junction
from ddsim.discretize.boundary import OhmicContact
from ddsim.mesh.mesh1d import graded_mesh_1d


def pn_diode(
    Na: float = 1e16,
    Nd: float = 1e16,
    length: float = 1e-4,
    junction: float = 0.5e-4,
    n_nodes: int = 201,
    h_min: float = 1e-7,
    anode_voltage: float = 0.0,
    cathode_voltage: float = 0.0,
    material: Material | None = None,
) -> Device:
    """An abrupt PN junction diode, p-type on the left.

    Args:
        Na: acceptor concentration on the p side [cm^-3], positive.
        Nd: donor concentration on the n side [cm^-3], positive.
        length: device length [cm].
        junction: junction position [cm].
        n_nodes: mesh node count.
        h_min: mesh spacing at the junction [cm].
        anode_voltage: bias on the p side contact [V].
        cathode_voltage: bias on the n side contact [V].
        material: defaults to silicon at 300 K.
    """
    mesh = graded_mesh_1d(
        length=length, n_nodes=n_nodes, refine_at=junction, h_min=h_min
    )
    contacts = (
        OhmicContact(name="anode", node=0, voltage=anode_voltage),
        OhmicContact(name="cathode", node=n_nodes - 1, voltage=cathode_voltage),
    )
    return build_device(
        mesh=mesh,
        doping=abrupt_junction(Na=Na, Nd=Nd, position=junction),
        contacts=contacts,
        material=material,
    )
