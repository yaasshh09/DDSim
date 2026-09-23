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
            Range 1e14 to 1e19, log. Below 1e14 the p side is close to
            intrinsic at 300 K, and above 1e18 the closed form V_bi this
            device is checked against degrades, as the module docstring says.
        Nd: donor concentration on the n side [cm^-3], positive.
            Range 1e14 to 1e19, log. The same ends and the same reason as Na.
        length: device length [cm]. Range 1e-5 to 1e-3, log. 0.1 um to 10 um,
            from a few Debye lengths of quasi-neutral material on each side
            of the junction out to the long base limit.
        junction: junction position [cm], inside length.
            Range 1e-5 to 9e-5. A junction outside the mesh is refused by
            graded_mesh_1d rather than clamped here: the pair this makes with
            length belongs to whoever set them, and a silent clamp would solve
            a device nobody asked for.
        n_nodes: mesh node count [1]. Range 51 to 1001. The low end is coarse
            enough to show what too few nodes across the depletion region do
            to the answer, which is worth being able to see.
        h_min: mesh spacing at the junction [cm]. Range 1e-8 to 1e-6, log.
            Half of the 41 nm Debye length at 1e16 sits in the middle of this.
        anode_voltage: bias on the p side contact [V]. Range -5 to 1. Above
            roughly 1.3 V the equilibrium guess a cold solve starts from stops
            converging on this device, see docs/07-decisions.md.
        cathode_voltage: bias on the n side contact [V]. Range -5 to 1. The
            same ends as the anode.
        material: defaults to silicon at 300 K.
    """
    if not 0.0 < junction < length:
        raise ValueError(
            f"the junction must sit inside the device, got junction={junction:g} "
            f"cm in a device {length:g} cm long"
        )
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
