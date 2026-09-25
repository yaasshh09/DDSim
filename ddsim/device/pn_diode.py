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
    Nd  :float  =1e16,
    length : float =  1e-4,
    junction :float  =  0.5e-4,
    n_nodes  :  int  = 201,
    h_min  : float = 1e-7,
    anode_voltage :  float =0.0,
    cathode_voltage  : float = 0.0,
    material  : Material |  None  = None,
) ->  Device:
    '''An abrupt PN junction diode, p-type on the left.

    Args:
        Na: how many acceptors per cubic centimetre on the p side [cm^-3].
            Range 1e14 to 1e19, log. Below 1e14 the p side is barely doped at
            all, and above 1e18 the textbook built in potential it's checked
            against starts to drift.
        Nd: how many donors per cubic centimetre on the n side [cm^-3].
            Range 1e14 to 1e19, log. Same limits as Na, for the same reasons.
        length: how long the diode is [cm]. Range 1e-5 to 1e-3, log. That's
            0.1 um to 10 um, from just enough room either side of the junction
            up to a long diode.
        junction: where the p side ends and the n side starts [cm], measured
            from the left and inside the length. Range 1e-5 to 9e-5.
        n_nodes: how many mesh points the diode is cut into [1].
            Range 51 to 1001. The low end is deliberately coarse, so you can
            see what too few points across the depletion region does.
        h_min: the smallest mesh spacing, right at the junction [cm].
            Range 1e-8 to 1e-6, log. The default of 1 nm easily resolves a
            1e16 junction.
        anode_voltage: voltage on the p side contact [V]. Range -5 to 1.
            Much above 1.3 V forward, the solver can't find a starting guess
            for this device.
        cathode_voltage: voltage on the n side contact [V]. Range -5 to 1.
            Same limits as the anode.
        material: defaults to silicon at 300 K.

    Why the ranges end where they do. Na and Nd stop at 1e18 to 1e19 because
    the closed form V_bi the tests hold this device to degrades there, as the
    module docstring says. length runs from a few Debye lengths of neutral
    material each side of the junction out to the long base limit. A junction
    outside the mesh is refused by graded_mesh_1d rather than clamped here:
    the pair it makes with length belongs to whoever set them, and a silent
    clamp would solve a device nobody asked for. Half of the 41 nm Debye
    length at 1e16 sits in the middle of the h_min range. The 1.3 V limit on
    a cold solve is in docs/07-decisions.md.
    '''
    if not 0.0 <  junction <length :
        raise ValueError(
            f"the junction must sit inside the device, got junction={junction:g} "
            f"cm in a device {length:g} cm long"
        )
    Mesh= graded_mesh_1d(
        length  = length, n_nodes = n_nodes, refine_at =junction, h_min = h_min
    )
    coontacts=(
        OhmicContact(name ="anode",node =0,voltage =anode_voltage),
        OhmicContact(name ="cathode",node = n_nodes-1,voltage = cathode_voltage),
    )
    return build_device(
        mesh=Mesh,
        doping=abrupt_junction(Na = Na,Nd=Nd,position= junction),
        contacts = coontacts,
        material=material,
    )
