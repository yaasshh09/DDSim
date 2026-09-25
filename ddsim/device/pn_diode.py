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
