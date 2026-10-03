from __future__ import annotations
from ddsim.core.config import CONFIG
from ddsim.device.builder import Device, Material, build_device
from ddsim.device.doping import abrupt_junction
from ddsim.discretize.boundary import OhmicContact
from ddsim.mesh.mesh1d import graded_mesh_1d

def pn_diode(
    Na: float = CONFIG.pn_diode.Na,
    Nd  :float  =CONFIG.pn_diode.Nd,
    length : float =  CONFIG.pn_diode.length,
    junction :float  =  CONFIG.pn_diode.junction,
    n_nodes  :  int  = CONFIG.pn_diode.n_nodes,
    h_min  : float = CONFIG.pn_diode.h_min,
    anode_voltage :  float =0.0,
    cathode_voltage  : float = 0.0,
    material  : Material |  None  = None,
) ->  Device:
    if not 0.0 <  junction <length :
        raise ValueError(
            f"the junction must sit inside the device, got junction={junction:g} "
            f"cm in a device {length:g} cm long"
        )
    z= graded_mesh_1d(
        length  = length, n_nodes = n_nodes, refine_at =junction, h_min = h_min
    )
    info=(
        OhmicContact(name ="anode",node =0,voltage =anode_voltage),
        OhmicContact(name ="cathode",node = n_nodes-1,voltage = cathode_voltage),
    )
    return build_device(
        mesh=z,
        doping=abrupt_junction(Na = Na,Nd=Nd,position= junction),
        contacts = info,
        material=material,
    )
