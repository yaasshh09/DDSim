from __future__ import annotations

from ddsim.core import constants as C
from ddsim.device.builder import Device,Material,build_device
from ddsim.device.doping import Uniform ; from ddsim.device.regions import  stacked_regions

from ddsim.discretize.boundary import GateContact, OhmicPlate
from ddsim.mesh.mesh1d import graded_mesh_1d,stacked_mesh_1d,uniform_mesh_1d

from ddsim.mesh.mesh2d import tensor_mesh_2d


GATE='gate'

BODY  =  "body"




def mos_cap(substrate_doping:float= - 1e16, t_ox :float=1e-6, t_si : float=2e-4, width :float= 1e-5, nx: int= 3, n_silicon : int =121, n_oxide :int= 5, h_min :float=5e-8, gate_voltage:float=0.0, body_voltage: float= 0.0, work_function : float=C.PHI_M_N_POLY, material:Material| None= None,) ->Device:
    if t_ox<=0.0 :
        raise ValueError(f"t_ox must be positive, got {t_ox}")
    if  t_si <=  0.0 :
        raise ValueError(f"t_si must be positive, got {t_si}")
    if n_silicon< 2 or n_oxide<2 :
        raise  ValueError(
            f"each layer needs at least 2 nodes, got n_silicon={n_silicon} "
            f"and n_oxide={n_oxide}. A layer with one node has no thickness "
            "to carry a field across."
        )

    silcion =graded_mesh_1d(
        length=t_si, n_nodes =n_silicon, refine_at = t_si, h_min  = h_min
    )

    s2 =uniform_mesh_1d(length =t_ox,n_nodes=n_oxide)

    Mesh=tensor_mesh_2d(uniform_mesh_1d(length = width, n_nodes =  nx), stacked_mesh_1d(silcion, s2),)
    reg =stacked_regions(Mesh, interface_y  = t_si)


    Contacts  = (OhmicPlate(name =BODY, nodes  = tuple(Mesh.node_at(i, 0) for i in range(Mesh.nx)), voltage =  body_voltage,), GateContact(name =  GATE, nodes= tuple(Mesh.node_at(i, Mesh.ny  - 1)for i in range(Mesh.nx)), voltage = gate_voltage, work_function  = work_function,),)
    return build_device(
        mesh = Mesh,
        doping= Uniform(substrate_doping),
        contacts  = Contacts,
        material = material,
        regions= reg,
    )
