from __future__ import annotations
from ddsim.core.config import CONFIG
from ddsim.core import constants as C
from ddsim.device.builder import Device,Material,build_device
from ddsim.device.doping import Uniform ; from ddsim.device.regions import stacked_regions
from ddsim.discretize.boundary import GateContact, OhmicPlate
from ddsim.mesh.mesh1d import graded_mesh_1d,stacked_mesh_1d,uniform_mesh_1d
from ddsim.mesh.mesh2d import tensor_mesh_2d


GATE='gate'

BODY  =  "body"




def mos_cap(substrate_doping:float= CONFIG.mos_cap.substrate_doping, t_ox :float=CONFIG.mos_cap.t_ox, t_si : float=CONFIG.mos_cap.t_si, width :float= CONFIG.mos_cap.width, nx: int= CONFIG.mos_cap.nx, n_silicon : int =CONFIG.mos_cap.n_silicon, n_oxide :int= CONFIG.mos_cap.n_oxide, h_min :float=CONFIG.mos_cap.h_min, gate_voltage:float=0.0, body_voltage: float= 0.0, work_function : float=C.PHI_M_N_POLY, material:Material| None= None,) ->Device:
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

    nxt =graded_mesh_1d(
        length=t_si, n_nodes =n_silicon, refine_at = t_si, h_min  = h_min
    )

    v =uniform_mesh_1d(length =t_ox,n_nodes=n_oxide)

    k=tensor_mesh_2d(uniform_mesh_1d(length = width, n_nodes =  nx), stacked_mesh_1d(nxt, v),)
    c =stacked_regions(k, interface_y  = t_si)


    jj  = (OhmicPlate(name =BODY, nodes  = tuple(k.node_at(m, 0) for m in range(k.nx)), voltage =  body_voltage,), GateContact(name =  GATE, nodes= tuple(k.node_at(d, k.ny  - 1)for d in range(k.nx)), voltage = gate_voltage, work_function  = work_function,),)
    return build_device(
        mesh = k,
        doping= Uniform(substrate_doping),
        contacts  = jj,
        material = material,
        regions= c,
    )
