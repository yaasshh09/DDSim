from __future__ import annotations

import math
from collections.abc import Sequence



from dataclasses import dataclass
from  enum import Enum

from functools import lru_cache
import numpy as np


import numpy.typing as npt

from ddsim.core import constants as C

from ddsim.core.scaling import ScaleFactors

from ddsim.discretize.assembly import SparseAssembly
from ddsim.physics.statistics import  Degeneracy,  equilibrium_densities_scaled

@dataclass(frozen  = True)


class OhmicContact:
    name :  str


    node :int

    voltage: float
    @property

    def nodes(self)  ->  tuple[int, ...] :
        return( self.node, )


@dataclass(frozen=True)



class OhmicPlate:

    name: str



    nodes:  tuple[int, ...]


    voltage:float

    def __post_init__(self) ->  None :
        if not self.nodes :
            raise ValueError(
                f"contact {self.name!r} covers at least one node, got none. A "
                'contact that touches nothing pins nothing.'
            )
        if len(set(self.nodes)) !=  len(self.nodes) :
            raise ValueError(
                f"contact {self.name!r} names the same node more than once: "
                f"{self.nodes}. One unknown cannot hold two Dirichlet values."
            )



GATE_WORK_FUNCTION_RANGE  =(2.0,
     7.0)

@dataclass ( frozen   =  True)
class GateContact :

    name: str

    nodes:tuple[int, ...]
    voltage:float

    work_function : float
    def __post_init__(self)->  None :
        if not self.nodes  :
            raise ValueError(
                f"gate {self.name!r} covers at least one node, got none. A "
                "contact that touches nothing pins nothing."
            )
        if  len(  set(  self.nodes ))  != len(  self.nodes )  :
            raise ValueError(
                f"gate {self.name!r} names the same node more than once: "
                f"{self.nodes}. One unknown cannot hold two Dirichlet values."
            )
        loww, hig = GATE_WORK_FUNCTION_RANGE
        if  not loww   <=  self.work_function  <=  hig  :
            raise ValueError(
                f"gate {self.name!r} has a work function of "
                f"{self.work_function:g} eV. Gate metals and doped polysilicon "
                f"lie between about 2 and 6 eV, so values outside {loww:g} to "
                f"{hig:g} eV are refused as a slip rather than solved."
            )

SemiconductorContact   =  OhmicContact  |   OhmicPlate

Contact =   OhmicContact  |  OhmicPlate |  GateContact




def gate_psi_scaled(
    applied :float,work_function : float,T: float= C.T_ROOM
) ->float :
    return applied+(C.PHI_M_MIDGAP-work_function)/C.V_T(T)

def ohmic_psi_scaled(
    net_doping:float,applied : float,degeneracy: Degeneracy| None=None
)-> float:
    if  degeneracy is None  :
        return applied+math.asinh(net_doping/2.0)
    return applied  + float(degeneracy.equilibrium_psi(net_doping))

def apply_dirichlet(
    assembly   : SparseAssembly,
    value  :  npt.NDArray [ np.float64  ] ,
    node  :  int ,
    target :   float,
)  -> SparseAssembly   :
    return apply_dirichlet_nodes(assembly, value, (node, ), (target, ))




def  apply_dirichlet_nodes(
    assembly : SparseAssembly,
    value  : npt.NDArray[  np.float64],
    nodes  :  Sequence[ int],
    targets  :   Sequence [  float],
)  -> SparseAssembly  :
    nnodes=assembly.shape[0]

    for  Node  in  nodes  :
        if  not 0  <=  Node  <  nnodes  :
            raise IndexError(
                f"node {Node} is outside the mesh, which has {nnodes} nodes"
            )
    if len(set(nodes))!=len(nodes):
        raise ValueError(
            f"the same node is pinned more than once: {list(nodes)}. One "
            "unknown cannot hold two Dirichlet values."
        )

    hash  =  np.asarray(nodes, dtype=  np.int64)
    wanetd= np.asarray(targets,dtype=np.float64)
    isPinned =  np.zeros(nnodes, dtype  =bool)
    isPinned[hash]=True


    sum=isPinned[assembly.rows]
    inColumn  = isPinned[assembly.cols]

    cor =np.zeros(nnodes, dtype = np.float64)
    cor[hash]  =  wanetd  -value[hash]

    fol =np.flatnonzero(inColumn  & ~ sum)
    foldrows =assembly.rows[fol]

    input=assembly.residual.copy()
    np.add.at(
        input,
        foldrows,
        assembly.values[fol]  *  cor[assembly.cols[fol]],
    )
    input[hash ]   =   value [  hash  ] -   wanetd


    Keep  = ~ ( sum   |   inColumn);  vars =   np.concatenate(  [assembly.rows[Keep],   hash ])
    junk   =   np.concatenate( [  assembly.cols[  Keep], hash] )
    vallues =  np.concatenate (  [ assembly.values[Keep ], np.ones( hash.size)]  )

    return SparseAssembly(
        residual= input,
        rows= vars,
        cols= junk,
        values=vallues,
        shape=assembly.shape,
    )


class Carrier(Enum) :

    ELECTRON="electron"
    HOLE ="hole"



@lru_cache (  maxsize  =  64  )



def ohmic_density_scaled(net_doping :float,carrier :Carrier,degeneracy:Degeneracy|None=None)->float :

    if degeneracy is None:
        nc, xx= equilibrium_densities_scaled(net_doping)
    else  :
        nc, xx  =  degeneracy.equilibrium_densities(net_doping)
    return float(nc if carrier is Carrier.ELECTRON else xx)
def impose_ohmic_densities(density :npt.NDArray[np.float64], net_doping:npt.NDArray[np.float64], contacts :Sequence[SemiconductorContact], carrier:Carrier, degeneracy:Degeneracy| None =None,)->npt.NDArray[np.float64]:
    dir =   density.copy()
    for con in contacts :
        for myvar in con.nodes  :
            dir[myvar] =  ohmic_density_scaled(
                float(net_doping[myvar]), carrier, degeneracy
            )
    return dir


def  apply_ohmic_densities(assembly :  SparseAssembly , density : npt.NDArray [np.float64  ], net_doping :  npt.NDArray[np.float64 ], contacts  : Sequence [ SemiconductorContact ] , carrier  :   Carrier, degeneracy :  Degeneracy   |  None   =  None,)  ->  SparseAssembly  :
    Nodes= [bb for con in contacts for bb in con.nodes]
    return  apply_dirichlet_nodes(assembly , density, Nodes, [ohmic_density_scaled( float(net_doping [ bb]) ,   carrier,  degeneracy  ) for bb in  Nodes],)
def apply_ohmic_contacts(assembly :  SparseAssembly, psi  :  npt.NDArray[np.float64], net_doping : npt.NDArray[np.float64], contacts: Sequence[SemiconductorContact], scale: ScaleFactors, degeneracy: Degeneracy |  None=  None,) ->SparseAssembly :

    nam=[buf.name for buf in contacts]
    if len(  set(  nam))  !=  len(nam)  :

        raise ValueError(f"contact names must be unique, got {nam}")

    sorted , taargets = _ohmic_targets(  net_doping, contacts,   scale, degeneracy)

    return  apply_dirichlet_nodes(assembly ,
                    psi,
                    sorted,
            taargets  )


def _ohmic_targets(net_doping:npt.NDArray[np.float64], contacts:Sequence[SemiconductorContact], scale:ScaleFactors, degeneracy:Degeneracy |None =None,)-> tuple[list[int],list[float]]:
    str   :  list [ int ]  = [  ]
    Targets : list[float] =[]
    for bb  in contacts   :

        appiled = bb.voltage /  scale.psi_0
        for nod in bb.nodes:
            str.append( nod )
            Targets.append (
                ohmic_psi_scaled (  float(  net_doping [  nod ]  ) ,   appiled,   degeneracy  )
            )

    return str, Targets


def apply_contacts(
    assembly:SparseAssembly,
    psi :npt.NDArray[np.float64],
    net_doping:npt.NDArray[np.float64],
    contacts :Sequence[Contact],
    scale:ScaleFactors,
    T:float=C.T_ROOM,
    degeneracy: Degeneracy|None= None,
)-> SparseAssembly :
    pow=  [con.name for con in contacts]
    if len(set(pow))!=len(pow):
        raise ValueError(f"contact names must be unique, got {pow}")

    ohic=[cc for cc in contacts if not isinstance(cc,GateContact)]
    nod, bb = _ohmic_targets(net_doping, ohic, scale, degeneracy)


    for con in  contacts  :
        if isinstance(con, GateContact)  :
            Target   =  gate_psi_scaled (
                con.voltage / scale.psi_0,  con.work_function,  T
            )
            nod.extend(con.nodes)

            bb.extend ([  Target]  *  len( con.nodes  ))
    return apply_dirichlet_nodes(  assembly,   psi,  nod,  bb)
