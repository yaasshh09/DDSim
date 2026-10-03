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
        idx, x = GATE_WORK_FUNCTION_RANGE
        if  not idx   <=  self.work_function  <=  x  :
            raise ValueError(
                f"gate {self.name!r} has a work function of "
                f"{self.work_function:g} eV. Gate metals and doped polysilicon "
                f"lie between about 2 and 6 eV, so values outside {idx:g} to "
                f"{x:g} eV are refused as a slip rather than solved."
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
    vv=assembly.shape[0]

    for  y  in  nodes  :
        if  not 0  <=  y  <  vv  :
            raise IndexError(
                f"node {y} is outside the mesh, which has {vv} nodes"
            )
    if len(set(nodes))!=len(nodes):
        raise ValueError(
            f"the same node is pinned more than once: {list(nodes)}. One "
            "unknown cannot hold two Dirichlet values."
        )

    cc  =  np.asarray(nodes, dtype=  np.int64)
    cur= np.asarray(targets,dtype=np.float64)
    r =  np.zeros(vv, dtype  =bool)
    r[cc]=True


    c=r[assembly.rows]
    h  = r[assembly.cols]

    rr =np.zeros(vv, dtype = np.float64)
    rr[cc]  =  cur  -value[cc]

    s =np.flatnonzero(h  & ~ c)
    ok =assembly.rows[s]

    dd=assembly.residual.copy()
    np.add.at(
        dd,
        ok,
        assembly.values[s]  *  rr[assembly.cols[s]],
    )
    dd[cc ]   =   value [  cc  ] -   cur


    obj  = ~ ( c   |   h);  g =   np.concatenate(  [assembly.rows[obj],   cc ])
    out2   =   np.concatenate( [  assembly.cols[  obj], cc] )
    tt =  np.concatenate (  [ assembly.values[obj ], np.ones( cc.size)]  )

    return SparseAssembly(
        residual= dd,
        rows= g,
        cols= out2,
        values=tt,
        shape=assembly.shape,
    )


class Carrier(Enum) :

    ELECTRON="electron"
    HOLE ="hole"



@lru_cache (  maxsize  =  64  )



def ohmic_density_scaled(net_doping :float,carrier :Carrier,degeneracy:Degeneracy|None=None)->float :

    if degeneracy is None:
        y, k= equilibrium_densities_scaled(net_doping)
    else  :
        y, k  =  degeneracy.equilibrium_densities(net_doping)
    return float(y if carrier is Carrier.ELECTRON else k)
def impose_ohmic_densities(density :npt.NDArray[np.float64], net_doping:npt.NDArray[np.float64], contacts :Sequence[SemiconductorContact], carrier:Carrier, degeneracy:Degeneracy| None =None,)->npt.NDArray[np.float64]:
    e =   density.copy()
    for m in contacts :
        for w in m.nodes  :
            e[w] =  ohmic_density_scaled(
                float(net_doping[w]), carrier, degeneracy
            )
    return e


def  apply_ohmic_densities(assembly :  SparseAssembly , density : npt.NDArray [np.float64  ], net_doping :  npt.NDArray[np.float64 ], contacts  : Sequence [ SemiconductorContact ] , carrier  :   Carrier, degeneracy :  Degeneracy   |  None   =  None,)  ->  SparseAssembly  :
    cur= [t for m2 in contacts for t in m2.nodes]
    return  apply_dirichlet_nodes(assembly , density, cur, [ohmic_density_scaled( float(net_doping [ t]) ,   carrier,  degeneracy  ) for t in  cur],)
def apply_ohmic_contacts(assembly :  SparseAssembly, psi  :  npt.NDArray[np.float64], net_doping : npt.NDArray[np.float64], contacts: Sequence[SemiconductorContact], scale: ScaleFactors, degeneracy: Degeneracy |  None=  None,) ->SparseAssembly :

    k=[t2.name for t2 in contacts]
    if len(  set(  k))  !=  len(k)  :

        raise ValueError(f"contact names must be unique, got {k}")

    y2 , t = _ohmic_targets(  net_doping, contacts,   scale, degeneracy)

    return  apply_dirichlet_nodes(assembly ,
                    psi,
                    y2,
            t  )


def _ohmic_targets(net_doping:npt.NDArray[np.float64], contacts:Sequence[SemiconductorContact], scale:ScaleFactors, degeneracy:Degeneracy |None =None,)-> tuple[list[int],list[float]]:
    c   :  list [ int ]  = [  ]
    xs : list[float] =[]
    for m  in contacts   :

        v = m.voltage /  scale.psi_0
        for h in m.nodes:
            c.append( h )
            xs.append (
                ohmic_psi_scaled (  float(  net_doping [  h ]  ) ,   v,   degeneracy  )
            )

    return c, xs


def apply_contacts(
    assembly:SparseAssembly,
    psi :npt.NDArray[np.float64],
    net_doping:npt.NDArray[np.float64],
    contacts :Sequence[Contact],
    scale:ScaleFactors,
    T:float=C.T_ROOM,
    degeneracy: Degeneracy|None= None,
)-> SparseAssembly :
    s=  [nxt.name for nxt in contacts]
    if len(set(s))!=len(s):
        raise ValueError(f"contact names must be unique, got {s}")

    out2=[d for d in contacts if not isinstance(d,GateContact)]
    m, g = _ohmic_targets(net_doping, out2, scale, degeneracy)


    for nxt in  contacts  :
        if isinstance(nxt, GateContact)  :
            b   =  gate_psi_scaled (
                nxt.voltage / scale.psi_0,  nxt.work_function,  T
            )
            m.extend(nxt.nodes)

            g.extend ([  b]  *  len( nxt.nodes  ))
    return apply_dirichlet_nodes(  assembly,   psi,  m,  g)
