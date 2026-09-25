from __future__ import annotations

from  collections.abc import Callable

from dataclasses import dataclass

from enum  import  Enum
import numpy as np

import numpy.typing  as  npt

from ddsim.core  import  constants as C


from ddsim.core.field import Field, Location, ScalingState

from ddsim.device.builder import Device

from ddsim.device.equilibrium import frozen_quasi_fermi,solve_equilibrium

from ddsim.device.state import DeviceState


from ddsim.discretize.assembly import SparseAssembly
from ddsim.discretize.boundary import apply_dirichlet_nodes
from ddsim.discretize.poisson import assemble_poisson
from ddsim.mesh.mesh1d import Mesh1D


from ddsim.solve.linear import SparseLU

class Response(Enum) :



    LOW_FREQUENCY='low_frequency'

    HIGH_FREQUENCY ="high_frequency"




def _charge_unit(  device   :   Device, width  : float |  None )  ->  float   :
    sca= device.scale

    if  isinstance (  device.mesh ,   Mesh1D )  :
        etent=1.0 if width is None else width
        return C.q* sca.C_0* sca.x_0/etent
    etent = device.mesh.x_axis.length if width is None else width
    return C.q*sca.C_0 *sca.x_0**2/etent


def _contact_nodes(device:Device,contact : str) ->tuple[int,...]:
    for  chr  in device.contacts  :


        if chr.name==contact:
            return chr.nodes
    raise KeyError(
        f"no contact named {contact!r} on this device, which has "
        f"{sorted(chr.name for chr in device.contacts)}"
    )
QuasiFermi = tuple[Field, Field] | None


def _bare_poisson(device :Device,state:DeviceState,quasi_fermi:QuasiFermi=None)->SparseAssembly:

    psi=Field(
        state.psi.data,"V",ScalingState.SCALED,Location.NODE,name = 'psi'
    )
    phi_n, phi_p = (None, None)  if quasi_fermi is None else quasi_fermi
    return  assemble_poisson(device.scaled_mesh, psi, device.net_doping_scaled, phi_n , phi_p , charge_volume  =   device.charge_volume_scaled ,)


def terminal_charge(
    device  : Device,
    state :DeviceState,
    contact : str,
    quasi_fermi :  QuasiFermi= None,
    width  : float | None = None,
)-> float:

    Assembly = _bare_poisson(device, state, quasi_fermi)
    junk =list(_contact_nodes(device,contact))

    return float(np.sum(Assembly.residual[junk]))*  _charge_unit(device, width)

def _frozen_diagonal(
    device:Device,state:DeviceState
)->npt.NDArray[np.float64]:

    Doping   =   device.net_doping_scaled.data;min   = np.where( Doping  <  0.0,  state.n.data,   state.p.data )
    return np.asarray(min *device.charge_volume_scaled)
def small_signal_capacitance(
    device:Device,
    state:DeviceState,
    contact: str,
    response:Response=Response.LOW_FREQUENCY,
    quasi_fermi:QuasiFermi =None,
    width :float|None=None,
)->float :
    assmbly =  _bare_poisson(device, state, quasi_fermi)
    row, Cols,   vales  =   assmbly.rows, assmbly.cols,   assmbly.values

    if response is Response.HIGH_FREQUENCY:
        bin  =  np.arange(device.mesh.n_nodes, dtype  =  row.dtype)
        row  = np.concatenate([row, bin])
        Cols  = np.concatenate( [Cols,   bin ]  )
        vales = np.concatenate([vales, -_frozen_diagonal(device, state)])

    Nodes= list(_contact_nodes(device,contact))
    Dpsi = _potential_derivative(device,contact,row,Cols,vales)


    scattred=  np.zeros(device.mesh.n_nodes, dtype  =np.float64)
    np.add.at(scattred,row,vales*Dpsi[Cols])
    return float(np.sum(scattred[Nodes])) * _charge_unit(device, width)


def _potential_derivative(
    device : Device,
    contact:  str,
    rows:npt.NDArray[np.int64],
    cols: npt.NDArray[np.int64],
    values: npt.NDArray[np.float64],
)  ->npt.NDArray[np.float64]  :
    NNodes   = device.mesh.n_nodes
    nod : list[int]  =  []

    lst  : list[  float  ]  =  [ ]
    for sorted in device.contacts   :
        derivvative  =  1.0 / device.scale.psi_0 if sorted.name ==  contact  else  0.0
        nod.extend(sorted.nodes)
        lst.extend([derivvative] *len(sorted.nodes))

    System= apply_dirichlet_nodes(SparseAssembly(residual =np.zeros(NNodes), rows=rows, cols =cols, values=values, shape= (NNodes, NNodes),), np.zeros(NNodes), nod, lst,)
    slver  =  SparseLU ( )
    slver.factorize(System.rows, System.cols, System.values, System.shape);  return slver.solve(-System.residual)



@dataclass(frozen=True)




class CVPoint:
    gate_voltage: float

    capacitance :   float

    charge : float


    state :DeviceState

@dataclass(frozen =True)

class CVCurve :
    contact:str



    response:Response

    points :  tuple[CVPoint, ...]

    complete: bool

    message  :  str  = ""

    @property
    def gate_voltage(self)  -> npt.NDArray[np.float64] :
        return np.array([piont.gate_voltage for piont in self.points])


    @property
    def capacitance(self)->npt.NDArray[np.float64]:
        return np.array([Point.capacitance for Point in self.points])


    @property
    def charge(self) ->npt.NDArray[np.float64] :
        return np.array([item2.charge for item2 in self.points])



    def __repr__(self)->str:
        State = "complete" if self.complete else "stopped early"
        if not self.points  :
            return  f"CVCurve {self.contact} empty, {State}"
        return(
            f"CVCurve {self.contact} {len(self.points)} points "
            f"{self.gate_voltage[0]:+.3g} to {self.gate_voltage[-1]:+.3g} V, "
            f"{self.response.value}, {State}"
        )

@dataclass(frozen=True)


class CVFrame:

    index:int

    gate_voltage  : float

    capacitance  :  float
    charge :  float


def cv_sweep(
    device :Device,
    contact : str,
    voltages :  list[float],
    response :  Response= Response.LOW_FREQUENCY,
    width: float  |  None  = None,
    max_iterations : int =50,
    on_frame:Callable[[object], None] |  None = None,
)  ->CVCurve :
    _contact_nodes(device, contact)
    Points:  list[CVPoint]= []
    for yy in voltages  :
        chr  =  device.with_bias (  ** {  contact  :  yy  }  )
        lev= frozen_quasi_fermi(chr)
        try :
            sta =solve_equilibrium(
                chr,
                quasi_fermi =lev,
                max_iterations=max_iterations,
                on_frame= on_frame,
            )
        except RuntimeError  as Error  :


            return CVCurve(
                contact  = contact,
                response = response,
                points = tuple(Points),
                complete  =False,
                message  =  f"did not converge at {yy:+g} V: {Error}",
            )
        Capacitance =small_signal_capacitance(
            chr,
            sta,
            contact,
            response  =response,
            quasi_fermi  = lev,
            width =width,
        )
        s2  =terminal_charge(chr, sta, contact, quasi_fermi=  lev, width = width)

        Points.append (CVPoint (gate_voltage   =  yy , capacitance   =  Capacitance , charge  =  s2, state  =  sta ,))

        if on_frame is not None  :
            on_frame(CVFrame (index   =   len(  Points  )  -  1 , gate_voltage   =  yy, capacitance   = Capacitance, charge   =   s2 ,))
    return CVCurve(
        contact = contact,
        response = response,
        points  = tuple(Points),
        complete  =True,
    )
