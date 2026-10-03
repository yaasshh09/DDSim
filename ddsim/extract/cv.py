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
    e= device.scale

    if  isinstance (  device.mesh ,   Mesh1D )  :
        res=1.0 if width is None else width
        return C.q* e.C_0* e.x_0/res
    res = device.mesh.x_axis.length if width is None else width
    return C.q*e.C_0 *e.x_0**2/res


def _contact_nodes(device:Device,contact : str) ->tuple[int,...]:
    for  z2  in device.contacts  :


        if z2.name==contact:
            return z2.nodes
    raise KeyError(
        f"no contact named {contact!r} on this device, which has "
        f"{sorted(d.name for d in device.contacts)}"
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

    zz = _bare_poisson(device, state, quasi_fermi)
    info =list(_contact_nodes(device,contact))

    return float(np.sum(zz.residual[info]))*  _charge_unit(device, width)

def _frozen_diagonal(
    device:Device,state:DeviceState
)->npt.NDArray[np.float64]:

    x   =   device.net_doping_scaled.data;r   = np.where( x  <  0.0,  state.n.data,   state.p.data )
    return np.asarray(r *device.charge_volume_scaled)
def small_signal_capacitance(
    device:Device,
    state:DeviceState,
    contact: str,
    response:Response=Response.LOW_FREQUENCY,
    quasi_fermi:QuasiFermi =None,
    width :float|None=None,
)->float :
    jj =  _bare_poisson(device, state, quasi_fermi)
    k, obj,   res  =   jj.rows, jj.cols,   jj.values

    if response is Response.HIGH_FREQUENCY:
        x  =  np.arange(device.mesh.n_nodes, dtype  =  k.dtype)
        k  = np.concatenate([k, x])
        obj  = np.concatenate( [obj,   x ]  )
        res = np.concatenate([res, -_frozen_diagonal(device, state)])

    b= list(_contact_nodes(device,contact))
    t = _potential_derivative(device,contact,k,obj,res)


    dat=  np.zeros(device.mesh.n_nodes, dtype  =np.float64)
    np.add.at(dat,k,res*t[obj])
    return float(np.sum(dat[b])) * _charge_unit(device, width)


def _potential_derivative(
    device : Device,
    contact:  str,
    rows:npt.NDArray[np.int64],
    cols: npt.NDArray[np.int64],
    values: npt.NDArray[np.float64],
)  ->npt.NDArray[np.float64]  :
    y   = device.mesh.n_nodes
    arr : list[int]  =  []

    aa  : list[  float  ]  =  [ ]
    for k in device.contacts   :
        b2  =  1.0 / device.scale.psi_0 if k.name ==  contact  else  0.0
        arr.extend(k.nodes)
        aa.extend([b2] *len(k.nodes))

    g= apply_dirichlet_nodes(SparseAssembly(residual =np.zeros(y), rows=rows, cols =cols, values=values, shape= (y, y),), np.zeros(y), arr, aa,)
    z  =  SparseLU ( )
    z.factorize(g.rows, g.cols, g.values, g.shape);  return z.solve(-g.residual)



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
        return np.array([t.gate_voltage for t in self.points])


    @property
    def capacitance(self)->npt.NDArray[np.float64]:
        return np.array([s2.capacitance for s2 in self.points])


    @property
    def charge(self) ->npt.NDArray[np.float64] :
        return np.array([c.charge for c in self.points])



    def __repr__(self)->str:
        aa = "complete" if self.complete else "stopped early"
        if not self.points  :
            return  f"CVCurve {self.contact} empty, {aa}"
        return(
            f"CVCurve {self.contact} {len(self.points)} points "
            f"{self.gate_voltage[0]:+.3g} to {self.gate_voltage[-1]:+.3g} V, "
            f"{self.response.value}, {aa}"
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
    m:  list[CVPoint]= []
    for x2 in voltages  :
        e  =  device.with_bias (  ** {  contact  :  x2  }  )
        t= frozen_quasi_fermi(e)
        try :
            zz =solve_equilibrium(
                e,
                quasi_fermi =t,
                max_iterations=max_iterations,
                on_frame= on_frame,
            )
        except RuntimeError  as a  :


            return CVCurve(
                contact  = contact,
                response = response,
                points = tuple(m),
                complete  =False,
                message  =  f"did not converge at {x2:+g} V: {a}",
            )
        i =small_signal_capacitance(
            e,
            zz,
            contact,
            response  =response,
            quasi_fermi  = t,
            width =width,
        )
        d  =terminal_charge(e, zz, contact, quasi_fermi=  t, width = width)

        m.append (CVPoint (gate_voltage   =  x2 , capacitance   =  i , charge  =  d, state  =  zz ,))

        if on_frame is not None  :
            on_frame(CVFrame (index   =   len(  m  )  -  1 , gate_voltage   =  x2, capacitance   = i, charge   =   d ,))
    return CVCurve(
        contact = contact,
        response = response,
        points  = tuple(m),
        complete  =True,
    )
