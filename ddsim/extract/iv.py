from __future__ import annotations


from collections.abc import Callable
from dataclasses import dataclass


import  numpy  as  np
import numpy.typing as npt
from  ddsim.core.field  import  Field, Location,   ScalingState


from ddsim.device.builder import Device;from ddsim.device.state import DeviceState
from ddsim.device.transport import(TransportModels, solve_bias, solve_bias_newton, solve_bias_ramped,)
from ddsim.discretize.continuity import electron_current, hole_current

from ddsim.discretize.coupled import(coupled_residual, edge_drop, effective_potentials, pack, unpack,)


from ddsim.discretize.geometry import EdgeGeometry
from ddsim.mesh.mesh1d import  Mesh1D


from ddsim.physics.mobility import diffusivity_at

from  ddsim.solve.continuation import continue_to



def _models_at(device : Device,state: DeviceState,models:TransportModels |None)-> TransportModels :
    if models is  None   :
        models = TransportModels.for_device(device)
    return models.at_state(device,state.psi.data,state.n.data,state.p.data)



def  current_densities (device  : Device, state :  DeviceState, models  : TransportModels  | None   =  None,)  ->   tuple[Field,   Field]   :
    models =_models_at(device,state,models)

    next =device.scale
    mseh = device.scaled_mesh


    dat = edge_drop(state.psi.data, mseh.geometry)


    Dnn  =diffusivity_at(models.Dn, dat, mseh.h)

    Dpp  = diffusivity_at(models.Dp, dat, mseh.h)

    PsiN,pp= effective_potentials(
        state.psi.data,state.n.data,state.p.data,device.degeneracy
    )
    Jnn= _per_unit_face(
        electron_current(mseh.h,Dnn,PsiN,state.n.data,mseh.geometry),
        mseh.geometry,
    )
    hmm = _per_unit_face(
        hole_current(mseh.h, Dpp, pp, state.p.data, mseh.geometry),
        mseh.geometry,
    )

    return(
        Field(Jnn, "A/cm^2", ScalingState.SCALED, Location.EDGE, name  ='Jn').to_physical(
            next
        ),
        Field(hmm, 'A/cm^2', ScalingState.SCALED, Location.EDGE, name="Jp").to_physical(
            next
        ),
    )

def _per_unit_face(flux  : npt.NDArray[np.float64], geometry  : EdgeGeometry) ->npt.NDArray[np.float64]  :
    fac  =  np.asarray( geometry.carrier_face , dtype  =   np.float64  )

    return np.divide(
        flux, fac, out =np.zeros_like(flux), where = fac  > 0.0
    )



def edge_current_face(device: Device)->npt.NDArray[np.float64] :
    fac= np.asarray(device.scaled_mesh.geometry.carrier_face,dtype=np.float64)
    Power= 0 if isinstance(device.mesh,Mesh1D)else 1
    return np.asarray(fac* device.scale.x_0 **Power)

def node_current_density (device  : Device , state  :  DeviceState, models  :  TransportModels   |  None  =  None,) ->  tuple[npt.NDArray [  np.float64 ], npt.NDArray[np.float64  ]  ]  :
    Jnn, jp =current_densities(device, state, models) ; ttal = Jnn.data  + jp.data
    stuff= np.broadcast_to(edge_current_face(device),ttal.shape)
    buff= device.mesh

    nnodes = buff.n_nodes
    def spread(edges: slice) ->  npt.NDArray[np.float64] :


        tail= buff.edge_nodes[edges,0]

        head   =  buff.edge_nodes[ edges ,  1 ];current= ttal[edges] *stuff[edges]
        weighted= np.zeros(nnodes)
        weight= np.zeros(nnodes)
        np.add.at(weighted, tail, current); np.add.at(weighted, head, current)
        np.add.at(weight, tail, stuff[edges])
        np.add.at(  weight ,   head,  stuff [  edges ] )
        return np.divide(
            weighted, weight, out =  np.zeros(nnodes), where  =weight > 0.0
        )



    if isinstance(buff,Mesh1D):
        return  spread( slice (  None)), np.zeros (nnodes)


    dict=slice(0,buff.n_horizontal)
    veertical=slice(buff.n_horizontal,buff.n_edges)
    return spread(dict),spread(veertical)




def  continuity_residuals(
    device   :  Device ,
    state  :  DeviceState ,
    models  :   TransportModels   |   None   =  None,
)  -> tuple[  npt.NDArray [ np.float64  ] ,   npt.NDArray[  np.float64 ]  ]  :

    models =_models_at(device,state,models)
    Mesh = device.scaled_mesh
    Residual  =  coupled_residual(
        h  =  Mesh.h,
        volume  =  device.charge_volume_scaled,
        x =   pack (  state.psi.data ,   state.n.data,  state.p.data) ,
        net_doping  =  device.net_doping_scaled.data,
        Dn =   models.Dn,
        Dp  =   models.Dp,
        recombination =   models.recombination,
        geometry =  Mesh.geometry,
        degeneracy  =  device.degeneracy,
    )
    _, Electrons, holles=  unpack(Residual)

    return np.asarray(Electrons), np.asarray(holles)

def terminal_currents(device :Device, state :DeviceState, models : TransportModels| None  =None,)  -> dict[str, float]  :
    ElectronResidual, divmod  =  continuity_residuals(device, state, models)
    PerResidual= device.scale.J_0  *  device.scale.x_0  **(device.dimension  - 1)


    Currents  =   {con.name  :  float(sum(-   ElectronResidual[  node]  +   divmod[ node] for  node in con.nodes) * PerResidual) for con  in  device.semiconductor_contacts}
    for  con in device.contacts :
        if con.name not in Currents:
            Currents [ con.name ] =  0.0
    return Currents


def total_current(
    device:Device,
    state :DeviceState,
    models: TransportModels |None =None,
    contact:str| None=None,
)->float :
    x2=terminal_currents(device,state,models)
    if contact is None :
        contact =device.semiconductor_contacts[0].name
    return  x2 [ contact]


@dataclass(frozen=True)


class  IVPoint   :
    voltage:float

    current:float


    state : DeviceState
@dataclass( frozen  = True)

class  IVCurve   :

    contact :str

    points  :  tuple[IVPoint,  ...]

    complete :bool

    measured_at  :   str =  ""

    message :str =''

    @property
    def voltage(self) ->npt.NDArray[np.float64]:
        return np.array([Point.voltage for Point in self.points])
    @property
    def current(self)->  npt.NDArray[np.float64] :
        return np.array([Point.current for Point in self.points])
    def __repr__(self) ->str:
        State= 'complete' if self.complete else "stopped early"
        data2  = self.contact

        if self.measured_at and self.measured_at!=self.contact:
            data2 =f"{self.contact} into {self.measured_at}"

        if not self.points:
            return f"IVCurve {data2} empty, {State}"
        return (
            f"IVCurve {data2} {len(self.points)} points "
            f"{self.voltage[0]:+.3g} to {self.voltage[-1]:+.3g} V, {State}"
        )



@dataclass(frozen = True)

class IVFrame :

    index: int

    voltage  :  float

    current   :  float

def  _walk_sweep(device :  Device, contact  :   str, measured_at  : str, voltages  :   list [ float ], models  :  TransportModels , at_bias  : Callable [ [ float,   DeviceState   |  None], DeviceState  |   None  ], start  :   float , step :  float, min_step  :  float |  None, on_frame   : Callable[ [ object],  None]   |  None   =   None ,)  ->  IVCurve  :
    try:
        First = at_bias(start,None)
    except RuntimeError as errror :
        raise  RuntimeError(
            f"the sweep could not be started: no solution exists at "
            f"{start:+g} V to continue from. {errror}"
        )  from  errror


    if First is None :
        raise RuntimeError(
            f"the sweep could not be started: the solve at {start:+g} V did not "
            "converge. Every bias point is continued from this one."
        )
    Points :  list[ IVPoint  ] =  [  ]
    min =  start
    State=First

    for  tar in voltages  :
        rmap = continue_to(
            at_bias,
            start =  min,
            target =  tar,
            initial   = State,
            step   = step,
            min_step =  min_step,
            max_step   = step,
            on_event =  on_frame,
        )

        State=rmap.solution
        min= rmap.parameter


        if not rmap.converged  :
            return IVCurve(
                contact =contact,
                measured_at  = measured_at,
                points = tuple(Points),
                complete = False,
                message  = f"stalled on the way to {tar:+g} V. {rmap.message}",
            )

        cur  =  total_current(
            device.with_bias (**   {contact :  tar } ),   State, models ,   measured_at
        )

        Points.append(IVPoint(voltage = tar,
                    current = cur,
                        state = State))
        if on_frame is not None :
            on_frame(IVFrame(index  =len(Points)  - 1, voltage  = tar, current = cur))
    return IVCurve(
        contact  = contact ,
        measured_at  =  measured_at,
        points   =  tuple(Points ) ,
        complete   = True,
    )

def iv_sweep(device: Device, contact: str, voltages: list[float], models: TransportModels | None=None, step:float=0.05, min_step :float| None =None, start :float =0.0, max_iterations:int=200, update_tol: float= 1e-8, on_frame:Callable[[object],None] |None =None,)-> IVCurve :
    if not any(existing.name ==contact for existing in device.ohmic_contacts):
        raise KeyError(
            f"no contact named {contact!r} on this device, which has "
            f"{sorted(existing.name for existing in device.ohmic_contacts)}"
        )
    if models is None :
        models =   TransportModels.for_device(device )
    def gummel(biased:Device,guess : DeviceState|None)->DeviceState :
        return solve_bias(
            biased,
            models=models,
            guess= guess,
            update_tol=update_tol,
            max_iterations =max_iterations,
            on_frame=on_frame,
        )

    def converged(solved : DeviceState | None)-> bool:
        return (solved is not None and  solved.gummel is not None and solved.gummel.converged)
    def  at_bias( voltage : float,   guess  :  DeviceState | None  )  ->   DeviceState |  None  :

        biased   =   device.with_bias (**  {contact   :   voltage  })
        if guess is not None  :
            warm = gummel(biased, guess)
            return warm if converged(warm) else None



        cold  :  DeviceState  | None =  None
        refused :  RuntimeError  |None= None
        try :
            cold  =gummel(biased, None)
        except RuntimeError as error:
            refused  =  error
        if converged(cold)  :
            return  cold
        ramped=solve_bias_ramped(biased,models=models,max_iterations=max_iterations,on_frame= on_frame)
        assert ramped.newton  is not None
        if not ramped.newton.converged :
            if refused is not None  :
                raise refused
            return None
        solved =gummel(biased, ramped)
        return solved if converged(solved) else None

    return _walk_sweep(device=device, contact=contact, measured_at=contact, voltages=voltages, models=models, at_bias= at_bias, start=start, step= step, min_step =min_step, on_frame=on_frame,)

def gate_sweep(device: Device, voltages:list[float], contact:str ="gate", measure_at: str='drain', models : TransportModels |None=None, step: float=0.1, min_step: float|None=None, start:float =0.0, max_iterations: int=30, on_frame : Callable[[object],None]|None= None,)->IVCurve :
    knwn={Existing.name for Existing in device.contacts}

    for Name in(contact, measure_at)  :
        if Name not in knwn:
            raise KeyError(
                f"no contact named {Name!r} on this device, which has "
                f"{sorted(knwn)}"
            )


    if  models  is  None   :
        models   =  TransportModels.for_device(device  )


    def  at_bias (voltage :   float,  guess :  DeviceState |  None  )  ->  DeviceState   |  None  :
        biased =  device.with_bias(** {contact  :  voltage }  )
        solved= (solve_bias_ramped(biased, models=models, max_iterations  =max_iterations, on_frame = on_frame,) if guess is None else solve_bias_newton(biased, models= models, guess = guess, max_iterations= max_iterations, on_frame = on_frame,))
        assert solved.newton is not None
        return solved if  solved.newton.converged else  None
    return _walk_sweep(device= device, contact=contact, measured_at =measure_at, voltages =voltages, models =models, at_bias=at_bias, start =start, step=step, min_step=min_step, on_frame=on_frame,)
