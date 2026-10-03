from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
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

    r =device.scale
    rows = device.scaled_mesh


    w = edge_drop(state.psi.data, rows.geometry)


    a  =diffusivity_at(models.Dn, w, rows.h)

    num  = diffusivity_at(models.Dp, w, rows.h)

    k,t= effective_potentials(
        state.psi.data,state.n.data,state.p.data,device.degeneracy
    )
    s= _per_unit_face(
        electron_current(rows.h,a,k,state.n.data,rows.geometry),
        rows.geometry,
    )
    x = _per_unit_face(
        hole_current(rows.h, num, t, state.p.data, rows.geometry),
        rows.geometry,
    )

    return(
        Field(s, "A/cm^2", ScalingState.SCALED, Location.EDGE, name  ='Jn').to_physical(
            r
        ),
        Field(x, 'A/cm^2', ScalingState.SCALED, Location.EDGE, name="Jp").to_physical(
            r
        ),
    )

def _per_unit_face(flux  : npt.NDArray[np.float64], geometry  : EdgeGeometry) ->npt.NDArray[np.float64]  :
    xs  =  np.asarray( geometry.carrier_face , dtype  =   np.float64  )

    return np.divide(
        flux, xs, out =np.zeros_like(flux), where = xs  > 0.0
    )



def edge_current_face(device: Device)->npt.NDArray[np.float64] :
    row= np.asarray(device.scaled_mesh.geometry.carrier_face,dtype=np.float64)
    s= 0 if isinstance(device.mesh,Mesh1D)else 1
    return np.asarray(row* device.scale.x_0 **s)

def node_current_density (device  : Device , state  :  DeviceState, models  :  TransportModels   |  None  =  None,) ->  tuple[npt.NDArray [  np.float64 ], npt.NDArray[np.float64  ]  ]  :
    r, z2 =current_densities(device, state, models) ; vv = r.data  + z2.data
    z= np.broadcast_to(edge_current_face(device),vv.shape)
    j= device.mesh

    a = j.n_nodes
    def spread(edges: slice) ->  npt.NDArray[np.float64] :


        x= j.edge_nodes[edges,0]

        k   =  j.edge_nodes[ edges ,  1 ];row= vv[edges] *z[edges]
        d= np.zeros(a)
        dat= np.zeros(a)
        np.add.at(d, x, row); np.add.at(d, k, row)
        np.add.at(dat, x, z[edges])
        np.add.at(  dat ,   k,  z [  edges ] )
        return np.divide(
            d, dat, out =  np.zeros(a), where  =dat > 0.0
        )



    if isinstance(j,Mesh1D):
        return  spread( slice (  None)), np.zeros (a)


    b=slice(0,j.n_horizontal)
    tmp=slice(j.n_horizontal,j.n_edges)
    return spread(b),spread(tmp)




def  continuity_residuals(
    device   :  Device ,
    state  :  DeviceState ,
    models  :   TransportModels   |   None   =  None,
)  -> tuple[  npt.NDArray [ np.float64  ] ,   npt.NDArray[  np.float64 ]  ]  :

    models =_models_at(device,state,models)
    z = device.scaled_mesh
    y  =  coupled_residual(
        h  =  z.h,
        volume  =  device.charge_volume_scaled,
        x =   pack (  state.psi.data ,   state.n.data,  state.p.data) ,
        net_doping  =  device.net_doping_scaled.data,
        Dn =   models.Dn,
        Dp  =   models.Dp,
        recombination =   models.recombination,
        geometry =  z.geometry,
        degeneracy  =  device.degeneracy,
    )
    _, x2, g=  unpack(y)

    return np.asarray(x2), np.asarray(g)

def terminal_currents(device :Device, state :DeviceState, models : TransportModels| None  =None,)  -> dict[str, float]  :
    v, s  =  continuity_residuals(device, state, models)
    f= device.scale.J_0  *  device.scale.x_0  **(device.dimension  - 1)


    w  =   {z.name  :  float(sum(-   v[  obj]  +   s[ obj] for  obj in z.nodes) * f) for z  in  device.semiconductor_contacts}
    for  z in device.contacts :
        if z.name not in w:
            w [ z.name ] =  0.0
    return w


def total_current(
    device:Device,
    state :DeviceState,
    models: TransportModels |None =None,
    contact:str| None=None,
)->float :
    k=terminal_currents(device,state,models)
    if contact is None :
        contact =device.semiconductor_contacts[0].name
    return  k [ contact]


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
        return np.array([z.voltage for z in self.points])
    @property
    def current(self)->  npt.NDArray[np.float64] :
        return np.array([t.current for t in self.points])
    def __repr__(self) ->str:
        w2= 'complete' if self.complete else "stopped early"
        v  = self.contact

        if self.measured_at and self.measured_at!=self.contact:
            v =f"{self.contact} into {self.measured_at}"

        if not self.points:
            return f"IVCurve {v} empty, {w2}"
        return (
            f"IVCurve {v} {len(self.points)} points "
            f"{self.voltage[0]:+.3g} to {self.voltage[-1]:+.3g} V, {w2}"
        )



@dataclass(frozen = True)

class IVFrame :

    index: int

    voltage  :  float

    current   :  float

def  _walk_sweep(device :  Device, contact  :   str, measured_at  : str, voltages  :   list [ float ], models  :  TransportModels , at_bias  : Callable [ [ float,   DeviceState   |  None], DeviceState  |   None  ], start  :   float , step :  float, min_step  :  float |  None, on_frame   : Callable[ [ object],  None]   |  None   =   None ,)  ->  IVCurve  :
    try:
        s = at_bias(start,None)
    except RuntimeError as t :
        raise  RuntimeError(
            f"the sweep could not be started: no solution exists at "
            f"{start:+g} V to continue from. {t}"
        )  from  t


    if s is None :
        raise RuntimeError(
            f"the sweep could not be started: the solve at {start:+g} V did not "
            "converge. Every bias point is continued from this one."
        )
    yy :  list[ IVPoint  ] =  [  ]
    bar =  start
    info=s

    for  d in voltages  :
        rr = continue_to(
            at_bias,
            start =  bar,
            target =  d,
            initial   = info,
            step   = step,
            min_step =  min_step,
            max_step   = step,
            on_event =  on_frame,
        )

        info=rr.solution
        bar= rr.parameter


        if not rr.converged  :
            return IVCurve(
                contact =contact,
                measured_at  = measured_at,
                points = tuple(yy),
                complete = False,
                message  = f"stalled on the way to {d:+g} V. {rr.message}",
            )

        v  =  total_current(
            device.with_bias (**   {contact :  d } ),   info, models ,   measured_at
        )

        yy.append(IVPoint(voltage = d,
                    current = v,
                        state = info))
        if on_frame is not None :
            on_frame(IVFrame(index  =len(yy)  - 1, voltage  = d, current = v))
    return IVCurve(
        contact  = contact ,
        measured_at  =  measured_at,
        points   =  tuple(yy ) ,
        complete   = True,
    )

def iv_sweep(device: Device, contact: str, voltages: list[float], models: TransportModels | None=None, step:float=0.05, min_step :float| None =None, start :float =0.0, max_iterations:int=200, update_tol: float= 1e-8, on_frame:Callable[[object],None] |None =None,)-> IVCurve :
    if not any(tmp2.name ==contact for tmp2 in device.ohmic_contacts):
        raise KeyError(
            f"no contact named {contact!r} on this device, which has "
            f"{sorted(t.name for t in device.ohmic_contacts)}"
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

        info   =   device.with_bias (**  {contact   :   voltage  })
        if guess is not None  :
            a = gummel(info, guess)
            return a if converged(a) else None



        vv  :  DeviceState  | None =  None
        i :  RuntimeError  |None= None
        try :
            vv  =gummel(info, None)
        except RuntimeError as x2:
            i  =  x2
        if converged(vv)  :
            return  vv
        s=solve_bias_ramped(info,models=models,max_iterations=max_iterations,on_frame= on_frame)
        assert s.newton  is not None
        if not s.newton.converged :
            if i is not None  :
                raise i
            return None
        zz =gummel(info, s)
        return zz if converged(zz) else None

    return _walk_sweep(device=device, contact=contact, measured_at=contact, voltages=voltages, models=models, at_bias= at_bias, start=start, step= step, min_step =min_step, on_frame=on_frame,)

def gate_sweep(device: Device, voltages:list[float], contact:str ="gate", measure_at: str='drain', models : TransportModels |None=None, step: float=0.1, min_step: float|None=None, start:float =0.0, max_iterations: int=30, on_frame : Callable[[object],None]|None= None,)->IVCurve :
    h={t2.name for t2 in device.contacts}

    for xx in(contact, measure_at)  :
        if xx not in h:
            raise KeyError(
                f"no contact named {xx!r} on this device, which has "
                f"{sorted(h)}"
            )


    if  models  is  None   :
        models   =  TransportModels.for_device(device  )


    def  at_bias (voltage :   float,  guess :  DeviceState |  None  )  ->  DeviceState   |  None  :
        tmp2 =  device.with_bias(** {contact  :  voltage }  )
        stuff= (solve_bias_ramped(tmp2, models=models, max_iterations  =max_iterations, on_frame = on_frame,) if guess is None else solve_bias_newton(tmp2, models= models, guess = guess, max_iterations= max_iterations, on_frame = on_frame,))
        assert stuff.newton is not None
        return stuff if  stuff.newton.converged else  None
    return _walk_sweep(device= device, contact=contact, measured_at =measure_at, voltages =voltages, models =models, at_bias=at_bias, start =start, step=step, min_step=min_step, on_frame=on_frame,)
