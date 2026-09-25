from __future__ import annotations

from collections.abc import Callable;  from dataclasses import dataclass, replace

import numpy as np; import numpy.typing as npt
from ddsim.core import constants as C

from ddsim.core.field import Field, Location, ScalingState


from ddsim.device.builder import  Device

from ddsim.device.equilibrium import(frozen_quasi_fermi, solve_equilibrium, solve_poisson,)
from ddsim.device.state import DeviceState

from ddsim.discretize.assembly import SparseAssembly
from ddsim.discretize.boundary import(
    Carrier,
    apply_ohmic_densities,
    impose_ohmic_densities,
)


from ddsim.discretize.continuity import(
    Diffusivity,
    assemble_electron_continuity,
    assemble_hole_continuity,
)


from ddsim.discretize.coupled import(
    apply_contacts_coupled,
    assemble_coupled_terms,
    coupled_update_by_family,
    edge_drop,
    limit_psi_step,
    pack,
    residual_measure_by_family,
    residual_term_scales,
    row_weights,
    scale_rows,
    unpack,
)
from ddsim.mesh.mesh2d import Mesh2D,normal_field
from ddsim.physics.mobility import(AroraMobility, CaugheyThomas, ConstantMobility, EdgeDiffusivity, LombardiSurface, diffusivity_at, edge_diffusivity,)

from ddsim.physics.recombination import(
    AugerRecombination,
    RecombinationModel,
    SRHRecombination,
    SumOfRecombination,
    scharfetter_lifetime,
)
from  ddsim.solve.continuation  import  continue_to

from ddsim.solve.gummel import BlockStep, GummelResult, gummel_solve; from ddsim.solve.linear import SparseLU

from ddsim.solve.newton import NewtonIteration,NewtonResult,newton_solve
MOBILITY_MODELS  =("constant", 'arora')

DENSITY_REFERENCE  =   1.0


class  TransportError( RuntimeError )  :



    def __init__(self, message: str, state  : DeviceState) -> None  :
        super().__init__(message)
        self.state   =  state



@dataclass(frozen=True)

class SurfaceScattering   :

    electrons  : LombardiSurface

    holes  :   LombardiSurface
    mu_bulk_n:npt.NDArray[np.float64]


    mu_bulk_p  : npt.NDArray[np.float64]
    total_doping  :  npt.NDArray [  np.float64]

    semiconductor :  npt.NDArray[np.bool_]

    def corrected(
        self,
        device:Device,
        psi :npt.NDArray[np.float64],
        n:npt.NDArray[np.float64],
        p : npt.NDArray[np.float64],
    ) ->tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
        mes = device.mesh
        if not  isinstance(  mes , Mesh2D  )  :
            raise  TypeError(
                'surface mobility needs a direction normal to the interface '
                f"and a {type(mes).__name__} has none. Build the device on a "
                'Mesh2D, which is what a MOSFET is on.'
            )

        sca = device.scale;eperp =normal_field(mes,psi *sca.psi_0)
        carirers =(n + p)  *  sca.C_0

        return(
            np.where(
                self.semiconductor,
                self.electrons(self.mu_bulk_n, eperp, self.total_doping, carirers),
                self.mu_bulk_n,
            ),
            np.where(
                self.semiconductor,
                self.holes(self.mu_bulk_p, eperp, self.total_doping, carirers),
                self.mu_bulk_p,
            ),
        )

@dataclass(frozen = True)



class TransportModels  :
    recombination :  RecombinationModel



    Dn  :  EdgeDiffusivity

    Dp :EdgeDiffusivity

    surface  :  SurfaceScattering  |   None  =   None

    field_dependent : bool = False


    def at_state(
        self,
        device :  Device,
        psi  : npt.NDArray[np.float64],
        n: npt.NDArray[np.float64],
        p :  npt.NDArray[np.float64],
    ) ->  TransportModels:

        if self.surface is None  :
            return self
        muu_n, muu_p = self.surface.corrected(device, psi, n, p)
        return replace(self, Dn = _from_nodal_mobility(device,Carrier.ELECTRON,muu_n,self.field_dependent), Dp= _from_nodal_mobility(device,Carrier.HOLE,muu_p,self.field_dependent),)
    @classmethod
    def for_device(cls, device  :  Device, recombination  :   RecombinationModel |   None  =  None, mobility  :   str  =   "constant" , auger  :  bool  =   False , field_dependent : bool   = False , surface   : bool   =  False,)  ->  TransportModels :

        res = device.scale
        TotalDoping  =   np.abs( device.net_doping.data)



        if  recombination  is  None :
            recombination =SRHRecombination(tau_n= scharfetter_lifetime(TotalDoping,tau_max =C.TAU_N_MAX,tau_min= C.TAU_N_MIN) /res.t_0, tau_p=scharfetter_lifetime(TotalDoping,tau_max = C.TAU_P_MAX,tau_min=C.TAU_P_MIN) / res.t_0, ni2=(device.material.n_i/res.C_0)**2, n1=device.material.n_i/res.C_0, p1 =device.material.n_i/ res.C_0,)
            if  auger  :
                recombination= SumOfRecombination((recombination, AugerRecombination(C_n =C.AUGER_C_N * res.C_0 **2 *res.t_0, C_p= C.AUGER_C_P* res.C_0**2 * res.t_0, ni2=(device.material.n_i/res.C_0)** 2,),))


        return cls(
            recombination=recombination,
            Dn = _scaled_diffusivity(
                device,Carrier.ELECTRON,mobility,field_dependent
            ),
            Dp=_scaled_diffusivity(
                device,Carrier.HOLE,mobility,field_dependent
            ),
            surface= (
                _surface_scattering(device,mobility,TotalDoping)
                if surface
                else None
            ),
            field_dependent=field_dependent,
        )


def _scaled_diffusivity(
    device : Device, carrier : Carrier, mobility : str, field_dependent  :  bool  =False
) ->  EdgeDiffusivity :
    Scale   =  device.scale
    t2  = device.material.T
    vals  =  carrier is Carrier.ELECTRON
    if  mobility  ==   "constant"  :
        consatnt= C.D_n(t2)if vals else C.D_p(t2)
        low_fiield: Diffusivity  =  consatnt /  Scale.D_0
    elif mobility=='arora'  :

        round  =(
            AroraMobility.electrons(t2)
            if vals
            else AroraMobility.holes(t2)
        )
        noadl  = round(np.abs(device.net_doping.data))
        edg=device.scaled_mesh.geometry.edge_nodes
        low_fiield=(edge_diffusivity(noadl,C.V_T(t2),edg)/Scale.D_0)
    else :
        raise ValueError (
            f"unknown mobility model {mobility!r}. Use "
            +   " or ".join(  repr( name  ) for name  in  MOBILITY_MODELS  )
            +  "."
        )

    return _wrapped_in_saturation ( device,   carrier,   low_fiield,   field_dependent )



def _surface_scattering(
    device :Device, mobility :str, total_doping  :npt.NDArray[np.float64]
)  -> SurfaceScattering:

    Temperature  =  device.material.T


    if not isinstance(device.mesh,Mesh2D) :
        raise TypeError(
            "surface mobility needs a direction normal to the interface "
            f"and a {type(device.mesh).__name__} has none. Build the device on "
            "a Mesh2D, which is what a MOSFET is on."
        )
    if  device.regions is  not  None  :
        cel=device.regions.cell_material
        if(cel[:, 1  :] !=  cel[:, :- 1]).any()  :
            raise ValueError(
                'surface mobility reads the field normal to a flat Si/SiO2 '
                "interface, dpsi/dy, and this device has a vertical one, an "
                'oxide wall beside silicon, where the normal is x. Solve it '
                "without surface scattering."
            )


    if mobility== 'constant':
        nod = ConstantMobility(C.mu_n(Temperature)) (total_doping)
        nodalp =ConstantMobility(C.mu_p(Temperature)) (total_doping)
    elif  mobility ==  'arora'  :

        nod=  AroraMobility.electrons(Temperature)  (total_doping)
        nodalp =AroraMobility.holes(Temperature)(total_doping)

    else:

        raise ValueError(
            f"unknown mobility model {mobility!r}. Use "
            +" or ".join(repr(name)for name in MOBILITY_MODELS)
            +"."
        )
    sem =np.ones(device.mesh.n_nodes,dtype =np.bool_)
    sem [  list(  device.carrier_free_nodes  )  ]  =  False
    return SurfaceScattering(electrons= LombardiSurface.electrons(Temperature), holes =LombardiSurface.holes(Temperature), mu_bulk_n =  nod, mu_bulk_p = nodalp, total_doping=np.maximum(total_doping, device.material.n_i), semiconductor  =sem,)

def  _from_nodal_mobility(device :  Device , carrier  :   Carrier , nodal  :  npt.NDArray[  np.float64  ], field_dependent   :  bool,)   ->  EdgeDiffusivity :
    EdgeNodes   =   device.scaled_mesh.geometry.edge_nodes
    res  =  C.V_T( device.material.T )
    low  =edge_diffusivity(nodal, res, EdgeNodes) / device.scale.D_0; return  _wrapped_in_saturation( device,   carrier,  low ,  field_dependent )




def _wrapped_in_saturation(device :Device, carrier  :Carrier, low_field  :  Diffusivity, field_dependent : bool,)  ->  EdgeDiffusivity  :
    if not field_dependent:
        return low_field

    Scale = device.scale


    temperaure  = device.material.T
    elecctrons=carrier is Carrier.ELECTRON
    vSat =   C.v_sat_n ( temperaure)   if elecctrons else C.v_sat_p(  temperaure)


    return CaugheyThomas(
        low_field =np.broadcast_to(
            np.asarray(low_field,dtype=np.float64),(device.scaled_mesh.h.size,)
        ).copy(),
        v_sat = vSat * Scale.x_0/Scale.D_0,
        beta= C.BETA_N if elecctrons else C.BETA_P,
    )




def _lagged_diffusivity(
    D:EdgeDiffusivity,device:Device,psi: npt.NDArray[np.float64]
)->Diffusivity:

    return diffusivity_at(D, edge_drop(psi), device.scaled_mesh.h)



def _node_field(values:  npt.NDArray[np.float64], unit: str, name  :  str)-> Field :

    return Field(values, unit, ScalingState.SCALED, Location.NODE, name= name)




def _density_update(
    old : npt.NDArray[np.float64], new : npt.NDArray[np.float64]
)  ->  float :


    return float(np.max(np.abs(new-old) /(np.abs(old) +DENSITY_REFERENCE)))




def poisson_block(device :Device) ->BlockStep[DeviceState]:

    Solver  = SparseLU(  )

    def step(state : DeviceState)  ->  tuple[DeviceState, float] :
        result =solve_poisson(device,state.psi.data,state.phi_n,state.phi_p,solver=Solver)
        if not result.converged  :
            raise TransportError(
                f"the Poisson block did not converge: {result.message}",state
            )


        shift=result.x - state.psi.data

        with np.errstate(over="ignore",under='ignore'):


            n = state.n.data *np.exp(shift);  p=state.p.data*np.exp(- shift)

        if not(np.all(np.isfinite(n))and np.all(np.isfinite(p))) :
            raise TransportError(
                f"the potential moved by {np.max(np.abs(shift)):.3g} V_T in one "
                "cycle and overflowed the Boltzmann densities. Ramp the bias in "
                'smaller steps.',
                state,
            )

        updated =  replace(state, psi = _node_field(result.x, 'V', 'psi'), n =  _node_field(n, 'cm^-3', "n"), p=  _node_field(p, "cm^-3", 'p'), newton= result,)
        return updated, float(np.max(np.abs(shift)))
    return step



def _lagged_effective_potential(
    device :  Device, state: DeviceState, carrier:  Carrier
) -> Field :
    deegeneracy=device.degeneracy
    if deegeneracy is  None  :
        return state.psi
    if  carrier is Carrier.ELECTRON :
        val=  deegeneracy.electron_potential(state.psi.data, state.n.data)
    else:

        val= deegeneracy.hole_potential(state.psi.data,state.p.data)
    return _node_field(np.asarray(val),
      'V',
              'psi_eff')


def  electron_block (
    device  : Device,  models  :   TransportModels
)   ->   BlockStep[DeviceState ]  :
    Doping = device.net_doping_scaled.data
    x2  = device.degeneracy
    sollver= SparseLU()
    def step(state  :DeviceState)->  tuple[DeviceState, float]:

        assembly =assemble_electron_continuity(device.mesh_1d, _lagged_effective_potential(device,state,Carrier.ELECTRON), state.n, state.p, models.recombination, device.scale, _lagged_diffusivity(models.Dn,device,state.psi.data),)
        assembly= apply_ohmic_densities(
            assembly,
            state.n.data,
            Doping,
            device.ohmic_contacts,
            Carrier.ELECTRON,
            x2,
        )

        sollver.factorize(assembly.rows, assembly.cols, assembly.values, assembly.shape)
        updated_n  =  impose_ohmic_densities(state.n.data  + sollver.solve(- assembly.residual), Doping, device.ohmic_contacts, Carrier.ELECTRON, x2,)

        _check_positive(updated_n,  'n',  state)
        return(replace(state, n =  _node_field(updated_n, "cm^-3", 'n')), _density_update(state.n.data, updated_n),)

    return step




def hole_block(device :Device,models:TransportModels)->BlockStep[DeviceState]:
    doipng =  device.net_doping_scaled.data; Degeneracy=device.degeneracy
    abs  =   SparseLU( )
    def step(state :DeviceState) -> tuple[DeviceState, float]:
        assembly  =   assemble_hole_continuity(
            device.mesh_1d ,
            _lagged_effective_potential( device, state,  Carrier.HOLE ),
            state.n,
            state.p,
            models.recombination,
            device.scale,
            _lagged_diffusivity(models.Dp, device, state.psi.data),
        )
        assembly=apply_ohmic_densities(
            assembly,
            state.p.data,
            doipng,
            device.ohmic_contacts,
            Carrier.HOLE,
            Degeneracy,
        )

        abs.factorize(assembly.rows,assembly.cols,assembly.values,assembly.shape)
        updated_p = impose_ohmic_densities(state.p.data+abs.solve(- assembly.residual), doipng, device.ohmic_contacts, Carrier.HOLE, Degeneracy,)
        _check_positive(updated_p,"p",state)
        return(
            replace(state, p = _node_field(updated_p, "cm^-3", "p")),
            _density_update(state.p.data, updated_p),
        )

    return step


def  _check_positive (
    density   :  npt.NDArray [  np.float64  ] , name : str ,   state : DeviceState
)   ->  None   :
    if np.all(density>0.0):
        return

    k2 =int(np.argmin(density))
    raise TransportError(
        f"{name} came out non-positive at node {k2}, value "
        f"{density[k2]:.3e}. The continuity matrix should be an M-matrix "
        "with a non-negative right hand side, so check signs before anything "
        'else, per docs/05-pitfalls.md. Do not clamp.',
        state,
    )

def initial_state(device : Device)->DeviceState:
    return solve_equilibrium(device,frozen_quasi_fermi(device))

def _low_field_models(models: TransportModels) ->  TransportModels  :
    temp2 =tuple(
        D.low_field if isinstance(D, CaugheyThomas)else D
        for D in(models.Dn, models.Dp)
    )
    return replace(
        models,
        Dn  = temp2[0],
        Dp  =  temp2[1],
        surface =None,
        field_dependent = False,
    )
def _needs_a_low_field_prelude(
    models :TransportModels, guess: DeviceState |None
)  -> bool  :

    return  guess is  None  and  models.field_dependent


def _low_field_edges(D:EdgeDiffusivity) ->  npt.NDArray[np.float64]  :
    if isinstance(D, CaugheyThomas)  :
        return D.low_field

    return np.asarray(D, dtype =  np.float64)


def _surface_moved(before  : TransportModels, after  : TransportModels)  -> float  :
    Worst  =  0.0
    for w,neww in((before.Dn,after.Dn),(before.Dp,after.Dp)):
        aa, bb =  _low_field_edges(w), _low_field_edges(neww)

        arr,t2= np.abs(bb -aa),np.abs(aa)


        Relative=np.divide(arr,t2,out=np.where(arr>0.0,np.inf,0.0),where = t2 > 0.0)
        Worst=max(Worst,float(np.max(Relative)))

    return Worst



def _surface_fixed_point(
    device  : Device,
    models : TransportModels,
    run :Callable[[TransportModels, npt.NDArray[np.float64]], NewtonResult],
    x0: npt.NDArray[np.float64],
    max_sweeps: int,
    rtol:float,
) ->NewtonResult:
    Active=models.at_state(device,*unpack(x0))
    X  =  x0 ; k2=0
    residualHistory: list[float] =[]

    cnt :list[float]=[]
    limmited_steps=0
    Sweeps =0


    while Sweeps<max_sweeps:
        Sweeps+=1
        res   =   run (  Active ,  X)

        k2+=res.iterations
        residualHistory.extend(res.residual_history)


        cnt.extend(res.update_history)
        limmited_steps+=res.limited_steps
        X  =   res.x

        com   =   replace (res , iterations  =   k2, residual_history = residualHistory, update_history =   cnt, limited_steps  = limmited_steps,)


        if not res.converged:
            return com


        pow= Active.at_state(device,*unpack(X))
        if _surface_moved(Active, pow)  < rtol:
            return com
        Active =  pow

    return replace(
        com,
        converged= False,
        message = (
            f"the surface mobility was still moving after {max_sweeps} "
            f"sweeps. The last Newton solve converged; what did not is the "
            f"fixed point between the mobility and the state it is read from."
        ),
    )


def _reported_by_family(
    residual_by_family : Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64]], dict[str, float]
    ],
    on_frame : Callable[[object], None] |None,
) -> tuple[
    Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float],
    Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float],
    Callable[[NewtonIteration], None] |  None,
] :
    laast: dict[str,dict[str,float]]={}

    def residual_norm(
        residual :  npt.NDArray[np.float64], x:npt.NDArray[np.float64]
    ) ->  float:
        split   =  residual_by_family(residual,   x)
        laast["residual"]= split;  return max(0.0, * split.values())

    def update_norm(
        delta : npt.NDArray[np.float64], x  :  npt.NDArray[np.float64]
    )  -> float :
        split =coupled_update_by_family(delta,x)
        laast ["update" ]   =   split
        return  max(  split.values ( ))

    if on_frame is None:
        return residual_norm, update_norm, None
    Send =on_frame
    def  matching (kind :  str ,   value :   float  | None  )   -> dict [  str ,  float  ]  |  None  :
        split = laast.get(kind)
        if split is None or value is None or max(split.values())  !=  value :
            return  None
        return split
    def report(frame : NewtonIteration)  -> None :

        Send(replace(frame, residual_by_family  =  matching('residual', frame.residual), update_by_family =  matching('update', frame.update),))

    return residual_norm,update_norm,report


def solve_bias_newton(
    device:Device,
    models :TransportModels |None=None,
    guess :DeviceState|None= None,
    max_psi_step: float=5.0,
    max_iterations :int=30,
    residual_rtol:float= 1e-10,
    update_tol:float=1e-10,
    max_surface_sweeps : int =20,
    surface_rtol :float=1e-8,
    on_frame:Callable[[object],None]|None=None,
) ->DeviceState:
    if models is None:
        models  = TransportModels.for_device(device)
    w=initial_state(device) if guess is None else guess
    Scale = device.scale

    msh=device.scaled_mesh


    dir=msh.h
    Volume=device.charge_volume_scaled
    gemetry   = msh.geometry
    net   =   device.net_doping_scaled.data
    cf  =   device.carrier_free_nodes

    x00 =   pack( w.psi.data,  w.n.data, w.p.data  )


    def assembler(
        active  : TransportModels,
    ) -> Callable[[npt.NDArray[np.float64]], SparseAssembly]  :
        def assemble(x : npt.NDArray[np.float64]) ->SparseAssembly :
            assembly, scales = assemble_coupled_terms(
                dir,
                Volume,
                x,
                net,
                active.Dn,
                active.Dp,
                active.recombination,
                gemetry,
                device.degeneracy,
            )
            assembly  = apply_contacts_coupled (assembly , x , net, device.contacts, Scale, cf, device.material.T, device.degeneracy,)
            return scale_rows(assembly, row_weights(scales, msh.n_nodes))
        return assemble


    def measured(active :  TransportModels  )   ->   Callable[[npt.NDArray[np.float64  ],   npt.NDArray[np.float64  ] ], dict[str,  float ]] :

        def norm(
            residual: npt.NDArray[np.float64],x: npt.NDArray[np.float64]
        )->dict[str,float] :
            _ ,  n,   p   =   unpack( x )
            scales =residual_term_scales(
                dir,
                Volume,
                x,
                net,
                active.Dn,
                active.Dp,
                np.asarray(active.recombination.rate(n,p),dtype =np.float64),
                gemetry,
                device.degeneracy,
            )
            return residual_measure_by_family(residual, scales, msh.n_nodes)



        return  norm

    def run(active :TransportModels,x :npt.NDArray[np.float64])->NewtonResult:
        residual_norm,update_norm,report=_reported_by_family(
            measured(active),on_frame
        )
        return newton_solve(
            assembler(active),
            x,
            limit=lambda delta : limit_psi_step(delta, max_psi_step),
            residual_scale=1.0,
            residual_norm = residual_norm,
            residual_rtol = residual_rtol,
            update_tol= update_tol,
            update_norm  =  update_norm,
            max_iterations =max_iterations,
            on_iteration =  report,
        )
    def solve_with(active:TransportModels,x : npt.NDArray[np.float64]) -> NewtonResult:
        if active.surface is None :
            return  run(  active, x  )

        return  _surface_fixed_point(
            device,   active, run, x, max_surface_sweeps,  surface_rtol
        )


    pre  :  NewtonResult   | None  =  None
    if _needs_a_low_field_prelude(models,guess):
        pre=  solve_with(_low_field_models(models), x00); x00  =   pre.x
    res = solve_with(models,x00)
    if pre is not None :
        res=replace(res, iterations=res.iterations+ pre.iterations, residual_history=pre.residual_history+res.residual_history, update_history= pre.update_history+res.update_history, limited_steps=res.limited_steps+pre.limited_steps,)
    psi, n, p =unpack(res.x)

    return DeviceState(psi =  _node_field(psi.copy(), 'V', 'psi'), n=  _node_field(n.copy(), "cm^-3", "n"), p  =_node_field(p.copy(), 'cm^-3', "p"), newton=  res, degeneracy= device.degeneracy,)

def solve_bias_ramped(
    device  :   Device ,
    models  : TransportModels |   None  =   None ,
    step  :  float  = 0.25,
    max_iterations   :   int  =   30 ,
    on_frame  :  Callable[[  object],  None  ]  |  None  =  None,
)  ->   DeviceState  :
    if models is None:
        models= TransportModels.for_device(device)


    Applied={Contact.name: Contact.voltage for Contact in device.contacts}

    def at_fraction(fraction: float,guess: DeviceState| None) ->DeviceState|None:
        solved = solve_bias_newton(
            device.with_bias(
                ** {name: fraction* volts for name, volts in Applied.items()}
            ),
            models = models,
            guess =guess,
            max_iterations  =  max_iterations,
            on_frame  = on_frame,
        )
        assert solved.newton is not None
        return solved if solved.newton.converged else None

    offf=solve_bias_newton(
        device.with_bias(**dict.fromkeys(Applied,0.0)),
        models=models,
        max_iterations=max_iterations,
        on_frame =on_frame,
    )

    Ramp  =  continue_to (
        at_fraction,
        start  = 0.0,
        target  = 1.0,
        initial  =   offf,
        step  =   step ,
        on_event  =  on_frame,
    )


    return solve_bias_newton (
        device,
        models  =  models,
        guess =  Ramp.solution,
        max_iterations  =   max_iterations,
        on_frame =  on_frame,
    )

def _gummel_prelude(device  :Device, models :  TransportModels, state : DeviceState, cycles : int, on_frame : Callable[[object], None]  | None  = None,)->  DeviceState :
    if cycles <=0:
        return state
    ste =[
        poisson_block(device),
        electron_block(device,models),
        hole_block(device,models),
    ]
    try :
        dat=gummel_solve(
            state,
            ste,
            update_tol=1e-300,
            max_iterations = cycles,
            on_iteration= on_frame,
        )

    except TransportError as fai :
        return fai.state
    return replace(dat.state,gummel =dat)


def solve_bias_hybrid(device  :   Device, models : TransportModels   |  None =  None, guess   :   DeviceState |  None  = None, gummel_cycles :  int  =   3, retry_cycles  :   int   =  5, max_psi_step  : float = 5.0, max_iterations :   int   =  30, on_frame   : Callable[[ object],  None  ]  |   None = None ,)  ->  DeviceState :

    if  models is  None  :
        models= TransportModels.for_device(device)

    sta =initial_state(device)if guess is None else guess
    def newton_from(state :DeviceState) -> DeviceState  :
        return solve_bias_newton(device, models  = models, guess  = state, max_psi_step= max_psi_step, max_iterations=max_iterations, on_frame  =on_frame,)
    slice  =  _gummel_prelude( device, models , sta,  gummel_cycles ,  on_frame) ; res=newton_from(slice)

    assert res.newton is not None
    if res.newton.converged or retry_cycles<=0:
        return replace(res, gummel= slice.gummel)

    slice=_gummel_prelude(device,models,slice,retry_cycles,on_frame)
    return replace(newton_from(slice), gummel  =  slice.gummel)
def solve_bias(
    device :Device,
    models:TransportModels|None= None,
    guess:DeviceState|None=None,
    update_tol:float =1e-8,
    max_iterations: int = 200,
    on_frame:Callable[[object],None]|None =None,
)->DeviceState:
    if models is None:

        models   =  TransportModels.for_device(  device)
    sta  =  initial_state(device)  if  guess is None  else  guess
    Steps= [
        poisson_block(device),
        electron_block(device,models),
        hole_block(device,models),
    ]

    try :
        Result  = gummel_solve(sta, Steps, update_tol  = update_tol, max_iterations =  max_iterations, on_iteration = on_frame,)
    except TransportError  as faiilure  :
        return replace(faiilure.state, gummel= GummelResult(state= faiilure.state, converged= False, iterations  = 0, message  = str(faiilure),),)
    return replace(Result.state,gummel = Result)
