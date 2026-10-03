from __future__ import annotations
from collections.abc import Callable; from dataclasses import dataclass, replace
import numpy as np; import numpy.typing as npt
from ddsim.core import constants as C
from ddsim.core.field import Field, Location, ScalingState
from ddsim.device.builder import Device
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
        dat = device.mesh
        if not  isinstance(  dat , Mesh2D  )  :
            raise  TypeError(
                'surface mobility needs a direction normal to the interface '
                f"and a {type(dat).__name__} has none. Build the device on a "
                'Mesh2D, which is what a MOSFET is on.'
            )

        i = device.scale;d =normal_field(dat,psi *i.psi_0)
        val =(n + p)  *  i.C_0

        return(
            np.where(
                self.semiconductor,
                self.electrons(self.mu_bulk_n, d, self.total_doping, val),
                self.mu_bulk_n,
            ),
            np.where(
                self.semiconductor,
                self.holes(self.mu_bulk_p, d, self.total_doping, val),
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
        c2, g = self.surface.corrected(device, psi, n, p)
        return replace(self, Dn = _from_nodal_mobility(device,Carrier.ELECTRON,c2,self.field_dependent), Dp= _from_nodal_mobility(device,Carrier.HOLE,g,self.field_dependent),)
    @classmethod
    def for_device(cls, device  :  Device, recombination  :   RecombinationModel |   None  =  None, mobility  :   str  =   "constant" , auger  :  bool  =   False , field_dependent : bool   = False , surface   : bool   =  False,)  ->  TransportModels :

        bb = device.scale
        c  =   np.abs( device.net_doping.data)



        if  recombination  is  None :
            recombination =SRHRecombination(tau_n= scharfetter_lifetime(c,tau_max =C.TAU_N_MAX,tau_min= C.TAU_N_MIN) /bb.t_0, tau_p=scharfetter_lifetime(c,tau_max = C.TAU_P_MAX,tau_min=C.TAU_P_MIN) / bb.t_0, ni2=(device.material.n_i/bb.C_0)**2, n1=device.material.n_i/bb.C_0, p1 =device.material.n_i/ bb.C_0,)
            if  auger  :
                recombination= SumOfRecombination((recombination, AugerRecombination(C_n =C.AUGER_C_N * bb.C_0 **2 *bb.t_0, C_p= C.AUGER_C_P* bb.C_0**2 * bb.t_0, ni2=(device.material.n_i/bb.C_0)** 2,),))


        return cls(
            recombination=recombination,
            Dn = _scaled_diffusivity(
                device,Carrier.ELECTRON,mobility,field_dependent
            ),
            Dp=_scaled_diffusivity(
                device,Carrier.HOLE,mobility,field_dependent
            ),
            surface= (
                _surface_scattering(device,mobility,c)
                if surface
                else None
            ),
            field_dependent=field_dependent,
        )


def _scaled_diffusivity(
    device : Device, carrier : Carrier, mobility : str, field_dependent  :  bool  =False
) ->  EdgeDiffusivity :
    out2   =  device.scale
    a  = device.material.T
    a2  =  carrier is Carrier.ELECTRON
    if  mobility  ==   "constant"  :
        s= C.D_n(a)if a2 else C.D_p(a)
        m: Diffusivity  =  s /  out2.D_0
    elif mobility=='arora'  :

        t  =(
            AroraMobility.electrons(a)
            if a2
            else AroraMobility.holes(a)
        )
        v  = t(np.abs(device.net_doping.data))
        z=device.scaled_mesh.geometry.edge_nodes
        m=(edge_diffusivity(v,C.V_T(a),z)/out2.D_0)
    else :
        raise ValueError (
            f"unknown mobility model {mobility!r}. Use "
            +   " or ".join(  repr( h  ) for h  in  MOBILITY_MODELS  )
            +  "."
        )

    return _wrapped_in_saturation ( device,   carrier,   m,   field_dependent )



def _surface_scattering(
    device :Device, mobility :str, total_doping  :npt.NDArray[np.float64]
)  -> SurfaceScattering:

    zz  =  device.material.T


    if not isinstance(device.mesh,Mesh2D) :
        raise TypeError(
            "surface mobility needs a direction normal to the interface "
            f"and a {type(device.mesh).__name__} has none. Build the device on "
            "a Mesh2D, which is what a MOSFET is on."
        )
    if  device.regions is  not  None  :
        j=device.regions.cell_material
        if(j[:, 1  :] !=  j[:, :- 1]).any()  :
            raise ValueError(
                'surface mobility reads the field normal to a flat Si/SiO2 '
                "interface, dpsi/dy, and this device has a vertical one, an "
                'oxide wall beside silicon, where the normal is x. Solve it '
                "without surface scattering."
            )


    if mobility== 'constant':
        y = ConstantMobility(C.mu_n(zz)) (total_doping)
        m2 =ConstantMobility(C.mu_p(zz)) (total_doping)
    elif  mobility ==  'arora'  :

        y=  AroraMobility.electrons(zz)  (total_doping)
        m2 =AroraMobility.holes(zz)(total_doping)

    else:

        raise ValueError(
            f"unknown mobility model {mobility!r}. Use "
            +" or ".join(repr(c)for c in MOBILITY_MODELS)
            +"."
        )
    cc =np.ones(device.mesh.n_nodes,dtype =np.bool_)
    cc [  list(  device.carrier_free_nodes  )  ]  =  False
    return SurfaceScattering(electrons= LombardiSurface.electrons(zz), holes =LombardiSurface.holes(zz), mu_bulk_n =  y, mu_bulk_p = m2, total_doping=np.maximum(total_doping, device.material.n_i), semiconductor  =cc,)

def  _from_nodal_mobility(device :  Device , carrier  :   Carrier , nodal  :  npt.NDArray[  np.float64  ], field_dependent   :  bool,)   ->  EdgeDiffusivity :
    val   =   device.scaled_mesh.geometry.edge_nodes
    r  =  C.V_T( device.material.T )
    a  =edge_diffusivity(nodal, r, val) / device.scale.D_0; return  _wrapped_in_saturation( device,   carrier,  a ,  field_dependent )




def _wrapped_in_saturation(device :Device, carrier  :Carrier, low_field  :  Diffusivity, field_dependent : bool,)  ->  EdgeDiffusivity  :
    if not field_dependent:
        return low_field

    m2 = device.scale


    bb  = device.material.T
    r=carrier is Carrier.ELECTRON
    yy =   C.v_sat_n ( bb)   if r else C.v_sat_p(  bb)


    return CaugheyThomas(
        low_field =np.broadcast_to(
            np.asarray(low_field,dtype=np.float64),(device.scaled_mesh.h.size,)
        ).copy(),
        v_sat = yy * m2.x_0/m2.D_0,
        beta= C.BETA_N if r else C.BETA_P,
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

    e  = SparseLU(  )

    def step(state : DeviceState)  ->  tuple[DeviceState, float] :
        c =solve_poisson(device,state.psi.data,state.phi_n,state.phi_p,solver=e)
        if not c.converged  :
            raise TransportError(
                f"the Poisson block did not converge: {c.message}",state
            )


        v=c.x - state.psi.data

        with np.errstate(over="ignore",under='ignore'):


            n = state.n.data *np.exp(v);  p=state.p.data*np.exp(- v)

        if not(np.all(np.isfinite(n))and np.all(np.isfinite(p))) :
            raise TransportError(
                f"the potential moved by {np.max(np.abs(v)):.3g} V_T in one "
                "cycle and overflowed the Boltzmann densities. Ramp the bias in "
                'smaller steps.',
                state,
            )

        ok =  replace(state, psi = _node_field(c.x, 'V', 'psi'), n =  _node_field(n, 'cm^-3', "n"), p=  _node_field(p, "cm^-3", 'p'), newton= c,)
        return ok, float(np.max(np.abs(v)))
    return step



def _lagged_effective_potential(
    device :  Device, state: DeviceState, carrier:  Carrier
) -> Field :
    bb=device.degeneracy
    if bb is  None  :
        return state.psi
    if  carrier is Carrier.ELECTRON :
        e=  bb.electron_potential(state.psi.data, state.n.data)
    else:

        e= bb.hole_potential(state.psi.data,state.p.data)
    return _node_field(np.asarray(e),
      'V',
              'psi_eff')


def  electron_block (
    device  : Device,  models  :   TransportModels
)   ->   BlockStep[DeviceState ]  :
    tmp3 = device.net_doping_scaled.data
    c  = device.degeneracy
    it= SparseLU()
    def step(state  :DeviceState)->  tuple[DeviceState, float]:

        val =assemble_electron_continuity(device.mesh_1d, _lagged_effective_potential(device,state,Carrier.ELECTRON), state.n, state.p, models.recombination, device.scale, _lagged_diffusivity(models.Dn,device,state.psi.data),)
        val= apply_ohmic_densities(
            val,
            state.n.data,
            tmp3,
            device.ohmic_contacts,
            Carrier.ELECTRON,
            c,
        )

        it.factorize(val.rows, val.cols, val.values, val.shape)
        v  =  impose_ohmic_densities(state.n.data  + it.solve(- val.residual), tmp3, device.ohmic_contacts, Carrier.ELECTRON, c,)

        _check_positive(v,  'n',  state)
        return(replace(state, n =  _node_field(v, "cm^-3", 'n')), _density_update(state.n.data, v),)

    return step




def hole_block(device :Device,models:TransportModels)->BlockStep[DeviceState]:
    m =  device.net_doping_scaled.data; e=device.degeneracy
    b  =   SparseLU( )
    def step(state :DeviceState) -> tuple[DeviceState, float]:
        y  =   assemble_hole_continuity(
            device.mesh_1d ,
            _lagged_effective_potential( device, state,  Carrier.HOLE ),
            state.n,
            state.p,
            models.recombination,
            device.scale,
            _lagged_diffusivity(models.Dp, device, state.psi.data),
        )
        y=apply_ohmic_densities(
            y,
            state.p.data,
            m,
            device.ohmic_contacts,
            Carrier.HOLE,
            e,
        )

        b.factorize(y.rows,y.cols,y.values,y.shape)
        f = impose_ohmic_densities(state.p.data+b.solve(- y.residual), m, device.ohmic_contacts, Carrier.HOLE, e,)
        _check_positive(f,"p",state)
        return(
            replace(state, p = _node_field(f, "cm^-3", "p")),
            _density_update(state.p.data, f),
        )

    return step


def  _check_positive (
    density   :  npt.NDArray [  np.float64  ] , name : str ,   state : DeviceState
)   ->  None   :
    if np.all(density>0.0):
        return

    h =int(np.argmin(density))
    raise TransportError(
        f"{name} came out non-positive at node {h}, value "
        f"{density[h]:.3e}. The continuity matrix should be an M-matrix "
        "with a non-negative right hand side, so check signs before anything "
        'else, per references/pitfalls.md. Do not clamp.',
        state,
    )

def initial_state(device : Device)->DeviceState:
    return solve_equilibrium(device,frozen_quasi_fermi(device))

def _low_field_models(models: TransportModels) ->  TransportModels  :
    b2 =tuple(
        y.low_field if isinstance(y, CaugheyThomas)else y
        for y in(models.Dn, models.Dp)
    )
    return replace(
        models,
        Dn  = b2[0],
        Dp  =  b2[1],
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
    g  =  0.0
    for rr,y in((before.Dn,after.Dn),(before.Dp,after.Dp)):
        h, s =  _low_field_edges(rr), _low_field_edges(y)

        e,a= np.abs(s -h),np.abs(h)


        dat=np.divide(e,a,out=np.where(e>0.0,np.inf,0.0),where = a > 0.0)
        g=max(g,float(np.max(dat)))

    return g



def _surface_fixed_point(
    device  : Device,
    models : TransportModels,
    run :Callable[[TransportModels, npt.NDArray[np.float64]], NewtonResult],
    x0: npt.NDArray[np.float64],
    max_sweeps: int,
    rtol:float,
) ->NewtonResult:
    obj=models.at_state(device,*unpack(x0))
    idx  =  x0 ; cc=0
    b: list[float] =[]

    r :list[float]=[]
    t=0
    c =0


    while c<max_sweeps:
        c+=1
        w2   =   run (  obj ,  idx)

        cc+=w2.iterations
        b.extend(w2.residual_history)


        r.extend(w2.update_history)
        t+=w2.limited_steps
        idx  =   w2.x

        m2   =   replace (w2 , iterations  =   cc, residual_history = b, update_history =   r, limited_steps  = t,)


        if not w2.converged:
            return m2


        s= obj.at_state(device,*unpack(idx))
        if _surface_moved(obj, s)  < rtol:
            return m2
        obj =  s

    return replace(
        m2,
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
    u: dict[str,dict[str,float]]={}

    def residual_norm(
        residual :  npt.NDArray[np.float64], x:npt.NDArray[np.float64]
    ) ->  float:
        r   =  residual_by_family(residual,   x)
        u["residual"]= r;  return max(0.0, * r.values())

    def update_norm(
        delta : npt.NDArray[np.float64], x  :  npt.NDArray[np.float64]
    )  -> float :
        f =coupled_update_by_family(delta,x)
        u ["update" ]   =   f
        return  max(  f.values ( ))

    if on_frame is None:
        return residual_norm, update_norm, None
    j =on_frame
    def  matching (kind :  str ,   value :   float  | None  )   -> dict [  str ,  float  ]  |  None  :
        rows = u.get(kind)
        if rows is None or value is None or max(rows.values())  !=  value :
            return  None
        return rows
    def report(frame : NewtonIteration)  -> None :

        j(replace(frame, residual_by_family  =  matching('residual', frame.residual), update_by_family =  matching('update', frame.update),))

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
    b=initial_state(device) if guess is None else guess
    vals = device.scale

    res2=device.scaled_mesh


    obj=res2.h
    a=device.charge_volume_scaled
    i   = res2.geometry
    g   =   device.net_doping_scaled.data
    t  =   device.carrier_free_nodes

    item =   pack( b.psi.data,  b.n.data, b.p.data  )


    def assembler(
        active  : TransportModels,
    ) -> Callable[[npt.NDArray[np.float64]], SparseAssembly]  :
        def assemble(x : npt.NDArray[np.float64]) ->SparseAssembly :
            v, dat = assemble_coupled_terms(
                obj,
                a,
                x,
                g,
                active.Dn,
                active.Dp,
                active.recombination,
                i,
                device.degeneracy,
            )
            v  = apply_contacts_coupled (v , x , g, device.contacts, vals, t, device.material.T, device.degeneracy,)
            return scale_rows(v, row_weights(dat, res2.n_nodes))
        return assemble


    def measured(active :  TransportModels  )   ->   Callable[[npt.NDArray[np.float64  ],   npt.NDArray[np.float64  ] ], dict[str,  float ]] :

        def norm(
            residual: npt.NDArray[np.float64],x: npt.NDArray[np.float64]
        )->dict[str,float] :
            _ ,  n,   p   =   unpack( x )
            s =residual_term_scales(
                obj,
                a,
                x,
                g,
                active.Dn,
                active.Dp,
                np.asarray(active.recombination.rate(n,p),dtype =np.float64),
                i,
                device.degeneracy,
            )
            return residual_measure_by_family(residual, s, res2.n_nodes)



        return  norm

    def run(active :TransportModels,x :npt.NDArray[np.float64])->NewtonResult:
        u,mm,k2=_reported_by_family(
            measured(active),on_frame
        )
        return newton_solve(
            assembler(active),
            x,
            limit=lambda delta : limit_psi_step(delta, max_psi_step),
            residual_scale=1.0,
            residual_norm = u,
            residual_rtol = residual_rtol,
            update_tol= update_tol,
            update_norm  =  mm,
            max_iterations =max_iterations,
            on_iteration =  k2,
        )
    def solve_with(active:TransportModels,x : npt.NDArray[np.float64]) -> NewtonResult:
        if active.surface is None :
            return  run(  active, x  )

        return  _surface_fixed_point(
            device,   active, run, x, max_surface_sweeps,  surface_rtol
        )


    m  :  NewtonResult   | None  =  None
    if _needs_a_low_field_prelude(models,guess):
        m=  solve_with(_low_field_models(models), item); item  =   m.x
    d = solve_with(models,item)
    if m is not None :
        d=replace(d, iterations=d.iterations+ m.iterations, residual_history=m.residual_history+d.residual_history, update_history= m.update_history+d.update_history, limited_steps=d.limited_steps+m.limited_steps,)
    psi, n, p =unpack(d.x)

    return DeviceState(psi =  _node_field(psi.copy(), 'V', 'psi'), n=  _node_field(n.copy(), "cm^-3", "n"), p  =_node_field(p.copy(), 'cm^-3', "p"), newton=  d, degeneracy= device.degeneracy,)

def solve_bias_ramped(
    device  :   Device ,
    models  : TransportModels |   None  =   None ,
    step  :  float  = 0.25,
    max_iterations   :   int  =   30 ,
    on_frame  :  Callable[[  object],  None  ]  |  None  =  None,
)  ->   DeviceState  :
    if models is None:
        models= TransportModels.for_device(device)


    d={dat.name: dat.voltage for dat in device.contacts}

    def at_fraction(fraction: float,guess: DeviceState| None) ->DeviceState|None:
        tmp3 = solve_bias_newton(
            device.with_bias(
                ** {arr: fraction* k for arr, k in d.items()}
            ),
            models = models,
            guess =guess,
            max_iterations  =  max_iterations,
            on_frame  = on_frame,
        )
        assert tmp3.newton is not None
        return tmp3 if tmp3.newton.converged else None

    x=solve_bias_newton(
        device.with_bias(**dict.fromkeys(d,0.0)),
        models=models,
        max_iterations=max_iterations,
        on_frame =on_frame,
    )

    buf  =  continue_to (
        at_fraction,
        start  = 0.0,
        target  = 1.0,
        initial  =   x,
        step  =   step ,
        on_event  =  on_frame,
    )


    return solve_bias_newton (
        device,
        models  =  models,
        guess =  buf.solution,
        max_iterations  =   max_iterations,
        on_frame =  on_frame,
    )

def _gummel_prelude(device  :Device, models :  TransportModels, state : DeviceState, cycles : int, on_frame : Callable[[object], None]  | None  = None,)->  DeviceState :
    if cycles <=0:
        return state
    xs =[
        poisson_block(device),
        electron_block(device,models),
        hole_block(device,models),
    ]
    try :
        y=gummel_solve(
            state,
            xs,
            update_tol=1e-300,
            max_iterations = cycles,
            on_iteration= on_frame,
        )

    except TransportError as g :
        return g.state
    return replace(y.state,gummel =y)


def solve_bias_hybrid(device  :   Device, models : TransportModels   |  None =  None, guess   :   DeviceState |  None  = None, gummel_cycles :  int  =   3, retry_cycles  :   int   =  5, max_psi_step  : float = 5.0, max_iterations :   int   =  30, on_frame   : Callable[[ object],  None  ]  |   None = None ,)  ->  DeviceState :

    if  models is  None  :
        models= TransportModels.for_device(device)

    j =initial_state(device)if guess is None else guess
    def newton_from(state :DeviceState) -> DeviceState  :
        return solve_bias_newton(device, models  = models, guess  = state, max_psi_step= max_psi_step, max_iterations=max_iterations, on_frame  =on_frame,)
    rr  =  _gummel_prelude( device, models , j,  gummel_cycles ,  on_frame) ; d2=newton_from(rr)

    assert d2.newton is not None
    if d2.newton.converged or retry_cycles<=0:
        return replace(d2, gummel= rr.gummel)

    rr=_gummel_prelude(device,models,rr,retry_cycles,on_frame)
    return replace(newton_from(rr), gummel  =  rr.gummel)
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
    k  =  initial_state(device)  if  guess is None  else  guess
    w= [
        poisson_block(device),
        electron_block(device,models),
        hole_block(device,models),
    ]

    try :
        r  = gummel_solve(k, w, update_tol  = update_tol, max_iterations =  max_iterations, on_iteration = on_frame,)
    except TransportError  as rr  :
        return replace(rr.state, gummel= GummelResult(state= rr.state, converged= False, iterations  = 0, message  = str(rr),),)
    return replace(r.state,gummel = r)
