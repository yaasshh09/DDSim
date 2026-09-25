from __future__ import annotations

from collections.abc import Callable

import  numpy as np, numpy.typing as npt
from ddsim.core.field import Field,Location,ScalingState

from ddsim.device.builder import Device

from ddsim.device.state  import  DeviceState
from  ddsim.discretize.assembly  import SparseAssembly



from ddsim.discretize.boundary import(GateContact, apply_contacts, apply_dirichlet_nodes,)


from ddsim.discretize.poisson import assemble_poisson
from ddsim.physics.statistics import(n_boltzmann_scaled, p_boltzmann_scaled, psi_equilibrium_scaled,)



from ddsim.solve.linear import SparseLU

from ddsim.solve.newton import NewtonResult,newton_solve


MAX_PSI_STEP= 5.0

EPS=  float(np.finfo(np.float64).eps)


FLUX_FLOOR_MARGIN =  16.0
def frozen_quasi_fermi(device : Device)  -> tuple[Field, Field] :
    Doping =   device.net_doping.data

    NNodes  =device.mesh.n_nodes

    ohm  =[cc for cc in device.contacts if not isinstance(cc, GateContact)]; buf = [cc for cc in ohm if Doping[cc.nodes[0]] >= 0.0]
    ps  = [cc for cc in ohm if Doping[cc.nodes[0]]  <  0.0]

    nbias  =   buf[ 0 ].voltage if buf else ps[  0  ].voltage
    pb   =   ps[ 0 ].voltage if  ps  else buf[0  ].voltage
    phi_n =Field(np.full(NNodes,nbias/device.scale.psi_0), "V", ScalingState.SCALED, Location.NODE, name="phi_n",)
    phi_p=Field(
        np.full(NNodes,pb/device.scale.psi_0),
        'V',
        ScalingState.SCALED,
        Location.NODE,
        name ="phi_p",
    )
    return phi_n,phi_p




def insulator_guess(
    device : Device, psi  : npt.NDArray[np.float64]
)->  npt.NDArray[np.float64]  :

    if device.regions is None or device.regions.oxide_nodes.size == 0:
        return psi

    Scale =  device.scale
    net_dooping   =  device.net_doping_scaled
    feild=Field(psi, "V", ScalingState.SCALED, Location.NODE, name =  'psi')

    asembly= assemble_poisson(
        device.scaled_mesh,
        feild,
        net_dooping,
        charge_volume=device.charge_volume_scaled,
        degeneracy =device.degeneracy,
    )
    asembly =  apply_contacts(
        asembly,
        psi,
        net_dooping.data,
        device.contacts,
        Scale ,
        device.material.T,
        device.degeneracy ,
    )



    Held= np.flatnonzero(device.regions.semiconductor_volume> 0.0)

    asembly= apply_dirichlet_nodes(
        asembly, psi, Held.tolist(), psi[Held].tolist()
    )
    tuple  =  SparseLU ( )
    tuple.factorize(
        asembly.rows,asembly.cols,asembly.values,asembly.shape
    )
    return psi   +   tuple.solve(-  asembly.residual )


def solve_poisson(device  :  Device, psi_initial  :   npt.NDArray [ np.float64 ], phi_n  : Field | None  =   None, phi_p   :  Field  |  None =   None, max_iterations  :  int   =  50, residual_rtol   :  float =  1e-10, update_tol  : float  =  1e-10, solver  :   SparseLU  | None  =  None , on_frame  :   Callable[ [ object  ] ,  None]  | None   =  None,)   ->  NewtonResult   :
    buff=device.scale
    meesh  = device.scaled_mesh
    chargevolume =device.charge_volume_scaled
    NetDoping = device.net_doping_scaled

    dv=NetDoping.data
    deg  = device.degeneracy
    def assemble(psi_values : npt.NDArray[np.float64])->SparseAssembly :
        psi=Field(psi_values,'V',ScalingState.SCALED,Location.NODE,name = "psi")
        assembly =  assemble_poisson(meesh, psi, NetDoping, phi_n, phi_p, charge_volume=chargevolume, degeneracy  = deg,)
        return apply_contacts(
            assembly,
            psi_values,
            dv,
            device.contacts,
            buff,
            device.material.T,
            deg,
        )
    min=float(np.max(np.abs(dv) * chargevolume))
    dir , Right = meesh.geometry.ends ( meesh.n_edges )
    ep = np.maximum(np.abs(psi_initial[dir]), np.abs(psi_initial[Right]))
    fluxfloor= FLUX_FLOOR_MARGIN* EPS*float(np.max(meesh.geometry.weight*ep /meesh.h))

    residal_scale=min
    if residual_rtol >  0.0 and residual_rtol  *   min  <   fluxfloor  :
        residal_scale  =fluxfloor/ residual_rtol
    return  newton_solve (assemble, psi_initial , max_step  =  MAX_PSI_STEP, residual_rtol  =  residual_rtol, residual_scale =  residal_scale , update_tol =  update_tol, max_iterations   =  max_iterations, solver  =  solver, on_iteration   = on_frame ,)

def solve_equilibrium(device : Device, quasi_fermi  :  tuple[Field, Field] | None = None, max_iterations : int =50, residual_rtol  :  float= 1e-10, update_tol  : float = 1e-10, on_frame :Callable[[object], None] | None  = None,)  -> DeviceState :
    phi_n, phi_p = (None, None) if quasi_fermi is None else quasi_fermi
    dopingvalues =  device.net_doping_scaled.data

    degeneeracy = device.degeneracy
    if degeneeracy is None:
        id= np.asarray(psi_equilibrium_scaled(dopingvalues),dtype=np.float64)
    else :
        id = np.asarray(
            degeneeracy.equilibrium_psi(dopingvalues), dtype =np.float64
        )
    if phi_n is not None and phi_p is not None:
        out2= np.where(dopingvalues >=0.0,
            phi_n.data,
                      phi_p.data)
        id   =  id  + out2

    id= insulator_guess(device,id)

    sum =solve_poisson(
        device,
        id,
        phi_n,
        phi_p,
        max_iterations =max_iterations,
        residual_rtol =residual_rtol,
        update_tol=update_tol,
        on_frame=on_frame,
    )

    if not  sum.converged  :
        raise RuntimeError(
            f"equilibrium solve did not converge: {sum.message}. "
            f"Residual history: {sum.residual_history}"
        )


    nlevel=0.0 if phi_n is None else phi_n.data
    pl  =0.0 if phi_p is None else phi_p.data;psi =  Field(sum.x, "V", ScalingState.SCALED, Location.NODE, name= "psi")

    xx =np.asarray(device.charge_volume_scaled)> 0.0
    with np.errstate(over= "ignore"):
        if degeneeracy is None  :
            NRaw = np.asarray(n_boltzmann_scaled(sum.x,nlevel))
            dir  =  np.asarray ( p_boltzmann_scaled (  sum.x,  pl  ))
        else:
            NRaw=degeneeracy.electron_density(sum.x - nlevel)

            dir = degeneeracy.hole_density(pl-sum.x)
        nData=np.where(xx,NRaw,0.0)
        PData   =  np.where(  xx,  dir,  0.0)



    return  DeviceState (
        psi  =  psi,
        n  = Field (nData, "cm^-3" , ScalingState.SCALED ,  Location.NODE,   name =  "n"),
        p  =   Field ( PData,   "cm^-3" ,   ScalingState.SCALED,  Location.NODE,   name   = "p" ),
        newton  =  sum ,
        degeneracy = degeneeracy ,
    )
