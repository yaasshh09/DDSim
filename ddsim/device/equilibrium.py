from __future__ import annotations
from collections.abc import Callable
import numpy as np, numpy.typing as npt
from ddsim.core.field import Field,Location,ScalingState
from ddsim.device.builder import Device
from ddsim.device.state import DeviceState
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
    k =   device.net_doping.data

    e  =device.mesh.n_nodes

    j  =[a for a in device.contacts if not isinstance(a, GateContact)]; d = [a for a in j if k[a.nodes[0]] >= 0.0]
    yy  = [a for a in j if k[a.nodes[0]]  <  0.0]

    it  =   d[ 0 ].voltage if d else yy[  0  ].voltage
    y   =   yy[ 0 ].voltage if  yy  else d[0  ].voltage
    phi_n =Field(np.full(e,it/device.scale.psi_0), "V", ScalingState.SCALED, Location.NODE, name="phi_n",)
    phi_p=Field(
        np.full(e,y/device.scale.psi_0),
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

    xx =  device.scale
    f   =  device.net_doping_scaled
    w2=Field(psi, "V", ScalingState.SCALED, Location.NODE, name =  'psi')

    tmp= assemble_poisson(
        device.scaled_mesh,
        w2,
        f,
        charge_volume=device.charge_volume_scaled,
        degeneracy =device.degeneracy,
    )
    tmp =  apply_contacts(
        tmp,
        psi,
        f.data,
        device.contacts,
        xx ,
        device.material.T,
        device.degeneracy ,
    )



    m= np.flatnonzero(device.regions.semiconductor_volume> 0.0)

    tmp= apply_dirichlet_nodes(
        tmp, psi, m.tolist(), psi[m].tolist()
    )
    bb  =  SparseLU ( )
    bb.factorize(
        tmp.rows,tmp.cols,tmp.values,tmp.shape
    )
    return psi   +   bb.solve(-  tmp.residual )


def solve_poisson(device  :  Device, psi_initial  :   npt.NDArray [ np.float64 ], phi_n  : Field | None  =   None, phi_p   :  Field  |  None =   None, max_iterations  :  int   =  50, residual_rtol   :  float =  1e-10, update_tol  : float  =  1e-10, solver  :   SparseLU  | None  =  None , on_frame  :   Callable[ [ object  ] ,  None]  | None   =  None,)   ->  NewtonResult   :
    ss=device.scale
    hh  = device.scaled_mesh
    f =device.charge_volume_scaled
    cur = device.net_doping_scaled

    tmp2=cur.data
    s  = device.degeneracy
    def assemble(psi_values : npt.NDArray[np.float64])->SparseAssembly :
        psi=Field(psi_values,'V',ScalingState.SCALED,Location.NODE,name = "psi")
        rows =  assemble_poisson(hh, psi, cur, phi_n, phi_p, charge_volume=f, degeneracy  = s,)
        return apply_contacts(
            rows,
            psi_values,
            tmp2,
            device.contacts,
            ss,
            device.material.T,
            s,
        )
    item=float(np.max(np.abs(tmp2) * f))
    y , v = hh.geometry.ends ( hh.n_edges )
    el = np.maximum(np.abs(psi_initial[y]), np.abs(psi_initial[v]))
    aa= FLUX_FLOOR_MARGIN* EPS*float(np.max(hh.geometry.weight*el /hh.h))

    arr=item
    if residual_rtol >  0.0 and residual_rtol  *   item  <   aa  :
        arr  =aa/ residual_rtol
    return  newton_solve (assemble, psi_initial , max_step  =  MAX_PSI_STEP, residual_rtol  =  residual_rtol, residual_scale =  arr , update_tol =  update_tol, max_iterations   =  max_iterations, solver  =  solver, on_iteration   = on_frame ,)

def solve_equilibrium(device : Device, quasi_fermi  :  tuple[Field, Field] | None = None, max_iterations : int =50, residual_rtol  :  float= 1e-10, update_tol  : float = 1e-10, on_frame :Callable[[object], None] | None  = None,)  -> DeviceState :
    phi_n, phi_p = (None, None) if quasi_fermi is None else quasi_fermi
    cc =  device.net_doping_scaled.data

    b2 = device.degeneracy
    if b2 is None:
        item= np.asarray(psi_equilibrium_scaled(cc),dtype=np.float64)
    else :
        item = np.asarray(
            b2.equilibrium_psi(cc), dtype =np.float64
        )
    if phi_n is not None and phi_p is not None:
        k= np.where(cc >=0.0,
            phi_n.data,
                      phi_p.data)
        item   =  item  + k

    item= insulator_guess(device,item)

    w =solve_poisson(
        device,
        item,
        phi_n,
        phi_p,
        max_iterations =max_iterations,
        residual_rtol =residual_rtol,
        update_tol=update_tol,
        on_frame=on_frame,
    )

    if not  w.converged  :
        raise RuntimeError(
            f"equilibrium solve did not converge: {w.message}. "
            f"Residual history: {w.residual_history}"
        )


    w2=0.0 if phi_n is None else phi_n.data
    yy  =0.0 if phi_p is None else phi_p.data;psi =  Field(w.x, "V", ScalingState.SCALED, Location.NODE, name= "psi")

    arr =np.asarray(device.charge_volume_scaled)> 0.0
    with np.errstate(over= "ignore"):
        if b2 is None  :
            s = np.asarray(n_boltzmann_scaled(w.x,w2))
            t  =  np.asarray ( p_boltzmann_scaled (  w.x,  yy  ))
        else:
            s=b2.electron_density(w.x - w2)

            t = b2.hole_density(yy-w.x)
        tmp=np.where(arr,s,0.0)
        tmp2   =  np.where(  arr,  t,  0.0)



    return  DeviceState (
        psi  =  psi,
        n  = Field (tmp, "cm^-3" , ScalingState.SCALED ,  Location.NODE,   name =  "n"),
        p  =   Field ( tmp2,   "cm^-3" ,   ScalingState.SCALED,  Location.NODE,   name   = "p" ),
        newton  =  w ,
        degeneracy = b2 ,
    )
