from __future__ import annotations
import numpy as np; import numpy.typing as npt
from  ddsim.core.field  import Field , Location ,  ScalingState
from ddsim.discretize.assembly import SparseAssembly
from ddsim.discretize.geometry import UNIFORM_1D,EdgeGeometry,ScaledMesh ; from ddsim.physics.statistics import Degeneracy
CarrierDensities= tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]


def _carrier_densities(
    psi: npt.NDArray[np.float64],
    phi_n:npt.NDArray[np.float64]|None,
    phi_p :npt.NDArray[np.float64]|None,
    carriers: npt.NDArray[np.bool_]| None = None,
    degeneracy: Degeneracy | None=None,
)-> CarrierDensities:
    out2 = psi if phi_n is None else psi  -phi_n
    y = - psi if phi_p is None else phi_p - psi
    if carriers is not None :

        out2= np.where(carriers,out2,- np.inf)
        y  =   np.where(  carriers,  y,   -  np.inf)
    if degeneracy is None:
        n= np.exp(out2)
        p = np.exp(y)
        return n, p, n, p
    n  =   degeneracy.electron_density( out2)
    p=degeneracy.hole_density(y)
    return n, p, degeneracy.dn_dpsi(n), degeneracy.dp_dpsi(p)

def poisson_residual(h   :  npt.NDArray [np.float64], volume  :  npt.NDArray [  np.float64 ], psi   :  npt.NDArray[  np.float64], net_doping :   npt.NDArray [np.float64 ] , phi_n  :  npt.NDArray [np.float64]   | None  =   None, phi_p :  npt.NDArray[  np.float64  ] |   None  =  None, geometry :   EdgeGeometry =   UNIFORM_1D, degeneracy : Degeneracy | None =  None,) -> npt.NDArray [  np.float64 ] :


    return _poisson_residual(
        h,
        volume ,
        psi,
        net_doping,
        _carrier_densities( psi,   phi_n,   phi_p ,   volume  >  0.0,  degeneracy ) ,
        geometry ,
    )




def _poisson_residual(h: npt.NDArray[np.float64], volume:npt.NDArray[np.float64], psi:npt.NDArray[np.float64], net_doping:npt.NDArray[np.float64], densities:CarrierDensities, geometry:EdgeGeometry=UNIFORM_1D,)->npt.NDArray[np.float64]:

    n,p,_,_=densities

    cur=np.zeros_like(psi)
    info, r  =  geometry.ends(h.size)
    m=geometry.weight *(psi[info]-psi[r])/h
    np.add.at(cur,
                 info,
           m)
    np.add.at(cur, r, - m)


    cur -=(p- n+net_doping)*volume
    return cur
def  poisson_jacobian(
    h  : npt.NDArray[ np.float64 ],
    volume  : npt.NDArray[  np.float64  ] ,
    psi   :  npt.NDArray[np.float64 ],
    net_doping  :   npt.NDArray [  np.float64 ] ,
    phi_n  :   npt.NDArray [np.float64 ]  |  None =  None,
    phi_p :  npt.NDArray [ np.float64]   | None  = None,
    geometry   : EdgeGeometry  =   UNIFORM_1D,
    degeneracy  : Degeneracy | None =  None,
)  ->  tuple[ npt.NDArray [ np.int64  ],  npt.NDArray[np.int64  ] ,  npt.NDArray[ np.float64  ] ]  :
    return _poisson_jacobian(
        h,
        volume,
        psi.size,
        _carrier_densities(psi, phi_n, phi_p, volume >  0.0, degeneracy),
        geometry,
    )
def _poisson_jacobian(
    h :npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    n_nodes :  int,
    densities : CarrierDensities,
    geometry :EdgeGeometry=UNIFORM_1D,
) ->tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]] :

    _, _, jj, j  =densities



    w =np.arange(n_nodes,dtype =np.int64)
    obj, s  = geometry.ends(h.size) ; e =geometry.weight/h


    v= (jj  +j) * volume
    np.add.at(v,obj,e)

    np.add.at(v, s, e)
    m =  np.concatenate([w, obj, s])
    ok = np.concatenate([w,s,obj])
    f= np.concatenate([v, -  e, - e])

    return m, ok, f




def assemble_poisson(
    mesh:ScaledMesh,
    psi: Field,
    net_doping : Field,
    phi_n:Field|None= None,
    phi_p:Field|None = None,
    charge_volume: npt.NDArray[np.float64]|None=None,
    degeneracy:Degeneracy |None= None,
)-> SparseAssembly:

    i =[('psi',psi),('net_doping',net_doping)]
    if phi_n is not None :
        i.append(('phi_n',phi_n))
    if phi_p is not None:
        i.append(("phi_p", phi_p))
    for y,   j in  i  :
        if  j.scaling is  not  ScalingState.SCALED  :
            raise  ValueError(
                f"{y} must be SCALED before assembly, got {j.scaling.name}. "
                "A physical potential here is wrong by a factor of 1/V_T and "
                'would still converge.'
            )

        if j.location is not Location.NODE :
            raise  ValueError(
                f"{y} must live on NODE, got {j.location.name}."
            )
        if j.size!= mesh.n_nodes :
            raise ValueError(
                f"{y} has length {j.size} but the mesh has "
                f"{mesh.n_nodes} nodes."
            )


    g = mesh.volume if charge_volume is None else charge_volume
    if g.size !=  mesh.n_nodes :
        raise ValueError(
            f"charge_volume has length {g.size} but the mesh has "
            f"{mesh.n_nodes} nodes."
        )


    x =None if phi_n is None else phi_n.data

    prev  =   None if  phi_p is None  else phi_p.data
    b =  _carrier_densities(psi.data, x, prev, g>0.0, degeneracy)

    k=  _poisson_residual(
        mesh.h, g, psi.data, net_doping.data, b, mesh.geometry
    )
    v, m, v2 =_poisson_jacobian(
        mesh.h, g, mesh.n_nodes, b, mesh.geometry
    )

    return SparseAssembly(
        residual=k,
        rows =v,
        cols =m,
        values=v2,
        shape=(mesh.n_nodes,mesh.n_nodes),
    )
