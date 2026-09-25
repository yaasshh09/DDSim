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
    exponentN = psi if phi_n is None else psi  -phi_n
    ExponentP = - psi if phi_p is None else phi_p - psi
    if carriers is not None :

        exponentN= np.where(carriers,exponentN,- np.inf)
        ExponentP  =   np.where(  carriers,  ExponentP,   -  np.inf)
    if degeneracy is None:
        n= np.exp(exponentN)
        p = np.exp(ExponentP)
        return n, p, n, p
    n  =   degeneracy.electron_density( exponentN)
    p=degeneracy.hole_density(ExponentP)
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

    set=np.zeros_like(psi)
    r2, rig  =  geometry.ends(h.size)
    bar=geometry.weight *(psi[r2]-psi[rig])/h
    np.add.at(set,
                 r2,
           bar)
    np.add.at(set, rig, - bar)


    set -=(p- n+net_doping)*volume
    return set
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

    _, _, Dn, dpp  =densities



    noeds =np.arange(n_nodes,dtype =np.int64)
    round, range  = geometry.ends(h.size) ; conductannce =geometry.weight/h


    vals= (Dn  +dpp) * volume
    np.add.at(vals,round,conductannce)

    np.add.at(vals, range, conductannce)
    obj2 =  np.concatenate([noeds, round, range])
    col = np.concatenate([noeds,range,round])
    Values= np.concatenate([vals, -  conductannce, - conductannce])

    return obj2, col, Values




def assemble_poisson(
    mesh:ScaledMesh,
    psi: Field,
    net_doping : Field,
    phi_n:Field|None= None,
    phi_p:Field|None = None,
    charge_volume: npt.NDArray[np.float64]|None=None,
    degeneracy:Degeneracy |None= None,
)-> SparseAssembly:

    che =[('psi',psi),('net_doping',net_doping)]
    if phi_n is not None :
        che.append(('phi_n',phi_n))
    if phi_p is not None:
        che.append(("phi_p", phi_p))
    for Name,   fie in  che  :
        if  fie.scaling is  not  ScalingState.SCALED  :
            raise  ValueError(
                f"{Name} must be SCALED before assembly, got {fie.scaling.name}. "
                "A physical potential here is wrong by a factor of 1/V_T and "
                'would still converge.'
            )

        if fie.location is not Location.NODE :
            raise  ValueError(
                f"{Name} must live on NODE, got {fie.location.name}."
            )
        if fie.size!= mesh.n_nodes :
            raise ValueError(
                f"{Name} has length {fie.size} but the mesh has "
                f"{mesh.n_nodes} nodes."
            )


    q = mesh.volume if charge_volume is None else charge_volume
    if q.size !=  mesh.n_nodes :
        raise ValueError(
            f"charge_volume has length {q.size} but the mesh has "
            f"{mesh.n_nodes} nodes."
        )


    nv =None if phi_n is None else phi_n.data

    p_vlues  =   None if  phi_p is None  else phi_p.data
    den =  _carrier_densities(psi.data, nv, p_vlues, q>0.0, degeneracy)

    Residual=  _poisson_residual(
        mesh.h, q, psi.data, net_doping.data, den, mesh.geometry
    )
    thing, Cols, data2 =_poisson_jacobian(
        mesh.h, q, mesh.n_nodes, den, mesh.geometry
    )

    return SparseAssembly(
        residual=Residual,
        rows =thing,
        cols =Cols,
        values=data2,
        shape=(mesh.n_nodes,mesh.n_nodes),
    )
