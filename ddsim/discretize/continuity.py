from __future__ import annotations
import numpy as np, numpy.typing as npt
from ddsim.core.field import Field, Location, ScalingState
from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.assembly import SparseAssembly; from ddsim.discretize.geometry import UNIFORM_1D,EdgeGeometry
from ddsim.mesh.mesh1d import Mesh1D
from ddsim.physics.bernoulli import B
from ddsim.physics.recombination import RecombinationModel
Diffusivity= float  |  npt.NDArray[np.float64]



BernoulliPair =tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]


def _bernoulli_pair(
    psi:npt.NDArray[np.float64],geometry :EdgeGeometry= UNIFORM_1D
)-> BernoulliPair:


    i, r2 =geometry.ends_of(psi.size)
    c = psi[r2]  -psi[i]

    return  np.asarray(  B(  c  ),   dtype   = np.float64 ),  np.asarray(B(  -   c),   dtype  =  np.float64)

def electron_current(h : npt.NDArray[np.float64], Dn: Diffusivity, psi: npt.NDArray[np.float64], n :npt.NDArray[np.float64], geometry : EdgeGeometry  = UNIFORM_1D,) ->  npt.NDArray[np.float64]:
    return  _electron_current(
        h,   Dn,  _bernoulli_pair (  psi,  geometry),  n,  geometry
    )


def _electron_current(
    h : npt.NDArray[np.float64],
    Dn  :  Diffusivity,
    bernoulli :BernoulliPair,
    n : npt.NDArray[np.float64],
    geometry : EdgeGeometry = UNIFORM_1D,
) -> npt.NDArray[np.float64]:
    m2, k2= bernoulli; v,t=geometry.ends(h.size)
    return np.asarray(
        (Dn*geometry.carrier_face/h)
        * (m2 * n[t]- k2* n[v])
    )


def hole_current(
    h  :  npt.NDArray [np.float64 ] ,
    Dp  :   Diffusivity,
    psi   :  npt.NDArray[ np.float64],
    p  : npt.NDArray[np.float64  ],
    geometry :  EdgeGeometry  =   UNIFORM_1D,
)   ->   npt.NDArray[  np.float64 ]  :

    return _hole_current(
        h, Dp, _bernoulli_pair(psi, geometry), p, geometry
    )



def _hole_current(
    h :npt.NDArray[np.float64],
    Dp:Diffusivity,
    bernoulli:BernoulliPair,
    p:npt.NDArray[np.float64],
    geometry: EdgeGeometry=UNIFORM_1D,
) ->npt.NDArray[np.float64] :
    yy, val = bernoulli
    out,   rows =  geometry.ends(  h.size )

    return np.asarray((Dp   *  geometry.carrier_face  / h ) *  (yy   * p [ out  ]   -  val   * p[  rows]  ))

def electron_continuity_residual(
    h:npt.NDArray[np.float64],
    volume:npt.NDArray[np.float64],
    Dn:Diffusivity,
    psi:npt.NDArray[np.float64],
    n: npt.NDArray[np.float64],
    R :npt.NDArray[np.float64],
    geometry:EdgeGeometry=UNIFORM_1D,
) ->npt.NDArray[np.float64]:
    return _electron_continuity_residual(h, volume, Dn, _bernoulli_pair(psi, geometry), n, R, geometry)

def _electron_continuity_residual(
    h: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    Dn:Diffusivity,
    bernoulli:BernoulliPair,
    n :npt.NDArray[np.float64],
    R : npt.NDArray[np.float64],
    geometry :EdgeGeometry=UNIFORM_1D,
) -> npt.NDArray[np.float64]:
    m=_electron_current(h,Dn,bernoulli,n,geometry)
    y,   i  =   geometry.ends ( h.size  )

    res2 =R *volume
    np.add.at(res2,y,-m)
    np.add.at(res2,
                  i,
          m)


    return np.asarray(res2)
def hole_continuity_residual(
    h   :   npt.NDArray [ np.float64  ],
    volume  :   npt.NDArray [ np.float64] ,
    Dp   :   Diffusivity,
    psi  :  npt.NDArray[  np.float64],
    p   :   npt.NDArray[  np.float64 ],
    R :  npt.NDArray[ np.float64],
    geometry   :   EdgeGeometry  = UNIFORM_1D,
) -> npt.NDArray[np.float64 ]  :
    return _hole_continuity_residual(
        h, volume, Dp, _bernoulli_pair(psi, geometry), p, R, geometry
    )
def _hole_continuity_residual(h : npt.NDArray[np.float64], volume : npt.NDArray[np.float64], Dp: Diffusivity, bernoulli :  BernoulliPair, p:npt.NDArray[np.float64], R  :npt.NDArray[np.float64], geometry:  EdgeGeometry =  UNIFORM_1D,) ->npt.NDArray[np.float64]:
    m = _hole_current(h, Dp, bernoulli, p, geometry)
    tt,g =geometry.ends(h.size)


    i  =   R  * volume
    np.add.at(i,tt,m)
    np.add.at(  i,   g ,   -   m)
    return np.asarray(i)




def electron_continuity_jacobian(
    h:  npt.NDArray[np.float64],
    volume  : npt.NDArray[np.float64],
    Dn:Diffusivity,
    psi :npt.NDArray[np.float64],
    dR_dn  : npt.NDArray[np.float64],
    geometry :EdgeGeometry  = UNIFORM_1D,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]] :
    return  _electron_continuity_jacobian(
        h,   volume ,  Dn,  _bernoulli_pair(psi,  geometry),   psi.size,   dR_dn , geometry
    )


def _electron_continuity_jacobian(
    h  :   npt.NDArray[np.float64  ] ,
    volume  :  npt.NDArray[ np.float64],
    Dn  : Diffusivity,
    bernoulli  :  BernoulliPair,
    n_nodes  : int ,
    dR_dn  : npt.NDArray [ np.float64],
    geometry :  EdgeGeometry  =  UNIFORM_1D ,
) ->   tuple[ npt.NDArray[ np.int64  ],   npt.NDArray[np.int64] , npt.NDArray[ np.float64  ]  ] :
    x, j = bernoulli
    bar  = np.asarray((Dn *geometry.carrier_face  / h) *  x);el = np.asarray( (Dn  *   geometry.carrier_face  /  h )   *  j  )

    tmp2  =   np.arange (n_nodes,   dtype =   np.int64)
    z2, e =  geometry.ends(h.size)

    y   =   dR_dn  *   volume
    np.add.at(  y,  z2,   el  )
    np.add.at(y,e,bar)

    out2= np.concatenate([tmp2, z2, e])
    a  =  np.concatenate (  [ tmp2 ,   e ,  z2 ])
    m =   np.concatenate (  [y,  -   bar ,   - el]  )
    return out2,a,m


def hole_continuity_jacobian(
    h : npt.NDArray[np.float64],
    volume :npt.NDArray[np.float64],
    Dp :Diffusivity,
    psi : npt.NDArray[np.float64],
    dR_dp :npt.NDArray[np.float64],
    geometry  :EdgeGeometry = UNIFORM_1D,
) ->  tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]] :

    return _hole_continuity_jacobian(
        h ,  volume, Dp, _bernoulli_pair(  psi, geometry  ) ,  psi.size ,   dR_dp,  geometry
    )


def  _hole_continuity_jacobian(h  :   npt.NDArray[  np.float64 ], volume   :   npt.NDArray[  np.float64], Dp  : Diffusivity , bernoulli   :   BernoulliPair, n_nodes  :  int, dR_dp  :  npt.NDArray[np.float64  ] , geometry : EdgeGeometry  =  UNIFORM_1D ,) ->   tuple[  npt.NDArray [np.int64] ,   npt.NDArray [  np.int64] , npt.NDArray[ np.float64  ]] :
    m,e=bernoulli

    idx=np.asarray((Dp* geometry.carrier_face / h)*m)

    c2 =np.asarray((Dp*geometry.carrier_face / h) *e)

    xx = np.arange(n_nodes, dtype  =  np.int64)
    j,k2 = geometry.ends(h.size)

    zz=dR_dp * volume
    np.add.at (  zz ,  j,   idx  )

    np.add.at(zz, k2, c2)


    a= np.concatenate([xx,j,k2])
    i  = np.concatenate([xx, k2, j])
    w =np.concatenate([zz, -c2, - idx])
    return a, i, w




def _check(mesh  :Mesh1D, named:tuple[tuple[str, Field], ...])-> None:
    for d2, mm in named :
        if mm.scaling is not ScalingState.SCALED :
            raise ValueError(
                f"{d2} must be SCALED before assembly, got {mm.scaling.name}. "
                "A physical density here is wrong by a factor of C_0 and would "
                'still converge.'
            )
        if mm.location is not Location.NODE:
            raise ValueError(f"{d2} must live on NODE, got {mm.location.name}.")
        if mm.size!=mesh.n_nodes:
            raise ValueError(
                f"{d2} has length {mm.size} but the mesh has "
                f"{mesh.n_nodes} nodes."
            )
def assemble_electron_continuity(
    mesh : Mesh1D,
    psi: Field,
    n  : Field,
    p: Field,
    recombination :  RecombinationModel,
    scale  :  ScaleFactors,
    Dn  : Diffusivity,
    geometry  :  EdgeGeometry= UNIFORM_1D,
) -> SparseAssembly :
    _check(mesh ,  ( ("psi" , psi) ,   ("n" ,   n), ("p", p))  )
    a = mesh.h/scale.x_0
    b2  =   mesh.volume  /  scale.x_0
    x2 = np.asarray(recombination.rate(n.data, p.data), dtype  =np.float64)
    w,_= recombination.electron_linearization(n.data,p.data)
    thing =_bernoulli_pair(psi.data,geometry)
    res2 = _electron_continuity_residual(
        a,b2,Dn,thing,n.data,x2,geometry
    )


    c,z,s= _electron_continuity_jacobian(a, b2, Dn, thing, mesh.n_nodes, np.asarray(w,dtype=np.float64), geometry,)

    return SparseAssembly(
        residual = res2,
        rows = c,
        cols  = z,
        values = s,
        shape =  (mesh.n_nodes, mesh.n_nodes),
    )

def assemble_hole_continuity(mesh  :  Mesh1D, psi :  Field, n  : Field, p :Field, recombination:  RecombinationModel, scale  :  ScaleFactors, Dp: Diffusivity, geometry :EdgeGeometry = UNIFORM_1D,) ->  SparseAssembly  :
    _check( mesh , ( ('psi',  psi  ),  ( 'n', n  ), ( "p",  p )  )  )

    ret  = mesh.h /scale.x_0
    aa =mesh.volume / scale.x_0
    idx =  np.asarray(  recombination.rate(  n.data ,   p.data ) ,  dtype = np.float64)
    y,_=recombination.hole_linearization(n.data,p.data)

    t  =  _bernoulli_pair ( psi.data,  geometry)
    k=_hole_continuity_residual(ret,aa,Dp,t,p.data,idx,geometry)
    s,v,b2=_hole_continuity_jacobian(
        ret,
        aa,
        Dp,
        t,
        mesh.n_nodes,
        np.asarray(y,dtype=np.float64),
        geometry,
    )

    return SparseAssembly(
        residual  = k,
        rows=  s,
        cols=v,
        values = b2,
        shape=  (mesh.n_nodes, mesh.n_nodes),
    )
