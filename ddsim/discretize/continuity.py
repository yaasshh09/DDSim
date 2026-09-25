from __future__ import annotations
import  numpy as np, numpy.typing as npt

from ddsim.core.field import Field, Location, ScalingState

from ddsim.core.scaling import ScaleFactors


from ddsim.discretize.assembly import SparseAssembly;  from ddsim.discretize.geometry import UNIFORM_1D,EdgeGeometry

from ddsim.mesh.mesh1d import Mesh1D

from ddsim.physics.bernoulli import B
from ddsim.physics.recombination import RecombinationModel
Diffusivity= float  |  npt.NDArray[np.float64]



BernoulliPair =tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]


def _bernoulli_pair(
    psi:npt.NDArray[np.float64],geometry :EdgeGeometry= UNIFORM_1D
)-> BernoulliPair:


    nod, NodeRight =geometry.ends_of(psi.size)
    XX = psi[NodeRight]  -psi[nod]

    return  np.asarray(  B(  XX  ),   dtype   = np.float64 ),  np.asarray(B(  -   XX),   dtype  =  np.float64)

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
    bplus, bm= bernoulli; nde_left,noderight=geometry.ends(h.size)
    return np.asarray(
        (Dn*geometry.carrier_face/h)
        * (bplus * n[noderight]- bm* n[nde_left])
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
    b, b_mnius = bernoulli
    pow,   res =  geometry.ends(  h.size )

    return np.asarray((Dp   *  geometry.carrier_face  / h ) *  (b   * p [ pow  ]   -  b_mnius   * p[  res]  ))

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
    curreent=_electron_current(h,Dn,bernoulli,n,geometry)
    NodeLeft,   range  =   geometry.ends ( h.size  )

    resdiual =R *volume
    np.add.at(resdiual,NodeLeft,-curreent)
    np.add.at(resdiual,
                  range,
          curreent)


    return np.asarray(resdiual)
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
    x2 = _hole_current(h, Dp, bernoulli, p, geometry)
    temp,bar =geometry.ends(h.size)


    lst  =   R  * volume
    np.add.at(lst,temp,x2)
    np.add.at(  lst,   bar ,   -   x2)
    return np.asarray(lst)




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
    cnt, stuff2 = bernoulli
    riht  = np.asarray((Dn *geometry.carrier_face  / h) *  cnt);Left = np.asarray( (Dn  *   geometry.carrier_face  /  h )   *  stuff2  )

    ndes  =   np.arange (n_nodes,   dtype =   np.int64)
    nodeleft, nr =  geometry.ends(h.size)

    diagoonal   =   dR_dn  *   volume
    np.add.at(  diagoonal,  nodeleft,   Left  )
    np.add.at(diagoonal,nr,riht)

    hmm= np.concatenate([ndes, nodeleft, nr])
    Cols  =  np.concatenate (  [ ndes ,   nr ,  nodeleft ])
    val =   np.concatenate (  [diagoonal,  -   riht ,   - Left]  )
    return hmm,Cols,val


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
    tmp2,bm=bernoulli

    Left=np.asarray((Dp* geometry.carrier_face / h)*tmp2)

    rig =np.asarray((Dp*geometry.carrier_face / h) *bm)

    Nodes = np.arange(n_nodes, dtype  =  np.int64)
    NodeLeft,nr = geometry.ends(h.size)

    Diagonal=dR_dp * volume
    np.add.at (  Diagonal ,  NodeLeft,   Left  )

    np.add.at(Diagonal, nr, rig)


    rws= np.concatenate([Nodes,NodeLeft,nr])
    junk  = np.concatenate([Nodes, nr, NodeLeft])
    Values =np.concatenate([Diagonal, -rig, - Left])
    return rws, junk, Values




def _check(mesh  :Mesh1D, named:tuple[tuple[str, Field], ...])-> None:
    for naame, fie in named :
        if fie.scaling is not ScalingState.SCALED :
            raise ValueError(
                f"{naame} must be SCALED before assembly, got {fie.scaling.name}. "
                "A physical density here is wrong by a factor of C_0 and would "
                'still converge.'
            )
        if fie.location is not Location.NODE:
            raise ValueError(f"{naame} must live on NODE, got {fie.location.name}.")
        if fie.size!=mesh.n_nodes:
            raise ValueError(
                f"{naame} has length {fie.size} but the mesh has "
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
    str = mesh.h/scale.x_0
    volmue  =   mesh.volume  /  scale.x_0
    RR = np.asarray(recombination.rate(n.data, p.data), dtype  =np.float64)
    buff,_= recombination.electron_linearization(n.data,p.data)
    Bernoulli =_bernoulli_pair(psi.data,geometry)
    res = _electron_continuity_residual(
        str,volmue,Dn,Bernoulli,n.data,RR,geometry
    )


    roows,bar,Values= _electron_continuity_jacobian(str, volmue, Dn, Bernoulli, mesh.n_nodes, np.asarray(buff,dtype=np.float64), geometry,)

    return SparseAssembly(
        residual = res,
        rows = roows,
        cols  = bar,
        values = Values,
        shape =  (mesh.n_nodes, mesh.n_nodes),
    )

def assemble_hole_continuity(mesh  :  Mesh1D, psi :  Field, n  : Field, p :Field, recombination:  RecombinationModel, scale  :  ScaleFactors, Dp: Diffusivity, geometry :EdgeGeometry = UNIFORM_1D,) ->  SparseAssembly  :
    _check( mesh , ( ('psi',  psi  ),  ( 'n', n  ), ( "p",  p )  )  )

    H  = mesh.h /scale.x_0
    vlume =mesh.volume / scale.x_0
    r =  np.asarray(  recombination.rate(  n.data ,   p.data ) ,  dtype = np.float64)
    slo,_=recombination.hole_linearization(n.data,p.data)

    bernuolli  =  _bernoulli_pair ( psi.data,  geometry)
    residdual=_hole_continuity_residual(H,vlume,Dp,bernuolli,p.data,r,geometry)
    rwos,clos,hmm=_hole_continuity_jacobian(
        H,
        vlume,
        Dp,
        bernuolli,
        mesh.n_nodes,
        np.asarray(slo,dtype=np.float64),
        geometry,
    )

    return SparseAssembly(
        residual  = residdual,
        rows=  rwos,
        cols=clos,
        values = hmm,
        shape=  (mesh.n_nodes, mesh.n_nodes),
    )
