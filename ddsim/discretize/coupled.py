from __future__ import annotations
from collections.abc import Sequence
from  enum  import  Enum,   IntEnum
from typing import NamedTuple, TypeVar, cast
import numpy as np, numpy.typing  as  npt
from  ddsim.core  import constants as C
from  ddsim.core.field  import  Field, Location,  ScalingState
from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.assembly import SparseAssembly
from ddsim.discretize.boundary import(
    Carrier,
    Contact,
    GateContact,
    apply_dirichlet_nodes,
    gate_psi_scaled,
    ohmic_density_scaled,
    ohmic_psi_scaled,
)
from ddsim.discretize.continuity import Diffusivity
from ddsim.discretize.geometry import UNIFORM_1D, EdgeGeometry
from ddsim.mesh.mesh1d  import  Mesh1D;  from ddsim.physics.bernoulli  import B ,  dB_dx
from ddsim.physics.mobility import(
    EdgeDiffusivity,
    diffusivity_at,
    diffusivity_tangent,
)
from ddsim.physics.recombination import Density, RecombinationModel
from ddsim.physics.statistics import Degeneracy

EPS=float(np.finfo(np.float64).eps)


class Unknown(IntEnum) :

    PSI =0

    N  = 1

    P  = 2
UNKNOWNS_PER_NODE=len(Unknown)

Number=  TypeVar("Number", np.float64, np.complex128)



TermScales =tuple[npt.NDArray[np.float64],npt.NDArray[np.float64],npt.NDArray[np.float64]]




def unknown_index(node  :int, component  :Unknown)->  int:
    return UNKNOWNS_PER_NODE  *  node   +  int(  component  )
def pack(psi : npt.NDArray[Number], n  : npt.NDArray[Number], p : npt.NDArray[Number],) -> npt.NDArray[Number]:
    if not psi.size==n.size==p.size :
        raise ValueError(
            "psi, n and p must have the same length, got "
            f"{psi.size}, {n.size} and {p.size}."
        )
    r= np.empty(UNKNOWNS_PER_NODE *psi.size,dtype= np.result_type(psi,n,p))
    r[Unknown.PSI ::UNKNOWNS_PER_NODE] = psi
    r [ Unknown.N  ::  UNKNOWNS_PER_NODE ]   =  n
    r[Unknown.P::UNKNOWNS_PER_NODE] = p

    return r



def unpack(
    x:npt.NDArray[Number],
)-> tuple[npt.NDArray[Number], npt.NDArray[Number], npt.NDArray[Number]] :
    return(
        x[ Unknown.PSI  ::  UNKNOWNS_PER_NODE  ] ,
        x [  Unknown.N   ::  UNKNOWNS_PER_NODE] ,
        x[ Unknown.P  ::   UNKNOWNS_PER_NODE  ] ,
    )
def edge_drop(
    psi  :  npt.NDArray[Number], geometry  :EdgeGeometry=  UNIFORM_1D
)  ->  npt.NDArray[Number] :
    arr , d  =   geometry.ends_of (psi.size)

    return psi[  d]   -  psi[ arr]

def effective_potentials(psi :npt.NDArray[Number], n:npt.NDArray[Number], p :npt.NDArray[Number], degeneracy:Degeneracy|None = None,)->tuple[npt.NDArray[Number],npt.NDArray[Number]]:
    if degeneracy is None:
        return  psi,  psi


    cur= cast('npt.NDArray[np.float64]',psi);i   =  cast(  "npt.NDArray[np.float64]" ,   n  )

    f  =cast('npt.NDArray[np.float64]', p)

    return(
        cast(
            'npt.NDArray[Number]' ,
            degeneracy.electron_potential(  cur , i ) ,
        ),
        cast ("npt.NDArray[Number]",  degeneracy.hole_potential (cur ,  f)),
    )

def _bernoulli_pair(psi :  npt.NDArray[Number], geometry : EdgeGeometry = UNIFORM_1D) -> tuple[npt.NDArray[Number], npt.NDArray[Number]]  :
    r =  edge_drop(psi, geometry)
    return np.asarray(B(r)), np.asarray(B(-r))



def _diffusivity_at(
    D:  EdgeDiffusivity,
    psi :  npt.NDArray[Number],
    h:npt.NDArray[np.float64],
    geometry: EdgeGeometry  = UNIFORM_1D,
)->  Diffusivity :
    return diffusivity_at(D, edge_drop(psi, geometry), h)


def _diffusivity_tangent(
    D:EdgeDiffusivity,
    psi: npt.NDArray[np.float64],
    h: npt.NDArray[np.float64],
    geometry : EdgeGeometry =UNIFORM_1D,
)->npt.NDArray[np.float64]|None :
    return diffusivity_tangent(D,edge_drop(psi,geometry),h)


def _bernoulli_derivative_pair(psi:npt.NDArray[np.float64],geometry:EdgeGeometry=UNIFORM_1D) ->tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]] :

    vals =edge_drop(psi,geometry)
    return(
        np.asarray(dB_dx(vals), dtype  =np.float64),
        np.asarray(dB_dx(-  vals), dtype  = np.float64),
    )

def  coupled_residual(
    h  :  npt.NDArray[np.float64 ],
    volume  :  npt.NDArray[  np.float64],
    x :  npt.NDArray[  Number ],
    net_doping   :  npt.NDArray [  np.float64],
    Dn  :  EdgeDiffusivity,
    Dp  :   EdgeDiffusivity,
    recombination :   RecombinationModel,
    geometry  : EdgeGeometry  =   UNIFORM_1D ,
    degeneracy   : Degeneracy   |  None  = None,
)  -> npt.NDArray[ Number  ]   :


    psi, n, p =unpack(x)
    z2 = cast (
        "npt.NDArray[Number]" ,
        recombination.rate(  cast(  Density,  n),  cast( Density,  p  )  ),
    )
    out2, s = effective_potentials( psi , n ,  p , degeneracy)
    return  _residual_from(
        h ,
        volume,
        x ,
        net_doping,
        _diffusivity_at(Dn , psi , h,   geometry  ),
        _diffusivity_at (Dp,   psi,   h ,  geometry ) ,
        _bernoulli_pair(  out2,   geometry ) ,
        _bernoulli_pair (s, geometry  ),
        z2,
        geometry ,
    )


def _residual_from(h :  npt.NDArray[ np.float64 ], volume  :  npt.NDArray [ np.float64  ], x   :  npt.NDArray [ Number ], net_doping  :  npt.NDArray[  np.float64  ], Dn   :  Diffusivity , Dp  : Diffusivity , bernoulli_n   : tuple[npt.NDArray[  Number], npt.NDArray[  Number ]  ], bernoulli_p  :  tuple[  npt.NDArray[ Number  ] ,   npt.NDArray [ Number]], R  :   npt.NDArray[  Number  ], geometry  : EdgeGeometry  = UNIFORM_1D ,)   ->   npt.NDArray [Number ]   :
    psi,  n, p  =  unpack(x)
    e, m =bernoulli_n
    nxt, info = bernoulli_p

    dd  = np.zeros_like( x )
    k, u ,   c   =  unpack( dd  )
    res,v =geometry.ends(h.size)

    k2  = geometry.weight  *   (psi[ res]  - psi[v ] )  /   h

    np.add.at(k, res, k2)
    np.add.at(k,  v ,   -  k2  )

    k-= (p-n + net_doping)*volume
    z =(Dn * geometry.carrier_face / h)  * (e * n[v]-  m * n[res])
    u += R *volume
    np.add.at(u,res,-z)


    np.add.at( u,   v , z)

    r=(Dp  *  geometry.carrier_face/ h) * (
        nxt*p[res]- info *p[v]
    )
    c  +=  R * volume
    np.add.at(c,res,r)
    np.add.at(c, v, - r)

    return dd



class NodeRange(Enum) :

    ALL = "all"

    LEFT = "left"
    RIGHT="right"

class _Triplets:

    def __init__(
        self, n_nodes  :  int, geometry  : EdgeGeometry = UNIFORM_1D
    ) ->  None  :
        w,k=geometry.ends_of(n_nodes)
        self._nodes  ={
            NodeRange.ALL:np.arange(n_nodes, dtype= np.int64),
            NodeRange.LEFT : w,
            NodeRange.RIGHT : k,
        }
        self._rows   : list[npt.NDArray[  np.int64 ]]  =  [ ] ; self._cols:  list[npt.NDArray[np.int64]]  = []

        self._values:list[npt.NDArray[np.float64]]= []
    def add(self , equation :  Unknown, at_nodes   :  NodeRange, unknown  :   Unknown, of_nodes   : NodeRange, values  :  npt.NDArray[  np.float64  ],)  ->  None   :

        self._rows.append(
            UNKNOWNS_PER_NODE * self._nodes[at_nodes]  + int(equation)
        )
        self._cols.append(
            UNKNOWNS_PER_NODE  * self._nodes[of_nodes]+  int(unknown)
        )

        self._values.append(np.asarray(values,dtype =np.float64))

    def build(
        self,
    )  ->tuple[
        npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]
    ]:
        return(
            np.concatenate(self._rows),
            np.concatenate(self._cols),
            np.concatenate(self._values),
        )
def coupled_jacobian(
    h :npt.NDArray[np.float64],
    volume:npt.NDArray[np.float64],
    x :npt.NDArray[np.float64],
    Dn: EdgeDiffusivity,
    Dp:EdgeDiffusivity,
    recombination: RecombinationModel,
    geometry:EdgeGeometry= UNIFORM_1D,
    degeneracy: Degeneracy |None=None,
)->tuple[npt.NDArray[np.int64],npt.NDArray[np.int64],npt.NDArray[np.float64]]:
    psi , n, p  = unpack ( x)
    u,   arr =   effective_potentials(  psi,  n,  p,  degeneracy)
    return _jacobian_from(h, volume, x, _diffusivity_at(Dn,psi,h,geometry), _diffusivity_at(Dp,psi,h,geometry), _bernoulli_pair(u,geometry), _bernoulli_pair(arr,geometry), _bernoulli_derivative_pair(u,geometry), _bernoulli_derivative_pair(arr,geometry), np.asarray(recombination.d_rate_dn(n,p),dtype=np.float64), np.asarray(recombination.d_rate_dp(n,p),dtype=np.float64), geometry, _diffusivity_tangent(Dn,psi,h,geometry), _diffusivity_tangent(Dp,psi,h,geometry), * _potential_tangents(n,p,degeneracy),)



def _potential_tangents(n :npt.NDArray[np.float64], p  : npt.NDArray[np.float64], degeneracy :  Degeneracy |None,)  -> tuple[npt.NDArray[np.float64]|None, npt.NDArray[np.float64] | None] :

    if degeneracy is None:
        return None,   None
    return degeneracy.d_electron_potential_dn(n),degeneracy.d_hole_potential_dp(p)

def _jacobian_from(h  :npt.NDArray[np.float64], volume : npt.NDArray[np.float64], x :  npt.NDArray[np.float64], Dn  : Diffusivity, Dp :  Diffusivity, bernoulli_n : tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], bernoulli_p: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], dbernoulli_n  :tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], dbernoulli_p: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], dR_dn :npt.NDArray[np.float64], dR_dp :  npt.NDArray[np.float64], geometry  : EdgeGeometry =UNIFORM_1D, dDn_dX : npt.NDArray[np.float64] |  None  = None, dDp_dX : npt.NDArray[np.float64]  | None =None, dpsi_n_dn :  npt.NDArray[np.float64] | None =  None, dpsi_p_dp : npt.NDArray[np.float64]  |  None  = None,)-> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    psi, n, p=unpack(x)
    ii  =  psi.size

    kk,dd =bernoulli_n
    c2, vals =dbernoulli_n

    vv,row=bernoulli_p

    r, cur = dbernoulli_p


    i =NodeRange.ALL
    b2 =NodeRange.LEFT;  k2  =  NodeRange.RIGHT

    b = _Triplets(ii, geometry)
    jj,d = geometry.ends(h.size)



    rows  =   geometry.weight  /  h
    f  =   np.zeros( ii )
    np.add.at(f,jj,rows)
    np.add.at(  f,  d ,  rows)
    b.add( Unknown.PSI, i ,  Unknown.PSI, i,  f) ; b.add(Unknown.PSI, b2, Unknown.PSI, k2, - rows)


    b.add (  Unknown.PSI ,   k2 ,  Unknown.PSI,  b2,  - rows )

    b.add( Unknown.PSI,  i , Unknown.N ,   i, volume  )
    b.add(Unknown.PSI,i,Unknown.P,i,-volume)

    z =(Dn *geometry.carrier_face/h)* (
        c2 * n[d]+vals*n[jj]
    )
    tt = z
    if dDn_dX is not None:
        tt= tt + (dDn_dX *geometry.carrier_face /h) * (
            kk  * n[d] - dd  *n[jj]
        )
    f= np.zeros(ii)
    np.add.at(f, jj, tt)
    np.add.at (f , d,  tt)
    b.add(Unknown.N,i,Unknown.PSI,i,f)
    b.add(Unknown.N,b2,Unknown.PSI,k2,- tt)
    b.add(Unknown.N ,
                     k2,
                    Unknown.PSI,
            b2,
        -  tt  )

    res= (Dn * geometry.carrier_face/ h)* kk

    a2 = (Dn * geometry.carrier_face/h)*  dd
    if  dpsi_n_dn is  not None   :

        res=res+ z * dpsi_n_dn[d]
        a2  =  a2  +  z *   dpsi_n_dn [ jj ]
    f= dR_dn* volume
    np.add.at(f,jj,a2)
    np.add.at(f, d, res)
    b.add(Unknown.N,i,Unknown.N,i,f) ; b.add(Unknown.N, b2, Unknown.N, k2, -res)
    b.add(  Unknown.N,   k2 ,  Unknown.N ,   b2 ,   -  a2)

    b.add(Unknown.N,i,Unknown.P,i,dR_dp *volume)

    ret   =   (  Dp   *  geometry.carrier_face /  h  ) *  (
        r *  p [ jj ]   + cur  *  p[d ]
    )
    v = ret
    if dDp_dX  is not None   :
        v=  v +  (dDp_dX * geometry.carrier_face/ h) *(
            vv * p[jj] - row*p[d]
        )
    f =  np.zeros (  ii  )
    np.add.at(f, jj, -v)
    np.add.at(  f ,  d ,  - v)
    b.add(Unknown.P, i, Unknown.PSI, i, f)

    b.add(Unknown.P,b2,Unknown.PSI,k2,v)
    b.add(Unknown.P, k2, Unknown.PSI, b2, v)


    b.add(Unknown.P,i,Unknown.N,i,dR_dn * volume)
    a2  =   (Dp *   geometry.carrier_face   /  h  )   * vv
    res= (Dp *  geometry.carrier_face /  h) *row
    if dpsi_p_dp  is  not None   :

        a2 =  a2 -  ret *dpsi_p_dp[jj];res = res  -ret *dpsi_p_dp[d]
    f=dR_dp*volume
    np.add.at(  f, jj, a2  )
    np.add.at(f, d, res)
    b.add(Unknown.P, i, Unknown.P, i, f); b.add(Unknown.P,b2,Unknown.P,k2,-res)

    b.add (Unknown.P,  k2,   Unknown.P , b2,  -   a2  )

    return b.build()


def assemble_coupled(mesh : Mesh1D, psi :Field, n  :Field, p  : Field, net_doping :Field, recombination : RecombinationModel, scale : ScaleFactors, Dn :EdgeDiffusivity, Dp :EdgeDiffusivity, degeneracy:Degeneracy | None  =  None,)  -> SparseAssembly  :
    for d, ii in(
        ('psi', psi),
        ("n", n),
        ("p", p),
        ("net_doping", net_doping),
    ):
        if ii.scaling is not ScalingState.SCALED:
            raise  ValueError (
                f"{d} must be SCALED before assembly, got {ii.scaling.name}. "
                'A physical value here is wrong by a fixed factor and would '
                "still converge."
            )
        if ii.location  is not  Location.NODE  :
            raise ValueError(f"{d} must live on NODE, got {ii.location.name}.")
        if ii.size  != mesh.n_nodes:


            raise ValueError(
                f"{d} has length {ii.size} but the mesh has "
                f"{mesh.n_nodes} nodes."
            )
    arr: npt.NDArray[np.float64]= pack(psi.data,
                     n.data,
                    p.data)

    return assemble_coupled_arrays (
        h =  mesh.h  /  scale.x_0 ,
        volume = mesh.volume   /   scale.x_0,
        x  = arr,
        net_doping =  net_doping.data,
        Dn  =   Dn ,
        Dp =  Dp,
        recombination  =   recombination,
        degeneracy =  degeneracy ,
    )

def assemble_coupled_arrays(h: npt.NDArray[np.float64], volume : npt.NDArray[np.float64], x : npt.NDArray[np.float64], net_doping : npt.NDArray[np.float64], Dn : EdgeDiffusivity, Dp:EdgeDiffusivity, recombination  : RecombinationModel, degeneracy : Degeneracy |None  = None,)->SparseAssembly  :
    return assemble_coupled_terms(
        h , volume,  x , net_doping , Dn,   Dp , recombination, degeneracy =   degeneracy
    ).assembly

class CoupledAssembly(NamedTuple) :



    assembly  :  SparseAssembly

    scales :TermScales

def  assemble_coupled_terms(
    h : npt.NDArray[  np.float64],
    volume :  npt.NDArray [  np.float64  ] ,
    x  :  npt.NDArray[  np.float64  ],
    net_doping  : npt.NDArray[np.float64],
    Dn  :  EdgeDiffusivity,
    Dp : EdgeDiffusivity ,
    recombination  : RecombinationModel ,
    geometry  :  EdgeGeometry =   UNIFORM_1D,
    degeneracy   : Degeneracy   |   None  =   None,
)  ->  CoupledAssembly :
    psi ,   n,   p   =   unpack(x )
    b, x2 =effective_potentials(psi, n, p, degeneracy)

    j = _bernoulli_pair(b, geometry)

    lst=(
        j if degeneracy is None else _bernoulli_pair(x2,geometry)
    )
    rr =_bernoulli_derivative_pair(b, geometry)
    z= (rr if degeneracy is None else _bernoulli_derivative_pair(x2,geometry))
    v2, f=  _potential_tangents(n, p, degeneracy)
    g = np.asarray(recombination.rate(n, p), dtype = np.float64)
    d  =np.asarray(recombination.d_rate_dn(n, p), dtype = np.float64)
    m=np.asarray(recombination.d_rate_dp(n,p),dtype=np.float64)


    u=_diffusivity_at(Dn,psi,h,geometry)
    v =_diffusivity_at(Dp, psi, h, geometry)
    y  =_diffusivity_tangent(Dn, psi, h, geometry)
    res2  = _diffusivity_tangent(Dp, psi, h, geometry)
    s  =  _residual_from(h, volume, x, net_doping, u, v, j, lst, g, geometry,)
    mm,   c ,  ss   =  _jacobian_from(h, volume, x, u , v, j, lst, rr , z, d, m , geometry, y, res2, v2 , f,)
    res=_term_scales_from(
        h,
        volume,
        x,
        net_doping,
        u,
        v,
        j,
        lst,
        g,
        geometry,
    )
    cc = x.size
    return CoupledAssembly(
        assembly =SparseAssembly(
            residual  = np.asarray(s, dtype = np.float64),
            rows =  mm,
            cols = c,
            values = ss,
            shape =  (cc, cc),
        ),
        scales  = res,
    )



def limit_psi_step(delta : npt.NDArray[np.float64],max_psi_step:float)-> npt.NDArray[np.float64]:
    vv  =   delta [Unknown.PSI  ::   UNKNOWNS_PER_NODE ]


    v  = float(np.max(np.abs(vv)))
    if  v  <=  max_psi_step :
        return delta
    d =delta.copy()
    d[Unknown.PSI ::UNKNOWNS_PER_NODE]=vv* (max_psi_step / v);return d



def apply_contacts_coupled(assembly: SparseAssembly, x :npt.NDArray[np.float64], net_doping  : npt.NDArray[np.float64], contacts :  Sequence[Contact], scale : ScaleFactors, carrier_free_nodes : Sequence[int] =  (), T:float =C.T_ROOM, degeneracy:Degeneracy | None  = None,)->SparseAssembly  :
    c=[e.name for e in contacts]
    if  len(  set(  c  )  )  !=  len(  c )   :
        raise ValueError(f"contact names must be unique, got {c}")
    h :list[int] = []
    u: list[float]=[]
    for e in contacts  :
        y=e.voltage /scale.psi_0

        if isinstance(e, GateContact) :
            it =gate_psi_scaled(y,e.work_function,T)
            for w in e.nodes:
                h.append(unknown_index(w, Unknown.PSI))
                u.append(it)
            continue


        for w in e.nodes  :
            xs =float(net_doping[w])

            h.append(unknown_index(w,
                       Unknown.PSI)); u.append(ohmic_psi_scaled(xs, y, degeneracy))
            h.append(unknown_index(w,
                          Unknown.N))
            u.append(ohmic_density_scaled(xs,Carrier.ELECTRON,degeneracy))

            h.append(unknown_index(w,Unknown.P))
            u.append(ohmic_density_scaled(  xs, Carrier.HOLE,   degeneracy))

    for w in carrier_free_nodes:
        h.append(  unknown_index (w,   Unknown.N  )  )
        u.append(0.0)
        h.append(  unknown_index(  w,   Unknown.P ))
        u.append(0.0)
    return apply_dirichlet_nodes(  assembly, x, h, u)



def residual_term_scales(
    h :npt.NDArray[np.float64],
    volume :  npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    net_doping :npt.NDArray[np.float64],
    Dn :EdgeDiffusivity,
    Dp  :  EdgeDiffusivity,
    R : npt.NDArray[np.float64]|None=None,
    geometry: EdgeGeometry=UNIFORM_1D,
    degeneracy : Degeneracy| None =  None,
)  -> TermScales :


    psi, n,  p   =   unpack (  x  )
    c, k= effective_potentials(psi, n, p, degeneracy)
    return _term_scales_from(h, volume, x, net_doping, _diffusivity_at(Dn,psi,h,geometry), _diffusivity_at(Dp,psi,h,geometry), _bernoulli_pair(c,geometry), _bernoulli_pair(k,geometry), R, geometry,)

def _term_scales_from(h :npt.NDArray[np.float64], volume: npt.NDArray[np.float64], x: npt.NDArray[np.float64], net_doping:npt.NDArray[np.float64], Dn:Diffusivity, Dp :Diffusivity, bernoulli_n:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]], bernoulli_p:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]], R:npt.NDArray[np.float64]| None, geometry : EdgeGeometry=UNIFORM_1D,)->TermScales :

    psi, n, p=  unpack(x)


    t2, flag = bernoulli_n
    bar, v = bernoulli_p
    e,i= geometry.ends(h.size)
    r= volume.size
    v2 = (geometry.weight  / h)* np.maximum(
        np.abs(psi[e]), np.abs(psi[i])
    )
    m= np.maximum(_largest_at_each_node(v2,e,i,r), (np.abs(p)+np.abs(n)+np.abs(net_doping)) *volume,)

    c =  (
        np.zeros(r) if R is None else np.abs(R)* volume
    )
    rows  = Dn * geometry.carrier_face/ h
    z =  Dp * geometry.carrier_face /  h
    s=np.maximum(
        _largest_at_each_node(
            np.maximum(rows* t2* n[i],rows *flag *n[e]),
            e,
            i,
            r,
        ),
        c,
    )
    obj = np.maximum(
        _largest_at_each_node(
            np.maximum(z*bar* p[e], z * v *  p[i]),
            e,
            i,
            r,
        ),
        c,
    )


    g= (m,s,obj)

    if not  all(bool(  np.all(  np.isfinite( tmp3  ) ) )  for tmp3 in  g)  :
        raise FloatingPointError(
            "a residual term overflowed, the iterate has diverged: term scales "
            f"max to {tuple(float(np.max(k)) for k in g)} for "
            '(psi, n, p).'
        )
    if  not all(np.any(it  >  0.0 )   for it in g  ) :
        raise ValueError(
            f"the state has no terms to measure a residual against: term "
            f"scales max to "
            f"{tuple(float(np.max(d2)) for d2 in g)} for (psi, n, p). "
            'Every equation is identically zero, which a device never is. '
            "Dividing by these would rename the problem as a singular matrix "
            'three call frames later.'
        )

    return g



def _largest_at_each_node(edge_term : npt.NDArray[np.float64], node_left  : npt.NDArray[np.int64], node_right : npt.NDArray[np.int64], n_nodes  :int,) ->  npt.NDArray[np.float64] :
    k= np.zeros(n_nodes)
    np.maximum.at(k,node_left,edge_term) ; np.maximum.at(k, node_right, edge_term)
    return k


def row_weights(scales :TermScales, n_nodes : int)  ->npt.NDArray[np.float64]:


    k2=  np.empty(UNKNOWNS_PER_NODE  *n_nodes)

    for lst, b in zip(Unknown, scales, strict= True) :
        if b.size!= n_nodes :
            raise ValueError(
                f"the {lst.name} term scale has {b.size} entries "
                f"for a mesh of {n_nodes} nodes"
            )
        k2[ lst   ::   UNKNOWNS_PER_NODE ]  =  np.max ( b )
    return  k2
FAMILIES  =  tuple(  w.name.lower(  )  for  w in Unknown  )




def residual_measure(residual : npt.NDArray[np.float64],scales:TermScales,n_nodes:int) ->float :

    return max(0.0,*residual_measure_by_family(residual,scales,n_nodes).values())


def residual_measure_by_family (
    residual :  npt.NDArray[np.float64  ],  scales : TermScales ,  n_nodes  : int
)  ->  dict [str,   float ]  :
    d2 =row_weights(scales,n_nodes)

    ys=np.abs(residual)*d2
    d:dict[str,float]= {}
    for j,bb,h in zip(Unknown,scales,FAMILIES,strict =True) :
        xs= ys[j:: UNKNOWNS_PER_NODE]
        s2 =EPS  *  float(np.max(bb))
        el = np.divide(
            xs, bb, out = np.zeros_like(xs), where  = bb > s2
        )
        d[  h  ]  =  float (np.max(  el  )  )
    return d



def scale_rows(
    assembly:SparseAssembly,weights: npt.NDArray[np.float64]
)-> SparseAssembly:
    return SparseAssembly (
        residual = assembly.residual   /   weights,
        rows  =  assembly.rows,
        cols =  assembly.cols ,
        values  =  assembly.values  / weights[assembly.rows  ],
        shape   =  assembly.shape,
    )



def coupled_update_norm(delta : npt.NDArray[np.float64],x :npt.NDArray[np.float64])->float:
    return max(coupled_update_by_family(delta,x).values())


def  coupled_update_by_family(
    delta   :  npt.NDArray [np.float64  ], x  :  npt.NDArray [np.float64 ]
)  -> dict[  str,  float]  :
    z, g, val = unpack(delta)
    _,n,p=unpack(x)
    return{
        "psi"  : float(np.max(np.abs(z))),
        "n"  :float(np.max(np.abs(g)  / (np.abs(n)+ 1.0))),
        "p"  :float(np.max(np.abs(val) / (np.abs(p)  + 1.0))),
    }
