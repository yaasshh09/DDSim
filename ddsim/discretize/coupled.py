from __future__  import  annotations
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
    xx= np.empty(UNKNOWNS_PER_NODE *psi.size,dtype= np.result_type(psi,n,p))
    xx[Unknown.PSI ::UNKNOWNS_PER_NODE] = psi
    xx [ Unknown.N  ::  UNKNOWNS_PER_NODE ]   =  n
    xx[Unknown.P::UNKNOWNS_PER_NODE] = p

    return xx



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
    nodeLeft , NodeRight  =   geometry.ends_of (psi.size)

    return psi[  NodeRight]   -  psi[ nodeLeft]

def effective_potentials(psi :npt.NDArray[Number], n:npt.NDArray[Number], p :npt.NDArray[Number], degeneracy:Degeneracy|None = None,)->tuple[npt.NDArray[Number],npt.NDArray[Number]]:
    if degeneracy is None:
        return  psi,  psi


    pottential= cast('npt.NDArray[np.float64]',psi);ele   =  cast(  "npt.NDArray[np.float64]" ,   n  )

    Holes  =cast('npt.NDArray[np.float64]', p)

    return(
        cast(
            'npt.NDArray[Number]' ,
            degeneracy.electron_potential(  pottential , ele ) ,
        ),
        cast ("npt.NDArray[Number]",  degeneracy.hole_potential (pottential ,  Holes)),
    )

def _bernoulli_pair(psi :  npt.NDArray[Number], geometry : EdgeGeometry = UNIFORM_1D) -> tuple[npt.NDArray[Number], npt.NDArray[Number]]  :
    XX =  edge_drop(psi, geometry)
    return np.asarray(B(XX)), np.asarray(B(-XX))



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

    XX =edge_drop(psi,geometry)
    return(
        np.asarray(dB_dx(XX), dtype  =np.float64),
        np.asarray(dB_dx(-  XX), dtype  = np.float64),
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
    id = cast (
        "npt.NDArray[Number]" ,
        recombination.rate(  cast(  Density,  n),  cast( Density,  p  )  ),
    )
    res, psip = effective_potentials( psi , n ,  p , degeneracy)
    return  _residual_from(
        h ,
        volume,
        x ,
        net_doping,
        _diffusivity_at(Dn , psi , h,   geometry  ),
        _diffusivity_at (Dp,   psi,   h ,  geometry ) ,
        _bernoulli_pair(  res,   geometry ) ,
        _bernoulli_pair (psip, geometry  ),
        id,
        geometry ,
    )


def _residual_from(h :  npt.NDArray[ np.float64 ], volume  :  npt.NDArray [ np.float64  ], x   :  npt.NDArray [ Number ], net_doping  :  npt.NDArray[  np.float64  ], Dn   :  Diffusivity , Dp  : Diffusivity , bernoulli_n   : tuple[npt.NDArray[  Number], npt.NDArray[  Number ]  ], bernoulli_p  :  tuple[  npt.NDArray[ Number  ] ,   npt.NDArray [ Number]], R  :   npt.NDArray[  Number  ], geometry  : EdgeGeometry  = UNIFORM_1D ,)   ->   npt.NDArray [Number ]   :
    psi,  n, p  =  unpack(x)
    any, bnminus =bernoulli_n
    BpPlus, blah = bernoulli_p

    outt  = np.zeros_like( x )
    FPsi, f ,   fp   =  unpack( outt  )
    nodeLeft,nr =geometry.ends(h.size)

    zz  = geometry.weight  *   (psi[ nodeLeft]  - psi[nr ] )  /   h

    np.add.at(FPsi, nodeLeft, zz)
    np.add.at(FPsi,  nr ,   -  zz  )

    FPsi-= (p-n + net_doping)*volume
    Jnn =(Dn * geometry.carrier_face / h)  * (any * n[nr]-  bnminus * n[nodeLeft])
    f += R *volume
    np.add.at(f,nodeLeft,-Jnn)


    np.add.at( f,   nr , Jnn)

    Jpp=(Dp  *  geometry.carrier_face/ h) * (
        BpPlus*p[nodeLeft]- blah *p[nr]
    )
    fp  +=  R * volume
    np.add.at(fp,nodeLeft,Jpp)
    np.add.at(fp, nr, - Jpp)

    return outt



class NodeRange(Enum) :

    ALL = "all"

    LEFT = "left"
    RIGHT="right"

class _Triplets:

    def __init__(
        self, n_nodes  :  int, geometry  : EdgeGeometry = UNIFORM_1D
    ) ->  None  :
        NodeLeft,zz=geometry.ends_of(n_nodes)
        self._nodes  ={
            NodeRange.ALL:np.arange(n_nodes, dtype= np.int64),
            NodeRange.LEFT : NodeLeft,
            NodeRange.RIGHT : zz,
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
    yy,   range =   effective_potentials(  psi,  n,  p,  degeneracy)
    return _jacobian_from(h, volume, x, _diffusivity_at(Dn,psi,h,geometry), _diffusivity_at(Dp,psi,h,geometry), _bernoulli_pair(yy,geometry), _bernoulli_pair(range,geometry), _bernoulli_derivative_pair(yy,geometry), _bernoulli_derivative_pair(range,geometry), np.asarray(recombination.d_rate_dn(n,p),dtype=np.float64), np.asarray(recombination.d_rate_dp(n,p),dtype=np.float64), geometry, _diffusivity_tangent(Dn,psi,h,geometry), _diffusivity_tangent(Dp,psi,h,geometry), * _potential_tangents(n,p,degeneracy),)



def _potential_tangents(n :npt.NDArray[np.float64], p  : npt.NDArray[np.float64], degeneracy :  Degeneracy |None,)  -> tuple[npt.NDArray[np.float64]|None, npt.NDArray[np.float64] | None] :

    if degeneracy is None:
        return None,   None
    return degeneracy.d_electron_potential_dn(n),degeneracy.d_hole_potential_dp(p)

def _jacobian_from(h  :npt.NDArray[np.float64], volume : npt.NDArray[np.float64], x :  npt.NDArray[np.float64], Dn  : Diffusivity, Dp :  Diffusivity, bernoulli_n : tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], bernoulli_p: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], dbernoulli_n  :tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], dbernoulli_p: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], dR_dn :npt.NDArray[np.float64], dR_dp :  npt.NDArray[np.float64], geometry  : EdgeGeometry =UNIFORM_1D, dDn_dX : npt.NDArray[np.float64] |  None  = None, dDp_dX : npt.NDArray[np.float64]  | None =None, dpsi_n_dn :  npt.NDArray[np.float64] | None =  None, dpsi_p_dp : npt.NDArray[np.float64]  |  None  = None,)-> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    psi, n, p=unpack(x)
    acc  =  psi.size

    str,hex =bernoulli_n
    dbPlus, db =dbernoulli_n

    blah,bp_mius=bernoulli_p

    dp, t2 = dbernoulli_p


    nod =NodeRange.ALL
    lef =NodeRange.LEFT;  rig  =  NodeRange.RIGHT

    j = _Triplets(acc, geometry)
    NodeLeft,filter = geometry.ends(h.size)



    Conductance  =   geometry.weight  /  h
    dia  =   np.zeros( acc )
    np.add.at(dia,NodeLeft,Conductance)
    np.add.at(  dia,  filter ,  Conductance)
    j.add( Unknown.PSI, nod ,  Unknown.PSI, nod,  dia) ; j.add(Unknown.PSI, lef, Unknown.PSI, rig, - Conductance)


    j.add (  Unknown.PSI ,   rig ,  Unknown.PSI,  lef,  - Conductance )

    j.add( Unknown.PSI,  nod , Unknown.N ,   nod, volume  )
    j.add(Unknown.PSI,nod,Unknown.P,nod,-volume)

    gs =(Dn *geometry.carrier_face/h)* (
        dbPlus * n[filter]+db*n[NodeLeft]
    )
    GG = gs
    if dDn_dX is not None:
        GG= GG + (dDn_dX *geometry.carrier_face /h) * (
            str  * n[filter] - hex  *n[NodeLeft]
        )
    dia= np.zeros(acc)
    np.add.at(dia, NodeLeft, GG)
    np.add.at (dia , filter,  GG)
    j.add(Unknown.N,nod,Unknown.PSI,nod,dia)
    j.add(Unknown.N,lef,Unknown.PSI,rig,- GG)
    j.add(Unknown.N ,
                     rig,
                    Unknown.PSI,
            lef,
        -  GG  )

    bb= (Dn * geometry.carrier_face/ h)* str

    tl = (Dn * geometry.carrier_face/h)*  hex
    if  dpsi_n_dn is  not None   :

        bb=bb+ gs * dpsi_n_dn[filter]
        tl  =  tl  +  gs *   dpsi_n_dn [ NodeLeft ]
    dia= dR_dn* volume
    np.add.at(dia,NodeLeft,tl)
    np.add.at(dia, filter, bb)
    j.add(Unknown.N,nod,Unknown.N,nod,dia) ; j.add(Unknown.N, lef, Unknown.N, rig, -bb)
    j.add(  Unknown.N,   rig ,  Unknown.N ,   lef ,   -  tl)

    j.add(Unknown.N,nod,Unknown.P,nod,dR_dp *volume)

    junk   =   (  Dp   *  geometry.carrier_face /  h  ) *  (
        dp *  p [ NodeLeft ]   + t2  *  p[filter ]
    )
    HH = junk
    if dDp_dX  is not None   :
        HH=  HH +  (dDp_dX * geometry.carrier_face/ h) *(
            blah * p[NodeLeft] - bp_mius*p[filter]
        )
    dia =  np.zeros (  acc  )
    np.add.at(dia, NodeLeft, -HH)
    np.add.at(  dia ,  filter ,  - HH)
    j.add(Unknown.P, nod, Unknown.PSI, nod, dia)

    j.add(Unknown.P,lef,Unknown.PSI,rig,HH)
    j.add(Unknown.P, rig, Unknown.PSI, lef, HH)


    j.add(Unknown.P,nod,Unknown.N,nod,dR_dn * volume)
    tl  =   (Dp *   geometry.carrier_face   /  h  )   * blah
    bb= (Dp *  geometry.carrier_face /  h) *bp_mius
    if dpsi_p_dp  is  not None   :

        tl =  tl -  junk *dpsi_p_dp[NodeLeft];bb = bb  -junk *dpsi_p_dp[filter]
    dia=dR_dp*volume
    np.add.at(  dia, NodeLeft, tl  )
    np.add.at(dia, filter, bb)
    j.add(Unknown.P, nod, Unknown.P, nod, dia); j.add(Unknown.P,lef,Unknown.P,rig,-bb)

    j.add (Unknown.P,  rig,   Unknown.P , lef,  -   tl  )

    return j.build()


def assemble_coupled(mesh : Mesh1D, psi :Field, n  :Field, p  : Field, net_doping :Field, recombination : RecombinationModel, scale : ScaleFactors, Dn :EdgeDiffusivity, Dp :EdgeDiffusivity, degeneracy:Degeneracy | None  =  None,)  -> SparseAssembly  :
    for Name, map in(
        ('psi', psi),
        ("n", n),
        ("p", p),
        ("net_doping", net_doping),
    ):
        if map.scaling is not ScalingState.SCALED:
            raise  ValueError (
                f"{Name} must be SCALED before assembly, got {map.scaling.name}. "
                'A physical value here is wrong by a fixed factor and would '
                "still converge."
            )
        if map.location  is not  Location.NODE  :
            raise ValueError(f"{Name} must live on NODE, got {map.location.name}.")
        if map.size  != mesh.n_nodes:


            raise ValueError(
                f"{Name} has length {map.size} but the mesh has "
                f"{mesh.n_nodes} nodes."
            )
    max: npt.NDArray[np.float64]= pack(psi.data,
                     n.data,
                    p.data)

    return assemble_coupled_arrays (
        h =  mesh.h  /  scale.x_0 ,
        volume = mesh.volume   /   scale.x_0,
        x  = max,
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
    chr, max =effective_potentials(psi, n, p, degeneracy)

    bernooulli_n = _bernoulli_pair(chr, geometry)

    bernoullip=(
        bernooulli_n if degeneracy is None else _bernoulli_pair(max,geometry)
    )
    dbernoullin =_bernoulli_derivative_pair(chr, geometry)
    dbe= (dbernoullin if degeneracy is None else _bernoulli_derivative_pair(max,geometry))
    dppsi_n_dn, dppsi_p_dp=  _potential_tangents(n, p, degeneracy)
    r = np.asarray(recombination.rate(n, p), dtype = np.float64)
    DrDn  =np.asarray(recombination.d_rate_dn(n, p), dtype = np.float64)
    drdp=np.asarray(recombination.d_rate_dp(n,p),dtype=np.float64)


    dn=_diffusivity_at(Dn,psi,h,geometry)
    dp =_diffusivity_at(Dp, psi, h, geometry)
    dd  =_diffusivity_tangent(Dn, psi, h, geometry)
    ddp  = _diffusivity_tangent(Dp, psi, h, geometry)
    xx  =  _residual_from(h, volume, x, net_doping, dn, dp, bernooulli_n, bernoullip, r, geometry,)
    Rows,   col ,  t2   =  _jacobian_from(h, volume, x, dn , dp, bernooulli_n, bernoullip, dbernoullin , dbe, DrDn, drdp , geometry, dd, ddp, dppsi_n_dn , dppsi_p_dp,)
    yy=_term_scales_from(
        h,
        volume,
        x,
        net_doping,
        dn,
        dp,
        bernooulli_n,
        bernoullip,
        r,
        geometry,
    )
    sum = x.size
    return CoupledAssembly(
        assembly =SparseAssembly(
            residual  = np.asarray(xx, dtype = np.float64),
            rows =  Rows,
            cols = col,
            values = t2,
            shape =  (sum, sum),
        ),
        scales  = yy,
    )



def limit_psi_step(delta : npt.NDArray[np.float64],max_psi_step:float)-> npt.NDArray[np.float64]:
    Dpsi  =   delta [Unknown.PSI  ::   UNKNOWNS_PER_NODE ]


    tmp  = float(np.max(np.abs(Dpsi)))
    if  tmp  <=  max_psi_step :
        return delta
    Limited =delta.copy()
    Limited[Unknown.PSI ::UNKNOWNS_PER_NODE]=Dpsi* (max_psi_step / tmp);return Limited



def apply_contacts_coupled(assembly: SparseAssembly, x :npt.NDArray[np.float64], net_doping  : npt.NDArray[np.float64], contacts :  Sequence[Contact], scale : ScaleFactors, carrier_free_nodes : Sequence[int] =  (), T:float =C.T_ROOM, degeneracy:Degeneracy | None  = None,)->SparseAssembly  :
    xx=[dat.name for dat in contacts]
    if  len(  set(  xx  )  )  !=  len(  xx )   :
        raise ValueError(f"contact names must be unique, got {xx}")
    Indices :list[int] = []
    tar: list[float]=[]
    for dat in contacts  :
        appllied=dat.voltage /scale.psi_0

        if isinstance(dat, GateContact) :
            Target =gate_psi_scaled(appllied,dat.work_function,T)
            for idx2 in dat.nodes:
                Indices.append(unknown_index(idx2, Unknown.PSI))
                tar.append(Target)
            continue


        for idx2 in dat.nodes  :
            data2 =float(net_doping[idx2])

            Indices.append(unknown_index(idx2,
                       Unknown.PSI)); tar.append(ohmic_psi_scaled(data2, appllied, degeneracy))
            Indices.append(unknown_index(idx2,
                          Unknown.N))
            tar.append(ohmic_density_scaled(data2,Carrier.ELECTRON,degeneracy))

            Indices.append(unknown_index(idx2,Unknown.P))
            tar.append(ohmic_density_scaled(  data2, Carrier.HOLE,   degeneracy))

    for idx2 in carrier_free_nodes:
        Indices.append(  unknown_index (idx2,   Unknown.N  )  )
        tar.append(0.0)
        Indices.append(  unknown_index(  idx2,   Unknown.P ))
        tar.append(0.0)
    return apply_dirichlet_nodes(  assembly, x, Indices, tar)



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
    PsiN, psiP= effective_potentials(psi, n, p, degeneracy)
    return _term_scales_from(h, volume, x, net_doping, _diffusivity_at(Dn,psi,h,geometry), _diffusivity_at(Dp,psi,h,geometry), _bernoulli_pair(PsiN,geometry), _bernoulli_pair(psiP,geometry), R, geometry,)

def _term_scales_from(h :npt.NDArray[np.float64], volume: npt.NDArray[np.float64], x: npt.NDArray[np.float64], net_doping:npt.NDArray[np.float64], Dn:Diffusivity, Dp :Diffusivity, bernoulli_n:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]], bernoulli_p:tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]], R:npt.NDArray[np.float64]| None, geometry : EdgeGeometry=UNIFORM_1D,)->TermScales :

    psi, n, p=  unpack(x)


    bn_pus, str = bernoulli_n
    bpPlus, buf = bernoulli_p
    NodeLeft,node_riight= geometry.ends(h.size)
    w= volume.size
    PsiEdge = (geometry.weight  / h)* np.maximum(
        np.abs(psi[NodeLeft]), np.abs(psi[node_riight])
    )
    psiscale= np.maximum(_largest_at_each_node(PsiEdge,NodeLeft,node_riight,w), (np.abs(p)+np.abs(n)+np.abs(net_doping)) *volume,)

    Recombined =  (
        np.zeros(w) if R is None else np.abs(R)* volume
    )
    gnn  = Dn * geometry.carrier_face/ h
    Gp =  Dp * geometry.carrier_face /  h
    stuff=np.maximum(
        _largest_at_each_node(
            np.maximum(gnn* bn_pus* n[node_riight],gnn *str *n[NodeLeft]),
            NodeLeft,
            node_riight,
            w,
        ),
        Recombined,
    )
    tmp = np.maximum(
        _largest_at_each_node(
            np.maximum(Gp*bpPlus* p[NodeLeft], Gp * buf *  p[node_riight]),
            NodeLeft,
            node_riight,
            w,
        ),
        Recombined,
    )


    Scales= (psiscale,stuff,tmp)

    if not  all(bool(  np.all(  np.isfinite( scale  ) ) )  for scale in  Scales)  :
        raise FloatingPointError(
            "a residual term overflowed, the iterate has diverged: term scales "
            f"max to {tuple(float(np.max(scale)) for scale in Scales)} for "
            '(psi, n, p).'
        )
    if  not all(np.any(scale  >  0.0 )   for scale in Scales  ) :
        raise ValueError(
            f"the state has no terms to measure a residual against: term "
            f"scales max to "
            f"{tuple(float(np.max(scale)) for scale in Scales)} for (psi, n, p). "
            'Every equation is identically zero, which a device never is. '
            "Dividing by these would rename the problem as a singular matrix "
            'three call frames later.'
        )

    return Scales



def _largest_at_each_node(edge_term : npt.NDArray[np.float64], node_left  : npt.NDArray[np.int64], node_right : npt.NDArray[np.int64], n_nodes  :int,) ->  npt.NDArray[np.float64] :
    laargest= np.zeros(n_nodes)
    np.maximum.at(laargest,node_left,edge_term) ; np.maximum.at(laargest, node_right, edge_term)
    return laargest


def row_weights(scales :TermScales, n_nodes : int)  ->npt.NDArray[np.float64]:


    Weights=  np.empty(UNKNOWNS_PER_NODE  *n_nodes)

    for componeent, scle in zip(Unknown, scales, strict= True) :
        if scle.size!= n_nodes :
            raise ValueError(
                f"the {componeent.name} term scale has {scle.size} entries "
                f"for a mesh of {n_nodes} nodes"
            )
        Weights[ componeent   ::   UNKNOWNS_PER_NODE ]  =  np.max ( scle )
    return  Weights
FAMILIES  =  tuple(  unknown.name.lower(  )  for  unknown in Unknown  )




def residual_measure(residual : npt.NDArray[np.float64],scales:TermScales,n_nodes:int) ->float :

    return max(0.0,*residual_measure_by_family(residual,scales,n_nodes).values())


def residual_measure_by_family (
    residual :  npt.NDArray[np.float64  ],  scales : TermScales ,  n_nodes  : int
)  ->  dict [str,   float ]  :
    Weights =row_weights(scales,n_nodes)

    raww=np.abs(residual)*Weights
    ByFamily:dict[str,float]= {}
    for cmoponent,bin,nmae in zip(Unknown,scales,FAMILIES,strict =True) :
        rwos= raww[cmoponent:: UNKNOWNS_PER_NODE]
        Floor =EPS  *  float(np.max(bin))
        Measured = np.divide(
            rwos, bin, out = np.zeros_like(rwos), where  = bin > Floor
        )
        ByFamily[  nmae  ]  =  float (np.max(  Measured  )  )
    return ByFamily



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
    Dpsi, dnn, dpp = unpack(delta)
    _,n,p=unpack(x)
    return{
        "psi"  : float(np.max(np.abs(Dpsi))),
        "n"  :float(np.max(np.abs(dnn)  / (np.abs(n)+ 1.0))),
        "p"  :float(np.max(np.abs(dpp) / (np.abs(p)  + 1.0))),
    }
