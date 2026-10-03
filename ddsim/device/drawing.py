from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np; import numpy.typing as npt
from scipy import ndimage
from ddsim.core import constants as C
from ddsim.device.builder import Device, Material, build_device
from ddsim.device.doping import Along,Coordinates,DopingProfile,Sum,Window;from ddsim.device.mosfet import implant_lengths
from ddsim.device.regions import OXIDE,SILICON,region_map
from ddsim.device.stack import DOPING_RANGE, NODES_INSIDE
from ddsim.discretize.boundary import Contact,GateContact,OhmicPlate
from ddsim.mesh.mesh1d import graded_mesh_1d_through
from ddsim.mesh.mesh2d import Mesh2D,tensor_mesh_2d

MATERIALS  ={"silicon"  :SILICON, "oxide":  OXIDE}
DEGENERATE_DOPING_TOP=  1e20
NODE_BUDGET =20000



@dataclass(frozen =True)

class Block :

    material : str


    x0  :  float

    x1:float

    y0 :  float

    y1:float


@dataclass(frozen  =  True)




class  Implant  :
    dopant  :  str


    concentration:float
    x0: float

    x1 : float


    y0:float
    y1 :float

    profile :  str = "uniform"

    straggle :float=0.0



    lateral:float =0.0


@dataclass(frozen=True)

class  Electrode  :
    name: str
    kind :str

    x0: float

    x1  :  float

    y0 :float
    y1:float
    voltage  : float  =  0.0


    work_function:float =C.PHI_M_N_POLY

Drawing =tuple[tuple[Block, ...], tuple[Implant, ...], tuple[Electrode, ...]]

def _nmos_drawing()->Drawing:
    c,z,el,Na,g =1e-4,4e-5,2e-5,1e17,1e20;d ,  u   = 2e-6, 1e-4
    x, info =2.0 *  z  + c, u  + d
    xs,h= implant_lengths(1.5e-5,1e-5,g,Na)
    return((Block("silicon", 0.0, x, 0.0, u), Block("oxide", 0.0, x, u, info),), (Implant('p', Na, 0.0, x, 0.0, u), Implant('n', g, 0.0, z, u, info, "gaussian", xs, h), Implant("n", g, x - z, x, u, info, "gaussian", xs, h),), (Electrode('source', "ohmic", 0.0, el, u, u), Electrode('drain', 'ohmic', x -el, x, u, u), Electrode('gate', "gate", z, x - z, info, info), Electrode('body', "ohmic", 0.0, x, 0.0, 0.0),),)

def _mos_cap_drawing()-> Drawing :
    Na, el, b, dd=  1e16, 1e-6, 2e-4, 1e-5
    m =  b  +  el
    return(
        (
            Block("silicon",0.0,dd,0.0,b),
            Block("oxide",0.0,dd,b,m),
        ),
        (Implant("p",Na,0.0,dd,0.0,b),),
        (
            Electrode('body',"ohmic",0.0,dd,0.0,0.0),
            Electrode('gate','gate',0.0,dd,m,m),
        ),
    )

NMOS_DRAWING   = _nmos_drawing()

MOS_CAP_DRAWING  =  _mos_cap_drawing (  )



def _box(what :str,x0 :float,x1 :float,y0:float,y1 : float)->str:
    return f"{what} ({x0:g} to {x1:g} cm across, {y0:g} to {y1:g} cm up)"



def _check_records(
    blocks :tuple[Block,...],
    implants: tuple[Implant,...],
    electrodes:tuple[Electrode,...],
    degenerate:bool,
)->None:
    for kk, e in enumerate(blocks, start=1)  :

        if e.material not in MATERIALS :
            raise ValueError(
                f"block {kk}: a block is silicon or oxide, got {e.material!r}"
            )
        if not(e.x0<e.x1 and e.y0<e.y1) :
            raise ValueError(
                f"block {kk}: a block needs a positive width and height, "
                "x0 < x1 and y0 < y1, got "
                + _box('it', e.x0, e.x1, e.y0, e.y1)
            )

    u=DOPING_RANGE[0]
    m2  =   DEGENERATE_DOPING_TOP  if  degenerate else  DOPING_RANGE[1  ]
    for kk, i in enumerate(implants, start =  1):
        if i.dopant not in('n','p'):


            raise ValueError(
                f"implant {kk}: the dopant is 'n' or 'p', got {i.dopant!r}"
            )
        if  i.profile not in(  'uniform', 'gaussian' )  :


            raise ValueError(
                f"implant {kk}: a profile is uniform or gaussian, got "
                f"{i.profile!r}"
            )
        if i.profile == "gaussian" and not(
            i.straggle> 0.0 and i.lateral > 0.0
        ) :
            raise ValueError(
                f"implant {kk}: a gaussian implant needs a positive straggle "
                f"and lateral, got straggle={i.straggle:g} and "
                f"lateral={i.lateral:g} cm"
            )

        if not(i.x0< i.x1 and i.y0< i.y1):

            raise  ValueError(
                f"implant {kk}: an implant needs a positive width and "
                "height, x0 < x1 and y0 < y1"
            )
        if not  u <=   i.concentration  <= m2   :
            d =(
                ''
                if degenerate or i.concentration<u
                else f" Above {m2:g}, Boltzmann statistics put the Fermi level "
                "in the wrong place. Turn on degenerate (Fermi-Dirac "
                "statistics), the way nmos does."
            )
            raise ValueError(
                f"implant {kk}: a concentration of {i.concentration:g} "
                f"cm^-3 is outside {u:g} to {m2:g} cm^-3, the range the "
                f"models here are built for (see references/physics.md).{d}"
            )
    for val in electrodes:
        if val.kind not in("ohmic",'gate'):
            raise ValueError(
                f"electrode {val.name!r}: an electrode is ohmic or gate, got "
                f"{val.kind!r}. A metal on silicon that is not ohmic is a "
                "Schottky contact, which this solver does not model."
            )
        val2=val.x1 -val.x0
        flag  = val.y1   - val.y0
        if not((val2 >  0.0 and flag ==  0.0)or(flag> 0.0 and val2  == 0.0)) :
            raise  ValueError(
                f"electrode {val.name!r}: an electrode is a straight line "
                'along x or along y, with x0 < x1 and y0 == y1 or the other way '
                "round, got "
                +  _box("it", val.x0,  val.x1 , val.y0 , val.y1 )
            )




def _lines(
    things :  list[tuple[str, float, float]], h_min:float, axis  :str
) ->  list[float]:
    j : dict[float,list[str]] ={}
    for dd,w,val2 in things:
        j.setdefault(w, []).append(dd)
        j.setdefault(val2,
          []).append(dd)
    u= sorted(j)
    for  z ,   prev in  zip( u [ :-   1  ],
          u[  1  :],
          strict  =  True  )  :

        if prev  - z<h_min:
            raise ValueError(
                f"{', '.join(j[z])} and {', '.join(j[prev])} put mesh "
                f"lines {prev - z:g} cm apart, at {axis} = {z:g} and {prev:g} cm, "
                f"closer than h_min_{axis}={h_min:g} cm. That is a feature "
                'smaller than the mesh resolves. Line the edges up, or move them '
                "at least h_min apart."
            )
    return u


def _paint(
    blocks  :tuple[Block, ...],
    x_edges : npt.NDArray[np.float64],
    y_edges :npt.NDArray[np.float64],
) ->  npt.NDArray[np.int64] :
    k  =   0.5  * (  x_edges [ 1   :  ]   +  x_edges [  :-  1])
    r  = 0.5  *   (y_edges[  1  :]   +  y_edges[ :- 1  ])
    foo= np.full((r.size, k.size), -1, dtype = np.int64)
    for e in  blocks   :
        tmp2= np.outer((r>e.y0) &(r<e.y1), (k>e.x0) &(k< e.x1),)
        foo[tmp2]= MATERIALS[e.material]
    return foo
def _interfaces(
    cells:npt.NDArray[np.int64],
    x_edges:npt.NDArray[np.float64],
    y_edges : npt.NDArray[np.float64],
) -> tuple[set[float],set[float]]:
    y  =   cells[ : ,  1 :]  !=   cells[  : ,   :-  1 ]
    dat =  cells[1  :, :]!= cells[:-1, :]
    tmp2= {float(x_edges[u + 1]) for u in np.flatnonzero(y.any(axis=0))}
    a2   =  {float(y_edges [res   + 1 ]  ) for  res  in np.flatnonzero(  dat.any( axis  =  1))}
    return tmp2,a2



def _implant_profile(implant  :  Implant,   width   :  float,   height :   float)  ->   DopingProfile  :
    tmp2= -math.inf if implant.x0 <=0.0 else implant.x0
    xx=math.inf if implant.x1 >= width else implant.x1
    jj= -math.inf if implant.y0<=0.0 else implant.y0
    lst= math.inf if implant.y1>= height else implant.y1
    if  implant.profile ==  'uniform'  :
        v =  Window(tmp2, xx)
        j = Window(jj, lst)
    else:
        v = Window(tmp2,xx,"erfc",implant.lateral)
        j= Window(jj,lst,'gaussian',implant.straggle)
    w=1.0 if implant.dopant== "n" else- 1.0

    return  Along ( v ,   "x" ) * Along(  j, 'y')  *   (  w   *  implant.concentration)



def _electrode_nodes(mesh  :  Mesh2D, electrode  : Electrode)->  tuple[int, ...]:

    r2 = (
        (mesh.node_x  >= electrode.x0)
        & (mesh.node_x  <=  electrode.x1)
        &(mesh.node_y >= electrode.y0)
        & (mesh.node_y <=electrode.y1)
    )
    return tuple ( int(out)   for out  in np.flatnonzero( r2  )  )



def drawing(blocks  : tuple[  Block,   ... ]  =  NMOS_DRAWING[  0] , implants  :  tuple[ Implant,   ... ]  = NMOS_DRAWING [ 1 ], electrodes  : tuple [ Electrode , ... ] =  NMOS_DRAWING[ 2  ], nx  :  int =  63, ny  :   int  =  133, h_min_x   :  float  = 2e-7, h_min_y  :  float   =  6.25e-9 , degenerate  :  bool   = True, material  :  Material   | None =   None ,)  ->  Device   :
    _check_records ( blocks, implants, electrodes ,   degenerate  )
    if not blocks:
        raise ValueError("nothing is drawn: a device needs at least one block")

    i=  min(jj.x0 for jj in blocks)
    z2 =  min(tt.y0  for  tt  in blocks )
    if  i  !=   0.0  or  z2   !=  0.0 :

        raise  ValueError(
            f"the drawing starts at x={i:g}, y={z2:g} cm; draw it "
            "from the origin, x = 0 and y = 0 at its lower left corner"
        )

    bar =max(ii.x1 for ii in blocks)
    b=max(item.y1 for item in blocks)

    zz  : list[tuple [  str,
      Block | Implant  |   Electrode]  ]  = [ ]
    zz  += [(f"block {n}",   u )   for  n ,   u  in  enumerate(  blocks , start  = 1)]
    zz  +=   [ (  f"implant {n}",  kk  ) for n, kk in enumerate(  implants , start =  1)]
    zz  +=   [  (  f"electrode {el.name!r}" ,   el  ) for el in electrodes]
    for dd, vv in zz :
        out  =  (0.0 <= vv.x0 and vv.x1   <= bar and 0.0  <= vv.y0 and  vv.y1  <=  b)
        if not out:
            raise ValueError(
                f"{_box(dd, vv.x0, vv.x1, vv.y0, vv.y1)} reaches "
                f"outside the drawing, which is the blocks' extent: 0 to "
                f"{bar:g} cm across and 0 to {b:g} cm up"
            )


    if nx  *  ny  >   NODE_BUDGET :
        raise ValueError(
            f"a mesh of {nx} by {ny} is {nx * ny} nodes, over the budget of "
            f"{NODE_BUDGET}. A 2D solve slows faster than its node count grows; "
            'the benchmark nmos is 8379 nodes and about 20 s for a transfer curve.'
        )


    ss = _lines([(thing, w2.x0, w2.x1) for thing, w2 in zz], h_min_x, 'x');  row=_lines([(thing,w2.y0,w2.y1) for thing,w2 in zz],h_min_y,'y')
    m =_paint(blocks,np.array(ss),np.array(row))
    if(m < 0).any() :
        nxt, kk =(int(bb[0]) for bb in np.nonzero(m < 0))
        raise ValueError(
            f"nothing is drawn between x = {ss[kk]:g} and {ss[kk + 1]:g} "
            f"cm, y = {row[nxt]:g} and {row[nxt + 1]:g} cm. An undrawn gap "
            'is vacuum, which this solver has no material for. Cover it with a '
            "block."
        )

    if not(m== SILICON).any() :
        raise ValueError("the drawing has no silicon in it, so nothing to simulate")
    mm, xs =  _interfaces(m, np.array(ss), np.array(row))
    a2  =  mm  |  {kk.x0  for kk in implants if  kk.x0  >  0.0  }
    a2 |= {kk.x1 for kk in implants if kk.x1  <  bar}

    res2 = xs|{kk.y0 for kk in implants if kk.y0 > 0.0}
    res2  |={kk.y1 for kk in implants if kk.y1 <b}
    cc  =  tensor_mesh_2d(graded_mesh_1d_through(bar,   nx, tuple(ss ) ,  tuple (sorted(a2 ) ), h_min_x) , graded_mesh_1d_through (b,  ny ,  tuple(  row) , tuple(  sorted ( res2) ),  h_min_y),)
    r2,v2 =cc.x_axis.x,cc.y_axis.x
    for dd,vv in zz[:len(blocks) +len(implants)]:
        prev=int(np.count_nonzero((r2>vv.x0)&(r2 < vv.x1)))
        val2 = int(np.count_nonzero((v2>vv.y0)  &(v2 < vv.y1)))
        if vv.x0  ==  0.0 and  vv.x1  == bar :
            prev =   NODES_INSIDE
        if  vv.y0 == 0.0 and  vv.y1  ==  b  :


            val2 = NODES_INSIDE
        if min(prev, val2)  <  NODES_INSIDE :
            raise ValueError(
                f"{_box(dd, vv.x0, vv.x1, vv.y0, vv.y1)} is smaller "
                f"than the mesh resolves: it holds {prev} node columns and "
                f"{val2} node rows inside it and needs at least "
                f"{NODES_INSIDE} of each. Make it bigger, or refine the mesh "
                'with more nodes or a smaller h_min.'
            )

    ok= _paint(blocks, r2, v2); xx =region_map(cc,ok)
    v  = xx.semiconductor_volume   > 0.0
    yy :  list[  Contact]  =  [ ]
    b2 :dict[int, str] = {}
    for d2 in electrodes :
        d=_electrode_nodes(cc,d2)
        aa=_box(
            f"electrode {d2.name!r}",
            d2.x0,
            d2.x1,
            d2.y0,
            d2.y1,
        )
        if d2.kind  ==  'ohmic'  :
            if not v[list(d)].all() :
                raise ValueError(
                    f"{aa} is ohmic, and part of it sits on oxide with no "
                    "silicon under it, so no carrier density to pin there. Keep "
                    "it on silicon, or make it a gate."
                )
            yy.append(OhmicPlate(name =d2.name,nodes=d,voltage = d2.voltage))
        else:
            if v[list(d)].any():
                raise  ValueError(
                    f"{aa} is a gate touching silicon. A metal on silicon is "
                    "a Schottky contact, which this solver does not model: a "
                    "gate sits on oxide. Put oxide under it, or make it ohmic."
                )
            yy.append(
                GateContact(
                    name =d2.name,
                    nodes = d,
                    voltage =d2.voltage,
                    work_function=d2.work_function,
                )
            )
        for hh in d:
            if hh  in b2  :
                raise ValueError(
                    f"electrodes {b2[hh]!r} and {d2.name!r} share "
                    f"the node at x={cc.node_x[hh]:g}, y={cc.node_y[hh]:g}"
                    " cm. One node cannot hold two contacts."
                )
            b2[hh]  = d2.name


    ys= {
        hh
        for val in yy
        if isinstance(val,OhmicPlate)
        for hh in val.nodes
    }
    lst,  h  = ndimage.label(ok  ==  SILICON )
    for t2 in range(1, h+  1) :
        c, vals  = np.nonzero(lst ==t2)
        m2 ={
            cc.node_at(int(kk)+ arr,int(nxt) +tmp)
            for nxt,kk in zip(c,vals,strict =True)
            for tmp in(0,1)
            for arr in(0,1)
        }
        if not m2 &  ys :
            out2 =0.5* (r2[vals[0]] + r2[vals[0]+1])
            z=0.5 *(v2[c[0]]+v2[c[0]+ 1])
            raise ValueError(
                f"the silicon around x={out2:g}, y={z:g} cm floats: no ohmic "
                'contact touches it, so nothing sets its Fermi level and its '
                "charge is whatever the solver starts from. Put an ohmic "
                'electrode on it.'
            )


    j= Sum(tuple(_implant_profile(tmp3, bar, b) for tmp3 in implants))

    foo =  j(  Coordinates ( cc.node_x,  cc.node_y)  ).reshape(  cc.ny ,  cc.nx)
    k2 =v.reshape(cc.ny, cc.nx)
    for s2, c2, w in(("p", 'holes', foo <  0.0), ('n', 'electrons', foo > 0.0),)  :

        y2, h =ndimage.label(k2 &w)
        for  cur  in  range(  1,  h   +  1 )  :
            rr=set(np.flatnonzero(y2.ravel()== cur).tolist())
            if not rr  & ys:
                hh =   min(rr  )
                raise ValueError(
                    f"the {s2} silicon around x={cc.node_x[hh]:g}, "
                    f"y={cc.node_y[hh]:g} cm floats: no ohmic contact "
                    f"touches it, so its {c2} reach a contact only "
                    'through a junction. That leakage is too small for the '
                    "solver to pin the region's potential, and the solve "
                    "stalls. This solver can't handle a floating body, so tie "
                    f"it down with an ohmic electrode on the {s2} region."
                )

    return  build_device(
        mesh  = cc ,
        doping  = j,
        contacts  = tuple (yy ),
        material = material,
        regions  = xx ,
        degenerate  =   degenerate,
    )
