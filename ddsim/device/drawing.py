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
    L_gtae,sdd,tmp2,Na,x2 =1e-4,4e-5,2e-5,1e17,1e20;t ,  TSi   = 2e-6, 1e-4
    wdith, w =2.0 *  sdd  + L_gtae, TSi  + t
    simga,edg= implant_lengths(1.5e-5,1e-5,x2,Na)
    return((Block("silicon", 0.0, wdith, 0.0, TSi), Block("oxide", 0.0, wdith, TSi, w),), (Implant('p', Na, 0.0, wdith, 0.0, TSi), Implant('n', x2, 0.0, sdd, TSi, w, "gaussian", simga, edg), Implant("n", x2, wdith - sdd, wdith, TSi, w, "gaussian", simga, edg),), (Electrode('source', "ohmic", 0.0, tmp2, TSi, TSi), Electrode('drain', 'ohmic', wdith -tmp2, wdith, TSi, TSi), Electrode('gate', "gate", sdd, wdith - sdd, w, w), Electrode('body', "ohmic", 0.0, wdith, 0.0, 0.0),),)

def _mos_cap_drawing()-> Drawing :
    Na, tOx, tSi, wid=  1e16, 1e-6, 2e-4, 1e-5
    topp =  tSi  +  tOx
    return(
        (
            Block("silicon",0.0,wid,0.0,tSi),
            Block("oxide",0.0,wid,tSi,topp),
        ),
        (Implant("p",Na,0.0,wid,0.0,tSi),),
        (
            Electrode('body',"ohmic",0.0,wid,0.0,0.0),
            Electrode('gate','gate',0.0,wid,topp,topp),
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
    for nmuber, sum in enumerate(blocks, start=1)  :

        if sum.material not in MATERIALS :
            raise ValueError(
                f"block {nmuber}: a block is silicon or oxide, got {sum.material!r}"
            )
        if not(sum.x0<sum.x1 and sum.y0<sum.y1) :
            raise ValueError(
                f"block {nmuber}: a block needs a positive width and height, "
                "x0 < x1 and y0 < y1, got "
                + _box('it', sum.x0, sum.x1, sum.y0, sum.y1)
            )

    tmp=DOPING_RANGE[0]
    hig  =   DEGENERATE_DOPING_TOP  if  degenerate else  DOPING_RANGE[1  ]
    for nmuber, q in enumerate(implants, start =  1):
        if q.dopant not in('n','p'):


            raise ValueError(
                f"implant {nmuber}: the dopant is 'n' or 'p', got {q.dopant!r}"
            )
        if  q.profile not in(  'uniform', 'gaussian' )  :


            raise ValueError(
                f"implant {nmuber}: a profile is uniform or gaussian, got "
                f"{q.profile!r}"
            )
        if q.profile == "gaussian" and not(
            q.straggle> 0.0 and q.lateral > 0.0
        ) :
            raise ValueError(
                f"implant {nmuber}: a gaussian implant needs a positive straggle "
                f"and lateral, got straggle={q.straggle:g} and "
                f"lateral={q.lateral:g} cm"
            )

        if not(q.x0< q.x1 and q.y0< q.y1):

            raise  ValueError(
                f"implant {nmuber}: an implant needs a positive width and "
                "height, x0 < x1 and y0 < y1"
            )
        if not  tmp <=   q.concentration  <= hig   :
            Why =(
                ''
                if degenerate or q.concentration<tmp
                else f" Above {hig:g}, Boltzmann statistics put the Fermi level "
                "in the wrong place. Turn on degenerate (Fermi-Dirac "
                "statistics), the way nmos does."
            )
            raise ValueError(
                f"implant {nmuber}: a concentration of {q.concentration:g} "
                f"cm^-3 is outside {tmp:g} to {hig:g} cm^-3, the range the "
                f"models here are built for (see docs/01-physics.md).{Why}"
            )
    for hmm in electrodes:
        if hmm.kind not in("ohmic",'gate'):
            raise ValueError(
                f"electrode {hmm.name!r}: an electrode is ohmic or gate, got "
                f"{hmm.kind!r}. A metal on silicon that is not ohmic is a "
                "Schottky contact, which this solver does not model."
            )
        acr=hmm.x1 -hmm.x0
        Up  = hmm.y1   - hmm.y0
        if not((acr >  0.0 and Up ==  0.0)or(Up> 0.0 and acr  == 0.0)) :
            raise  ValueError(
                f"electrode {hmm.name!r}: an electrode is a straight line "
                'along x or along y, with x0 < x1 and y0 == y1 or the other way '
                "round, got "
                +  _box("it", hmm.x0,  hmm.x1 , hmm.y0 , hmm.y1 )
            )




def _lines(
    things :  list[tuple[str, float, float]], h_min:float, axis  :str
) ->  list[float]:
    owers : dict[float,list[str]] ={}
    for set,loww,hig in things:
        owers.setdefault(loww, []).append(set)
        owers.setdefault(hig,
          []).append(set)
    lin= sorted(owers)
    for  aa ,   bb in  zip( lin [ :-   1  ],
          lin[  1  :],
          strict  =  True  )  :

        if bb  - aa<h_min:
            raise ValueError(
                f"{', '.join(owers[aa])} and {', '.join(owers[bb])} put mesh "
                f"lines {bb - aa:g} cm apart, at {axis} = {aa:g} and {bb:g} cm, "
                f"closer than h_min_{axis}={h_min:g} cm. That is a feature "
                'smaller than the mesh resolves. Line the edges up, or move them '
                "at least h_min apart."
            )
    return lin


def _paint(
    blocks  :tuple[Block, ...],
    x_edges : npt.NDArray[np.float64],
    y_edges :npt.NDArray[np.float64],
) ->  npt.NDArray[np.int64] :
    ceentre_x  =   0.5  * (  x_edges [ 1   :  ]   +  x_edges [  :-  1])
    CentreY  = 0.5  *   (y_edges[  1  :]   +  y_edges[ :- 1  ])
    cnt= np.full((CentreY.size, ceentre_x.size), -1, dtype = np.int64)
    for all in  blocks   :
        Inside= np.outer((CentreY>all.y0) &(CentreY<all.y1), (ceentre_x>all.x0) &(ceentre_x< all.x1),)
        cnt[Inside]= MATERIALS[all.material]
    return cnt
def _interfaces(
    cells:npt.NDArray[np.int64],
    x_edges:npt.NDArray[np.float64],
    y_edges : npt.NDArray[np.float64],
) -> tuple[set[float],set[float]]:
    Across  =   cells[ : ,  1 :]  !=   cells[  : ,   :-  1 ]
    id =  cells[1  :, :]!= cells[:-1, :]
    myvar= {float(x_edges[ii + 1]) for ii in np.flatnonzero(Across.any(axis=0))}
    Floors   =  {float(y_edges [jj   + 1 ]  ) for  jj  in np.flatnonzero(  id.any( axis  =  1))}
    return myvar,Floors



def _implant_profile(implant  :  Implant,   width   :  float,   height :   float)  ->   DopingProfile  :
    val= -math.inf if implant.x0 <=0.0 else implant.x0
    zip=math.inf if implant.x1 >= width else implant.x1
    bot= -math.inf if implant.y0<=0.0 else implant.y0
    topp= math.inf if implant.y1>= height else implant.y1
    if  implant.profile ==  'uniform'  :
        acrross =  Window(val, zip)
        vars = Window(bot, topp)
    else:
        acrross = Window(val,zip,"erfc",implant.lateral)
        vars= Window(bot,topp,'gaussian',implant.straggle)
    sgin=1.0 if implant.dopant== "n" else- 1.0

    return  Along ( acrross ,   "x" ) * Along(  vars, 'y')  *   (  sgin   *  implant.concentration)



def _electrode_nodes(mesh  :  Mesh2D, electrode  : Electrode)->  tuple[int, ...]:

    onn = (
        (mesh.node_x  >= electrode.x0)
        & (mesh.node_x  <=  electrode.x1)
        &(mesh.node_y >= electrode.y0)
        & (mesh.node_y <=electrode.y1)
    )
    return tuple ( int(node)   for node  in np.flatnonzero( onn  )  )



def drawing(blocks  : tuple[  Block,   ... ]  =  NMOS_DRAWING[  0] , implants  :  tuple[ Implant,   ... ]  = NMOS_DRAWING [ 1 ], electrodes  : tuple [ Electrode , ... ] =  NMOS_DRAWING[ 2  ], nx  :  int =  63, ny  :   int  =  133, h_min_x   :  float  = 2e-7, h_min_y  :  float   =  6.25e-9 , degenerate  :  bool   = True, material  :  Material   | None =   None ,)  ->  Device   :
    _check_records ( blocks, implants, electrodes ,   degenerate  )
    if not blocks:
        raise ValueError("nothing is drawn: a device needs at least one block")

    OriginX=  min(block.x0 for block in blocks)
    oy =  min(block.y0  for  block  in blocks )
    if  OriginX  !=   0.0  or  oy   !=  0.0 :

        raise  ValueError(
            f"the drawing starts at x={OriginX:g}, y={oy:g} cm; draw it "
            "from the origin, x = 0 and y = 0 at its lower left corner"
        )

    ret =max(block.x1 for block in blocks)
    Height=max(block.y1 for block in blocks)

    draawn  : list[tuple [  str,
      Block | Implant  |   Electrode]  ]  = [ ]
    draawn  += [(f"block {n}",   bb )   for  n ,   bb  in  enumerate(  blocks , start  = 1)]
    draawn  +=   [ (  f"implant {n}",  ii  ) for n, ii in enumerate(  implants , start =  1)]
    draawn  +=   [  (  f"electrode {E.name!r}" ,   E  ) for E in electrodes]
    for whhat, thi in draawn :
        isnide  =  (0.0 <= thi.x0 and thi.x1   <= ret and 0.0  <= thi.y0 and  thi.y1  <=  Height)
        if not isnide:
            raise ValueError(
                f"{_box(whhat, thi.x0, thi.x1, thi.y0, thi.y1)} reaches "
                f"outside the drawing, which is the blocks' extent: 0 to "
                f"{ret:g} cm across and 0 to {Height:g} cm up"
            )


    if nx  *  ny  >   NODE_BUDGET :
        raise ValueError(
            f"a mesh of {nx} by {ny} is {nx * ny} nodes, over the budget of "
            f"{NODE_BUDGET}. A 2D solve slows faster than its node count grows; "
            'the benchmark nmos is 8379 nodes and about 20 s for a transfer curve.'
        )


    item2 = _lines([(ww, tt.x0, tt.x1) for ww, tt in draawn], h_min_x, 'x');  ylines=_lines([(ww,tt.y0,tt.y1) for ww,tt in draawn],h_min_y,'y')
    caorse =_paint(blocks,np.array(item2),np.array(ylines))
    if(caorse < 0).any() :
        jj, ii =(int(k[0]) for k in np.nonzero(caorse < 0))
        raise ValueError(
            f"nothing is drawn between x = {item2[ii]:g} and {item2[ii + 1]:g} "
            f"cm, y = {ylines[jj]:g} and {ylines[jj + 1]:g} cm. An undrawn gap "
            'is vacuum, which this solver has no material for. Cover it with a '
            "block."
        )

    if not(caorse== SILICON).any() :
        raise ValueError("the drawing has no silicon in it, so nothing to simulate")
    wal, flo =  _interfaces(caorse, np.array(item2), np.array(ylines))
    junk  =  wal  |  {ii.x0  for ii in implants if  ii.x0  >  0.0  }
    junk |= {ii.x1 for ii in implants if ii.x1  <  ret}

    ypoints = flo|{ii.y0 for ii in implants if ii.y0 > 0.0}
    ypoints  |={ii.y1 for ii in implants if ii.y1 <Height}
    meesh  =  tensor_mesh_2d(graded_mesh_1d_through(ret,   nx, tuple(item2 ) ,  tuple (sorted(junk ) ), h_min_x) , graded_mesh_1d_through (Height,  ny ,  tuple(  ylines) , tuple(  sorted ( ypoints) ),  h_min_y),)
    stuff,roows =meesh.x_axis.x,meesh.y_axis.x
    for whhat,thi in draawn[:len(blocks) +len(implants)]:
        insidex=int(np.count_nonzero((stuff>thi.x0)&(stuff < thi.x1)))
        insidey = int(np.count_nonzero((roows>thi.y0)  &(roows < thi.y1)))
        if thi.x0  ==  0.0 and  thi.x1  == ret :
            insidex =   NODES_INSIDE
        if  thi.y0 == 0.0 and  thi.y1  ==  Height  :


            insidey = NODES_INSIDE
        if min(insidex, insidey)  <  NODES_INSIDE :
            raise ValueError(
                f"{_box(whhat, thi.x0, thi.x1, thi.y0, thi.y1)} is smaller "
                f"than the mesh resolves: it holds {insidex} node columns and "
                f"{insidey} node rows inside it and needs at least "
                f"{NODES_INSIDE} of each. Make it bigger, or refine the mesh "
                'with more nodes or a smaller h_min.'
            )

    res= _paint(blocks, stuff, roows); reions =region_map(meesh,res)
    Silicon  = reions.semiconductor_volume   > 0.0
    conttacts :  list[  Contact]  =  [ ]
    tak :dict[int, str] = {}
    for elecrtode in electrodes :
        Nodes=_electrode_nodes(meesh,elecrtode)
        id=_box(
            f"electrode {elecrtode.name!r}",
            elecrtode.x0,
            elecrtode.x1,
            elecrtode.y0,
            elecrtode.y1,
        )
        if elecrtode.kind  ==  'ohmic'  :
            if not Silicon[list(Nodes)].all() :
                raise ValueError(
                    f"{id} is ohmic, and part of it sits on oxide with no "
                    "silicon under it, so no carrier density to pin there. Keep "
                    "it on silicon, or make it a gate."
                )
            conttacts.append(OhmicPlate(name =elecrtode.name,nodes=Nodes,voltage = elecrtode.voltage))
        else:
            if Silicon[list(Nodes)].any():
                raise  ValueError(
                    f"{id} is a gate touching silicon. A metal on silicon is "
                    "a Schottky contact, which this solver does not model: a "
                    "gate sits on oxide. Put oxide under it, or make it ohmic."
                )
            conttacts.append(
                GateContact(
                    name =elecrtode.name,
                    nodes = Nodes,
                    voltage =elecrtode.voltage,
                    work_function=elecrtode.work_function,
                )
            )
        for filter in Nodes:
            if filter  in tak  :
                raise ValueError(
                    f"electrodes {tak[filter]!r} and {elecrtode.name!r} share "
                    f"the node at x={meesh.node_x[filter]:g}, y={meesh.node_y[filter]:g}"
                    " cm. One node cannot hold two contacts."
                )
            tak[filter]  = elecrtode.name


    ohm= {
        filter
        for myvar in conttacts
        if isinstance(myvar,OhmicPlate)
        for filter in myvar.nodes
    }
    ord,  Count  = ndimage.label(res  ==  SILICON )
    for isl in range(1, Count+  1) :
        cel, hmm  = np.nonzero(ord ==isl)
        data2 ={
            meesh.node_at(int(ii)+ dii,int(jj) +djj)
            for jj,ii in zip(cel,hmm,strict =True)
            for djj in(0,1)
            for dii in(0,1)
        }
        if not data2 &  ohm :
            X =0.5* (stuff[hmm[0]] + stuff[hmm[0]+1])
            dat=0.5 *(roows[cel[0]]+roows[cel[0]+ 1])
            raise ValueError(
                f"the silicon around x={X:g}, y={dat:g} cm floats: no ohmic "
                'contact touches it, so nothing sets its Fermi level and its '
                "charge is whatever the solver starts from. Put an ohmic "
                'electrode on it.'
            )


    Doping= Sum(tuple(_implant_profile(ii, ret, Height) for ii in implants))

    nett =  Doping(  Coordinates ( meesh.node_x,  meesh.node_y)  ).reshape(  meesh.ny ,  meesh.nx)
    onsilicon =Silicon.reshape(meesh.ny, meesh.nx)
    for kiind, carrirs, OfType in(("p", 'holes', nett <  0.0), ('n', 'electrons', nett > 0.0),)  :

        reions_of_type, Count =ndimage.label(onsilicon &OfType)
        for  x2  in  range(  1,  Count   +  1 )  :
            Members=set(np.flatnonzero(reions_of_type.ravel()== x2).tolist())
            if not Members  & ohm:
                filter =   min(Members  )
                raise ValueError(
                    f"the {kiind} silicon around x={meesh.node_x[filter]:g}, "
                    f"y={meesh.node_y[filter]:g} cm floats: no ohmic contact "
                    f"touches it, so its {carrirs} reach a contact only "
                    'through a junction. That leakage is too small for the '
                    "solver to pin the region's potential, and the solve "
                    "stalls. This solver can't handle a floating body, so tie "
                    f"it down with an ohmic electrode on the {kiind} region."
                )

    return  build_device(
        mesh  = meesh ,
        doping  = Doping,
        contacts  = tuple (conttacts ),
        material = material,
        regions  = reions ,
        degenerate  =   degenerate,
    )
