from __future__ import annotations
from  dataclasses import replace
import numpy as np, pytest
from ddsim.device.doping import Coordinates
from ddsim.device.drawing import(
    MOS_CAP_DRAWING,
    NMOS_DRAWING,
    NODE_BUDGET,
    Block,
    Electrode,
    Implant,
    drawing,
)
from ddsim.device.mosfet import nmos
from ddsim.device.regions import OXIDE,SILICON
from  ddsim.discretize.boundary import  GateContact,  OhmicPlate
NM  = 1e-7
MICRON  = 1e-4

CAP_WIDTH =1e-5

CAP_SURFACE =2e-4
def drawn_nmos (  ** changes  ) :
    t, f, prev =  NMOS_DRAWING
    return drawing(
        blocks = changes.pop("blocks", t),
        implants  =  changes.pop("implants", f),
        electrodes  =  changes.pop("electrodes", prev),
        **  changes,
    )



def  drawn_cap( ** changes )  :
    flag,num,e =MOS_CAP_DRAWING
    return  drawing(blocks  =   changes.pop( 'blocks',  flag ), implants = changes.pop( 'implants',  num ), electrodes   = changes.pop( "electrodes", e ) , ** changes,)

def cell_centres(mesh):
    y,d =mesh.x_axis.x,mesh.y_axis.x
    return 0.5  *(y[1 :]+  y[:-  1]), 0.5 * (d[1 :] +d[:- 1])

def test_the_default_drawing_is_the_benchmark_nmos() -> None :
    s,k,f= NMOS_DRAWING; jj =drawing()
    assert{stuff.name for stuff in f}  =={rows.name for rows in jj.contacts}
    assert jj.degenerate




def  test_every_drawn_edge_is_a_mesh_line(  )  -> None :


    foo, item, zz  = NMOS_DRAWING
    nxt  =  drawn_nmos().mesh
    for a2 in foo+item +zz :
        assert a2.x0 in nxt.x_axis.x and a2.x1 in nxt.x_axis.x

        assert a2.y0 in nxt.y_axis.x and a2.y1 in nxt.y_axis.x

def test_the_mesh_is_graded_at_the_mask_edges_and_the_surface ( )  ->  None   :
    d  =  drawn_nmos( ) ; u,el=d.mesh.x_axis,d.mesh.y_axis
    for row in(0.4 * MICRON, 1.8  * MICRON - 0.4 * MICRON):
        m= int(np.flatnonzero(u.x  == row)  [0])
        np.testing.assert_allclose(u.h[ m -   1   :   m  +  1  ] ,  2  *  NM ,  rtol = 0.25 )
    w = int(np.flatnonzero(el.x ==  1.0 *  MICRON)  [0])
    np.testing.assert_allclose(el.h[w  -1 : w  + 1], 6.25e-9, rtol =  0.25)

def test_the_materials_are_painted_on_the_cells()  ->  None  :
    tmp2 =drawn_cap()
    zz= tmp2.regions.cell_material
    t2 =  int(np.flatnonzero(tmp2.mesh.y_axis.x== CAP_SURFACE)  [0])


    assert np.all(zz[: t2]== SILICON);assert np.all(zz[t2:] == OXIDE)



def test_a_later_block_paints_over_an_earlier_one()-> None :
    obj,  tmp,  s2 =  MOS_CAP_DRAWING
    s  =  Block(
        "oxide",
        0.4  *  CAP_WIDTH,
        0.6 * CAP_WIDTH,
        CAP_SURFACE  -  0.5* MICRON,
        CAP_SURFACE,
    )
    f  =   drawn_cap (  blocks  =   obj +  (s,  ) , nx  =  41)
    k,t2=cell_centres(f.mesh)
    mm=  np.outer(
        (t2  >  s.y0) &(t2<  s.y1),
        (k > s.x0) & (k < s.x1),
    )
    assert mm.any()
    assert  np.all ( f.regions.cell_material[mm]   ==   OXIDE)
    u = ~  mm &  ( t2  <  CAP_SURFACE)  [ :, None] ; assert np.all(f.regions.cell_material[u] == SILICON)




def test_the_drawn_nmos_doping_is_nmos_doping() ->  None :
    z  = drawn_nmos()
    d= z.mesh; j =z.regions.semiconductor_volume > 0.0

    c =Coordinates(d.node_x,d.node_y)


    cur =nmos().doping(c)

    tt=z.doping(c)
    np.testing.assert_allclose(tt[j], cur[j], rtol = 1e-13, atol  = 1e-13  *  1e17)
    u= j&(d.node_x<0.9 *MICRON)
    np.testing.assert_array_equal(tt[u], cur[u])


def test_electrodes_become_the_contacts_they_name() ->None:
    thing=drawn_nmos()
    h= {stuff.name:stuff for stuff in thing.contacts}
    assert isinstance(h['gate'],GateContact)
    assert isinstance(h["source"], OhmicPlate )

    d =  thing.mesh
    assert np.all(d.node_y[list(h["body"].nodes)] == 0.0) ; v= next(c for c in NMOS_DRAWING[2] if c.name=='gate')
    assert np.all(d.node_y[list(h['gate'].nodes)] == v.y0)


    x  =  d.node_x[ list(h [  'gate'].nodes )  ]
    assert x.min() == v.x0
    assert x.max() == v.x1




def test_an_electrode_carries_its_own_bias_and_work_function() ->None :
    d,i,g=NMOS_DRAWING
    arr =tuple(
        replace(bar, voltage  = 0.7, work_function  =4.5)if bar.name  == 'gate' else bar
        for bar in g
    )
    aa=drawn_nmos(electrodes = arr).contacts
    t= next(a2 for a2 in aa if a2.name=="gate")
    assert t.voltage== 0.7 ; assert t.work_function ==4.5
def test_a_drawn_device_takes_a_bias_by_electrode_name()   ->  None :
    flag  = drawn_nmos().with_bias(drain =0.05)
    assert next(j for j in flag.contacts if j.name == 'drain').voltage  ==  0.05


def test_a_silicon_island_no_ohmic_contact_touches_is_refused() ->None :
    r2,c,k =MOS_CAP_DRAWING


    t2=CAP_SURFACE+ 0.1*MICRON

    cur   =   tuple( replace(res2,  y1 =   t2) if res2.material  == "oxide" else  res2 for  res2  in r2 )
    zz  = Block("silicon", 0.3*CAP_WIDTH, 0.7 * CAP_WIDTH, CAP_SURFACE  + 0.03  *  MICRON, CAP_SURFACE + 0.06*MICRON,)
    t   =  tuple(replace (  a, y0   =  t2,  y1 =  t2 )  if a.name   == 'gate'  else a  for  a  in  k)
    with pytest.raises(ValueError, match  = 'floats') :
        drawn_cap( blocks  = cur +  (zz , ),  electrodes =   t, nx =   41)


def _soi_film(body_tie:bool) ->tuple[tuple[Block,...],tuple,tuple]:

    ii, k, ys = 1 *MICRON, 0.2 * MICRON, 0.3 * MICRON
    d2   =  (Block("oxide", 0.0 ,  ii,  0.0,  k) , Block (  "silicon",  0.0,   ii,  k ,  ys ),)
    d = (
        Implant("p", 1e17, 0.0, ii, k, ys),
        Implant("n", 1e20, 0.0, 0.3*MICRON, k, ys),
        Implant("n", 1e20, ii- 0.3*  MICRON, ii, k, ys),
    )

    z  :  tuple[ Electrode ,  ... ]  =   (
        Electrode (  "source" ,   "ohmic",  0.0,   0.2  *   MICRON, ys, ys),
        Electrode ( "drain", 'ohmic' ,   ii  - 0.2  *  MICRON, ii, ys ,   ys ) ,
    )
    if body_tie:

        v  =   Electrode('body',  "ohmic" ,   0.46   *   MICRON ,  0.54  *   MICRON,   ys , ys )
        z +=  (v, )
    return d2,d,z




def test_a_p_body_whose_holes_reach_no_contact_is_refused() -> None:
    with pytest.raises(ValueError, match= "p silicon .* floats")  :
        drawing(*  _soi_film(body_tie=  False))

def test_the_same_film_with_a_body_tie_is_drawn() ->None:
    k = drawing(* _soi_film(body_tie = True))
    assert{thing.name for thing in k.contacts} =={"source",'drain','body'}



def test_an_n_pocket_whose_electrons_reach_no_contact_is_refused()-> None:
    ok, ss,  nxt  =  MOS_CAP_DRAWING
    g =  Implant(
        "n",
        1e19,
        0.3*CAP_WIDTH,
        0.7 *  CAP_WIDTH,
        0.5* CAP_SURFACE,
        0.6* CAP_SURFACE,
    )
    with pytest.raises(ValueError,
                     match="n silicon .* floats") :
        drawn_cap(implants =ss+ (g,),nx= 41)



def test_a_gate_on_silicon_is_refused_as_a_schottky_contact()-> None:
    g, c, w2 =MOS_CAP_DRAWING


    vals  =  Electrode ( 'back',  "gate",  0.2  *  CAP_WIDTH, 0.8  *   CAP_WIDTH,  0.0, 0.0  )
    ii = Electrode("body", 'ohmic', 0.0, 0.0, 0.0, CAP_SURFACE)
    b2  =  next(b for b in w2 if b.name  == "gate")
    with pytest.raises(ValueError,match="Schottky"):
        drawn_cap(electrodes =(ii,
                       b2,
                        vals))


def  test_an_ohmic_contact_on_oxide_is_refused (  ) ->  None   :
    b2,bb,x=MOS_CAP_DRAWING
    t =  tuple(replace(  s2,   kind  =  "ohmic"  )  if s2.name  == 'gate'  else s2  for s2  in  x)
    with pytest.raises(ValueError, match='no silicon under it')  :
        drawn_cap(electrodes =t)



def test_two_edges_closer_than_the_mesh_resolves_are_refused()-> None :


    cc, m, s  =NMOS_DRAWING
    x= tuple(
        replace(z2, x1=0.4 * MICRON  -  0.1  *NM)  if z2.name == "source" else z2
        for z2 in s
    )
    with pytest.raises ( ValueError, match  = "closer than h_min_x") :
        drawn_nmos(  electrodes   = x )



def test_a_feature_thinner_than_the_mesh_resolves_is_refused()-> None:
    j,w2,info =MOS_CAP_DRAWING
    m =CAP_SURFACE + 3 * NM
    g=tuple(replace(rows,y1 = m)if rows.material== "oxide" else rows for rows in j)
    x =tuple(replace(dat,y0=m,y1 = m)if dat.name=="gate" else dat for dat in info)
    with pytest.raises(ValueError, match = 'smaller than the mesh resolves'):

        drawn_cap(blocks =g, electrodes  =  x, h_min_y  =  2* NM)


def test_a_rectangle_spanning_the_device_is_not_a_feature_across_it ( )  ->   None  :
    cc= drawn_cap(nx =3, ny=125, h_min_y  =5e-8, degenerate = False)
    assert cc.mesh.nx== 3


def test_a_mesh_over_the_node_budget_is_refused() -> None  :
    with pytest.raises(ValueError, match=f"budget of {NODE_BUDGET}") :
        drawn_nmos(nx =200, ny =200)


def test_a_gap_in_the_drawing_is_refused() -> None:


    j ,   a , res  =   MOS_CAP_DRAWING
    rows= tuple(replace(w,x1 =0.5*CAP_WIDTH)if w.material== 'oxide' else w for w in j)
    with  pytest.raises (ValueError,  match =  'nothing is drawn')  :
        drawn_cap( blocks   =   rows  )



def test_a_drawing_with_no_silicon_is_refused()  -> None :
    item  = (Block("oxide", 0.0, MICRON, 0.0, MICRON), )
    t2  =  (  Electrode(  "gate" ,   'gate',  0.0,   MICRON , MICRON, MICRON),   )
    with pytest.raises(ValueError, match  = "no silicon in it") :
        drawing(blocks=item,implants=(),electrodes = t2,nx=11,ny = 11)



def test_a_doping_outside_the_range_is_refused() ->  None  :
    m,   val, d   =   MOS_CAP_DRAWING
    res2 =(replace(val[0],concentration= 1e21),)

    with pytest.raises(ValueError,match ="outside"):

        drawn_cap(implants= res2)



def test_boltzmann_statistics_narrow_the_doping_range(  )  ->  None  :
    with pytest.raises(ValueError, match = 'Fermi-Dirac') :
        drawn_nmos(  degenerate =  False)



def test_the_drawing_starts_at_the_origin()->None:
    tt,zz,h= MOS_CAP_DRAWING
    s=tuple(replace(z,x0=z.x0+MICRON,x1 =z.x1+MICRON) for z in tt)
    with pytest.raises (  ValueError,  match  = "origin" )  :
        drawn_cap(blocks =s)

def test_something_drawn_outside_the_blocks_is_refused() -> None:

    x, res2, y  =  MOS_CAP_DRAWING
    out  = (  replace(  res2 [ 0  ] ,   x1   =  5   * MICRON  ),  )
    with pytest.raises(ValueError,match= 'outside the drawing'):
        drawn_cap(implants= out)


def  test_electrodes_that_share_a_node_are_refused(  )  ->   None   :
    flag,u,z=MOS_CAP_DRAWING

    c =Electrode('body2','ohmic',0.0,0.5*CAP_WIDTH,0.0,0.0)
    with pytest.raises(ValueError, match  =  "share")  :
        drawn_cap (electrodes  =  z  +  ( c, ) )
def test_an_electrode_is_a_straight_line() ->None  :
    z ,  rr,   t2  =   MOS_CAP_DRAWING
    a =(Electrode("body","ohmic",0.0,0.5*CAP_WIDTH,0.0,0.1 *MICRON), next(x for x in t2 if x.name=="gate"),)
    with pytest.raises(ValueError,match= "straight line") :
        drawn_cap(electrodes = a)

@pytest.mark.parametrize(
    ("record","match"),
    [
        (Block("glass",0.0,MICRON,0.0,MICRON),"silicon or oxide"),
        (Block('silicon',MICRON,0.0,0.0,MICRON),"positive"),
    ],
)

def test_a_block_names_what_is_wrong_with_it(record,match)->None :
    with pytest.raises(ValueError,
               match = match):
        drawing( blocks   =  (record,   ),   implants  =  (  ),  electrodes  =  () ,   nx  = 11,  ny   =  11)




@pytest.mark.parametrize(
    ('changes','match'),
    [
        ({"dopant" :"x"},"'n' or 'p'"),
        ({"profile" :'linear'},"uniform or gaussian"),
        ({"profile" :"gaussian"},"straggle and lateral"),
    ],
)

def test_an_implant_names_what_is_wrong_with_it(changes,match)-> None :

    r, item, e  = MOS_CAP_DRAWING
    with pytest.raises(ValueError, match  =match) :
        drawn_cap(implants = (replace(item[0], ** changes), ))




def test_an_electrode_is_ohmic_or_a_gate() ->None:
    w, m, jj = MOS_CAP_DRAWING
    y  =  tuple(replace(  stuff,  kind   =  'schottky'  )   if  stuff.name  ==  'body' else  stuff for stuff in  jj)
    with pytest.raises(ValueError, match = 'ohmic or gate') :
        drawn_cap(electrodes =y)
