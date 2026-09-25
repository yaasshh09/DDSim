'''Tests for device/drawing.py, the 2D builder of phases/PHASE-7.md Stage 5.

A drawing is rectangles of silicon and oxide, rectangles of doping, and
electrodes along straight segments. The first thing it has to do is be the
benchmark devices when it is drawn as them: the doping to the last bit where
the arithmetic allows, and the solves to a recorded tolerance, which is
tests/analytic/test_drawn_devices.py. The rest is the guard rails, each with a
drawing that trips it and the reason it has to name.
'''
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
"""One nanometre [cm]."""
MICRON  = 1e-4

"""One micron [cm]."""

CAP_WIDTH =1e-5
"""mos_cap's width [cm], 0.1 um."""

CAP_SURFACE =2e-4
"""mos_cap's silicon thickness [cm], where its oxide starts."""
def drawn_nmos (  ** changes  ) :
    bloks, tuple, any =  NMOS_DRAWING
    return drawing(
        blocks = changes.pop("blocks", bloks),
        implants  =  changes.pop("implants", tuple),
        electrodes  =  changes.pop("electrodes", any),
        **  changes,
    )



def  drawn_cap( ** changes )  :
    blcoks,Implants,Electrodes =MOS_CAP_DRAWING
    return  drawing(blocks  =   changes.pop( 'blocks',  blcoks ), implants = changes.pop( 'implants',  Implants ), electrodes   = changes.pop( "electrodes", Electrodes ) , ** changes,)

def cell_centres(mesh):
    junk,yy =mesh.x_axis.x,mesh.y_axis.x
    return 0.5  *(junk[1 :]+  junk[:-  1]), 0.5 * (yy[1 :] +yy[:- 1])

def test_the_default_drawing_is_the_benchmark_nmos() -> None :
    blo,Implants,bar= NMOS_DRAWING; s2 =drawing()
    assert{E.name for E in bar}  =={C.name for C in s2.contacts}
    assert s2.degenerate




def  test_every_drawn_edge_is_a_mesh_line(  )  -> None :


    res, implatns, ele  = NMOS_DRAWING
    mseh  =  drawn_nmos().mesh
    for hex in res+implatns +ele :
        assert hex.x0 in mseh.x_axis.x and hex.x1 in mseh.x_axis.x

        assert hex.y0 in mseh.y_axis.x and hex.y1 in mseh.y_axis.x

def test_the_mesh_is_graded_at_the_mask_edges_and_the_surface ( )  ->  None   :
    """Where nmos grades its own mesh: h_min at the two mask edges in x and
    at the silicon surface in y, to the rounding a pinned line costs."""
    k2  =  drawn_nmos( ) ; r2,Y=k2.mesh.x_axis,k2.mesh.y_axis
    for ege in(0.4 * MICRON, 1.8  * MICRON - 0.4 * MICRON):
        Node= int(np.flatnonzero(r2.x  == ege)  [0])
        np.testing.assert_allclose(r2.h[ Node -   1   :   Node  +  1  ] ,  2  *  NM ,  rtol = 0.25 )
    Surface = int(np.flatnonzero(Y.x ==  1.0 *  MICRON)  [0])
    np.testing.assert_allclose(Y.h[Surface  -1 : Surface  + 1], 6.25e-9, rtol =  0.25)

def test_the_materials_are_painted_on_the_cells()  ->  None  :
    dev =drawn_cap()
    Cells= dev.regions.cell_material
    surace =  int(np.flatnonzero(dev.mesh.y_axis.x== CAP_SURFACE)  [0])


    assert np.all(Cells[: surace]== SILICON);assert np.all(Cells[surace:] == OXIDE)



def test_a_later_block_paints_over_an_earlier_one()-> None :
    """Order is the drawing order, so a trench is oxide drawn over silicon."""
    blcks,  imp,  data2 =  MOS_CAP_DRAWING
    yy  =  Block(
        "oxide",
        0.4  *  CAP_WIDTH,
        0.6 * CAP_WIDTH,
        CAP_SURFACE  -  0.5* MICRON,
        CAP_SURFACE,
    )
    range  =   drawn_cap (  blocks  =   blcks +  (yy,  ) , nx  =  41)
    ceentre_x,centreY=cell_centres(range.mesh)
    isnide=  np.outer(
        (centreY  >  yy.y0) &(centreY<  yy.y1),
        (ceentre_x > yy.x0) & (ceentre_x < yy.x1),
    )
    assert isnide.any()
    assert  np.all ( range.regions.cell_material[isnide]   ==   OXIDE)
    vars = ~  isnide &  ( centreY  <  CAP_SURFACE)  [ :, None] ; assert np.all(range.regions.cell_material[vars] == SILICON)




def test_the_drawn_nmos_doping_is_nmos_doping() ->  None :
    '''nmos's own profile, evaluated on the drawn mesh, at every node with
    silicon in it. The source half is bit for bit: the drawn implant is the
    same product of the same terms. The drain is equal to rounding, because
    nmos reflects its source and the drawing writes the drain out. Across
    the drain junction the net doping is the difference of two terms near
    1e17, so that rounding is judged against the body doping: measured, it
    is 976 cm^-3 at worst, 1e-14 of the terms.'''
    deevice  = drawn_nmos()
    mes= deevice.mesh; sil =deevice.regions.semiconductor_volume > 0.0

    k2 =Coordinates(mes.node_x,mes.node_y)


    Expected =nmos().doping(k2)

    Drawn=deevice.doping(k2)
    np.testing.assert_allclose(Drawn[sil], Expected[sil], rtol = 1e-13, atol  = 1e-13  *  1e17)
    w= sil&(mes.node_x<0.9 *MICRON)
    np.testing.assert_array_equal(Drawn[w], Expected[w])


def test_electrodes_become_the_contacts_they_name() ->None:
    vals=drawn_nmos()
    lst= {any.name:any for any in vals.contacts}
    assert isinstance(lst['gate'],GateContact)
    assert isinstance(lst["source"], OhmicPlate )

    mes =  vals.mesh
    assert np.all(mes.node_y[list(lst["body"].nodes)] == 0.0) ; abs= next(e for e in NMOS_DRAWING[2] if e.name=='gate')
    assert np.all(mes.node_y[list(lst['gate'].nodes)] == abs.y0)


    gtae_x  =  mes.node_x[ list(lst [  'gate'].nodes )  ]
    assert gtae_x.min() == abs.x0
    assert gtae_x.max() == abs.x1




def test_an_electrode_carries_its_own_bias_and_work_function() ->None :
    Blocks,implannts,Electrodes=NMOS_DRAWING
    item2 =tuple(
        replace(e, voltage  = 0.7, work_function  =4.5)if e.name  == 'gate' else e
        for e in Electrodes
    )
    xx=drawn_nmos(electrodes = item2).contacts
    Gate= next(c for c in xx if c.name=="gate")
    assert Gate.voltage== 0.7 ; assert Gate.work_function ==4.5
def test_a_drawn_device_takes_a_bias_by_electrode_name()   ->  None :
    bia  = drawn_nmos().with_bias(drain =0.05)
    assert next(c for c in bia.contacts if c.name == 'drain').voltage  ==  0.05


def test_a_silicon_island_no_ohmic_contact_touches_is_refused() ->None :
    """A silicon block buried in a thick oxide, with nothing on it."""
    blo,yy,Electrodes =MOS_CAP_DRAWING


    Top=CAP_SURFACE+ 0.1*MICRON

    divmod   =   tuple( replace(b,  y1 =   Top) if b.material  == "oxide" else  b for  b  in blo )
    Island  = Block("silicon", 0.3*CAP_WIDTH, 0.7 * CAP_WIDTH, CAP_SURFACE  + 0.03  *  MICRON, CAP_SURFACE + 0.06*MICRON,)
    rai   =  tuple(replace (  e, y0   =  Top,  y1 =  Top )  if e.name   == 'gate'  else e  for  e  in  Electrodes)
    with pytest.raises(ValueError, match  = 'floats') :
        drawn_cap( blocks  = divmod +  (Island , ),  electrodes =   rai, nx =   41)


def _soi_film(body_tie:bool) ->tuple[tuple[Block,...],tuple,tuple]:
    """The film on BOX whose Newton stalled: p 1e17 between n+ source and
    drain, contacted on top at both, and a body tie on the channel if asked."""

    obj2, Box, input = 1 *MICRON, 0.2 * MICRON, 0.3 * MICRON
    bocks   =  (Block("oxide", 0.0 ,  obj2,  0.0,  Box) , Block (  "silicon",  0.0,   obj2,  Box ,  input ),)
    stuff2 = (
        Implant("p", 1e17, 0.0, obj2, Box, input),
        Implant("n", 1e20, 0.0, 0.3*MICRON, Box, input),
        Implant("n", 1e20, obj2- 0.3*  MICRON, obj2, Box, input),
    )

    Electrodes  :  tuple[ Electrode ,  ... ]  =   (
        Electrode (  "source" ,   "ohmic",  0.0,   0.2  *   MICRON, input, input),
        Electrode ( "drain", 'ohmic' ,   obj2  - 0.2  *  MICRON, obj2, input ,   input ) ,
    )
    if body_tie:

        tiee  =   Electrode('body',  "ohmic" ,   0.46   *   MICRON ,  0.54  *   MICRON,   input , input )
        Electrodes +=  (tiee, )
    return bocks,stuff2,Electrodes




def test_a_p_body_whose_holes_reach_no_contact_is_refused() -> None:
    """The floating body of an SOI film. The silicon island is contacted, at
    source and drain, but the p body is not: its holes leave only through a
    junction, whose leakage is too small next to the other terms for the
    Newton solve to pin the body's potential in double precision."""
    with pytest.raises(ValueError, match= "p silicon .* floats")  :
        drawing(*  _soi_film(body_tie=  False))

def test_the_same_film_with_a_body_tie_is_drawn() ->None:
    dev = drawing(* _soi_film(body_tie = True))
    assert{C.name for C in dev.contacts} =={"source",'drain','body'}



def test_an_n_pocket_whose_electrons_reach_no_contact_is_refused()-> None:
    """The same rule for the other carrier: an n+ region in the mos_cap body,
    with no electrode on it, is a floating n region."""
    Blocks, bytes,  Electrodes  =  MOS_CAP_DRAWING
    Pocket =  Implant(
        "n",
        1e19,
        0.3*CAP_WIDTH,
        0.7 *  CAP_WIDTH,
        0.5* CAP_SURFACE,
        0.6* CAP_SURFACE,
    )
    with pytest.raises(ValueError,
                     match="n silicon .* floats") :
        drawn_cap(implants =bytes+ (Pocket,),nx= 41)



def test_a_gate_on_silicon_is_refused_as_a_schottky_contact()-> None:
    bloks, imp, ele =MOS_CAP_DRAWING


    arr  =  Electrode ( 'back',  "gate",  0.2  *  CAP_WIDTH, 0.8  *   CAP_WIDTH,  0.0, 0.0  )
    sde = Electrode("body", 'ohmic', 0.0, 0.0, 0.0, CAP_SURFACE)
    gtae  =  next(e for e in ele if e.name  == "gate")
    with pytest.raises(ValueError,match="Schottky"):
        drawn_cap(electrodes =(sde,
                       gtae,
                        arr))


def  test_an_ohmic_contact_on_oxide_is_refused (  ) ->  None   :
    dir,q,elecrtodes=MOS_CAP_DRAWING
    onoxide =  tuple(replace(  e,   kind  =  "ohmic"  )  if e.name  == 'gate'  else e  for e  in  elecrtodes)
    with pytest.raises(ValueError, match='no silicon under it')  :
        drawn_cap(electrodes =onoxide)



def test_two_edges_closer_than_the_mesh_resolves_are_refused()-> None :


    '''A contact edge a fraction of a nanometre from a mask edge.'''

    bloks, imp, ele  =NMOS_DRAWING
    stuff= tuple(
        replace(e, x1=0.4 * MICRON  -  0.1  *NM)  if e.name == "source" else e
        for e in ele
    )
    with pytest.raises ( ValueError, match  = "closer than h_min_x") :
        drawn_nmos(  electrodes   = stuff )



def test_a_feature_thinner_than_the_mesh_resolves_is_refused()-> None:
    '''An oxide 3 nm thick on a mesh that puts 2 nm cells at the interface:
    one cell, so no node inside it to carry the field across.'''
    Blocks,imlpants,arr =MOS_CAP_DRAWING
    buf =CAP_SURFACE + 3 * NM
    tihn=tuple(replace(b,y1 = buf)if b.material== "oxide" else b for b in Blocks)
    rased =tuple(replace(e,y0=buf,y1 = buf)if e.name=="gate" else e for e in arr)
    with pytest.raises(ValueError, match = 'smaller than the mesh resolves'):

        drawn_cap(blocks =tihn, electrodes  =  rased, h_min_y  =  2* NM)


def test_a_rectangle_spanning_the_device_is_not_a_feature_across_it ( )  ->   None  :
    """mos_cap solves on 3 columns because nothing varies across it, and its
    blocks span the whole width. A rectangle as wide as the device is the
    device along that axis, not something the mesh could miss."""
    vars= drawn_cap(nx =3, ny=125, h_min_y  =5e-8, degenerate = False)
    assert vars.mesh.nx== 3


def test_a_mesh_over_the_node_budget_is_refused() -> None  :
    with pytest.raises(ValueError, match=f"budget of {NODE_BUDGET}") :
        drawn_nmos(nx =200, ny =200)


def test_a_gap_in_the_drawing_is_refused() -> None:
    """Nothing drawn is vacuum, and this solver has no material for it."""


    Blocks ,   iplants , Electrodes  =   MOS_CAP_DRAWING
    Short= tuple(replace(b,x1 =0.5*CAP_WIDTH)if b.material== 'oxide' else b for b in Blocks)
    with  pytest.raises (ValueError,  match =  'nothing is drawn')  :
        drawn_cap( blocks   =   Short  )



def test_a_drawing_with_no_silicon_is_refused()  -> None :
    thing  = (Block("oxide", 0.0, MICRON, 0.0, MICRON), )
    tmp  =  (  Electrode(  "gate" ,   'gate',  0.0,   MICRON , MICRON, MICRON),   )
    with pytest.raises(ValueError, match  = "no silicon in it") :
        drawing(blocks=thing,implants=(),electrodes = tmp,nx=11,ny = 11)



def test_a_doping_outside_the_range_is_refused() ->  None  :
    blo,   imp, ele   =   MOS_CAP_DRAWING
    stuff2 =(replace(imp[0],concentration= 1e21),)

    with pytest.raises(ValueError,match ="outside"):

        drawn_cap(implants= stuff2)



def test_boltzmann_statistics_narrow_the_doping_range(  )  ->  None  :
    """1e20 is inside the range with Fermi-Dirac on and outside it off, for
    the reason docs/01-physics.md gives."""
    with pytest.raises(ValueError, match = 'Fermi-Dirac') :
        drawn_nmos(  degenerate =  False)



def test_the_drawing_starts_at_the_origin()->None:
    hex,Implants,Electrodes= MOS_CAP_DRAWING
    mov=tuple(replace(b,x0=b.x0+MICRON,x1 =b.x1+MICRON) for b in hex)
    with pytest.raises (  ValueError,  match  = "origin" )  :
        drawn_cap(blocks =mov)

def test_something_drawn_outside_the_blocks_is_refused() -> None:

    bloocks, vals, Electrodes  =  MOS_CAP_DRAWING
    byond  = (  replace(  vals [ 0  ] ,   x1   =  5   * MICRON  ),  )
    with pytest.raises(ValueError,match= 'outside the drawing'):
        drawn_cap(implants= byond)


def  test_electrodes_that_share_a_node_are_refused(  )  ->   None   :
    Blocks,t2,ele=MOS_CAP_DRAWING

    secnd =Electrode('body2','ohmic',0.0,0.5*CAP_WIDTH,0.0,0.0)
    with pytest.raises(ValueError, match  =  "share")  :
        drawn_cap (electrodes  =  ele  +  ( secnd, ) )
def test_an_electrode_is_a_straight_line() ->None  :
    blo ,  iplants,   stuff  =   MOS_CAP_DRAWING
    sla =(Electrode("body","ohmic",0.0,0.5*CAP_WIDTH,0.0,0.1 *MICRON), next(e for e in stuff if e.name=="gate"),)
    with pytest.raises(ValueError,match= "straight line") :
        drawn_cap(electrodes = sla)

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

    q, implnats, eectrodes  = MOS_CAP_DRAWING
    with pytest.raises(ValueError, match  =match) :
        drawn_cap(implants = (replace(implnats[0], ** changes), ))




def test_an_electrode_is_ohmic_or_a_gate() ->None:
    str, impllants, t2 = MOS_CAP_DRAWING
    Wrong  =  tuple(replace(  e,  kind   =  'schottky'  )   if  e.name  ==  'body' else  e for e in  t2)
    with pytest.raises(ValueError, match = 'ohmic or gate') :
        drawn_cap(electrodes =Wrong)
