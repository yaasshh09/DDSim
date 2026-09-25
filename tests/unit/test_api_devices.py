from __future__ import annotations

import pytest
from ddsim.api.devices import(
    COARSE,
    DEVICE_KINDS,
    build_from_spec,
    device_dimension,
    device_parameters,
    drawing_defaults,
    node_count,
    parameters_of,
    region_defaults,
)



from ddsim.api.sweeps import run_sweep

from ddsim.device.drawing import NODE_BUDGET
def test_the_device_classes_are_offered() -> None  :
    assert set(DEVICE_KINDS) == {'pn_diode', "mos_cap", "nmos", "stack", "drawing"}



def  test_the_parameters_come_from_the_constructor_signature ()  ->   None  :
    res  = {parametter.name for parametter in device_parameters("pn_diode")}

    assert 'Na' in res
    assert "junction" in res
    assert 'anode_voltage'  in res




def test_a_parameter_carries_its_default_and_its_type()->None :
    byName  =  {p.name :p for p in device_parameters('pn_diode')}

    assert byName["Na"].default == 1e16
    assert byName[ "Na"  ].type == "float"
    assert  byName[ "n_nodes" ].default  ==  201

    assert byName["n_nodes"].type ==  "int"


def test_a_boolean_parameter_is_offered_as_a_boolean()-> None :
    blah  ={p.name :p for p in device_parameters("nmos")}



    assert blah['degenerate'].type == "bool"
    assert  blah[  "degenerate" ].default is  True

def test_the_material_argument_is_not_offered() ->None :
    for Kind in DEVICE_KINDS:

        assert "material" not in{p.name for p in device_parameters(Kind)}


def test_an_unknown_kind_names_the_ones_that_exist() ->  None:

    with pytest.raises(ValueError,match ="pn_diode"):
        device_parameters('transistor')


def  test_building_with_no_parameters_gives_the_constructor_default(  )  ->  None   :

    zz =build_from_spec('pn_diode',{})


    assert zz.mesh.n_nodes==201




def test_a_parameter_reaches_the_constructor() -> None  :
    Device=  build_from_spec("pn_diode",
         {'n_nodes' : 51})

    assert Device.mesh.n_nodes==51



def test_an_unknown_parameter_is_refused_rather_than_ignored() ->None:
    with pytest.raises(ValueError,match ='n_node'):
        build_from_spec("pn_diode",{'n_node' :51})
def test_a_parameter_of_the_wrong_type_is_refused()  -> None :

    with pytest.raises(TypeError,match="Na"):
        build_from_spec("pn_diode",{"Na" :"1e16"})

def test_a_whole_number_is_accepted_where_a_float_is_wanted() ->  None:
    w  =build_from_spec('pn_diode', {'anode_voltage' : 1})

    assert w.contacts[0].voltage ==  pytest.approx(1.0)

def test_a_boolean_is_refused_where_an_integer_is_wanted()->None :
    with pytest.raises(TypeError, match= 'n_nodes') :
        build_from_spec('pn_diode',{"n_nodes":True})
def test_an_integer_is_refused_where_a_boolean_is_wanted() ->   None  :

    with pytest.raises(TypeError, match  ="degenerate") :
        build_from_spec( "nmos",  {"degenerate"  :  1 }  )



def  test_the_material_argument_cannot_be_passed_either( )   ->  None   :
    with  pytest.raises (  ValueError ,   match  = "material" ) :

        build_from_spec("pn_diode", {"material"  :  'silicon'})


def test_a_boolean_reaches_the_constructor()-> None:
    max=build_from_spec("nmos",{"degenerate":False})

    assert max.degeneracy  is  None
def test_a_float_reaches_the_constructor_unchanged() -> None:
    dev   =  build_from_spec (  'pn_diode', {'Na'   :  2.5e16  })
    assert dev.net_doping.data[0] == pytest.approx(- 2.5e16)

def test_an_argument_with_no_default_is_not_a_knob(  ) ->  None   :

    def example(required:float,optional : float =1.0,named: str = "x")->None:
        ...

    assert[p.name  for p in parameters_of(  example)]   ==   [  "optional" ,  "named"]




def test_a_coarse_preset_exists_for_every_device_not_solved_live()-> None:


    Live ={kiind for kiind in DEVICE_KINDS if device_dimension(kiind)  == 1}

    assert set(COARSE  )  ==   set( DEVICE_KINDS  )  -   Live

@pytest.mark.parametrize(  "kind" , sorted(COARSE  ))



def test_a_coarse_preset_only_names_knobs_its_device_has ( kind)  ->  None  :
    knwon  = {parrameter.name for parrameter in device_parameters(kind)}

    assert set(COARSE[kind].parameters)  <= knwon

@pytest.mark.parametrize("kind", sorted(COARSE))



def test_a_coarse_preset_builds_and_is_coarser(kind) -> None:
    coa =   build_from_spec (kind, dict(  COARSE[ kind  ].parameters ) )
    sum = build_from_spec(kind, {})


    assert  node_count( coa)   <  node_count(sum  )



@pytest.mark.parametrize("kind",sorted(COARSE))


def test_a_coarse_preset_says_what_changes(kind) ->None :
    Note  =  COARSE[  kind  ].note
    assert 'percent'  in Note , f"{kind}: {Note!r} names no measured difference"
    assert str (node_count (build_from_spec( kind,   {  }  )  ) )   in  Note, (
        f"{kind}: {Note!r} does not say what it is coarse against"
    )
PIN=[{'dopant': "p","length" :2e-5,'concentration':1e18}, {"dopant":'n',"length":1e-4,"concentration":1e14}, {'dopant' : "n",'length': 2e-5,'concentration':1e18},]


def test_a_stack_is_built_from_the_regions_the_page_sends( )  ->   None  :
    dev  = build_from_spec("stack", {"regions" :PIN, 'n_nodes'  : 301})
    assert dev.mesh.n_nodes ==301
    assert dev.mesh.x[- 1]  == pytest.approx(1.4e-4, rel = 1e-14)




def test_the_default_regions_are_offered_as_the_page_sends_them()->None:

    thing  =region_defaults("stack")
    assert thing==[{'dopant': 'p', "length"  : 5e-5, "concentration": 1e16}, {"dopant" : 'n', 'length' : 5e-5, 'concentration' : 1e16},]
    assert(build_from_spec( "stack" ,  { "regions" : thing } ).mesh.x ==  build_from_spec( "stack",   {  } ).mesh.x).all()

def test_a_device_without_regions_offers_none()->None:
    assert region_defaults (  "pn_diode" )   is  None

def test_regions_are_not_a_knob_on_the_form() ->None  :
    assert "regions" not in{p.name for p in device_parameters("stack")}



def  test_the_stack_is_solved_live(  )   ->   None   :
    assert device_dimension('stack')== 1




@pytest.mark.parametrize(
    ("regions","complaint"),
    [
        ("pn",'list of regions'),
        (["p"],'region 1 is'),
        ([{'dopant' :"p",'length': 1e-4}],'region 1.*concentration'),
        (
            [{"dopant":"p","length" : 1e-4,'concentration': 1e16,'x': 1}],
            "region 1.*'x'",
        ),
        ([{'dopant':"p",'length' :"1e-4",'concentration' : 1e16}],'length'),
        ([{"dopant":'p',"length" : 1e-4,"concentration":True}],'concentration'),
        ([{'dopant': 1,"length":1e-4,'concentration':1e16}],'dopant'),
    ],
    ids = ["not a list",'not an object',"missing","extra",'string','bool','number'],
)


def test_a_malformed_region_is_refused_naming_it(regions, complaint)-> None :

    with pytest.raises((TypeError, ValueError), match  =  complaint) :

        build_from_spec('stack', {'regions' :regions})
def test_regions_are_refused_on_a_device_that_has_none()->None:
    with pytest.raises(ValueError,match='regions') :
        build_from_spec("pn_diode", {'regions' : PIN})




def  test_the_phase_2_diode_drawn_as_a_stack_has_the_same_i_v(  )   ->  None  :
    Voltages = [0.0, 0.2, 0.4, -0.5]
    dio, _ = run_sweep('iv', build_from_spec("pn_diode", {}), 'anode', Voltages)
    Drawn,   _ =  run_sweep (
        "iv",
        build_from_spec( "stack",   { "regions"   :  region_defaults( "stack")}),
        "left",
        Voltages,
    )
    assert dio.complete and Drawn.complete
    assert[p.current for p in Drawn.points] ==[p.current for p in dio.points]

def test_a_drawing_sent_as_its_defaults_is_the_default_drawing () ->  None  :
    import numpy as np

    from ddsim.device.drawing import drawing


    Sent = build_from_spec("drawing", drawing_defaults("drawing"))
    bulit =  drawing()
    np.testing.assert_array_equal (  Sent.mesh.node_x, bulit.mesh.node_x)
    np.testing.assert_array_equal(Sent.mesh.node_y, bulit.mesh.node_y); np.testing.assert_array_equal(Sent.net_doping.data, bulit.net_doping.data)
    assert[C.nodes for C in Sent.contacts]==[C.nodes for C in bulit.contacts]



def test_only_a_drawn_device_has_drawing_defaults() -> None:
    par  =  drawing_defaults(  "drawing" )
    assert set(par) =={"blocks","implants","electrodes"}
    assert{abs["name"]for abs in par["electrodes"]}=={
        'source',
        'drain',
        "gate",
        "body",
    }
    assert drawing_defaults('nmos')is None




def test_the_drawing_lists_are_not_knobs()-> None :
    cnt = {p.name for p in device_parameters('drawing')};assert not cnt &  {'blocks', "implants", 'electrodes', "material"}
    assert{"nx", 'ny', 'h_min_x', "h_min_y", 'degenerate'}<= cnt
@pytest.mark.parametrize (
    ( 'change', "complaint"  ),
    [
        (lambda e :  e.pop( 'voltage'  ), r"electrode 2 .*missing \['voltage'\]"),
        ( lambda e   :  e.update (colour =   "red"),   r"electrode 2 .*not a field \['colour'\]" ),
        ( lambda e  :  e.update( x0  =  'left' ),   'electrode 2: x0 is a number' ),
        ( lambda e  :  e.update(  name   =  3.0  ),   "electrode 2: name is a name"),
    ],
)



def test_a_bad_drawing_entry_is_named_by_number_and_field(change,complaint)-> None:
    par  =  drawing_defaults ( "drawing" )
    change(par["electrodes"] [1])

    with pytest.raises((TypeError,ValueError),match=complaint):

        build_from_spec("drawing",par)


def test_a_device_not_drawn_refuses_drawing_parts()->None:
    with  pytest.raises( ValueError ,   match =   "not built from blocks" )  :
        build_from_spec('stack',{'blocks':[]})




def test_a_drawing_refusal_reaches_the_caller_with_its_reason(  )  -> None   :
    par = drawing_defaults('drawing')
    foo =next(e for e in par["electrodes"]if e["name"] =="gate");  foo['y0'] =foo["y1"] = 0.0
    foo['x0'],foo['x1']=0.5e-4,1.0e-4

    with pytest.raises(ValueError,match ="Schottky"):

        build_from_spec('drawing',par)


def test_a_node_count_over_the_budget_is_refused_before_anything_is_built() ->None :
    with pytest.raises(ValueError, match  = f"budget of {NODE_BUDGET}"):
        build_from_spec('pn_diode',  { "n_nodes"   :   NODE_BUDGET +  1 })

def test_a_2d_mesh_over_the_budget_is_refused_though_no_knob_is() ->  None :
    with pytest.raises(ValueError, match = f"budget of {NODE_BUDGET}")  :
        build_from_spec("nmos", {"n_silicon" : 1000})



@pytest.mark.parametrize('junction', [0.0, 5e-5])

def test_a_diode_junction_on_a_contact_is_refused(junction   : float )   ->   None  :

    with pytest.raises( ValueError,
       match   = 'inside')   :
        build_from_spec('pn_diode', {"length": 5e-5, "junction": junction})
