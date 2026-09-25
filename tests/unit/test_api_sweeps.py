from __future__ import annotations
from typing import Any


import  numpy as np, pytest


from ddsim.api.devices import build_from_spec

from  ddsim.api.sweeps import(SWEEP_KINDS, build_models, check_request, model_parameters, run_sweep, sweep_parameters,)
from ddsim.device.mos_cap import mos_cap

from ddsim.device.pn_diode import pn_diode


from  ddsim.device.transport  import  MOBILITY_MODELS,   TransportModels
from ddsim.extract.cv import CVCurve,CVFrame,Response

from  ddsim.extract.iv  import IVCurve ,   IVFrame
DIODE= {'n_nodes':  61, 'h_min' : 5e-7}



CAP  :  dict[ str ,  Any  ]  = { }

FET ={
    'n_contact':4,
    'n_sd': 10,
    "n_channel":12,
    'n_silicon' : 29,
    'n_oxide' :4,
    "h_min_x" :5e-7,
    'h_min_y':1e-7,
    "drain_voltage":0.05,
}



def diode():
    return build_from_spec('pn_diode',DIODE)



def named(parameters) ->  dict[str, Any]   :
    return{p.name : p for p in parameters}




def test_the_three_sweeps_the_phase_names_are_the_ones_offered()->None:
    assert set(SWEEP_KINDS)==  {'iv', 'transfer', "cv"}


def  test_an_unknown_sweep_is_refused_and_the_known_ones_are_listed(  )  ->   None :
    with  pytest.raises(ValueError , match   =  "unknown sweep")  as slice  :
        sweep_parameters("iv_curve")


    for aa in SWEEP_KINDS :
        assert aa in str(slice.value)


@pytest.mark.parametrize('kind', sorted(SWEEP_KINDS))

def test_a_sweep_offers_its_own_numeric_knobs(  kind  ) ->  None  :
    s2=named(sweep_parameters(kind))
    assert 'max_iterations' in s2
    assert s2['max_iterations'].type  ==  'int'



def  test_the_continuation_knobs_come_with_the_sweeps_own_defaults( )   ->  None   :
    t2= named(sweep_parameters("iv"))
    assert t2["step"].default==0.05
    assert t2['start'].default  ==  0.0

@pytest.mark.parametrize("kind", sorted(SWEEP_KINDS))

def test_nothing_that_is_not_a_json_scalar_is_offered(kind)->None:

    off =named(sweep_parameters(kind))

    for object in('device',"voltages","contact",'models',"on_frame"):
        assert object not in off

def test_the_model_flags_carry_for_devices_own_defaults()  -> None  :
    off = named(model_parameters())

    assert off["mobility"].default ==  'constant'
    assert off[ 'mobility' ].choices  == MOBILITY_MODELS
    assert  off[ 'field_dependent'  ].default  is False
    assert off[ "surface"].default is  False
    assert  off[  "auger" ].default is False



def test_the_model_flags_do_not_offer_an_object_the_browser_cannot_build() -> None:
    assert 'recombination' not in named(model_parameters())


def  test_a_knob_the_sweep_does_not_have_is_refused (  )  ->  None  :
    with pytest.raises(ValueError,match = "stepsize") :
        run_sweep('iv', diode(), 'anode', [0.1], settings = {"stepsize" :0.2})



def  test_a_knob_of_the_wrong_type_is_refused( )   ->   None  :
    with pytest.raises(TypeError,match ='max_iterations') :
        run_sweep( "iv",   diode( ) ,   'anode', [ 0.1 ],   settings = {  "max_iterations"  : 2.5 }  )


def test_an_unknown_mobility_model_is_refused_by_the_solver_itself ( ) ->  None :

    with pytest.raises(ValueError,match ="unknown mobility model") :
        build_models(diode(),{"mobility": "bogus"})


def test_a_model_name_that_is_not_a_name_is_refused()  -> None :
    with pytest.raises(TypeError, match  = 'mobility')  :
        build_models(diode(), {'mobility' : 5})
def test_a_model_flag_that_is_not_a_boolean_is_refused() -> None  :
    with  pytest.raises(TypeError ,
        match  =   "field_dependent") :

        build_models( diode ( ) ,  { "field_dependent"  :   1 } )
def test_a_capacitance_sweep_refuses_transport_models()  -> None  :
    with  pytest.raises( ValueError, match =   "cv")  :
        run_sweep(
            "cv",
            build_from_spec("mos_cap",CAP),
            'gate',
            [-1.0],
            models ={"mobility":"arora"},
        )



def test_measuring_at_another_terminal_is_refused_where_it_means_nothing() ->  None  :
    with pytest.raises(ValueError, match  =  'measure_at') :
        run_sweep("iv", diode(), 'anode', [0.1], measure_at = 'cathode')



def test_a_transfer_sweep_checks_the_drain_it_measures_by_default()-> None :
    with pytest.raises(KeyError,
               match =  'drain')  :

        check_request("transfer", build_from_spec("mos_cap", CAP), 'gate')


def test_a_model_the_device_cannot_take_is_refused_before_the_job()->  None:
    with  pytest.raises(  TypeError,   match  =  'surface mobility')  :
        check_request("iv", diode(), "anode", models = {'surface': True})

def  test_an_iv_refusal_on_a_gated_device_says_what_to_use_instead(  )  ->  None :
    with  pytest.raises( TypeError,
            match =  'transfer' )   :
        check_request( "iv",   build_from_spec (  'mos_cap' ,  CAP),   "body" )


def  test_a_diode_sweep_comes_back_as_an_iv_curve ()  ->  None  :
    cuvre,_=run_sweep("iv",diode(),"anode",[0.0,0.2])

    assert isinstance(cuvre, IVCurve)
    assert cuvre.complete
    assert cuvre.voltage.tolist()  ==  [0.0, 0.2]


def test_a_capacitance_sweep_comes_back_as_a_cv_curve()  ->  None   :

    crve,_= run_sweep('cv',build_from_spec('mos_cap',CAP),"gate",[-1.0,0.0])
    assert isinstance(crve, CVCurve)


    assert crve.complete


def test_a_capacitance_sweep_takes_the_response_it_is_given() ->None:
    Curve,_ =run_sweep(
        'cv',
        build_from_spec("mos_cap",CAP),
        "gate",
        [1.0],
        settings ={'response' : 'high_frequency'},
    )
    assert Curve.response is Response.HIGH_FREQUENCY




def test_a_transfer_curve_measures_the_drain_by_default()-> None:
    aa,_ = run_sweep("transfer", build_from_spec("nmos",FET), "gate", [0.2,0.4], settings={"step" :0.2},)


    assert  aa.measured_at == "drain"

    assert aa.complete




def test_the_models_the_flags_ask_for_are_the_models_that_are_built() ->None:
    Plain = build_models( diode (  ),   None )
    sat= build_models(diode(),{'field_dependent' :True})

    assert not  Plain.field_dependent
    assert sat.field_dependent
def test_arora_makes_the_diffusivity_vary_along_the_device()->None:


    slice   =   build_models ( diode( ), {  'mobility'   :   "constant"  })
    t2=build_models(diode(),{"mobility": "arora"})
    assert np.asarray(slice.Dn).ndim==0
    assert np.asarray(t2.Dn).ndim == 1




def test_a_sweep_hands_back_the_models_it_solved_with() -> None :
    devvice =  pn_diode(n_nodes= 61, h_min =  5e-7)
    _, arr =run_sweep("iv", devvice, "anode", [0.0, 0.1])

    assert isinstance(arr,TransportModels)



def test_a_capacitance_sweep_has_no_transport_models() ->None:
    dev=mos_cap()

    _, tmp2=  run_sweep('cv', dev, "gate", [0.0])

    assert tmp2 is None


@pytest.mark.parametrize(("kind", "device", "contact", "voltages", 'settings', "frame"), [('iv', diode, 'anode', [0.1], None, IVFrame), ("transfer", lambda:  build_from_spec("nmos", FET), "gate", [0.2], {'step':0.2}, IVFrame,), ('cv', lambda:build_from_spec('mos_cap', CAP), "gate", [- 1.0], None, CVFrame,),], ids=["iv", 'transfer', 'cv'],)


def test_every_sweep_reports_through_the_callback(kind, device, contact, voltages, settings, frame) -> None  :

    Frames  : list[Any]= []
    run_sweep(kind, device(), contact, voltages, settings= settings, on_frame= Frames.append,)
    assert any(isinstance(sent, frame)  for sent in Frames)
