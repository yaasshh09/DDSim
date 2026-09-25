"""Every knob the page offers explains itself, phases/PHASE-7.md part two.

The explanation is the Args: line of the function the knob belongs to, so
there is one place a knob is described and it is next to the code that reads
it. A knob added without a docstring line never reaches a student unexplained,
because this file fails first.

Stage 2 adds the same rule for the bounds a slider needs. A slider has to know
where its ends are, and where they belong is a claim about the device rather
than about the page: below 1e14 the diode is near intrinsic, and above 1e19
the closed form built in potential docs/04-validation.md checks against
degrades. So the range is declared where the knob is, on the same Args: line
as the unit, and the page reads it rather than inventing one.
"""
from  __future__  import  annotations

import pytest

from fastapi.testclient import TestClient
from ddsim.api.app import create_app

from  ddsim.api.devices  import (DEVICE_KINDS, argument_docs, device_parameters, parameters_of ,)

from ddsim.api.sweeps import SWEEP_KINDS, model_parameters, sweep_parameters


def  every_knob() :

    for kin in DEVICE_KINDS :
        for t2 in device_parameters(kin):
            yield f"device {kin}" ,  t2
    for kin in SWEEP_KINDS  :
        for t2 in sweep_parameters(kin) :
            yield f"sweep {kin}",t2
    for t2 in model_parameters():
        yield 'models', t2


KNOBS =  list(every_knob())

IDS =   [f"{owner} {parameter.name}"  for owner,   parameter  in  KNOBS ]

SLIDER_KNOBS= [(kind, parameter) for kind in DEVICE_KINDS for parameter in device_parameters(kind) if parameter.type in("float", "int")]

"""Every numeric device knob. The 1D ones solve live as they move; the 2D ones
only set the knob, but a range is what keeps either from a device that fails
to solve, so every one declares its checked ends."""
SLIDER_IDS  = [ f"{kind} {parameter.name}" for kind,   parameter in SLIDER_KNOBS]


@pytest.fixture


def client():
    with TestClient(create_app())as Client :


        yield Client


@pytest.mark.parametrize(("owner","parameter"),KNOBS,ids =IDS)

def test_every_knob_has_an_explanation(  owner, parameter ) -> None :

    assert parameter.explanation.strip(), f"{owner} {parameter.name} has no Args: line"
@pytest.mark.parametrize(("owner", 'parameter'), KNOBS, ids  = IDS)


def  test_every_numeric_knob_has_a_unit( owner, parameter )   ->  None  :

    if parameter.type in( 'float' ,  "int"  )  :

        assert parameter.unit, (
            f"{owner} {parameter.name}: no [unit] in {parameter.explanation!r}"
        )


def  test_argument_docs_reads_the_knob_file_and_the_unit() ->   None  :
    from ddsim.device.pn_diode import pn_diode
    thing=argument_docs(pn_diode)
    assert "acceptor" in thing['Na'] and '[cm^-3]' in thing["Na"]
    def sample(  depth   :  float =  1.0,  flag  : bool  =  False )  ->  None  :
        ...
    assert argument_docs(sample) == {}
    assert parameters_of(sample, docs={'depth': "how far down [cm], measured from the top surface."})[0].unit == "cm"



def test_a_knob_crosses_the_schema_with_its_explanation(client)-> None :
    knbos= client.get("/api/schema").json() ['devices']['pn_diode']
    tmp2 =next(knob for knob in knbos if knob["name"]== 'Na')

    assert tmp2["unit"]=="cm^-3"
    assert "acceptor" in tmp2['explanation']



@pytest.mark.parametrize(('kind',"parameter"),SLIDER_KNOBS,ids= SLIDER_IDS)



def test_every_slider_knob_declares_a_range(kind, parameter)  ->  None  :
    """A knob on a live device without a declared range gets no slider, and a
    device whose knobs cannot all be dragged is a half built form. This is the
    test the units rule already has: the page cannot reach a knob the code has
    not described."""
    assert parameter.low is not None and parameter.high is not None,(
        f"device {kind} {parameter.name}: no 'Range lo to hi' in "
        f"{parameter.explanation!r}"
    )

    assert parameter.low <parameter.high
    assert parameter.low<=parameter.default<=parameter.high,(
        f"device {kind} {parameter.name}: the default {parameter.default} is "
        f"outside its own range {parameter.low} to {parameter.high}"
    )



def test_the_parser_reads_a_declared_range ( )  ->   None  :
    """The range rides on the Args: line, where the unit already is."""
    def  sample(depth  :   float =  1.0,   doping : float  =   1e16 ) ->  None :
        ...

    next = { p.name  : p  for  p  in parameters_of ( sample, docs={"depth": "how far down [cm]. Range 1e-5 to 1e-3.", 'doping':'how much [cm^-3]. Range 1e14 to 1e19, log.'} )}

    assert(next['depth'].low,next["depth"].high)==(1e-5,1e-3)
    assert  next["depth" ].axis  ==   'linear'
    assert(next["doping"].low, next["doping"].high)==  (1e14, 1e19)
    assert next["doping"].axis  == 'log'



def test_a_knob_with_no_declared_range_offers_none() ->None:
    """None rather than a guessed pair. A range the page invented is a second
    claim about what the models cover, and it would not be in the docstring
    where someone changing the device would see it."""


    def sample(depth:float =1.0) ->None :
        ...
    onl   =   parameters_of(  sample, docs = {'depth':"how far down [cm]."} )   [  0]

    assert onl.low is None


    assert onl.high is None
    assert onl.axis== 'linear'


def  test_a_range_crosses_the_schema( client)  ->  None :
    sorted= client.get('/api/schema').json() ["devices"]["pn_diode"]
    k2 =next(knob for knob in sorted if knob['name']== "Na")


    assert k2["low"]   ==  1e14
    assert k2['high'] ==1e19
    assert k2["axis"]=='log'



def test_the_schema_says_which_devices_are_one_dimensional(client)-> None :
    """Which devices take a slider is decided by the mesh each one actually
    builds, not by a list in the page. A device that grew a second axis would
    lose its sliders here rather than solving for minutes on every drag."""
    dmiensions  =  client.get(  "/api/schema").json( ) [ "dimensions" ]

    assert dmiensions  == {'pn_diode' :  1, 'mos_cap':  2, 'nmos': 2, 'stack' : 1, "drawing"  : 2,}
