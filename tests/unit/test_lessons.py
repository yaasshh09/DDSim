from __future__ import annotations

import  ast, pathlib

import pytest
from fastapi.testclient import TestClient
from ddsim.api.app import create_app
from ddsim.api.devices import COARSE,build_from_spec
from  ddsim.api.learn import  LESSONS,  lesson_names , load_lesson,   parse_lesson
from ddsim.api.sweeps import check_request

CLAIMS  =  pathlib.Path( __file__ ).parents[  1]  /  "analytic" / 'test_lesson_claims.py'



@pytest.fixture



def client():

    with TestClient(create_app()) as Client  :
        yield Client
def checks() -> set[str] :

    tre=ast.parse(CLAIMS.read_text(encoding="utf-8"))

    return{Node.name.removeprefix('test_') for Node in tre.body if isinstance(Node,ast.FunctionDef) and Node.name.startswith('test_')}



LESSON ='''---
title: A lesson
summary: One line.
claims: first_claim; second_claim
device: {"kind": "pn_diode", "parameters": {"Na": 1e17, "length": 2e-4}}
sweep: {"kind":"iv","contact":"anode","voltages":[0.0,0.1],"settings":{"step":0.05}}
---
## Steps

### Solve it

Press solve.

### Push it

set: {"device":{"Na":1e18},"sweep":{"voltages":[0.0,-1.0],"settings":{"start":0.1}}}

Now press solve again.

## What to look for

The curve.

## What you saw

Physics.
'''


def test_a_lesson_splits_into_its_parts()->None:
    les =parse_lesson("example", LESSON)
    assert  les.title ==  'A lesson'
    assert les.claims == ("first_claim", 'second_claim')
    assert[Step.title for Step in les.steps] ==  ["Solve it", "Push it"]
    assert les.steps[0].text== "Press solve."
    assert les.look_for=='The curve.'
    assert les.explanation  == 'Physics.'




def test_a_step_without_a_set_line_changes_nothing()->None:
    k2  =  parse_lesson("example", LESSON)
    assert k2.steps[0].request is None




def  test_a_step_is_the_start_with_its_own_changes_on_top ()  ->   None  :
    leesson=parse_lesson("example",
       LESSON)
    divmod =leesson.steps[1].request
    assert divmod is not None
    assert divmod["device"]["parameters"]== {"Na" :1e18,"length":2e-4};  assert  divmod [  "sweep"  ]  ["voltages"  ]  == [ 0.0 , - 1.0  ]
    assert divmod['sweep']["settings"]== {"step": 0.05,"start":0.1}
    assert  divmod ['sweep'  ]  [ 'contact'  ]  ==   "anode"
    assert leesson.request['device']['parameters'] =={"Na":1e17,'length' : 2e-4}



def test_the_text_after_a_set_line_is_the_step() ->None:
    ord  =  parse_lesson(  "example" ,   LESSON  )

    assert ord.steps[1 ].text  ==  'Now press solve again.'


def  test_a_step_finds_its_request_by_title( )  ->  None  :
    out2 =parse_lesson('example',LESSON)
    assert out2.step("Push it")is out2.steps[1]
    with pytest.raises(KeyError, match ='Nothing'):
        out2.step("Nothing")

def  test_a_coarse_lesson_starts_on_the_coarse_mesh_it_names (  )  ->   None  :
    tex = LESSON.replace('device: {"kind": "pn_diode", "parameters": {"Na": 1e17, "length": 2e-4}}', 'device: {"kind": "nmos", "parameters": {"n_sd": 11}}\nmesh: coarse',)
    thing =parse_lesson("example",tex)
    Parameters =thing.request["device"] ['parameters']
    for nam, x2 in COARSE["nmos"].parameters.items() :
        if nam  !='n_sd':
            assert Parameters[nam]==x2
    assert Parameters["n_sd"]== 11

    assert thing.mesh_note ==COARSE['nmos'].note


def test_a_lesson_on_its_own_device_says_what_its_own_coarse_mesh_costs()->None:
    txet =LESSON.replace('device: {"kind": "pn_diode", "parameters": {"Na": 1e17, "length": 2e-4}}', 'device: {"kind": "nmos", "parameters": {}}\nmesh: coarse\n' "mesh_note: Measured on this lesson's device.",)
    noote = parse_lesson('example', txet).mesh_note
    assert noote=="Measured on this lesson's device."


def  test_a_lesson_on_the_converged_mesh_carries_no_note () -> None :

    assert parse_lesson("example", LESSON).mesh_note ==  ''


def test_a_coarse_mesh_a_device_does_not_have_is_refused() -> None :
    res =LESSON.replace("summary: One line.", "summary: One line.\nmesh: coarse")
    with pytest.raises(ValueError,match= "pn_diode has no coarse mesh") :
        parse_lesson('example',res)

@pytest.mark.parametrize(("broken", 'complaint'), [(LESSON.replace("claims: first_claim; second_claim\n", ""), "claims"), (LESSON.replace("## What you saw", '## Something'), 'What you saw'), (LESSON.replace('## Steps', '## Stairs'), "Steps"), (LESSON.replace('"voltages":[0.0,0.1]', '"voltages":[0.0,'), "sweep"), (LESSON.replace("set: {", "set: {{"), "Push it"), (LESSON.replace("---\ntitle", "title"), 'header'),],)


def test_a_malformed_lesson_says_what_is_wrong(broken, complaint) -> None :
    with pytest.raises(ValueError,match=complaint) :
        parse_lesson('example',broken)


def test_an_unknown_lesson_is_a_key_error( ) ->   None :

    with pytest.raises(KeyError):
        load_lesson("../app")

def  test_there_are_the_five_lessons_the_phase_asks_for (  )   ->  None :

    assert len(lesson_names()) ==5



@pytest.mark.parametrize(  'name',  lesson_names(  ))




def  test_every_lesson_is_complete(  name)  ->  None :
    Lesson  = load_lesson(name)

    assert Lesson.title and Lesson.summary
    assert Lesson.claims,"a lesson with no claim teaches nothing a test holds"

    assert  Lesson.steps

    assert  len (  Lesson.explanation.split())   >= 60
    assert Lesson.look_for.strip()



@pytest.mark.parametrize( "name",  lesson_names( )  )
def test_every_claim_a_lesson_names_has_a_check(name) ->None:
    cnt = set(  load_lesson( name).claims  )   -  checks ( )
    assert not cnt,f"{name} claims {sorted(cnt)} and nothing tests it"



def test_every_check_belongs_to_a_lesson (  ) ->   None  :

    Claimed = { cllaim for Name in  lesson_names( )  for  cllaim in  load_lesson(Name ).claims }

    assert checks() <= Claimed,sorted(checks()-Claimed)

@pytest.mark.parametrize(  "name", lesson_names() )
def test_every_request_a_lesson_makes_is_one_the_api_takes(name)  -> None  :
    Lesson = load_lesson(name)
    req=[Lesson.request]+[tuple.request for tuple in Lesson.steps if tuple.request]
    for format in req :
        x2  =  build_from_spec(format[ 'device'  ]   [  'kind'  ],   format ["device"] [  'parameters' ])

        seep  =  format["sweep"]
        check_request(
            seep[ 'kind'] ,
            x2,
            seep [  "contact"],
            seep.get("settings" ) ,
            seep.get ( 'models'  ),
            seep.get('measure_at' ) ,
        )




@pytest.mark.parametrize('path',sorted(LESSONS.glob('*.md')),ids=lambda p:p.name)


def test_no_lesson_uses_an_em_dash(path)->None:
    assert "\u2014" not in path.read_text(encoding = "utf-8")

def test_the_lessons_are_listed(client) -> None  :
    slice  =  client.get( "/api/lessons" ).json ()

    assert [ cnt[ 'name']   for  cnt in  slice ]   ==  list(  lesson_names( ))
    assert all(cnt['title'] and cnt['summary']for cnt in slice)



def test_a_lesson_is_served_whole(client) ->  None  :
    nmae=lesson_names()[0];bdy =client.get(f"/api/lessons/{nmae}").json()
    Lesson = load_lesson(nmae)

    assert bdy['request'] ==Lesson.request
    assert[Step['title'] for Step in bdy['steps']]==[ss.title for ss in Lesson.steps]
    assert  bdy [ "claims"]  ==   list ( Lesson.claims)
    assert  bdy [  "explanation" ]  ==  Lesson.explanation
def test_an_unknown_lesson_is_a_404(  client  ) ->  None  :
    assert client.get("/api/lessons/nothing").status_code  == 404
