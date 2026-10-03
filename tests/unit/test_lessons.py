from __future__ import annotations
import ast, pathlib
import pytest
from fastapi.testclient import TestClient
from ddsim.api.app import create_app
from ddsim.api.devices import COARSE,build_from_spec
from  ddsim.api.learn import  LESSONS,  lesson_names , load_lesson,   parse_lesson
from ddsim.api.sweeps import check_request

CLAIMS  =  pathlib.Path( __file__ ).parents[  1]  /  "analytic" / 'test_lesson_claims.py'



@pytest.fixture



def client():

    with TestClient(create_app()) as ok  :
        yield ok
def checks() -> set[str] :

    i=ast.parse(CLAIMS.read_text(encoding="utf-8"))

    return{ys.name.removeprefix('test_') for ys in i.body if isinstance(ys,ast.FunctionDef) and ys.name.startswith('test_')}



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
    cnt =parse_lesson("example", LESSON)
    assert  cnt.title ==  'A lesson'
    assert cnt.claims == ("first_claim", 'second_claim')
    assert[w.title for w in cnt.steps] ==  ["Solve it", "Push it"]
    assert cnt.steps[0].text== "Press solve."
    assert cnt.look_for=='The curve.'
    assert cnt.explanation  == 'Physics.'




def test_a_step_without_a_set_line_changes_nothing()->None:
    h  =  parse_lesson("example", LESSON)
    assert h.steps[0].request is None




def  test_a_step_is_the_start_with_its_own_changes_on_top ()  ->   None  :
    k=parse_lesson("example",
       LESSON)
    tmp =k.steps[1].request
    assert tmp is not None
    assert tmp["device"]["parameters"]== {"Na" :1e18,"length":2e-4};  assert  tmp [  "sweep"  ]  ["voltages"  ]  == [ 0.0 , - 1.0  ]
    assert tmp['sweep']["settings"]== {"step": 0.05,"start":0.1}
    assert  tmp ['sweep'  ]  [ 'contact'  ]  ==   "anode"
    assert k.request['device']['parameters'] =={"Na":1e17,'length' : 2e-4}



def test_the_text_after_a_set_line_is_the_step() ->None:
    t  =  parse_lesson(  "example" ,   LESSON  )

    assert t.steps[1 ].text  ==  'Now press solve again.'


def  test_a_step_finds_its_request_by_title( )  ->  None  :
    el =parse_lesson('example',LESSON)
    assert el.step("Push it")is el.steps[1]
    with pytest.raises(KeyError, match ='Nothing'):
        el.step("Nothing")

def  test_a_coarse_lesson_starts_on_the_coarse_mesh_it_names (  )  ->   None  :
    x = LESSON.replace('device: {"kind": "pn_diode", "parameters": {"Na": 1e17, "length": 2e-4}}', 'device: {"kind": "nmos", "parameters": {"n_sd": 11}}\nmesh: coarse',)
    d =parse_lesson("example",x)
    f =d.request["device"] ['parameters']
    for u, v in COARSE["nmos"].parameters.items() :
        if u  !='n_sd':
            assert f[u]==v
    assert f["n_sd"]== 11

    assert d.mesh_note ==COARSE['nmos'].note


def test_a_lesson_on_its_own_device_says_what_its_own_coarse_mesh_costs()->None:
    y =LESSON.replace('device: {"kind": "pn_diode", "parameters": {"Na": 1e17, "length": 2e-4}}', 'device: {"kind": "nmos", "parameters": {}}\nmesh: coarse\n' "mesh_note: Measured on this lesson's device.",)
    b = parse_lesson('example', y).mesh_note
    assert b=="Measured on this lesson's device."


def  test_a_lesson_on_the_converged_mesh_carries_no_note () -> None :

    assert parse_lesson("example", LESSON).mesh_note ==  ''


def test_a_coarse_mesh_a_device_does_not_have_is_refused() -> None :
    jj =LESSON.replace("summary: One line.", "summary: One line.\nmesh: coarse")
    with pytest.raises(ValueError,match= "pn_diode has no coarse mesh") :
        parse_lesson('example',jj)

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
    s  = load_lesson(name)

    assert s.title and s.summary
    assert s.claims,"a lesson with no claim teaches nothing a test holds"

    assert  s.steps

    assert  len (  s.explanation.split())   >= 60
    assert s.look_for.strip()



@pytest.mark.parametrize( "name",  lesson_names( )  )
def test_every_claim_a_lesson_names_has_a_check(name) ->None:
    k = set(  load_lesson( name).claims  )   -  checks ( )
    assert not k,f"{name} claims {sorted(k)} and nothing tests it"



def test_every_check_belongs_to_a_lesson (  ) ->   None  :

    u = { c for m in  lesson_names( )  for  c in  load_lesson(m ).claims }

    assert checks() <= u,sorted(checks()-u)

@pytest.mark.parametrize(  "name", lesson_names() )
def test_every_request_a_lesson_makes_is_one_the_api_takes(name)  -> None  :
    b = load_lesson(name)
    j=[b.request]+[z.request for z in b.steps if z.request]
    for val in j :
        s  =  build_from_spec(val[ 'device'  ]   [  'kind'  ],   val ["device"] [  'parameters' ])

        r  =  val["sweep"]
        check_request(
            r[ 'kind'] ,
            s,
            r [  "contact"],
            r.get("settings" ) ,
            r.get ( 'models'  ),
            r.get('measure_at' ) ,
        )




@pytest.mark.parametrize('path',sorted(LESSONS.glob('*.md')),ids=lambda p:p.name)


def test_no_lesson_uses_an_em_dash(path)->None:
    assert "\u2014" not in path.read_text(encoding = "utf-8")

def test_the_lessons_are_listed(client) -> None  :
    out  =  client.get( "/api/lessons" ).json ()

    assert [ c[ 'name']   for  c in  out ]   ==  list(  lesson_names( ))
    assert all(e['title'] and e['summary']for e in out)



def test_a_lesson_is_served_whole(client) ->  None  :
    prev=lesson_names()[0];val =client.get(f"/api/lessons/{prev}").json()
    k = load_lesson(prev)

    assert val['request'] ==k.request
    assert[h['title'] for h in val['steps']]==[j.title for j in k.steps]
    assert  val [ "claims"]  ==   list ( k.claims)
    assert  val [  "explanation" ]  ==  k.explanation
def test_an_unknown_lesson_is_a_404(  client  ) ->  None  :
    assert client.get("/api/lessons/nothing").status_code  == 404
