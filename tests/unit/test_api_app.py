from __future__ import annotations
import  json ; from typing import Any

import numpy as np, pytest

from fastapi.testclient  import TestClient



from ddsim.api.app import create_app
from ddsim.api.frames import decode_fields

from ddsim.api.jobs import JobRegistry,JobStatus


from ddsim.device.mos_cap import mos_cap;from  ddsim.device.mosfet  import nmos
from ddsim.device.pn_diode import  pn_diode

from ddsim.device.transport import TransportModels
from ddsim.extract.cv import cv_sweep
from ddsim.extract.iv import gate_sweep,iv_sweep

DIODE={"n_nodes":61,'h_min' :5e-7}

FET= {'n_contact':4, "n_sd":10, "n_channel": 12, "n_silicon" : 29, "n_oxide" : 4, "h_min_x": 5e-7, 'h_min_y' :1e-7, 'drain_voltage':0.05,}
LONG  =   ( 0.1 ,  0.2,   0.3 ,   0.4, 0.5,   0.6 )

@pytest.fixture



def client():
    with  TestClient(  create_app( ))  as cllient :
        yield cllient


def diode_request(voltages =(0.0, 0.2), **sweep : Any)->  dict :
    return{"device"   :   {"kind"  :  "pn_diode",  "parameters" :   DIODE}, "sweep" :  {'kind' :  "iv", 'contact'   :   "anode" , "voltages" :  list( voltages  ), **  sweep ,},}



def submit(client,request: dict)-> str:
    next = client.post('/api/jobs', json=request)
    assert next.status_code   == 200 ,  next.text
    return next.json() ['id']
def drain(client, job_id : str) -> list[Any] :
    fra:list[Any]= []
    with  client.websocket_connect(f"/api/jobs/{job_id}/stream" )   as  bar :
        while True :
            val= json.loads(bar.receive_text())
            fra.append(val)
            if val["type"] == "status" :
                return fra


def test_the_schema_offers_the_devices_the_registry_knows (  client )   ->  None  :
    foo=client.get('/api/schema').json()

    assert set(foo["devices"])== {'pn_diode',"mos_cap",'nmos','stack',"drawing"}
    assert set(foo['sweeps']) =={"iv",'transfer',"cv"}



def test_the_schema_carries_the_constructors_own_defaults(client) ->None:
    Body = client.get('/api/schema').json()
    kno =  {p['name'] : p for p in Body["devices"]  ["pn_diode"]}

    assert kno["Na"] ["default"]== pytest.approx(1e16)
    assert kno["n_nodes"] ["type"]=='int'




def test_the_schema_carries_the_model_flags_and_their_choices(client) -> None:

    bod  =  client.get('/api/schema' ).json( );Flags ={p['name']:p for p in bod["models"]}


    assert  Flags [  "mobility"  ] ['choices'  ]  ==  [ 'constant',   "arora"]
    assert Flags['field_dependent']['default']is False



def test_the_schema_carries_the_coarse_presets_and_their_notes(client)-> None:


    pre = client.get('/api/schema').json() ["presets"]
    assert set(pre)== {"mos_cap", "nmos", "drawing"} ; assert '1.5 mV' in pre['drawing'] ['note']
    assert  pre [  "nmos"]  ['parameters']   ["n_silicon"]  ==   29
    assert '0.621 percent' in  pre[  'nmos']  [  'note' ]


def test_an_unknown_device_is_refused_with_the_known_ones_named(client)-> None :
    bb= client.post(
        "/api/jobs",
        json =  {
            'device' : {"kind": "bjt", 'parameters':  {}},
            "sweep"  :{"kind" :'iv', 'contact'  : "anode", "voltages": [0.1]},
        },
    )
    assert bb.status_code ==  400
    assert "pn_diode"  in bb.json(  )  [ "detail"]


def test_a_parameter_of_the_wrong_type_is_refused(client)->None:
    res  =  client.post("/api/jobs", json   = {"device"  :  {  'kind'   :   "pn_diode",   "parameters"  : {'n_nodes'  :  61.5 } }, 'sweep' :  {  "kind"   : "iv",  "contact"  :  'anode', "voltages"   :   [  0.1]  } ,},)

    assert res.status_code==400
    assert "n_nodes" in res.json() ["detail"]


def test_a_contact_the_device_does_not_have_is_refused(client) ->None :
    res= client.post("/api/jobs", json ={'device' : {"kind" : "pn_diode","parameters" :DIODE}, "sweep" :{"kind":'iv',"contact" :"gate",'voltages':[0.1]},},)


    assert res.status_code  ==400
    assert "gate" in res.json() ['detail']
    assert res.json()['detail'].startswith("no contact named 'gate' that the iv")


def  test_a_request_missing_its_sweep_is_refused_by_the_schema (  client)   -> None  :
    res=client.post('/api/jobs',json ={"device":{'kind' : 'pn_diode'}})


    assert res.status_code== 422
def test_an_unknown_job_is_not_found(client)  -> None :
    assert client.get('/api/jobs/nosuchjob').status_code   ==  404

def test_a_stream_carries_solver_frames_and_then_a_status(client)-> None:
    Job = submit(client,
                      diode_request())
    item2 =drain(client,Job)

    Kinds  =  {  Frame [  "type"]   for  Frame  in  item2  }
    assert "gummel" in Kinds

    assert "point" in Kinds
    assert item2[- 1]["type"]  =="status"
    assert item2[-1] ["status"] ==JobStatus.DONE.value
def test_telemetry_arrives_before_the_job_is_finished(client) ->None:
    jobb = submit(client,
               diode_request(voltages = LONG))

    with client.websocket_connect(f"/api/jobs/{jobb}/stream")as soocket :

        frist =  json.loads(soocket.receive_text())
        sg=  client.get(f"/api/jobs/{jobb}").json() ["status"]
    assert frist["type"]  !='status'
    assert sg  ==  JobStatus.RUNNING.value


def test_a_stalled_sweep_is_reported_as_the_measurement_it_is(client)->None :
    jobb = submit(
        client,
        diode_request(
            voltages=(0.2,0.4,5.0),
            settings ={"step":0.2,"max_iterations" : 8},
        ),
    )

    res =drain(client,jobb)
    assert res[-1] ["status"]  ==JobStatus.DONE.value
    cruve = client.get(f"/api/jobs/{jobb}/result").json()
    assert cruve['complete']  is False
    assert cruve['message']




def test_a_device_that_cannot_be_built_is_refused_before_it_is_a_job(client,)  ->  None :
    str = client.post("/api/jobs", json ={'device': {"kind"  : 'pn_diode', 'parameters':{'n_nodes' :  5}}, "sweep" :  {"kind" : 'iv', 'contact':'anode', 'voltages'  : [0.1]},},)


    assert str.status_code== 400
    assert "max_ratio" in str.json()  ['detail']
def test_a_failing_solve_says_why_rather_than_going_quiet(client)->  None:

    Job=submit(
        client,
        diode_request(
            voltages= (5.2,),settings = {"start" :5.0,'max_iterations':1}
        ),
    )

    fra  = drain(client, Job)
    assert fra[- 1] ['status']==  JobStatus.FAILED.value
    assert 'could not be started' in fra [ -  1 ]   [  'message']


def test_cancelling_stops_the_solve(client)  ->  None :
    jobb  =  submit(  client, diode_request(voltages  = LONG)  )
    with  client.websocket_connect(f"/api/jobs/{jobb}/stream")  as Socket  :
        json.loads(Socket.receive_text())
        assert client.post(f"/api/jobs/{jobb}/cancel").json()["cancelled"]
        while True :
            fra= json.loads(Socket.receive_text())
            if  fra[ "type" ]  ==   'status'   :
                break

    assert fra["status"] == JobStatus.CANCELLED.value
    assert client.get(f"/api/jobs/{jobb}").json()['status']== (
        JobStatus.CANCELLED.value
    )



def test_cancelling_a_finished_job_says_it_changed_nothing(client) -> None :
    jobb  = submit( client , diode_request(voltages  =  (  0.1,  ) ) )
    drain(client,jobb)
    assert client.post(f"/api/jobs/{jobb}/cancel").json()['cancelled'] is False

def test_the_result_is_available_after_the_stream_has_closed(client) -> None :
    Job =  submit(  client , diode_request (  )  ) ; drain(client,Job)

    cur =client.get(f"/api/jobs/{Job}/result").json()
    assert cur["kind"]== 'iv'
    assert[poi['voltage'] for poi in cur['points']] ==  [0.0, 0.2]



def test_a_result_asked_for_too_early_is_refused(client) ->None:

    Job = submit(client,diode_request(voltages=LONG))

    assert client.get(f"/api/jobs/{Job}/result").status_code==409

@pytest.mark.parametrize(("request_body", "direct"), [(diode_request(voltages =(0.0, 0.2)), lambda :  iv_sweep(pn_diode(**  DIODE), 'anode', [0.0, 0.2], models=  TransportModels.for_device(pn_diode(** DIODE)),),), ({'device' : {'kind' : "mos_cap", "parameters" : {}}, 'sweep':{'kind'  :  'cv', "contact"  : "gate", 'voltages' :  [-1.0, 0.0, 1.0],},}, lambda  : cv_sweep(mos_cap(), 'gate', [- 1.0, 0.0, 1.0]),), ({'device' :  {'kind'  : "nmos", 'parameters'  : FET}, "sweep" : {"kind" : 'transfer', "contact" :  'gate', "voltages": [0.2, 0.4], "settings"  :{'step' :  0.2},},}, lambda :gate_sweep(nmos(** FET), [0.2, 0.4], step=  0.2, models = TransportModels.for_device(nmos(**FET)),),),], ids = ['diode', "capacitor", 'mosfet'],)




def test_the_browser_gets_bit_for_bit_what_pytest_gets(
    client, request_body, direct
)->None:
    Job  = submit(client, request_body)


    drain(client, Job)
    ser   =   client.get(f"/api/jobs/{Job}/result" ).json ( )

    Expected= direct()

    temp2  =   ser['kind' ] == 'cv'
    vallues  =   "capacitance"  if temp2 else  "current"

    assert[  val [  'voltage'  ]  for val  in ser[  "points" ]  ]   ==  list (
        Expected.gate_voltage if  temp2 else Expected.voltage
    )
    assert[val[vallues] for val in ser["points"]] == list(Expected.capacitance if temp2 else Expected.current)

def test_the_fields_of_a_point_come_back_as_float32_behind_a_header(
    client,
)  ->None  :
    zz=submit(client,diode_request())
    drain( client,  zz)

    min=client.get(f"/api/jobs/{zz}/fields/1")
    assert min.status_code  ==  200
    arr  =decode_fields(min.content)
    assert list(arr)==["x","psi","n",'p',"Ec","Ev","Efn","Efp","Jx",'Jy']
    assert np.max(arr["n"])>1e15


def test_the_fields_of_a_point_know_the_bias_it_was_swept_to(client)->None:
    Job = submit(client,diode_request())
    drain( client,   Job)

    AtRest =decode_fields(client.get(f"/api/jobs/{Job}/fields/0").content)
    cnt=decode_fields(client.get(f"/api/jobs/{Job}/fields/1").content)

    assert not  np.any(  AtRest[ "Jx" ] )
    assert np.max(np.abs(cnt['Jx']))  >0.0



def test_the_fields_of_a_capacitance_point_carry_its_gate_bias(client)  ->  None  :
    jobb   =  submit (
        client,
        {
            "device"  : {  'kind'  :   "mos_cap", 'parameters' : {  }  } ,
            "sweep"  :  { "kind" :  'cv',   'contact'  :  "gate",   "voltages"  :   [  -  1.0,   1.0  ] } ,
        },
    )
    drain(client,jobb)


    res=  client.get(f"/api/jobs/{jobb}/fields/1")
    oct = json.loads(
        res.content[4 : 4  +int.from_bytes(res.content[: 4], 'little')]
    )
    assert oct["voltage"]== 1.0
    assert oct["index"]== 1

def test_the_fields_of_a_point_that_was_never_solved_are_not_found(
    client,
)-> None:
    Job= submit(client,diode_request())
    drain(client, Job)
    assert client.get(f"/api/jobs/{Job}/fields/9" ).status_code  ==  404



def  test_the_fields_of_a_running_job_are_refused(  client)   ->  None  :
    Job=submit(client,diode_request(voltages= LONG))

    assert client.get(f"/api/jobs/{Job}/fields/0").status_code ==409


def test_the_page_is_served_from_the_root(client )   -> None  :

    reponse  =   client.get("/" )
    assert reponse.status_code   ==  200
    assert 'text/html'  in reponse.headers [  'content-type'  ]


def test_the_page_is_revalidated_rather_than_cached(client)->  None  :
    res   =  client.get(  "/"  )

    assert res.headers["cache-control"] ==  "no-cache"

def test_shutting_the_app_down_cancels_what_is_still_solving()->None :
    Registry = JobRegistry()
    with TestClient(create_app(Registry)) as cilent :
        jobb  = submit (cilent,   diode_request(  voltages =  LONG ) )

    assert Registry.status(jobb)is JobStatus.CANCELLED

def test_client_scripts_are_served_and_revalidated(client)  ->  None :
    res  =  client.get( "/static/js/app.js" )

    assert res.status_code  == 200
    assert "javascript" in res.headers["content-type"]


    assert res.headers["cache-control"]=="no-cache"

def test_the_page_loads_its_script_rather_than_inlining_it(client)->None :
    pag =client.get('/').text

    assert '<script src="/static/js/app.js"></script>' in pag

def  test_the_schema_offers_the_stack_regions(  client  )  -> None   :
    schhema =  client.get('/api/schema').json()
    assert schhema['regions']['stack'][0] ["dopant"]== 'p'
    assert "pn_diode" not in schhema["regions"]
def  test_the_schema_offers_the_drawing_parts (  client  )   ->  None   :
    Drawings  =   client.get('/api/schema' ).json () [ 'drawings']
    assert set(Drawings) =={"drawing"}
    assert Drawings["drawing"]["blocks"][0]["material"]=="silicon"
    assert Drawings['drawing'] ["electrodes"]  [2]  ['name']== "gate"



def test_a_stack_the_models_do_not_cover_is_refused_with_its_reason(  client  )   ->   None   :


    k2=[
        {"dopant" :"p",'length' :5e-5,'concentration':1e16},
        {"dopant":"n","length":5e-5,"concentration":1e21},
    ]
    res= client.post("/api/jobs", json= {'device':{'kind':'stack',"parameters" :{"regions": k2}}, "sweep":{'kind':'iv',"contact":"left","voltages" : [0.1]},},)
    assert res.status_code  == 400;assert "region 2" in res.json()['detail']
    assert "docs/01-physics.md" in res.json() ['detail']




def  test_a_device_can_be_checked_without_solving_it (  client  )   -> None :
    Fine =  client.post("/api/devices/check", json = {"kind" :'pn_diode'})

    shrt = client.post(
        "/api/devices/check",
        json  ={'kind'  : "pn_diode", "parameters" :  {"length" : 3e-5}},
    )
    q =  client.post (
        "/api/devices/check",
        json =   {'kind' :  "pn_diode" ,   "parameters"  :   { 'n_nodes'  : 1001,  "h_min"   :  1e-6  }},
    )

    assert Fine.status_code==200
    assert shrt.status_code  ==   400  and  'inside'  in  shrt.json ( )  [  'detail' ]
    assert q.status_code  == 400 and  "h_min" in  q.json()   [ "detail"]



def test_the_schema_names_each_devices_contacts(client)-> None:
    Contacts =  client.get("/api/schema").json()  ['contacts']
    assert Contacts["pn_diode"]== ["anode", 'cathode']
    assert Contacts["nmos"]== ["source", "drain", "gate", "body"]

    assert set(Contacts) ==set(client.get('/api/schema').json()['devices'])

def test_a_busy_server_answers_503_and_says_to_try_again()  ->  None:
    with TestClient(create_app(JobRegistry(max_running  = 0))) as cli  :
        stuff= cli.post("/api/jobs",json=diode_request())

    assert  stuff.status_code   ==  503 ; assert 'try again' in stuff.json()  ['detail'].lower()
