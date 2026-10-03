from __future__ import annotations
import json ; from typing import Any
import numpy as np, pytest
from fastapi.testclient import TestClient
from ddsim.api.app import create_app
from ddsim.api.frames import decode_fields
from ddsim.api.jobs import JobRegistry,JobStatus
from ddsim.device.mos_cap import mos_cap;from ddsim.device.mosfet import nmos
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import TransportModels
from ddsim.extract.cv import cv_sweep
from ddsim.extract.iv import gate_sweep,iv_sweep

DIODE={"n_nodes":61,'h_min' :5e-7}

FET= {'n_contact':4, "n_sd":10, "n_channel": 12, "n_silicon" : 29, "n_oxide" : 4, "h_min_x": 5e-7, 'h_min_y' :1e-7, 'drain_voltage':0.05,}
LONG  =   ( 0.1 ,  0.2,   0.3 ,   0.4, 0.5,   0.6 )

@pytest.fixture



def client():
    with  TestClient(  create_app( ))  as dd :
        yield dd


def diode_request(voltages =(0.0, 0.2), **sweep : Any)->  dict :
    return{"device"   :   {"kind"  :  "pn_diode",  "parameters" :   DIODE}, "sweep" :  {'kind' :  "iv", 'contact'   :   "anode" , "voltages" :  list( voltages  ), **  sweep ,},}



def submit(client,request: dict)-> str:
    vals = client.post('/api/jobs', json=request)
    assert vals.status_code   == 200 ,  vals.text
    return vals.json() ['id']
def drain(client, job_id : str) -> list[Any] :
    v:list[Any]= []
    with  client.websocket_connect(f"/api/jobs/{job_id}/stream" )   as  stuff :
        while True :
            lst= json.loads(stuff.receive_text())
            v.append(lst)
            if lst["type"] == "status" :
                return v


def test_the_schema_offers_the_devices_the_registry_knows (  client )   ->  None  :
    f=client.get('/api/schema').json()

    assert set(f["devices"])== {'pn_diode',"mos_cap",'nmos','stack',"drawing"}
    assert set(f['sweeps']) =={"iv",'transfer',"cv"}



def test_the_schema_carries_the_constructors_own_defaults(client) ->None:
    vals = client.get('/api/schema').json()
    d =  {p['name'] : p for p in vals["devices"]  ["pn_diode"]}

    assert d["Na"] ["default"]== pytest.approx(1e16)
    assert d["n_nodes"] ["type"]=='int'




def test_the_schema_carries_the_model_flags_and_their_choices(client) -> None:

    b2  =  client.get('/api/schema' ).json( );res2 ={p['name']:p for p in b2["models"]}


    assert  res2 [  "mobility"  ] ['choices'  ]  ==  [ 'constant',   "arora"]
    assert res2['field_dependent']['default']is False



def test_the_schema_carries_the_coarse_presets_and_their_notes(client)-> None:


    z = client.get('/api/schema').json() ["presets"]
    assert set(z)== {"mos_cap", "nmos", "drawing"} ; assert '1.5 mV' in z['drawing'] ['note']
    assert  z [  "nmos"]  ['parameters']   ["n_silicon"]  ==   29
    assert '0.621 percent' in  z[  'nmos']  [  'note' ]


def test_an_unknown_device_is_refused_with_the_known_ones_named(client)-> None :
    z= client.post(
        "/api/jobs",
        json =  {
            'device' : {"kind": "bjt", 'parameters':  {}},
            "sweep"  :{"kind" :'iv', 'contact'  : "anode", "voltages": [0.1]},
        },
    )
    assert z.status_code ==  400
    assert "pn_diode"  in z.json(  )  [ "detail"]


def test_a_parameter_of_the_wrong_type_is_refused(client)->None:
    z  =  client.post("/api/jobs", json   = {"device"  :  {  'kind'   :   "pn_diode",   "parameters"  : {'n_nodes'  :  61.5 } }, 'sweep' :  {  "kind"   : "iv",  "contact"  :  'anode', "voltages"   :   [  0.1]  } ,},)

    assert z.status_code==400
    assert "n_nodes" in z.json() ["detail"]


def test_a_contact_the_device_does_not_have_is_refused(client) ->None :
    num= client.post("/api/jobs", json ={'device' : {"kind" : "pn_diode","parameters" :DIODE}, "sweep" :{"kind":'iv',"contact" :"gate",'voltages':[0.1]},},)


    assert num.status_code  ==400
    assert "gate" in num.json() ['detail']
    assert num.json()['detail'].startswith("no contact named 'gate' that the iv")


def  test_a_request_missing_its_sweep_is_refused_by_the_schema (  client)   -> None  :
    bb=client.post('/api/jobs',json ={"device":{'kind' : 'pn_diode'}})


    assert bb.status_code== 422
def test_an_unknown_job_is_not_found(client)  -> None :
    assert client.get('/api/jobs/nosuchjob').status_code   ==  404

def test_a_stream_carries_solver_frames_and_then_a_status(client)-> None:
    i = submit(client,
                      diode_request())
    m =drain(client,i)

    k  =  {  flag [  "type"]   for  flag  in  m  }
    assert "gummel" in k

    assert "point" in k
    assert m[- 1]["type"]  =="status"
    assert m[-1] ["status"] ==JobStatus.DONE.value
def test_telemetry_arrives_before_the_job_is_finished(client) ->None:
    v2 = submit(client,
               diode_request(voltages = LONG))

    with client.websocket_connect(f"/api/jobs/{v2}/stream")as t :

        ss =  json.loads(t.receive_text())
        u=  client.get(f"/api/jobs/{v2}").json() ["status"]
    assert ss["type"]  !='status'
    assert u  ==  JobStatus.RUNNING.value


def test_a_stalled_sweep_is_reported_as_the_measurement_it_is(client)->None :
    r = submit(
        client,
        diode_request(
            voltages=(0.2,0.4,5.0),
            settings ={"step":0.2,"max_iterations" : 8},
        ),
    )

    b =drain(client,r)
    assert b[-1] ["status"]  ==JobStatus.DONE.value
    y = client.get(f"/api/jobs/{r}/result").json()
    assert y['complete']  is False
    assert y['message']




def test_a_device_that_cannot_be_built_is_refused_before_it_is_a_job(client,)  ->  None :
    j = client.post("/api/jobs", json ={'device': {"kind"  : 'pn_diode', 'parameters':{'n_nodes' :  5}}, "sweep" :  {"kind" : 'iv', 'contact':'anode', 'voltages'  : [0.1]},},)


    assert j.status_code== 400
    assert "max_ratio" in j.json()  ['detail']
def test_a_failing_solve_says_why_rather_than_going_quiet(client)->  None:

    y2=submit(
        client,
        diode_request(
            voltages= (5.2,),settings = {"start" :5.0,'max_iterations':1}
        ),
    )

    y  = drain(client, y2)
    assert y[- 1] ['status']==  JobStatus.FAILED.value
    assert 'could not be started' in y [ -  1 ]   [  'message']


def test_cancelling_stops_the_solve(client)  ->  None :
    rr  =  submit(  client, diode_request(voltages  = LONG)  )
    with  client.websocket_connect(f"/api/jobs/{rr}/stream")  as xx  :
        json.loads(xx.receive_text())
        assert client.post(f"/api/jobs/{rr}/cancel").json()["cancelled"]
        while True :
            g= json.loads(xx.receive_text())
            if  g[ "type" ]  ==   'status'   :
                break

    assert g["status"] == JobStatus.CANCELLED.value
    assert client.get(f"/api/jobs/{rr}").json()['status']== (
        JobStatus.CANCELLED.value
    )



def test_cancelling_a_finished_job_says_it_changed_nothing(client) -> None :
    x  = submit( client , diode_request(voltages  =  (  0.1,  ) ) )
    drain(client,x)
    assert client.post(f"/api/jobs/{x}/cancel").json()['cancelled'] is False

def test_the_result_is_available_after_the_stream_has_closed(client) -> None :
    val2 =  submit(  client , diode_request (  )  ) ; drain(client,val2)

    z =client.get(f"/api/jobs/{val2}/result").json()
    assert z["kind"]== 'iv'
    assert[s['voltage'] for s in z['points']] ==  [0.0, 0.2]



def test_a_result_asked_for_too_early_is_refused(client) ->None:

    x = submit(client,diode_request(voltages=LONG))

    assert client.get(f"/api/jobs/{x}/result").status_code==409

@pytest.mark.parametrize(("request_body", "direct"), [(diode_request(voltages =(0.0, 0.2)), lambda :  iv_sweep(pn_diode(**  DIODE), 'anode', [0.0, 0.2], models=  TransportModels.for_device(pn_diode(** DIODE)),),), ({'device' : {'kind' : "mos_cap", "parameters" : {}}, 'sweep':{'kind'  :  'cv', "contact"  : "gate", 'voltages' :  [-1.0, 0.0, 1.0],},}, lambda  : cv_sweep(mos_cap(), 'gate', [- 1.0, 0.0, 1.0]),), ({'device' :  {'kind'  : "nmos", 'parameters'  : FET}, "sweep" : {"kind" : 'transfer', "contact" :  'gate', "voltages": [0.2, 0.4], "settings"  :{'step' :  0.2},},}, lambda :gate_sweep(nmos(** FET), [0.2, 0.4], step=  0.2, models = TransportModels.for_device(nmos(**FET)),),),], ids = ['diode', "capacitor", 'mosfet'],)




def test_the_browser_gets_bit_for_bit_what_pytest_gets(
    client, request_body, direct
)->None:
    v  = submit(client, request_body)


    drain(client, v)
    w   =   client.get(f"/api/jobs/{v}/result" ).json ( )

    prev= direct()

    a  =   w['kind' ] == 'cv'
    d  =   "capacitance"  if a else  "current"

    assert[  z [  'voltage'  ]  for z  in w[  "points" ]  ]   ==  list (
        prev.gate_voltage if  a else prev.voltage
    )
    assert[z[d] for z in w["points"]] == list(prev.capacitance if a else prev.current)

def test_the_fields_of_a_point_come_back_as_float32_behind_a_header(
    client,
)  ->None  :
    i=submit(client,diode_request())
    drain( client,  i)

    r=client.get(f"/api/jobs/{i}/fields/1")
    assert r.status_code  ==  200
    c  =decode_fields(r.content)
    assert list(c)==["x","psi","n",'p',"Ec","Ev","Efn","Efp","Jx",'Jy']
    assert np.max(c["n"])>1e15


def test_the_fields_of_a_point_know_the_bias_it_was_swept_to(client)->None:
    a = submit(client,diode_request())
    drain( client,   a)

    d2 =decode_fields(client.get(f"/api/jobs/{a}/fields/0").content)
    dd=decode_fields(client.get(f"/api/jobs/{a}/fields/1").content)

    assert not  np.any(  d2[ "Jx" ] )
    assert np.max(np.abs(dd['Jx']))  >0.0



def test_the_fields_of_a_capacitance_point_carry_its_gate_bias(client)  ->  None  :
    s   =  submit (
        client,
        {
            "device"  : {  'kind'  :   "mos_cap", 'parameters' : {  }  } ,
            "sweep"  :  { "kind" :  'cv',   'contact'  :  "gate",   "voltages"  :   [  -  1.0,   1.0  ] } ,
        },
    )
    drain(client,s)


    xx=  client.get(f"/api/jobs/{s}/fields/1")
    c = json.loads(
        xx.content[4 : 4  +int.from_bytes(xx.content[: 4], 'little')]
    )
    assert c["voltage"]== 1.0
    assert c["index"]== 1

def test_the_fields_of_a_point_that_was_never_solved_are_not_found(
    client,
)-> None:
    info= submit(client,diode_request())
    drain(client, info)
    assert client.get(f"/api/jobs/{info}/fields/9" ).status_code  ==  404



def  test_the_fields_of_a_running_job_are_refused(  client)   ->  None  :
    s=submit(client,diode_request(voltages= LONG))

    assert client.get(f"/api/jobs/{s}/fields/0").status_code ==409


def test_the_page_is_served_from_the_root(client )   -> None  :

    u  =   client.get("/" )
    assert u.status_code   ==  200
    assert 'text/html'  in u.headers [  'content-type'  ]


def test_the_page_is_revalidated_rather_than_cached(client)->  None  :
    num   =  client.get(  "/"  )

    assert num.headers["cache-control"] ==  "no-cache"

def test_shutting_the_app_down_cancels_what_is_still_solving()->None :
    ss = JobRegistry()
    with TestClient(create_app(ss)) as z :
        x2  = submit (z,   diode_request(  voltages =  LONG ) )

    assert ss.status(x2)is JobStatus.CANCELLED

def test_client_scripts_are_served_and_revalidated(client)  ->  None :
    cc  =  client.get( "/static/js/app.js" )

    assert cc.status_code  == 200
    assert "javascript" in cc.headers["content-type"]


    assert cc.headers["cache-control"]=="no-cache"

def test_the_page_loads_its_script_rather_than_inlining_it(client)->None :
    t2 =client.get('/').text

    assert '<script src="/static/js/app.js"></script>' in t2

def  test_the_schema_offers_the_stack_regions(  client  )  -> None   :
    u =  client.get('/api/schema').json()
    assert u['regions']['stack'][0] ["dopant"]== 'p'
    assert "pn_diode" not in u["regions"]
def  test_the_schema_offers_the_drawing_parts (  client  )   ->  None   :
    e  =   client.get('/api/schema' ).json () [ 'drawings']
    assert set(e) =={"drawing"}
    assert e["drawing"]["blocks"][0]["material"]=="silicon"
    assert e['drawing'] ["electrodes"]  [2]  ['name']== "gate"



def test_a_stack_the_models_do_not_cover_is_refused_with_its_reason(  client  )   ->   None   :


    m=[
        {"dopant" :"p",'length' :5e-5,'concentration':1e16},
        {"dopant":"n","length":5e-5,"concentration":1e21},
    ]
    el= client.post("/api/jobs", json= {'device':{'kind':'stack',"parameters" :{"regions": m}}, "sweep":{'kind':'iv',"contact":"left","voltages" : [0.1]},},)
    assert el.status_code  == 400;assert "region 2" in el.json()['detail']
    assert "references/physics.md" in el.json() ['detail']




def  test_a_device_can_be_checked_without_solving_it (  client  )   -> None :
    info =  client.post("/api/devices/check", json = {"kind" :'pn_diode'})

    ok = client.post(
        "/api/devices/check",
        json  ={'kind'  : "pn_diode", "parameters" :  {"length" : 3e-5}},
    )
    b =  client.post (
        "/api/devices/check",
        json =   {'kind' :  "pn_diode" ,   "parameters"  :   { 'n_nodes'  : 1001,  "h_min"   :  1e-6  }},
    )

    assert info.status_code==200
    assert ok.status_code  ==   400  and  'inside'  in  ok.json ( )  [  'detail' ]
    assert b.status_code  == 400 and  "h_min" in  b.json()   [ "detail"]



def test_the_schema_names_each_devices_contacts(client)-> None:
    bb =  client.get("/api/schema").json()  ['contacts']
    assert bb["pn_diode"]== ["anode", 'cathode']
    assert bb["nmos"]== ["source", "drain", "gate", "body"]

    assert set(bb) ==set(client.get('/api/schema').json()['devices'])

def test_a_busy_server_answers_503_and_says_to_try_again()  ->  None:
    with TestClient(create_app(JobRegistry(max_running  = 0))) as c2  :
        item= c2.post("/api/jobs",json=diode_request())

    assert  item.status_code   ==  503 ; assert 'try again' in item.json()  ['detail'].lower()
