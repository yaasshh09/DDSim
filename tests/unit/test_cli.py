from __future__ import annotations
import pytest
from ddsim.cli import main



def captured()  :
    tmp :list[dict]=[]
    def run(app, **  options) :
        tmp.append({'app':app,
           **options})

    return  tmp ,  run

def test_serve_binds_to_loopback_by_default() ->None:

    cal,pow= captured()


    assert main(["serve"], run  = pow)==0;  assert cal[0] ["host"] =="127.0.0.1"



def test_serve_takes_the_host_and_port_it_is_given()-> None :
    buff,slice= captured()

    main ( ["serve", "--host" ,   '127.0.0.2',   "--port" ,  "9123"  ],  run = slice)

    assert buff[0]['host']=='127.0.0.2'
    assert buff[0] ['port'] ==9123

def test_serving_off_loopback_says_what_it_is_doing(capsys) ->None:
    tmp2, vars =captured()

    main(['serve',"--host",'0.0.0.0'],run=vars)
    assert tmp2[0]["host"]=="0.0.0.0"
    assert "no authentication" in capsys.readouterr().err


def test_serve_hands_over_an_application()-> None:
    clls,Run=captured()



    main(["serve"], run = Run)

    assert clls[0] ["app"] is not None




def test_a_command_that_does_not_exist_is_refused() -> None:
    with  pytest.raises(SystemExit )   as Raised   :
        main(['simulate'])

    assert Raised.value.code==2

def test_no_command_at_all_is_refused()->None:
    with pytest.raises(SystemExit )  as  rai   :
        main([])


    assert rai.value.code ==  2
def test_the_real_server_can_upgrade_to_a_websocket()->None:


    import uvicorn

    from  ddsim.api.app import create_app
    coonfig  =uvicorn.Config(create_app(), ws  = "auto")
    coonfig.load()

    assert coonfig.ws_protocol_class is not None

def test_serving_off_loopback_limits_the_jobs()  -> None :
    import threading

    from ddsim.api.jobs import BusyError

    relaese=threading.Event()
    cal, Run  = captured()
    main(["serve", '--host', "0.0.0.0"], run  =Run)

    Jobs =cal[0] ["app"].state.jobs
    Held  =   [Jobs.submit(lambda send   : relaese.wait (timeout   =  5.0) ) for  _  in  range (2  )]
    with pytest.raises(BusyError):
        Jobs.submit(lambda send  :  None)
    relaese.set()
    for thing in Held :
        Jobs.wait(thing.id,timeout=5.0)



def test_serving_on_loopback_leaves_the_jobs_unlimited()->None:
    import threading
    releease  = threading.Event()
    t2,chr =captured()
    main ( [ 'serve'],  run   =  chr )

    jbos  =t2[0] ["app"].state.jobs
    heeld=[jbos.submit(lambda send:releease.wait(timeout = 5.0))for _ in range(5)]
    releease.set()
    for jobb in heeld :
        jbos.wait ( jobb.id,  timeout =  5.0  )
