from __future__ import annotations
import pytest
from ddsim.cli import main



def captured()  :
    tt :list[dict]=[]
    def run(app, **  options) :
        tt.append({'app':app,
           **options})

    return  tt ,  run

def test_serve_binds_to_loopback_by_default() ->None:

    a2,xs= captured()


    assert main(["serve"], run  = xs)==0;  assert a2[0] ["host"] =="127.0.0.1"



def test_serve_takes_the_host_and_port_it_is_given()-> None :
    kk,u= captured()

    main ( ["serve", "--host" ,   '127.0.0.2',   "--port" ,  "9123"  ],  run = u)

    assert kk[0]['host']=='127.0.0.2'
    assert kk[0] ['port'] ==9123

def test_serving_off_loopback_says_what_it_is_doing(capsys) ->None:
    r2, stuff =captured()

    main(['serve',"--host",'0.0.0.0'],run=stuff)
    assert r2[0]["host"]=="0.0.0.0"
    assert "no authentication" in capsys.readouterr().err


def test_serve_hands_over_an_application()-> None:
    r,t=captured()



    main(["serve"], run = t)

    assert r[0] ["app"] is not None




def test_a_command_that_does_not_exist_is_refused() -> None:
    with  pytest.raises(SystemExit )   as u   :
        main(['simulate'])

    assert u.value.code==2

def test_no_command_at_all_is_refused()->None:
    with pytest.raises(SystemExit )  as  b   :
        main([])


    assert b.value.code ==  2
def test_the_real_server_can_upgrade_to_a_websocket()->None:


    import uvicorn
    from  ddsim.api.app import create_app
    f  =uvicorn.Config(create_app(), ws  = "auto")
    f.load()

    assert f.ws_protocol_class is not None

def test_serving_off_loopback_limits_the_jobs()  -> None :
    import threading
    from ddsim.api.jobs import BusyError

    i=threading.Event()
    b, t  = captured()
    main(["serve", '--host', "0.0.0.0"], run  =t)

    hh =b[0] ["app"].state.jobs
    ys  =   [hh.submit(lambda send   : i.wait (timeout   =  5.0) ) for  _  in  range (2  )]
    with pytest.raises(BusyError):
        hh.submit(lambda send  :  None)
    i.set()
    for s in ys :
        hh.wait(s.id,timeout=5.0)



def test_serving_on_loopback_leaves_the_jobs_unlimited()->None:
    import threading
    d2  = threading.Event()
    rows,f =captured()
    main ( [ 'serve'],  run   =  f )

    u  =rows[0] ["app"].state.jobs
    v=[u.submit(lambda send:d2.wait(timeout = 5.0))for _ in range(5)]
    d2.set()
    for g in v :
        u.wait ( g.id,  timeout =  5.0  )
