from __future__ import annotations
from  collections.abc  import AsyncIterator
from contextlib import asynccontextmanager
from  dataclasses import  dataclass
from pathlib import Path
from typing import Any
import anyio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect;  from fastapi.responses  import  FileResponse, Response
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles
from  ddsim.api.devices  import (COARSE, DEVICE_KINDS, build_from_spec, contact_names, device_dimension, device_parameters, drawing_defaults, region_defaults,)
from ddsim.api.frames import(
    Status,
    curve_body,
    encode,
    field_frame,
    point_voltage,
)
from ddsim.api.jobs import BusyError, JobRegistry, JobStatus, Send
from ddsim.api.learn import (KNOB_LABELS, KNOB_TOPICS, PLOT_TOPICS , STATUS_TOPICS , lesson_names, load_lesson , load_topic , topic_names,)
from ddsim.api.sweeps import(
    SWEEP_KINDS,
    check_request,
    model_parameters,
    run_sweep,
    sweep_parameters,
)
from ddsim.device.builder import Device
from  ddsim.device.drawing  import NODE_BUDGET
from ddsim.device.transport  import  TransportModels; from ddsim.extract.cv import CVCurve
from ddsim.extract.iv  import IVCurve


SHUTDOWN_TIMEOUT=30.0


PAGE =Path(__file__).parent  /'static'/ "index.html"


class _Revalidated(StaticFiles):

    async def get_response(self, path  :  str, scope :  Any) -> Response:
        d = await super().get_response(path, scope)
        d.headers["Cache-Control"] ='no-cache'
        return d
_QUIET_POLL = 0.25


_TERMINAL  = (JobStatus.DONE,
               JobStatus.FAILED,
      JobStatus.CANCELLED)
_ENDED   =  object( )

_NOTHING_YET= object()
class DeviceSpec(BaseModel) :
    kind : str
    parameters :  dict[str, Any]  =Field(default_factory  =dict)

class SweepSpec(BaseModel):


    kind :str
    contact:str
    voltages:list[float]
    settings:dict[str,Any] = Field(default_factory= dict)
    models: dict[str, Any] = Field(default_factory  =  dict)
    measure_at:str |None=None

class JobRequest(BaseModel) :

    device:DeviceSpec
    sweep: SweepSpec

@dataclass(frozen = True)


class Finished:


    device  : Device

    curve:IVCurve| CVCurve
    models :  TransportModels  | None
def create_app(registry :JobRegistry|None =None)-> FastAPI:

    c=registry if registry is not None else JobRegistry()

    @asynccontextmanager

    async def lifespan(_ :FastAPI)-> AsyncIterator[None]:
        yield
        c.close(timeout= SHUTDOWN_TIMEOUT)
    j  =  FastAPI(title =  'DDSim', lifespan  =   lifespan  )
    j.state.jobs =  c
    j.mount('/static',_Revalidated(directory=PAGE.parent),name='static')

    @j.get("/api/schema")
    def schema() -> dict[str, Any] :
        return{'devices' : {el  : [ _knob(  p  )  for  p  in  device_parameters(el )] for el in DEVICE_KINDS}, "dimensions"  :  {el   :  device_dimension (el  ) for el in  DEVICE_KINDS}, 'contacts'   :  {el  : list (  contact_names( el)  )  for  el  in  DEVICE_KINDS}, 'regions'   : {el   : h for  el  in  DEVICE_KINDS if ( h  :=   region_defaults(el))  is  not  None}, 'drawings'   :  {el  : c2 for el in  DEVICE_KINDS if( c2  :=  drawing_defaults(  el))  is  not None}, "node_budget"   : NODE_BUDGET, 'presets'  : {el :   {'parameters'  :   dict(  k2.parameters  ),  'note'   :  k2.note } for  el ,   k2 in  COARSE.items (  )}, "sweeps"  :   {el   : [ _knob(  p ) for  p in sweep_parameters (el  )  ]   for  el  in SWEEP_KINDS}, "models" :   [_knob(p )  for  p  in model_parameters()  ], 'plots'  :   dict (PLOT_TOPICS  ) , "statuses"  :   dict(STATUS_TOPICS  ),}
    @j.get('/api/learn')
    def  topics( ) ->  list [  dict[  str,   str  ]  ] :
        return[
            {'name': t.name,'title':t.title,'summary': t.summary}
            for t in(load_topic(row)for row in topic_names())
        ]
    @j.get('/api/learn/{name}')
    def topic(name  :  str)  ->  dict[str, Any] :
        kk=_found(lambda :load_topic(name))
        return{"name" :  kk.name, 'title'   :  kk.title , "summary" :  kk.summary, 'docs'  :   list(kk.docs  ), 'plain' :   kk.plain, "depth"  :   kk.depth, "knobs"   :  sorted(m for  m ,   lst  in  KNOB_TOPICS.items( )  if  lst == kk.name ),}

    @j.get(  "/api/lessons" )
    def lessons()  -> list[dict[str, str]] :
        return [
            {"name"  :   dat.name,  "title" :   dat.title, 'summary'   : dat.summary  }
            for dat in(load_lesson(res2  )   for  res2  in lesson_names(  ))
        ]


    @j.get("/api/lessons/{name}")
    def lesson(name :str)->  dict[str, Any] :
        i = _found(lambda :load_lesson(name))
        return{
            "name": i.name,
            'title':i.title,
            "summary":i.summary,
            'claims':list(i.claims),
            "request":i.request,
            'mesh_note' :i.mesh_note,
            'steps' :[
                {"title":s.title,"text": s.text,"request": s.request}
                for s in i.steps
            ],
            "look_for" : i.look_for,
            "explanation":i.explanation,
        }
    @j.post('/api/devices/check')
    def  check( device   :  DeviceSpec  ) ->  dict[  str,   bool ]  :
        _checked (lambda  :  build_from_spec(device.kind , device.parameters)  )

        return{  "builds"  :  True }

    @j.post('/api/jobs')

    def  submit(request   :   JobRequest  )   ->  dict [ str,  str  ]  :
        z2   = _checked (
            lambda   : build_from_spec(  request.device.kind,  request.device.parameters )
        )
        nxt  = request.sweep
        _checked(lambda :  check_request (nxt.kind, z2 , nxt.contact, nxt.settings , nxt.models, nxt.measure_at,))

        def work(send : Send) ->  Finished  :
            jj,v2= run_sweep(nxt.kind, z2, nxt.contact, nxt.voltages, settings =nxt.settings, models=nxt.models or None, measure_at = nxt.measure_at, on_frame=send,)
            return Finished(device  = z2, curve =jj, models= v2)
        try:
            return{'id' : c.submit(work).id}
        except BusyError as d2:
            raise HTTPException(status_code =  503, detail =  str(d2)) from d2

    @j.get('/api/jobs/{job_id}')
    def  status (job_id :  str  )  ->  dict[str,  Any  ]  :
        return{
            "id": job_id,
            "status" :_found(lambda:c.status(job_id)).value,
            "message" : c.message(job_id),
            'dropped' :c.dropped(job_id),
        }
    @j.post('/api/jobs/{job_id}/cancel')
    def cancel(job_id : str)  ->dict[str, bool] :
        return{"cancelled" : _found(lambda: c.cancel(job_id))}

    @j.get('/api/jobs/{job_id}/result')


    def  result(  job_id :   str )  -> dict[ str,   Any ]  :
        return curve_body(_finished(c,job_id).curve)
    @j.get( "/api/jobs/{job_id}/fields/{index}" )
    def fields(job_id :str, index  :int) -> Response :
        w2   =  _finished (c ,   job_id )
        res= w2.curve.points
        if not 0<= index < len(res):
            raise HTTPException(
                status_code =404,
                detail  = (
                    f"this sweep reached {len(res)} points, so there is no "
                    f"point {index} to show a field at"
                ),
            )
        out2   = res[  index ]
        s2  = point_voltage(out2 )
        zz  =encode(field_frame(w2.device.with_bias(**  {w2.curve.contact : s2}), out2.state, index = index, voltage =  s2, models =w2.models,))
        assert  isinstance( zz,   bytes  )


        return Response(content=zz,media_type='application/octet-stream')

    @j.websocket("/api/jobs/{job_id}/stream")
    async  def stream(  socket  : WebSocket,   job_id  :   str)   ->  None :
        await socket.accept()
        try :
            c.status(job_id)
        except KeyError as r2 :
            await socket.close(code= 1008, reason = str(r2))
            return

        try :
            while True  :
                it =await anyio.to_thread.run_sync(_poll,c,job_id)
                if  it  is _ENDED  :
                    break
                if it is _NOTHING_YET:
                    if c.status(job_id)in _TERMINAL:
                        break
                    continue
                await _send (  socket,  encode( it)  )
            await _send (socket, encode(Status(status  =  c.status( job_id).value, message =  c.message (  job_id), dropped  =   c.dropped(  job_id  ),)) ,)

        except WebSocketDisconnect :
            return

    @j.get ( '/')
    def  page( )  -> FileResponse  :

        return FileResponse(
            PAGE,media_type='text/html',headers={'Cache-Control': 'no-cache'}
        )
    return  j




def _knob(parameter: Any) ->dict[str, Any]:

    return{
        "name" : parameter.name,
        'label': KNOB_LABELS.get(parameter.name, parameter.name),
        'default' :  parameter.default,
        'type':  parameter.type,
        "choices" :list(parameter.choices),
        'explanation'  : parameter.explanation,
        'unit': parameter.unit,
        "topic" : KNOB_TOPICS.get(parameter.name, ""),
        "low"  :  parameter.low,
        "high"  :parameter.high,
        "axis"  : parameter.axis,
    }


def _checked (call :   Any) -> Any :
    try :
        return call()
    except (  ValueError,  TypeError, KeyError  )   as  u   :
        row  =u.args[0]if u.args else str(u)

        raise  HTTPException(status_code   =  400 ,  detail  =  str(  row )  ) from  u

def  _found(  call   : Any ) -> Any  :
    try :
        return call()
    except  KeyError  as ii   :
        raise HTTPException(status_code=404,detail =str(ii))from ii

def _finished(jobs : JobRegistry, job_id : str)  ->Finished:
    s2 =_found(lambda  :  jobs.result(job_id))
    if s2 is None  :
        raise HTTPException(
            status_code  =   409,
            detail = (
                f"job {job_id} is {jobs.status(job_id).value} and has no "
                f"curve to show. {jobs.message(job_id)}".strip(  )
            ),
        )

    assert isinstance(s2, Finished)

    return s2

def _poll(jobs: JobRegistry,
          job_id:str) ->Any:


    try   :
        return  next(  jobs.frames ( job_id,   timeout =  _QUIET_POLL)  )

    except StopIteration  :

        return _ENDED
    except TimeoutError:
        return  _NOTHING_YET

async def _send(socket  :WebSocket, message  :str  |bytes)  -> None  :
    if isinstance(message,str) :
        await socket.send_text(message)
    else   :
        await  socket.send_bytes(  message )
