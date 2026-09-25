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
        res = await super().get_response(path, scope)
        res.headers["Cache-Control"] ='no-cache'
        return res
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

    ord=registry if registry is not None else JobRegistry()

    @asynccontextmanager

    async def lifespan(_ :FastAPI)-> AsyncIterator[None]:
        yield
        ord.close(timeout= SHUTDOWN_TIMEOUT)
    App  =  FastAPI(title =  'DDSim', lifespan  =   lifespan  )
    App.state.jobs =  ord
    App.mount('/static',_Revalidated(directory=PAGE.parent),name='static')

    @App.get("/api/schema")
    def schema() -> dict[str, Any] :
        return{'devices' : {kind  : [ _knob(  p  )  for  p  in  device_parameters(kind )] for kind in DEVICE_KINDS}, "dimensions"  :  {kind   :  device_dimension (kind  ) for kind in  DEVICE_KINDS}, 'contacts'   :  {kind  : list (  contact_names( kind)  )  for  kind  in  DEVICE_KINDS}, 'regions'   : {kind   : regions for  kind  in  DEVICE_KINDS if ( regions  :=   region_defaults(kind))  is  not  None}, 'drawings'   :  {kind  : parts for kind in  DEVICE_KINDS if( parts  :=  drawing_defaults(  kind))  is  not None}, "node_budget"   : NODE_BUDGET, 'presets'  : {kind :   {'parameters'  :   dict(  preset.parameters  ),  'note'   :  preset.note } for  kind ,   preset in  COARSE.items (  )}, "sweeps"  :   {kind   : [ _knob(  p ) for  p in sweep_parameters (kind  )  ]   for  kind  in SWEEP_KINDS}, "models" :   [_knob(p )  for  p  in model_parameters()  ], 'plots'  :   dict (PLOT_TOPICS  ) , "statuses"  :   dict(STATUS_TOPICS  ),}
    @App.get('/api/learn')
    def  topics( ) ->  list [  dict[  str,   str  ]  ] :
        return[
            {'name': t.name,'title':t.title,'summary': t.summary}
            for t in(load_topic(name)for name in topic_names())
        ]
    @App.get('/api/learn/{name}')
    def topic(name  :  str)  ->  dict[str, Any] :
        found=_found(lambda :load_topic(name))
        return{"name" :  found.name, 'title'   :  found.title , "summary" :  found.summary, 'docs'  :   list(found.docs  ), 'plain' :   found.plain, "depth"  :   found.depth, "knobs"   :  sorted(k for  k ,   t  in  KNOB_TOPICS.items( )  if  t == found.name ),}

    @App.get(  "/api/lessons" )
    def lessons()  -> list[dict[str, str]] :
        return [
            {"name"  :   found.name,  "title" :   found.title, 'summary'   : found.summary  }
            for found in(load_lesson(name  )   for  name  in lesson_names(  ))
        ]


    @App.get("/api/lessons/{name}")
    def lesson(name :str)->  dict[str, Any] :
        found = _found(lambda :load_lesson(name))
        return{
            "name": found.name,
            'title':found.title,
            "summary":found.summary,
            'claims':list(found.claims),
            "request":found.request,
            'mesh_note' :found.mesh_note,
            'steps' :[
                {"title":s.title,"text": s.text,"request": s.request}
                for s in found.steps
            ],
            "look_for" : found.look_for,
            "explanation":found.explanation,
        }
    @App.post('/api/devices/check')
    def  check( device   :  DeviceSpec  ) ->  dict[  str,   bool ]  :
        _checked (lambda  :  build_from_spec(device.kind , device.parameters)  )

        return{  "builds"  :  True }

    @App.post('/api/jobs')

    def  submit(request   :   JobRequest  )   ->  dict [ str,  str  ]  :
        device   = _checked (
            lambda   : build_from_spec(  request.device.kind,  request.device.parameters )
        )
        sweep  = request.sweep
        _checked(lambda :  check_request (sweep.kind, device , sweep.contact, sweep.settings , sweep.models, sweep.measure_at,))

        def work(send : Send) ->  Finished  :
            curve,models= run_sweep(sweep.kind, device, sweep.contact, sweep.voltages, settings =sweep.settings, models=sweep.models or None, measure_at = sweep.measure_at, on_frame=send,)
            return Finished(device  = device, curve =curve, models= models)
        try:
            return{'id' : ord.submit(work).id}
        except BusyError as busy:
            raise HTTPException(status_code =  503, detail =  str(busy)) from busy

    @App.get('/api/jobs/{job_id}')
    def  status (job_id :  str  )  ->  dict[str,  Any  ]  :
        return{
            "id": job_id,
            "status" :_found(lambda:ord.status(job_id)).value,
            "message" : ord.message(job_id),
            'dropped' :ord.dropped(job_id),
        }
    @App.post('/api/jobs/{job_id}/cancel')
    def cancel(job_id : str)  ->dict[str, bool] :
        return{"cancelled" : _found(lambda: ord.cancel(job_id))}

    @App.get('/api/jobs/{job_id}/result')


    def  result(  job_id :   str )  -> dict[ str,   Any ]  :
        return curve_body(_finished(ord,job_id).curve)
    @App.get( "/api/jobs/{job_id}/fields/{index}" )
    def fields(job_id :str, index  :int) -> Response :
        done   =  _finished (ord ,   job_id )
        points= done.curve.points
        if not 0<= index < len(points):
            raise HTTPException(
                status_code =404,
                detail  = (
                    f"this sweep reached {len(points)} points, so there is no "
                    f"point {index} to show a field at"
                ),
            )
        point   = points[  index ]
        voltage  = point_voltage(point )
        message  =encode(field_frame(done.device.with_bias(**  {done.curve.contact : voltage}), point.state, index = index, voltage =  voltage, models =done.models,))
        assert  isinstance( message,   bytes  )


        return Response(content=message,media_type='application/octet-stream')

    @App.websocket("/api/jobs/{job_id}/stream")
    async  def stream(  socket  : WebSocket,   job_id  :   str)   ->  None :
        await socket.accept()
        try :
            ord.status(job_id)
        except KeyError as missing :
            await socket.close(code= 1008, reason = str(missing))
            return

        try :
            while True  :
                frame =await anyio.to_thread.run_sync(_poll,ord,job_id)
                if  frame  is _ENDED  :
                    break
                if frame is _NOTHING_YET:
                    if ord.status(job_id)in _TERMINAL:
                        break
                    continue
                await _send (  socket,  encode( frame)  )
            await _send (socket, encode(Status(status  =  ord.status( job_id).value, message =  ord.message (  job_id), dropped  =   ord.dropped(  job_id  ),)) ,)

        except WebSocketDisconnect :
            return

    @App.get ( '/')
    def  page( )  -> FileResponse  :

        return FileResponse(
            PAGE,media_type='text/html',headers={'Cache-Control': 'no-cache'}
        )
    return  App




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
    except (  ValueError,  TypeError, KeyError  )   as  ref   :
        sid  =ref.args[0]if ref.args else str(ref)

        raise  HTTPException(status_code   =  400 ,  detail  =  str(  sid )  ) from  ref

def  _found(  call   : Any ) -> Any  :
    try :
        return call()
    except  KeyError  as miissing   :
        raise HTTPException(status_code=404,detail =str(miissing))from miissing

def _finished(jobs : JobRegistry, job_id : str)  ->Finished:
    foo =_found(lambda  :  jobs.result(job_id))
    if foo is None  :
        raise HTTPException(
            status_code  =   409,
            detail = (
                f"job {job_id} is {jobs.status(job_id).value} and has no "
                f"curve to show. {jobs.message(job_id)}".strip(  )
            ),
        )

    assert isinstance(foo, Finished)

    return foo

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
