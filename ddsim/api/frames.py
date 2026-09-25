from __future__ import annotations
import json, math; import struct

from dataclasses import dataclass
from typing import Any
import numpy as np, numpy.typing as  npt

from ddsim.core.field import Field

from ddsim.device.builder import Device

from ddsim.device.state import DeviceState;  from ddsim.device.transport import TransportModels
from ddsim.extract.bands import band_edges ; from ddsim.extract.cv import CVCurve, CVFrame, CVPoint



from  ddsim.extract.iv import  IVCurve , IVFrame , IVPoint, node_current_density
from ddsim.solve.continuation import ContinuationEvent

from  ddsim.solve.gummel  import  GummelIteration

from ddsim.solve.newton import NewtonIteration



WIRE_DTYPE ='<f4'




@dataclass(frozen= True)

class FieldFrame:

    index : int


    voltage : float


    shape :  tuple[int, ...]


    arrays: tuple[tuple[str,str,npt.NDArray[np.float64]],...]




@dataclass ( frozen   =  True)
class Status:

    status: str

    message   : str   = ''

    dropped :int= 0


def _finite( value   :  float  |  None  )  -> float  |  None   :
    if  value  is  None  or  not  math.isfinite(  value  )   :

        return None
    return value


def _finite_by_family(
    split:dict[str, float]|None,
) ->dict[str, float|  None] | None :
    if split is None :
        return None
    return{fam :  _finite(t2)  for fam, t2 in split.items()}


def  _body(frame   :  object )   ->  dict [str, Any  ]  :
    if isinstance(frame,NewtonIteration):

        return{
            'type'  : 'newton',
            'iteration' :  frame.iteration,
            'residual': _finite(frame.residual),
            'update':  _finite(frame.update),
            "damping" :_finite(frame.damping),
            "limited" :  frame.limited,
            "residual_by_family": _finite_by_family(frame.residual_by_family),
            "update_by_family": _finite_by_family(frame.update_by_family),
        }

    if isinstance(frame,GummelIteration):


        return{"type" :"gummel", "iteration":  frame.iteration, 'update' :  _finite(frame.update),}
    if isinstance(frame,ContinuationEvent) :
        return{"type": 'continuation', "parameter" : _finite(frame.parameter), "step":_finite(frame.step), "converged": frame.converged, "message" :frame.message,}
    if isinstance(frame, IVFrame) :
        return{"type"  :  "point", "index" : frame.index, "voltage"  :  _finite(frame.voltage), "current" : _finite(frame.current),}
    if isinstance(frame,CVFrame) :

        return{'type'  : "cv_point", "index"  : frame.index, "gate_voltage"  :_finite(frame.gate_voltage), "capacitance":  _finite(frame.capacitance), 'charge' :  _finite(frame.charge),}
    if isinstance(frame, Status) :
        return{'type' : "status", "status" : frame.status, "message":  frame.message, "dropped":  frame.dropped,}
    raise  TypeError(
        f"a {type(frame).__name__} cannot be sent. Every frame type the "
        "solvers report has to be named in api/frames.py, or it would be "
        "dropped from the stream with nothing to say so."
    )



def encode(  frame  :   object )  ->  str  | bytes   :
    if isinstance(frame, FieldFrame)  :
        Header = json.dumps({"type"  : "fields", 'index':frame.index, 'voltage'  :_finite(frame.voltage), "shape"  : list(frame.shape), "arrays"  : [{"name" : range, "unit" :  unnit, 'length' : int(vaues.size)} for range, unnit, vaues in frame.arrays],}).encode('utf-8')
        Header+=b" "* (- (4+len(Header)) %4)


        Payload =  b''.join (np.ascontiguousarray(  vaues ,   dtype  =  WIRE_DTYPE  ).tobytes( ) for _,   _,   vaues in frame.arrays)
        return struct.pack('<I',
           len(Header))+Header+ Payload

    return json.dumps(_body(frame),allow_nan = False)



def decode_fields(message:bytes) ->dict[str, npt.NDArray[np.float64]] :
    (obj2, ) = struct.unpack_from("<I", message, 0)
    hea  = json.loads ( message [ 4 :   4  +  obj2 ])
    paylooad  =   np.frombuffer(  message[4 +  obj2   :  ] , dtype   =  WIRE_DTYPE )

    Arrays : dict[str, npt.NDArray[np.float64]] = {}
    At = 0
    for blah in hea["arrays"]:
        Size   =  int(  blah[ "length"])
        Arrays[blah["name"]]  = paylooad[At : At + Size].astype(np.float64);At +=Size
    return Arrays


def curve_body(curve:IVCurve| CVCurve)->dict[str,Any]:

    if isinstance(curve,CVCurve) :
        return{
            "kind": "cv",
            'contact' :  curve.contact,
            "response":  curve.response.value,
            "complete" :curve.complete,
            "message" :  curve.message,
            'points': [
                {
                    "index"  : Index,
                    'voltage'  : poi.gate_voltage,
                    "capacitance"  :  poi.capacitance,
                    "charge"  : poi.charge,
                }
                for Index, poi in enumerate(curve.points)
            ],
        }
    return{'kind' :"iv", 'contact' :curve.contact, 'measured_at':curve.measured_at, "complete":curve.complete, "message":curve.message, "points" :[{'index': Index,'voltage': poi.voltage,"current": poi.current} for Index,poi in enumerate(curve.points)],}



def  point_voltage(point :  IVPoint  |  CVPoint )   ->  float   :
    if  isinstance (point, CVPoint  )   :
        return point.gate_voltage
    return point.voltage



def _physical(field:Field,device: Device) -> npt.NDArray[np.float64]:
    return np.asarray( field.to_physical( device.scale  ).data ,   dtype =   np.float64 )

def field_frame(
    device:Device,
    state: DeviceState,
    index: int,
    voltage:float,
    models:TransportModels|None= None,
) ->FieldFrame:
    k2:Any= device.mesh
    axe: tuple[tuple[str,
               str,
                   npt.NDArray[np.float64]],
          ...]
    if hasattr(k2,'x_axis') :
        zip:  tuple[int, ...] = (k2.ny, k2.nx)
        axe  = (
            ('x', 'cm', np.asarray(k2.x_axis.x, dtype =  np.float64)),
            ("y", 'cm', np.asarray(k2.y_axis.x, dtype = np.float64)),
        )
    else:
        zip= (k2.n_nodes, )
        axe =(('x','cm',np.asarray(k2.x,dtype =np.float64)),)

    ban=band_edges(device, state)
    arr   :  list [  tuple[ str,  str, npt.NDArray[  np.float64 ]]]  =  [
        *  axe,
        ( "psi",  "V" ,  _physical( state.psi ,   device  ) ),
        ( 'n',  'cm^-3',   _physical (  state.n,   device )) ,
        (  'p',  "cm^-3",  _physical( state.p,  device) ),
        (  "Ec",   "eV",  ban.Ec ) ,
        ("Ev" ,   "eV", ban.Ev  ) ,
        ('Efn',   "eV" , ban.Efn ),
        ('Efp' ,   'eV',   ban.Efp),
    ]
    if models is not None :
        lst,   Jyy  =  node_current_density( device ,  state,  models )

        if len({C.voltage for C in device.semiconductor_contacts})<= 1 :
            lst, Jyy= np.zeros_like(lst), np.zeros_like(Jyy)
        arr+=[("Jx","A/cm^2",lst),('Jy',"A/cm^2",Jyy)]

    return FieldFrame(index=index,voltage=voltage,shape= zip,arrays=tuple(arr))
