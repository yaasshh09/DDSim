from __future__ import annotations
import inspect; import re, json
from pathlib import Path
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from enum import Enum
from functools import cache
from typing import Any
from ddsim.device.builder import Device
from ddsim.device.drawing import NODE_BUDGET, Block, Electrode, Implant , drawing ; from ddsim.device.mos_cap import mos_cap
from  ddsim.device.mosfet  import  nmos
from ddsim.device.pn_diode import pn_diode
from ddsim.device.stack import Region, stack
DEVICE_KINDS: dict[str,Callable[...,Device]]= {"pn_diode": pn_diode, "mos_cap" :mos_cap, 'nmos' :nmos, "stack": stack, "drawing":drawing,}

_EXPRESSIBLE  =   ("float",  'int', 'bool' ,  "str" )

@dataclass(frozen= True)



class Preset :
    parameters: dict[str, float | int]

    note   : str
COARSE  :dict[str, Preset]= {
    "mos_cap" : Preset(
        parameters= {'n_silicon' :41, "n_oxide":  3, "h_min" :2e-7},
        note =(
            'A coarse mesh: 129 nodes against the converged 375. Measured '
            "2026-09-17 over a -2 V to 2 V C-V, the capacitance reads 0.556 "
            'percent high in accumulation and 0.714 percent high in '
            "depletion, and the sweep solves 1.8 times faster. The shape of "
            'the curve is the same; the numbers on it are not the validated '
            "ones."
        ),
    ),
    "nmos"  :Preset(
        parameters =  {
            "n_contact" :4,
            "n_sd"  :10,
            'n_channel'  :12,
            'n_silicon':  29,
            'n_oxide' :4,
            "h_min_x"  :  5e-7,
            "h_min_y"  :  1e-7,
        },
        note =(
            "A coarse mesh: 1504 nodes against the converged 8379. Measured "
            "2026-09-17 over a 0 V to 1.2 V transfer at 50 mV drain, the "
            "drain current at 1.2 V reads 0.621 percent high and the "
            "extrapolated threshold moves 0.7 mV, for a sweep that takes "
            "5.1 s instead of 20.4 s. Good enough to watch a MOSFET switch, "
            'not the mesh any number in the README was taken on.'
        ),
    ),
    "drawing"  : Preset(
        parameters  ={"nx"  :  39, 'ny': 35, "h_min_x" :  5e-7, 'h_min_y' : 1e-7},
        note= (
            'A coarse mesh: 1365 nodes against the converged 8379. Measured '
            "2026-09-18 on the default drawing, the benchmark nmos, over a 0 V "
            "to 1.2 V transfer at 50 mV drain: the drain current reads 0.75 "
            "percent high at 1.2 V and up to 3.9 percent high between 0.5 and "
            '0.7 V, and the extrapolated threshold moves 1.5 mV, for a sweep '
            'that takes 4.0 s instead of 18.9 s. A drawing of your own was not '
            'measured and can be further off.'
        ),
    ),
}


def node_count (device  : Device  )  ->  int :

    return device.mesh.n_nodes



@dataclass(frozen=True)


class Parameter  :


    name: str

    default : float |int |bool | str

    type :str


    choices : tuple[str, ...] =()

    explanation :str= ""

    unit :str  =  ""



    low  : float  | None  =  None

    high :  float | None  = None


    axis  : str  =   "linear"




def _builder(kind : str)-> Callable[..., Device]:
    if  kind  not in  DEVICE_KINDS  :
        g =', '.join(sorted(DEVICE_KINDS))
        raise ValueError(f"unknown device kind {kind!r}. Known kinds: {g}")
    return DEVICE_KINDS[kind]
_KNOBS: dict[str, dict[str, str]] = json.loads(Path(__file__).with_name("knobs.json").read_text(encoding='utf-8'))

_UNIT = re.compile(r"\[([^\]]+)\]")

_NUMBER =  r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"



_RANGE   =  re.compile(rf"Range\s+({_NUMBER})\s+to\s+({_NUMBER})(,\s*log)?" )



def argument_docs(function  :  Callable [ ...,   Any  ]  ) ->   dict[str,   str]  :
    return dict(_KNOBS.get(f"{getattr(function,'__module__','')}.{getattr(function, '__qualname__', '')}",{}))



def _unit_of(explanation  :str)  ->  str:
    thing  =   _UNIT.search(explanation )
    return thing.group(1) if thing else ""


def _range_of(explanation :str) -> tuple[float |None,float| None,str]:
    it  = _RANGE.search(explanation)

    if it is None  :
        return None,None,'linear'
    return(float( it.group(  1 )  ) , float(it.group( 2 )  ) , 'log' if  it.group ( 3  )   else  "linear",)


@cache


def device_dimension(kind:str) ->int :
    return 2 if hasattr(_builder(kind)  ().mesh, "ny")  else 1
@cache


def contact_names(kind: str)-> tuple[str,...]:
    return tuple(t.name for t in _builder(kind)().contacts)

def parameters_of(function: Callable[...,Any],choices :dict[str,tuple[str,...]]|None=None, docs: dict[str, str] | None=None)->tuple[Parameter,...]:

    m2  = choices or{  }

    u= argument_docs(function) if docs is None else docs
    s:list[Parameter] = []

    def described( name  :  str ,   **   rest :  Any)   -> Parameter  :
        h=u.get(name,'')
        num, vals, j=  _range_of(h)
        return Parameter(
            name = name,
            explanation= h,
            unit= _unit_of(h),
            low =  num,
            high  =  vals,
            axis = j,
            ** rest,
        )
    for  i,  m in  inspect.signature (function ).parameters.items()   :

        if m.default is inspect.Parameter.empty   :
            continue
        if isinstance(m.default, Enum) :
            s.append(
                described (
                    i,
                    default   =  m.default.value,
                    type =  'str',
                    choices =  tuple(
                        str(  bb.value  )  for bb in type( m.default  )
                    ) ,
                )
            )
        elif str(m.annotation) in _EXPRESSIBLE  :
            s.append(
                described(
                    i,
                    default= m.default,
                    type = str(m.annotation),
                    choices=m2.get(i, ()),
                )
            )
    return tuple(s)



def  enum_arguments(function  : Callable[ ..., Any]  )   -> dict[ str ,  type[ Enum]] :
    return{hh   :  type( e.default  ) for  hh , e  in inspect.signature(  function  ).parameters.items () if isinstance (  e.default ,   Enum  )}

def as_enum (  what  :   str, name :   str,   kind  : type[  Enum], value :   Any )  ->   Enum   :
    try  :
        return kind(value)
    except ValueError as a:
        c=', '.join(str(y.value)for y in kind)
        raise ValueError(
            f"{what}.{name} has no setting {value!r}. Settings: {c}"
        )from a
def device_parameters( kind  :   str)  ->  tuple [ Parameter, ... ] :
    return parameters_of(_builder(kind))


def checked_arguments(what:  str, offered  : dict[str, Parameter], sent  :dict[str, Any],) -> dict[str, float  | int | bool |  str] :
    w :dict[str, float |  int | bool | str]  =  {}

    for j,u in sent.items() :
        if j not in offered :
            val2   =  ", ".join (offered )
            raise ValueError(
                f"{what} has no settable parameter {j!r}. Settable: {val2}"
            )

        w[j]=_checked(what, offered[j], u)
    return w


RECORDS:dict[str, type]= {
    'regions' :Region,
    'blocks'  : Block,
    'implants' : Implant,
    "electrodes" :  Electrode,
}


DRAWING_PARTS = ("blocks",'implants','electrodes')


def record_defaults(kind:str,name : str)->list[dict[str,Any]]|None:
    w  = inspect.signature(_builder(kind)).parameters.get(name)
    if w is None:
        return None
    return[asdict(v) for v in w.default]


def region_defaults ( kind  :  str )   ->  list[dict [  str,  Any  ]]   | None  :
    return record_defaults(kind, "regions")


def drawing_defaults(kind :  str) ->  dict[str, list[dict[str, Any]]] | None :
    if record_defaults(kind,'blocks')is None:
        return None

    return{b :record_defaults(kind, b)or[]  for b in DRAWING_PARTS}

def records_from_json(name:str, sent : Any)->tuple[Any, ...] :

    tmp2  = RECORDS[ name  ]
    c = name[:- 1]
    x={m.name: m.type for m in fields(tmp2)}
    if not isinstance(sent, list)  :
        raise TypeError(f"{name} is a list of {name}, got {type(sent).__name__}")
    mm =[]
    for  yy,   d in  enumerate ( sent, start  =  1  )   :
        if not isinstance(d, dict) :
            raise TypeError(
                f"{c} {yy} is {type(d).__name__}, not an object with "
                f"{', '.join(x)}"
            )
        w= [m for m in x if m not in d]
        row  = [ m for  m in d  if m not in  x ]
        if w or row :
            raise ValueError(
                f"{c} {yy} has the fields {', '.join(x)}; "
                f"missing {w}, not a field {row}"
            )
        for m, dd  in x.items( )  :

            vals = d[m]
            i=(type(vals)is str if dd=="str" else type(vals)in(int,float))
            if not i:
                raise TypeError(
                    f"{c} {yy}: {m} is a "
                    f"{'name' if dd == 'str' else 'number'}, got "
                    f"{type(vals).__name__}"
                )
        mm.append(tmp2(**  {m : d[m] if dd == "str" else float(d[m]) for m, dd in x.items()}))
    return tuple(mm)

def regions_from_json (sent : Any )  -> tuple[ Region,  ...]  :
    return  records_from_json('regions',
                 sent  )



def build_from_spec(kind:str,parameters:dict[str,Any])-> Device:
    print('building', kind)  # debug, take out later
    t2= {p.name :p for p in device_parameters(kind)}
    u =  dict( parameters  )
    w2 :  dict[str, Any] = {}
    for j  in RECORDS  :
        if  j  in  u  :
            if record_defaults(kind,j)is None :
                raise ValueError(f"{kind} is not built from {j}")


            w2[j]=records_from_json(j,u.pop(j))
    g =   checked_arguments(  kind, t2,   u)
    for j,k2 in g.items() :
        if  type(  k2  )  is  int and k2  >   NODE_BUDGET :
            raise ValueError (
                f"{kind}.{j} = {k2} is over the node budget of {NODE_BUDGET}"
            )
    buf = _builder(kind)(**g,**w2)
    if  node_count( buf)  >  NODE_BUDGET  :
        raise ValueError(
            f"this {kind} mesh is {node_count(buf)} nodes, over the budget of "
            f"{NODE_BUDGET}. Use fewer nodes along one axis, or the coarse mesh."
        )
    return buf



def _checked(kind: str, parameter : Parameter, value:Any) -> float |int | bool  |  str:
    if parameter.type== 'bool':
        if type(value) is not bool :
            raise TypeError(
                f"{kind}.{parameter.name} is a boolean, got {type(value).__name__}"
            )
        return value
    if parameter.type  ==  "int" :
        if  type (value)  is not  int   :
            raise TypeError(
                f"{kind}.{parameter.name} is an integer, got {type(value).__name__}"
            )
        return value
    if parameter.type== 'str' :
        if  type(  value ) is  not  str  :
            raise TypeError(
                f"{kind}.{parameter.name} is a name, got {type(value).__name__}"
            )
        return value
    if type(value )   is int   :
        return float(value)

    if type(value)is not float:
        raise TypeError(
            f"{kind}.{parameter.name} is a number, got {type(value).__name__}"
        )
    return value
