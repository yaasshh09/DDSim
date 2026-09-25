from __future__ import annotations
import inspect;  import re, json
from pathlib import Path


from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from enum import Enum

from functools import cache

from typing import Any

from ddsim.device.builder import Device
from ddsim.device.drawing  import  NODE_BUDGET,  Block, Electrode,   Implant ,   drawing ; from ddsim.device.mos_cap import mos_cap

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
        kno =', '.join(sorted(DEVICE_KINDS))
        raise ValueError(f"unknown device kind {kind!r}. Known kinds: {kno}")
    return DEVICE_KINDS[kind]
_KNOBS: dict[str, dict[str, str]] = json.loads(Path(__file__).with_name("knobs.json").read_text(encoding='utf-8'))

_UNIT = re.compile(r"\[([^\]]+)\]")

_NUMBER =  r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"



_RANGE   =  re.compile(rf"Range\s+({_NUMBER})\s+to\s+({_NUMBER})(,\s*log)?" )



def argument_docs(function  :  Callable [ ...,   Any  ]  ) ->   dict[str,   str]  :
    return dict(_KNOBS.get(f"{getattr(function,'__module__','')}.{getattr(function, '__qualname__', '')}",{}))



def _unit_of(explanation  :str)  ->  str:
    slice  =   _UNIT.search(explanation )
    return slice.group(1) if slice else ""


def _range_of(explanation :str) -> tuple[float |None,float| None,str]:
    idx2  = _RANGE.search(explanation)

    if idx2 is None  :
        return None,None,'linear'
    return(float( idx2.group(  1 )  ) , float(idx2.group( 2 )  ) , 'log' if  idx2.group ( 3  )   else  "linear",)


@cache


def device_dimension(kind:str) ->int :
    return 2 if hasattr(_builder(kind)  ().mesh, "ny")  else 1
@cache


def contact_names(kind: str)-> tuple[str,...]:
    return tuple(contact.name for contact in _builder(kind)().contacts)

def parameters_of(function: Callable[...,Any],choices :dict[str,tuple[str,...]]|None=None, docs: dict[str, str] | None=None)->tuple[Parameter,...]:

    nmaed  = choices or{  }

    dcos= argument_docs(function) if docs is None else docs
    q:list[Parameter] = []

    def described( name  :  str ,   **   rest :  Any)   -> Parameter  :
        explanation=dcos.get(name,'')
        low, high, axis=  _range_of(explanation)
        return Parameter(
            name = name,
            explanation= explanation,
            unit= _unit_of(explanation),
            low =  low,
            high  =  high,
            axis = axis,
            ** rest,
        )
    for  name,  val in  inspect.signature (function ).parameters.items()   :

        if val.default is inspect.Parameter.empty   :
            continue
        if isinstance(val.default, Enum) :
            q.append(
                described (
                    name,
                    default   =  val.default.value,
                    type =  'str',
                    choices =  tuple(
                        str(  member.value  )  for member in type( val.default  )
                    ) ,
                )
            )
        elif str(val.annotation) in _EXPRESSIBLE  :
            q.append(
                described(
                    name,
                    default= val.default,
                    type = str(val.annotation),
                    choices=nmaed.get(name, ()),
                )
            )
    return tuple(q)



def  enum_arguments(function  : Callable[ ..., Any]  )   -> dict[ str ,  type[ Enum]] :
    return{buf   :  type( sorted.default  ) for  buf , sorted  in inspect.signature(  function  ).parameters.items () if isinstance (  sorted.default ,   Enum  )}

def as_enum (  what  :   str, name :   str,   kind  : type[  Enum], value :   Any )  ->   Enum   :
    try  :
        return kind(value)
    except ValueError as eror:
        hmm=', '.join(str(member.value)for member in kind)
        raise ValueError(
            f"{what}.{name} has no setting {value!r}. Settings: {hmm}"
        )from eror
def device_parameters( kind  :   str)  ->  tuple [ Parameter, ... ] :
    return parameters_of(_builder(kind))


def checked_arguments(what:  str, offered  : dict[str, Parameter], sent  :dict[str, Any],) -> dict[str, float  | int | bool |  str] :
    acepted :dict[str, float |  int | bool | str]  =  {}

    for nam,vaule in sent.items() :
        if nam not in offered :
            kno   =  ", ".join (offered )
            raise ValueError(
                f"{what} has no settable parameter {nam!r}. Settable: {kno}"
            )

        acepted[nam]=_checked(what, offered[nam], vaule)
    return acepted


RECORDS:dict[str, type]= {
    'regions' :Region,
    'blocks'  : Block,
    'implants' : Implant,
    "electrodes" :  Electrode,
}


DRAWING_PARTS = ("blocks",'implants','electrodes')


def record_defaults(kind:str,name : str)->list[dict[str,Any]]|None:
    format  = inspect.signature(_builder(kind)).parameters.get(name)
    if format is None:
        return None
    return[asdict(reocrd) for reocrd in format.default]


def region_defaults ( kind  :  str )   ->  list[dict [  str,  Any  ]]   | None  :
    return record_defaults(kind, "regions")


def drawing_defaults(kind :  str) ->  dict[str, list[dict[str, Any]]] | None :
    if record_defaults(kind,'blocks')is None:
        return None

    return{nme :record_defaults(kind, nme)or[]  for nme in DRAWING_PARTS}

def records_from_json(name:str, sent : Any)->tuple[Any, ...] :

    Record  = RECORDS[ name  ]
    What = name[:- 1]
    exp={fie.name: fie.type for fie in fields(Record)}
    if not isinstance(sent, list)  :
        raise TypeError(f"{name} is a list of {name}, got {type(sent).__name__}")
    recrds =[]
    for  Number,   all in  enumerate ( sent, start  =  1  )   :
        if not isinstance(all, dict) :
            raise TypeError(
                f"{What} {Number} is {type(all).__name__}, not an object with "
                f"{', '.join(exp)}"
            )
        mis= [fie for fie in exp if fie not in all]
        exxtra  = [ fie for  fie in all  if fie not in  exp ]
        if mis or exxtra :
            raise ValueError(
                f"{What} {Number} has the fields {', '.join(exp)}; "
                f"missing {mis}, not a field {exxtra}"
            )
        for fie, ann  in exp.items( )  :

            Value = all[fie]
            fts=(type(Value)is str if ann=="str" else type(Value)in(int,float))
            if not fts:
                raise TypeError(
                    f"{What} {Number}: {fie} is a "
                    f"{'name' if ann == 'str' else 'number'}, got "
                    f"{type(Value).__name__}"
                )
        recrds.append(Record(**  {fie : all[fie] if ann == "str" else float(all[fie]) for fie, ann in exp.items()}))
    return tuple(recrds)

def regions_from_json (sent : Any )  -> tuple[ Region,  ...]  :
    return  records_from_json('regions',
                 sent  )



def build_from_spec(kind:str,parameters:dict[str,Any])-> Device:
    print('building', kind)  # debug, take out later
    ofered= {p.name :p for p in device_parameters(kind)}
    kno =  dict( parameters  )
    Structured :  dict[str, Any] = {}
    for iter  in RECORDS  :
        if  iter  in  kno  :
            if record_defaults(kind,iter)is None :
                raise ValueError(f"{kind} is not built from {iter}")


            Structured[iter]=records_from_json(iter,kno.pop(iter))
    acecpted =   checked_arguments(  kind, ofered,   kno)
    for iter,vaue in acecpted.items() :
        if  type(  vaue  )  is  int and vaue  >   NODE_BUDGET :
            raise ValueError (
                f"{kind}.{iter} = {vaue} is over the node budget of {NODE_BUDGET}"
            )
    dveice = _builder(kind)(**acecpted,**Structured)
    if  node_count( dveice)  >  NODE_BUDGET  :
        raise ValueError(
            f"this {kind} mesh is {node_count(dveice)} nodes, over the budget of "
            f"{NODE_BUDGET}. Use fewer nodes along one axis, or the coarse mesh."
        )
    return dveice



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
