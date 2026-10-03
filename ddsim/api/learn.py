from __future__ import annotations
import copy, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from ddsim.api.devices import COARSE
LEARN= Path(__file__).parent/"static"/'learn'
LESSONS = Path(__file__).parent/'lessons'

_PLAIN  =   '## In plain words'

_DEPTH  =  '## In more depth'




@dataclass(frozen= True)
class Topic :


    name  :   str
    title :str
    summary   :  str

    docs: tuple[str,...]

    plain :str
    depth :str
def topic_names() -> tuple[str, ...] :

    return tuple(sorted(f.stem for f in LEARN.glob("*.md")))



def load_topic(name :str)-> Topic:
    if name not in topic_names():
        raise KeyError(f"no topic {name!r}")
    z =  (LEARN  /  f"{name}.md").read_text(encoding  = "utf-8")
    w, g= _split_header(name, z, ("title", "summary", 'docs'))

    if _PLAIN not in g  :
        raise  ValueError (  f"{name}: no '{_PLAIN}' section"  )
    if _DEPTH  not  in  g   :
        raise  ValueError(  f"{name}: no '{_DEPTH}' section"  )
    out,   y = g.split( _PLAIN, 1 )   [  1 ].split(_DEPTH ,  1 )
    return Topic(name =name, title=w["title"], summary=w["summary"], docs =tuple(x2.strip() for x2 in w["docs"].split(";") if x2.strip()), plain= out.strip(), depth=y.strip(),)



def _split_header(name :str,text: str,required:tuple[str,...])->tuple[dict[str,str],str]:


    if not text.startswith("---\n") or "\n---\n" not in text[4  :]:
        raise ValueError(f"{name}: no --- header")
    obj,nxt=text[4 :].split("\n---\n",1)
    g  : dict[str, str] = {}
    for y in  obj.splitlines(  )   :
        if ":" in y :
            d2,  a  =  y.split( ":", 1 )
            g[d2.strip()]=a.strip()


    for d2 in required :
        if not g.get(d2)  :
            raise ValueError(f"{name}: header has no {d2}")
    return g, nxt

_STEPS="## Steps"

_LOOK  = "## What to look for"

_SAW='## What you saw'
_SET = "set:"


@dataclass(frozen = True)




class  Step  :
    title  :  str
    text : str

    request :dict[str,Any] |None


@dataclass(frozen   = True)




class Lesson  :

    name  :  str
    title:str; summary  : str

    claims:tuple[str,...]

    request :  dict[str, Any]
    mesh_note:str

    steps:tuple[Step, ...]
    look_for :str
    explanation: str
    def  step(  self, title  :  str ) -> Step  :

        for k in self.steps:
            if k.title ==title:
                return k
        raise KeyError(f"{self.name} has no step {title!r}")



def lesson_names() -> tuple[str,...]:

    return tuple(sorted(r2.stem for r2 in LESSONS.glob("*.md")))

def load_lesson(name : str) ->Lesson:


    if  name  not  in  lesson_names()  :

        raise KeyError (  f"no lesson {name!r}" )
    return parse_lesson(name, (LESSONS  /f"{name}.md").read_text(encoding  ="utf-8"))
def parse_lesson( name  :   str,   text  : str ) -> Lesson  :
    x , h  =  _split_header(
        name, text,  ( "title",  'summary' , "claims", 'device' ,   "sweep" )
    )
    cur = _json(name, 'device', x["device"])
    buf  =  { "device"  : cur , "sweep"   :   _json (  name, 'sweep',   x [  'sweep' ])}
    cur.setdefault("parameters",{})
    arr  =  ''
    if x.get("mesh") ==  "coarse" :
        t =cur.get("kind")
        if t not in COARSE :
            raise ValueError(f"{name}: {t} has no coarse mesh")

        res2 =COARSE[t]
        cur["parameters"] =  {**res2.parameters, **cur["parameters"]}
        arr= x.get('mesh_note')or res2.note

    for r2 in(_STEPS, _LOOK, _SAW) :
        if r2 not in h:
            raise ValueError ( f"{name}: no '{r2}' section")
    ii,  y  =  h.split(_STEPS, 1  )   [ 1  ].split(  _LOOK, 1  )


    j ,   xs   =  y.split(_SAW, 1  )


    return  Lesson(name  =   name , title   =  x ["title"  ], summary  =  x[  "summary" ] , claims  =  tuple (u.strip( )   for u in x [  'claims' ].split(';')  if u.strip ( )), request  =  buf , mesh_note  =  arr , steps  =  tuple(_step(  name ,   buf, tmp3 )   for  tmp3 in ii.split ( "### ") [1  : ]), look_for  =   j.strip () , explanation  =   xs.strip (),)

def _step(name:str,start: dict[str,Any],chunk:str) ->Step:

    kk,_,s2 =chunk.partition("\n");kk = kk.strip()
    s2  =s2.strip()
    if not s2.startswith(_SET):
        return Step(title=kk,text=s2,request = None)


    d,_,s2=s2.partition("\n")
    a =_json(name,kk,d[len(_SET) :])

    z = copy.deepcopy(start)
    z['device'] ['parameters'].update(a.get("device", {}))
    for c, y in a.get("sweep", {}).items() :
        if isinstance(y,dict):

            z['sweep'].setdefault(c, {}).update(y)
        else :
            z['sweep'] [c]=  y
    return Step(title   =  kk,  text   = s2.strip(  ) ,  request  =  z  )
def _json(name : str,what :str,text:str)-> Any :
    try:
        return json.loads(text)
    except json.JSONDecodeError as d:
        raise ValueError(f"{name}: {what} is not JSON: {d}") from d

KNOB_TOPICS: dict[str,str] ={"Na":"doping", 'Nd':'doping', 'length' :'pn-diode', "junction":"pn-diode", 'n_nodes' :"mesh", "h_min" :"mesh", 'anode_voltage': "contacts-and-bias", 'cathode_voltage' :'contacts-and-bias', 'left_voltage' :"contacts-and-bias", 'right_voltage':"contacts-and-bias", "substrate_doping": 'doping', "t_ox":"mos-capacitor", "t_si" : 'mos-capacitor', 'width': 'mos-capacitor', "nx": "mesh", "ny":'mesh', 'n_silicon':'mesh', "n_oxide": "mesh", "gate_voltage" : "contacts-and-bias", "body_voltage":"contacts-and-bias", 'work_function': "mos-capacitor", 'L_gate' :'mosfet', 'sd_length':"mosfet", 'contact_length': 'mosfet', "sd_peak":"doping", "x_j":"doping", 'lateral_diffusion':"doping", 'n_contact':"mesh", 'n_sd':"mesh", 'n_channel': "mesh", "h_min_x":'mesh', "h_min_y":"mesh", "drain_voltage":"contacts-and-bias", 'source_voltage' : 'contacts-and-bias', "degenerate" :"fermi-dirac-statistics", 'step':"continuation", "start": "continuation", 'max_iterations':"convergence", 'update_tol':"convergence", 'response': "cv-sweep", 'mobility':'mobility', "auger":'recombination', "field_dependent" :'velocity-saturation', "surface":'surface-scattering',}

KNOB_LABELS:dict[str, str]  =  {
    "Na" :  'P-side doping',
    "Nd": "N-side doping",
    "length" : 'Device length',
    "junction" :  "Junction position",
    "n_nodes" : "Mesh points",
    "h_min" :  "Finest mesh spacing",
    'anode_voltage' : "Anode voltage",
    "cathode_voltage" : 'Cathode voltage',
    "left_voltage" : 'Left contact voltage',
    "right_voltage" : "Right contact voltage",
    'substrate_doping':'Body doping',
    "t_ox": 'Oxide thickness',
    "t_si" : "Silicon thickness",
    'width' :  "Device width",
    "nx": 'Mesh lines across',
    'ny' : 'Mesh lines up',
    "n_silicon"  :'Mesh lines in the silicon',
    "n_oxide" :'Mesh lines in the oxide',
    "gate_voltage" : "Gate voltage",
    "body_voltage"  : 'Body voltage',
    'work_function' :"Gate work function",
    "L_gate"  :'Gate length',
    'sd_length': "Source and drain length",
    "contact_length" : "Contact length",
    "sd_peak"  : "Source and drain doping",
    'x_j' : "Junction depth",
    'lateral_diffusion': 'Sideways diffusion',
    "n_contact" : 'Mesh lines at the contacts',
    'n_sd' : "Mesh lines in source and drain",
    'n_channel' : "Mesh lines along the channel",
    "h_min_x" :  "Finest spacing across",
    'h_min_y'  :  "Finest spacing up",
    'drain_voltage':  "Drain voltage",
    'source_voltage'  :  "Source voltage",
    "degenerate"  :"Heavy doping statistics",
    'step' :'First ramp step',
    'start' :  "Start from",
    "max_iterations" : 'Iteration budget',
    'update_tol' : "Convergence tolerance",
    'response' : "Which carriers follow",
    "mobility": "Mobility model",
    "auger" :  'Auger recombination',
    "field_dependent" : 'Velocity saturation',
    "surface"  : 'Surface scattering',
}

PLOT_TOPICS : dict[str, str] ={
    'residual-plot' : 'residual-plot',
    "legend-psi-residual"  : "newton",
    'legend-n-residual': "newton",
    "legend-p-residual":'newton',
    'legend-gummel-update'  : 'gummel',
    "legend-rejected-step" : 'continuation',
    "curve-plot" :  "curve-plot",
    "compare-runs"  :  'curve-plot',
    "profile-plot" : "profile-plot",
    "legend-psi" :"potential",
    'legend-n' : 'carrier-densities',
    "legend-p": 'carrier-densities',
    'bands-view': 'band-diagram',
    'legend-quasi-fermi':  'quasi-fermi-levels',
    "cutline" :  "cutline",
    "streamlines":  "current-flow",
    "sweep-kind" :"sweep-kinds",
    'device-kind'  : "devices",
    'stack-regions'  : "device-builder",
    "device-file" : 'device-builder',
    "drawing-parts" :'device-drawing',
}

STATUS_TOPICS  :   dict [ str,  str] =  {'done' :  "convergence", "failed"   :  'failed' , 'cancelled'  : 'cancelled', "stalled"  :  "stalled", "dropped"   : "dropped-frames",}
