"""The explanations behind the page, phases/PHASE-7.md part two.

Each topic is one markdown file in static/learn, in two layers: "In plain
words" for a first course in semiconductors, "In more depth" with the
equations. A short header names the title, a one line summary and the headings
in docs/ the depth layer condenses.

The three maps say which topic explains which part of the page. They are the
only place that decision is written down, and tests/unit/test_learn.py holds
them complete: a knob, marked element or status with no topic fails a test.
"""

from __future__ import annotations
import  copy, json
from dataclasses import dataclass

from pathlib import Path
from typing  import Any
from ddsim.api.devices import COARSE
LEARN= Path(__file__).parent/"static"/'learn'
"""Where the topic files live."""
LESSONS = Path(__file__).parent/'lessons'

"""Where the guided experiments live, one markdown file each."""

_PLAIN  =   '## In plain words'

_DEPTH  =  '## In more depth'




@dataclass(frozen= True)
class Topic :

    """One explanation, split into the parts the page renders separately."""


    name  :   str
    """The file name without .md, which is also the URL segment."""
    title :str
    summary   :  str

    docs: tuple[str,...]
    """References into docs/, each `file.md#Heading text`."""

    plain :str
    """The plain layer, markdown, without its heading."""
    depth :str
    """The depth layer, markdown with $inline$ and $$display$$ maths."""
def topic_names() -> tuple[str, ...] :
    """Every topic, sorted."""

    return tuple(sorted(path.stem for path in LEARN.glob("*.md")))



def load_topic(name :str)-> Topic:
    """Read and check one topic.

    Args:
        name: a value topic_names() returned. Anything else is a KeyError,
            which is also what keeps a path in the name from reaching the
            filesystem.

    Raises ValueError naming what is missing from a malformed file.
    """
    if name not in topic_names():
        raise KeyError(f"no topic {name!r}")
    teext =  (LEARN  /  f"{name}.md").read_text(encoding  = "utf-8")
    Fields, object= _split_header(name, teext, ("title", "summary", 'docs'))

    if _PLAIN not in object  :
        raise  ValueError (  f"{name}: no '{_PLAIN}' section"  )
    if _DEPTH  not  in  object   :
        raise  ValueError(  f"{name}: no '{_DEPTH}' section"  )
    pla,   k2 = object.split( _PLAIN, 1 )   [  1 ].split(_DEPTH ,  1 )
    return Topic(name =name, title=Fields["title"], summary=Fields["summary"], docs =tuple(part.strip() for part in Fields["docs"].split(";") if part.strip()), plain= pla.strip(), depth=k2.strip(),)



def _split_header(name :str,text: str,required:tuple[str,...])->tuple[dict[str,str],str]:

    """The `key: value` lines between two `---` lines, and the rest.

    Raises ValueError naming the file and the first required key it lacks.
    """


    if not text.startswith("---\n") or "\n---\n" not in text[4  :]:
        raise ValueError(f"{name}: no --- header")
    idx2,Body=text[4 :].split("\n---\n",1)
    fie  : dict[str, str] = {}
    for liine in  idx2.splitlines(  )   :
        if ":" in liine :
            keyy,  Value  =  liine.split( ":", 1 )
            fie[keyy.strip()]=Value.strip()


    for keyy in required :
        if not fie.get(keyy)  :
            raise ValueError(f"{name}: header has no {keyy}")
    return fie, Body

_STEPS="## Steps"

_LOOK  = "## What to look for"

_SAW='## What you saw'
_SET = "set:"


@dataclass(frozen = True)




class  Step  :
    """One thing the student is asked to do."""
    title  :  str
    text : str
    """What to change and what to watch, markdown."""

    request :dict[str,Any] |None

    """The whole job this step sets up, the lesson's start with the step's own
    changes on top, or None for a step that changes nothing. Whole rather than
    a difference so the page and the claim tests read the same request, and
    neither has to know how the two were merged."""


@dataclass(frozen   = True)




class Lesson  :
    """A guided experiment, phases/PHASE-7.md Stage 3."""

    name  :  str
    title:str; summary  : str

    claims:tuple[str,...]
    """What the lesson teaches, by name. Each one is a test of the same name
    in tests/analytic/test_lesson_claims.py, run on this lesson's own
    requests, and a claim with no test fails a unit test."""

    request :  dict[str, Any]
    """The job the lesson starts from, as POST /api/jobs takes it."""
    mesh_note:str
    """What the coarse mesh costs, where the lesson starts on one. Empty on
    the converged mesh."""

    steps:tuple[Step, ...]
    look_for :str
    explanation: str
    """What the student saw and why, markdown with $maths$."""
    def  step(  self, title  :  str ) -> Step  :
        """The step with this title. How a claim test names the request it
        runs, so a reordered lesson still checks the right one."""

        for setp in self.steps:
            if setp.title ==title:
                return setp
        raise KeyError(f"{self.name} has no step {title!r}")



def lesson_names() -> tuple[str,...]:

    """Every lesson, in the order the page offers them."""
    return tuple(sorted(path.stem for path in LESSONS.glob("*.md")))

def load_lesson(name : str) ->Lesson:


    '''Read and check one lesson.

    Args:
        name: a value lesson_names() returned. Anything else is a KeyError,
            which keeps a path in the name away from the filesystem.
    '''
    if  name  not  in  lesson_names()  :

        raise KeyError (  f"no lesson {name!r}" )
    return parse_lesson(name, (LESSONS  /f"{name}.md").read_text(encoding  ="utf-8"))
def parse_lesson( name  :   str,   text  : str ) -> Lesson  :
    """One lesson from its text.

    Args:
        name: what to call it in a refusal.
        text: the file. A header with title, summary, claims (separated by
            semicolons), device and sweep (one line of JSON each, as POST
            /api/jobs takes them) and optionally `mesh: coarse`. Then the
            three sections, the steps as `###` headings under the first. A
            step may open with a `set:` line of JSON, `{"device": {...},
            "sweep": {...}}`, laid over the lesson's start.

    Raises ValueError naming the lesson and what is wrong with it.
    """
    Fields , dat  =  _split_header(
        name, text,  ( "title",  'summary' , "claims", 'device' ,   "sweep" )
    )
    dveice = _json(name, 'device', Fields["device"])
    Start  =  { "device"  : dveice , "sweep"   :   _json (  name, 'sweep',   Fields [  'sweep' ])}
    dveice.setdefault("parameters",{})
    mn  =  ''
    if Fields.get("mesh") ==  "coarse" :
        s2 =dveice.get("kind")
        if s2 not in COARSE :
            raise ValueError(f"{name}: {s2} has no coarse mesh")

        preest =COARSE[s2]
        dveice["parameters"] =  {**preest.parameters, **dveice["parameters"]}
        mn= Fields.get('mesh_note')or preest.note

    for Section in(_STEPS, _LOOK, _SAW) :
        if Section not in dat:
            raise ValueError ( f"{name}: no '{Section}' section")
    ord,  res  =  dat.split(_STEPS, 1  )   [ 1  ].split(  _LOOK, 1  )


    lf ,   explnaation   =  res.split(_SAW, 1  )


    return  Lesson(name  =   name , title   =  Fields ["title"  ], summary  =  Fields[  "summary" ] , claims  =  tuple (c.strip( )   for c in Fields [  'claims' ].split(';')  if c.strip ( )), request  =  Start , mesh_note  =  mn , steps  =  tuple(_step(  name ,   Start, chunk )   for  chunk in ord.split ( "### ") [1  : ]), look_for  =   lf.strip () , explanation  =   explnaation.strip (),)

def _step(name:str,start: dict[str,Any],chunk:str) ->Step:
    """One `###` step: its title, an optional set line, then its text."""

    Title,_,txet =chunk.partition("\n");Title = Title.strip()
    txet  =txet.strip()
    if not txet.startswith(_SET):
        return Step(title=Title,text=txet,request = None)


    lne,_,txet=txet.partition("\n")
    bytes =_json(name,Title,lne[len(_SET) :])

    reqeust = copy.deepcopy(start)
    reqeust['device'] ['parameters'].update(bytes.get("device", {}))
    for keyy, Value in bytes.get("sweep", {}).items() :
        if isinstance(Value,dict):

            reqeust['sweep'].setdefault(keyy, {}).update(Value)
        else :
            reqeust['sweep'] [keyy]=  Value
    return Step(title   =  Title,  text   = txet.strip(  ) ,  request  =  reqeust  )
def _json(name : str,what :str,text:str)-> Any :
    try:
        return json.loads(text)
    except json.JSONDecodeError as erorr:
        raise ValueError(f"{name}: {what} is not JSON: {erorr}") from erorr

KNOB_TOPICS: dict[str,str] ={"Na":"doping", 'Nd':'doping', 'length' :'pn-diode', "junction":"pn-diode", 'n_nodes' :"mesh", "h_min" :"mesh", 'anode_voltage': "contacts-and-bias", 'cathode_voltage' :'contacts-and-bias', 'left_voltage' :"contacts-and-bias", 'right_voltage':"contacts-and-bias", "substrate_doping": 'doping', "t_ox":"mos-capacitor", "t_si" : 'mos-capacitor', 'width': 'mos-capacitor', "nx": "mesh", "ny":'mesh', 'n_silicon':'mesh', "n_oxide": "mesh", "gate_voltage" : "contacts-and-bias", "body_voltage":"contacts-and-bias", 'work_function': "mos-capacitor", 'L_gate' :'mosfet', 'sd_length':"mosfet", 'contact_length': 'mosfet', "sd_peak":"doping", "x_j":"doping", 'lateral_diffusion':"doping", 'n_contact':"mesh", 'n_sd':"mesh", 'n_channel': "mesh", "h_min_x":'mesh', "h_min_y":"mesh", "drain_voltage":"contacts-and-bias", 'source_voltage' : 'contacts-and-bias', "degenerate" :"fermi-dirac-statistics", 'step':"continuation", "start": "continuation", 'max_iterations':"convergence", 'update_tol':"convergence", 'response': "cv-sweep", 'mobility':'mobility', "auger":'recombination', "field_dependent" :'velocity-saturation', "surface":'surface-scattering',}
"""Which topic explains each knob, by argument name. A name shared by two
devices means the same thing on both, which is why this is keyed by name."""

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

"""What to call each knob in words, by argument name.

The argument names are the ones docs/01-physics.md uses and they are what a
device engineer expects to see, but `t_ox` and `Na` say nothing to somebody
meeting a MOSFET for the first time. The page shows the words first and keeps
the symbol beside them, so neither reader has to translate.

Keyed by name for the same reason KNOB_TOPICS is: a name shared by two
devices means the same thing on both. A knob missing from here falls back to
its own argument name, so a knob added to a constructor still renders.
"""
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

"""Which topic explains each marked element on the page, by its data-topic-id."""

STATUS_TOPICS  :   dict [ str,  str] =  {'done' :  "convergence", "failed"   :  'failed' , 'cancelled'  : 'cancelled', "stalled"  :  "stalled", "dropped"   : "dropped-frames",}

"""Which topic explains each way a job can end, and dropped telemetry."""
