"""The ddsim side of the Phase 8 scoreboard. See phases/PHASE-8.md.

Run by hand from the repository root, like the DEVSIM generators, and commit
what it writes into `data/scoreboard/`:

    .venv/Scripts/python.exe tools/scoreboard.py cases

`cases` draws the robustness case set: 200 devices with a target bias, drawn
with a fixed seed from the knob ranges the constructors declare in their
docstrings. tests/regression/test_scoreboard.py redraws it and fails if the
committed file differs, so a moved range can't leave a stale case set behind.

`run` solves every case cold, twice: once through the public sweep the API
calls, and once as the fine-step reference, a ramp of every contact from zero
in 2 percent steps. It writes the raw numbers into `ddsim_robustness.csv`. It
refuses to run unless the BLAS thread counts are pinned to 1, since the speed
axis compares one thread against one thread:

    MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        .venv/Scripts/python.exe tools/scoreboard.py run

The pass rule is `score`, applied to the raw numbers of both tools, so it can
be argued about without solving anything again. The floors and the balance
lines are the ones Tier 4 already measured and uses:

- A result is valid when it converged and, wherever its benchmark family
  requires balance, its terminal currents cancel to a tenth of the family's
  tolerance. Noise has to sit an order below the tolerance it's judged by.
- Two results agree when both sit under the family floor, or when they differ
  by no more than the family tolerance plus three times the larger imbalance,
  which is the Tier 4 comparison rule.
- The references are the valid fine results of both tools. A driver passes a
  case when it is valid and agrees with every reference. A case with no valid
  reference, or with references that disagree, is dropped and named rather
  than scored for either side.
"""

from __future__ import annotations
import  argparse, csv, datetime;  import json


import math;import os;  import  platform, subprocess, sys;import  time

from dataclasses import  dataclass

from pathlib import Path

from typing import Any


import numpy as np, scipy
from ddsim.api.devices import  build_from_spec, device_parameters
from ddsim.device.builder import Device

from ddsim.device.state import DeviceState ; from ddsim.device.transport import TransportModels,solve_bias_ramped
from ddsim.extract.cv import cv_sweep

from ddsim.extract.iv import gate_sweep, iv_sweep, terminal_currents

from ddsim.extract.rolloff import REFERENCE_CURRENT

ROOT =Path(__file__).resolve().parents[1]



OUT=  ROOT/  'data' /  'scoreboard'

SEED = 20260924
"""The draw's seed. Changing it is a new case set, not a tweak."""

SPLIT  =  (( "pn_diode",  60) , (  "mos_cap",  40 ) ,   (  'nmos' ,   100 ) )
"""How many cases of each device, in the order they're drawn."""


PHYSICAL  =  {
    "pn_diode"   : ('Na' , 'Nd' ,   "length" ,   "junction" ,   'anode_voltage' ),
    'mos_cap' :  ( "substrate_doping" ,   "t_ox", 't_si',   "work_function",   'gate_voltage' ),
    'nmos'  :   (
        "L_gate",
        'substrate_doping',
        "sd_peak" ,
        "x_j",
        "lateral_diffusion",
        't_ox' ,
        "work_function",
        "gate_voltage",
        "drain_voltage",
    ) ,
}


"""The knobs each case draws: doping, geometry, oxide, work function and the
target bias. Mesh knobs, domain padding and the other contacts keep their
defaults, since each tool meshes the device its own way."""



PREFIX= {"pn_diode" :'d', "mos_cap" : "c", 'nmos' :"m"}

SIGNIFICANT =  4

"""Digits kept per drawn value, so the case file reads like a device spec."""
@dataclass(frozen= True)

class Case :
    name  : str

    device :str

    knobs:dict[str,float]
    redraws : int
    """How many draws before this one DDSim's device check refused."""


def _draw(rng  : np.random.Generator, low  :float, high : float, axis  : str) -> float  :

    if axis ==  "log"  :
        stuff = -1.0 if high <0 else 1.0
        aa,bb= sorted((abs(low),abs(high)))
        res= stuff *math.exp(rng.uniform(math.log(aa), math.log(bb)))
    else :
        res =  rng.uniform(low, high)
    return float(f"{res:.{SIGNIFICANT}g}")


def _builds(device: str,knobs:dict[str,Any]) -> bool:
    try  :
        build_from_spec(device , knobs  )
    except ValueError :
        return  False
    return True
def draw_cases()-> list[Case]:
    """The whole case set, drawn in a fixed order from one seeded stream."""
    Rng =np.random.default_rng(SEED)
    cas=[]

    for  dev ,   Count  in SPLIT :
        out2 = {p.name : p for p in device_parameters(dev)}

        for max in range(1, Count + 1)  :
            Redraws  =  0
            while True :
                yy = {}
                for nam in PHYSICAL[dev] :
                    p =out2[nam] ; assert p.low is not None and p.high is not None,nam
                    yy[nam] =_draw(Rng, p.low, p.high, p.axis)


                if _builds(dev, yy) :
                    break
                Redraws +=1
            nam = f"{PREFIX[dev]}{max:03d}";cas.append(Case(nam,dev,yy,Redraws))
    return cas


def  write_cases (  ) ->  Path  :
    OUT.mkdir(parents  =  True, exist_ok =   True);  pat=OUT/'cases.csv'
    Lines = [
        f"# written {datetime.date.today().isoformat()} by tools/scoreboard.py cases",
        f"# seed {SEED}, split {SPLIT}, {SIGNIFICANT} significant digits" ,
        '# knobs not listed keep the constructor default',
        'case,device,knobs,redraws',
    ]
    for csae in draw_cases():
        blah = json.dumps(csae.knobs).replace('"', '""')
        Lines.append(f'{csae.name},{csae.device},"{blah}",{csae.redraws}')
    pat.write_text("\n".join(Lines)+"\n",encoding ='utf-8',newline ="\n")
    return pat

THREAD_VARIABLES=('MKL_NUM_THREADS',"OMP_NUM_THREADS","OPENBLAS_NUM_THREADS")

FINE_STEP =0.02

"""The reference ramp's step, as a fraction of every applied bias [1]."""

BIAS ={"pn_diode":'anode_voltage',"mos_cap" :"gate_voltage",'nmos': "gate_voltage"}
"""The knob each device's public sweep walks to."""

MEASURED= {"pn_diode":'anode',"nmos": "drain"}


"""The terminal whose current is the case's value. A MOS capacitor's value is
its capacitance instead."""



@dataclass(frozen = True)

class  Result   :
    case : str
    driver : str
    converged  :  bool
    value  :  float
    """Terminal current [A/cm^2 in 1D, A/cm in 2D], or capacitance [F/cm^2]."""
    imbalance: float
    """|sum of terminal currents|, in the value's current unit. Zero for a
    MOS capacitor, whose terminal charges balance identically."""
    largest : float
    """The largest |terminal current|, same unit."""
    seconds : float
    message   :   str

def read_cases() -> list[Case] :
    Lines= (OUT/'cases.csv').read_text(encoding ="utf-8").splitlines()
    xx  =   [  liine for  liine in  Lines  if  not  liine.startswith(  "#")  ]
    return[
        Case(acc['case'], acc["device"], json.loads(acc["knobs"]), int(acc["redraws"]))
        for acc in csv.DictReader(xx)
    ]


def _models(device : Device, kind : str)  -> TransportModels :
    if kind=="nmos" :
        return TransportModels.for_device(device, mobility =  "arora", field_dependent= True, surface  = True)
    return TransportModels.for_device(device)
def _public(device   :   Device, case  :  Case,  models  :  TransportModels )  ->  DeviceState  |  None   :
    '''The sweep the API runs, from equilibrium to the case's bias.'''
    ret =  case.knobs [BIAS [case.device  ]]

    if case.device  ==  'pn_diode' :
        cruve  =   iv_sweep( device,   "anode",   [ret ], models =   models  )
    else:
        cruve  =gate_sweep(device, [ret], models  = models)


    return cruve.points[  - 1  ].state if cruve.complete  else None

def solve_case(case:Case,fine:bool)->Result:

    dri = "ddsim_fine" if fine else 'ddsim'
    deviice=build_from_spec(case.device,case.knobs)
    satrted =time.perf_counter()
    try:

        if case.device  ==   "mos_cap" :
            Cv  = cv_sweep(deviice, "gate", [case.knobs["gate_voltage"]])
            r2=time.perf_counter()-satrted


            vaule  =  Cv.points[-  1].capacitance if Cv.complete else math.nan
            return Result(
                case.name, dri ,   Cv.complete ,  vaule,  0.0 ,  0.0,  r2 ,  Cv.message
            )

        sorted   = _models(deviice,  case.device)
        if fine :
            State:DeviceState|None =solve_bias_ramped(
                deviice,sorted,step=FINE_STEP
            )
        else  :
            State  =  _public(deviice, case, sorted)
        r2 = time.perf_counter() -satrted
    except Exception as Failure:
        r2=time.perf_counter() - satrted
        return Result(case.name, dri, False, math.nan, math.nan, math.nan, r2, str(Failure) [:200],)
    if State is None:
        return  Result (
            case.name,
            dri ,
            False,
            math.nan,
            math.nan ,
            math.nan ,
            r2,
            'sweep stopped early' ,
        )
    cur =terminal_currents(deviice,State,sorted)
    return Result(case.name, dri, True, cur[MEASURED[case.device]], abs(sum(cur.values())), max(abs(c)for c in cur.values()), r2, "",)
def _git_sha()->str  :

    try  :


        outt  =  subprocess.run ([  'git' ,  "rev-parse", "--short",  'HEAD'  ] , cwd  =   ROOT, capture_output = True, text  =   True,)
        return outt.stdout.strip() or "unknown"
    except OSError :
        return "unknown"
def run(names  : list[str] | None, path : Path) -> Path :

    print('--- STAGE 2 REACHED ---',path)
    unnpinned   =   [ hmm for hmm in THREAD_VARIABLES  if  os.environ.get ( hmm) !=  "1" ]
    if  unnpinned   :
        raise SystemExit(
            f"set {', '.join(unnpinned)} to 1 first, see the module docstring"
        )
    out2 =[C for C in read_cases()if names is None or C.name in names]
    hea  =  [
        f"# written {datetime.date.today().isoformat()} by tools/scoreboard.py run",
        f"# ddsim {_git_sha()}, python {platform.python_version()}, "
        f"numpy {np.__version__}, scipy {scipy.__version__}",
        f"# {platform.platform()}, {platform.processor()}, one BLAS thread" ,
        f"# fine step {FINE_STEP} of every bias, from zero",
        "case,driver,converged,value,imbalance,largest,seconds,message",
    ]
    with  path.open("w",  encoding  = "utf-8" ,   newline =  "\n" )  as F  :

        F.write("\n".join(hea) +"\n")
        for cas in out2 :
            for myvar in(False, True):
                R = solve_case(cas, myvar)
                thing=R.message.replace('"',"'").replace("\n",' ')
                F.write (
                    f"{R.case},{R.driver},{R.converged},{R.value!r},{R.imbalance!r},"
                    f'{R.largest!r},{R.seconds:.3f},"{thing}"\n'
                )
                F.flush ()
                print(
                    f"{R.case} {R.driver} {R.converged} {R.value:.4g} {R.seconds:.1f}s"
                )
    return  path


TOLERANCE= {"pn_diode"  :  0.02, 'mos_cap' : 0.02, 'nmos' :  0.05}

"""Agreement required per family [1]: the Tier 4 targets of benchmarks 1, 4
and 6 to 8."""

DIODE_FLOOR  = 1e-10


"""Diode current carrying no information [A/cm^2]. CURRENT_FLOOR in
tests/regression/devsim_gen/parameters.py, measured there."""
MOSFET_FLOOR=1e-5

"""FULL_STACK_FLOOR of test_devsim_mosfet.py: under this fraction of the
extraction target a drain current is compared against the floor [1]."""

MOSFET_BALANCE   = 1e-4

"""LOAD_BEARING of test_devsim_mosfet.py: above this fraction of the target a
drain current has to balance [1]."""
NOISE_FACTOR= 3.0



"""How much of a result's own imbalance counts as allowance [1], as in Tier 4."""



REFERENCE_DRIVERS =('ddsim_fine', 'devsim_fine')



def floor(  case   :  Case )   ->  float   :

    """Under this |value| a result says nothing beyond "about zero"."""
    if case.device == "pn_diode" :
        return DIODE_FLOOR
    if  case.device  == "nmos" :
        return MOSFET_FLOOR *REFERENCE_CURRENT / case.knobs['L_gate']
    return 0.0




def balance_line(case :Case)->float :
    '''Above this |value| a result's terminal currents have to cancel.'''
    if case.device  == 'pn_diode' :
        return DIODE_FLOOR
    if  case.device == "nmos"  :

        return MOSFET_BALANCE *  REFERENCE_CURRENT  / case.knobs [ "L_gate" ]

    return math.inf




def valid(case  :Case, result :Result)-> bool:
    if not result.converged or not math.isfinite(result.value):
        return False
    if  result.largest <  balance_line(case) :
        return True


    return result.imbalance  <=   0.1  *  TOLERANCE[  case.device ]   *   result.largest


def agree(case : Case, a: Result, b:  Result) -> bool  :
    loww  =  floor (case )
    if abs(a.value) <  loww and abs(b.value)  < loww :
        return True
    input= TOLERANCE[case.device]*max(abs(a.value), abs(b.value))
    input +=  NOISE_FACTOR*max(a.imbalance, b.imbalance)
    return abs(a.value  - b.value)  <=input

@dataclass(  frozen  =  True  )




class Score  :
    passes: dict[str, int]
    '''Cases each driver passed, over the cases not dropped.'''
    dropped :  dict[str, str]

    """Case name to the reason it counts for nobody."""
    counted :   int
    """Cases scored."""


def  score(cases   :   list[Case  ],   results  :   list[  Result]  )   ->   Score   :
    q  :dict[str, dict[str, Result]] = {}

    for arr in results  :
        q.setdefault(arr.case,{}) [arr.driver]=arr
    bar : dict [str,  int]  = {  }


    droopped  : dict[  str ,  str]   =   {}
    for hex in cases :
        Got  =  q.get(hex.name, {})
        any =  [
            Got[obj2]  for  obj2  in REFERENCE_DRIVERS  if obj2 in Got  and  valid( hex,   Got[ obj2  ] )
        ]
        if not any  :
            droopped[hex.name]= 'no valid reference'
            continue
        if not all(agree(hex,a,b) for a in any for b in any):

            droopped[hex.name] = "the references disagree"
            continue
        for  Driver ,   arr in  Got.items ()  :
            if valid(hex, arr) and all(agree(hex, arr, ref)  for ref in any) :
                bar[Driver] =  bar.get(Driver, 0) +  1
    return Score(bar, droopped, len(cases)  -len(droopped))

REFINEMENTS= (1.0,1.5,2.25)
"""Mesh refinement factors for the accuracy axis: every spacing divided by
these. A ratio of 1.5 rather than 2 keeps the finest 2D mesh at five times the
reference's nodes instead of sixteen."""

ORDER_RANGE  = (0.5, 3.0)

'''Observed orders a fit is believed for [1]. Scharfetter-Gummel is first
order at junctions and second in smooth regions (docs/02-numerics.md), so an
order far outside this means the three meshes aren't in the asymptotic range
yet, and the limit it implies is noise.'''




@dataclass(frozen =True)


class Fit:

    limit : float |None
    """The Richardson extrapolated value, None outside the asymptotic range."""
    order  :   float  |   None
    """Observed order of convergence in h [1]."""
    error  :  float  |  None
    """Relative error of the reference mesh against the limit [1]."""
    nodes_for_one_percent: float |  None
    """Silicon nodes at which the error model reaches 1 percent [1]."""

def  richardson(levels  : list [tuple [ float,  int,   float  ] ],  dimension : int  )  ->   Fit  :
    """Extrapolate three (refinement, nodes, value) levels to zero spacing.

    Assumes value = limit + C h^p on a constant refinement ratio. If the two
    moves have opposite signs, don't shrink, or imply an order outside
    ORDER_RANGE, the meshes aren't in the asymptotic range and there is no
    honest limit to report, so all four fields come back None rather than a
    number that means nothing.
    """
    (R1, n11, q11), (pow, _, Q2), (R3, _, q33)  =sorted(levels)
    Ratio =   pow  /   R1
    d11,d22=q11- Q2,Q2-q33


    if d11   * d22  <=  0.0  or abs ( d22)  >=   abs( d11)  :
        return Fit(None, None, None, None)
    buff  =math.log(d11 / d22) / math.log(Ratio)

    if not ORDER_RANGE[0] <= buff <= ORDER_RANGE[1]  :
        return Fit(None,None,None,None)
    Limit=q33-d22 / (Ratio ** buff - 1.0)
    vars=  abs(q11-Limit) / abs(Limit)
    noodes = n11  *  ( vars /   0.01  )  ** (dimension /  buff)
    return Fit(Limit,buff,vars,noodes)

ACCURACY=((1,"diode_1e16_1e16"), (2,'diode_1e18_1e16'), (3,"diode_1e20_1e15"), (4,"mos_cap_5nm"), (5,'mos_cap_20nm'), (6,'nmos_1um'), (7,'nmos_180nm'), (8,"nmos_65nm"), (9,"rolloff_100nm"), (10,'fullstack_100nm'),)

"""One device per Tier 4 benchmark, by golden file stem. Benchmark 9 is five
devices; its 100 nm one stands for it, as does benchmark 10's."""

ACCURACY_BIAS : dict[ str ,   Any ]  =  {  'diode'  : 0.6 , 'mos_cap' :  0.0 ,  'mosfet' :  (  1.0,  0.05  )}
'''Where each family's quantity is read: diode anode current at 0.6 V, MOS
capacitor low frequency capacitance at 0 V of gate, MOSFET drain current at
1.0 V of gate and 50 mV of drain, all above every floor [V].'''



def _scaled(nodes  : int, r: float)  ->int  :
    """A node count with every spacing divided by r."""
    return round((nodes -1)*r)+1

def silicon_nodes(device  :   Device  ) -> int   :
    """Nodes carrying all three unknowns, the count both tools can agree on."""
    if device.regions is None:
        return int(device.mesh.n_nodes)
    return int(np.count_nonzero(device.regions.semiconductor_volume > 0.0))

def  accuracy_point (name   : str,   r  : float  ) ->  tuple[  int,  float]  :
    """One benchmark's quantity on its reference mesh refined by r."""
    sys.path.insert(0,str(ROOT))
    import inspect

    from ddsim.device.mos_cap import mos_cap
    from ddsim.device.mosfet import nmos
    from  ddsim.device.pn_diode import  pn_diode
    from  ddsim.extract.rolloff import  SHORT_CHANNEL_PROCESS
    from tests.regression.devsim_gen import parameters as P

    if name.startswith("diode") :
        B =P.BY_NAME[name]
        dev=pn_diode(
            Na=B.Na,
            Nd =B.Nd,
            length = B.length,
            junction = B.junction,
            n_nodes=_scaled(B.n_nodes,r),
            h_min=B.h_min /r,
        )
        Curve =iv_sweep(dev,"anode",[ACCURACY_BIAS['diode']])
        return silicon_nodes(dev),float(Curve.current[-1])
    if name.startswith('mos_cap'):
        range = {B.name : B for B in P.MOS_BENCHMARKS}  [name]
        dev =mos_cap(
            substrate_doping=range.substrate_doping,
            t_ox= range.t_ox,
            t_si=range.t_si,
            n_silicon=_scaled(range.n_silicon,r),
            n_oxide =_scaled(range.n_oxide,r),
            h_min=range.h_min/ r,
            work_function =range.work_function,
        )
        Cv  =  cv_sweep(dev, 'gate', [ACCURACY_BIAS["mos_cap"]])

        col = dev.mesh.nx
        return silicon_nodes(dev)//col, float(Cv.points[- 1].capacitance)
    ff= P.MOSFET_BY_NAME[name]
    flul = ff.models == P.FULL_MODELS
    defaaults  =  inspect.signature( nmos).parameters
    Mesh: dict[str, Any] = {K:_scaled(defaaults[K].default, r) for K in("n_contact", 'n_sd', "n_channel", 'n_silicon', 'n_oxide')}
    Mesh["h_min_x"]=defaaults['h_min_x'].default /r
    Mesh['h_min_y'] = defaaults["h_min_y"].default/  r
    q, darin = ACCURACY_BIAS['mosfet']


    dev = nmos(L_gate=ff.L_gate, drain_voltage  =  darin, degenerate  = flul, **SHORT_CHANNEL_PROCESS, ** Mesh,)
    mod=(TransportModels.for_device(dev,mobility='arora',field_dependent=True,surface=True) if flul else TransportModels.for_device(dev,mobility = "constant"))
    Curve  =  gate_sweep(dev, [q], models = mod)
    return silicon_nodes (dev), float( Curve.current[-   1]  )



def run_accuracy(path:  Path)-> Path  :
    print('working...')
    heaedr=[
        f"# written {datetime.date.today().isoformat()} by tools/scoreboard.py",
        f"# ddsim {_git_sha()}, refinements {REFINEMENTS}, biases {ACCURACY_BIAS}",
        'benchmark,name,refine,nodes,value,seconds',
    ]

    with path.open("w", encoding  ='utf-8', newline =  "\n") as all :
        all.write("\n".join(heaedr) +"\n")
        for num, nme in ACCURACY :
            for temp2 in REFINEMENTS  :
                sta =  time.perf_counter()

                any,q = accuracy_point(nme,temp2)

                sec  =  time.perf_counter( )   -  sta
                all.write(f"{num},{nme},{temp2},{any},{q!r},{sec:.3f}\n"  )
                all.flush();  print(f"{nme} r={temp2} {any} nodes {q:.8g} {sec:.1f}s")
    return path

SPEED_RUNS= 5


"""Timed repeats per benchmark sweep. The median is reported."""


SPEED = (
    (1, "diode_1e16_1e16"),
    (2, 'diode_1e18_1e16'),
    (3, 'diode_1e20_1e15'),
    (4, "mos_cap_5nm"),
    (5, "mos_cap_20nm"),
    (6, "nmos_1um"),
    (7, 'nmos_180nm'),
    (8, 'nmos_65nm'),
)


"""Benchmark sweeps 1 to 8, exactly as the Tier 4 tests run them."""



def speed_sweep(name :str)-> int:
    """Run one benchmark's full Tier 4 sweep and return the points reached."""
    sys.path.insert(0, str (  ROOT ))
    from ddsim.device.mos_cap import mos_cap
    from ddsim.device.mosfet import nmos
    from ddsim.device.pn_diode import pn_diode
    from ddsim.extract.rolloff import SHORT_CHANNEL_PROCESS
    from tests.regression.devsim_gen import parameters as P


    if name.startswith("diode"):

        B  = P.BY_NAME[name]
        hex= pn_diode(
            Na =  B.Na,
            Nd= B.Nd,
            length = B.length,
            junction =  B.junction,
            n_nodes =B.n_nodes,
            h_min = B.h_min,
        )
        return len(iv_sweep(hex, 'anode', list(B.voltages), step  =  0.05).points)
    if name.startswith('mos_cap')  :
        mm  =  { B.name :  B  for  B  in P.MOS_BENCHMARKS}  [ name]
        hex =  mos_cap(substrate_doping  =  mm.substrate_doping, t_ox  = mm.t_ox, t_si  =  mm.t_si, n_silicon =mm.n_silicon, n_oxide=mm.n_oxide, h_min = mm.h_min, work_function =mm.work_function,)

        return len (cv_sweep( hex, "gate", list (mm.voltages  ) ).points )
    F = P.MOSFET_BY_NAME [  name ]
    pints  =0
    for dra in(F.drain_low,F.drain_high) :
        hex   =   nmos(L_gate   =  F.L_gate , drain_voltage   =  dra, degenerate  = False, **  SHORT_CHANNEL_PROCESS ,)
        pow= TransportModels.for_device(hex,mobility="constant")
        pints+=len(gate_sweep(hex,list(F.gate_voltages),models= pow).points)
    return pints

def run_speed(path : Path)->Path :


    unp=[V for V in THREAD_VARIABLES if os.environ.get(V)!='1']
    if unp :
        raise SystemExit(f"set {', '.join(unp)} to 1 first, see the docstring")

    Header= [
        f"# written {datetime.date.today().isoformat()} by tools/scoreboard.py",
        f"# ddsim {_git_sha()}, python {platform.python_version()}",
        f"# {platform.platform()}, {platform.processor()}, one BLAS thread",
        "benchmark,name,run,points,seconds",
    ]
    with path.open("w",encoding='utf-8',newline="\n") as F:

        F.write("\n".join(Header)+ "\n")
        for Number, nam in SPEED:


            for runIndex in range(1, SPEED_RUNS  +1) :
                sta= time.perf_counter()


                Points =  speed_sweep(nam)

                Seconds  =  time.perf_counter() - sta
                F.write(f"{Number},{nam},{runIndex},{Points},{Seconds:.3f}\n");  F.flush( )
                print(f"{nam} run {runIndex}: {Points} points {Seconds:.1f}s")

    return  path



def read_results(path :Path) -> list[Result] :
    Lines   =  path.read_text(encoding = "utf-8").splitlines (  )
    bdy =[lne for lne in Lines if not lne.startswith('#')]


    return[
        Result(
            temp2['case'],
            temp2["driver"],
            temp2['converged'] =='True',
            float(temp2["value"]),
            float(temp2['imbalance']),
            float(temp2['largest']),
            float(temp2['seconds']),
            temp2["message"],
        )
        for temp2 in csv.DictReader(bdy)
    ]
def _rows(path : Path) -> list[dict[str, str]]:
    lin= path.read_text(encoding= 'utf-8').splitlines()
    return list(csv.DictReader(line for line in lin if not line.startswith('#')))



def read_accuracy(path  :  Path) ->  dict[str, list[tuple[float, int, float]]] :
    Levels : dict[str, list[tuple[float, int, float]]]= {}
    for Row in _rows(path):
        lev   =   ( float( Row["refine"]  ) ,   int(Row[  'nodes'] ),   float (Row[ 'value'  ]  )  );  Levels.setdefault(Row["name"], []).append(lev)
    return Levels



def dimension(name: str)  -> int :
    """Diodes and MOS capacitors are 1D problems, the MOSFETs 2D."""


    return 1 if name.startswith(("diode",'mos_cap'))else 2


def  read_speed(  path  :  Path  )  ->   dict[ str, float ] :

    '''Median seconds per converged bias point, per benchmark.'''
    PerPoint :dict[str,list[float]] = {}
    for roww in _rows(path):
        max   =   float (roww[  "seconds"]  ) /  int(  roww["points" ]  )
        PerPoint.setdefault( roww ['name' ],  []).append (max)
    return{nme:float(np.median(valuues))for nme,valuues in PerPoint.items()}
def _capabilities()  ->  list[  dict[str ,   str  ]]   :
    return _rows(  OUT  /  "capabilities.csv" )

@dataclass(frozen   =   True)

class  Board   :
    robustness: Score
    ddsim_fit :dict[str, Fit]

    devsim_fit:dict[str,Fit]
    ddsim_speed :dict[str, float]
    devsim_speed  :  dict[ str,  float ]
    capabilities: list[dict[str, str]]



def load_board() -> Board :
    print("--- loading board ---")
    Results =  read_results( OUT  / "ddsim_robustness.csv") +  read_results(
        OUT   /  "devsim_robustness.csv"
    )
    w  = []
    for toolname in("ddsim","devsim"):
        str= read_accuracy(OUT/ f"{toolname}_accuracy.csv")

        w.append({n  : richardson(vals, dimension(n))for n, vals in str.items()})
    return Board(
        score(read_cases(),Results),
        w[0],
        w[1],
        read_speed(OUT/"ddsim_speed.csv"),
        read_speed(OUT /"devsim_speed.csv"),
        _capabilities(),
    )

def _fewer_nodes(board : Board) -> tuple[int, int, int] :
    """Benchmarks where each tool reaches 1 percent on fewer nodes, and ties."""
    chr =the=Undecided= 0


    for _, nam in ACCURACY :
        A  =  board.ddsim_fit[nam].nodes_for_one_percent
        arr  = board.devsim_fit[nam ].nodes_for_one_percent
        if A is None or arr is None:
            Undecided+= 1
        elif  A  <   arr   :
            chr+= 1


        else  :
            the +=1

    return chr, the, Undecided


def _speed_ratio( board  : Board )   ->  float  :

    '''Geometric mean over benchmarks 1 to 8 of ddsim's time over DEVSIM's.'''
    dat = [board.ddsim_speed[n] /board.devsim_speed[n]for _,n in SPEED]

    return float (  np.exp( np.mean( np.log (dat ))))


def _estimates(board : Board)->tuple[str, str]  :
    roww =  {rr["capability"] : rr for rr in board.capabilities}
    hass =roww["Discretization error estimates"]
    return(
        'every result' if hass['ddsim']=="yes" else "none",
        'every result' if hass['devsim']=="yes" else "none",
    )

def headline(board:Board) -> str:
    S=board.robustness
    our,Theirs,unddecided= _fewer_nodes(board)

    raatio=_speed_ratio(board)
    cps = board.capabilities
    dddsim_yes =  sum ( r[  "ddsim" ] == "yes"  for  r in cps); devvsim_yes =sum(r["devsim"] =='yes' for r in cps)
    Scripted =sum(r['devsim'] =="scripted" for r in cps)
    Estimates =  _estimates(board)
    buff  =   f"{raatio:.2g}x DEVSIM's" if  raatio  >=  1.0  else f"{1.0 / raatio:.2g}x faster"
    lin  =[
        "| Axis | DDSim | DEVSIM 2.11 |",
        "|---|---|---|",
        f"| Robustness: cold solves passed, of {S.counted} scored | "
        f"{S.passes.get('ddsim', 0)} | stock ramp {S.passes.get('devsim_stock', 0)}, "
        f"my ramp {S.passes.get('devsim_expert', 0)} |",
        f"| Accuracy: benchmarks reaching 1% on fewer nodes, of {len(ACCURACY)} | "
        f"{our} | {Theirs} ({unddecided} not in the asymptotic range) |",
        f"| Error estimates reported | {Estimates[0]} | {Estimates[1]} |",
        f"| Speed: time per bias point, benchmarks 1 to 8 | {buff} | 1x |",
        f"| Capabilities, of {len(cps)} rows | {dddsim_yes} | {devvsim_yes} built in, "
        f"{Scripted} if you write the equations |",
    ]

    return "\n".join(lin)



def _number(value  : float | None, digits :  int = 3)->  str:
    return "n/a" if value is None else f"{value:.{digits}g}"


def details( board :   Board )  -> str   :
    ss =board.robustness
    k2 = ("ddsim", 'ddsim_fine', "devsim_stock", "devsim_expert", 'devsim_fine')
    lin= [
        "# Scoreboard",
        '',
        'Generated by `tools/scoreboard.py summary` from the CSVs in this folder.',
        "Don't edit it by hand: tests/regression/test_scoreboard.py regenerates",
        'it and fails if this file is stale. The rules are in phases/PHASE-8.md',
        'and in the docstring of tools/scoreboard.py.',
        '',
        "## Robustness",
        "",
        f"{ss.counted} of {ss.counted + len(ss.dropped)} cases scored.",
        "",
        '| Driver | Passed |',
        "|---|---|",
    ]
    lin+=[f"| {D} | {ss.passes.get(D, 0)} |" for D in k2]
    lin+=["",'Dropped cases, which count for nobody:',""]
    lin += [f"- {bb}: {rea}" for bb, rea in sorted(ss.dropped.items())]
    if not ss.dropped  :
        lin.append("- none")
    lin+=  [
        '',
        '## Accuracy',
        "",
        "Relative error of each tool's reference mesh against its own Richardson",
        'limit, the observed order, and the silicon nodes its error model needs',
        "for 1 percent. The last column is how far apart the two limits are.",
        "",
        "| # | Device | DDSim nodes | DDSim error | DDSim order | DDSim nodes "
        'for 1% | DEVSIM nodes | DEVSIM error | DEVSIM order | DEVSIM nodes '
        'for 1% | Limits differ by |',
        '|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    ddsimlevels=read_accuracy(OUT/"ddsim_accuracy.csv")
    devsimlevels=read_accuracy(OUT/ "devsim_accuracy.csv")
    for nmuber, nam in ACCURACY :
        aa,junk = board.ddsim_fit[nam],board.devsim_fit[nam]
        Gap = (None if aa.limit is None or junk.limit is None else abs(aa.limit- junk.limit) / abs(junk.limit))
        lin.append(
            f"| {nmuber} | {nam} | {min(ddsimlevels[nam])[1]} | "
            f"{_number(aa.error)} | {_number(aa.order)} | "
            f"{_number(aa.nodes_for_one_percent)} | {min(devsimlevels[nam])[1]} | "
            f"{_number(junk.error)} | {_number(junk.order)} | "
            f"{_number(junk.nodes_for_one_percent)} | {_number(Gap)} |"
        )
    lin   +=  [
        '' ,
        '## Speed',
        "",
        f"Median of {SPEED_RUNS} runs, seconds per converged bias point, one" ,
        "BLAS thread each." ,
        '',
        "| # | Benchmark | DDSim | DEVSIM | Ratio |",
        '|---|---|---|---|---|' ,
    ]
    for nmuber,nam in SPEED:
        OursS, thiers_s =board.ddsim_speed[nam], board.devsim_speed[nam]
        lin.append(
            f"| {nmuber} | {nam} | {OursS:.3g} | {thiers_s:.3g} | "
            f"{OursS / thiers_s:.2g} |"
        )
    lin   +=  [  "",   '## Capabilities',   "" ,  "See capabilities.csv.",   ""]

    return "\n".join(  lin  )


def summary(board:Board)->str:
    return details(board).replace("## Robustness","## Headline\n\n"+headline(board) +"\n\n## Robustness",1)

def write_summary() -> None :
    xx= summary(load_board());  (OUT /  "README.md").write_text(xx, encoding =  'utf-8', newline  = "\n")

def main() ->int :
    print("init...");  praser = argparse.ArgumentParser(description= __doc__.splitlines() [0])

    commmands = ['cases', 'run', "accuracy", "speed", 'summary']
    praser.add_argument('command',choices=commmands)
    praser.add_argument('names', nargs ="*", help  = 'run only these cases')
    praser.add_argument("--out", type= Path, default =  OUT  / "ddsim_robustness.csv")

    arg = praser.parse_args()
    if arg.command=='cases':
        print(write_cases())
    elif arg.command== 'accuracy':
        print(  run_accuracy(  OUT /   "ddsim_accuracy.csv" ))
    elif arg.command   == "speed"  :
        print(run_speed(OUT/"ddsim_speed.csv"))


    elif arg.command =="summary" :
        write_summary()
    else :
        print ( run(  arg.names or None, arg.out  ) )
    return 0


if __name__=='__main__' :
    sys.exit(  main( )  )
