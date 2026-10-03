from __future__ import annotations
import argparse, csv, datetime; import json
import math;import os; import platform, subprocess, sys;import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np, scipy
from ddsim.api.devices import build_from_spec, device_parameters
from ddsim.device.builder import Device
from ddsim.mesh.mesh2d import Mesh2D
from ddsim.device.state import DeviceState ; from ddsim.device.transport import TransportModels,solve_bias_ramped
from ddsim.extract.cv import cv_sweep
from ddsim.extract.iv import gate_sweep, iv_sweep, terminal_currents
from ddsim.extract.rolloff import REFERENCE_CURRENT

ROOT =Path(__file__).resolve().parents[1]



OUT=  ROOT/  'data' /  'scoreboard'

SEED = 20260924

SPLIT  =  (( "pn_diode",  60) , (  "mos_cap",  40 ) ,   (  'nmos' ,   100 ) )


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



PREFIX= {"pn_diode" :'d', "mos_cap" : "c", 'nmos' :"m"}

SIGNIFICANT =  4

@dataclass(frozen= True)

class Case :
    name  : str

    device :str

    knobs:dict[str,float]
    redraws : int


def _draw(rng  : np.random.Generator, low  :float, high : float, axis  : str) -> float  :

    if axis ==  "log"  :
        m = -1.0 if high <0 else 1.0
        d,r= sorted((abs(low),abs(high)))
        t= m *math.exp(rng.uniform(math.log(d), math.log(r)))
    else :
        t =  rng.uniform(low, high)
    return float(f"{t:.{SIGNIFICANT}g}")


def _builds(device: str,knobs:dict[str,Any]) -> bool:
    try  :
        build_from_spec(device , knobs  )
    except ValueError :
        return  False
    return True
def draw_cases()-> list[Case]:
    k =np.random.default_rng(SEED)
    a=[]

    for  a2 ,   hh  in SPLIT :
        cur = {p.name : p for p in device_parameters(a2)}

        for c in range(1, hh + 1)  :
            it  =  0
            while True :
                obj = {}
                for b in PHYSICAL[a2] :
                    p =cur[b] ; assert p.low is not None and p.high is not None,b
                    obj[b] =_draw(k, p.low, p.high, p.axis)


                if _builds(a2, obj) :
                    break
                it +=1
            b = f"{PREFIX[a2]}{c:03d}";a.append(Case(b,a2,obj,it))
    return a


def  write_cases (  ) ->  Path  :
    OUT.mkdir(parents  =  True, exist_ok =   True);  row=OUT/'cases.csv'
    buf = [
        f"# written {datetime.date.today().isoformat()} by tools/scoreboard.py cases",
        f"# seed {SEED}, split {SPLIT}, {SIGNIFICANT} significant digits" ,
        '# knobs not listed keep the constructor default',
        'case,device,knobs,redraws',
    ]
    for a in draw_cases():
        it = json.dumps(a.knobs).replace('"', '""')
        buf.append(f'{a.name},{a.device},"{it}",{a.redraws}')
    row.write_text("\n".join(buf)+"\n",encoding ='utf-8',newline ="\n")
    return row

THREAD_VARIABLES=('MKL_NUM_THREADS',"OMP_NUM_THREADS","OPENBLAS_NUM_THREADS")

FINE_STEP =0.02

BIAS ={"pn_diode":'anode_voltage',"mos_cap" :"gate_voltage",'nmos': "gate_voltage"}

MEASURED= {"pn_diode":'anode',"nmos": "drain"}



@dataclass(frozen = True)

class  Result   :
    case : str
    driver : str
    converged  :  bool
    value  :  float
    imbalance: float
    largest : float
    seconds : float
    message   :   str

def read_cases() -> list[Case] :
    val= (OUT/'cases.csv').read_text(encoding ="utf-8").splitlines()
    w  =   [  tmp2 for  tmp2 in  val  if  not  tmp2.startswith(  "#")  ]
    return[
        Case(b['case'], b["device"], json.loads(b["knobs"]), int(b["redraws"]))
        for b in csv.DictReader(w)
    ]


def _models(device : Device, kind : str)  -> TransportModels :
    if kind=="nmos" :
        return TransportModels.for_device(device, mobility =  "arora", field_dependent= True, surface  = True)
    return TransportModels.for_device(device)
def _public(device   :   Device, case  :  Case,  models  :  TransportModels )  ->  DeviceState  |  None   :
    y =  case.knobs [BIAS [case.device  ]]

    if case.device  ==  'pn_diode' :
        f  =   iv_sweep( device,   "anode",   [y ], models =   models  )
    else:
        f  =gate_sweep(device, [y], models  = models)


    return f.points[  - 1  ].state if f.complete  else None

def solve_case(case:Case,fine:bool)->Result:

    yy = "ddsim_fine" if fine else 'ddsim'
    j=build_from_spec(case.device,case.knobs)
    tmp3 =time.perf_counter()
    try:

        if case.device  ==   "mos_cap" :
            d  = cv_sweep(j, "gate", [case.knobs["gate_voltage"]])
            dd=time.perf_counter()-tmp3


            lst  =  d.points[-  1].capacitance if d.complete else math.nan
            return Result(
                case.name, yy ,   d.complete ,  lst,  0.0 ,  0.0,  dd ,  d.message
            )

        y   = _models(j,  case.device)
        if fine :
            a2:DeviceState|None =solve_bias_ramped(
                j,y,step=FINE_STEP
            )
        else  :
            a2  =  _public(j, case, y)
        dd = time.perf_counter() -tmp3
    except Exception as f:
        dd=time.perf_counter() - tmp3
        return Result(case.name, yy, False, math.nan, math.nan, math.nan, dd, str(f) [:200],)
    if a2 is None:
        return  Result (
            case.name,
            yy ,
            False,
            math.nan,
            math.nan ,
            math.nan ,
            dd,
            'sweep stopped early' ,
        )
    s =terminal_currents(j,a2,y)
    return Result(case.name, yy, True, s[MEASURED[case.device]], abs(sum(s.values())), max(abs(u)for u in s.values()), dd, "",)
def _git_sha()->str  :

    try  :


        res  =  subprocess.run ([  'git' ,  "rev-parse", "--short",  'HEAD'  ] , cwd  =   ROOT, capture_output = True, text  =   True,)
        return res.stdout.strip() or "unknown"
    except OSError :
        return "unknown"
def run(names  : list[str] | None, path : Path) -> Path :

    print('--- STAGE 2 REACHED ---',path)
    dat   =   [ a for a in THREAD_VARIABLES  if  a in os.environ ]
    if  dat   :
        raise SystemExit(
            f"unset {', '.join(dat)} first, robustness runs at each tool's default threading, see references/decisions.md 2026-09-26"
        )
    x =[y for y in read_cases()if names is None or y.name in names]
    e  =  [
        f"# written {datetime.date.today().isoformat()} by tools/scoreboard.py run",
        f"# ddsim {_git_sha()}, python {platform.python_version()}, "
        f"numpy {np.__version__}, scipy {scipy.__version__}",
        f"# {platform.platform()}, {platform.processor()}, default BLAS threads" ,
        f"# fine step {FINE_STEP} of every bias, from zero",
        "case,driver,converged,value,imbalance,largest,seconds,message",
    ]
    with  path.open("w",  encoding  = "utf-8" ,   newline =  "\n" )  as w2  :

        w2.write("\n".join(e) +"\n")
        for x2 in x :
            for v in(False, True):
                yy = solve_case(x2, v)
                h=yy.message.replace('"',"'").replace("\n",' ')
                w2.write (
                    f"{yy.case},{yy.driver},{yy.converged},{yy.value!r},{yy.imbalance!r},"
                    f'{yy.largest!r},{yy.seconds:.3f},"{h}"\n'
                )
                w2.flush ()
                print(
                    f"{yy.case} {yy.driver} {yy.converged} {yy.value:.4g} {yy.seconds:.1f}s"
                )
    return  path


TOLERANCE= {"pn_diode"  :  0.02, 'mos_cap' : 0.02, 'nmos' :  0.05}

DIODE_FLOOR  = 1e-10


MOSFET_FLOOR=1e-5

MOSFET_BALANCE   = 1e-4

NOISE_FACTOR= 3.0



REFERENCE_DRIVERS =('ddsim_fine', 'devsim_fine')



def floor(  case   :  Case )   ->  float   :

    if case.device == "pn_diode" :
        return DIODE_FLOOR
    if  case.device  == "nmos" :
        return MOSFET_FLOOR *REFERENCE_CURRENT / case.knobs['L_gate']
    return 0.0




def balance_line(case :Case)->float :
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
    xx  =  floor (case )
    if abs(a.value) <  xx and abs(b.value)  < xx :
        return True
    nxt= TOLERANCE[case.device]*max(abs(a.value), abs(b.value))
    nxt +=  NOISE_FACTOR*max(a.imbalance, b.imbalance)
    return abs(a.value  - b.value)  <=nxt

@dataclass(  frozen  =  True  )




class Score  :
    passes: dict[str, int]
    dropped :  dict[str, str]

    counted :   int


def  score(cases   :   list[Case  ],   results  :   list[  Result]  )   ->   Score   :
    vv  :dict[str, dict[str, Result]] = {}

    for m in results  :
        vv.setdefault(m.case,{}) [m.driver]=m
    z2 : dict [str,  int]  = {  }


    y  : dict[  str ,  str]   =   {}
    for f in cases :
        g  =  vv.get(f.name, {})
        i =  [
            g[out]  for  out  in REFERENCE_DRIVERS  if out in g  and  valid( f,   g[ out  ] )
        ]
        if not i  :
            y[f.name]= 'no valid reference'
            continue
        if not all(agree(f,x,c) for x in i for c in i):

            y[f.name] = "the references disagree"
            continue
        for  bb ,   m in  g.items ()  :
            if valid(f, m) and all(agree(f, m, c2)  for c2 in i) :
                z2[bb] =  z2.get(bb, 0) +  1
    return Score(z2, y, len(cases)  -len(y))

REFINEMENTS= (1.0,1.5,2.25)

ORDER_RANGE  = (0.5, 3.0)




@dataclass(frozen =True)


class Fit:

    limit : float |None
    order  :   float  |   None
    error  :  float  |  None
    nodes_for_one_percent: float |  None

def  richardson(levels  : list [tuple [ float,  int,   float  ] ],  dimension : int  )  ->   Fit  :
    (t2, b, w), (ii, _, a), (zz, _, f)  =sorted(levels)
    e =   ii  /   t2
    g,tt=w- a,a-f


    if g   * tt  <=  0.0  or abs ( tt)  >=   abs( g)  :
        return Fit(None, None, None, None)
    foo  =math.log(g / tt) / math.log(e)

    if not ORDER_RANGE[0] <= foo <= ORDER_RANGE[1]  :
        return Fit(None,None,None,None)
    r=f-tt / (e ** foo - 1.0)
    s=  abs(w-r) / abs(r)
    out2 = b  *  ( s /   0.01  )  ** (dimension /  foo)
    return Fit(r,foo,s,out2)

ACCURACY=((1,"diode_1e16_1e16"), (2,'diode_1e18_1e16'), (3,"diode_1e20_1e15"), (4,"mos_cap_5nm"), (5,'mos_cap_20nm'), (6,'nmos_1um'), (7,'nmos_180nm'), (8,"nmos_65nm"), (9,"rolloff_100nm"), (10,'fullstack_100nm'),)

ACCURACY_BIAS : dict[ str ,   Any ]  =  {  'diode'  : 0.6 , 'mos_cap' :  0.0 ,  'mosfet' :  (  1.0,  0.05  )}



def _scaled(nodes  : int, r: float)  ->int  :
    return round((nodes -1)*r)+1

def silicon_nodes(device  :   Device  ) -> int   :
    if device.regions is None:
        return int(device.mesh.n_nodes)
    return int(np.count_nonzero(device.regions.semiconductor_volume > 0.0))

def  accuracy_point (name   : str,   r  : float  ) ->  tuple[  int,  float]  :
    sys.path.insert(0,str(ROOT))
    import inspect
    from ddsim.device.mos_cap import mos_cap
    from ddsim.device.mosfet import nmos
    from  ddsim.device.pn_diode import  pn_diode
    from  ddsim.extract.rolloff import  SHORT_CHANNEL_PROCESS
    from tests.regression.devsim_gen import parameters as P

    if name.startswith("diode") :
        x =P.BY_NAME[name]
        b=pn_diode(
            Na=x.Na,
            Nd =x.Nd,
            length = x.length,
            junction = x.junction,
            n_nodes=_scaled(x.n_nodes,r),
            h_min=x.h_min /r,
        )
        i =iv_sweep(b,"anode",[ACCURACY_BIAS['diode']])
        return silicon_nodes(b),float(i.current[-1])
    if name.startswith('mos_cap'):
        z = {x.name : x for x in P.MOS_BENCHMARKS}  [name]
        b =mos_cap(
            substrate_doping=z.substrate_doping,
            t_ox= z.t_ox,
            t_si=z.t_si,
            n_silicon=_scaled(z.n_silicon,r),
            n_oxide =_scaled(z.n_oxide,r),
            h_min=z.h_min/ r,
            work_function =z.work_function,
        )
        c  =  cv_sweep(b, 'gate', [ACCURACY_BIAS["mos_cap"]])

        assert isinstance(b.mesh, Mesh2D)
        res2 = b.mesh.nx
        return silicon_nodes(b)//res2, float(c.points[- 1].capacitance)
    u= P.MOSFET_BY_NAME[name]
    tmp = u.models == P.FULL_MODELS
    t  =  inspect.signature( nmos).parameters
    v: dict[str, Any] = {y:_scaled(t[y].default, r) for y in("n_contact", 'n_sd', "n_channel", 'n_silicon', 'n_oxide')}
    v["h_min_x"]=t['h_min_x'].default /r
    v['h_min_y'] = t["h_min_y"].default/  r
    k, h = ACCURACY_BIAS['mosfet']; k=min(k, max(u.gate_voltages))


    b = nmos(L_gate=u.L_gate, drain_voltage  =  h, degenerate  = tmp, **SHORT_CHANNEL_PROCESS, ** v,)
    j=(TransportModels.for_device(b,mobility='arora',field_dependent=True,surface=True) if tmp else TransportModels.for_device(b,mobility = "constant"))
    i  =  gate_sweep(b, [k], models = j)
    return silicon_nodes (b), float( i.current[-   1]  )



def run_accuracy(path:  Path)-> Path  :
    print('working...')
    prev=[
        f"# written {datetime.date.today().isoformat()} by tools/scoreboard.py",
        f"# ddsim {_git_sha()}, refinements {REFINEMENTS}, biases {ACCURACY_BIAS}, mosfet gate capped at its golden curve's last point",
        'benchmark,name,refine,nodes,value,seconds',
    ]

    with path.open("w", encoding  ='utf-8', newline =  "\n") as k :
        k.write("\n".join(prev) +"\n")
        for y2, v in ACCURACY :
            for c in REFINEMENTS  :
                h =  time.perf_counter()

                e,thing = accuracy_point(v,c)

                t  =  time.perf_counter( )   -  h
                k.write(f"{y2},{v},{c},{e},{thing!r},{t:.3f}\n"  )
                k.flush();  print(f"{v} r={c} {e} nodes {thing:.8g} {t:.1f}s")
    return path

SPEED_RUNS= 5


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



def speed_sweep(name :str)-> int:
    sys.path.insert(0, str (  ROOT ))
    from ddsim.device.mos_cap import mos_cap
    from ddsim.device.mosfet import nmos
    from ddsim.device.pn_diode import pn_diode
    from ddsim.extract.rolloff import SHORT_CHANNEL_PROCESS
    from tests.regression.devsim_gen import parameters as P


    if name.startswith("diode"):

        ys  = P.BY_NAME[name]
        u= pn_diode(
            Na =  ys.Na,
            Nd= ys.Nd,
            length = ys.length,
            junction =  ys.junction,
            n_nodes =ys.n_nodes,
            h_min = ys.h_min,
        )
        return len(iv_sweep(u, 'anode', list(ys.voltages), step  =  0.05).points)
    if name.startswith('mos_cap')  :
        d  =  { ys.name :  ys  for  ys  in P.MOS_BENCHMARKS}  [ name]
        u =  mos_cap(substrate_doping  =  d.substrate_doping, t_ox  = d.t_ox, t_si  =  d.t_si, n_silicon =d.n_silicon, n_oxide=d.n_oxide, h_min = d.h_min, work_function =d.work_function,)

        return len (cv_sweep( u, "gate", list (d.voltages  ) ).points )
    cur = P.MOSFET_BY_NAME [  name ]
    ok  =0
    for z in(cur.drain_low,cur.drain_high) :
        u   =   nmos(L_gate   =  cur.L_gate , drain_voltage   =  z, degenerate  = False, **  SHORT_CHANNEL_PROCESS ,)
        idx= TransportModels.for_device(u,mobility="constant")
        ok+=len(gate_sweep(u,list(cur.gate_voltages),models= idx).points)
    return ok

def run_speed(path : Path)->Path :


    i=[val for val in THREAD_VARIABLES if os.environ.get(val)!='1']
    if i :
        raise SystemExit(f"set {', '.join(i)} to 1 first, see references/decisions.md 2026-09-26")

    b= [
        f"# written {datetime.date.today().isoformat()} by tools/scoreboard.py",
        f"# ddsim {_git_sha()}, python {platform.python_version()}",
        f"# {platform.platform()}, {platform.processor()}, one BLAS thread",
        "benchmark,name,run,points,seconds",
    ]
    with path.open("w",encoding='utf-8',newline="\n") as cur:

        cur.write("\n".join(b)+ "\n")
        for idx, k in SPEED:


            for e in range(1, SPEED_RUNS  +1) :
                h= time.perf_counter()


                aa =  speed_sweep(k)

                k2  =  time.perf_counter() - h
                cur.write(f"{idx},{k},{e},{aa},{k2:.3f}\n");  cur.flush( )
                print(f"{k} run {e}: {aa} points {k2:.1f}s")

    return  path



def read_results(path :Path) -> list[Result] :
    cc   =  path.read_text(encoding = "utf-8").splitlines (  )
    i =[bar for bar in cc if not bar.startswith('#')]


    return[
        Result(
            x['case'],
            x["driver"],
            x['converged'] =='True',
            float(x["value"]),
            float(x['imbalance']),
            float(x['largest']),
            float(x['seconds']),
            x["message"],
        )
        for x in csv.DictReader(i)
    ]
def _rows(path : Path) -> list[dict[str, str]]:
    b= path.read_text(encoding= 'utf-8').splitlines()
    return list(csv.DictReader(k for k in b if not k.startswith('#')))



def read_accuracy(path  :  Path) ->  dict[str, list[tuple[float, int, float]]] :
    val2 : dict[str, list[tuple[float, int, float]]]= {}
    for m in _rows(path):
        f   =   ( float( m["refine"]  ) ,   int(m[  'nodes'] ),   float (m[ 'value'  ]  )  );  val2.setdefault(m["name"], []).append(f)
    return val2



def dimension(name: str)  -> int :


    return 1 if name.startswith(("diode",'mos_cap'))else 2


def  read_speed(  path  :  Path  )  ->   dict[ str, float ] :

    b :dict[str,list[float]] = {}
    for tmp3 in _rows(path):
        if int(tmp3["points"])==0:
            continue
        r2   =   float (tmp3[  "seconds"]  ) /  int(  tmp3["points" ]  )
        b.setdefault( tmp3 ['name' ],  []).append (r2)
    return{j:float(np.median(c))for j,c in b.items()}
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
    v =  read_results( OUT  / "ddsim_robustness.csv") +  read_results(
        OUT   /  "devsim_robustness.csv"
    )
    buf  = []
    for c in("ddsim","devsim"):
        u= read_accuracy(OUT/ f"{c}_accuracy.csv")

        buf.append({n  : richardson(ys, dimension(n))for n, ys in u.items()})
    return Board(
        score(read_cases(),v),
        buf[0],
        buf[1],
        read_speed(OUT/"ddsim_speed.csv"),
        read_speed(OUT /"devsim_speed.csv"),
        _capabilities(),
    )

def _fewer_nodes(board : Board) -> tuple[int, int, int] :
    mm =ii=cc= 0


    for _, obj in ACCURACY :
        d2  =  board.ddsim_fit[obj].nodes_for_one_percent
        f  = board.devsim_fit[obj ].nodes_for_one_percent
        if d2 is None or f is None:
            cc+= 1
        elif  d2  <   f   :
            mm+= 1


        else  :
            ii +=1

    return mm, ii, cc


def _speed_ratio( board  : Board )   ->  float  :

    info = [board.ddsim_speed[n] /board.devsim_speed[n]for _,n in SPEED if n in board.ddsim_speed and n in board.devsim_speed]

    return float (  np.exp( np.mean( np.log (info ))))


def _estimates(board : Board)->tuple[str, str]  :
    r =  {b2["capability"] : b2 for b2 in board.capabilities}
    k =r["Discretization error estimates"]
    return(
        'every result' if k['ddsim']=="yes" else "none",
        'every result' if k['devsim']=="yes" else "none",
    )

def headline(board:Board) -> str:
    tmp=board.robustness
    v,tt,d= _fewer_nodes(board)

    res=_speed_ratio(board)
    out = board.capabilities
    t =  sum ( j[  "ddsim" ] == "yes"  for  j in out); u =sum(y2["devsim"] =='yes' for y2 in out)
    ok =sum(i['devsim'] =="scripted" for i in out)
    yy =  _estimates(board)
    ii  =   f"{res:.2g}x DEVSIM's" if  res  >=  1.0  else f"{1.0 / res:.2g}x faster"
    x2  =[
        "| Axis | DDSim | DEVSIM 2.11 |",
        "|---|---|---|",
        f"| Robustness: cold solves passed, of {tmp.counted} scored | "
        f"{tmp.passes.get('ddsim', 0)} | stock ramp {tmp.passes.get('devsim_stock', 0)}, "
        f"my ramp {tmp.passes.get('devsim_expert', 0)} |",
        f"| Accuracy: benchmarks reaching 1% on fewer nodes, of {len(ACCURACY)} | "
        f"{v} | {tt} ({d} not in the asymptotic range) |",
        f"| Error estimates reported | {yy[0]} | {yy[1]} |",
        f"| Speed: time per bias point, benchmarks 1 to 8 | {ii} | 1x |",
        f"| Capabilities, of {len(out)} rows | {t} | {u} built in, "
        f"{ok} if you write the equations |",
    ]

    return "\n".join(x2)



def _number(value  : float | None, digits :  int = 3)->  str:
    return "n/a" if value is None else f"{value:.{digits}g}"


def _speed_row(number:int, name:str, ours:dict[str,float], theirs:dict[str,float]) -> str:
    nxt,k = ours.get(name), theirs.get(name)
    u = "n/a" if nxt is None or k is None else f"{nxt / k:.2g}"
    return f"| {number} | {name} | {'failed' if nxt is None else f'{nxt:.3g}'} | {'failed' if k is None else f'{k:.3g}'} | {u} |"


def details( board :   Board )  -> str   :
    ys =board.robustness
    tmp3 = ("ddsim", 'ddsim_fine', "devsim_stock", "devsim_expert", 'devsim_fine')
    v= [
        "# Scoreboard",
        '',
        'Generated by `tools/scoreboard.py summary` from the CSVs in this folder.',
        "Don't edit it by hand: tests/regression/test_scoreboard.py regenerates",
        'it and fails if this file is stale. The rules are the 2026-09-25 and',
        '2026-09-26 rows of references/decisions.md.',
        '',
        "## Robustness",
        "",
        f"{ys.counted} of {ys.counted + len(ys.dropped)} cases scored.",
        "",
        '| Driver | Passed |',
        "|---|---|",
    ]
    v+=[f"| {out2} | {ys.passes.get(out2, 0)} |" for out2 in tmp3]
    v+=["",'Dropped cases, which count for nobody:',""]
    v += [f"- {j}: {row}" for j, row in sorted(ys.dropped.items())]
    if not ys.dropped  :
        v.append("- none")
    v+=  [
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
    b=read_accuracy(OUT/"ddsim_accuracy.csv")
    foo=read_accuracy(OUT/ "devsim_accuracy.csv")
    for u, x in ACCURACY :
        k,r = board.ddsim_fit[x],board.devsim_fit[x]
        a = (None if k.limit is None or r.limit is None else abs(k.limit- r.limit) / abs(r.limit))
        v.append(
            f"| {u} | {x} | {min(b[x])[1]} | "
            f"{_number(k.error)} | {_number(k.order)} | "
            f"{_number(k.nodes_for_one_percent)} | {min(foo[x])[1]} | "
            f"{_number(r.error)} | {_number(r.order)} | "
            f"{_number(r.nodes_for_one_percent)} | {_number(a)} |"
        )
    v   +=  [
        '' ,
        '## Speed',
        "",
        f"Median of {SPEED_RUNS} runs, seconds per converged bias point, one" ,
        "BLAS thread each." ,
        '',
        "| # | Benchmark | DDSim | DEVSIM | Ratio |",
        '|---|---|---|---|---|' ,
    ]
    v+=[_speed_row(u, x, board.ddsim_speed, board.devsim_speed) for u,x in SPEED]
    v   +=  [  "",   '## Capabilities',   "" ,  "See capabilities.csv.",   ""]

    return "\n".join(  v  )


def summary(board:Board)->str:
    return details(board).replace("## Robustness","## Headline\n\n"+headline(board) +"\n\n## Robustness",1)

def write_summary() -> None :
    vv= summary(load_board());  (OUT /  "README.md").write_text(vv, encoding =  'utf-8', newline  = "\n")

def main() ->int :
    print("init...");  w = argparse.ArgumentParser(description="The ddsim side of the Phase 8 scoreboard. See references/decisions.md, 2026-09-25.")

    f = ['cases', 'run', "accuracy", "speed", 'summary']
    w.add_argument('command',choices=f)
    w.add_argument('names', nargs ="*", help  = 'run only these cases')
    w.add_argument("--out", type= Path, default =  OUT  / "ddsim_robustness.csv")

    k2 = w.parse_args()
    if k2.command=='cases':
        print(write_cases())
    elif k2.command== 'accuracy':
        print(  run_accuracy(  OUT /   "ddsim_accuracy.csv" ))
    elif k2.command   == "speed"  :
        print(run_speed(OUT/"ddsim_speed.csv"))


    elif k2.command =="summary" :
        write_summary()
    else :
        print ( run(  k2.names or None, k2.out  ) )
    return 0


if __name__=='__main__' :
    sys.exit(  main( )  )
