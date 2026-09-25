"""The DEVSIM side of the Phase 8 scoreboard.

Runs under `.venv-devsim`, like the other generators here, and writes into
`data/scoreboard/`. See phases/PHASE-8.md.

    .venv-devsim/Scripts/python.exe tests/regression/devsim_gen/scoreboard.py api

`api` writes `devsim_api.txt`: every public name devsim exports, the solve
types and linear solvers its docstring lists as `solve:<type>` and
`solver:<type>`, the element types its Gmsh import accepts as
`create_gmsh_mesh:<element>`, and every function in its bundled
`python_packages` as `python_packages.<module>.<function>`. The capability
matrix cites these names, and test_scoreboard.py checks each citation against
this file, so a claim about what DEVSIM can do is a claim about this snapshot
rather than about my memory of its manual.

`run` solves the robustness cases in `cases.csv` three ways and writes
`devsim_robustness.csv`, in the same columns as ddsim's file:

- `devsim_stock`: DEVSIM's own `python_packages/ramp.py` walks the bias, one
  solve per step, halving on failure and never growing back.
- `devsim_expert`: the driver that produced the golden data for that device
  family, which is the best I know how to write for DEVSIM.
- `devsim_fine`: the expert driver with its step held ten times smaller, as
  the reference.

All three share the physics, the mesh, the equilibrium start and the solver
tolerances, so what differs between them is only how the bias gets walked.
Every device is deleted after its case, since devsim solves every live device
at once (see README.md here).

    MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        .venv-devsim/Scripts/python.exe tests/regression/devsim_gen/scoreboard.py run
"""
from __future__ import annotations

import argparse

import ast, csv

import datetime ; import json ; import math, os, platform


import  re ; import  sys;import time
from typing import Any


import devsim
HERE  = os.path.dirname(  os.path.abspath(  __file__))


ROOT= os.path.abspath(os.path.join(HERE, "..", "..", ".."))

OUT = os.path.join(ROOT,
        'data',
     'scoreboard')
sys.path.insert( 0,  HERE )

sys.path.insert(0, ROOT)

import generate_diodes as GD;import generate_mos_cv as GC, generate_mosfet as GM, parameters  as  P
from devsim.python_packages.ramp import rampbias

def _choices(doc : str| None,
      argument : str)-> list[str] :
    '''The {a, b, c} choices a devsim docstring lists for one argument.'''
    iter   =  re.search(rf"\b{argument} : \{{([^}}]*)\}}",   doc or ""  )
    if  iter is  None  :
        raise RuntimeError(  f"devsim's docstring no longer lists choices for {argument}"  )
    return[T.strip().strip("'") for T in iter.group(1).split(",")]
def solve_types() ->list[str] :
    """solve's `type` and `solver_type` choices, e.g. solve:ac, solver:iterative."""
    Doc  = devsim.solve.__doc__
    return[f"solve:{vals}" for vals in _choices(Doc, "type")]  +  [
        f"solver:{vals}" for vals in _choices(Doc, 'solver_type')
    ]
def gmsh_elements()-> list[str] :
    """The element types create_gmsh_mesh's docstring accepts, e.g. tetrahedron."""

    Found=re.findall(r"- \d+ (\w+)",devsim.create_gmsh_mesh.__doc__ or "")
    if 'triangle' not in Found:
        raise  RuntimeError("create_gmsh_mesh's docstring no longer lists element types")
    return[f"create_gmsh_mesh:{cnt}" for cnt in Found]
def package_functions()->  list[str] :
    """Top level functions in devsim/python_packages, read from source."""
    import devsim.python_packages as packages


    bar=os.path.dirname(packages.__file__)
    naames = []
    for fil in sorted(os.listdir(bar)) :
        if not fil.endswith(".py")  or fil.startswith('__') :
            continue
        with  open(os.path.join(  bar, fil) , encoding   = 'utf-8')  as  bin  :

            tre=ast.parse(bin.read())
        mod   =   fil[ :-  3  ]
        for Node in tre.body:
            if isinstance(Node, ast.FunctionDef)  :
                naames.append(f"python_packages.{mod}.{Node.name}")
    return naames


def write_api( ) ->  str  :
    Public=sorted(n for n in dir(devsim)if not n.startswith('_'))
    tdoay= datetime.date.today().isoformat()
    myvar =[
        f"# devsim {devsim.__version__}",
        f"# written {tdoay} by devsim_gen/scoreboard.py api",
        "# public names, solve types, gmsh elements, python_packages functions",
    ]
    myvar  += Public


    myvar += solve_types()


    myvar += gmsh_elements()
    myvar +=package_functions()
    os.makedirs(OUT,exist_ok=True);  obj2= os.path.join(OUT, "devsim_api.txt")
    with open(obj2, 'w', encoding = "utf-8", newline = "\n")  as ff :
        ff.write(  "\n".join( myvar  ) +  "\n"  )
    return  obj2

THREAD_VARIABLES=('MKL_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')


DRIVERS = ("devsim_stock", 'devsim_expert', 'devsim_fine')

FINE  =  10.0
"""How much smaller the reference's step is than the expert's [1]."""


CV_STEP =  0.005
'''Half the gate span of DEVSIM's central difference capacitance [V]. ddsim
takes the derivative exactly, so this is kept small enough that its
truncation error sits far under the 2 percent the comparison allows.'''


def read_cases( )   ->  list[ dict [str , Any  ] ] :
    with open(os.path.join(OUT,'cases.csv'),encoding="utf-8")as ff :
        bdy=[myvar for myvar in ff.read().splitlines()if not myvar.startswith("#")]
    return[{"case":abs['case'], "device" : abs['device'], "knobs" :json.loads(abs['knobs']),} for abs in csv.DictReader(bdy)]




def _noop(device : str)->None:
    return None



def _stock_ramp ( device  :  str, contact   : str, target   :  float,   step :   float)   ->   None  :
    """DEVSIM's shipped ramp, at the expert driver's tolerances and budget."""
    rampbias(device,contact,target,step,1e-4,100,1e-8,1e30,_noop)

def  _cleanup(  ) -> None :
    """Delete every device and mesh. devsim solves every live device at once."""
    for Name in list(devsim.get_device_list()) :
        devsim.delete_device(  device  =  Name  )
    for Name in list(devsim.get_mesh_list()) :
        devsim.delete_mesh (  mesh   =  Name )


def _diode(case:dict[str,Any],driver:str)-> tuple[float,float,float]:
    K  =  case['knobs']
    zz = P.DiodeBenchmark(
        name   =   f"{case['case']}_{driver}",
        number =  0,
        Na   =   K[  'Na'  ] ,
        Nd  = K[ "Nd" ],
        length  =  K ["length" ],
        junction   =  K [ 'junction'],
        voltages =  (K[ "anode_voltage" ],  ),
        tolerance = 0.02,
    )
    Device= zz.name
    Target= K['anode_voltage']
    with GM.quiet() :
        GD.build_mesh(zz,
                       Device)
        GD.set_doping(zz, Device)
        GD.set_silicon_parameters(Device)
        GD.build_physics(Device)
        if driver=="devsim_stock":
            _stock_ramp(Device,GD.ANODE,Target,0.05)
        else :


            Step  =   0.05 if  driver  == 'devsim_expert' else  0.05  /  FINE
            GD.ramp_to(Device, Target, 0.0, step = Step)
    x2  =  GD.anode_current(Device  )
    cnt= GD.cathode_current(Device)
    return x2, abs(x2 +  cnt), max(abs(x2), abs(cnt))

def _mos_cap(case :  dict[str, Any], driver  :  str)->tuple[float, float, float] :
    kk = case['knobs']
    Bench   =  P.MosBenchmark(
        name   =  f"{case['case']}_{driver}",
        number =  0,
        substrate_doping =   kk[ "substrate_doping"],
        t_ox =  kk ['t_ox'],
        t_si  =   kk[  "t_si" ],
        work_function   = kk[  'work_function' ],
        voltages  = (kk [  "gate_voltage"], ),
        tolerance  =   0.02,
    )
    Device =  Bench.name
    trget =kk["gate_voltage"]

    def gate(v :float)->None:
        devsim.set_parameter(
            device  = Device, name  =  f"{GC.GATE}_bias" ,  value =  GC.gate_potential(  Bench,  v )
        )
        devsim.solve(type='dc', absolute_error=1e-10, relative_error=1e-12, maximum_iterations =100,)


    with GM.quiet():
        GC.build_mesh( Bench , Device )

        GC.set_material_parameters(Device)
        GC.build_physics(Bench,Device)
        sttart =trget -CV_STEP
        if driver=='devsim_stock':
            gate(0.0)

            endd= GC.gate_potential(Bench,sttart)

            rampbias(Device, GC.GATE, endd, 0.1, 1e-4, 100, 1e-12, 1e-10, _noop)
        elif driver   ==  'devsim_expert'   :
            gate(sttart)
        else :
            Steps   =  max(1, math.ceil( abs(  sttart  )  /   0.01  ))
            for inedx in range(1,Steps +1) :
                gate(sttart*inedx/Steps)
        Charges =[]
        for v in(sttart, trget+ CV_STEP)  :
            gate(v)
            Charges.append (devsim.get_contact_charge(device =   Device ,   contact   =   GC.GATE, equation  = 'PotentialEquation'))
    foo =  (Charges[1] -Charges[0])  / (2.0 *  CV_STEP)
    return foo, 0.0, 0.0


def _nmos(case : dict[str,Any],driver: str)->tuple[float,float,float]:


    kk=  case['knobs']
    x2  = dict(P.MOSFET_PROCESS)
    for keyy in("substrate_doping", "sd_peak", 'x_j', "lateral_diffusion", "t_ox")  :
        x2[keyy]=kk[keyy]
    stuff2  = kk['L_gate']
    hSurface  = 6.25e-9
    ben   =  P.MosfetBenchmark(
        name  =  f"{case['case']}_{driver}",
        number   =  0 ,
        L_gate  = stuff2,
        gate_voltages  = (  kk[ 'gate_voltage'] ,  ),
        drain_low   =  kk["drain_voltage"  ] ,
        drain_high  =   kk ["drain_voltage"],
        tolerance  =  0.05,
        devsim_h_junction  =   min ( stuff2  / 80.0,  5e-7) ,
        devsim_h_channel = min( stuff2  /  40.0,   2e-6 ) ,
        devsim_h_surface   =   hSurface,
        devsim_h_depth   =  P.implant_shape(  x2) [0  ] *  P.H_DEPTH_SIGMAS,
        devsim_oxide_cells = min (  64,   max(  4,  round( x2 [  't_ox' ]  /  hSurface )) ),
        models = P.FULL_MODELS,
    )
    myvar= ben.name
    wff   =  kk ["work_function" ]
    gte  = GM.gate_potential(kk['gate_voltage' ] ,
                     wff)
    drian=kk["drain_voltage"]

    GM._surface_is_live = False
    try :
        with  GM.quiet ()  :
            GM.build_mesh(ben, myvar, process  = x2)


            GM.set_material_parameters(myvar)
            GM.set_doping(ben, myvar, process =x2)
            if driver  == 'devsim_expert' :
                GM.build_physics(myvar, 0.0, drian, P.FULL_MODELS, wff)
                GM.ramp_to(myvar, GM.GATE, gte)
            else :
                GM.build_physics(myvar, 0.0, 0.0, P.FULL_MODELS, wff)
                if driver==  "devsim_stock" :


                    _stock_ramp(myvar,GM.DRAIN,drian,0.1)
                    _stock_ramp(myvar, GM.GATE, gte, 0.1)
                else:
                    smmall =0.1 /  FINE
                    GM.ramp_to ( myvar, GM.DRAIN, drian , step  = smmall, max_step =  smmall)
                    GM.ramp_to(myvar,GM.GATE,gte,step =smmall,max_step=smmall)
            GM.settle(myvar, balance_tol =  GM.BALANCE_TOL)
        tmp= [
            GM.terminal_current(myvar,junk)for junk in(GM.DRAIN,GM.SOURCE,GM.BODY)
        ]
    finally:

        GM._surface_is_live  =  False
    return tmp[0], abs(sum(tmp)), max(abs(junk)for junk in tmp)

SOLVERS = {"pn_diode" :_diode, "mos_cap" : _mos_cap, "nmos"  :  _nmos}


def  _done (  path  : str ) ->  set[ tuple[str ,   str]  ] :
    """The (case, driver) pairs a previous run already wrote to path."""

    if not os.path.exists(path):
        return set ( )

    with open(path, encoding = "utf-8") as ff  :
        boody   =   [s2  for s2 in  ff.read (  ).splitlines (  )  if  not  s2.startswith( "#"  )]
    return{ ( Row[ "case"  ],   Row[ 'driver'  ] ) for Row  in csv.DictReader (  boody  ) }



def run(names:list[str] |None,path:str,resume: bool = False) ->str:
    print( "resume??", resume , len( names  or [  ] )  ) ; Unpinned=[V for V in THREAD_VARIABLES if os.environ.get(V)!='1']
    if Unpinned   :
        raise  SystemExit(f"set {', '.join(Unpinned)} to 1 first, see phases/PHASE-8.md item 5")
    casses  =  [  cc for  cc  in read_cases ( )  if names  is None  or cc[ "case"] in names  ]
    iter = _done(path)  if resume else set();  tod=  datetime.date.today().isoformat()
    hea  = [
        f"# written {tod} by devsim_gen/scoreboard.py run",
        f"# devsim {devsim.__version__}, python {platform.python_version()}",
        f"# {platform.platform()}, {platform.processor()}, one BLAS thread",
        f"# fine step {1.0 / FINE:g} of the expert step",
        'case,driver,converged,value,imbalance,largest,seconds,message',
    ]


    with open(path,'a' if iter else "w",encoding='utf-8',newline="\n") as F:
        if  not iter   :
            F.write("\n".join(hea)+ "\n")

        for Case in casses :
            for myvar in DRIVERS  :
                if(Case['case'], myvar) in iter :
                    continue
                sta =   time.perf_counter( )
                try :
                    vallue, dir, max  =SOLVERS[Case["device"]]  (Case, myvar)
                    ord, Message =True, ''
                except Exception as zip  :
                    vallue  = dir = max=math.nan
                    ord=False
                    Message =  str(zip).replace("\n", ' ').replace('"', "'") [: 200]


                finally :
                    _cleanup()
                sec  =   time.perf_counter () -  sta
                F.write(
                    f"{Case['case']},{myvar},{ord},{vallue!r},{dir!r},"
                    f'{max!r},{sec:.3f},"{Message}"\n'
                )
                F.flush()
                print(f"{Case['case']} {myvar} {ord} {vallue:.4g} {sec:.1f}s")

    return path
REFINEMENTS= (1.0,1.5,2.25)
"""The accuracy axis's refinements, the same three tools/scoreboard.py uses."""
ACCURACY= (
    (1, "diode_1e16_1e16"),
    (2, 'diode_1e18_1e16'),
    (3, "diode_1e20_1e15"),
    (4, 'mos_cap_5nm'),
    (5, "mos_cap_20nm"),
    (6, "nmos_1um"),
    (7, "nmos_180nm"),
    (8, "nmos_65nm"),
    (9, "rolloff_100nm"),
    (10, 'fullstack_100nm'),
)

"""The same benchmark devices, quantities and biases as tools/scoreboard.py:
diode anode current at 0.6 V, MOS capacitor capacitance at 0 V, MOSFET drain
current at 1.0 V of gate and 50 mV of drain."""


def _silicon_nodes(device :str,region :str)-> int:
    return  len(  devsim.get_node_model_values ( device   =  device , region = region ,  name   =  "x"  ) )

def accuracy_point(name  :  str, r  :  float)-> tuple[int, float] :
    """One benchmark's quantity on DEVSIM's reference mesh refined by r."""
    if name.startswith("diode") :
        Bench=P.BY_NAME[name]
        ret = f"{name}_acc"
        with GM.quiet() :
            GD.build_mesh(Bench, ret, refine  = r)
            GD.set_doping(Bench,
                 ret)
            GD.set_silicon_parameters(ret)


            GD.build_physics(ret)
            GD.ramp_to (  ret ,   0.6 ,  0.0,   step  = 0.05)
        return _silicon_nodes(ret,GD.REGION),GD.anode_current(ret)
    if name.startswith('mos_cap')  :
        Bench  =  {  bb.name  :   bb for  bb  in P.MOS_BENCHMARKS} [name ]

        ret =  f"{name}_acc"
        cha =[]
        with GM.quiet():
            GC.build_mesh(Bench,ret,refine=r)

            GC.set_material_parameters(ret)
            GC.build_physics( Bench,   ret  )

            for V in(- CV_STEP , CV_STEP ) :
                devsim.set_parameter(
                    device=  ret,
                    name  = f"{GC.GATE}_bias",
                    value =GC.gate_potential(Bench, V),
                )
                devsim.solve(
                    type="dc",
                    absolute_error=1e-10,
                    relative_error= 1e-12,
                    maximum_iterations =100,
                )


                cha.append(
                    devsim.get_contact_charge(
                        device= ret, contact  = GC.GATE, equation = "PotentialEquation"
                    )
                )


        Capacitance =  (cha[1]-cha[0]) /(2.0 *  CV_STEP);return _silicon_nodes(ret, GC.SILICON), Capacitance
    import dataclasses
    gold = P.MOSFET_BY_NAME[name].gate_voltages; stp=round(gold[1] - gold[0], 10)
    vg=[v for v in gold if v <= 1.0 + 1e-9]
    while vg[-1] < 1.0 - 1e-9 :  # walk it like the golden run or nmos_1um hangs
        vg.append(round(vg[-1] + stp, 10))
    Bench=dataclasses.replace(P.MOSFET_BY_NAME[name], gate_voltages=  tuple(vg))
    pow ,   s2  =  GM.transfer_curve(  Bench , 0.05,  refine  =   r )
    return s2,pow[-1] ["drain"]


def  run_accuracy(path   :   str )   ->  str  :
    print( "working..." )

    min= datetime.date.today().isoformat()

    Header  = [
        f"# written {min} by devsim_gen/scoreboard.py accuracy" ,
        f"# devsim {devsim.__version__}, refinements {REFINEMENTS}",
        "benchmark,name,refine,nodes,value,seconds",
    ]
    with open (  path,  'w', encoding  =  'utf-8',   newline  =  "\n"  ) as ff  :
        ff.write("\n".join(Header)+"\n")

        for nmber, Name in ACCURACY :


            for R in REFINEMENTS  :

                type = time.perf_counter()
                try  :
                    noodes,stuff2 =accuracy_point(Name,R)

                finally :
                    _cleanup()

                tuple = time.perf_counter()- type
                ff.write(f"{nmber},{Name},{R},{noodes},{stuff2!r},{tuple:.3f}\n")

                ff.flush()
                print(f"{Name} r={R} {noodes} nodes {stuff2:.8g} {tuple:.1f}s")
    return path

SPEED_RUNS=5

SPEED=(
    (1,'diode_1e16_1e16'),
    (2,'diode_1e18_1e16'),
    (3,"diode_1e20_1e15"),
    (4,'mos_cap_5nm'),
    (5,"mos_cap_20nm"),
    (6,'nmos_1um'),
    (7,'nmos_180nm'),
    (8,'nmos_65nm'),
)

"""Benchmark sweeps 1 to 8, exactly as the golden generators run them, with
no mesh convergence check."""


def speed_sweep(name :  str)  -> int  :
    """Run one benchmark's golden sweep and return the points reached."""
    if name.startswith("diode"):
        with GM.quiet() :
            return len(GD.sweep(P.BY_NAME[name]))
    if  name.startswith("mos_cap"  )  :


        Bench  ={B.name :B for B in P.MOS_BENCHMARKS}  [name]
        with GM.quiet() :
            return len(GC.sweep(Bench))

    Low, High, _ = GM.sweep(P.MOSFET_BY_NAME[name])
    return len(Low)+len(High)


def run_speed(path:  str) ->  str :
    arr = [tuple for tuple in THREAD_VARIABLES if os.environ.get(tuple)!="1"]
    if arr :
        raise SystemExit(f"set {', '.join(arr)} to 1 first, see phases/PHASE-8.md item 5")
    tod =  datetime.date.today().isoformat()
    headder= [
        f"# written {tod} by devsim_gen/scoreboard.py speed",
        f"# devsim {devsim.__version__}, python {platform.python_version()}",
        f"# {platform.platform()}, {platform.processor()}, one BLAS thread",
        'benchmark,name,run,points,seconds',
    ]
    with open(path,'w',encoding = "utf-8",newline= "\n")as F:
        F.write("\n".join(headder)  +  "\n")
        for yy, nam in SPEED :
            for ri in range(1,SPEED_RUNS+1):
                sta=time.perf_counter()
                try :
                    poi = speed_sweep(nam)
                finally :


                    _cleanup()
                Seconds =time.perf_counter()  - sta; F.write(f"{yy},{nam},{ri},{poi},{Seconds:.3f}\n")
                F.flush ( )
                print(f"{nam} run {ri}: {poi} points {Seconds:.1f}s")

    return path
def main() -> int :

    arr  =   argparse.ArgumentParser( description  = "The DEVSIM side of the Phase 8 scoreboard." )
    arr.add_argument ( "command" ,  choices  =  [  'api', "run",  "accuracy" , "speed"] )
    arr.add_argument('names', nargs=  "*", help  =  'run only these cases')
    arr.add_argument (  "--out",   default  =   None)
    arr.add_argument('--resume',action='store_true',help= "keep finished rows")
    Args   =  arr.parse_args ( )
    if Args.command =="api"  :
        print(write_api())
    elif Args.command ==  'accuracy' :
        print(run_accuracy(Args.out or os.path.join(OUT,'devsim_accuracy.csv')))

    elif Args.command=="speed" :
        print(run_speed(Args.out or os.path.join(OUT,'devsim_speed.csv')))
    else:

        obj2  =  Args.out  or os.path.join( OUT, 'devsim_robustness.csv'  ); print(run(Args.names or None, obj2, resume  =  Args.resume))
    return 0
if __name__ ==  '__main__'  :
    sys.exit (main(  )  )
