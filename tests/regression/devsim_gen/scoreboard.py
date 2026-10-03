from __future__ import annotations
import argparse
import ast, csv
import datetime ; import json ; import math, os, platform
import re ; import sys;import time
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
    ii   =  re.search(rf"\b{argument} : \{{([^}}]*)\}}",   doc or ""  )
    if  ii is  None  :
        raise RuntimeError(  f"devsim's docstring no longer lists choices for {argument}"  )
    return[ok.strip().strip("'") for ok in ii.group(1).split(",")]
def solve_types() ->list[str] :
    m  = devsim.solve.__doc__
    return[f"solve:{w2}" for w2 in _choices(m, "type")]  +  [
        f"solver:{w2}" for w2 in _choices(m, 'solver_type')
    ]
def gmsh_elements()-> list[str] :

    obj=re.findall(r"- \d+ (\w+)",devsim.create_gmsh_mesh.__doc__ or "")
    if 'triangle' not in obj:
        raise  RuntimeError("create_gmsh_mesh's docstring no longer lists element types")
    return[f"create_gmsh_mesh:{z2}" for z2 in obj]
def package_functions()->  list[str] :
    import devsim.python_packages as packages


    val=os.path.dirname(packages.__file__)
    m = []
    for t2 in sorted(os.listdir(val)) :
        if not t2.endswith(".py")  or t2.startswith('__') :
            continue
        with  open(os.path.join(  val, t2) , encoding   = 'utf-8')  as  aa  :

            z=ast.parse(aa.read())
        prev   =   t2[ :-  3  ]
        for item in z.body:
            if isinstance(item, ast.FunctionDef)  :
                m.append(f"python_packages.{prev}.{item.name}")
    return m


def write_api( ) ->  str  :
    idx=sorted(n for n in dir(devsim)if not n.startswith('_'))
    g= datetime.date.today().isoformat()
    d =[
        f"# devsim {devsim.__version__}",
        f"# written {g} by devsim_gen/scoreboard.py api",
        "# public names, solve types, gmsh elements, python_packages functions",
    ]
    d  += idx


    d += solve_types()


    d += gmsh_elements()
    d +=package_functions()
    os.makedirs(OUT,exist_ok=True);  v= os.path.join(OUT, "devsim_api.txt")
    with open(v, 'w', encoding = "utf-8", newline = "\n")  as h :
        h.write(  "\n".join( d  ) +  "\n"  )
    return  v

THREAD_VARIABLES=('MKL_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')


DRIVERS = ("devsim_stock", 'devsim_expert', 'devsim_fine')

FINE  =  10.0


CV_STEP =  0.005


def read_cases( )   ->  list[ dict [str , Any  ] ] :
    with open(os.path.join(OUT,'cases.csv'),encoding="utf-8")as k :
        c=[j for j in k.read().splitlines()if not j.startswith("#")]
    return[{"case":g['case'], "device" : g['device'], "knobs" :json.loads(g['knobs']),} for g in csv.DictReader(c)]




def _noop(device : str)->None:
    return None



def _stock_ramp ( device  :  str, contact   : str, target   :  float,   step :   float)   ->   None  :
    rampbias(device,contact,target,step,1e-4,100,1e-8,1e30,_noop)

def  _cleanup(  ) -> None :
    for zz in list(devsim.get_device_list()) :
        devsim.delete_device(  device  =  zz  )
    for zz in list(devsim.get_mesh_list()) :
        devsim.delete_mesh (  mesh   =  zz )


def _diode(case:dict[str,Any],driver:str)-> tuple[float,float,float]:
    w  =  case['knobs']
    c = P.DiodeBenchmark(
        name   =   f"{case['case']}_{driver}",
        number =  0,
        Na   =   w[  'Na'  ] ,
        Nd  = w[ "Nd" ],
        length  =  w ["length" ],
        junction   =  w [ 'junction'],
        voltages =  (w[ "anode_voltage" ],  ),
        tolerance = 0.02,
    )
    y= c.name
    ys= w['anode_voltage']
    with GM.quiet() :
        GD.build_mesh(c,
                       y)
        GD.set_doping(c, y)
        GD.set_silicon_parameters(y)
        GD.build_physics(y)
        if driver=="devsim_stock":
            _stock_ramp(y,GD.ANODE,ys,0.05)
        else :


            v  =   0.05 if  driver  == 'devsim_expert' else  0.05  /  FINE
            GD.ramp_to(y, ys, 0.0, step = v)
    el  =  GD.anode_current(y  )
    t= GD.cathode_current(y)
    return el, abs(el +  t), max(abs(el), abs(t))

def _mos_cap(case :  dict[str, Any], driver  :  str)->tuple[float, float, float] :
    xs = case['knobs']
    m   =  P.MosBenchmark(
        name   =  f"{case['case']}_{driver}",
        number =  0,
        substrate_doping =   xs[ "substrate_doping"],
        t_ox =  xs ['t_ox'],
        t_si  =   xs[  "t_si" ],
        work_function   = xs[  'work_function' ],
        voltages  = (xs [  "gate_voltage"], ),
        tolerance  =   0.02,
    )
    g =  m.name
    ss =xs["gate_voltage"]

    def gate(v :float)->None:
        devsim.set_parameter(
            device  = g, name  =  f"{GC.GATE}_bias" ,  value =  GC.gate_potential(  m,  v )
        )
        devsim.solve(type='dc', absolute_error=1e-10, relative_error=1e-12, maximum_iterations =100,)


    with GM.quiet():
        GC.build_mesh( m , g )

        GC.set_material_parameters(g)
        GC.build_physics(m,g)
        vv =ss -CV_STEP
        if driver=='devsim_stock':
            gate(0.0)

            s= GC.gate_potential(m,vv)

            rampbias(g, GC.GATE, s, 0.1, 1e-4, 100, 1e-12, 1e-10, _noop)
        elif driver   ==  'devsim_expert'   :
            gate(vv)
        else :
            el   =  max(1, math.ceil( abs(  vv  )  /   0.01  ))
            for arr in range(1,el +1) :
                gate(vv*arr/el)
        tmp =[]
        for v in(vv, ss+ CV_STEP)  :
            gate(v)
            tmp.append (devsim.get_contact_charge(device =   g ,   contact   =   GC.GATE, equation  = 'PotentialEquation'))
    b =  (tmp[1] -tmp[0])  / (2.0 *  CV_STEP)
    return b, 0.0, 0.0


def _nmos(case : dict[str,Any],driver: str)->tuple[float,float,float]:


    idx=  case['knobs']
    t  = dict(P.MOSFET_PROCESS)
    for m in("substrate_doping", "sd_peak", 'x_j', "lateral_diffusion", "t_ox")  :
        t[m]=idx[m]
    i  = idx['L_gate']
    rr  = 6.25e-9
    w   =  P.MosfetBenchmark(
        name  =  f"{case['case']}_{driver}",
        number   =  0 ,
        L_gate  = i,
        gate_voltages  = (  idx[ 'gate_voltage'] ,  ),
        drain_low   =  idx["drain_voltage"  ] ,
        drain_high  =   idx ["drain_voltage"],
        tolerance  =  0.05,
        devsim_h_junction  =   min ( i  / 80.0,  5e-7) ,
        devsim_h_channel = min( i  /  40.0,   2e-6 ) ,
        devsim_h_surface   =   rr,
        devsim_h_depth   =  P.implant_shape(  t) [0  ] *  P.H_DEPTH_SIGMAS,
        devsim_oxide_cells = min (  64,   max(  4,  round( t [  't_ox' ]  /  rr )) ),
        models = P.FULL_MODELS,
    )
    arr= w.name
    v   =  idx ["work_function" ]
    y  = GM.gate_potential(idx['gate_voltage' ] ,
                     v)
    s=idx["drain_voltage"]

    GM._surface_is_live = False
    try :
        with  GM.quiet ()  :
            GM.build_mesh(w, arr, process  = t)


            GM.set_material_parameters(arr)
            GM.set_doping(w, arr, process =t)
            if driver  == 'devsim_expert' :
                GM.build_physics(arr, 0.0, s, P.FULL_MODELS, v)
                GM.ramp_to(arr, GM.GATE, y)
            else :
                GM.build_physics(arr, 0.0, 0.0, P.FULL_MODELS, v)
                if driver==  "devsim_stock" :


                    _stock_ramp(arr,GM.DRAIN,s,0.1)
                    _stock_ramp(arr, GM.GATE, y, 0.1)
                else:
                    vv =0.1 /  FINE
                    GM.ramp_to ( arr, GM.DRAIN, s , step  = vv, max_step =  vv)
                    GM.ramp_to(arr,GM.GATE,y,step =vv,max_step=vv)
            GM.settle(arr, balance_tol =  GM.BALANCE_TOL)
        k= [
            GM.terminal_current(arr,h)for h in(GM.DRAIN,GM.SOURCE,GM.BODY)
        ]
    finally:

        GM._surface_is_live  =  False
    return k[0], abs(sum(k)), max(abs(b)for b in k)

SOLVERS = {"pn_diode" :_diode, "mos_cap" : _mos_cap, "nmos"  :  _nmos}


def  _done (  path  : str ) ->  set[ tuple[str ,   str]  ] :

    if not os.path.exists(path):
        return set ( )

    with open(path, encoding = "utf-8") as t  :
        j   =   [m  for m in  t.read (  ).splitlines (  )  if  not  m.startswith( "#"  )]
    return{ ( y[ "case"  ],   y[ 'driver'  ] ) for y  in csv.DictReader (  j  ) }



def run(names:list[str] |None,path:str,resume: bool = False) ->str:
    print( "resume??", resume , len( names  or [  ] )  ) ; arr=[e for e in THREAD_VARIABLES if e in os.environ]
    if arr   :
        raise  SystemExit(f"unset {', '.join(arr)} first, robustness runs at each tool's default threading, see references/decisions.md 2026-09-26")
    y  =  [  c for  c  in read_cases ( )  if names  is None  or c[ "case"] in names  ]
    k = _done(path)  if resume else set();  h=  datetime.date.today().isoformat()
    kk  = [
        f"# written {h} by devsim_gen/scoreboard.py run",
        f"# devsim {devsim.__version__}, python {platform.python_version()}",
        f"# {platform.platform()}, {platform.processor()}, default BLAS threads",
        f"# fine step {1.0 / FINE:g} of the expert step",
        'case,driver,converged,value,imbalance,largest,seconds,message',
    ]


    with open(path,'a' if k else "w",encoding='utf-8',newline="\n") as a2:
        if  not k   :
            a2.write("\n".join(kk)+ "\n")

        for bb in y :
            for b in DRIVERS  :
                if(bb['case'], b) in k :
                    continue
                z =   time.perf_counter( )
                try :
                    xs, m, j  =SOLVERS[bb["device"]]  (bb, b)
                    v, x =True, ''
                except Exception as d  :
                    xs  = m = j=math.nan
                    v=False
                    x =  str(d).replace("\n", ' ').replace('"', "'") [: 200]


                finally :
                    _cleanup()
                i  =   time.perf_counter () -  z
                a2.write(
                    f"{bb['case']},{b},{v},{xs!r},{m!r},"
                    f'{j!r},{i:.3f},"{x}"\n'
                )
                a2.flush()
                print(f"{bb['case']} {b} {v} {xs:.4g} {i:.1f}s")

    return path
REFINEMENTS= (1.0,1.5,2.25)
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


def _silicon_nodes(device :str,region :str)-> int:
    return  len(  devsim.get_node_model_values ( device   =  device , region = region ,  name   =  "x"  ) )

def accuracy_point(name  :  str, r  :  float)-> tuple[int, float] :
    if name.startswith("diode") :
        b2=P.BY_NAME[name]
        item = f"{name}_acc"
        with GM.quiet() :
            GD.build_mesh(b2, item, refine  = r)
            GD.set_doping(b2,
                 item)
            GD.set_silicon_parameters(item)


            GD.build_physics(item)
            GD.ramp_to (  item ,   0.6 ,  0.0,   step  = 0.05)
        return _silicon_nodes(item,GD.REGION),GD.anode_current(item)
    if name.startswith('mos_cap')  :
        b2  =  {  foo.name  :   foo for  foo  in P.MOS_BENCHMARKS} [name ]

        item =  f"{name}_acc"
        w =[]
        with GM.quiet():
            GC.build_mesh(b2,item,refine=r)

            GC.set_material_parameters(item)
            GC.build_physics( b2,   item  )

            for d in(- CV_STEP , CV_STEP ) :
                devsim.set_parameter(
                    device=  item,
                    name  = f"{GC.GATE}_bias",
                    value =GC.gate_potential(b2, d),
                )
                devsim.solve(
                    type="dc",
                    absolute_error=1e-10,
                    relative_error= 1e-12,
                    maximum_iterations =100,
                )


                w.append(
                    devsim.get_contact_charge(
                        device= item, contact  = GC.GATE, equation = "PotentialEquation"
                    )
                )


        i =  (w[1]-w[0]) /(2.0 *  CV_STEP);return _silicon_nodes(item, GC.SILICON), i
    import dataclasses
    prev = P.MOSFET_BY_NAME[name].gate_voltages
    g=[k for k in prev if k <= 1.0 + 1e-9]  # golden grid, 1 V or wherever golden stops
    b2=dataclasses.replace(P.MOSFET_BY_NAME[name], gate_voltages=  tuple(g))
    y ,   a  =  GM.transfer_curve(  b2 , 0.05,  refine  =   r )
    return a,y[-1] ["drain"]


def  run_accuracy(path   :   str )   ->  str  :
    print( "working..." )

    ys= datetime.date.today().isoformat()

    res2  = [
        f"# written {ys} by devsim_gen/scoreboard.py accuracy" ,
        f"# devsim {devsim.__version__}, refinements {REFINEMENTS}",
        "benchmark,name,refine,nodes,value,seconds",
    ]
    with open (  path,  'w', encoding  =  'utf-8',   newline  =  "\n"  ) as tt  :
        tt.write("\n".join(res2)+"\n")

        for flag, v in ACCURACY :


            for u in REFINEMENTS  :

                cc = time.perf_counter()
                try  :
                    y,b =accuracy_point(v,u)

                finally :
                    _cleanup()

                a = time.perf_counter()- cc
                tt.write(f"{flag},{v},{u},{y},{b!r},{a:.3f}\n")

                tt.flush()
                print(f"{v} r={u} {y} nodes {b:.8g} {a:.1f}s")
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


def speed_sweep(name :  str)  -> int  :
    if name.startswith("diode"):
        with GM.quiet() :
            return len(GD.sweep(P.BY_NAME[name]))
    if  name.startswith("mos_cap"  )  :


        r  ={buf.name :buf for buf in P.MOS_BENCHMARKS}  [name]
        with GM.quiet() :
            return len(GC.sweep(r))

    v, num, _ = GM.sweep(P.MOSFET_BY_NAME[name])
    return len(v)+len(num)


def run_speed(path:  str) ->  str :
    b = [t for t in THREAD_VARIABLES if os.environ.get(t)!="1"]
    if b :
        raise SystemExit(f"set {', '.join(b)} to 1 first, see references/decisions.md 2026-09-26")
    rr =  datetime.date.today().isoformat()
    cnt= [
        f"# written {rr} by devsim_gen/scoreboard.py speed",
        f"# devsim {devsim.__version__}, python {platform.python_version()}",
        f"# {platform.platform()}, {platform.processor()}, one BLAS thread",
        'benchmark,name,run,points,seconds',
    ]
    with open(path,'w',encoding = "utf-8",newline= "\n")as tt:
        tt.write("\n".join(cnt)  +  "\n")
        for ret, it in SPEED :
            for k in range(1,SPEED_RUNS+1):
                w2=time.perf_counter()
                try :
                    y2 = speed_sweep(it)
                except Exception as k2 :
                    y2 = 0;  print(f"{it} run {k} failed: {k2}")
                finally :


                    _cleanup()
                ok =time.perf_counter()  - w2; tt.write(f"{ret},{it},{k},{y2},{ok:.3f}\n")
                tt.flush ( )
                print(f"{it} run {k}: {y2} points {ok:.1f}s")

    return path
def main() -> int :

    tmp2  =   argparse.ArgumentParser( description  = "The DEVSIM side of the Phase 8 scoreboard." )
    tmp2.add_argument ( "command" ,  choices  =  [  'api', "run",  "accuracy" , "speed"] )
    tmp2.add_argument('names', nargs=  "*", help  =  'run only these cases')
    tmp2.add_argument (  "--out",   default  =   None)
    tmp2.add_argument('--resume',action='store_true',help= "keep finished rows")
    xs   =  tmp2.parse_args ( )
    if xs.command =="api"  :
        print(write_api())
    elif xs.command ==  'accuracy' :
        print(run_accuracy(xs.out or os.path.join(OUT,'devsim_accuracy.csv')))

    elif xs.command=="speed" :
        print(run_speed(xs.out or os.path.join(OUT,'devsim_speed.csv')))
    else:

        v2  =  xs.out  or os.path.join( OUT, 'devsim_robustness.csv'  ); print(run(xs.names or None, v2, resume  =  xs.resume))
    return 0
if __name__ ==  '__main__'  :
    sys.exit (main(  )  )
