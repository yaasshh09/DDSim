from __future__ import annotations


import argparse
import datetime

import os;  import sys

from typing import Any

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))



import parameters as P
from devsim import(
    add_1d_contact,
    add_1d_mesh_line,
    add_1d_region,
    create_1d_mesh,
    create_device,
    delete_device,
    delete_mesh,
    finalize_mesh,
    get_contact_current,
    node_model,
    set_node_values,
    set_parameter,
    solve,
)
from devsim.python_packages.model_create import(
    CreateNodeModel,
    CreateSolution,
)

from devsim.python_packages.simple_physics import(CreateSiliconDriftDiffusion, CreateSiliconDriftDiffusionAtContact, CreateSiliconPotentialOnly, CreateSiliconPotentialOnlyContact, ece_name, hce_name,)
REGION  =  "bulk"

ANODE ="anode"
CATHODE = "cathode"

def devsim_version()->str :
    try:
        from  importlib.metadata import version
        return version('devsim')


    except Exception:
        return 'unknown'



def build_mesh(benchmark   : P.DiodeBenchmark,   device  :  str,  refine   :  float  =   1.0)  ->  None  :

    Mesh =device
    hj = benchmark.devsim_h_junction  / refine
    hb = benchmark.devsim_h_bulk/refine


    create_1d_mesh(mesh=Mesh)
    add_1d_mesh_line(mesh = Mesh, pos = 0.0, ps = hb, tag ='top')
    add_1d_mesh_line(mesh = Mesh, pos = benchmark.junction, ps = hj, ns = hj, tag ="mid")
    add_1d_mesh_line( mesh =   Mesh, pos =   benchmark.length,   ps  = hb,  ns =  hb,  tag  =  "bot"  )

    add_1d_contact(mesh=Mesh,name =ANODE,tag= 'top',material="metal")
    add_1d_contact(mesh=Mesh,name = CATHODE,tag ="bot",material='metal')
    add_1d_region(mesh=Mesh,material ='Silicon',region=REGION,tag1="top",tag2='bot')
    finalize_mesh(mesh =Mesh);create_device(mesh= Mesh,device=device)


def set_silicon_parameters(device :str) -> None :

    s2 :dict[str, float] = {'Permittivity'  : P.EPS_R_SI *P.EPS_0, 'ElectronCharge'  :P.Q, "n_i": P.N_I, "T" : P.T, "kT" : P.K_B *P.T, 'V_t' : P.V_T, 'mu_n'  :P.MU_N, "mu_p": P.MU_P, 'n1' : P.N_I, "p1"  : P.N_I,}
    for r2, t2 in s2.items() :
        set_parameter(device  =  device, region =REGION, name =r2, value  = t2)

def set_doping(benchmark  :  P.DiodeBenchmark,
          device:str)  ->None:
    tmp   =  (
        f"ifelse(x < {benchmark.junction:.16e}, "
        f"{-benchmark.Na:.16e}, {benchmark.Nd:.16e})"
    )
    node_model(device=device,region =REGION,name='NetDoping',equation=tmp)


def set_lifetimes(device :str) -> None :
    for Name,tauMax,tauu_min in(
        ('taun',P.TAU_N_MAX,P.TAU_N_MIN),
        ("taup",P.TAU_P_MAX,P.TAU_P_MIN),
    ) :
        equ  =(
            f"{tauu_min:.16e} + ({tauMax:.16e} - {tauu_min:.16e}) / "
            f"(1 + (abs(NetDoping)/{P.N_REF_SRH:.16e})^{P.GAMMA_SRH:.16e})"
        )
        CreateNodeModel(device,REGION,Name,equ)
def build_physics(device :str) ->None :
    CreateSolution(device, REGION, 'Potential')
    CreateSiliconPotentialOnly(device,REGION)
    for conntact in(ANODE,
            CATHODE) :
        set_parameter(device=device,name =f"{conntact}_bias",value=0.0)
        CreateSiliconPotentialOnlyContact(device,REGION,conntact)

    solve(
        type  =   "dc",   absolute_error   = 1.0, relative_error  =   1e-12 ,  maximum_iterations  = 60
    )

    for buf,sou in(
        ('Electrons',"IntrinsicElectrons"),
        ('Holes',"IntrinsicHoles"),
    ):
        CreateSolution(device,REGION,buf)
        set_node_values(device  =device, region =REGION, name =  buf, init_from= sou)

    set_lifetimes( device )
    CreateSiliconDriftDiffusion(device, REGION, mu_n=  'mu_n', mu_p =  'mu_p')
    for conntact in(ANODE,
           CATHODE)  :
        CreateSiliconDriftDiffusionAtContact(device,REGION,conntact)
    solve(type="dc",absolute_error=1e10,relative_error = 1e-12,maximum_iterations=60)

def anode_current(device :str)->float :
    Electrons= get_contact_current(device = device, contact =  ANODE, equation =ece_name)
    Holes=get_contact_current(device =device,contact= ANODE,equation =hce_name)

    return Electrons +Holes




def cathode_current (  device   : str  ) ->   float   :
    format =get_contact_current(device=device, contact =  CATHODE, equation  = ece_name)
    dict= get_contact_current(device =device, contact = CATHODE, equation = hce_name)
    return format+ dict

def ramp_to(  device : str,  target  :  float,   present   :  float,  step  :  float ) ->   float   :
    if  abs(  target  - present  )  < 1e-15  :
        return present



    len= 1.0 if target  > present else  -  1.0
    set  =   max( 1 ,   int( round( abs(target  -  present )   /   step  )) )
    for ind  in  range (1 ,  set  +  1)  :
        filter = present + len * step* ind
        if ind== set :
            filter =  target
        set_parameter(device = device, name  = f"{ANODE}_bias", value =  filter)


        solve(type= 'dc', absolute_error=1e10, relative_error=1e-12, maximum_iterations=60,)
    return target
def sweep(benchmark:  P.DiodeBenchmark, refine  : float  = 1.0) -> list[dict[str, Any]]  :


    all=f"{benchmark.name}_r{refine:g}".replace(".",
                  '_')
    build_mesh(  benchmark ,  all,   refine = refine)
    set_doping(benchmark, all)
    set_silicon_parameters(all)
    build_physics(all)


    Negatives  =  sorted((  V for V  in  benchmark.voltages if V  < 0.0  ) ,  reverse  =  True  )
    pos =  sorted(V for V in benchmark.voltages if V>=0.0)

    bar: dict[float, dict[str, Any]]= {}
    Present   =  0.0

    for Leg in(Negatives,pos):

        Present= ramp_to(all, 0.0, Present, step =0.05)

        for tar in Leg:
            Present   = ramp_to(  all ,
                          tar,
                       Present,
                    step =  0.05)
            bar[tar]  =  {
                'voltage' :tar,
                "current"  : anode_current(all),
                "cathode": cathode_current(all),
            }
    delete_device(device = all)
    delete_mesh(  mesh  =  all )

    return[bar[V] for V in sorted(bar)]



def relative_difference(
    coarse: list[dict[str,Any]],fine:list[dict[str,Any]]
)->tuple[float,float]:
    thing=0.0
    whe = 0.0

    for A, B in zip(coarse, fine, strict  =True):
        if abs(A["current"])  <P.CURRENT_FLOOR and abs(B["current"]) <P.CURRENT_FLOOR:
            continue
        denominaotr  =  max( abs(A["current"  ]  ), abs (B[ 'current'] )  )
        res=abs(A['current']-B['current'])/denominaotr
        if res > thing  :
            thing,  whe  =  res,   A[  "voltage"]
    return thing,whe



def write_csv(
    benchmark :P.DiodeBenchmark,
    rows:list[dict[str,Any]],
    mesh_check:tuple[float,float]|None,
    path:str,
)->None:

    Stamp   =  datetime.datetime.now( datetime.UTC ).strftime("%Y-%m-%d" )
    temp2: list[str] = [
        f"# device: {benchmark.name}",
        f"# benchmark: {benchmark.number} in docs/04-validation.md",
        '# generated by: tests/regression/devsim_gen/generate_diodes.py',
        f"# generator: devsim {devsim_version()} on python "
        f"{sys.version.split()[0]}",
        f"# generated on: {Stamp}",
        f"# tolerance: {benchmark.tolerance}",
        f"# Na: {benchmark.Na:.6e}",
        f"# Nd: {benchmark.Nd:.6e}",
        f"# length: {benchmark.length:.6e}",
        f"# junction: {benchmark.junction:.6e}",
        f"# devsim h_junction: {benchmark.devsim_h_junction:.6e}",
        f"# devsim h_bulk: {benchmark.devsim_h_bulk:.6e}",
    ]
    if mesh_check is not None :


        Worst,Where=mesh_check
        temp2.append(
            f"# mesh convergence: {Worst:.3e} worst relative change in current "
            f"when every spacing is halved, at {Where:+g} V"
        )
    temp2.append( "# notes: "   + benchmark.notes)
    temp2.append('# models:')

    temp2.extend('#   '+line for line in P.MODEL_SUMMARY)
    temp2.append(
        "# columns: anode bias [V], anode current [A/cm^2], "
        "cathode current [A/cm^2]"
    )
    temp2.append("voltage,current,cathode_current")
    for  sorted  in rows  :
        temp2.append(
            f"{sorted['voltage']:.10g},{sorted['current']:.12e},{sorted['cathode']:.12e}"
        )
    with open(path, "w", encoding ="utf-8", newline =  "\n")  as Handle  :
        Handle.write("\n".join(temp2)+"\n")

def main()->int:
    paser= argparse.ArgumentParser(description  =  "Generate the tier 4 golden diode curves with DEVSIM.")
    paser.add_argument(
        "names",
        nargs =  "*",
        default=None,
        help=  "benchmark names to generate, default all",
    )
    paser.add_argument('--out', default = os.path.join('data', "golden"), help=  "output directory for the CSV files",)

    paser.add_argument(
        "--no-mesh-check" ,
        action  =   "store_true" ,
        help = "skip the halved mesh rerun, which roughly doubles the runtime" ,
    )
    Args=paser.parse_args()

    choesn  =   P.BENCHMARKS
    if Args.names :
        choesn =   tuple(P.BY_NAME [name]   for name  in Args.names)
    os.makedirs(Args.out, exist_ok=True)
    for  hash  in  choesn  :
        print(f"[{hash.name}] solving on the reference mesh")
        row=sweep(hash,refine= 1.0)

        stuff2 :  tuple[float, float] |  None  = None
        if  not Args.no_mesh_check :
            print(f"[{hash.name}] solving again on a halved mesh")
            dict= sweep(hash,refine=2.0)

            stuff2 = relative_difference(row,dict)
            print(
                f"[{hash.name}] mesh convergence {stuff2[0]:.3e} "
                f"at {stuff2[1]:+g} V"
            )



        ptah= os.path.join(Args.out,f"{hash.name}.csv")
        write_csv (hash,  row , stuff2, ptah )
        print(f"[{hash.name}] wrote {ptah} with {len(row)} points")

    return 0

if __name__=="__main__" :

    raise  SystemExit( main(  ))
