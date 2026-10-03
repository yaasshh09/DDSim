from __future__ import annotations
import argparse
import datetime
import os; import sys
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

    e =device
    j = benchmark.devsim_h_junction  / refine
    f = benchmark.devsim_h_bulk/refine


    create_1d_mesh(mesh=e)
    add_1d_mesh_line(mesh = e, pos = 0.0, ps = f, tag ='top')
    add_1d_mesh_line(mesh = e, pos = benchmark.junction, ps = j, ns = j, tag ="mid")
    add_1d_mesh_line( mesh =   e, pos =   benchmark.length,   ps  = f,  ns =  f,  tag  =  "bot"  )

    add_1d_contact(mesh=e,name =ANODE,tag= 'top',material="metal")
    add_1d_contact(mesh=e,name = CATHODE,tag ="bot",material='metal')
    add_1d_region(mesh=e,material ='Silicon',region=REGION,tag1="top",tag2='bot')
    finalize_mesh(mesh =e);create_device(mesh= e,device=device)


def set_silicon_parameters(device :str) -> None :

    cur :dict[str, float] = {'Permittivity'  : P.EPS_R_SI *P.EPS_0, 'ElectronCharge'  :P.Q, "n_i": P.N_I, "T" : P.T, "kT" : P.K_B *P.T, 'V_t' : P.V_T, 'mu_n'  :P.MU_N, "mu_p": P.MU_P, 'n1' : P.N_I, "p1"  : P.N_I,}
    for info, tt in cur.items() :
        set_parameter(device  =  device, region =REGION, name =info, value  = tt)

def set_doping(benchmark  :  P.DiodeBenchmark,
          device:str)  ->None:
    res2   =  (
        f"ifelse(x < {benchmark.junction:.16e}, "
        f"{-benchmark.Na:.16e}, {benchmark.Nd:.16e})"
    )
    node_model(device=device,region =REGION,name='NetDoping',equation=res2)


def set_lifetimes(device :str) -> None :
    for w,idx,u in(
        ('taun',P.TAU_N_MAX,P.TAU_N_MIN),
        ("taup",P.TAU_P_MAX,P.TAU_P_MIN),
    ) :
        k  =(
            f"{u:.16e} + ({idx:.16e} - {u:.16e}) / "
            f"(1 + (abs(NetDoping)/{P.N_REF_SRH:.16e})^{P.GAMMA_SRH:.16e})"
        )
        CreateNodeModel(device,REGION,w,k)
def build_physics(device :str) ->None :
    CreateSolution(device, REGION, 'Potential')
    CreateSiliconPotentialOnly(device,REGION)
    for u in(ANODE,
            CATHODE) :
        set_parameter(device=device,name =f"{u}_bias",value=0.0)
        CreateSiliconPotentialOnlyContact(device,REGION,u)

    solve(
        type  =   "dc",   absolute_error   = 1.0, relative_error  =   1e-12 ,  maximum_iterations  = 60
    )

    for y,r2 in(
        ('Electrons',"IntrinsicElectrons"),
        ('Holes',"IntrinsicHoles"),
    ):
        CreateSolution(device,REGION,y)
        set_node_values(device  =device, region =REGION, name =  y, init_from= r2)

    set_lifetimes( device )
    CreateSiliconDriftDiffusion(device, REGION, mu_n=  'mu_n', mu_p =  'mu_p')
    for u in(ANODE,
           CATHODE)  :
        CreateSiliconDriftDiffusionAtContact(device,REGION,u)
    solve(type="dc",absolute_error=1e10,relative_error = 1e-12,maximum_iterations=60)

def anode_current(device :str)->float :
    res= get_contact_current(device = device, contact =  ANODE, equation =ece_name)
    zz=get_contact_current(device =device,contact= ANODE,equation =hce_name)

    return res +zz




def cathode_current (  device   : str  ) ->   float   :
    zz =get_contact_current(device=device, contact =  CATHODE, equation  = ece_name)
    cnt= get_contact_current(device =device, contact = CATHODE, equation = hce_name)
    return zz+ cnt

def ramp_to(  device : str,  target  :  float,   present   :  float,  step  :  float ) ->   float   :
    if  abs(  target  - present  )  < 1e-15  :
        return present



    x2= 1.0 if target  > present else  -  1.0
    num  =   max( 1 ,   int( round( abs(target  -  present )   /   step  )) )
    for item  in  range (1 ,  num  +  1)  :
        d = present + x2 * step* item
        if item== num :
            d =  target
        set_parameter(device = device, name  = f"{ANODE}_bias", value =  d)


        solve(type= 'dc', absolute_error=1e10, relative_error=1e-12, maximum_iterations=60,)
    return target
def sweep(benchmark:  P.DiodeBenchmark, refine  : float  = 1.0) -> list[dict[str, Any]]  :


    d=f"{benchmark.name}_r{refine:g}".replace(".",
                  '_')
    build_mesh(  benchmark ,  d,   refine = refine)
    set_doping(benchmark, d)
    set_silicon_parameters(d)
    build_physics(d)


    j  =  sorted((  item for item  in  benchmark.voltages if item  < 0.0  ) ,  reverse  =  True  )
    v =  sorted(x for x in benchmark.voltages if x>=0.0)

    z2: dict[float, dict[str, Any]]= {}
    a   =  0.0

    for z in(j,v):

        a= ramp_to(d, 0.0, a, step =0.05)

        for b in z:
            a   = ramp_to(  d ,
                          b,
                       a,
                    step =  0.05)
            z2[b]  =  {
                'voltage' :b,
                "current"  : anode_current(d),
                "cathode": cathode_current(d),
            }
    delete_device(device = d)
    delete_mesh(  mesh  =  d )

    return[z2[vals] for vals in sorted(z2)]



def relative_difference(
    coarse: list[dict[str,Any]],fine:list[dict[str,Any]]
)->tuple[float,float]:
    num=0.0
    f = 0.0

    for j, y2 in zip(coarse, fine, strict  =True):
        if abs(j["current"])  <P.CURRENT_FLOOR and abs(y2["current"]) <P.CURRENT_FLOOR:
            continue
        a2  =  max( abs(j["current"  ]  ), abs (y2[ 'current'] )  )
        i=abs(j['current']-y2['current'])/a2
        if i > num  :
            num,  f  =  i,   j[  "voltage"]
    return num,f



def write_csv(
    benchmark :P.DiodeBenchmark,
    rows:list[dict[str,Any]],
    mesh_check:tuple[float,float]|None,
    path:str,
)->None:

    out   =  datetime.datetime.now( datetime.UTC ).strftime("%Y-%m-%d" )
    h: list[str] = [
        f"# device: {benchmark.name}",
        f"# benchmark: {benchmark.number} in docs/04-validation.md",
        '# generated by: tests/regression/devsim_gen/generate_diodes.py',
        f"# generator: devsim {devsim_version()} on python "
        f"{sys.version.split()[0]}",
        f"# generated on: {out}",
        f"# tolerance: {benchmark.tolerance}",
        f"# Na: {benchmark.Na:.6e}",
        f"# Nd: {benchmark.Nd:.6e}",
        f"# length: {benchmark.length:.6e}",
        f"# junction: {benchmark.junction:.6e}",
        f"# devsim h_junction: {benchmark.devsim_h_junction:.6e}",
        f"# devsim h_bulk: {benchmark.devsim_h_bulk:.6e}",
    ]
    if mesh_check is not None :


        item,k=mesh_check
        h.append(
            f"# mesh convergence: {item:.3e} worst relative change in current "
            f"when every spacing is halved, at {k:+g} V"
        )
    h.append( "# notes: "   + benchmark.notes)
    h.append('# models:')

    h.extend('#   '+b2 for b2 in P.MODEL_SUMMARY)
    h.append(
        "# columns: anode bias [V], anode current [A/cm^2], "
        "cathode current [A/cm^2]"
    )
    h.append("voltage,current,cathode_current")
    for  z  in rows  :
        h.append(
            f"{z['voltage']:.10g},{z['current']:.12e},{z['cathode']:.12e}"
        )
    with open(path, "w", encoding ="utf-8", newline =  "\n")  as obj  :
        obj.write("\n".join(h)+"\n")

def main()->int:
    z= argparse.ArgumentParser(description  =  "Generate the tier 4 golden diode curves with DEVSIM.")
    z.add_argument(
        "names",
        nargs =  "*",
        default=None,
        help=  "benchmark names to generate, default all",
    )
    z.add_argument('--out', default = os.path.join('data', "golden"), help=  "output directory for the CSV files",)

    z.add_argument(
        "--no-mesh-check" ,
        action  =   "store_true" ,
        help = "skip the halved mesh rerun, which roughly doubles the runtime" ,
    )
    bb=z.parse_args()

    y  =   P.BENCHMARKS
    if bb.names :
        y =   tuple(P.BY_NAME [flag]   for flag  in bb.names)
    os.makedirs(bb.out, exist_ok=True)
    for  rr  in  y  :
        print(f"[{rr.name}] solving on the reference mesh")
        hh=sweep(rr,refine= 1.0)

        rows :  tuple[float, float] |  None  = None
        if  not bb.no_mesh_check :
            print(f"[{rr.name}] solving again on a halved mesh")
            val2= sweep(rr,refine=2.0)

            rows = relative_difference(hh,val2)
            print(
                f"[{rr.name}] mesh convergence {rows[0]:.3e} "
                f"at {rows[1]:+g} V"
            )



        k= os.path.join(bb.out,f"{rr.name}.csv")
        write_csv (rr,  hh , rows, k )
        print(f"[{rr.name}] wrote {k} with {len(hh)} points")

    return 0

if __name__=="__main__" :

    raise  SystemExit( main(  ))
