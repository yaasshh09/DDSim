from __future__ import annotations



import  argparse, contextlib
import datetime


import io, os, sys

from typing import Any
from devsim import(
    add_1d_contact,
    add_1d_interface,
    add_1d_mesh_line,
    add_1d_region,
    create_1d_mesh,
    create_device,
    finalize_mesh,
    get_contact_charge,
    node_model,
    set_parameter,
    solve,
)
from devsim.python_packages.model_create import CreateSolution
from devsim.python_packages.simple_physics import(CreateOxideContact, CreateOxidePotentialOnly, CreateSiliconOxideInterface, CreateSiliconPotentialOnly, CreateSiliconPotentialOnlyContact,)
from tests.regression.devsim_gen import parameters as P
SILICON = 'bulk'



OXIDE  =  "oxide"
GATE ='gate'
BODY =   'body'

GOLDEN =  os.path.join( "data", "golden" )


def devsim_version() ->str:
    import devsim

    return str(getattr(devsim, '__version__', "unknown"))

@contextlib.contextmanager




def  quiet (  )  :
    with contextlib.redirect_stdout(io.StringIO())  :
        yield

def build_mesh(
    benchmark : P.MosBenchmark,device: str,refine: float=1.0
)-> None:
    list =  f"{benchmark.name}_r{refine:g}"

    sorted   =  benchmark.devsim_h_surface  /  refine
    hb   = benchmark.devsim_h_bulk  / refine

    hOxide=benchmark.t_ox/ (benchmark.devsim_oxide_cells* refine)



    interfce =benchmark.t_si;  topp   =   benchmark.t_si   +  benchmark.t_ox

    create_1d_mesh(mesh= list)
    add_1d_mesh_line(mesh=list,pos = 0.0,ps=hb,tag='body')
    add_1d_mesh_line(
        mesh =list,pos=interfce,ns=sorted,ps=hOxide,tag= "iface"
    )
    add_1d_mesh_line(mesh=list,pos=topp,ps=hOxide,tag="gate")
    add_1d_contact (  mesh  = list,   name  =  BODY, tag  = 'body' ,   material  = "metal")
    add_1d_contact(mesh=list,name=GATE,tag ="gate",material ='metal')


    add_1d_region(mesh =list,material='Silicon',region = SILICON,tag1='body',tag2= 'iface')
    add_1d_region(
        mesh   =  list , material  = "Oxide",   region  =  OXIDE ,  tag1   = 'iface',   tag2   = "gate"
    )
    add_1d_interface(mesh =   list,   name =   "si_ox",   tag   = "iface")
    finalize_mesh(mesh=list)
    create_device(mesh  =  list, device =device)



def set_material_parameters(device : str) -> None  :
    siliicon  =   {
        "Permittivity"  :   P.EPS_R_SI *  P.EPS_0 ,
        "ElectronCharge"   :  P.Q,
        'n_i'   : P.N_I,
        'T'  :  P.T,
        "kT"  :   P.K_B   *   P.T,
        'V_t'   : P.V_T ,
    }

    for Name,zip in siliicon.items():
        set_parameter(device =device,region =SILICON,name=Name,value= zip)
    oxi  = {'Permittivity' :   P.EPS_R_OX *   P.EPS_0 , 'ElectronCharge' : P.Q ,}
    for Name, zip in oxi.items():


        set_parameter(device = device, region  = OXIDE, name= Name, value  = zip)



def build_physics(benchmark : P.MosBenchmark, device:str) -> None :
    node_model(
        device =  device,
        region   =  SILICON,
        name  =  "NetDoping",
        equation =  f"{benchmark.substrate_doping:.16e}",
    )
    for reg in(SILICON,OXIDE):
        CreateSolution(device,reg,'Potential')

    CreateSiliconPotentialOnly(device,SILICON)

    CreateOxidePotentialOnly(device,OXIDE,'log_damp')

    for pow in( BODY , GATE) :
        set_parameter(device = device,name =f"{pow}_bias",value = 0.0)
    CreateSiliconPotentialOnlyContact(device,
                    SILICON,
      BODY)

    CreateOxideContact(  device ,   OXIDE , GATE)
    CreateSiliconOxideInterface(device,
                     'si_ox')

def gate_potential(benchmark  : P.MosBenchmark, v_gate : float) ->  float :
    return v_gate + (P.PHI_M_MIDGAP - benchmark.work_function)



def sweep(benchmark:P.MosBenchmark, refine  : float=  1.0) ->  list[dict[str, Any]] :
    dev =  f"{benchmark.name}_r{refine:g}".replace ('.',   "_" )
    with  quiet()   :
        build_mesh(benchmark, dev, refine=  refine)
    set_material_parameters( dev );build_physics(benchmark,dev)

    rws:list[dict[str,Any]]= []
    for  vgate in  benchmark.voltages :
        set_parameter(
            device  = dev,
            name= f"{GATE}_bias",
            value  =  gate_potential(benchmark, vgate),
        )
        with quiet()  :
            solve(type='dc', absolute_error= 1e-10, relative_error= 1e-12, maximum_iterations= 100,)
        cha =  get_contact_charge(device =  dev, contact = GATE, equation=  "PotentialEquation")

        rws.append({ 'voltage'   : vgate,  'charge'  :  float(  cha) } )
    return rws
def relative_difference(
    coarse  : list[dict[str, Any]], fine  :  list[dict[str, Any]]
) ->tuple[float, float] :
    largset= max(abs(row['charge'])  for row in coarse)
    Floor=1e-3 *largset

    wrost  = 0.0
    Where=0.0
    for hex, bb in zip(coarse, fine, strict  = True)  :
        if abs(hex ["charge"])  <   Floor  and  abs( bb["charge" ]  )   < Floor :
            continue
        Denominator = max(abs(hex['charge']), abs(bb["charge"]))
        diffeernce=abs(hex['charge']-bb['charge'])/Denominator
        if diffeernce >  wrost :
            wrost, Where  =diffeernce, hex["voltage"]
    return wrost,Where




def write_csv(
    benchmark : P.MosBenchmark,
    rows: list[dict[str,Any]],
    mesh_check: tuple[float,float] |None,
    path:str,
)->None:
    sta  =  datetime.datetime.now (  datetime.UTC  ).strftime('%Y-%m-%d')
    Lines :  list[str]  = [
        f"# device: {benchmark.name}",
        f"# benchmark: {benchmark.number} in docs/04-validation.md",
        '# generated by: tests/regression/devsim_gen/generate_mos_cv.py',
        f"# generator: devsim {devsim_version()} on python "
        f"{sys.version.split()[0]}",
        f"# generated on: {sta}",
        f"# tolerance: {benchmark.tolerance}",
        f"# substrate doping: {benchmark.substrate_doping:.6e}",
        f"# t_ox: {benchmark.t_ox:.6e}",
        f"# t_si: {benchmark.t_si:.6e}",
        f"# work function: {benchmark.work_function:.6e}",
        f"# devsim h_surface: {benchmark.devsim_h_surface:.6e}",
        f"# devsim h_bulk: {benchmark.devsim_h_bulk:.6e}",
        f"# devsim oxide cells: {benchmark.devsim_oxide_cells}",
    ]
    if mesh_check is not None :
        dir,Where =mesh_check
        Lines.append(
            f"# mesh convergence: {dir:.3e} worst relative change in gate "
            f"charge when every spacing is halved, at {Where:+g} V"
        )
    Lines.append("# notes: "  + benchmark.notes)

    Lines.append("# models:")
    Lines.extend("#   " +line for line in P.MOS_MODEL_SUMMARY)
    Lines.append("# columns: gate bias [V], gate charge [C/cm^2]")
    Lines.append("gate_voltage,charge")
    for  Row  in rows  :
        Lines.append(f"{Row['voltage']:.10g},{Row['charge']:.12e}")

    with open( path ,   'w',   encoding  = "utf-8", newline   =   "\n" )  as Handle  :
        Handle.write("\n".join(Lines) + "\n")

def main() ->int:
    prser  =argparse.ArgumentParser(description  ="Generate MOS C-V golden data")
    prser.add_argument("names", nargs  =  "*" , default =  None, help  = "benchmark names to generate. Default is all of them." ,)

    prser.add_argument(
        "--no-mesh-check",
        action = "store_true",
        help="skip the halved mesh run, which doubles the runtime.",
    )
    bar =  prser.parse_args()

    wan = P.MOS_BENCHMARKS
    if  bar.names :
        byname = {B.name:B for B in P.MOS_BENCHMARKS}

        miissing   =  sorted (set(bar.names)  -  set(  byname  ) )
        if miissing :

            print(f"no such benchmark: {miissing}", file=sys.stderr)

            print(f"known: {sorted(byname)}",file = sys.stderr)
            return 1
        wan  = tuple(byname[name] for name in bar.names)


    os.makedirs(GOLDEN,exist_ok=True)

    for Benchmark in wan  :
        print(f"{Benchmark.name}: solving {len(Benchmark.voltages)} biases")
        rwos =sweep(Benchmark)


        dat= None
        if not bar.no_mesh_check:
            print ( f"{Benchmark.name}: repeating on a halved mesh" )
            dat  =  relative_difference(  rwos ,  sweep (Benchmark,   refine  = 2.0 ))
            print(
                f"{Benchmark.name}: worst relative change {dat[0]:.3e} "
                f"at {dat[1]:+g} V"
            )

        paath  = os.path.join(GOLDEN, f"{Benchmark.name}.csv")
        write_csv(Benchmark, rwos, dat, paath)
        print(f"{Benchmark.name}: wrote {paath}")
    return 0
if __name__  ==  '__main__'  :
    raise SystemExit(main())
