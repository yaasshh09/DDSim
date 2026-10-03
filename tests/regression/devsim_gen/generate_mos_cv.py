from __future__ import annotations
import argparse, contextlib
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
    mm =  f"{benchmark.name}_r{refine:g}"

    vv   =  benchmark.devsim_h_surface  /  refine
    s   = benchmark.devsim_h_bulk  / refine

    r2=benchmark.t_ox/ (benchmark.devsim_oxide_cells* refine)



    t =benchmark.t_si;  j   =   benchmark.t_si   +  benchmark.t_ox

    create_1d_mesh(mesh= mm)
    add_1d_mesh_line(mesh=mm,pos = 0.0,ps=s,tag='body')
    add_1d_mesh_line(
        mesh =mm,pos=t,ns=vv,ps=r2,tag= "iface"
    )
    add_1d_mesh_line(mesh=mm,pos=j,ps=r2,tag="gate")
    add_1d_contact (  mesh  = mm,   name  =  BODY, tag  = 'body' ,   material  = "metal")
    add_1d_contact(mesh=mm,name=GATE,tag ="gate",material ='metal')


    add_1d_region(mesh =mm,material='Silicon',region = SILICON,tag1='body',tag2= 'iface')
    add_1d_region(
        mesh   =  mm , material  = "Oxide",   region  =  OXIDE ,  tag1   = 'iface',   tag2   = "gate"
    )
    add_1d_interface(mesh =   mm,   name =   "si_ox",   tag   = "iface")
    finalize_mesh(mesh=mm)
    create_device(mesh  =  mm, device =device)



def set_material_parameters(device : str) -> None  :
    b  =   {
        "Permittivity"  :   P.EPS_R_SI *  P.EPS_0 ,
        "ElectronCharge"   :  P.Q,
        'n_i'   : P.N_I,
        'T'  :  P.T,
        "kT"  :   P.K_B   *   P.T,
        'V_t'   : P.V_T ,
    }

    for d,vals in b.items():
        set_parameter(device =device,region =SILICON,name=d,value= vals)
    y  = {'Permittivity' :   P.EPS_R_OX *   P.EPS_0 , 'ElectronCharge' : P.Q ,}
    for d, vals in y.items():


        set_parameter(device = device, region  = OXIDE, name= d, value  = vals)



def build_physics(benchmark : P.MosBenchmark, device:str) -> None :
    node_model(
        device =  device,
        region   =  SILICON,
        name  =  "NetDoping",
        equation =  f"{benchmark.substrate_doping:.16e}",
    )
    for i in(SILICON,OXIDE):
        CreateSolution(device,i,'Potential')

    CreateSiliconPotentialOnly(device,SILICON)

    CreateOxidePotentialOnly(device,OXIDE,'log_damp')

    for val in( BODY , GATE) :
        set_parameter(device = device,name =f"{val}_bias",value = 0.0)
    CreateSiliconPotentialOnlyContact(device,
                    SILICON,
      BODY)

    CreateOxideContact(  device ,   OXIDE , GATE)
    CreateSiliconOxideInterface(device,
                     'si_ox')

def gate_potential(benchmark  : P.MosBenchmark, v_gate : float) ->  float :
    return v_gate + (P.PHI_M_MIDGAP - benchmark.work_function)



def sweep(benchmark:P.MosBenchmark, refine  : float=  1.0) ->  list[dict[str, Any]] :
    yy =  f"{benchmark.name}_r{refine:g}".replace ('.',   "_" )
    with  quiet()   :
        build_mesh(benchmark, yy, refine=  refine)
    set_material_parameters( yy );build_physics(benchmark,yy)

    x:list[dict[str,Any]]= []
    for  v in  benchmark.voltages :
        set_parameter(
            device  = yy,
            name= f"{GATE}_bias",
            value  =  gate_potential(benchmark, v),
        )
        with quiet()  :
            solve(type='dc', absolute_error= 1e-10, relative_error= 1e-12, maximum_iterations= 100,)
        b =  get_contact_charge(device =  yy, contact = GATE, equation=  "PotentialEquation")

        x.append({ 'voltage'   : v,  'charge'  :  float(  b) } )
    return x
def relative_difference(
    coarse  : list[dict[str, Any]], fine  :  list[dict[str, Any]]
) ->tuple[float, float] :
    s2= max(abs(jj['charge'])  for jj in coarse)
    d=1e-3 *s2

    yy  = 0.0
    z=0.0
    for ok, c in zip(coarse, fine, strict  = True)  :
        if abs(ok ["charge"])  <   d  and  abs( c["charge" ]  )   < d :
            continue
        res = max(abs(ok['charge']), abs(c["charge"]))
        m=abs(ok['charge']-c['charge'])/res
        if m >  yy :
            yy, z  =m, ok["voltage"]
    return yy,z




def write_csv(
    benchmark : P.MosBenchmark,
    rows: list[dict[str,Any]],
    mesh_check: tuple[float,float] |None,
    path:str,
)->None:
    r  =  datetime.datetime.now (  datetime.UTC  ).strftime('%Y-%m-%d')
    u :  list[str]  = [
        f"# device: {benchmark.name}",
        f"# benchmark: {benchmark.number} in docs/04-validation.md",
        '# generated by: tests/regression/devsim_gen/generate_mos_cv.py',
        f"# generator: devsim {devsim_version()} on python "
        f"{sys.version.split()[0]}",
        f"# generated on: {r}",
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
        hh,nxt =mesh_check
        u.append(
            f"# mesh convergence: {hh:.3e} worst relative change in gate "
            f"charge when every spacing is halved, at {nxt:+g} V"
        )
    u.append("# notes: "  + benchmark.notes)

    u.append("# models:")
    u.extend("#   " +foo for foo in P.MOS_MODEL_SUMMARY)
    u.append("# columns: gate bias [V], gate charge [C/cm^2]")
    u.append("gate_voltage,charge")
    for  c2  in rows  :
        u.append(f"{c2['voltage']:.10g},{c2['charge']:.12e}")

    with open( path ,   'w',   encoding  = "utf-8", newline   =   "\n" )  as tmp  :
        tmp.write("\n".join(u) + "\n")

def main() ->int:
    d  =argparse.ArgumentParser(description  ="Generate MOS C-V golden data")
    d.add_argument("names", nargs  =  "*" , default =  None, help  = "benchmark names to generate. Default is all of them." ,)

    d.add_argument(
        "--no-mesh-check",
        action = "store_true",
        help="skip the halved mesh run, which doubles the runtime.",
    )
    tmp =  d.parse_args()

    w = P.MOS_BENCHMARKS
    if  tmp.names :
        z2 = {j.name:j for j in P.MOS_BENCHMARKS}

        b   =  sorted (set(tmp.names)  -  set(  z2  ) )
        if b :

            print(f"no such benchmark: {b}", file=sys.stderr)

            print(f"known: {sorted(z2)}",file = sys.stderr)
            return 1
        w  = tuple(z2[v] for v in tmp.names)


    os.makedirs(GOLDEN,exist_ok=True)

    for mm in w  :
        print(f"{mm.name}: solving {len(mm.voltages)} biases")
        r =sweep(mm)


        c= None
        if not tmp.no_mesh_check:
            print ( f"{mm.name}: repeating on a halved mesh" )
            c  =  relative_difference(  r ,  sweep (mm,   refine  = 2.0 ))
            print(
                f"{mm.name}: worst relative change {c[0]:.3e} "
                f"at {c[1]:+g} V"
            )

        k  = os.path.join(GOLDEN, f"{mm.name}.csv")
        write_csv(mm, r, c, k)
        print(f"{mm.name}: wrote {k}")
    return 0
if __name__  ==  '__main__'  :
    raise SystemExit(main())
