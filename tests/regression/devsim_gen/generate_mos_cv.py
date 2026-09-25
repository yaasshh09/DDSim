"""Generate the tier 4 MOS capacitor golden data with devsim.

Run under the devsim interpreter, not the project one:

    .venv-devsim/Scripts/python.exe -m tests.regression.devsim_gen.generate_mos_cv

It writes one CSV per benchmark into `data/golden/`, each carrying a header
that records the devsim version, the device, the model choices and the mesh
convergence check, so a golden file can be read years later without guessing
how it was made. See the README in this directory.

Why devsim solves this in one dimension and ddsim in two
--------------------------------------------------------
A MOS capacitor is a one dimensional problem, and devsim is the reference
rather than the thing under test. Solving it in 1D here makes the reference as
simple as it can be, and it makes the comparison stronger rather than weaker:
ddsim's answer comes off a structured 2D mesh with a different grading and a
different assembly, so the two agree because the physics agrees rather than
because the meshes match.

What is compared, and why the charge rather than the capacitance
----------------------------------------------------------------
The gate charge is what each code actually computes. Both get it the same way,
as the flux of D over the contact's own cell, which is the discrete Gauss law
there rather than a second calculation of the same thing.

The capacitance is then a derivative, and the two codes take it differently:
ddsim differentiates its solved system exactly, devsim has no such path here.
Storing the charge and applying one central difference to both keeps the
comparison about the physics instead of about the differentiation, and the
exact derivative is checked against ddsim's own central difference in
tests/analytic/test_mos_cv.py where that is the question being asked.

The potential reference is the same in both codes
-------------------------------------------------
devsim's `IntrinsicElectrons = n_i*exp(Potential/V_t)` puts its zero at the
intrinsic level, which is where ddsim's psi has always been. That is what lets
the gate bias be handed over as `V_gate + (PHI_M_MIDGAP - Phi_M)` with no
further translation, and it is worth stating because a constant offset between
the two references would shift the whole C-V curve while leaving its shape
perfect.
"""
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
    """The devsim version string, for the golden file header."""
    import devsim

    return str(getattr(devsim, '__version__', "unknown"))

@contextlib.contextmanager




def  quiet (  )  :
    """Swallow devsim's per iteration convergence report.

    It writes to stdout, and the generator's stdout is a progress log a person
    reads. The errors still surface: a solve that fails raises.
    """
    with contextlib.redirect_stdout(io.StringIO())  :
        yield

def build_mesh(
    benchmark : P.MosBenchmark,device: str,refine: float=1.0
)-> None:
    """Create and finalise the stack for one benchmark.

    Args:
        benchmark: the device definition.
        device: devsim device name.
        refine: divide every spacing by this. 1 is the golden mesh.

    Silicon below, oxide above, the interface on the node line they share. The
    silicon is graded to the surface, where the inversion layer sits; the oxide
    is uniform, because with no charge in it its potential is a straight line
    and uniform cells resolve a straight line exactly.

    devsim grades between consecutive mesh lines, and each line names its
    spacing separately for each direction: `ps` walking in +x, `ns` walking in
    -x. The interface line is the only one with material on both sides, so it
    is the only one where the two differ and the only one where getting them
    the wrong way round is silent. It buys a mesh that looks refined, refines
    the wrong region, and converges to the right answer slowly.
    """
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
    """Push every ddsim constant into devsim, overriding its own defaults.

    devsim's own `simple_physics` carries eps_r(Si) = 11.1, q = 1.6e-19 and
    eps_0 = 8.85e-14, all of which are the right numbers rounded. Left alone
    they would put a percent into the comparison before any physics happened.
    """
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
    """Poisson with Boltzmann carriers in the silicon, Laplace in the oxide."""
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
    '''The potential to pin the gate at [V], measured from the intrinsic level.

    docs/01-physics.md writes the gate condition as psi_gate = V_gate - Phi_MS,
    which needs the doping under the gate. Since psi is measured from the
    intrinsic level in both codes, and the work function of intrinsic silicon
    is exactly chi + Eg/2, the same statement is

        psi_gate = V_gate + (chi + Eg/2 - Phi_M)

    and the doping has cancelled. ddsim uses the second form for the same
    reason: a contact has no business reading the substrate under it.
    '''
    return v_gate + (P.PHI_M_MIDGAP - benchmark.work_function)



def sweep(benchmark:P.MosBenchmark, refine  : float=  1.0) ->  list[dict[str, Any]] :
    """Solve one benchmark at every requested gate bias and return the curve.

    Walked in ascending order and continued from the point before it, since the
    stack at one bias is a good guess for the next. The sweep starts at the
    most negative bias, which is deep accumulation and the easiest end.
    """
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
    '''Worst relative charge difference between two meshes, and where [1, V].

    Skips the points where the charge passes through zero near flatband, since
    a relative comparison between two numbers that are both nearly nothing says
    nothing about the mesh. The floor is a thousandth of the largest charge on
    the curve, which is four decades below anything the comparison cares about.
    '''
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
    """Write one golden curve, header and all."""
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
    """Generate every benchmark named on the command line, or all of them."""
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
