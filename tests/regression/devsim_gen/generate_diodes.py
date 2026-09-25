"""Generate the tier 4 golden diode curves with DEVSIM.

This is the only file in the repository that imports devsim, and it does not
run under the project interpreter. See README.md in this directory for the
environment. Run it from the repository root:

    .venv-devsim/Scripts/python.exe tests/regression/devsim_gen/generate_diodes.py

It writes one CSV per benchmark into `data/golden/`, each carrying a header
that records the devsim version, the model choices and a mesh refinement self
check, so a curve can be read years later without having to guess how it was
made.

The physics is built from devsim's own `simple_physics` helpers, with every
parameter overridden to the ddsim value. The helpers ship with eps_r = 11.1,
q = 1.6e-19 and mu_n = 400, none of which are what ddsim uses, so the override
is the whole point rather than a detail. docs/04-validation.md: match the
models before comparing numbers.
"""


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

"""The single silicon region. Every device here is one material."""
ANODE ="anode"
'''p side contact, the one that gets swept.'''
CATHODE = "cathode"
'''n side contact, held at zero.'''

def devsim_version()->str :
    """The installed devsim version [1], for the golden file header."""
    try:
        from  importlib.metadata import version
        return version('devsim')


    except Exception:
        return 'unknown'



def build_mesh(benchmark   : P.DiodeBenchmark,   device  :  str,  refine   :  float  =   1.0)  ->  None  :
    """Create and finalise the 1D mesh for one benchmark.

    Args:
        benchmark: the device definition.
        device: devsim device name.
        refine: divide every mesh spacing by this. 2.0 halves the spacing
            everywhere, which is how the mesh convergence self check is run.

    The junction sits on a mesh line, so the abrupt doping step lands on a node
    rather than inside a cell in either code. docs/04-validation.md warns that
    putting it inside a cell moves the metallurgical junction by half a cell.
    """

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
    """Push every ddsim constant into devsim, overriding its own defaults.

    Nothing here is left at a devsim default. A parameter that is not written
    down in `parameters.py` is a parameter the two codes are free to disagree
    about.
    """

    s2 :dict[str, float] = {'Permittivity'  : P.EPS_R_SI *P.EPS_0, 'ElectronCharge'  :P.Q, "n_i": P.N_I, "T" : P.T, "kT" : P.K_B *P.T, 'V_t' : P.V_T, 'mu_n'  :P.MU_N, "mu_p": P.MU_P, 'n1' : P.N_I, "p1"  : P.N_I,}
    for r2, t2 in s2.items() :
        set_parameter(device  =  device, region =REGION, name =r2, value  = t2)

def set_doping(benchmark  :  P.DiodeBenchmark,
          device:str)  ->None:
    """The abrupt junction, right continuous at the junction like ddsim's Step."""
    tmp   =  (
        f"ifelse(x < {benchmark.junction:.16e}, "
        f"{-benchmark.Na:.16e}, {benchmark.Nd:.16e})"
    )
    node_model(device=device,region =REGION,name='NetDoping',equation=tmp)


def set_lifetimes(device :str) -> None :
    """Scharfetter doping dependent lifetimes as node models, not parameters.

    devsim's SRH expression names `taun` and `taup`. Creating node models under
    those names shadows the scalar parameters its helpers would otherwise set,
    which is how the doping dependence gets in without touching the helper. The
    lifetimes do not depend on the carrier densities, so the SRH derivatives
    devsim differentiates symbolically are unaffected.

    ddsim feeds the Scharfetter relation abs(net doping) rather than Na + Nd,
    because only the net is available on a step profile. Same choice here, for
    the same reason, so the two agree node for node.
    """
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
    """Equilibrium solve, then the full drift diffusion system."""
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
    """Terminal current into the anode [A/cm^2].

    devsim reports the electron and hole contact currents separately and their
    sum already carries ddsim's convention: positive means conventional current
    flowing from the contact into the device, so a forward biased diode is
    positive at the anode and the two terminals sum to zero. That was measured
    against ddsim rather than assumed, on the 1e16 symmetric diode: both codes
    give about +5.2e2 A/cm^2 at 0.7 V and about -8.3e-9 A/cm^2 at -1 V.

    A 1D devsim device has unit cross section, so the number is already a
    density and needs no area division.
    """
    Electrons= get_contact_current(device = device, contact =  ANODE, equation =ece_name)
    Holes=get_contact_current(device =device,contact= ANODE,equation =hce_name)

    return Electrons +Holes




def cathode_current (  device   : str  ) ->   float   :
    """Terminal current into the cathode [A/cm^2]. Sums to zero with the anode."""
    format =get_contact_current(device=device, contact =  CATHODE, equation  = ece_name)
    dict= get_contact_current(device =device, contact = CATHODE, equation = hce_name)
    return format+ dict

def ramp_to(  device : str,  target  :  float,   present   :  float,  step  :  float ) ->   float   :
    """Walk the anode bias from present to target [V], solving at each step.

    Returns the bias actually reached, which is the target unless a solve
    raised. Continuation exists because a diode solved cold at 0.7 V does not
    converge, in devsim any more than in ddsim.
    """
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


    '''Solve one benchmark at every requested bias and return the curve.

    The voltage list is walked outward from zero in each direction, negatives
    descending and then positives ascending, each leg continued from the
    equilibrium solution rather than from the far end of the other leg.
    '''

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
    """Worst relative current difference between two meshes, and where [1, V].

    Points under P.CURRENT_FLOOR are skipped, since a relative comparison
    between two roundoff residues says nothing about the mesh.

    Refining does not always improve this number and is not expected to. In
    reverse bias the terminal current is a cancellation between drift and
    diffusion terms that scale as 1/h, so halving the spacing doubles the
    quantity being cancelled and costs about a factor of two in the last digits
    that survive. That is a property of Scharfetter-Gummel at low current, not
    of the mesh being too coarse, and it is why the golden mesh is the coarse
    one rather than the finest that would still run.
    """
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

    """Write one golden curve, header and all."""

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
    '''Generate every benchmark named on the command line, or all of them.'''
    paser= argparse.ArgumentParser(description  =  __doc__)
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
