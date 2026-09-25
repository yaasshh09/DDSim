from __future__ import annotations
import  math

from scipy.special import erfcinv as _erfcinv
from ddsim.core import constants as C;from ddsim.device.builder import Device,Material,build_device
from ddsim.device.doping import Along,Erfc,Gaussian,Mirrored,Uniform

from ddsim.device.regions import stacked_regions
from ddsim.discretize.boundary import GateContact, OhmicPlate
from ddsim.mesh.mesh1d import(
    Mesh1D,
    graded_mesh_1d,
    stacked_mesh_1d,
    uniform_mesh_1d,
)

from ddsim.mesh.mesh2d import tensor_mesh_2d

SOURCE =  'source'


DRAIN= "drain"
GATE =  "gate"
BODY =  'body'

def _junction_mesh(
    length  : float, n_nodes : int, refine_at :float, h_min  : float
) ->Mesh1D :
    if h_min  * (n_nodes- 1)>= length  :
        return uniform_mesh_1d (  length = length,   n_nodes   =  n_nodes)
    return graded_mesh_1d (
        length =   length,   n_nodes  =  n_nodes,   refine_at =  refine_at, h_min =   h_min
    )




def  implant_lengths(x_j   :  float,   lateral_diffusion :   float,   sd_peak   :  float,   Na :   float)  -> tuple[ float,   float  ]  :
    sima = x_j   /   math.sqrt ( 2.0  *   math.log( sd_peak / Na )  )
    edg=lateral_diffusion /float(_erfcinv(2.0*Na/sd_peak))
    return sima, edg


def nmos(L_gate : float = 1e-4, sd_length : float=4e-5, contact_length : float = 2e-5, substrate_doping:float = - 1e17, sd_peak : float = 1e20, x_j :  float  = 1.5e-5, lateral_diffusion :float  = 1e-5, t_ox  :float = 2e-6, t_si : float = 1e-4, n_contact : int=6, n_sd: int = 12, n_channel  :  int  = 16, n_silicon: int =  101, n_oxide : int  =33, h_min_x :  float = 2e-7, h_min_y:float =6.25e-9, gate_voltage  :  float=0.0, drain_voltage : float =0.0, source_voltage: float  = 0.0, body_voltage: float  = 0.0, work_function: float= C.PHI_M_N_POLY, material  : Material| None  = None, degenerate :bool  = True,)  ->Device :

    for str,Value in(
        ("L_gate",L_gate),
        ("sd_length",sd_length),
        ('contact_length',contact_length),
        ('x_j',x_j),
        ('lateral_diffusion',lateral_diffusion),
        ('t_ox',t_ox),
        ("t_si",t_si),
    ):
        if Value<= 0.0 :
            raise  ValueError(  f"{str} must be positive, got {Value}" )
    if substrate_doping  >=  0.0 :
        raise ValueError(
            f"nmos builds an n-channel device, so the body must be p-type and "
            f"substrate_doping negative, got {substrate_doping:g}. The sign "
            "convention is net doping, Nd - Na, everywhere in this codebase."
        )


    Na = - substrate_doping
    if sd_peak<=Na:
        raise ValueError(
            f"sd_peak must exceed the substrate doping, got sd_peak="
            f"{sd_peak:g} against {Na:g} cm^-3. A source no heavier than the "
            "body it sits in makes no junction, and there would be no depth "
            "at which to put one."
        )
    if x_j   >=  t_si :
        raise ValueError(
            f"x_j must be less than t_si, got x_j={x_j:g} cm in {t_si:g} cm "
            'of silicon. The source would reach the body contact.'
        )

    if 2.0 *lateral_diffusion>= L_gate :
        raise ValueError(
            f"no channel is left: the source and drain each reach "
            f"{lateral_diffusion:g} cm under a gate {L_gate:g} cm long, so "
            "the two junctions meet or cross. That geometry is a short "
            'circuit, and a solver would happily return one instead of an '
            "error. Shrink lateral_diffusion or lengthen the gate."
        )



    if contact_length >= sd_length:


        raise  ValueError(
            f"contact_length must be less than sd_length, got "
            f"{contact_length:g} against {sd_length:g} cm. A contact reaching "
            'the mask edge pins the junction itself, and the built in '
            "potential stops being something the solve works out."
        )
    if n_contact<2 or n_sd <2 or n_channel < 2 or n_silicon<2 or n_oxide<2:

        raise ValueError(
            'every mesh segment needs at least 2 nodes, got n_contact='
            f"{n_contact}, n_sd={n_sd}, n_channel={n_channel}, n_silicon="
            f"{n_silicon}, n_oxide={n_oxide}. A segment with one node has no "
            'extent to carry a field across.'
        )

    wid=2.0 * sd_length+L_gate

    list  =  sd_length  -  contact_length
    hlf_gate = 0.5 * L_gate

    xa  =  stacked_mesh_1d(uniform_mesh_1d(length  = contact_length, n_nodes  = n_contact), _junction_mesh(list, n_sd, refine_at =  list, h_min=h_min_x), _junction_mesh(hlf_gate, n_channel, refine_at= 0.0, h_min =  h_min_x), _junction_mesh(hlf_gate, n_channel, refine_at = hlf_gate, h_min = h_min_x), _junction_mesh(list, n_sd, refine_at = 0.0, h_min= h_min_x), uniform_mesh_1d(length = contact_length, n_nodes = n_contact),)

    yaxis =stacked_mesh_1d(
        graded_mesh_1d(
            length=t_si,n_nodes=n_silicon,refine_at =t_si,h_min=h_min_y
        ),
        uniform_mesh_1d(length =t_ox,n_nodes=n_oxide),
    )

    mes =  tensor_mesh_2d( xa, yaxis)
    Regions = stacked_regions( mes , interface_y =  t_si)
    iContactEnd  =  n_contact - 1


    i_gate_strat= n_contact+n_sd -2


    q  =  i_gate_strat  +  2 * (  n_channel  -  1  )
    idrainstart = q+n_sd-1

    max= n_silicon -1
    Top = mes.ny -  1
    arr =(OhmicPlate(name =SOURCE, nodes=tuple(mes.node_at(i,max) for i in range(iContactEnd +1)), voltage=source_voltage,), OhmicPlate(name =DRAIN, nodes=tuple(mes.node_at(i,max)for i in range(idrainstart,mes.nx)), voltage=drain_voltage,), GateContact(name=GATE, nodes=tuple(mes.node_at(i,Top)for i in range(i_gate_strat,q +1)), voltage=gate_voltage, work_function =work_function,), OhmicPlate(name=BODY, nodes=tuple(mes.node_at(i,0)for i in range(mes.nx)), voltage= body_voltage,),)

    Sigma,edg= implant_lengths(x_j,lateral_diffusion,sd_peak,Na)
    sou =(Along(Erfc(peak =0.5, position=sd_length, length =  edg), "x") * Along(Gaussian(peak = 1.0, centre  = t_si, sigma  =Sigma), "y") *  sd_peak)
    dooping   =  Uniform(  substrate_doping  ) +  sou +  Mirrored(  sou,  0.5  *  wid )
    return build_device(mesh =mes, doping=dooping, contacts=arr, material=material, regions= Regions, degenerate=degenerate,)
