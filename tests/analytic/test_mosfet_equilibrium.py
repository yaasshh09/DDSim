"""The MOSFET at equilibrium, before any current flows.

Every terminal at zero, or at flatband, and nothing but Poisson. That is a
deliberately small ambition for a device this complicated, and it is the right
first one: equilibrium is the only state of a MOSFET whose answer is known in
closed form, so it is the only place the geometry can be checked against
something other than a previous run of the same code.

Three things are being asked here, in order of how much they would cost to get
wrong.

The built in potential across the source junction. It is set by the two doping
levels and nothing else, and it is the potential the channel has to be lifted
over before a current flows. If it is wrong then every threshold voltage this
project reports is wrong by the same amount, and nothing in a converged
solution says so.

Flatband under the gate. At V_G equal to Phi_MS the bands are flat right up to
the surface, so the surface potential equals the neutral body potential. That
one number needs the gate work function, the body contact potential and the
intrinsic reference to agree exactly, and those are three pieces of code that
never otherwise meet. This is the same check the MOS capacitor gates on, asked
again on a geometry where the gate no longer spans the whole device.

Symmetry. With the source and drain at the same bias the solution has to be
symmetric about the centre of the device, node for node. Nothing in the
assembly knows the device is symmetric, so this is a real check on the 2D
Jacobian, on the region map, and on the mesh: if a horizontal face carried the
wrong area or the wrong permittivity, the two halves would disagree.
"""

from  __future__ import  annotations

import numpy  as np, pytest
from  ddsim.core import  constants  as C

from ddsim.device.builder import build_device

from ddsim.device.doping import Along,Step,Uniform
from ddsim.device.equilibrium import solve_equilibrium

from ddsim.device.mosfet import nmos


NA = 1e17

"""Substrate acceptor concentration [cm^-3]."""
SD_PEAK   =  1e20



"""Source and drain surface concentration [cm^-3]."""


L_GATE= 1e-4

"""Gate length [cm], 1 um."""

SD_LENGTH=4e-5
"""Source and drain length per side [cm]."""
T_SI= 1e-4


"""Silicon thickness [cm]."""
WIDTH= 2.0* SD_LENGTH  +  L_GATE
"""Total device length [cm]."""



MILLIVOLT  = 1e-3

def build(** overrides):
    return  nmos(L_gate  =  L_GATE, sd_length   =  SD_LENGTH, substrate_doping  =-  NA , sd_peak  =  SD_PEAK , t_si = T_SI, **  overrides,)


@pytest.fixture(scope='module')

def fet():

    return  build()



@pytest.fixture(scope  ="module")



def state(fet):


    return solve_equilibrium(fet)


def  psi_volts(  fet ,  state )   :
    """The potential in volts, off the scaled solution."""
    return state.psi.data* fet.scale.psi_0



def node_nearest(fet,x:float,y:float)-> int:


    """The node closest to a position, for reading a solution at a place."""
    range =(fet.mesh.node_x-x)** 2+ (fet.mesh.node_y -y)** 2
    return int(np.argmin(range))
def test_the_equilibrium_solve_converges(state) :
    """Four terminals, two materials and four orders of magnitude of doping.
    Nothing in this project has asked Newton for all of that at once before.
    """
    assert state.newton.converged;  assert  state.newton.iterations  <  30


def test_no_carrier_density_is_reported_inside_the_oxide(fet,
      state):


    """An insulator holds none. The zero charge volume makes the equations
    agree, and this is the array everything downstream reads."""
    assert fet.regions is not None
    t2= fet.regions.oxide_nodes

    np.testing.assert_array_equal(state.n.data[t2], 0.0)
    np.testing.assert_array_equal(state.p.data[t2], 0.0)


def test_the_body_is_neutral_far_from_everything(fet, state) :
    """Deep under the channel, where nothing has depleted anything. Space
    charge there is p - n + N, and it has to vanish."""
    zip= node_nearest(fet, 0.5  *  WIDTH, 0.0)
    Scale  =  fet.scale


    chagre  =  (state.p.data[  zip] -  state.n.data[ zip  ] +  fet.net_doping_scaled.data[zip])

    assert abs( chagre ) *   Scale.C_0   <  1e-6   *  NA

def test_the_neutral_body_sits_at_the_potential_its_doping_asks_for(fet, state) :
    """psi is measured from the intrinsic level, so neutral p-type silicon
    sits at V_T asinh(N / 2 n_i) and nowhere else."""
    nod = node_nearest(fet, 0.5 * WIDTH, 0.0)
    dict =  C.V_T()* np.arcsinh(-  NA  / (2.0 * C.n_i()))
    assert psi_volts(fet,state)[nod]==pytest.approx(float(dict),abs=MILLIVOLT)




def test_the_built_in_potential_across_the_source_junction(fet, state):
    """The number every threshold voltage in this project rests on.

    Neutral n+ source to neutral p body, both read where they are neutral, and
    the answer is fixed by the two doping levels alone. Under Boltzmann that
    is

        V_bi = V_T [ asinh(Nd / 2 n_i) + asinh(Na / 2 n_i) ]

    Not the textbook V_T ln(Na Nd / n_i^2), for the reason
    docs/05-pitfalls.md gives: the log form is the same number wherever it is
    valid and returns nonsense at low doping, and the solver uses asinh.

    The source is at 1e20, where n/Nc is 3.5 and Boltzmann is not on, so this
    MOSFET is solved with Fermi-Dirac and the closed form is the degenerate
    neutrality solution instead. It reduces to the asinh form exactly at low
    doping, so this is the same statement rather than a second one, and it is
    read off the device's own statistics rather than typed in. Row 119 of
    docs/07-decisions.md predicted the 30.5 mV this moved by, before there was
    anything to apply it to.
    """

    psi =  psi_volts(fet, state);  Source= node_nearest(fet,0.0,T_SI);  bod  = node_nearest(fet, 0.5*WIDTH, 0.0)

    myvar  = fet.degeneracy

    assert myvar is not None,"this MOSFET is supposed to be degenerate"

    v=C.V_T() * float(
        myvar.equilibrium_psi(SD_PEAK/C.n_i())
        -myvar.equilibrium_psi(- NA/ C.n_i())
    )


    assert psi[Source] -psi[bod] == pytest.approx(v, abs =  5* MILLIVOLT)

    dir =C.V_T() *  float(np.arcsinh(SD_PEAK / (2.0 *C.n_i())) + np.arcsinh(NA/ (2.0* C.n_i())))
    assert(v-  dir)  /  MILLIVOLT==  pytest.approx(30.5, rel = 1e-2)


def test_the_source_is_n_type_and_the_channel_is_p_type(fet, state):
    """The junction is where the majority carrier changes, and a solve that
    lost the implant would still converge on a uniform p-type bar."""
    iter  = node_nearest(fet, 0.0, T_SI)
    cha  = node_nearest( fet,  0.5  *  WIDTH , T_SI )

    assert state.n.data[iter]  >  state.p.data[iter]
    assert state.p.data[cha] > state.n.data[cha]
def test_at_flatband_the_surface_potential_is_the_bulk_potential (  fet  )  :
    """The sharpest single check available on a MOS gate.

    At V_G = Phi_MS there is no field anywhere in the semiconductor, so the
    surface under the middle of the channel sits at exactly the potential the
    neutral body does. Getting the sign of Phi_MS wrong moves this by nearly
    two volts, and getting the intrinsic reference wrong moves it by the
    Fermi potential, and neither shows up anywhere else.
    """
    vf=float(C.work_function_difference(C.PHI_M_N_POLY,-NA))
    map= build(gate_voltage  = vf)

    psi= psi_volts(map,solve_equilibrium(map))

    surace= node_nearest(map,0.5 *WIDTH,T_SI)
    myvar  =  node_nearest (map ,   0.5  *  WIDTH, 0.0 )
    assert psi[surace]-psi[myvar]==pytest.approx(0.0,abs= 2*MILLIVOLT)




def test_a_gate_above_flatband_bends_the_surface_upward(fet):
    """And the direction is the whole of an n-channel device. A positive gate
    pushes the surface potential up, which is what eventually inverts it."""
    v = float(C.work_function_difference(C.PHI_M_N_POLY,-NA))
    res=build(gate_voltage=v + 0.5)
    psi=psi_volts(res,solve_equilibrium(res))

    sur  =   node_nearest( res,  0.5 * WIDTH, T_SI ) ; arr=node_nearest(res,0.5*WIDTH,0.0)


    assert psi[sur]- psi[arr] > 0.1


def test_the_gate_bias_only_reaches_the_channel_it_covers(fet) :
    """The gate spans the channel alone, so the oxide over the source has no
    electrode on it and the silicon under the source does not know the gate
    moved. If the gate were pinning the whole top row, it would."""
    vFb  =  float(C.work_function_difference(C.PHI_M_N_POLY, -NA))
    af=build(gate_voltage= vFb)


    d2=build(gate_voltage=vFb+ 1.0)

    hmm =  psi_volts(  af,   solve_equilibrium(  af ) )
    lif=psi_volts(d2,solve_equilibrium(d2))
    UnderSource  = node_nearest(d2, 0.0, T_SI)
    type =  node_nearest ( d2 , 0.5  * WIDTH ,  T_SI)


    assert abs(lif[UnderSource] -hmm[UnderSource])< MILLIVOLT
    assert lif[type]- hmm[type]> 0.1


def test_the_solution_is_symmetric_about_the_centre(fet,state) :
    """Source and drain at the same bias, so the two halves have to agree
    node for node. Nothing in the assembly knows the device is symmetric.
    """
    Mesh=fet.mesh
    psi=state.psi.data
    Mirror = np.empty(Mesh.n_nodes,dtype= np.int64)
    for ii in range(Mesh.nx):
        for jj in range(Mesh.ny)  :
            Mirror [ Mesh.node_at(  ii,  jj)] =  Mesh.node_at( Mesh.nx  -  1  - ii, jj)

    np.testing.assert_allclose(psi, psi[Mirror], rtol= 1e-9, atol  = 1e-12)
def test_an_asymmetric_doping_profile_breaks_the_symmetry(fet)  :
    """The check above is only worth having if it can fail.

    Broken with the doping rather than with a drain bias, because a drain bias
    is not an equilibrium problem. Source and drain are both n-type contacts,
    so two different biases on them means two different electron quasi-Fermi
    levels, and a single frozen level cannot express that. That device is the
    next stage, not this one.
    """

    Mesh=  fet.mesh
    Lopsided = build_device(mesh=Mesh, doping=Uniform(- NA) + Along(Step(left= 2.0*NA,right= 0.0,position =0.3 * WIDTH),"x"), contacts =fet.contacts, regions =fet.regions,)
    psi = solve_equilibrium(Lopsided).psi.data

    mrror =   np.empty(Mesh.n_nodes ,
                 dtype   =   np.int64  )
    for ii in  range(  Mesh.nx ) :


        for  jj in range( Mesh.ny )  :
            mrror[Mesh.node_at(ii, jj)]=  Mesh.node_at(Mesh.nx - 1  -  ii, jj)

    assert np.max(np.abs(psi -psi[mrror])) >1.0
