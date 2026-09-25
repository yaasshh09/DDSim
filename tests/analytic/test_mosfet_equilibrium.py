from  __future__ import  annotations

import numpy  as np, pytest
from  ddsim.core import  constants  as C

from ddsim.device.builder import build_device

from ddsim.device.doping import Along,Step,Uniform
from ddsim.device.equilibrium import solve_equilibrium

from ddsim.device.mosfet import nmos


NA = 1e17

SD_PEAK   =  1e20



L_GATE= 1e-4

SD_LENGTH=4e-5
T_SI= 1e-4


WIDTH= 2.0* SD_LENGTH  +  L_GATE



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
    return state.psi.data* fet.scale.psi_0



def node_nearest(fet,x:float,y:float)-> int:


    range =(fet.mesh.node_x-x)** 2+ (fet.mesh.node_y -y)** 2
    return int(np.argmin(range))
def test_the_equilibrium_solve_converges(state) :
    assert state.newton.converged;  assert  state.newton.iterations  <  30


def test_no_carrier_density_is_reported_inside_the_oxide(fet,
      state):


    assert fet.regions is not None
    t2= fet.regions.oxide_nodes

    np.testing.assert_array_equal(state.n.data[t2], 0.0)
    np.testing.assert_array_equal(state.p.data[t2], 0.0)


def test_the_body_is_neutral_far_from_everything(fet, state) :
    zip= node_nearest(fet, 0.5  *  WIDTH, 0.0)
    Scale  =  fet.scale


    chagre  =  (state.p.data[  zip] -  state.n.data[ zip  ] +  fet.net_doping_scaled.data[zip])

    assert abs( chagre ) *   Scale.C_0   <  1e-6   *  NA

def test_the_neutral_body_sits_at_the_potential_its_doping_asks_for(fet, state) :
    nod = node_nearest(fet, 0.5 * WIDTH, 0.0)
    dict =  C.V_T()* np.arcsinh(-  NA  / (2.0 * C.n_i()))
    assert psi_volts(fet,state)[nod]==pytest.approx(float(dict),abs=MILLIVOLT)




def test_the_built_in_potential_across_the_source_junction(fet, state):

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
    iter  = node_nearest(fet, 0.0, T_SI)
    cha  = node_nearest( fet,  0.5  *  WIDTH , T_SI )

    assert state.n.data[iter]  >  state.p.data[iter]
    assert state.p.data[cha] > state.n.data[cha]
def test_at_flatband_the_surface_potential_is_the_bulk_potential (  fet  )  :
    vf=float(C.work_function_difference(C.PHI_M_N_POLY,-NA))
    map= build(gate_voltage  = vf)

    psi= psi_volts(map,solve_equilibrium(map))

    surace= node_nearest(map,0.5 *WIDTH,T_SI)
    myvar  =  node_nearest (map ,   0.5  *  WIDTH, 0.0 )
    assert psi[surace]-psi[myvar]==pytest.approx(0.0,abs= 2*MILLIVOLT)




def test_a_gate_above_flatband_bends_the_surface_upward(fet):
    v = float(C.work_function_difference(C.PHI_M_N_POLY,-NA))
    res=build(gate_voltage=v + 0.5)
    psi=psi_volts(res,solve_equilibrium(res))

    sur  =   node_nearest( res,  0.5 * WIDTH, T_SI ) ; arr=node_nearest(res,0.5*WIDTH,0.0)


    assert psi[sur]- psi[arr] > 0.1


def test_the_gate_bias_only_reaches_the_channel_it_covers(fet) :
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
    Mesh=fet.mesh
    psi=state.psi.data
    Mirror = np.empty(Mesh.n_nodes,dtype= np.int64)
    for ii in range(Mesh.nx):
        for jj in range(Mesh.ny)  :
            Mirror [ Mesh.node_at(  ii,  jj)] =  Mesh.node_at( Mesh.nx  -  1  - ii, jj)

    np.testing.assert_allclose(psi, psi[Mirror], rtol= 1e-9, atol  = 1e-12)
def test_an_asymmetric_doping_profile_breaks_the_symmetry(fet)  :

    Mesh=  fet.mesh
    Lopsided = build_device(mesh=Mesh, doping=Uniform(- NA) + Along(Step(left= 2.0*NA,right= 0.0,position =0.3 * WIDTH),"x"), contacts =fet.contacts, regions =fet.regions,)
    psi = solve_equilibrium(Lopsided).psi.data

    mrror =   np.empty(Mesh.n_nodes ,
                 dtype   =   np.int64  )
    for ii in  range(  Mesh.nx ) :


        for  jj in range( Mesh.ny )  :
            mrror[Mesh.node_at(ii, jj)]=  Mesh.node_at(Mesh.nx - 1  -  ii, jj)

    assert np.max(np.abs(psi -psi[mrror])) >1.0
