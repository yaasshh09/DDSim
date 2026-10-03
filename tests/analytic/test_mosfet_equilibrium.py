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


    y2 =(fet.mesh.node_x-x)** 2+ (fet.mesh.node_y -y)** 2
    return int(np.argmin(y2))
def test_the_equilibrium_solve_converges(state) :
    assert state.newton.converged;  assert  state.newton.iterations  <  30


def test_no_carrier_density_is_reported_inside_the_oxide(fet,
      state):


    assert fet.regions is not None
    zz= fet.regions.oxide_nodes

    np.testing.assert_array_equal(state.n.data[zz], 0.0)
    np.testing.assert_array_equal(state.p.data[zz], 0.0)


def test_the_body_is_neutral_far_from_everything(fet, state) :
    thing= node_nearest(fet, 0.5  *  WIDTH, 0.0)
    f  =  fet.scale


    d  =  (state.p.data[  thing] -  state.n.data[ thing  ] +  fet.net_doping_scaled.data[thing])

    assert abs( d ) *   f.C_0   <  1e-6   *  NA

def test_the_neutral_body_sits_at_the_potential_its_doping_asks_for(fet, state) :
    a = node_nearest(fet, 0.5 * WIDTH, 0.0)
    k2 =  C.V_T()* np.arcsinh(-  NA  / (2.0 * C.n_i()))
    assert psi_volts(fet,state)[a]==pytest.approx(float(k2),abs=MILLIVOLT)




def test_the_built_in_potential_across_the_source_junction(fet, state):

    psi =  psi_volts(fet, state);  tmp2= node_nearest(fet,0.0,T_SI);  j  = node_nearest(fet, 0.5*WIDTH, 0.0)

    r  = fet.degeneracy

    assert r is not None,"this MOSFET is supposed to be degenerate"

    k=C.V_T() * float(
        r.equilibrium_psi(SD_PEAK/C.n_i())
        -r.equilibrium_psi(- NA/ C.n_i())
    )


    assert psi[tmp2] -psi[j] == pytest.approx(k, abs =  5* MILLIVOLT)

    m2 =C.V_T() *  float(np.arcsinh(SD_PEAK / (2.0 *C.n_i())) + np.arcsinh(NA/ (2.0* C.n_i())))
    assert(k-  m2)  /  MILLIVOLT==  pytest.approx(30.5, rel = 1e-2)


def test_the_source_is_n_type_and_the_channel_is_p_type(fet, state):
    z  = node_nearest(fet, 0.0, T_SI)
    b  = node_nearest( fet,  0.5  *  WIDTH , T_SI )

    assert state.n.data[z]  >  state.p.data[z]
    assert state.p.data[b] > state.n.data[b]
def test_at_flatband_the_surface_potential_is_the_bulk_potential (  fet  )  :
    b=float(C.work_function_difference(C.PHI_M_N_POLY,-NA))
    e= build(gate_voltage  = b)

    psi= psi_volts(e,solve_equilibrium(e))

    t= node_nearest(e,0.5 *WIDTH,T_SI)
    tt  =  node_nearest (e ,   0.5  *  WIDTH, 0.0 )
    assert psi[t]-psi[tt]==pytest.approx(0.0,abs= 2*MILLIVOLT)




def test_a_gate_above_flatband_bends_the_surface_upward(fet):
    k = float(C.work_function_difference(C.PHI_M_N_POLY,-NA))
    val2=build(gate_voltage=k + 0.5)
    psi=psi_volts(val2,solve_equilibrium(val2))

    a  =   node_nearest( val2,  0.5 * WIDTH, T_SI ) ; m=node_nearest(val2,0.5*WIDTH,0.0)


    assert psi[a]- psi[m] > 0.1


def test_the_gate_bias_only_reaches_the_channel_it_covers(fet) :
    s  =  float(C.work_function_difference(C.PHI_M_N_POLY, -NA))
    ret=build(gate_voltage= s)


    r=build(gate_voltage=s+ 1.0)

    y =  psi_volts(  ret,   solve_equilibrium(  ret ) )
    g=psi_volts(r,solve_equilibrium(r))
    v  = node_nearest(r, 0.0, T_SI)
    m2 =  node_nearest ( r , 0.5  * WIDTH ,  T_SI)


    assert abs(g[v] -y[v])< MILLIVOLT
    assert g[m2]- y[m2]> 0.1


def test_the_solution_is_symmetric_about_the_centre(fet,state) :
    xs=fet.mesh
    psi=state.psi.data
    y = np.empty(xs.n_nodes,dtype= np.int64)
    for r2 in range(xs.nx):
        for bar in range(xs.ny)  :
            y [ xs.node_at(  r2,  bar)] =  xs.node_at( xs.nx  -  1  - r2, bar)

    np.testing.assert_allclose(psi, psi[y], rtol= 1e-9, atol  = 1e-12)
def test_an_asymmetric_doping_profile_breaks_the_symmetry(fet)  :

    y=  fet.mesh
    b = build_device(mesh=y, doping=Uniform(- NA) + Along(Step(left= 2.0*NA,right= 0.0,position =0.3 * WIDTH),"x"), contacts =fet.contacts, regions =fet.regions,)
    psi = solve_equilibrium(b).psi.data

    h =   np.empty(y.n_nodes ,
                 dtype   =   np.int64  )
    for z in  range(  y.nx ) :


        for  ok in range( y.ny )  :
            h[y.node_at(z, ok)]=  y.node_at(y.nx - 1  -  z, ok)

    assert np.max(np.abs(psi -psi[h])) >1.0
