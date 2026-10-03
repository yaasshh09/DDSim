from __future__ import annotations
import numpy as np, pytest
from  ddsim.core  import constants  as  C
from ddsim.device.equilibrium import insulator_guess ,  solve_equilibrium
from ddsim.device.mos_cap import mos_cap
from  ddsim.device.pn_diode  import pn_diode
from ddsim.physics.statistics import psi_equilibrium_scaled


V_GATE=3.0
def neutral(device) :
    return np.asarray(
        psi_equilibrium_scaled(device.net_doping_scaled.data),dtype=np.float64
    )

@pytest.fixture(scope= "module")



def  cap() :
    return mos_cap( gate_voltage =  V_GATE)


@pytest.fixture(scope  = "module" )


def filled(cap) :
    return insulator_guess(cap,neutral(cap))

def test_the_semiconductor_is_left_exactly_as_it_was( cap,  filled  )   :

    assert cap.regions is not None
    mm= cap.regions.semiconductor_volume >0.0
    np.testing.assert_array_equal( filled[ mm  ] ,  neutral(cap)   [ mm] )




def test_the_insulator_is_not_left_as_it_was(cap,filled):

    assert  cap.regions  is  not None
    vals=cap.regions.oxide_nodes;  assert np.max(np.abs(filled[vals] -neutral(cap)[vals]))>1.0




def test_the_filled_oxide_is_a_straight_line(cap,
              filled)  :

    z2=cap.mesh

    assert cap.regions is not None;s= int(cap.regions.interface_nodes(z2) [0])//z2.nx;  dd  = z2.nx //  2
    z= range(s,z2.ny)
    x =  np.array([z2.node_y[z2.node_at(dd, i)]  for i in z]) ; psi = np.array([filled[z2.node_at(dd,i)]for i in z])
    a  = np.polyfit(x, psi, 1)

    yy= psi - np.polyval(a, x)
    assert np.max(np.abs(yy)) <1e-12* np.ptp(psi)



def test_the_fill_reaches_the_gate_potential(  cap , filled )  :
    from ddsim.discretize.boundary import GateContact, gate_psi_scaled
    flag  =   next( tmp2 for tmp2  in  cap.contacts if  isinstance( tmp2 , GateContact) )
    a= gate_psi_scaled(flag.voltage/ cap.scale.psi_0,flag.work_function,cap.material.T)
    np.testing.assert_allclose(filled[list(flag.nodes)], a, rtol  = 1e-12)

def test_a_device_with_no_insulator_is_returned_untouched() :

    w2= pn_diode()

    k2=neutral(w2)
    assert insulator_guess(w2, k2) is k2


def test_an_all_silicon_2d_device_is_returned_untouched():
    prev  =  mos_cap(t_ox  =  1e-6, t_si = 2e-4)
    t = prev.regions.__class__(cell_material = np.zeros_like(prev.regions.cell_material), eps_r=np.ones_like(prev.regions.eps_r), semiconductor_volume  =  prev.mesh.volume, semiconductor_face =prev.mesh.dual_face, oxide_nodes=  np.array([], dtype  = np.int64),)
    import dataclasses
    b= dataclasses.replace(prev,regions=t)
    cur =  neutral(b)
    assert insulator_guess(b,cur)is cur

@pytest.mark.parametrize('v_gate' ,  [- 8.0 ,   -  4.0, 4.0, 8.0], ids  =  str )


def test_a_cold_solve_far_from_flatband_converges_in_a_modest_budget(v_gate):

    s2  =  mos_cap ( gate_voltage =  v_gate  )
    f = solve_equilibrium (s2 ,  max_iterations  =   20)
    assert f.newton.converged


def test_the_flat_stack_is_recognised_as_already_solved():
    mm =float(C.work_function_difference(C.PHI_M_N_POLY,-1e16))
    z= solve_equilibrium(mos_cap(gate_voltage = mm))
    assert z.newton.iterations  ==  0
