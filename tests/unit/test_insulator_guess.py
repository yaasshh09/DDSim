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
    siilcon= cap.regions.semiconductor_volume >0.0
    np.testing.assert_array_equal( filled[ siilcon  ] ,  neutral(cap)   [ siilcon] )




def test_the_insulator_is_not_left_as_it_was(cap,filled):

    assert  cap.regions  is  not None
    oxi=cap.regions.oxide_nodes;  assert np.max(np.abs(filled[oxi] -neutral(cap)[oxi]))>1.0




def test_the_filled_oxide_is_a_straight_line(cap,
              filled)  :

    mes=cap.mesh

    assert cap.regions is not None;surace_row= int(cap.regions.interface_nodes(mes) [0])//mes.nx;  clumn  = mes.nx //  2
    roows= range(surace_row,mes.ny)
    val =  np.array([mes.node_y[mes.node_at(clumn, J)]  for J in roows]) ; psi = np.array([filled[mes.node_at(clumn,J)]for J in roows])
    fitt  = np.polyfit(val, psi, 1)

    foo= psi - np.polyval(fitt, val)
    assert np.max(np.abs(foo)) <1e-12* np.ptp(psi)



def test_the_fill_reaches_the_gate_potential(  cap , filled )  :
    from ddsim.discretize.boundary import GateContact, gate_psi_scaled
    idx2  =   next( c for c  in  cap.contacts if  isinstance( c , GateContact) )
    exp= gate_psi_scaled(idx2.voltage/ cap.scale.psi_0,idx2.work_function,cap.material.T)
    np.testing.assert_allclose(filled[list(idx2.nodes)], exp, rtol  = 1e-12)

def test_a_device_with_no_insulator_is_returned_untouched() :

    doide= pn_diode()

    Guess=neutral(doide)
    assert insulator_guess(doide, Guess) is Guess


def test_an_all_silicon_2d_device_is_returned_untouched():
    capp  =  mos_cap(t_ox  =  1e-6, t_si = 2e-4)
    all = capp.regions.__class__(cell_material = np.zeros_like(capp.regions.cell_material), eps_r=np.ones_like(capp.regions.eps_r), semiconductor_volume  =  capp.mesh.volume, semiconductor_face =capp.mesh.dual_face, oxide_nodes=  np.array([], dtype  = np.int64),)
    import dataclasses
    dev= dataclasses.replace(capp,regions=all)
    geuss =  neutral(dev)
    assert insulator_guess(dev,geuss)is geuss

@pytest.mark.parametrize('v_gate' ,  [- 8.0 ,   -  4.0, 4.0, 8.0], ids  =  str )


def test_a_cold_solve_far_from_flatband_converges_in_a_modest_budget(v_gate):

    deice  =  mos_cap ( gate_voltage =  v_gate  )
    buff = solve_equilibrium (deice ,  max_iterations  =   20)
    assert buff.newton.converged


def test_the_flat_stack_is_recognised_as_already_solved():
    vfb =float(C.work_function_difference(C.PHI_M_N_POLY,-1e16))
    State= solve_equilibrium(mos_cap(gate_voltage = vfb))
    assert State.newton.iterations  ==  0
