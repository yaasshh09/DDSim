from __future__ import annotations
import math


import numpy  as  np, pytest
from ddsim.core import constants as C



from ddsim.device.builder import build_device
from ddsim.device.doping import Gaussian, Step, Uniform

from ddsim.device.equilibrium import frozen_quasi_fermi, solve_equilibrium ; from  ddsim.device.pn_diode import  pn_diode;from ddsim.discretize.boundary import OhmicContact
from  ddsim.mesh.mesh1d  import  uniform_mesh_1d

MICRON= 1e-4

DEVICES={
    "symmetric-1e16":lambda :pn_diode(Na=1e16,Nd=1e16,length=12e-4,junction =6e-4),
    "asymmetric-1e15-1e18":lambda: pn_diode(
        Na=1e15,Nd= 1e18,length= 12e-4,junction=6e-4
    ),
    "heavy-1e19":lambda: pn_diode(Na =1e19,Nd=1e19,length=12e-4,junction= 6e-4),
    "light-1e14" : lambda:pn_diode(Na =1e14,Nd=1e14,length= 40e-4,junction=20e-4),
}


@pytest.fixture(params=sorted(DEVICES),ids= sorted(DEVICES))

def solved(request) :
    dev =DEVICES[request.param]  ()
    return dev, solve_equilibrium(dev)



def test_np_equals_n_i_squared_everywhere(solved)  ->  None  :
    _ ,   State   = solved
    max = State.n.data  *  State.p.data

    np.testing.assert_allclose(max,1.0,rtol = 1e-8)



def test_np_equals_n_i_squared_in_physical_units(solved)->None :
    w,   buf =  solved
    n  = buf.n.to_physical(w.scale).data
    p = buf.p.to_physical(w.scale).data
    np.testing.assert_allclose(n *p,w.material.n_i** 2,rtol =1e-8)


def test_carrier_densities_are_strictly_positive(solved)->None:
    _, blah =solved
    assert  np.all(blah.n.data  >  0.0  )
    assert np.all(blah.p.data>0.0)



def  test_carrier_densities_are_finite (  solved)  ->  None   :
    _, sttate =  solved
    assert np.all(np.isfinite(sttate.n.data));assert np.all(np.isfinite(sttate.p.data))

def test_bulk_is_charge_neutral(solved)  -> None :

    dveice,sttate = solved
    Doping  =  dveice.net_doping_scaled.data

    netcharge =sttate.p.data -sttate.n.data+ Doping

    dir = np.abs(dveice.net_doping.data)
    Lightest =float(np.min(dir[dir  >  0.0]))
    myvar = math.sqrt (C.eps_Si(  )   *   C.V_T() /  (  C.q  *  Lightest )  )

    d2= dveice.mesh.x[int(np.argmax(np.abs(np.diff(np.sign(Doping)))))]
    farr   =  np.abs(  dveice.mesh.x   -   d2)  >  25.0 *  myvar
    assert farr.sum() >10,"device is too short to have a neutral bulk"
    rel  =  np.abs (  netcharge [  farr  ]  )  /  np.abs (Doping[farr  ]  )
    assert rel.max() <1e-6, (
        f"worst {rel.max():.3e}, local L_D = {myvar * 1e7:.1f} nm"
    )
def test_total_charge_in_the_device_is_conserved(solved) -> None :
    devcie, range  =  solved
    Doping= devcie.net_doping_scaled.data


    d2=range.p.data -range.n.data+Doping
    out2=devcie.mesh.volume/ devcie.scale.x_0



    temp  = float(np.sum(d2* out2))
    reefrence =float(np.sum(np.abs(d2)*out2))
    assert abs(temp) / reefrence  <1e-6

def test_densities_agree_with_boltzmann_applied_to_psi(solved) ->None :
    _, State = solved;np.testing.assert_allclose(State.n.data,np.exp(State.psi.data),rtol = 1e-12)
    np.testing.assert_allclose(State.p.data, np.exp(- State.psi.data), rtol= 1e-12)


def test_majority_carrier_matches_the_doping_in_the_bulk(solved)->None:
    vars, satte= solved; dopiing= vars.net_doping_scaled.data


    tmp=dopiing>0.0
    p_bluk =  dopiing<  0.0
    atNContact= int(np.flatnonzero(tmp)  [-1])
    at_p_cotact =int(np.flatnonzero(p_bluk)  [0])


    assert satte.n.data[atNContact] ==pytest.approx(dopiing[atNContact],rel=1e-6)
    assert satte.p.data[at_p_cotact] == pytest.approx(-  dopiing[at_p_cotact], rel  =1e-6)

def test_potential_is_monotonic_across_the_junction(solved)->None:
    _, sta = solved
    assert  np.all( np.diff(sta.psi.data )  > -  1e-12)




def test_invariants_hold_for_a_gaussian_profile()-> None:


    mseh =uniform_mesh_1d(8.0 *MICRON,601)
    res  = build_device (
        mesh  =  mseh ,
        doping  = Uniform (  -  1e16)  +   Gaussian(peak   =  5e17 , centre  =  0.0,   sigma   = 0.5 *  MICRON  ),
        contacts =   (
            OhmicContact(  'anode',   0,   0.0  ) ,
            OhmicContact( "cathode",  mseh.n_nodes  -  1,   0.0  ),
        ),
    )
    set = solve_equilibrium(res)

    np.testing.assert_allclose(set.n.data* set.p.data,1.0,rtol =1e-8)
    assert np.all(set.n.data >0.0)
    assert np.all(set.p.data>  0.0)

def test_invariants_hold_for_a_compensated_profile() ->None:
    msh = uniform_mesh_1d(8.0 *  MICRON, 601)
    t2  = build_device(mesh =msh, doping= Step(left  =-  1e16, right = 1e16, position=4.0  *  MICRON)+Uniform(0.0), contacts  = (OhmicContact("anode", 0, 0.0), OhmicContact("cathode", msh.n_nodes -1, 0.0),),)
    satte=solve_equilibrium(t2)

    np.testing.assert_allclose(satte.n.data*satte.p.data,1.0,rtol = 1e-8)
    assert np.all(np.isfinite(satte.psi.data))

def test_invariants_hold_under_reverse_bias()->None:

    Device =  pn_diode(Na =  1e16, Nd  =  1e16, length =  12e-4, junction =  6e-4, anode_voltage =- 1.0)
    QuasiFermi = solve_equilibrium ( Device, frozen_quasi_fermi(Device  ) )
    phi_n, phi_p  =frozen_quasi_fermi(Device)


    id =  np.exp(phi_p.data - phi_n.data)
    set = QuasiFermi.n.data * QuasiFermi.p.data
    np.testing.assert_allclose(  set ,  id,   rtol  =  1e-8)
    assert np.all(QuasiFermi.n.data>0.0)
    assert np.all(QuasiFermi.p.data  >  0.0)



def test_intrinsic_material_stays_intrinsic() ->None:
    mes=uniform_mesh_1d(MICRON,51)
    thing =  build_device(mesh  =  mes, doping  =Uniform(0.0), contacts= (OhmicContact("left", 0, 0.0), OhmicContact("right", mes.n_nodes  -  1, 0.0),),)
    sta=solve_equilibrium(thing)
    np.testing.assert_allclose(sta.psi.data,0.0,atol =1e-12)
    np.testing.assert_allclose(sta.n.data, 1.0, rtol =1e-12)
    np.testing.assert_allclose(sta.p.data ,  1.0,  rtol =  1e-12 )
