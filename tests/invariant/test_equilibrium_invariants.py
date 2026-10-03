from __future__ import annotations
import math
import numpy as np, pytest
from ddsim.core import constants as C
from ddsim.device.builder import build_device
from ddsim.device.doping import Gaussian, Step, Uniform
from ddsim.device.equilibrium import frozen_quasi_fermi, solve_equilibrium ; from ddsim.device.pn_diode import pn_diode;from ddsim.discretize.boundary import OhmicContact
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
    c =DEVICES[request.param]  ()
    return c, solve_equilibrium(c)



def test_np_equals_n_i_squared_everywhere(solved)  ->  None  :
    _ ,   f   = solved
    xx = f.n.data  *  f.p.data

    np.testing.assert_allclose(xx,1.0,rtol = 1e-8)



def test_np_equals_n_i_squared_in_physical_units(solved)->None :
    f,   m =  solved
    n  = m.n.to_physical(f.scale).data
    p = m.p.to_physical(f.scale).data
    np.testing.assert_allclose(n *p,f.material.n_i** 2,rtol =1e-8)


def test_carrier_densities_are_strictly_positive(solved)->None:
    _, ys =solved
    assert  np.all(ys.n.data  >  0.0  )
    assert np.all(ys.p.data>0.0)



def  test_carrier_densities_are_finite (  solved)  ->  None   :
    _, e =  solved
    assert np.all(np.isfinite(e.n.data));assert np.all(np.isfinite(e.p.data))

def test_bulk_is_charge_neutral(solved)  -> None :

    c,buf = solved
    y  =  c.net_doping_scaled.data

    j =buf.p.data -buf.n.data+ y

    b = np.abs(c.net_doping.data)
    w =float(np.min(b[b  >  0.0]))
    m2 = math.sqrt (C.eps_Si(  )   *   C.V_T() /  (  C.q  *  w )  )

    cnt= c.mesh.x[int(np.argmax(np.abs(np.diff(np.sign(y)))))]
    u   =  np.abs(  c.mesh.x   -   cnt)  >  25.0 *  m2
    assert u.sum() >10,"device is too short to have a neutral bulk"
    ok  =  np.abs (  j [  u  ]  )  /  np.abs (y[u  ]  )
    assert ok.max() <1e-6, (
        f"worst {ok.max():.3e}, local L_D = {m2 * 1e7:.1f} nm"
    )
def test_total_charge_in_the_device_is_conserved(solved) -> None :
    rr, f  =  solved
    bar= rr.net_doping_scaled.data


    j=f.p.data -f.n.data+bar
    z=rr.mesh.volume/ rr.scale.x_0



    v  = float(np.sum(j* z))
    v2 =float(np.sum(np.abs(j)*z))
    assert abs(v) / v2  <1e-6

def test_densities_agree_with_boltzmann_applied_to_psi(solved) ->None :
    _, f = solved;np.testing.assert_allclose(f.n.data,np.exp(f.psi.data),rtol = 1e-12)
    np.testing.assert_allclose(f.p.data, np.exp(- f.psi.data), rtol= 1e-12)


def test_majority_carrier_matches_the_doping_in_the_bulk(solved)->None:
    v2, zz= solved; s= v2.net_doping_scaled.data


    m=s>0.0
    s2 =  s<  0.0
    res= int(np.flatnonzero(m)  [-1])
    h =int(np.flatnonzero(s2)  [0])


    assert zz.n.data[res] ==pytest.approx(s[res],rel=1e-6)
    assert zz.p.data[h] == pytest.approx(-  s[h], rel  =1e-6)

def test_potential_is_monotonic_across_the_junction(solved)->None:
    _, w = solved
    assert  np.all( np.diff(w.psi.data )  > -  1e-12)




def test_invariants_hold_for_a_gaussian_profile()-> None:


    j =uniform_mesh_1d(8.0 *MICRON,601)
    z  = build_device (
        mesh  =  j ,
        doping  = Uniform (  -  1e16)  +   Gaussian(peak   =  5e17 , centre  =  0.0,   sigma   = 0.5 *  MICRON  ),
        contacts =   (
            OhmicContact(  'anode',   0,   0.0  ) ,
            OhmicContact( "cathode",  j.n_nodes  -  1,   0.0  ),
        ),
    )
    h = solve_equilibrium(z)

    np.testing.assert_allclose(h.n.data* h.p.data,1.0,rtol =1e-8)
    assert np.all(h.n.data >0.0)
    assert np.all(h.p.data>  0.0)

def test_invariants_hold_for_a_compensated_profile() ->None:
    x = uniform_mesh_1d(8.0 *  MICRON, 601)
    a  = build_device(mesh =x, doping= Step(left  =-  1e16, right = 1e16, position=4.0  *  MICRON)+Uniform(0.0), contacts  = (OhmicContact("anode", 0, 0.0), OhmicContact("cathode", x.n_nodes -1, 0.0),),)
    g=solve_equilibrium(a)

    np.testing.assert_allclose(g.n.data*g.p.data,1.0,rtol = 1e-8)
    assert np.all(np.isfinite(g.psi.data))

def test_invariants_hold_under_reverse_bias()->None:

    cc =  pn_diode(Na =  1e16, Nd  =  1e16, length =  12e-4, junction =  6e-4, anode_voltage =- 1.0)
    h = solve_equilibrium ( cc, frozen_quasi_fermi(cc  ) )
    phi_n, phi_p  =frozen_quasi_fermi(cc)


    y =  np.exp(phi_p.data - phi_n.data)
    bb = h.n.data * h.p.data
    np.testing.assert_allclose(  bb ,  y,   rtol  =  1e-8)
    assert np.all(h.n.data>0.0)
    assert np.all(h.p.data  >  0.0)



def test_intrinsic_material_stays_intrinsic() ->None:
    w=uniform_mesh_1d(MICRON,51)
    arr =  build_device(mesh  =  w, doping  =Uniform(0.0), contacts= (OhmicContact("left", 0, 0.0), OhmicContact("right", w.n_nodes  -  1, 0.0),),)
    m=solve_equilibrium(arr)
    np.testing.assert_allclose(m.psi.data,0.0,atol =1e-12)
    np.testing.assert_allclose(m.n.data, 1.0, rtol =1e-12)
    np.testing.assert_allclose(m.p.data ,  1.0,  rtol =  1e-12 )
