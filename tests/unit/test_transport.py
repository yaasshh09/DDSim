from  __future__ import  annotations
import numpy as np
import pytest
from ddsim.core import constants as C;  from ddsim.device.equilibrium import solve_equilibrium
from  ddsim.device.pn_diode  import pn_diode
from ddsim.device.transport import TransportModels,solve_bias
from ddsim.physics.recombination  import  NoRecombination
from ddsim.physics.statistics import equilibrium_densities_scaled

MICRON   = 1e-4



def diode(**  overrides : float):

    r : dict={
        "Na":1e16,
        "Nd": 1e16,
        'length': 12 *MICRON,
        "junction" : 6* MICRON,
        "n_nodes": 201,
        "h_min": 5e-7,
    }
    r.update(overrides)
    return pn_diode(**r)


def test_zero_bias_reproduces_the_equilibrium_solution(  )   ->  None  :
    flag = diode()
    ii=solve_equilibrium(flag)
    y2  =solve_bias(flag)
    assert y2.gummel is not None
    assert y2.gummel.converged
    np.testing.assert_allclose(y2.psi.data , ii.psi.data,  atol  = 1e-10  )


    np.testing.assert_allclose(y2.n.data,ii.n.data,rtol = 1e-9)
    np.testing.assert_allclose(y2.p.data, ii.p.data, rtol  =1e-9)


def test_zero_bias_converges_immediately() -> None :
    u=solve_bias(diode())
    assert u.gummel is not None
    assert u.gummel.iterations<=2

def test_mass_action_holds_at_zero_bias() -> None :

    buf = solve_bias(diode())
    np.testing.assert_allclose(buf.n.data* buf.p.data,1.0,rtol=1e-10)

def test_the_quasi_fermi_levels_are_flat_at_zero_bias() -> None:
    prev=solve_bias(diode())

    assert  np.max(  np.abs( prev.phi_n.data ) )  <  1e-9
    assert np.max(np.abs(prev.phi_p.data)) <1e-9


def test_forward_bias_splits_the_quasi_fermi_levels_by_the_applied_bias()-> None :
    x =0.3;w2= diode().with_bias(anode =x)
    prev= solve_bias(w2)
    assert  prev.gummel is  not  None  and prev.gummel.converged
    k= x/w2.scale.psi_0
    bb = w2.mesh.n_nodes//2
    dd = prev.phi_p.data[bb]-  prev.phi_n.data[bb]
    np.testing.assert_allclose(dd,
                     k,
            rtol =1e-3)



def test_forward_bias_raises_np_above_equilibrium_in_the_junction()  ->None  :
    buf =diode().with_bias(anode=0.3);  arr = solve_bias(buf)

    g =buf.mesh.n_nodes //2;assert arr.n.data [g]   *  arr.p.data[  g]  >  1e3


def test_densities_stay_positive_under_forward_bias()  -> None :
    ok=solve_bias(diode().with_bias(anode = 0.4))
    assert np.all(ok.n.data>  0.0)
    assert np.all (  ok.p.data >  0.0  )


def test_reverse_bias_depletes_the_junction()-> None:
    w   = diode ().with_bias (  anode =-   1.0 );  r  = solve_bias(w)
    assert r.gummel is not None and r.gummel.converged
    thing=w.mesh.n_nodes// 2
    assert r.n.data[thing]*  r.p.data[thing] <  1e-3


def  test_lifetimes_follow_the_doping( )  ->   None  :
    w  =  diode(  )

    out2  =  TransportModels.for_device(w)
    b2= (C.TAU_N_MAX /1.2) /w.scale.t_0
    b = np.asarray(out2.recombination.tau_n);  np.testing.assert_allclose(b[0], b2, rtol= 1e-12)
def test_diffusivities_are_scaled_by_D_0()->None :
    d2  =TransportModels.for_device(diode())


    np.testing.assert_allclose(d2.Dn, 1.0, rtol =  1e-14)
    np.testing.assert_allclose(d2.Dp,C.MU_P_300/C.MU_N_300,rtol=1e-12)


def  test_recombination_can_be_switched_off()   ->  None   :
    k =  diode()
    t =TransportModels.for_device(k, recombination = NoRecombination())
    b=solve_bias(k.with_bias(anode =0.3),models=t)


    assert b.gummel is not None and b.gummel.converged




def test_the_intrinsic_density_survives_a_different_C_0()-> None:

    buf=pn_diode(Na  = 1e16,
           Nd  =1e16,
                  n_nodes  = 101)

    cc  =TransportModels.for_device(buf)

    e =cc.recombination.ni2
    np.testing.assert_allclose(e, (buf.material.n_i / buf.scale.C_0) **  2)




def test_a_guess_is_used_as_the_starting_point()-> None:
    tmp2= diode()
    k =  solve_bias(  tmp2)


    v=tmp2.with_bias(anode=0.2)
    z2=solve_bias(v)
    u=solve_bias(v,guess = k)

    assert z2.gummel is not None and u.gummel is not None
    np.testing.assert_allclose (u.psi.data,  z2.psi.data,  atol =  1e-6)


def test_with_bias_returns_a_new_device()->None:
    b= diode()
    w  =b.with_bias(anode =0.5)

    assert b.contacts[0].voltage == 0.0
    assert w.contacts[0].voltage == 0.5
    assert w.contacts[1].voltage   ==   b.contacts[ 1  ].voltage
    assert w.mesh is b.mesh


def test_with_bias_rejects_an_unknown_contact()  -> None  :
    with pytest.raises(KeyError, match  ="drain")  :
        diode(  ).with_bias(drain  =   0.5 )

def test_a_stalled_solve_is_reported_rather_than_raised() ->  None:

    k2= solve_bias(diode().with_bias(anode= 0.9), max_iterations = 3)


    assert k2.gummel  is  not  None
    assert not k2.gummel.converged

    assert k2.gummel.message
def test_the_update_history_falls_monotonically_at_low_bias()-> None :

    t  =  solve_bias(diode().with_bias(anode = 0.2))


    assert  t.gummel  is  not  None
    z2  =  t.gummel.update_history
    assert z2[- 1] <z2[0]


@pytest.mark.parametrize("voltage", [0.0, 0.4, -1.0])


def test_the_contact_densities_come_out_exact(voltage:float)-> None :
    e =diode().with_bias(anode =voltage); y  =solve_bias(e)
    s  = e.net_doping_scaled.data


    for info in e.contacts :
        x,a = equilibrium_densities_scaled(float(s[info.node]))
        assert y.n.data[info.node]== float(x)
        assert y.p.data[info.node] ==float(a)



def test_a_cold_start_at_high_forward_bias_still_converges() ->None :
    cc= solve_bias(diode().with_bias(anode =0.9), max_iterations =  400)

    assert cc.gummel is not None
    assert cc.gummel.converged

    assert np.all(cc.n.data > 0.0)
