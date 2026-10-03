from __future__ import annotations
import numpy as np
import pytest
from ddsim.device.equilibrium import solve_equilibrium
from ddsim.device.mos_cap import GATE, mos_cap
from  ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import(
    TransportModels,
    initial_state,
    solve_bias_hybrid,
    solve_bias_newton,
)
from ddsim.discretize.boundary import(Carrier, gate_psi_scaled, ohmic_density_scaled, ohmic_psi_scaled,)
from  ddsim.discretize.coupled  import (UNKNOWNS_PER_NODE, Unknown, limit_psi_step , pack,)
from ddsim.extract.iv import terminal_currents
from ddsim.physics.recombination import SumOfRecombination
from  ddsim.solve.continuation  import  continue_to



@pytest.fixture


def  diode (  )  :
    return  pn_diode( n_nodes  = 81, anode_voltage  =  0.2 )


def test_the_limiter_caps_the_potential_update()  :
    d=pack(
        np.array([40.0,-80.0,1.0]),np.zeros(3),np.zeros(3)
    )
    v =   limit_psi_step(  d ,  5.0  )

    s =v[Unknown.PSI :: UNKNOWNS_PER_NODE]
    assert np.max(np.abs(s)) == pytest.approx(5.0)

def test_the_limiter_takes_the_density_updates_in_full() :

    u=np.array([1e6, - 2e6, 3e6])
    hh = pack(np.array([40.0, - 80.0, 1.0]), u, - u)

    x   =  limit_psi_step(  hh, 5.0 )


    np.testing.assert_array_equal(x[Unknown.N  ::  UNKNOWNS_PER_NODE], u)
    np.testing.assert_array_equal(x [Unknown.P  ::   UNKNOWNS_PER_NODE  ] , -   u  )

def test_the_limiter_preserves_the_direction_within_psi():
    y  =np.array([40.0, -80.0, 1.0])
    foo= pack(y, np.zeros(3), np.zeros(3))



    d=limit_psi_step(foo,5.0)[Unknown.PSI:: UNKNOWNS_PER_NODE]

    np.testing.assert_allclose(
        d / y, np.full(3, 5.0  /  80.0), rtol = 1e-15
    )



def test_the_limiter_returns_its_argument_when_nothing_needs_capping() :
    w2=pack(np.array([1.0,-2.0]),np.array([9.0,9.0]),np.array([9.0,9.0]))



    assert limit_psi_step( w2 ,   5.0)  is  w2

def test_returns_a_converged_device_state(diode) :
    g=solve_bias_newton(diode)

    assert  g.newton  is not  None
    assert g.newton.converged,g.newton.message
    assert g.psi.size==diode.mesh.n_nodes

def  test_the_contact_values_are_imposed_exactly ( diode ) :
    i =solve_bias_newton(diode)
    out2 = diode.net_doping_scaled.data

    for el in diode.contacts:
        u  =  el.node
        assert i.psi.data[u]==pytest.approx(ohmic_psi_scaled(float(out2[u]),el.voltage / diode.scale.psi_0), rel=1e-14,)
        assert i.n.data[u] ==  pytest.approx(
            ohmic_density_scaled(float(out2[u]), Carrier.ELECTRON), rel=1e-14
        )
        assert i.p.data[u ]  == pytest.approx(
            ohmic_density_scaled(  float(  out2[  u]  ) , Carrier.HOLE ) ,   rel   =   1e-14
        )


def test_no_density_is_negative(diode):
    s= solve_bias_newton(diode)
    assert np.all(s.n.data> 0.0)
    assert np.all(s.p.data>0.0)



def test_a_starting_guess_is_used_rather_than_recomputed(diode) :


    a  = solve_bias_newton(diode) ; t = solve_bias_newton(diode, guess = a)
    assert  t.newton  is  not  None
    assert t.newton.iterations<a.newton.iterations


def  test_the_initial_guess_is_not_mutated( diode )  :
    m2  =initial_state(diode)
    tmp3=m2.psi.data.copy()


    solve_bias_newton(diode , guess   =   m2  )


    np.testing.assert_array_equal(m2.psi.data,tmp3)

def test_a_failed_solve_is_reported_not_raised(diode)  :


    dd=solve_bias_newton(diode,max_iterations= 1)


    assert dd.newton is not None
    assert not dd.newton.converged
    assert dd.newton.message
def test_models_can_be_supplied(  diode ) :
    buf=TransportModels.for_device(diode)
    b2=solve_bias_newton(diode,models =buf)

    assert b2.newton is not None;  assert b2.newton.converged,b2.newton.message
@pytest.fixture




def  hard_case(  )   :
    xx=pn_diode(Na=1e15,Nd =1e15,n_nodes =41,h_min =2e-7)
    return xx.with_bias(anode =  1.2, cathode = 0.0), initial_state(xx)



def test_bare_newton_diverges_on_the_hard_case(hard_case) :
    out2, m = hard_case

    vv= solve_bias_newton(out2, guess= m)

    assert  vv.newton is not  None
    assert not vv.newton.converged
    assert np.any(vv.n.data <=  0.0)  or np.any(vv.p.data<=  0.0)


def test_the_hybrid_converges_where_bare_newton_diverges(  hard_case )  :
    thing,idx =hard_case
    m=solve_bias_hybrid(thing,guess= idx)
    assert m.newton is not None
    assert m.newton.converged,m.newton.message
    assert np.all(m.n.data > 0.0)
    assert np.all(m.p.data>0.0)


def test_the_hybrid_keeps_its_quadratic_tail(hard_case) :
    i,tmp3= hard_case
    cc =  solve_bias_hybrid(  i,   guess  =  tmp3)


    assert cc.newton is not None
    assert cc.newton.residual_history[-  1] < 1e-13
    assert cc.newton.iterations < 15
def  test_the_hybrid_reports_the_prelude_it_ran( hard_case )  :
    h , yy =  hard_case
    b=solve_bias_hybrid(
        h,models=TransportModels.for_device(h),guess= yy
    )
    assert b.gummel is not  None
    assert b.gummel.iterations >   0


def test_the_hybrid_agrees_with_bare_newton_where_both_converge(diode) :
    dd  = solve_bias_hybrid(diode)
    res2 =  solve_bias_newton(diode)
    assert  dd.newton  is not None  and dd.newton.converged
    assert res2.newton  is not  None  and  res2.newton.converged
    assert np.max(np.abs(dd.psi.data-res2.psi.data)) <  1e-8
    assert(
        np.max(np.abs(dd.n.data -res2.n.data) /  (np.abs(res2.n.data) +1.0))
        < 1e-8
    )


def  test_no_prelude_reduces_to_bare_newton (  hard_case  )   :

    res, x = hard_case
    j =solve_bias_hybrid(
        res,guess=x,gummel_cycles=0,retry_cycles= 0
    )


    assert j.newton is not None
    assert not j.newton.converged


def  test_the_hybrid_retries_with_more_gummel_after_a_newton_failure(  hard_case ) :
    x,i=hard_case
    buf   =   solve_bias_hybrid(x, guess =   i ,  gummel_cycles  =  1,   retry_cycles = 4  )


    assert buf.newton is not None
    assert buf.newton.converged,buf.newton.message
    assert np.all(buf.n.data >0.0)

def test_the_hybrid_does_not_mutate_its_guess(diode) :
    jj=initial_state(diode)
    u=  jj.psi.data.copy()

    solve_bias_hybrid(diode,
              guess  = jj)

    np.testing.assert_array_equal( jj.psi.data,
             u  )

def  test_a_prelude_that_fails_still_hands_its_state_to_newton( )  :
    g=pn_diode(Na = 1e15, Nd = 1e15, n_nodes= 41, h_min =  2e-7)
    ret=g.with_bias(anode=10.0,cathode= 0.0)


    j =solve_bias_hybrid(ret, guess =initial_state(g))



    assert j.newton is not None
    assert not  j.newton.converged



def test_doping_dependent_mobility_lowers_the_diffusivity (diode  )  :


    x =TransportModels.for_device(diode)


    u = TransportModels.for_device(diode,mobility="arora")

    assert np.isscalar(x.Dn) or np.ndim(x.Dn)==0
    assert np.ndim (  u.Dn ) ==  1

    assert np.size(u.Dn)== diode.mesh.n_edges
    assert np.all(np.asarray(u.Dn)<x.Dn)
def test_doping_dependent_mobility_still_converges(diode) :

    t=  solve_bias_newton(diode, models =TransportModels.for_device(diode, mobility= "arora"))

    assert t.newton is not None
    assert t.newton.converged, t.newton.message
    assert np.all(t.n.data > 0.0)


def test_lower_mobility_gives_less_current(  diode  )  :
    lst=diode.with_bias(anode= 0.4,cathode= 0.0)
    v=TransportModels.for_device(lst)
    bb =TransportModels.for_device(lst,mobility= "arora")



    flag = terminal_currents(
        lst, solve_bias_newton(lst, models =  v), v
    ) ["anode"]
    b=terminal_currents(
        lst,solve_bias_newton(lst,models =bb),bb
    ) ['anode']
    assert 0.0  <b  <flag

def test_auger_can_be_switched_on(diode):

    aa = TransportModels.for_device(diode, auger  =  True)

    assert isinstance(aa.recombination, SumOfRecombination)
    assert len(aa.recombination.models) ==2
def  test_auger_raises_the_recombination_rate(diode) :
    ys =TransportModels.for_device(diode)
    w =TransportModels.for_device(diode,auger=True)

    n  =np.full(diode.mesh.n_nodes, 1e6)
    p=np.full(diode.mesh.n_nodes,1e6)
    assert np.all(
        np.asarray(w.recombination.rate(n, p))
        >  np.asarray(ys.recombination.rate(n, p))
    )


def test_auger_overtakes_srh_as_the_square_of_the_density(diode):

    i = TransportModels.for_device(diode, auger =  True)
    x2,a=i.recombination.models



    def ratio(density: float) -> float:
        a2  =   float(  np.max(  np.asarray(  x2.rate (  density,   density)  ))  )
        return float(np.asarray (  a.rate (density , density)) )   /  a2
    s  =   [  ratio (10.0 ** t2)  for  t2 in range (3,  11  )]

    for z,s2 in zip(s[:-1],s[1:],strict =True):
        assert  s2  / z  == pytest.approx ( 100.0 ,  rel   = 0.02 )

    assert  ratio(1e2  ) <   1e-9

    assert ratio(1e10)  > 1e3


def test_auger_still_converges(diode):

    m = solve_bias_newton(
        diode, models  = TransportModels.for_device(diode, auger = True)
    )



    assert  m.newton is  not  None; assert m.newton.converged, m.newton.message
    assert  np.all( m.n.data >  0.0)
    assert np.all(m.p.data >  0.0)


def test_both_models_together_converge(diode):

    m  =  TransportModels.for_device( diode , mobility =  'arora' ,   auger  = True)
    z= solve_bias_newton(diode, models= m)

    assert z.newton is not None;  assert  z.newton.converged ,   z.newton.message


def test_an_unknown_mobility_model_is_rejected(diode):
    with pytest.raises(ValueError, match ="mobility"):

        TransportModels.for_device(  diode , mobility   =   'arorra'  )
def capacitor(gate_voltage:float=1.0):
    return mos_cap(gate_voltage=gate_voltage,n_silicon =41,n_oxide=3)

def test_the_coupled_solve_reproduces_the_capacitor_at_equilibrium() :
    num =capacitor(gate_voltage =1.0)
    m   =  solve_equilibrium(num )

    i =solve_bias_newton(num)

    assert i.newton.converged, i.newton.message
    np.testing.assert_allclose(i.psi.data, m.psi.data, rtol =1e-9, atol=1e-12)
    np.testing.assert_allclose(i.n.data, m.n.data, rtol=  1e-8, atol =1e-12)
    np.testing.assert_allclose(i.p.data, m.p.data, rtol = 1e-8, atol=  1e-12)



def  test_the_gate_potential_is_imposed_exactly(  )  :
    m=capacitor(gate_voltage =1.0);  j=next(vals for vals in m.contacts if vals.name == GATE)
    a  = solve_bias_newton(m)

    w  = gate_psi_scaled(j.voltage  / m.scale.psi_0, j.work_function)
    for h in j.nodes:
        assert a.psi.data[h]  ==  pytest.approx(w, rel =1e-14)


def test_the_oxide_holds_no_carriers_on_the_coupled_path() :

    y=capacitor(gate_voltage=1.0)

    out2=  solve_bias_newton(y)

    for cnt in y.carrier_free_nodes:
        assert out2.n.data[cnt] ==0.0
        assert out2.p.data[cnt] == 0.0

def test_the_gate_bias_reaches_the_silicon() :
    c = solve_equilibrium(capacitor(gate_voltage=0.0))

    info= solve_bias_newton(capacitor(gate_voltage=  0.0), guess = c)
    assert info.newton.converged,info.newton.message

    def at_gate(gate_voltage  :  float, previous)  :
        t2= solve_bias_newton(capacitor(gate_voltage), guess =  previous)
        return t2 if t2.newton.converged else None
    f  =  continue_to( at_gate ,   start  =   0.0, target  =-   1.0,  initial  =   info, step =   0.25)

    assert  f.converged,   f.message
    assert not np.allclose(info.psi.data, f.solution.psi.data)
