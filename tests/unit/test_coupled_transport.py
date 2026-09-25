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
    Delta=pack(
        np.array([40.0,-80.0,1.0]),np.zeros(3),np.zeros(3)
    )
    limtied =   limit_psi_step(  Delta ,  5.0  )

    k2 =limtied[Unknown.PSI :: UNKNOWNS_PER_NODE]
    assert np.max(np.abs(k2)) == pytest.approx(5.0)

def test_the_limiter_takes_the_density_updates_in_full() :

    obj2=np.array([1e6, - 2e6, 3e6])
    Delta = pack(np.array([40.0, - 80.0, 1.0]), obj2, - obj2)

    lmited   =  limit_psi_step(  Delta, 5.0 )


    np.testing.assert_array_equal(lmited[Unknown.N  ::  UNKNOWNS_PER_NODE], obj2)
    np.testing.assert_array_equal(lmited [Unknown.P  ::   UNKNOWNS_PER_NODE  ] , -   obj2  )

def test_the_limiter_preserves_the_direction_within_psi():
    psi_updte  =np.array([40.0, -80.0, 1.0])
    dleta= pack(psi_updte, np.zeros(3), np.zeros(3))



    lim=limit_psi_step(dleta,5.0)[Unknown.PSI:: UNKNOWNS_PER_NODE]

    np.testing.assert_allclose(
        lim / psi_updte, np.full(3, 5.0  /  80.0), rtol = 1e-15
    )



def test_the_limiter_returns_its_argument_when_nothing_needs_capping() :
    deta=pack(np.array([1.0,-2.0]),np.array([9.0,9.0]),np.array([9.0,9.0]))



    assert limit_psi_step( deta ,   5.0)  is  deta

def test_returns_a_converged_device_state(diode) :
    sta=solve_bias_newton(diode)

    assert  sta.newton  is not  None
    assert sta.newton.converged,sta.newton.message
    assert sta.psi.size==diode.mesh.n_nodes

def  test_the_contact_values_are_imposed_exactly ( diode ) :
    State =solve_bias_newton(diode)
    dopnig = diode.net_doping_scaled.data

    for temp2 in diode.contacts:
        Node  =  temp2.node
        assert State.psi.data[Node]==pytest.approx(ohmic_psi_scaled(float(dopnig[Node]),temp2.voltage / diode.scale.psi_0), rel=1e-14,)
        assert State.n.data[Node] ==  pytest.approx(
            ohmic_density_scaled(float(dopnig[Node]), Carrier.ELECTRON), rel=1e-14
        )
        assert State.p.data[Node ]  == pytest.approx(
            ohmic_density_scaled(  float(  dopnig[  Node]  ) , Carrier.HOLE ) ,   rel   =   1e-14
        )


def test_no_density_is_negative(diode):
    sttate= solve_bias_newton(diode)
    assert np.all(sttate.n.data> 0.0)
    assert np.all(sttate.p.data>0.0)



def test_a_starting_guess_is_used_rather_than_recomputed(diode) :


    war  = solve_bias_newton(diode) ; Again = solve_bias_newton(diode, guess = war)
    assert  Again.newton  is  not  None
    assert Again.newton.iterations<war.newton.iterations


def  test_the_initial_guess_is_not_mutated( diode )  :
    Guess  =initial_state(diode)
    lst=Guess.psi.data.copy()


    solve_bias_newton(diode , guess   =   Guess  )


    np.testing.assert_array_equal(Guess.psi.data,lst)

def test_a_failed_solve_is_reported_not_raised(diode)  :


    w=solve_bias_newton(diode,max_iterations= 1)


    assert w.newton is not None
    assert not w.newton.converged
    assert w.newton.message
def test_models_can_be_supplied(  diode ) :
    Models=TransportModels.for_device(diode)
    object=solve_bias_newton(diode,models =Models)

    assert object.newton is not None;  assert object.newton.converged,object.newton.message
@pytest.fixture




def  hard_case(  )   :
    Unbiased=pn_diode(Na=1e15,Nd =1e15,n_nodes =41,h_min =2e-7)
    return Unbiased.with_bias(anode =  1.2, cathode = 0.0), initial_state(Unbiased)



def test_bare_newton_diverges_on_the_hard_case(hard_case) :
    dev, w = hard_case

    staate= solve_bias_newton(dev, guess= w)

    assert  staate.newton is not  None
    assert not staate.newton.converged
    assert np.any(staate.n.data <=  0.0)  or np.any(staate.p.data<=  0.0)


def test_the_hybrid_converges_where_bare_newton_diverges(  hard_case )  :
    obj2,buff =hard_case
    idx2=solve_bias_hybrid(obj2,guess= buff)
    assert idx2.newton is not None
    assert idx2.newton.converged,idx2.newton.message
    assert np.all(idx2.n.data > 0.0)
    assert np.all(idx2.p.data>0.0)


def test_the_hybrid_keeps_its_quadratic_tail(hard_case) :
    type,tmp2= hard_case
    sta =  solve_bias_hybrid(  type,   guess  =  tmp2)


    assert sta.newton is not None
    assert sta.newton.residual_history[-  1] < 1e-13
    assert sta.newton.iterations < 15
def  test_the_hybrid_reports_the_prelude_it_ran( hard_case )  :
    dev , guss =  hard_case
    tmp2=solve_bias_hybrid(
        dev,models=TransportModels.for_device(dev),guess= guss
    )
    assert tmp2.gummel is not  None
    assert tmp2.gummel.iterations >   0


def test_the_hybrid_agrees_with_bare_newton_where_both_converge(diode) :
    object  = solve_bias_hybrid(diode)
    Bare =  solve_bias_newton(diode)
    assert  object.newton  is not None  and object.newton.converged
    assert Bare.newton  is not  None  and  Bare.newton.converged
    assert np.max(np.abs(object.psi.data-Bare.psi.data)) <  1e-8
    assert(
        np.max(np.abs(object.n.data -Bare.n.data) /  (np.abs(Bare.n.data) +1.0))
        < 1e-8
    )


def  test_no_prelude_reduces_to_bare_newton (  hard_case  )   :

    item2, bytes = hard_case
    tmp =solve_bias_hybrid(
        item2,guess=bytes,gummel_cycles=0,retry_cycles= 0
    )


    assert tmp.newton is not None
    assert not tmp.newton.converged


def  test_the_hybrid_retries_with_more_gummel_after_a_newton_failure(  hard_case ) :
    dev,Guess=hard_case
    sate   =   solve_bias_hybrid(dev, guess =   Guess ,  gummel_cycles  =  1,   retry_cycles = 4  )


    assert sate.newton is not None
    assert sate.newton.converged,sate.newton.message
    assert np.all(sate.n.data >0.0)

def test_the_hybrid_does_not_mutate_its_guess(diode) :
    gess=initial_state(diode)
    bef=  gess.psi.data.copy()

    solve_bias_hybrid(diode,
              guess  = gess)

    np.testing.assert_array_equal( gess.psi.data,
             bef  )

def  test_a_prelude_that_fails_still_hands_its_state_to_newton( )  :
    unb=pn_diode(Na = 1e15, Nd = 1e15, n_nodes= 41, h_min =  2e-7)
    dev=unb.with_bias(anode=10.0,cathode= 0.0)


    State =solve_bias_hybrid(dev, guess =initial_state(unb))



    assert State.newton is not None
    assert not  State.newton.converged



def test_doping_dependent_mobility_lowers_the_diffusivity (diode  )  :


    s2 =TransportModels.for_device(diode)


    zz = TransportModels.for_device(diode,mobility="arora")

    assert np.isscalar(s2.Dn) or np.ndim(s2.Dn)==0
    assert np.ndim (  zz.Dn ) ==  1

    assert np.size(zz.Dn)== diode.mesh.n_edges
    assert np.all(np.asarray(zz.Dn)<s2.Dn)
def test_doping_dependent_mobility_still_converges(diode) :

    sta=  solve_bias_newton(diode, models =TransportModels.for_device(diode, mobility= "arora"))

    assert sta.newton is not None
    assert sta.newton.converged, sta.newton.message
    assert np.all(sta.n.data > 0.0)


def test_lower_mobility_gives_less_current(  diode  )  :
    bia=diode.with_bias(anode= 0.4,cathode= 0.0)
    costant=TransportModels.for_device(bia)
    aro =TransportModels.for_device(bia,mobility= "arora")



    withconstant = terminal_currents(
        bia, solve_bias_newton(bia, models =  costant), costant
    ) ["anode"]
    q=terminal_currents(
        bia,solve_bias_newton(bia,models =aro),aro
    ) ['anode']
    assert 0.0  <q  <withconstant

def test_auger_can_be_switched_on(diode):

    myvar = TransportModels.for_device(diode, auger  =  True)

    assert isinstance(myvar.recombination, SumOfRecombination)
    assert len(myvar.recombination.models) ==2
def  test_auger_raises_the_recombination_rate(diode) :
    pla =TransportModels.for_device(diode)
    myvar =TransportModels.for_device(diode,auger=True)

    n  =np.full(diode.mesh.n_nodes, 1e6)
    p=np.full(diode.mesh.n_nodes,1e6)
    assert np.all(
        np.asarray(myvar.recombination.rate(n, p))
        >  np.asarray(pla.recombination.rate(n, p))
    )


def test_auger_overtakes_srh_as_the_square_of_the_density(diode):

    mod = TransportModels.for_device(diode, auger =  True)
    srhh,aug=mod.recombination.models



    def ratio(density: float) -> float:
        srh_rate  =   float(  np.max(  np.asarray(  srhh.rate (  density,   density)  ))  )
        return float(np.asarray (  aug.rate (density , density)) )   /  srh_rate
    val  =   [  ratio (10.0 ** E)  for  E in range (3,  11  )]

    for vars,Upper in zip(val[:-1],val[1:],strict =True):
        assert  Upper  / vars  == pytest.approx ( 100.0 ,  rel   = 0.02 )

    assert  ratio(1e2  ) <   1e-9

    assert ratio(1e10)  > 1e3


def test_auger_still_converges(diode):

    sta = solve_bias_newton(
        diode, models  = TransportModels.for_device(diode, auger = True)
    )



    assert  sta.newton is  not  None; assert sta.newton.converged, sta.newton.message
    assert  np.all( sta.n.data >  0.0)
    assert np.all(sta.p.data >  0.0)


def test_both_models_together_converge(diode):

    format  =  TransportModels.for_device( diode , mobility =  'arora' ,   auger  = True)
    State= solve_bias_newton(diode, models= format)

    assert State.newton is not None;  assert  State.newton.converged ,   State.newton.message


def test_an_unknown_mobility_model_is_rejected(diode):
    with pytest.raises(ValueError, match ="mobility"):

        TransportModels.for_device(  diode , mobility   =   'arorra'  )
def capacitor(gate_voltage:float=1.0):
    return mos_cap(gate_voltage=gate_voltage,n_silicon =41,n_oxide=3)

def test_the_coupled_solve_reproduces_the_capacitor_at_equilibrium() :
    Device =capacitor(gate_voltage =1.0)
    Reference   =  solve_equilibrium(Device )

    sate =solve_bias_newton(Device)

    assert sate.newton.converged, sate.newton.message
    np.testing.assert_allclose(sate.psi.data, Reference.psi.data, rtol =1e-9, atol=1e-12)
    np.testing.assert_allclose(sate.n.data, Reference.n.data, rtol=  1e-8, atol =1e-12)
    np.testing.assert_allclose(sate.p.data, Reference.p.data, rtol = 1e-8, atol=  1e-12)



def  test_the_gate_potential_is_imposed_exactly(  )  :
    Device=capacitor(gate_voltage =1.0);  gat=next(c for c in Device.contacts if c.name == GATE)
    sta  = solve_bias_newton(Device)

    expectted  = gate_psi_scaled(gat.voltage  / Device.scale.psi_0, gat.work_function)
    for round in gat.nodes:
        assert sta.psi.data[round]  ==  pytest.approx(expectted, rel =1e-14)


def test_the_oxide_holds_no_carriers_on_the_coupled_path() :

    dev=capacitor(gate_voltage=1.0)

    sta=  solve_bias_newton(dev)

    for noode in dev.carrier_free_nodes:
        assert sta.n.data[noode] ==0.0
        assert sta.p.data[noode] == 0.0

def test_the_gate_bias_reaches_the_silicon() :
    guses = solve_equilibrium(capacitor(gate_voltage=0.0))

    hel= solve_bias_newton(capacitor(gate_voltage=  0.0), guess = guses)
    assert hel.newton.converged,hel.newton.message

    def at_gate(gate_voltage  :  float, previous)  :
        solved= solve_bias_newton(capacitor(gate_voltage), guess =  previous)
        return solved if solved.newton.converged else None
    ram  =  continue_to( at_gate ,   start  =   0.0, target  =-   1.0,  initial  =   hel, step =   0.25)

    assert  ram.converged,   ram.message
    assert not np.allclose(hel.psi.data, ram.solution.psi.data)
