"""The coupled Newton driver, at the level of one bias point.

The convergence claims phases/PHASE-3.md is actually graded on live in
tests/convergence/test_newton_convergence.py. This file covers the wiring:
that the contacts arrive imposed rather than approached, that the limiter
does what docs/02-numerics.md prescribes, that a failure is reported rather
than swallowed, and that the state handed back is a state and not a vector.
"""


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
    """The Phase 2 device, at a small forward bias the Poisson guess reaches."""
    return  pn_diode( n_nodes  = 81, anode_voltage  =  0.2 )


def test_the_limiter_caps_the_potential_update()  :
    '''5*V_T per step, which is 5.0 scaled. docs/02-numerics.md.'''
    Delta=pack(
        np.array([40.0,-80.0,1.0]),np.zeros(3),np.zeros(3)
    )
    limtied =   limit_psi_step(  Delta ,  5.0  )

    k2 =limtied[Unknown.PSI :: UNKNOWNS_PER_NODE]
    assert np.max(np.abs(k2)) == pytest.approx(5.0)

def test_the_limiter_takes_the_density_updates_in_full() :

    """Damping n and p slows convergence without buying robustness.

    docs/05-pitfalls.md is explicit about this. The density update is six
    decades larger than the potential update in scaled units, so a single
    factor over the whole vector would be set by the densities and would
    leave psi effectively frozen.
    """
    obj2=np.array([1e6, - 2e6, 3e6])
    Delta = pack(np.array([40.0, - 80.0, 1.0]), obj2, - obj2)

    lmited   =  limit_psi_step(  Delta, 5.0 )


    np.testing.assert_array_equal(lmited[Unknown.N  ::  UNKNOWNS_PER_NODE], obj2)
    np.testing.assert_array_equal(lmited [Unknown.P  ::   UNKNOWNS_PER_NODE  ] , -   obj2  )

def test_the_limiter_preserves_the_direction_within_psi():
    """Scaled by one factor, not clipped entry by entry."""
    psi_updte  =np.array([40.0, -80.0, 1.0])
    dleta= pack(psi_updte, np.zeros(3), np.zeros(3))



    lim=limit_psi_step(dleta,5.0)[Unknown.PSI:: UNKNOWNS_PER_NODE]

    np.testing.assert_allclose(
        lim / psi_updte, np.full(3, 5.0  /  80.0), rtol = 1e-15
    )



def test_the_limiter_returns_its_argument_when_nothing_needs_capping() :
    """Identity, so that newton_solve does not count an inactive limiter."""
    deta=pack(np.array([1.0,-2.0]),np.array([9.0,9.0]),np.array([9.0,9.0]))



    assert limit_psi_step( deta ,   5.0)  is  deta

def test_returns_a_converged_device_state(diode) :
    """The happy path, at a bias the Poisson guess already reaches."""
    sta=solve_bias_newton(diode)

    assert  sta.newton  is not  None
    assert sta.newton.converged,sta.newton.message
    assert sta.psi.size==diode.mesh.n_nodes

def  test_the_contact_values_are_imposed_exactly ( diode ) :
    """A Dirichlet condition is a statement about the answer, not a target.

    Checked to the last bit rather than to a tolerance, because
    apply_dirichlet_nodes eliminates the column as well as the row and that
    is the whole reason it does.
    """
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
    """phases/PHASE-3.md, and docs/05-pitfalls.md forbids clamping to get it."""
    sttate= solve_bias_newton(diode)
    assert np.all(sttate.n.data> 0.0)
    assert np.all(sttate.p.data>0.0)



def test_a_starting_guess_is_used_rather_than_recomputed(diode) :
    """Continuation depends on this. The guess is the whole mechanism."""


    war  = solve_bias_newton(diode) ; Again = solve_bias_newton(diode, guess = war)
    assert  Again.newton  is  not  None
    assert Again.newton.iterations<war.newton.iterations


def  test_the_initial_guess_is_not_mutated( diode )  :
    """A driver that edits its guess makes continuation unrepeatable."""
    Guess  =initial_state(diode)
    lst=Guess.psi.data.copy()


    solve_bias_newton(diode , guess   =   Guess  )


    np.testing.assert_array_equal(Guess.psi.data,lst)

def test_a_failed_solve_is_reported_not_raised(diode)  :
    """A failed solve is information to inspect, matching newton_solve."""


    w=solve_bias_newton(diode,max_iterations= 1)


    assert w.newton is not None
    assert not w.newton.converged
    assert w.newton.message
def test_models_can_be_supplied(  diode ) :
    """Phase 4 swaps the mobility model in here, so it has to be a seam."""
    Models=TransportModels.for_device(diode)
    object=solve_bias_newton(diode,models =Models)

    assert object.newton is not None;  assert object.newton.converged,object.newton.message
@pytest.fixture




def  hard_case(  )   :
    """A device and guess where cold Newton diverges, so the hybrid has a job.

    1e15 doping on 41 nodes at 1.2 V. Coarse enough that the starting guess is
    a long way from the answer and lightly doped enough that the depletion
    region is a large fraction of the device. Measured bare: 30 steps, 28 of
    them against the limiter, 42 nodes with a negative density, and a final
    residual of 1.3e5.

    Not a contrived case. It is the same diode as everywhere else, one decade
    lighter and five times coarser.

    The guess is the equilibrium of the unbiased device, which is what the
    first step of any ramp starts from. It cannot be initial_state of the
    biased device: the Poisson solve at frozen quasi-Fermi levels does not
    itself converge at 1.2 V here, so there is no cold guess at that bias to
    be had. That is a separate limit and it is recorded in
    docs/07-decisions.md.

    Returns (device at 1.2 V, guess at 0 V).
    """
    Unbiased=pn_diode(Na=1e15,Nd =1e15,n_nodes =41,h_min =2e-7)
    return Unbiased.with_bias(anode =  1.2, cathode = 0.0), initial_state(Unbiased)



def test_bare_newton_diverges_on_the_hard_case(hard_case) :
    """Pins the premise the hybrid test rests on.

    Without this, a hybrid that passes proves nothing: the case has to be one
    that bare Newton actually fails. If this ever starts passing, the cold
    start got better and the hybrid case needs to be made harder or retired.
    """
    dev, w = hard_case

    staate= solve_bias_newton(dev, guess= w)

    assert  staate.newton is not  None
    assert not staate.newton.converged
    assert np.any(staate.n.data <=  0.0)  or np.any(staate.p.data<=  0.0)


def test_the_hybrid_converges_where_bare_newton_diverges(  hard_case )  :
    """docs/02-numerics.md: Gummel for 3 to 5 cycles, then switch to Newton.

    Measured on this device: two cycles are enough to turn a divergence into
    an eight step Newton solve, three into seven, five into six.
    """
    obj2,buff =hard_case
    idx2=solve_bias_hybrid(obj2,guess= buff)
    assert idx2.newton is not None
    assert idx2.newton.converged,idx2.newton.message
    assert np.all(idx2.n.data > 0.0)
    assert np.all(idx2.p.data>0.0)


def test_the_hybrid_keeps_its_quadratic_tail(hard_case) :
    '''The prelude buys the basin. It must not cost the convergence rate.'''
    type,tmp2= hard_case
    sta =  solve_bias_hybrid(  type,   guess  =  tmp2)


    assert sta.newton is not None
    assert sta.newton.residual_history[-  1] < 1e-13
    assert sta.newton.iterations < 15
def  test_the_hybrid_reports_the_prelude_it_ran( hard_case )  :
    """Log every switch, per docs/02-numerics.md.

    A solve that needed a prelude and a solve that did not are different
    events, and only one of them is a sign the guess is getting thin.
    """
    dev , guss =  hard_case
    tmp2=solve_bias_hybrid(
        dev,models=TransportModels.for_device(dev),guess= guss
    )
    assert tmp2.gummel is not  None
    assert tmp2.gummel.iterations >   0


def test_the_hybrid_agrees_with_bare_newton_where_both_converge(diode) :
    '''One set of equations, so the route to the answer cannot change it.'''
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
    """The prelude is the only difference, so switching it off removes it."""

    item2, bytes = hard_case
    tmp =solve_bias_hybrid(
        item2,guess=bytes,gummel_cycles=0,retry_cycles= 0
    )


    assert tmp.newton is not None
    assert not tmp.newton.converged


def  test_the_hybrid_retries_with_more_gummel_after_a_newton_failure(  hard_case ) :
    """The fallback half of the prescription, exercised rather than assumed.

    One prelude cycle is measured to be not enough on this device: Newton
    still diverges to 41 negative densities. The retry has to notice that and
    run more Gummel from the state before Newton touched it, never from the
    diverged one.
    """
    dev,Guess=hard_case
    sate   =   solve_bias_hybrid(dev, guess =   Guess ,  gummel_cycles  =  1,   retry_cycles = 4  )


    assert sate.newton is not None
    assert sate.newton.converged,sate.newton.message
    assert np.all(sate.n.data >0.0)

def test_the_hybrid_does_not_mutate_its_guess(diode) :
    """Continuation calls this in a loop and reuses the state it passed in."""
    gess=initial_state(diode)
    bef=  gess.psi.data.copy()

    solve_bias_hybrid(diode,
              guess  = gess)

    np.testing.assert_array_equal( gess.psi.data,
             bef  )

def  test_a_prelude_that_fails_still_hands_its_state_to_newton( )  :
    """A prelude is not a solve, so a block giving up inside it is not fatal.

    At 10 V forward on this device the Poisson block of the first Gummel cycle
    exhausts its own budget and raises. The state at that point is still a
    better guess than the one the cycle started from, and Newton is entitled
    to try it and fail in its own way. Swallowing the exception here rather
    than propagating it is what keeps continuation able to read a failure and
    halve its step.
    """
    unb=pn_diode(Na = 1e15, Nd = 1e15, n_nodes= 41, h_min =  2e-7)
    dev=unb.with_bias(anode=10.0,cathode= 0.0)


    State =solve_bias_hybrid(dev, guess =initial_state(unb))



    assert State.newton is not None
    assert not  State.newton.converged



def test_doping_dependent_mobility_lowers_the_diffusivity (diode  )  :
    """Arora at 1e16 gives about 1230 against the constant model's 1417.

    The seam works the way physics/mobility.py says it does: Dn becomes an
    array over edges and nothing in the assembly changes.
    """


    s2 =TransportModels.for_device(diode)


    zz = TransportModels.for_device(diode,mobility="arora")

    assert np.isscalar(s2.Dn) or np.ndim(s2.Dn)==0
    assert np.ndim (  zz.Dn ) ==  1

    assert np.size(zz.Dn)== diode.mesh.n_edges
    assert np.all(np.asarray(zz.Dn)<s2.Dn)
def test_doping_dependent_mobility_still_converges(diode) :

    """A per edge diffusivity has to solve exactly as a scalar one does."""
    sta=  solve_bias_newton(diode, models =TransportModels.for_device(diode, mobility= "arora"))

    assert sta.newton is not None
    assert sta.newton.converged, sta.newton.message
    assert np.all(sta.n.data > 0.0)


def test_lower_mobility_gives_less_current(  diode  )  :
    """The whole reason the model matters, and a sign check on the wiring.

    Arora reduces the mobility at 1e16, so it has to reduce the current. If it
    raised it, the model would be inverted somewhere between the doping and
    the diffusivity and every quantitative result downstream would be wrong in
    a way no convergence check would notice.
    """
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
    """Phase 3 scope item 6. SRH and Auger act in parallel, so they add."""

    myvar = TransportModels.for_device(diode, auger  =  True)

    assert isinstance(myvar.recombination, SumOfRecombination)
    assert len(myvar.recombination.models) ==2
def  test_auger_raises_the_recombination_rate(diode) :
    """Adding a parallel path can only add rate, never remove it."""
    pla =TransportModels.for_device(diode)
    myvar =TransportModels.for_device(diode,auger=True)

    n  =np.full(diode.mesh.n_nodes, 1e6)
    p=np.full(diode.mesh.n_nodes,1e6)
    assert np.all(
        np.asarray(myvar.recombination.rate(n, p))
        >  np.asarray(pla.recombination.rate(n, p))
    )


def test_auger_overtakes_srh_as_the_square_of_the_density(diode):

    """Cubic against linear, which is why Auger is a high injection mechanism.

    The physical content of the model is not that Auger is large or small, it
    is how fast it takes over. SRH at high injection goes as n and Auger goes
    as n^3, so their ratio has to go as n^2: exactly a hundredfold per decade
    of density. Measured across nine decades it is 99.1, 99.9, then 100.0 to
    four figures the whole way, and the crossover where Auger overtakes SRH
    lands at about 6e17 cm^-3.

    A coefficient wrong by orders of magnitude would move the crossover and
    leave every other test here passing. A coefficient with the wrong power of
    the density scale would break this one.
    """
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

    """The exact Auger tangent is not sign definite, so this is worth running."""
    sta = solve_bias_newton(
        diode, models  = TransportModels.for_device(diode, auger = True)
    )



    assert  sta.newton is  not  None; assert sta.newton.converged, sta.newton.message
    assert  np.all( sta.n.data >  0.0)
    assert np.all(sta.p.data >  0.0)


def test_both_models_together_converge(diode):

    """Scope items 6 and 7 at once, which is how Phase 4 will run."""
    format  =  TransportModels.for_device( diode , mobility =  'arora' ,   auger  = True)
    State= solve_bias_newton(diode, models= format)

    assert State.newton is not None;  assert  State.newton.converged ,   State.newton.message


def test_an_unknown_mobility_model_is_rejected(diode):
    """A typo must not silently fall back to the constant model."""
    with pytest.raises(ValueError, match ="mobility"):

        TransportModels.for_device(  diode , mobility   =   'arorra'  )
def capacitor(gate_voltage:float=1.0):
    """A small MOS capacitor. Coarse on purpose: nothing here is a physics
    claim about the capacitor, only about what the coupled path does with a
    contact that pins psi alone."""
    return mos_cap(gate_voltage=gate_voltage,n_silicon =41,n_oxide=3)

def test_the_coupled_solve_reproduces_the_capacitor_at_equilibrium() :
    """The gating test for the gate, and the sharpest form available.

    No current flows through an ideal insulator, so the semiconductor of a MOS
    capacitor stays in equilibrium with its body contact at every gate bias.
    Equilibrium is an exact fixed point of the coupled system: the continuity
    residuals vanish because there is no flux and no net recombination, and
    what is left is the Poisson equation the equilibrium path already solved.

    So the coupled solve must not move off the equilibrium answer. If the gate
    row were left out, mispinned, or pinned at the ohmic target instead of the
    work function one, the coupled residual there would not vanish and Newton
    would walk away from it. One check covering the gate boundary condition,
    the carrier free pinning and the charge volume together.
    """
    Device =capacitor(gate_voltage =1.0)
    Reference   =  solve_equilibrium(Device )

    sate =solve_bias_newton(Device)

    assert sate.newton.converged, sate.newton.message
    np.testing.assert_allclose(sate.psi.data, Reference.psi.data, rtol =1e-9, atol=1e-12)
    np.testing.assert_allclose(sate.n.data, Reference.n.data, rtol=  1e-8, atol =1e-12)
    np.testing.assert_allclose(sate.p.data, Reference.p.data, rtol = 1e-8, atol=  1e-12)



def  test_the_gate_potential_is_imposed_exactly(  )  :
    """A gate pins psi at its own work function, not at the doping under it.

    ohmic_psi_scaled would read the net doping at a node in the middle of an
    insulator, which is zero, and return the applied bias alone. That is a
    plausible looking number and it is wrong by Phi_MS, which slides the whole
    curve sideways with every regime still looking correct.
    """
    Device=capacitor(gate_voltage =1.0);  gat=next(c for c in Device.contacts if c.name == GATE)
    sta  = solve_bias_newton(Device)

    expectted  = gate_psi_scaled(gat.voltage  / Device.scale.psi_0, gat.work_function)
    for round in gat.nodes:
        assert sta.psi.data[round]  ==  pytest.approx(expectted, rel =1e-14)


def test_the_oxide_holds_no_carriers_on_the_coupled_path() :
    """Including the gate nodes, which are metal sitting on the insulator.

    Their continuity rows have no flux and no volume, so they read 0 = 0 and
    the matrix is singular unless something pins them. carrier_free_nodes
    already does, and a gate must not fight it for those rows.
    """

    dev=capacitor(gate_voltage=1.0)

    sta=  solve_bias_newton(dev)

    for noode in dev.carrier_free_nodes:
        assert sta.n.data[noode] ==0.0
        assert sta.p.data[noode] == 0.0

def test_the_gate_bias_reaches_the_silicon() :
    """Guards the case where the gate is accepted and then ignored.

    A gate quietly left out of the coupled assembly gives a floating oxide,
    which converges perfectly happily and reports a surface that does not care
    what the gate is doing.

    Both solves start from the same guess, which is what makes this a
    statement about the assembly. Started from their own guesses the two would
    differ whatever the assembly did, because initial_state is the equilibrium
    Poisson solve and that path has applied the gate correctly since Phase 4.
    The mutation it exists to catch is a gate that is accepted and then
    ignored: keep the gate row but build its target from 0.0 instead of the
    applied bias. Both solves then converge and hand back the same psi to the
    last bit, and the last assertion is what fails. Deleting the gate row
    outright is caught too, earlier and more bluntly, because a floating oxide
    leaves the first solve unable to converge at all.

    The ramp to -1 V is `continue_to` and not one Newton solve. One solve used
    to reach it from this guess, and that was luck rather than a property: the
    basin of the undamped solve on this capacitor is ragged, -0.25 V misses it
    while -1.0 V lands, and -1.0 V lands only by spending 9 of its 16 steps on
    the limiter. CI on Linux drew the other side of that coin and reported a
    residual of 7.095e+10. Every ramp step is still a coupled solve carrying
    the gate, so the ramp costs the test none of that.
    """
    guses = solve_equilibrium(capacitor(gate_voltage=0.0))

    hel= solve_bias_newton(capacitor(gate_voltage=  0.0), guess = guses)
    assert hel.newton.converged,hel.newton.message

    def at_gate(gate_voltage  :  float, previous)  :
        solved= solve_bias_newton(capacitor(gate_voltage), guess =  previous)
        return solved if solved.newton.converged else None
    ram  =  continue_to( at_gate ,   start  =   0.0, target  =-   1.0,  initial  =   hel, step =   0.25)

    assert  ram.converged,   ram.message
    assert not np.allclose(hel.psi.data, ram.solution.psi.data)
