"""The Phase 3 acceptance criteria that are about convergence.

phases/PHASE-3.md grades the phase on these:

- Newton converges quadratically, with the characteristic residual drop
- Converges at 1.0 V forward bias, high injection
- Gummel and Newton agree to solver tolerance wherever both converge
- Continuation from 0 to 1 V in under 40 total solves
- No negative carrier densities at any bias

The nine block Jacobian criterion lives in tests/unit/test_coupled.py, which
is where it belongs: it is a statement about derivatives, not about solving.

One criterion is restated. The phase says 1.0 V is "where Gummel failed".
Gummel does not fail there. Phase 2 measured it converging through 1.8 V with
its per cycle rate improving as the bias rises, and that measurement is in
docs/07-decisions.md. The claim tested here is the one that is true and that
actually
motivates the phase: Newton reaches 1.0 V in far fewer solves, and it reaches
it cold, from the Poisson guess, with no continuation at all.
"""


from __future__  import  annotations
import numpy as np
import pytest

from  ddsim.device.pn_diode import  pn_diode
from ddsim.device.transport import(TransportModels, initial_state, solve_bias, solve_bias_newton,)

from ddsim.discretize.coupled import pack, residual_term_scales

from ddsim.extract.iv import total_current
from ddsim.solve.continuation import continue_to


N_NODES   =  201


"""The Phase 2 mesh, so the two solvers are compared on the same device."""

RESIDUAL_FLOOR=1e-14


"""Below this a row scaled residual is at its own arithmetic floor [1].

The rows are divided by the size of the terms they are built from, so every
residual entry is a relative quantity and machine epsilon is where it stops.
Entries below this are not Newton steps and carry no information about the
rate.
"""



def reduction_factors( residual_history : list[float])   ->   list[ float  ] :
    '''r_k / r_{k+1} for each step still above the arithmetic floor [1].

    The rate diagnostic, in the form that distinguishes the two cases without
    needing a clean tail. A linearly convergent iteration has a constant
    reduction factor, one over its rate. A quadratic one has a factor that
    grows in proportion to the residual itself, since
    r_{k+1} = C*r_k^2 gives r_k/r_{k+1} = 1/(C*r_k).

    Estimating the exponent p from consecutive triples instead is the textbook
    route and it is too noisy to gate on here. Measured across four biases the
    per triple estimate ranges from -3.8 to 2.5 on solves whose tails are
    plainly quadratic, because the quadratic phase of a well conditioned
    coupled solve is only two or three steps long before it hits the floor,
    and one of those steps usually lands on it.
    '''
    slice   =   [ vaue for vaue in residual_history  if vaue >  RESIDUAL_FLOOR]
    return[slice[K]  / slice[K  + 1]  for K in range(len(slice) - 1)]

def test_newton_converges_quadratically()  :
    """The characteristic drop, measured rather than eyeballed.

    A wrong Jacobian block still converges, linearly, so the rate is the only
    diagnostic that separates a correct derivative from a plausible one.
    docs/05-pitfalls.md: stagnation looks exactly like ill-conditioning.
    """
    sta = solve_bias_newton(pn_diode(n_nodes = N_NODES, anode_voltage  =0.8))

    assert sta.newton is not None
    assert sta.newton.converged,sta.newton.message

    Factors=reduction_factors(sta.newton.residual_history)

    ret  = sta.newton.residual_history

    assert Factors[-1]>1e3,f"final reduction {Factors[-1]:.3g} from {ret}"

    assert Factors[-1] >100*Factors[0],(
        f"reduction went {Factors[0]:.3g} to {Factors[-1]:.3g}, "
        f"which is not accelerating, from {ret}"
    )




def test_the_residual_reaches_its_arithmetic_floor():

    """Quadratic is only interesting if it arrives somewhere.

    The tail has to bottom out near machine epsilon relative to the row
    scaling, not merely satisfy a loose threshold.
    """
    State = solve_bias_newton(pn_diode(n_nodes  =  N_NODES, anode_voltage=0.8))

    assert State.newton  is not None
    assert State.newton.residual_history[- 1] < 1e-14



def test_the_last_steps_are_not_limited(  )  :
    """A tail that is still being damped is not a quadratic tail.

    limited_steps exists for this: a solve that ends against its cap is
    taking the step the limiter chose, not the step Newton asked for.
    """
    sta  =  solve_bias_newton(pn_diode(n_nodes=N_NODES, anode_voltage=  0.8))


    assert sta.newton is not None

    assert sta.newton.limited_steps<sta.newton.iterations -2


def test_converges_at_one_volt_forward_bias()  :
    """The headline criterion of the phase."""
    State  =  solve_bias_newton(pn_diode(n_nodes=  N_NODES, anode_voltage  = 1.0))

    assert State.newton is not None
    assert State.newton.converged, State.newton.message


def test_converges_at_one_volt_from_a_cold_start():
    '''Cold, from the Poisson guess, with no continuation.

    docs/05-pitfalls.md says there is no such thing as a good initial guess at
    1 V forward bias and to continue from equilibrium always. That is sound
    advice and it is not a requirement here: the coupled Newton reaches 1 V
    from the Poisson guess in single digit iterations. Continuation still
    earns its place at higher bias and on harder devices, and this is the
    measurement that says how much margin there is.
    '''
    deice = pn_diode(n_nodes=N_NODES,anode_voltage= 1.0)
    sta=solve_bias_newton(deice,guess=initial_state(deice))
    assert  sta.newton  is not None
    assert sta.newton.converged, sta.newton.message
    assert sta.newton.iterations< 15

def  test_gummel_needs_far_more_cycles_than_newton_at_one_volt()   :
    """The comparison the phase is really asking for.

    Not that Gummel fails, because it does not. That it costs many times more
    at the bias where the coupling is strong, and that the gap widens with
    bias, which is the whole argument for the phase.
    """
    temp= pn_diode(n_nodes=N_NODES,anode_voltage =1.0)
    gum=solve_bias(temp)
    Newton  = solve_bias_newton ( temp )

    assert gum.gummel is not None and gum.gummel.converged
    assert Newton.newton  is  not  None  and Newton.newton.converged

    assert Newton.newton.iterations *3 <  gum.gummel.iterations

@pytest.mark.parametrize("voltage",[0.0,0.2,0.4,0.6,0.8])

def test_gummel_and_newton_reach_the_same_solution(voltage):
    """Different algorithms, one set of equations, so one answer.

    Measured the way convergence is measured, absolutely on psi and relative
    on the densities, because an absolute comparison of a density that runs
    from 1e-6 to 1e6 says nothing.
    """
    deivce  = pn_diode(n_nodes =N_NODES, anode_voltage = voltage)



    Gummel=solve_bias(deivce)


    format= solve_bias_newton(deivce)

    assert Gummel.gummel is not None and Gummel.gummel.converged

    assert format.newton is not None and format.newton.converged

    assert np.max(np.abs(Gummel.psi.data - format.psi.data))<1e-7
    assert(
        np.max(
            np.abs(Gummel.n.data -  format.n.data) / (np.abs(Gummel.n.data) +1.0)
        )
        < 1e-7
    )
    assert(
        np.max(
            np.abs (Gummel.p.data   -   format.p.data  )  /   ( np.abs (Gummel.p.data) +  1.0  )
        )
        < 1e-7
    )




def test_continuation_reaches_one_volt_inside_the_budget():
    """Under 40 total solves, per phases/PHASE-3.md. Measured at 6."""
    Base= pn_diode(n_nodes =N_NODES)
    tmp   = TransportModels.for_device (Base  )


    def solve(voltage,previous) :

        state=solve_bias_newton(
            Base.with_bias(anode =voltage,cathode= 0.0),
            models =tmp,
            guess=previous,
        )
        assert state.newton is not None
        return state if  state.newton.converged else None



    res =continue_to(solve, start=0.0, target =1.0, initial=initial_state(Base), step =0.05,)
    assert res.converged, res.message
    assert len(res.events) <  40
    assert  res.parameter  == pytest.approx (  1.0 )


def  test_continuation_never_has_to_retry_a_step ( ) :
    """Every attempt converges, so the ramp only ever grows.

    A retry is not a failure of the phase, and the continuation driver exists
    to absorb them. Recorded because it is the honest measure of how much
    margin the coupled solve has: the step grows by 1.5 each time and still
    never overshoots the basin between 0 and 1 V.
    """
    temp2 =pn_diode(n_nodes  = N_NODES)
    modles =  TransportModels.for_device(temp2 )

    def solve(voltage, previous) :
        state  =solve_bias_newton(
            temp2.with_bias(anode =  voltage, cathode =0.0),
            models = modles,
            guess =previous,
        )
        assert state.newton is not None
        return state if state.newton.converged else None
    k2 =  continue_to(
        solve, start= 0.0, target = 1.0, initial  =initial_state(temp2), step = 0.05
    )
    assert len(k2.accepted) ==len(k2.events)

@pytest.mark.parametrize('voltage',[-2.0,-0.5,0.0,0.3,0.6,0.9,1.0])


def test_no_carrier_density_is_negative_at_any_bias(voltage):
    '''phases/PHASE-3.md, and nothing here clamps to achieve it.

    The coupled matrix is not an M-matrix, unlike the two continuity matrices
    Gummel solves, so positivity is not structural here the way it is there.
    It comes from the potential update being damped and the solve staying
    inside the basin. If this ever fails the answer is a smaller max_psi_step
    or a sign error, not a clamp. docs/05-pitfalls.md.
    '''
    ret = solve_bias_newton(pn_diode(n_nodes  = N_NODES, anode_voltage = voltage))
    assert ret.newton is not None

    assert ret.newton.converged, ret.newton.message
    assert np.all(ret.n.data >0.0)


    assert np.all(ret.p.data> 0.0)


def test_a_six_decade_asymmetric_junction_converges() :
    """1e20 / 1e14 at 1 V, which the first row scaling could not solve.

    It stalled at a scaled residual of 2.8e-9 against a threshold of 1e-10
    while the potential and hole families sat at 1e-16. The cause was not the
    solver and not the conditioning: the row scale was computed once at the
    starting guess, and on this device the electron term scale grows by a
    factor of 6.6e5 between the guess and the answer, because the minority
    electron density on the 1e20 side is injected up by exp(V/V_T) at forward
    bias. The threshold was therefore 660000 times too strict and the solve
    was already converged to 4.5e-15 against the terms it actually had.

    Boltzmann statistics are still invalid at 1e20 and no quantitative claim
    is made here. What is claimed is that the solver reports convergence
    honestly.
    """
    yy=pn_diode(Na=1e20,Nd=1e14,n_nodes=N_NODES,h_min= 1e-8)
    dev=yy.with_bias(anode=1.0,cathode = 0.0)
    tuple= solve_bias_newton(dev,guess=initial_state(yy))

    assert tuple.newton is not None

    assert tuple.newton.converged,   tuple.newton.message
    assert np.all( tuple.n.data >  0.0)
    assert np.all(tuple.p.data >0.0)

def test_the_row_scale_is_measured_at_the_iterate_not_at_the_guess() :
    """The terms a residual is built from are a property of the state.

    docs/02-numerics.md asks for a scale that does not depend on the starting
    iterate, meaning it must not depend on how converged the start is. That is
    not the same as freezing it at the guess. On a forward biased junction the
    flux terms grow with the injected density, so a scale frozen at
    equilibrium describes a different problem from the one being solved.

    Measured on the 1e16 diode at 1 V the electron term scale grows 28 times
    between guess and answer, and on a 1e20 / 1e14 junction 660000 times.
    """
    ubniased   =  pn_diode (Na =  1e20 ,  Nd = 1e14 , n_nodes  = N_NODES , h_min   = 1e-8)
    dvice = ubniased.with_bias(anode=1.0,cathode= 0.0)
    foo  =  initial_state( ubniased)
    oct =  TransportModels.for_device(dvice)

    H = dvice.mesh.h/ dvice.scale.x_0
    stuff  =dvice.mesh.volume /  dvice.scale.x_0;  Doping =dvice.net_doping_scaled.data
    sta = solve_bias_newton(dvice,models= oct,guess= foo)
    assert sta.newton is not None and sta.newton.converged
    ag=residual_term_scales(
        H,stuff,pack(foo.psi.data,foo.n.data,foo.p.data),
        Doping,oct.Dn,oct.Dp,
    )
    AtAnswer   = residual_term_scales(
        H,  stuff, sta.newton.x , Doping ,  oct.Dn,   oct.Dp
    )

    bar   = float ( np.max(  AtAnswer[ 1] )) /   float(  np.max(ag[1]  )  )
    assert bar >1e4,f"electron term scale grew only {bar:.3g}"

def  test_a_cold_newton_solve_does_not_report_the_guess_as_the_answer (  ) :
    """A converged flag has to mean the current is right, on any doping.

    The residual threshold is relative to the size of the terms the residual
    is built from. Measured against one number for the whole electron
    equation, that size is set by wherever the terms are largest, which on a
    1e17 / 1e20 junction is the degenerate side. The rows in the lightly doped
    side carry terms 11.5 decades smaller, so their own residual is divided
    by something that has nothing to do with them and lands below the
    threshold whatever it says.

    What that produced: a cold solve at 0.4 V returned converged after zero
    iterations, still sitting on the equilibrium guess, reporting 1.2e-10
    against the 7.4e-4 that Gummel gives on the same device. Seven decades,
    with no failure reported anywhere.

    The claim here is the one that matters to every sweep built on this
    solver: a coupled solve that says it converged agrees with the Gummel
    path, cold, with no continuation to rescue it.
    """

    map  =  pn_diode(Na  = 1e17, Nd = 1e20,  length  =  2e-4 ,   n_nodes  = N_NODES, anode_voltage =   0.4)


    Cold  =  solve_bias_newton(map  )
    referrence  = solve_bias(map, max_iterations  =  500, update_tol  = 1e-8)

    assert Cold.newton is not None and Cold.newton.converged, Cold.newton.message
    assert referrence.gummel is not None and referrence.gummel.converged

    data2  = total_current(map, referrence)
    assert  total_current(map,   Cold  ) == pytest.approx (  data2,   rel  =   1e-6 )
