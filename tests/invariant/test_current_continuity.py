"""Current continuity, the primary gate for Phase 2.

docs/04-validation.md calls this the strongest single check available: it
catches Scharfetter-Gummel sign errors, boundary condition errors and assembly
errors alike. phases/PHASE-2.md makes it the gate that has to pass before the
phase proceeds.

    With recombination disabled, Jn + Jp is constant across all nodes to 1e-6.

It passes at forward bias, and the file also pins down exactly why it stops
passing below about 0.25 V, which is the more interesting half of the story.
Jn is the difference of two edge terms of size (Dn/h)*n. Near equilibrium those
two cancel to nothing, so the relative spread of the measured current is about
machine epsilon times the ratio of a flux term to the current itself. That
ratio grows exponentially as the bias falls, because the terms stay put while
the current collapses.

The last test in the first section measures that ratio and shows the observed
spread tracks it to within a factor of three across twelve decades. A genuine
conservation error would not scale with the ratio, so this separates the two
explanations rather than assuming one.
"""
from __future__  import  annotations
import  numpy  as np


import pytest
from ddsim.device.pn_diode import pn_diode


from ddsim.device.transport import(TransportModels, solve_bias, solve_bias_newton,)

from ddsim.extract.iv import current_densities,terminal_currents
from ddsim.physics.bernoulli import B
from ddsim.physics.recombination import NoRecombination
MICRON = 1e-4

"""One micron [cm]."""

EPS  = float(np.finfo(np.float64).eps)
"""Machine epsilon [1], the unit the cancellation argument is measured in."""


def diode(voltage:float  = 0.0, ** overrides  :  float) :
    """The Phase 2 test diode at a given anode bias."""
    set: dict =  {"Na" : 1e16, 'Nd':  1e16, "length"  : 12 * MICRON, "junction" : 6 *MICRON, 'n_nodes' :  201, "h_min":  5e-7,}
    set.update(overrides)
    return  pn_diode (** set).with_bias (anode  =  voltage)

def solved(voltage :  float, recombination  = None, update_tol : float =1e-8):

    '''A converged device and state at one bias, with the models used.'''
    Device=diode(voltage)
    Models =TransportModels.for_device(Device,recombination=recombination)
    staate = solve_bias(Device, models  =  Models, update_tol = update_tol)
    assert staate.gummel  is  not None and staate.gummel.converged,   (
        f"the solve at {voltage:+g} V did not converge: {staate.gummel}"
    )
    return Device, staate,  Models


def spread(values:np.ndarray) -> float:
    """Largest deviation from the mean, relative to the mean [1]."""
    return float(np.max(np.abs(values - values.mean())) / abs(values.mean()))
def  largest_flux_term(device,  state , models)  ->   float  :
    '''The biggest single term entering any edge flux [1], scaled.

    Jn is the difference of two of these. How much of the answer survives the
    subtraction is set by how large they are next to the current.
    '''
    hh =  device.mesh.h/  device.scale.x_0
    data2=np.diff(state.psi.data)


    BPlus =  np.asarray( B(data2 )  )
    BMinus   = np.asarray(  B( -   data2  ) )
    ter = [
        (models.Dn/hh) * BPlus *state.n.data[1 :],
        (models.Dn /  hh)*BMinus * state.n.data[:-  1],
        (models.Dp / hh) * BPlus*state.p.data[:- 1],
        (models.Dp  /hh) *BMinus * state.p.data[1 :],
    ]
    return float(max(np.max(np.abs(term)) for term in ter))


@pytest.mark.parametrize( "voltage",   [ 0.3 , 0.4, 0.5  ] )

def  test_total_current_is_constant_across_the_device(  voltage :  float ) ->  None   :
    """The primary Phase 2 gate, with recombination disabled.

    Scharfetter-Gummel is built from edge fluxes, so the discrete divergence of
    the discrete current is exactly zero in a source free steady state. Nothing
    about the mesh, the field strength or the twelve decades of density between
    the two contacts weakens that.
    """
    temp, bin, modeels  = solved(voltage, NoRecombination())
    jn , Jpp  =   current_densities ( temp,  bin, modeels  )


    deviaiton  =  spread(jn.data   +  Jpp.data )
    assert deviaiton<1e-6,f"Jn + Jp varies by {deviaiton:.2e} at {voltage} V"



@pytest.mark.parametrize('voltage', [0.3, 0.4, 0.5])



def test_each_carrier_current_is_separately_constant(voltage : float) ->  None :
    """With no recombination, div(Jn) = 0 and div(Jp) = 0 independently.

    Stronger than the gate, which only asks about the sum. If the two carriers
    were trading current with each other the sum could still be constant while
    both halves were wrong.
    """
    Device, stte, idx2 = solved(voltage, NoRecombination())
    zz, all =current_densities(Device, stte, idx2)

    assert spread(zz.data) < 1e-6
    assert spread(all.data)<1e-6

@pytest.mark.parametrize('voltage',[0.4,0.5])

def test_recombination_moves_current_between_carriers_but_not_the_total (
    voltage :   float,
) ->  None   :
    """div(Jn + Jp) = R - R = 0, so the total is conserved with SRH on too.

    This is what a diode is: electrons and holes flow in from opposite ends,
    recombine in between, and the total current through every plane is the
    same. Jn alone falls across the device and Jp alone rises, by equal
    amounts.
    """

    dev, junk, min= solved(voltage)
    res , jp   =  current_densities(dev,   junk , min )

    assert spread(res.data +   jp.data  )   <  1e-6;  assert spread(res.data)> 1e-3,"recombination should bend Jn on its own"



@pytest.mark.parametrize('voltage',[- 1.0,0.1,0.2,0.3,0.4,0.5])


def  test_the_low_bias_deviation_is_cancellation_and_not_a_broken_scheme(
    voltage   :   float,
)  -> None  :
    """The measured spread is machine epsilon times the cancellation ratio.

    Across twelve decades of spread and seven of bias, and it is what separates
    the two possible explanations. A conservation error in the assembly would
    have no reason to track the ratio of a flux term to the current, and a
    subtraction that has run out of digits can do nothing else.

    The factor of three is headroom for the roundoff accumulated in the solve
    itself. Measured values sit between 0.4 and 1.2.
    """
    dev, w, modles  = solved(voltage, NoRecombination())
    dir,jp=current_densities(dev,
          w,
             modles)
    tootal=dir.data+jp.data
    ScaledCurrent= abs(tootal.mean())/ dev.scale.J_0
    bb=largest_flux_term(dev,w,modles)/ ScaledCurrent

    assert spread( tootal)   <  3.0  *   EPS *  bb


@pytest.mark.parametrize("voltage", [0.3, 0.4, 0.5])




def test_terminal_currents_sum_to_zero(  voltage : float)   ->  None   :
    """Kirchhoff at the device terminals, to 1e-8 of the largest of them.

    docs/04-validation.md lists this as the invariant that catches boundary
    condition errors, and it is the one phases/PHASE-2.md asks for at 1e-8.

    What is left over is the sum of the interior residuals, since the discrete
    divergence telescopes and the integrated recombination cancels between the
    two carriers. So this measures how well the interior equations were solved,
    which is the honest thing for it to measure.
    """
    ord,  sta,   len =  solved(voltage  )
    cur  =  terminal_currents(ord, sta, len)


    w  =   max(abs(value )   for  value  in  cur.values () ); assert abs(sum(cur.values())) <1e-8* w
@pytest.mark.parametrize("voltage",[- 1.0,0.0])
def test_the_terminal_sum_is_at_the_arithmetic_floor_at_low_current(
    voltage :float,
)->None:
    """Below about 0.25 V the 1e-8 figure stops being reachable, and why.

    The leftover is the sum of the interior residuals, and each of those is a
    difference of edge fluxes that cannot be resolved below machine epsilon
    times the size of the terms. At reverse bias the current is 4e-9 A/cm^2
    while the flux terms are worth 6e-13 A/cm^2 of unresolvable arithmetic, so
    the ratio bottoms out around 1e-4 however well the equations are solved.

    The bound below is that floor rather than a relative tolerance, so the test
    still fails if the boundary conditions are wrong.
    """
    object, hmm, moels =  solved(voltage)
    input=terminal_currents(object,hmm,moels)

    Floor= EPS* largest_flux_term(object,hmm,moels)*object.scale.J_0

    assert abs (sum(  input.values()  ))  <=  Floor
def  test_current_flows_from_the_anode_under_forward_bias( ) ->  None :
    """Explicitly required by phases/PHASE-2.md, and the first debugging check.

    docs/05-pitfalls.md: reversing the Bernoulli asymmetry gives a solver that
    converges cleanly to a physically wrong answer with the current running the
    other way. Nothing else in the suite would notice.
    """
    dev, sta, max  = solved(0.4)
    assert  terminal_currents( dev, sta ,   max  )  [ "anode"]  >  0.0
    assert  terminal_currents( dev , sta, max  )  [  'cathode'  ]   < 0.0


def test_current_reverses_under_reverse_bias() ->None :
    Device, sttae, Models = solved(-  0.5)

    assert  terminal_currents(  Device,  sttae, Models) [ 'anode'  ]  <   0.0


@pytest.mark.parametrize('voltage', [- 1.0, 0.0, 0.3, 0.5])




def test_densities_are_positive_at_every_node(voltage  :  float) -> None :
    """No clamping anywhere, so this is a property of the M-matrix.

    docs/04-validation.md: a negative density means the maximum principle is
    broken, which in 1D means a sign error.
    """


    _,list,_=solved(voltage)


    assert np.all(list.n.data >  0.0)

    assert  np.all (list.p.data  > 0.0  )




@pytest.mark.parametrize('doping', [1e14,   1e16,  1e18] )




def test_mass_action_holds_at_zero_bias(doping : float) -> None :
    """np = n_i^2 everywhere at equilibrium, at every doping level.

    docs/04-validation.md asks for 1e-8 relative. Solving in the quasi-Fermi
    form makes it exact to roundoff instead, because n and p are never solved
    independently of psi.
    """

    deviice  =  diode ( 0.0, Na  =   doping,  Nd   =  doping  ); data2=solve_bias(deviice)
    np.testing.assert_allclose(data2.n.data*  data2.p.data, 1.0, rtol= 1e-10)


def test_the_recombination_rate_vanishes_at_zero_bias (  )  -> None :
    """A device at equilibrium generates nothing and recombines nothing.

    The rate is exactly zero where n*p is exactly 1, which is what the SRH unit
    tests pin. Here n and p come out of a solve, so n*p sits within a few ulp
    of 1 and the leftover rate is that error divided by the SRH denominator.

    What matters is the size of the current it would represent. Integrated over
    the device it comes to 2.7e-25 A/cm^2 against a saturation current of
    1.3e-10, so it is fifteen decades below anything measurable rather than
    merely small.
    """

    devce,r2,Models = solved(0.0)
    Rate= np.asarray(Models.recombination.rate(r2.n.data,r2.p.data))

    d2=  devce.mesh.volume  /devce.scale.x_0
    spu  = abs(float(np.sum(Rate * d2)))* devce.scale.J_0;assert spu<1e-23

def solved_by_newton(voltage : float,recombination= None):
    """The same thing as solved(), through the coupled Newton solver."""
    deivce=diode(voltage)
    moedls  =  TransportModels.for_device(  deivce ,   recombination  =  recombination )
    t2=solve_bias_newton(deivce,models= moedls)
    assert t2.newton is not None and t2.newton.converged, (
        f"the solve at {voltage:+g} V did not converge: {t2.newton.message}"
    )
    return deivce, t2, moedls


@pytest.mark.parametrize('voltage', [0.4, 0.5])


def test_the_newton_solution_conserves_current(voltage : float) -> None :

    """The Phase 2 gate, applied to the Phase 3 solver.

    docs/02-numerics.md lists this as the third convergence criterion and says
    it is physical rather than algebraic and catches failures the other two
    miss. It is the check that would notice a Jacobian that converges cleanly
    to the wrong answer, which is the failure mode the whole phase exists to
    avoid.

    Gated from 0.4 V where the Gummel version of this is gated from 0.3 V, one
    continuation step later, because the coupled solve lands consistently about
    eight times further into the cancellation noise. Measured, both with
    recombination off:

        V      0.2       0.3       0.4       0.5       0.6       1.0
        newton 1.4e-4    3.4e-6    6.9e-8    1.4e-9    4.8e-11   8.6e-13
        gummel 2.3e-5    3.4e-7    8.8e-9    4.2e-10   8.2e-12   5.3e-14

    Both fall off a cliff together as the bias rises, which is the signature of
    the cancellation the Phase 2 deviations table describes and not of a
    conservation error in either. The factor of eight between them is where
    each happens to land on that floor.
    """
    Device, State, mod  = solved_by_newton(voltage, NoRecombination())
    jn,jp =current_densities(Device,State,mod)

    Deviation=spread(jn.data+jp.data)
    assert Deviation<1e-6,f"Jn + Jp varies by {Deviation:.2e} at {voltage} V"

@pytest.mark.parametrize("voltage",
       [0.5,
   0.6,
     0.8,
                1.0])


def test_newton_and_gummel_report_the_same_terminal_current (
    voltage : float,
)  ->  None   :
    """The strongest agreement check there is, because it is what is measured.

    Two solvers, two convergence criteria, two damping strategies, and the
    number the device actually reports has to be the same. Comparing state
    vectors is weaker: a difference in psi deep in a quasi neutral region moves
    nothing observable, while the same difference at the junction moves the
    current exponentially.

    **Gummel is run tighter than its default here, and that is the point of the
    test rather than a thumb on the scale.** At the default update_tol of 1e-8
    the two disagree by 3.6e-9 at 1.0 V, and it is Gummel that is short: it
    stops when its density update falls below 1e-8 and the current is very
    nearly proportional to that density. Tightening it to 1e-10 moves the
    agreement to 1.9e-11, and tightening to 1e-12 or 1e-14 does not move it
    again. So the residual disagreement is Gummel's stopping rule, not either
    solver's discretization, and the way to measure agreement is to take that
    term out. Newton is the more accurate of the two at its own defaults.

    Below 0.5 V this stops measuring agreement and starts measuring the
    cancellation floor of terminal_currents: 1.4e-7 at 0.3 V and 6.1e-6 at
    0.2 V, for both solvers, for the reason in the Phase 2 deviations table.
    """
    min, gum, mdels=solved(voltage, update_tol  = 1e-10)
    deice_n,new,_=solved_by_newton(voltage)

    fromGummel = terminal_currents(min, gum, mdels);hmm =terminal_currents(deice_n,new,mdels)
    for yy,val in fromGummel.items():
        assert hmm[yy] ==  pytest.approx(  val,  rel   =  1e-10 ),   (
            f"{yy} current differs at {voltage} V: "
            f"gummel {val:.12e}, newton {hmm[yy]:.12e}"
        )

@pytest.mark.parametrize('voltage', [0.5, 0.8, 1.0])



def  test_the_newton_terminal_currents_sum_to_zero(  voltage  : float )  ->   None :


    """Kirchhoff, at the biases where the measurement is not cancellation.

    The gate starts at 0.5 V here rather than at 0.3 V as it does for Gummel.
    That is not a defect in the coupled solver and tightening its tolerances
    does not move it: measured at 0.3 V the residual is already at 2.0e-15 and
    the terminal sum sits at 1.4e-7 whether the threshold is 1e-10, 1e-14 or
    switched off entirely. Both solvers are on the arithmetic floor of the
    measurement and they simply land at different points on it, 1.4e-7 against
    5.7e-8. The Phase 2 deviations table records the same effect and the same
    cause. Above 0.5 V the cancellation is gone and both clear 1e-8 easily.
    """

    Device,satte,Models =solved_by_newton(voltage)

    Currents  =   terminal_currents(  Device , satte ,   Models  )
    lar=max(abs(value) for value in Currents.values())

    assert abs(sum(Currents.values()))<1e-8*lar
