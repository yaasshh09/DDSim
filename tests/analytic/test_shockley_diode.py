"""The Shockley diode, Tier 2 of docs/04-validation.md.

A full solve compared against closed form device physics. Nothing in the solver
knows any of the formulas in this file.

    I = I_s * (exp(V / (n V_T)) - 1)

Two things have to come out right, and they are checked separately because they
test different parts of the physics.

**The saturation current** is set by minority carrier diffusion into the two
quasi-neutral regions, so it tests the continuity equations, the contact
boundary conditions and the lifetimes together. This diode is short based: the
hole diffusion length is 55 um against a 6 um n side, so almost every injected
carrier reaches the contact rather than recombining on the way, and the
coth(W/L) factor in the general expression matters by a factor of nine.

**The ideality factor** is set by which mechanism dominates, and it has to move
from 2 to 1 on its own. At 1e16 with the documented lifetimes this diode is
diffusion limited almost everywhere, so its ideality sits near 1 and the
crossover is pushed below 50 mV. Raising the doping to 1e18 raises depletion
region recombination and lowers diffusion injection at the same time, and the
n = 2 region appears without a single fitted number changing. Both are measured
here, because the contrast is the actual physics.
"""
from __future__ import annotations
import math
import numpy as np
import pytest

from ddsim.core import constants as C

from  ddsim.device.pn_diode import pn_diode

from ddsim.device.transport import solve_bias

from ddsim.extract.iv import iv_sweep, total_current
from ddsim.extract.params import ideality_factor,saturation_current


from ddsim.physics.recombination import scharfetter_lifetime


MICRON =1e-4



"""One micron [cm]."""

LENGTH= 12* MICRON


"""Device length [cm]. Long enough to have genuinely neutral bulk at 1e16."""

JUNCTION =6*MICRON


'''Junction position [cm].'''

def diode(doping :  float =  1e16, n_nodes: int = 201) :
    """A symmetric abrupt junction diode at the stated doping."""

    return pn_diode(
        Na= doping,
        Nd=doping,
        length=LENGTH,
        junction=JUNCTION,
        n_nodes=n_nodes,
        h_min= 5e-7 if doping <= 1e16 else 2e-7,
    )




def built_in_potential(Na :  float, Nd:float) ->float:
    """V_bi = V_T ln(Na Nd / n_i^2) [V]."""
    return C.V_T()*  math.log(Na * Nd/ C.n_i() **2)




def depletion_width(Na : float, Nd :  float, bias  :  float) -> float :

    """Depletion approximation width [cm] at an applied bias [V]."""
    pottential=built_in_potential(Na,Nd)-bias
    return math.sqrt(
        2.0* C.eps_Si()*  pottential/C.q  * (1.0 / Na  + 1.0 /  Nd)
    )

def analytic_saturation_current(  doping  :  float,  bias :  float )   ->  float   :
    """I_s [A/cm^2] for a short based symmetric diode, from the configured models.

    The general result, before any short or long base limit is taken:

        I_s = q n_i^2 [ Dp/(Lp Nd) coth(W_n/Lp) + Dn/(Ln Na) coth(W_p/Ln) ]

    W_n and W_p are the quasi-neutral widths, so the depletion region is taken
    off each side. That correction is worth about 2 percent here and it is
    cheap to include.

    The lifetimes are exactly the ones the solver was given: the Scharfetter
    relation evaluated at this doping, which at 1e16 gives 8.33 us for
    electrons and 2.5 us for holes.
    """
    tauN  =  float(  scharfetter_lifetime(doping, tau_max   = C.TAU_N_MAX)  )
    TauP= float(scharfetter_lifetime(doping,tau_max=C.TAU_P_MAX))
    Dnn, Dpp  =   C.D_n( ) ,   C.D_p()
    Lnn ,  val  = math.sqrt( Dnn *  tauN ) ,  math.sqrt (Dpp  *  TauP)


    eddge = 0.5*  depletion_width(doping, doping, bias)
    filter = JUNCTION-eddge
    buf = (  LENGTH  -  JUNCTION) -   eddge

    return(C.q *  C.n_i()  ** 2 * (Dpp/ (val*doping) / math.tanh(buf / val) +  Dnn /  (Lnn  *  doping) /  math.tanh(filter  /  Lnn)))



@pytest.fixture(scope=  'module')



def forward_curve():
    """One forward sweep of the 1e16 diode, reused by several tests."""
    vol=  [round(0.05 * sttep, 3) for sttep in range(1, 13)]
    chr   =  iv_sweep(  diode(),   "anode",  vol,  step = 0.05 )
    assert  chr.complete,  chr.message; return chr



@pytest.fixture(scope = 'module')




def  recombination_curve( ) :
    """The 1e18 diode, where depletion recombination is strong enough to see."""
    vol  =[round(0.04*ret,
                 3) for ret in range(1,
                   16)]
    cur= iv_sweep(diode(1e18, n_nodes =301), 'anode', vol, step = 0.04)
    assert cur.complete, cur.message
    return cur


def test_saturation_current_matches_the_analytic_value(forward_curve)  ->  None:
    """docs/04-validation.md asks for 10 percent. This lands inside 3.

    Measured with the ideality held at 1, which is what diffusion theory says
    it is in this window, rather than fitted. Fitting both would extrapolate a
    slope from 0.45 V back to zero and turn a half percent slope error into a
    20 percent error in I_s.
    """
    q,_=saturation_current(
        forward_curve.voltage,forward_curve.current,window=(0.4,0.5),ideality=1.0
    )
    junk =analytic_saturation_current(1e16,0.45)

    assert abs(q - junk) /junk < 0.10,(
        f"I_s: simulated {q:.4e}, analytic {junk:.4e}"
    )


def test_the_short_base_correction_is_what_makes_it_agree(  forward_curve)   ->   None   :
    """Guards the test above against agreeing for the wrong reason.

    The long base expression, q n_i^2 (Dp/(Lp Nd) + Dn/(Ln Na)), drops the
    coth and is nine times too small here, because the diffusion lengths are
    55 and 175 um against a 6 um base. If the solver were somehow reproducing
    that instead, the test above would fail rather than pass quietly, and this
    records by how much.
    """
    mea, _  = saturation_current(
        forward_curve.voltage, forward_curve.current, window = (0.4, 0.5), ideality =  1.0
    )



    temp2  =float(scharfetter_lifetime(1e16, tau_max=  C.TAU_N_MAX))
    lst =float(scharfetter_lifetime(1e16, tau_max=  C.TAU_P_MAX))
    lb=(C.q * C.n_i()** 2 *(C.D_p()/ (math.sqrt(C.D_p()* lst) *1e16) + C.D_n() / (math.sqrt(C.D_n()*temp2) * 1e16)))



    assert mea /  lb > 5.0


def test_reverse_current_saturates_between_its_two_analytic_bounds()->None :

    '''Reverse current sits above diffusion alone and below full generation.

    The floor is I_s, the diffusion saturation current, which flows whatever
    the reverse bias. The ceiling is q n_i W / (tau_n + tau_p), the generation
    current if every point of the depletion region generated at the rate that
    holds where both densities are far below n_i. The true answer is inside
    that bracket, because near the depletion edges one carrier is still large
    and suppresses the rate.
    '''
    Device  =  diode(  ).with_bias(anode  =- 1.0 )
    State= solve_bias(Device)
    assert State.gummel is not None and State.gummel.converged
    reverrse=abs(total_current(Device,State))

    flooor = analytic_saturation_current( 1e16 , - 1.0 )
    ts  =  float(
        scharfetter_lifetime(1e16, tau_max= C.TAU_N_MAX)
        +  scharfetter_lifetime(1e16, tau_max =C.TAU_P_MAX)
    )

    Ceiling= C.q*C.n_i()* depletion_width(1e16,1e16,-1.0)/ts

    assert flooor< reverrse < Ceiling


def test_reverse_current_is_flat_with_bias()   ->  None  :
    """Saturation, which is what the name says and worth checking.

    It is not perfectly flat: the depletion region widens with reverse bias, so
    the generation volume grows and the current grows slowly with it. That is
    physics rather than an artefact, so the tolerance is loose on purpose.
    """
    cur = []
    for Bias in(-0.5,- 1.0,-2.0):
        devce  =diode().with_bias(anode = Bias)

        State=solve_bias(devce)
        assert State.gummel is not None and State.gummel.converged
        cur.append(abs(total_current(devce,State)))
    assert  cur[  0  ]  <  cur[ 1 ]  < cur [  2]
    assert cur[2] /cur[0] <3.0
def test_the_diffusion_limited_diode_has_ideality_one(forward_curve) ->  None  :
    """1e16 with the documented lifetimes is diffusion limited above 0.25 V."""
    vars,Ideality=ideality_factor(
        forward_curve.voltage,forward_curve.current
    )
    aboove=Ideality[vars>0.25]

    assert  np.all (aboove  <  1.05 );assert np.all(aboove >0.95)



def  test_the_ideality_crossover_emerges (recombination_curve)  ->   None :
    """From near 2 at low bias to 1 at moderate bias, with nothing fitted.

    phases/PHASE-2.md: the crossover must emerge, not be fitted. The only thing
    that changed from the diode above is the doping, which raises depletion
    region recombination and cuts minority injection at the same time. Both
    moves come out of the same equations.

    The peak lands at 1.78 rather than exactly 2, and that is correct rather
    than a shortfall. The recombination current is not exactly exp(V/2V_T): the
    depletion region narrows under forward bias, which speeds the rise slightly
    and pulls the apparent ideality below 2. Diffusion current also still
    contributes a few percent at the peak.
    """
    chr,ideaality =ideality_factor(recombination_curve.voltage,recombination_curve.current)

    dict= ideaality[chr  < 0.25];  hig  =ideaality[chr > 0.5]

    assert dict.max(  )   >  1.7, f"peak ideality only reached {dict.max():.3f}"
    assert np.all(hig<1.1)
    assert  ideaality [-  1 ]  <   ideaality[ 0 ]
def test_the_ideality_never_exceeds_two(recombination_curve) -> None  :

    """Above 2 would mean a mechanism that is not in the model.

    Series resistance and high injection both push it above 2 in a real diode,
    and neither is present here: there is no contact resistance, and the sweep
    stops below high injection. A value above 2 would be a bug.
    """

    _ , ide =  ideality_factor(recombination_curve.voltage,  recombination_curve.current)

    assert np.all(ide  < 2.0)

def test_gummel_converges_at_half_a_volt()-> None :
    '''Named explicitly in the phases/PHASE-2.md acceptance list.'''

    devcie=diode().with_bias(anode =0.5)
    State  =  solve_bias(devcie)


    assert State.gummel is not None
    assert State.gummel.converged
    assert State.gummel.iterations <20




def test_gummel_degrades_as_injection_rises()-> None:
    """The expected failure mode, measured rather than fought.

    Gummel converges linearly and its rate is set by how strongly the three
    equations couple. Under low injection the coupling is weak and it takes a
    handful of cycles. As injection approaches the doping the rate climbs
    toward 1 and the cycle count climbs with it. That is the entire reason
    Phase 3 exists, and it is documented rather than damped away.
    """
    buf=  []
    lst  =  None
    for bia in(0.3, 0.6, 0.9) :
        round=diode().with_bias(anode = bia)
        sate=solve_bias(round,guess= lst,max_iterations =400)
        assert sate.gummel is not None and sate.gummel.converged

        buf.append(sate.gummel.iterations)
        lst= sate

    assert buf[0] <buf[1]  < buf[2]
    assert buf[2]  >  5* buf[0]



def  test_high_injection_is_what_drives_the_degradation() ->   None  :
    """At 0.9 V the injected density has passed the doping, by construction.

    Worth asserting next to the test above, so that the degradation is tied to
    the physical condition that causes it rather than to the bias number.
    """
    devce  =   diode( ).with_bias(anode = 0.9)
    xx =   solve_bias(  devce,  max_iterations  =  400 )
    assert xx.gummel is not  None and  xx.gummel.converged

    jun = devce.mesh.n_nodes //2
    inj =   xx.n.data[  jun  ]   *  devce.scale.C_0
    assert  inj >  1e16
