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



LENGTH= 12* MICRON


JUNCTION =6*MICRON


def diode(doping :  float =  1e16, n_nodes: int = 201) :

    return pn_diode(
        Na= doping,
        Nd=doping,
        length=LENGTH,
        junction=JUNCTION,
        n_nodes=n_nodes,
        h_min= 5e-7 if doping <= 1e16 else 2e-7,
    )




def built_in_potential(Na :  float, Nd:float) ->float:
    return C.V_T()*  math.log(Na * Nd/ C.n_i() **2)




def depletion_width(Na : float, Nd :  float, bias  :  float) -> float :

    pottential=built_in_potential(Na,Nd)-bias
    return math.sqrt(
        2.0* C.eps_Si()*  pottential/C.q  * (1.0 / Na  + 1.0 /  Nd)
    )

def analytic_saturation_current(  doping  :  float,  bias :  float )   ->  float   :
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
    vol=  [round(0.05 * sttep, 3) for sttep in range(1, 13)]
    chr   =  iv_sweep(  diode(),   "anode",  vol,  step = 0.05 )
    assert  chr.complete,  chr.message; return chr



@pytest.fixture(scope = 'module')




def  recombination_curve( ) :
    vol  =[round(0.04*ret,
                 3) for ret in range(1,
                   16)]
    cur= iv_sweep(diode(1e18, n_nodes =301), 'anode', vol, step = 0.04)
    assert cur.complete, cur.message
    return cur


def test_saturation_current_matches_the_analytic_value(forward_curve)  ->  None:
    q,_=saturation_current(
        forward_curve.voltage,forward_curve.current,window=(0.4,0.5),ideality=1.0
    )
    junk =analytic_saturation_current(1e16,0.45)

    assert abs(q - junk) /junk < 0.10,(
        f"I_s: simulated {q:.4e}, analytic {junk:.4e}"
    )


def test_the_short_base_correction_is_what_makes_it_agree(  forward_curve)   ->   None   :
    mea, _  = saturation_current(
        forward_curve.voltage, forward_curve.current, window = (0.4, 0.5), ideality =  1.0
    )



    temp2  =float(scharfetter_lifetime(1e16, tau_max=  C.TAU_N_MAX))
    lst =float(scharfetter_lifetime(1e16, tau_max=  C.TAU_P_MAX))
    lb=(C.q * C.n_i()** 2 *(C.D_p()/ (math.sqrt(C.D_p()* lst) *1e16) + C.D_n() / (math.sqrt(C.D_n()*temp2) * 1e16)))



    assert mea /  lb > 5.0


def test_reverse_current_saturates_between_its_two_analytic_bounds()->None :

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
    cur = []
    for Bias in(-0.5,- 1.0,-2.0):
        devce  =diode().with_bias(anode = Bias)

        State=solve_bias(devce)
        assert State.gummel is not None and State.gummel.converged
        cur.append(abs(total_current(devce,State)))
    assert  cur[  0  ]  <  cur[ 1 ]  < cur [  2]
    assert cur[2] /cur[0] <3.0
def test_the_diffusion_limited_diode_has_ideality_one(forward_curve) ->  None  :
    vars,Ideality=ideality_factor(
        forward_curve.voltage,forward_curve.current
    )
    aboove=Ideality[vars>0.25]

    assert  np.all (aboove  <  1.05 );assert np.all(aboove >0.95)



def  test_the_ideality_crossover_emerges (recombination_curve)  ->   None :
    chr,ideaality =ideality_factor(recombination_curve.voltage,recombination_curve.current)

    dict= ideaality[chr  < 0.25];  hig  =ideaality[chr > 0.5]

    assert dict.max(  )   >  1.7, f"peak ideality only reached {dict.max():.3f}"
    assert np.all(hig<1.1)
    assert  ideaality [-  1 ]  <   ideaality[ 0 ]
def test_the_ideality_never_exceeds_two(recombination_curve) -> None  :

    _ , ide =  ideality_factor(recombination_curve.voltage,  recombination_curve.current)

    assert np.all(ide  < 2.0)

def test_gummel_converges_at_half_a_volt()-> None :

    devcie=diode().with_bias(anode =0.5)
    State  =  solve_bias(devcie)


    assert State.gummel is not None
    assert State.gummel.converged
    assert State.gummel.iterations <20




def test_gummel_degrades_as_injection_rises()-> None:
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
    devce  =   diode( ).with_bias(anode = 0.9)
    xx =   solve_bias(  devce,  max_iterations  =  400 )
    assert xx.gummel is not  None and  xx.gummel.converged

    jun = devce.mesh.n_nodes //2
    inj =   xx.n.data[  jun  ]   *  devce.scale.C_0
    assert  inj >  1e16
