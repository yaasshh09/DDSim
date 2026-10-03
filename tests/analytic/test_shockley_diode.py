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

    t=built_in_potential(Na,Nd)-bias
    return math.sqrt(
        2.0* C.eps_Si()*  t/C.q  * (1.0 / Na  + 1.0 /  Nd)
    )

def analytic_saturation_current(  doping  :  float,  bias :  float )   ->  float   :
    x2  =  float(  scharfetter_lifetime(doping, tau_max   = C.TAU_N_MAX)  )
    u= float(scharfetter_lifetime(doping,tau_max=C.TAU_P_MAX))
    b, t  =   C.D_n( ) ,   C.D_p()
    tmp ,  a  = math.sqrt( b *  x2 ) ,  math.sqrt (t  *  u)


    r2 = 0.5*  depletion_width(doping, doping, bias)
    d = JUNCTION-r2
    hh = (  LENGTH  -  JUNCTION) -   r2

    return(C.q *  C.n_i()  ** 2 * (t/ (a*doping) / math.tanh(hh / a) +  b /  (tmp  *  doping) /  math.tanh(d  /  tmp)))



@pytest.fixture(scope=  'module')



def forward_curve():
    lst=  [round(0.05 * val, 3) for val in range(1, 13)]
    res   =  iv_sweep(  diode(),   "anode",  lst,  step = 0.05 )
    assert  res.complete,  res.message; return res



@pytest.fixture(scope = 'module')




def  recombination_curve( ) :
    x  =[round(0.04*z,
                 3) for z in range(1,
                   16)]
    dd= iv_sweep(diode(1e18, n_nodes =301), 'anode', x, step = 0.04)
    assert dd.complete, dd.message
    return dd


def test_saturation_current_matches_the_analytic_value(forward_curve)  ->  None:
    d,_=saturation_current(
        forward_curve.voltage,forward_curve.current,window=(0.4,0.5),ideality=1.0
    )
    r =analytic_saturation_current(1e16,0.45)

    assert abs(d - r) /r < 0.10,(
        f"I_s: simulated {d:.4e}, analytic {r:.4e}"
    )


def test_the_short_base_correction_is_what_makes_it_agree(  forward_curve)   ->   None   :
    j, _  = saturation_current(
        forward_curve.voltage, forward_curve.current, window = (0.4, 0.5), ideality =  1.0
    )



    obj  =float(scharfetter_lifetime(1e16, tau_max=  C.TAU_N_MAX))
    t =float(scharfetter_lifetime(1e16, tau_max=  C.TAU_P_MAX))
    s2=(C.q * C.n_i()** 2 *(C.D_p()/ (math.sqrt(C.D_p()* t) *1e16) + C.D_n() / (math.sqrt(C.D_n()*obj) * 1e16)))



    assert j /  s2 > 5.0


def test_reverse_current_saturates_between_its_two_analytic_bounds()->None :

    g  =  diode(  ).with_bias(anode  =- 1.0 )
    z= solve_bias(g)
    assert z.gummel is not None and z.gummel.converged
    h=abs(total_current(g,z))

    ss = analytic_saturation_current( 1e16 , - 1.0 )
    r  =  float(
        scharfetter_lifetime(1e16, tau_max= C.TAU_N_MAX)
        +  scharfetter_lifetime(1e16, tau_max =C.TAU_P_MAX)
    )

    dat= C.q*C.n_i()* depletion_width(1e16,1e16,-1.0)/r

    assert ss< h < dat


def test_reverse_current_is_flat_with_bias()   ->  None  :
    w = []
    for j in(-0.5,- 1.0,-2.0):
        h  =diode().with_bias(anode = j)

        a=solve_bias(h)
        assert a.gummel is not None and a.gummel.converged
        w.append(abs(total_current(h,a)))
    assert  w[  0  ]  <  w[ 1 ]  < w [  2]
    assert w[2] /w[0] <3.0
def test_the_diffusion_limited_diode_has_ideality_one(forward_curve) ->  None  :
    zz,d=ideality_factor(
        forward_curve.voltage,forward_curve.current
    )
    s=d[zz>0.25]

    assert  np.all (s  <  1.05 );assert np.all(s >0.95)



def  test_the_ideality_crossover_emerges (recombination_curve)  ->   None :
    j,a =ideality_factor(recombination_curve.voltage,recombination_curve.current)

    v= a[j  < 0.25];  prev  =a[j > 0.5]

    assert v.max(  )   >  1.7, f"peak ideality only reached {v.max():.3f}"
    assert np.all(prev<1.1)
    assert  a [-  1 ]  <   a[ 0 ]
def test_the_ideality_never_exceeds_two(recombination_curve) -> None  :

    _ , d =  ideality_factor(recombination_curve.voltage,  recombination_curve.current)

    assert np.all(d  < 2.0)

def test_gummel_converges_at_half_a_volt()-> None :

    b=diode().with_bias(anode =0.5)
    e  =  solve_bias(b)


    assert e.gummel is not None
    assert e.gummel.converged
    assert e.gummel.iterations <20




def test_gummel_degrades_as_injection_rises()-> None:
    thing=  []
    foo  =  None
    for dd in(0.3, 0.6, 0.9) :
        bb=diode().with_bias(anode = dd)
        flag=solve_bias(bb,guess= foo,max_iterations =400)
        assert flag.gummel is not None and flag.gummel.converged

        thing.append(flag.gummel.iterations)
        foo= flag

    assert thing[0] <thing[1]  < thing[2]
    assert thing[2]  >  5* thing[0]



def  test_high_injection_is_what_drives_the_degradation() ->   None  :
    c  =   diode( ).with_bias(anode = 0.9)
    r =   solve_bias(  c,  max_iterations  =  400 )
    assert r.gummel is not  None and  r.gummel.converged

    r2 = c.mesh.n_nodes //2
    rr =   r.n.data[  r2  ]   *  c.scale.C_0
    assert  rr >  1e16
