from __future__ import annotations
import numpy as np, pytest
from ddsim.core import constants as C
from  ddsim.device.mosfet  import  BODY,  DRAIN,  GATE,   SOURCE ,  nmos
from  ddsim.device.transport  import  TransportModels,   solve_bias_newton
from ddsim.extract.iv import gate_sweep,terminal_currents
from ddsim.extract.params import subthreshold_slope


THERMAL_LIMIT=  59.5



NA_SUBSTRATE  =1e17


T_OX= 2e-6


V_DS_LINEAR= 0.05



@pytest.fixture(scope ='module')

def  transfer()   :

    y  = nmos(drain_voltage  =  V_DS_LINEAR)
    return gate_sweep(y, voltages = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5], models = TransportModels.for_device(y), step= 0.1,)


@pytest.fixture(scope= 'module')


def solved_on():
    j  =   nmos(  gate_voltage   = 1.5,   drain_voltage  =  V_DS_LINEAR )
    z  =  TransportModels.for_device( j )
    a = None
    for arr in(0.0, 0.5, 1.0, 1.5) :
        r=nmos(gate_voltage= arr,drain_voltage =V_DS_LINEAR)
        a =  solve_bias_newton(r, models= z, guess  =  a);assert  a.newton.converged
    return j,a

def test_the_terminal_currents_sum_to_zero(solved_on):
    a, zz  = solved_on
    hh =  terminal_currents(a,
            zz)



    x= sum(hh.values())
    assert abs(x) <1e-6 *abs(hh[DRAIN])




def test_the_gate_carries_no_dc_current(solved_on) :
    i, u  =solved_on

    assert  terminal_currents(  i,   u) [GATE]   == 0.0
def test_the_drain_current_comes_out_of_the_source(solved_on):
    a, w2 = solved_on
    h = terminal_currents(a, w2)

    assert h [ DRAIN  ]   > 0.0
    assert h[ SOURCE ]  ==   pytest.approx( - h[DRAIN ] ,  rel  =   1e-6)
    assert abs(h[BODY]) <1e-9*abs(h[DRAIN])

def test_the_sweep_reaches_every_gate_bias ( transfer  )   :
    assert  transfer.complete,  transfer.message; assert transfer.contact == GATE
    assert transfer.measured_at ==DRAIN



def  test_the_curve_says_which_terminal_it_measured (  transfer)  :
    assert 'gate into drain' in repr(transfer)


def test_the_drain_current_rises_with_every_step_of_gate_bias(transfer):
    assert np.all(np.diff(transfer.current)> 0.0)


def test_the_transistor_switches(transfer):
    u = transfer.current[- 1]  / transfer.current[0]

    assert u >1e5


def test_the_subthreshold_slope_beats_no_thermal_limit( transfer  )  :

    assert subthreshold_slope(transfer.voltage, transfer.current)>= THERMAL_LIMIT



def depletion_approximation_slope(Na  :  float,  t_ox :  float) ->  float  :
    rows =  C.V_T(  )  *  np.log( Na   /  C.n_i( ) )
    b =np.sqrt(2.0*C.eps_Si() *2.0*rows /(C.q*Na))
    return  float (1e3  *   C.V_T()  *   np.log( 10.0 )  * (1.0   +  (  C.eps_Si(  ) /   b )  /  (  C.eps_ox()   / t_ox)  ))

def test_the_subthreshold_slope_matches_the_body_factor(transfer)  :
    v = depletion_approximation_slope(NA_SUBSTRATE, T_OX); j =  subthreshold_slope( transfer.voltage,   transfer.current  )


    assert v  == pytest.approx(93.9, abs = 0.5)


    assert j==pytest.approx(v,rel=0.15)
    assert j  >  v

def test_the_channel_is_ohmic_at_a_small_drain_bias():
    z  = TransportModels.for_device(nmos())
    m  =  None
    for  tmp2 in(0.0 ,  0.5,   1.0 ,   1.5)  :

        m =solve_bias_newton(
            nmos(gate_voltage=tmp2), models  =  z, guess = m
        )
        assert m.newton.converged
    w= []
    for d2 in(0.02, 0.04):
        ys  =   nmos( gate_voltage   =   1.5 ,  drain_voltage   =  d2)

        b2 =solve_bias_newton(ys,models=z,guess=m)

        assert  b2.newton.converged
        w.append(terminal_currents(ys, b2)  [DRAIN] /  d2)

    assert w[1] == pytest.approx(w[0],rel=0.03)


def test_the_conductance_rises_with_gate_bias() :
    i= TransportModels.for_device(nmos())
    yy  =  None
    u={}
    for d2 in(0.0,  0.5,  1.0,  1.5  )  :
        y= nmos(gate_voltage=d2,drain_voltage=V_DS_LINEAR)
        yy   = solve_bias_newton(y,   models  =  i ,  guess  =  yy)
        assert  yy.newton.converged
        u[d2]=(
            terminal_currents(y,yy)[DRAIN]/V_DS_LINEAR
        )

    assert u[1.5]>1e4 *  u[0.0]

def test_a_gate_sweep_stops_and_says_where_when_it_stalls():
    w = gate_sweep(nmos(drain_voltage = V_DS_LINEAR), voltages = [0.4, 4.0], step=  0.4, min_step = 0.25, max_iterations =  7,)
    assert not w.complete
    assert "stalled on the way to +4 V" in w.message
    assert list(w.voltage  ) ==   [  0.4  ]


def test_a_gate_sweep_that_cannot_even_start_raises():


    with pytest.raises(RuntimeError, match =  'could not be started') :
        gate_sweep(nmos(drain_voltage =  V_DS_LINEAR), voltages=[0.5], max_iterations = 2,)




def test_a_gate_sweep_rejects_a_terminal_the_device_does_not_have()  :

    with pytest.raises(KeyError, match =  "no contact named 'collector'")  :
        gate_sweep(nmos(), voltages=[0.5], measure_at ='collector')
