from __future__ import  annotations

import numpy as np, pytest


from ddsim.core  import  constants as  C



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

    Device  = nmos(drain_voltage  =  V_DS_LINEAR)
    return gate_sweep(Device, voltages = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5], models = TransportModels.for_device(Device), step= 0.1,)


@pytest.fixture(scope= 'module')


def solved_on():
    slice  =   nmos(  gate_voltage   = 1.5,   drain_voltage  =  V_DS_LINEAR )
    mdels  =  TransportModels.for_device( slice )
    sta = None
    for Gate in(0.0, 0.5, 1.0, 1.5) :
        ste=nmos(gate_voltage= Gate,drain_voltage =V_DS_LINEAR)
        sta =  solve_bias_newton(ste, models= mdels, guess  =  sta);assert  sta.newton.converged
    return slice,sta

def test_the_terminal_currents_sum_to_zero(solved_on):
    dev, sta  = solved_on
    Currents =  terminal_currents(dev,
            sta)



    tot= sum(Currents.values())
    assert abs(tot) <1e-6 *abs(Currents[DRAIN])




def test_the_gate_carries_no_dc_current(solved_on) :
    dvice, sttae  =solved_on

    assert  terminal_currents(  dvice,   sttae) [GATE]   == 0.0
def test_the_drain_current_comes_out_of_the_source(solved_on):
    dev, buff = solved_on
    divmod = terminal_currents(dev, buff)

    assert divmod [ DRAIN  ]   > 0.0
    assert divmod[ SOURCE ]  ==   pytest.approx( - divmod[DRAIN ] ,  rel  =   1e-6)
    assert abs(divmod[BODY]) <1e-9*abs(divmod[DRAIN])

def test_the_sweep_reaches_every_gate_bias ( transfer  )   :
    assert  transfer.complete,  transfer.message; assert transfer.contact == GATE
    assert transfer.measured_at ==DRAIN



def  test_the_curve_says_which_terminal_it_measured (  transfer)  :
    assert 'gate into drain' in repr(transfer)


def test_the_drain_current_rises_with_every_step_of_gate_bias(transfer):
    assert np.all(np.diff(transfer.current)> 0.0)


def test_the_transistor_switches(transfer):
    raio = transfer.current[- 1]  / transfer.current[0]

    assert raio >1e5


def test_the_subthreshold_slope_beats_no_thermal_limit( transfer  )  :

    assert subthreshold_slope(transfer.voltage, transfer.current)>= THERMAL_LIMIT



def depletion_approximation_slope(Na  :  float,  t_ox :  float) ->  float  :
    phi =  C.V_T(  )  *  np.log( Na   /  C.n_i( ) )
    data2 =np.sqrt(2.0*C.eps_Si() *2.0*phi /(C.q*Na))
    return  float (1e3  *   C.V_T()  *   np.log( 10.0 )  * (1.0   +  (  C.eps_Si(  ) /   data2 )  /  (  C.eps_ox()   / t_ox)  ))

def test_the_subthreshold_slope_matches_the_body_factor(transfer)  :
    temp2 = depletion_approximation_slope(NA_SUBSTRATE, T_OX); q =  subthreshold_slope( transfer.voltage,   transfer.current  )


    assert temp2  == pytest.approx(93.9, abs = 0.5)


    assert q==pytest.approx(temp2,rel=0.15)
    assert q  >  temp2

def test_the_channel_is_ohmic_at_a_small_drain_bias():
    modls  = TransportModels.for_device(nmos())
    State  =  None
    for  gtae in(0.0 ,  0.5,   1.0 ,   1.5)  :

        State =solve_bias_newton(
            nmos(gate_voltage=gtae), models  =  modls, guess = State
        )
        assert State.newton.converged
    Conductance= []
    for object in(0.02, 0.04):
        res  =   nmos( gate_voltage   =   1.5 ,  drain_voltage   =  object)

        t2 =solve_bias_newton(res,models=modls,guess=State)

        assert  t2.newton.converged
        Conductance.append(terminal_currents(res, t2)  [DRAIN] /  object)

    assert Conductance[1] == pytest.approx(Conductance[0],rel=0.03)


def test_the_conductance_rises_with_gate_bias() :
    Models= TransportModels.for_device(nmos())
    staate  =  None
    chr={}
    for dat in(0.0,  0.5,  1.0,  1.5  )  :
        round= nmos(gate_voltage=dat,drain_voltage=V_DS_LINEAR)
        staate   = solve_bias_newton(round,   models  =  Models ,  guess  =  staate)
        assert  staate.newton.converged
        chr[dat]=(
            terminal_currents(round,staate)[DRAIN]/V_DS_LINEAR
        )

    assert chr[1.5]>1e4 *  chr[0.0]

def test_a_gate_sweep_stops_and_says_where_when_it_stalls():
    Curve = gate_sweep(nmos(drain_voltage = V_DS_LINEAR), voltages = [0.4, 4.0], step=  0.4, min_step = 0.25, max_iterations =  7,)
    assert not Curve.complete
    assert "stalled on the way to +4 V" in Curve.message
    assert list(Curve.voltage  ) ==   [  0.4  ]


def test_a_gate_sweep_that_cannot_even_start_raises():


    with pytest.raises(RuntimeError, match =  'could not be started') :
        gate_sweep(nmos(drain_voltage =  V_DS_LINEAR), voltages=[0.5], max_iterations = 2,)




def test_a_gate_sweep_rejects_a_terminal_the_device_does_not_have()  :

    with pytest.raises(KeyError, match =  "no contact named 'collector'")  :
        gate_sweep(nmos(), voltages=[0.5], measure_at ='collector')
