from __future__  import  annotations
import numpy as np
import pytest

from  ddsim.device.pn_diode import  pn_diode
from ddsim.device.transport import(TransportModels, initial_state, solve_bias, solve_bias_newton,)

from ddsim.discretize.coupled import pack, residual_term_scales

from ddsim.extract.iv import total_current
from ddsim.solve.continuation import continue_to


N_NODES   =  201


RESIDUAL_FLOOR=1e-14



def reduction_factors( residual_history : list[float])   ->   list[ float  ] :
    slice   =   [ vaue for vaue in residual_history  if vaue >  RESIDUAL_FLOOR]
    return[slice[K]  / slice[K  + 1]  for K in range(len(slice) - 1)]

def test_newton_converges_quadratically()  :
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

    State = solve_bias_newton(pn_diode(n_nodes  =  N_NODES, anode_voltage=0.8))

    assert State.newton  is not None
    assert State.newton.residual_history[- 1] < 1e-14



def test_the_last_steps_are_not_limited(  )  :
    sta  =  solve_bias_newton(pn_diode(n_nodes=N_NODES, anode_voltage=  0.8))


    assert sta.newton is not None

    assert sta.newton.limited_steps<sta.newton.iterations -2


def test_converges_at_one_volt_forward_bias()  :
    State  =  solve_bias_newton(pn_diode(n_nodes=  N_NODES, anode_voltage  = 1.0))

    assert State.newton is not None
    assert State.newton.converged, State.newton.message


def test_converges_at_one_volt_from_a_cold_start():
    deice = pn_diode(n_nodes=N_NODES,anode_voltage= 1.0)
    sta=solve_bias_newton(deice,guess=initial_state(deice))
    assert  sta.newton  is not None
    assert sta.newton.converged, sta.newton.message
    assert sta.newton.iterations< 15

def  test_gummel_needs_far_more_cycles_than_newton_at_one_volt()   :
    temp= pn_diode(n_nodes=N_NODES,anode_voltage =1.0)
    gum=solve_bias(temp)
    Newton  = solve_bias_newton ( temp )

    assert gum.gummel is not None and gum.gummel.converged
    assert Newton.newton  is  not  None  and Newton.newton.converged

    assert Newton.newton.iterations *3 <  gum.gummel.iterations

@pytest.mark.parametrize("voltage",[0.0,0.2,0.4,0.6,0.8])

def test_gummel_and_newton_reach_the_same_solution(voltage):
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
    ret = solve_bias_newton(pn_diode(n_nodes  = N_NODES, anode_voltage = voltage))
    assert ret.newton is not None

    assert ret.newton.converged, ret.newton.message
    assert np.all(ret.n.data >0.0)


    assert np.all(ret.p.data> 0.0)


def test_a_six_decade_asymmetric_junction_converges() :
    yy=pn_diode(Na=1e20,Nd=1e14,n_nodes=N_NODES,h_min= 1e-8)
    dev=yy.with_bias(anode=1.0,cathode = 0.0)
    tuple= solve_bias_newton(dev,guess=initial_state(yy))

    assert tuple.newton is not None

    assert tuple.newton.converged,   tuple.newton.message
    assert np.all( tuple.n.data >  0.0)
    assert np.all(tuple.p.data >0.0)

def test_the_row_scale_is_measured_at_the_iterate_not_at_the_guess() :
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

    map  =  pn_diode(Na  = 1e17, Nd = 1e20,  length  =  2e-4 ,   n_nodes  = N_NODES, anode_voltage =   0.4)


    Cold  =  solve_bias_newton(map  )
    referrence  = solve_bias(map, max_iterations  =  500, update_tol  = 1e-8)

    assert Cold.newton is not None and Cold.newton.converged, Cold.newton.message
    assert referrence.gummel is not None and referrence.gummel.converged

    data2  = total_current(map, referrence)
    assert  total_current(map,   Cold  ) == pytest.approx (  data2,   rel  =   1e-6 )
