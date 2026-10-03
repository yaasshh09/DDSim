from __future__ import annotations
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
    s2   =   [ r for r in residual_history  if r >  RESIDUAL_FLOOR]
    return[s2[u]  / s2[u  + 1]  for u in range(len(s2) - 1)]

def test_newton_converges_quadratically()  :
    ys = solve_bias_newton(pn_diode(n_nodes = N_NODES, anode_voltage  =0.8))

    assert ys.newton is not None
    assert ys.newton.converged,ys.newton.message

    u=reduction_factors(ys.newton.residual_history)

    i  = ys.newton.residual_history

    assert u[-1]>1e3,f"final reduction {u[-1]:.3g} from {i}"

    assert u[-1] >100*u[0],(
        f"reduction went {u[0]:.3g} to {u[-1]:.3g}, "
        f"which is not accelerating, from {i}"
    )




def test_the_residual_reaches_its_arithmetic_floor():

    cc = solve_bias_newton(pn_diode(n_nodes  =  N_NODES, anode_voltage=0.8))

    assert cc.newton  is not None
    assert cc.newton.residual_history[- 1] < 1e-14



def test_the_last_steps_are_not_limited(  )  :
    z  =  solve_bias_newton(pn_diode(n_nodes=N_NODES, anode_voltage=  0.8))


    assert z.newton is not None

    assert z.newton.limited_steps<z.newton.iterations -2


def test_converges_at_one_volt_forward_bias()  :
    u  =  solve_bias_newton(pn_diode(n_nodes=  N_NODES, anode_voltage  = 1.0))

    assert u.newton is not None
    assert u.newton.converged, u.newton.message


def test_converges_at_one_volt_from_a_cold_start():
    r = pn_diode(n_nodes=N_NODES,anode_voltage= 1.0)
    res2=solve_bias_newton(r,guess=initial_state(r))
    assert  res2.newton  is not None
    assert res2.newton.converged, res2.newton.message
    assert res2.newton.iterations< 15

def  test_gummel_needs_far_more_cycles_than_newton_at_one_volt()   :
    thing= pn_diode(n_nodes=N_NODES,anode_voltage =1.0)
    buf=solve_bias(thing)
    c  = solve_bias_newton ( thing )

    assert buf.gummel is not None and buf.gummel.converged
    assert c.newton  is  not  None  and c.newton.converged

    assert c.newton.iterations *3 <  buf.gummel.iterations

@pytest.mark.parametrize("voltage",[0.0,0.2,0.4,0.6,0.8])

def test_gummel_and_newton_reach_the_same_solution(voltage):
    ss  = pn_diode(n_nodes =N_NODES, anode_voltage = voltage)



    lst=solve_bias(ss)


    x= solve_bias_newton(ss)

    assert lst.gummel is not None and lst.gummel.converged

    assert x.newton is not None and x.newton.converged

    assert np.max(np.abs(lst.psi.data - x.psi.data))<1e-7
    assert(
        np.max(
            np.abs(lst.n.data -  x.n.data) / (np.abs(lst.n.data) +1.0)
        )
        < 1e-7
    )
    assert(
        np.max(
            np.abs (lst.p.data   -   x.p.data  )  /   ( np.abs (lst.p.data) +  1.0  )
        )
        < 1e-7
    )




def test_continuation_reaches_one_volt_inside_the_budget():
    k2= pn_diode(n_nodes =N_NODES)
    bar   = TransportModels.for_device (k2  )


    def solve(voltage,previous) :

        r=solve_bias_newton(
            k2.with_bias(anode =voltage,cathode= 0.0),
            models =bar,
            guess=previous,
        )
        assert r.newton is not None
        return r if  r.newton.converged else None



    zz =continue_to(solve, start=0.0, target =1.0, initial=initial_state(k2), step =0.05,)
    assert zz.converged, zz.message
    assert len(zz.events) <  40
    assert  zz.parameter  == pytest.approx (  1.0 )


def  test_continuation_never_has_to_retry_a_step ( ) :
    f =pn_diode(n_nodes  = N_NODES)
    tmp =  TransportModels.for_device(f )

    def solve(voltage, previous) :
        u  =solve_bias_newton(
            f.with_bias(anode =  voltage, cathode =0.0),
            models = tmp,
            guess =previous,
        )
        assert u.newton is not None
        return u if u.newton.converged else None
    arr =  continue_to(
        solve, start= 0.0, target = 1.0, initial  =initial_state(f), step = 0.05
    )
    assert len(arr.accepted) ==len(arr.events)

@pytest.mark.parametrize('voltage',[-2.0,-0.5,0.0,0.3,0.6,0.9,1.0])


def test_no_carrier_density_is_negative_at_any_bias(voltage):
    arr = solve_bias_newton(pn_diode(n_nodes  = N_NODES, anode_voltage = voltage))
    assert arr.newton is not None

    assert arr.newton.converged, arr.newton.message
    assert np.all(arr.n.data >0.0)


    assert np.all(arr.p.data> 0.0)


def test_a_six_decade_asymmetric_junction_converges() :
    rows=pn_diode(Na=1e20,Nd=1e14,n_nodes=N_NODES,h_min= 1e-8)
    w=rows.with_bias(anode=1.0,cathode = 0.0)
    s= solve_bias_newton(w,guess=initial_state(rows))

    assert s.newton is not None

    assert s.newton.converged,   s.newton.message
    assert np.all( s.n.data >  0.0)
    assert np.all(s.p.data >0.0)

def test_the_row_scale_is_measured_at_the_iterate_not_at_the_guess() :
    u   =  pn_diode (Na =  1e20 ,  Nd = 1e14 , n_nodes  = N_NODES , h_min   = 1e-8)
    dat = u.with_bias(anode=1.0,cathode= 0.0)
    ok  =  initial_state( u)
    item =  TransportModels.for_device(dat)

    s = dat.mesh.h/ dat.scale.x_0
    c  =dat.mesh.volume /  dat.scale.x_0;  d =dat.net_doping_scaled.data
    c2 = solve_bias_newton(dat,models= item,guess= ok)
    assert c2.newton is not None and c2.newton.converged
    jj=residual_term_scales(
        s,c,pack(ok.psi.data,ok.n.data,ok.p.data),
        d,item.Dn,item.Dp,
    )
    z   = residual_term_scales(
        s,  c, c2.newton.x , d ,  item.Dn,   item.Dp
    )

    m   = float ( np.max(  z[ 1] )) /   float(  np.max(jj[1]  )  )
    assert m >1e4,f"electron term scale grew only {m:.3g}"

def  test_a_cold_newton_solve_does_not_report_the_guess_as_the_answer (  ) :

    hh  =  pn_diode(Na  = 1e17, Nd = 1e20,  length  =  2e-4 ,   n_nodes  = N_NODES, anode_voltage =   0.4)


    v  =  solve_bias_newton(hh  )
    b  = solve_bias(hh, max_iterations  =  500, update_tol  = 1e-8)

    assert v.newton is not None and v.newton.converged, v.newton.message
    assert b.gummel is not None and b.gummel.converged

    ok  = total_current(hh, b)
    assert  total_current(hh,   v  ) == pytest.approx (  ok,   rel  =   1e-6 )
