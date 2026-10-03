from __future__ import annotations
import numpy as np
import pytest
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import(TransportModels, solve_bias, solve_bias_newton,)
from ddsim.extract.iv import current_densities,terminal_currents
from ddsim.physics.bernoulli import B
from ddsim.physics.recombination import NoRecombination
MICRON = 1e-4

EPS  = float(np.finfo(np.float64).eps)


def diode(voltage:float  = 0.0, ** overrides  :  float) :
    tmp: dict =  {"Na" : 1e16, 'Nd':  1e16, "length"  : 12 * MICRON, "junction" : 6 *MICRON, 'n_nodes' :  201, "h_min":  5e-7,}
    tmp.update(overrides)
    return  pn_diode (** tmp).with_bias (anode  =  voltage)

def solved(voltage :  float, recombination  = None, update_tol : float =1e-8):

    h=diode(voltage)
    c2 =TransportModels.for_device(h,recombination=recombination)
    j = solve_bias(h, models  =  c2, update_tol = update_tol)
    assert j.gummel  is  not None and j.gummel.converged,   (
        f"the solve at {voltage:+g} V did not converge: {j.gummel}"
    )
    return h, j,  c2


def spread(values:np.ndarray) -> float:
    return float(np.max(np.abs(values - values.mean())) / abs(values.mean()))
def  largest_flux_term(device,  state , models)  ->   float  :
    ys =  device.mesh.h/  device.scale.x_0
    v=np.diff(state.psi.data)


    c =  np.asarray( B(v )  )
    a   = np.asarray(  B( -   v  ) )
    x = [
        (models.Dn/ys) * c *state.n.data[1 :],
        (models.Dn /  ys)*a * state.n.data[:-  1],
        (models.Dp / ys) * c*state.p.data[:- 1],
        (models.Dp  /ys) *a * state.p.data[1 :],
    ]
    return float(max(np.max(np.abs(ret)) for ret in x))


@pytest.mark.parametrize( "voltage",   [ 0.3 , 0.4, 0.5  ] )

def  test_total_current_is_constant_across_the_device(  voltage :  float ) ->  None   :
    x, w, a  = solved(voltage, NoRecombination())
    it , thing  =   current_densities ( x,  w, a  )


    j  =  spread(it.data   +  thing.data )
    assert j<1e-6,f"Jn + Jp varies by {j:.2e} at {voltage} V"



@pytest.mark.parametrize('voltage', [0.3, 0.4, 0.5])



def test_each_carrier_current_is_separately_constant(voltage : float) ->  None :
    cc, jj, tt = solved(voltage, NoRecombination())
    h, xs =current_densities(cc, jj, tt)

    assert spread(h.data) < 1e-6
    assert spread(xs.data)<1e-6

@pytest.mark.parametrize('voltage',[0.4,0.5])

def test_recombination_moves_current_between_carriers_but_not_the_total (
    voltage :   float,
) ->  None   :

    row, c, arr= solved(voltage)
    z , mm   =  current_densities(row,   c , arr )

    assert spread(z.data +   mm.data  )   <  1e-6;  assert spread(z.data)> 1e-3,"recombination should bend Jn on its own"



@pytest.mark.parametrize('voltage',[- 1.0,0.1,0.2,0.3,0.4,0.5])


def  test_the_low_bias_deviation_is_cancellation_and_not_a_broken_scheme(
    voltage   :   float,
)  -> None  :
    mm, t2, m  = solved(voltage, NoRecombination())
    w2,i=current_densities(mm,
          t2,
             m)
    obj=w2.data+i.data
    a2= abs(obj.mean())/ mm.scale.J_0
    y=largest_flux_term(mm,t2,m)/ a2

    assert spread( obj)   <  3.0  *   EPS *  y


@pytest.mark.parametrize("voltage", [0.3, 0.4, 0.5])




def test_terminal_currents_sum_to_zero(  voltage : float)   ->  None   :
    f,  tt,   b2 =  solved(voltage  )
    s2  =  terminal_currents(f, tt, b2)


    c  =   max(abs(r )   for  r  in  s2.values () ); assert abs(sum(s2.values())) <1e-8* c
@pytest.mark.parametrize("voltage",[- 1.0,0.0])
def test_the_terminal_sum_is_at_the_arithmetic_floor_at_low_current(
    voltage :float,
)->None:
    z, v, jj =  solved(voltage)
    h=terminal_currents(z,v,jj)

    ret= EPS* largest_flux_term(z,v,jj)*z.scale.J_0

    assert abs (sum(  h.values()  ))  <=  ret
def  test_current_flows_from_the_anode_under_forward_bias( ) ->  None :
    r, a2, kk  = solved(0.4)
    assert  terminal_currents( r, a2 ,   kk  )  [ "anode"]  >  0.0
    assert  terminal_currents( r , a2, kk  )  [  'cathode'  ]   < 0.0


def test_current_reverses_under_reverse_bias() ->None :
    a, cnt, w = solved(-  0.5)

    assert  terminal_currents(  a,  cnt, w) [ 'anode'  ]  <   0.0


@pytest.mark.parametrize('voltage', [- 1.0, 0.0, 0.3, 0.5])




def test_densities_are_positive_at_every_node(voltage  :  float) -> None :


    _,c,_=solved(voltage)


    assert np.all(c.n.data >  0.0)

    assert  np.all (c.p.data  > 0.0  )




@pytest.mark.parametrize('doping', [1e14,   1e16,  1e18] )




def test_mass_action_holds_at_zero_bias(doping : float) -> None :

    w  =  diode ( 0.0, Na  =   doping,  Nd   =  doping  ); t2=solve_bias(w)
    np.testing.assert_allclose(t2.n.data*  t2.p.data, 1.0, rtol= 1e-10)


def test_the_recombination_rate_vanishes_at_zero_bias (  )  -> None :

    s,u,c2 = solved(0.0)
    s2= np.asarray(c2.recombination.rate(u.n.data,u.p.data))

    arr=  s.mesh.volume  /s.scale.x_0
    z  = abs(float(np.sum(s2 * arr)))* s.scale.J_0;assert z<1e-23

def solved_by_newton(voltage : float,recombination= None):
    item=diode(voltage)
    c  =  TransportModels.for_device(  item ,   recombination  =  recombination )
    res=solve_bias_newton(item,models= c)
    assert res.newton is not None and res.newton.converged, (
        f"the solve at {voltage:+g} V did not converge: {res.newton.message}"
    )
    return item, res, c


@pytest.mark.parametrize('voltage', [0.4, 0.5])


def test_the_newton_solution_conserves_current(voltage : float) -> None :

    k, r, d  = solved_by_newton(voltage, NoRecombination())
    out2,x =current_densities(k,r,d)

    dd=spread(out2.data+x.data)
    assert dd<1e-6,f"Jn + Jp varies by {dd:.2e} at {voltage} V"

@pytest.mark.parametrize("voltage",
       [0.5,
   0.6,
     0.8,
                1.0])


def test_newton_and_gummel_report_the_same_terminal_current (
    voltage : float,
)  ->  None   :
    d, d2, w2=solved(voltage, update_tol  = 1e-10)
    t,kk,_=solved_by_newton(voltage)

    tmp3 = terminal_currents(d, d2, w2);j =terminal_currents(t,kk,w2)
    for ii,out in tmp3.items():
        assert j[ii] ==  pytest.approx(  out,  rel   =  1e-10 ),   (
            f"{ii} current differs at {voltage} V: "
            f"gummel {out:.12e}, newton {j[ii]:.12e}"
        )

@pytest.mark.parametrize('voltage', [0.5, 0.8, 1.0])



def  test_the_newton_terminal_currents_sum_to_zero(  voltage  : float )  ->   None :


    w,x2,z =solved_by_newton(voltage)

    tt  =   terminal_currents(  w , x2 ,   z  )
    t=max(abs(h) for h in tt.values())

    assert abs(sum(tt.values()))<1e-8*t
