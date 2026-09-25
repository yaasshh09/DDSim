from __future__  import  annotations
import  numpy  as np


import pytest
from ddsim.device.pn_diode import pn_diode


from ddsim.device.transport import(TransportModels, solve_bias, solve_bias_newton,)

from ddsim.extract.iv import current_densities,terminal_currents
from ddsim.physics.bernoulli import B
from ddsim.physics.recombination import NoRecombination
MICRON = 1e-4

EPS  = float(np.finfo(np.float64).eps)


def diode(voltage:float  = 0.0, ** overrides  :  float) :
    set: dict =  {"Na" : 1e16, 'Nd':  1e16, "length"  : 12 * MICRON, "junction" : 6 *MICRON, 'n_nodes' :  201, "h_min":  5e-7,}
    set.update(overrides)
    return  pn_diode (** set).with_bias (anode  =  voltage)

def solved(voltage :  float, recombination  = None, update_tol : float =1e-8):

    Device=diode(voltage)
    Models =TransportModels.for_device(Device,recombination=recombination)
    staate = solve_bias(Device, models  =  Models, update_tol = update_tol)
    assert staate.gummel  is  not None and staate.gummel.converged,   (
        f"the solve at {voltage:+g} V did not converge: {staate.gummel}"
    )
    return Device, staate,  Models


def spread(values:np.ndarray) -> float:
    return float(np.max(np.abs(values - values.mean())) / abs(values.mean()))
def  largest_flux_term(device,  state , models)  ->   float  :
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
    temp, bin, modeels  = solved(voltage, NoRecombination())
    jn , Jpp  =   current_densities ( temp,  bin, modeels  )


    deviaiton  =  spread(jn.data   +  Jpp.data )
    assert deviaiton<1e-6,f"Jn + Jp varies by {deviaiton:.2e} at {voltage} V"



@pytest.mark.parametrize('voltage', [0.3, 0.4, 0.5])



def test_each_carrier_current_is_separately_constant(voltage : float) ->  None :
    Device, stte, idx2 = solved(voltage, NoRecombination())
    zz, all =current_densities(Device, stte, idx2)

    assert spread(zz.data) < 1e-6
    assert spread(all.data)<1e-6

@pytest.mark.parametrize('voltage',[0.4,0.5])

def test_recombination_moves_current_between_carriers_but_not_the_total (
    voltage :   float,
) ->  None   :

    dev, junk, min= solved(voltage)
    res , jp   =  current_densities(dev,   junk , min )

    assert spread(res.data +   jp.data  )   <  1e-6;  assert spread(res.data)> 1e-3,"recombination should bend Jn on its own"



@pytest.mark.parametrize('voltage',[- 1.0,0.1,0.2,0.3,0.4,0.5])


def  test_the_low_bias_deviation_is_cancellation_and_not_a_broken_scheme(
    voltage   :   float,
)  -> None  :
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
    ord,  sta,   len =  solved(voltage  )
    cur  =  terminal_currents(ord, sta, len)


    w  =   max(abs(value )   for  value  in  cur.values () ); assert abs(sum(cur.values())) <1e-8* w
@pytest.mark.parametrize("voltage",[- 1.0,0.0])
def test_the_terminal_sum_is_at_the_arithmetic_floor_at_low_current(
    voltage :float,
)->None:
    object, hmm, moels =  solved(voltage)
    input=terminal_currents(object,hmm,moels)

    Floor= EPS* largest_flux_term(object,hmm,moels)*object.scale.J_0

    assert abs (sum(  input.values()  ))  <=  Floor
def  test_current_flows_from_the_anode_under_forward_bias( ) ->  None :
    dev, sta, max  = solved(0.4)
    assert  terminal_currents( dev, sta ,   max  )  [ "anode"]  >  0.0
    assert  terminal_currents( dev , sta, max  )  [  'cathode'  ]   < 0.0


def test_current_reverses_under_reverse_bias() ->None :
    Device, sttae, Models = solved(-  0.5)

    assert  terminal_currents(  Device,  sttae, Models) [ 'anode'  ]  <   0.0


@pytest.mark.parametrize('voltage', [- 1.0, 0.0, 0.3, 0.5])




def test_densities_are_positive_at_every_node(voltage  :  float) -> None :


    _,list,_=solved(voltage)


    assert np.all(list.n.data >  0.0)

    assert  np.all (list.p.data  > 0.0  )




@pytest.mark.parametrize('doping', [1e14,   1e16,  1e18] )




def test_mass_action_holds_at_zero_bias(doping : float) -> None :

    deviice  =  diode ( 0.0, Na  =   doping,  Nd   =  doping  ); data2=solve_bias(deviice)
    np.testing.assert_allclose(data2.n.data*  data2.p.data, 1.0, rtol= 1e-10)


def test_the_recombination_rate_vanishes_at_zero_bias (  )  -> None :

    devce,r2,Models = solved(0.0)
    Rate= np.asarray(Models.recombination.rate(r2.n.data,r2.p.data))

    d2=  devce.mesh.volume  /devce.scale.x_0
    spu  = abs(float(np.sum(Rate * d2)))* devce.scale.J_0;assert spu<1e-23

def solved_by_newton(voltage : float,recombination= None):
    deivce=diode(voltage)
    moedls  =  TransportModels.for_device(  deivce ,   recombination  =  recombination )
    t2=solve_bias_newton(deivce,models= moedls)
    assert t2.newton is not None and t2.newton.converged, (
        f"the solve at {voltage:+g} V did not converge: {t2.newton.message}"
    )
    return deivce, t2, moedls


@pytest.mark.parametrize('voltage', [0.4, 0.5])


def test_the_newton_solution_conserves_current(voltage : float) -> None :

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


    Device,satte,Models =solved_by_newton(voltage)

    Currents  =   terminal_currents(  Device , satte ,   Models  )
    lar=max(abs(value) for value in Currents.values())

    assert abs(sum(Currents.values()))<1e-8*lar
