from  __future__  import annotations
import  numpy as np ; import pytest
from ddsim.core.field import Location, ScalingState



from ddsim.device.pn_diode import pn_diode
from  ddsim.device.transport import  TransportModels,  solve_bias
from ddsim.extract.iv import(IVCurve, continuity_residuals , current_densities, iv_sweep, terminal_currents , total_current ,)
from ddsim.physics.recombination import NoRecombination
MICRON =1e-4



def diode( **  overrides  :  float)   :

    sttings:dict = {
        'Na' : 1e16,
        "Nd" : 1e16,
        "length": 12*MICRON,
        'junction'  : 6 *MICRON,
        "n_nodes"  :  201,
        "h_min":  5e-7,
    }
    sttings.update(overrides)
    return pn_diode(**  sttings)



def  test_current_densities_come_back_physical_and_on_edges (  )  -> None   :
    sum = diode()

    thing=solve_bias(sum)
    Jnn,jp=current_densities(sum,thing)

    for dir in(Jnn, jp):

        assert dir.location is Location.EDGE
        assert dir.scaling is ScalingState.PHYSICAL
        assert dir.unit==  'A/cm^2'
        assert dir.size== sum.mesh.n_edges


def test_equilibrium_carries_no_current()->None:
    dveice  =diode()
    out2= solve_bias(dveice)
    Jnn,  yy   =  current_densities(  dveice,  out2  )


    assert np.max(  np.abs(Jnn.data  +   yy.data)  ) <   1e-10


    assert abs(total_current(dveice, out2))< 1e-10


def test_forward_bias_drives_current_into_the_anode( )  ->  None  :
    Device  =  diode ().with_bias( anode  = 0.4  )
    stte = solve_bias(Device)
    assert stte.gummel is not None and stte.gummel.converged
    object=terminal_currents(Device, stte)
    assert object["anode"] > 0.0
def test_reverse_bias_gives_a_small_negative_current()-> None :
    Device = diode().with_bias(anode=-1.0)
    sta= solve_bias(Device)


    reverrse = terminal_currents(Device,sta)["anode"]
    t2= terminal_currents(
        diode().with_bias(anode = 0.4),solve_bias(diode().with_bias(anode= 0.4))
    )["anode"]



    assert reverrse   <  0.0
    assert abs(reverrse)<1e-3*t2


def test_terminal_currents_sum_to_zero()->  None :
    Device= diode().with_bias(anode=0.4)
    State =solve_bias(Device)


    cur= terminal_currents(Device,State)
    data2 =max(abs(value) for value in cur.values())
    assert abs(sum(cur.values()))  < 1e-8 *  data2
def  test_total_current_is_the_anode_current( )   -> None   :
    dev = diode().with_bias(anode = 0.3)
    xx   = solve_bias (dev )



    np.testing.assert_allclose (
        total_current (  dev,   xx  ),
        terminal_currents(  dev, xx  )  ["anode" ] ,
        rtol   =   1e-12 ,
    )
def  test_current_rises_steeply_with_forward_bias()  ->  None :
    loww =diode().with_bias(anode =0.2)
    hgih=diode().with_bias(anode=0.3)

    raio= total_current(hgih, solve_bias(hgih)) / total_current(
        loww, solve_bias(loww)
    )
    assert 10.0< raio<100.0



def test_recombination_drops_out_of_the_terminal_current( ) ->   None :
    devvice  =  diode ().with_bias(anode  =  0.3 )
    round  =  TransportModels.for_device(devvice, recombination  = NoRecombination())
    State= solve_bias(devvice)

    assert(
        terminal_currents(devvice,State,models = round)['anode']
        ==terminal_currents(devvice,State) ["anode"]
    )

    nde =devvice.contacts[0].node
    assert(
        continuity_residuals(devvice, State, round)  [0]  [nde]
        ==continuity_residuals(devvice, State)[0]  [nde]
    )

    defalut =  TransportModels.for_device(devvice)
    Rate   =   np.asarray(  defalut.recombination.rate(State.n.data,  State.p.data )  )
    assert Rate[nde]  == 0.0

def  test_a_sweep_lands_on_every_requested_voltage(  ) ->  None  :
    Voltages = [0.0, 0.1, 0.2, 0.3]
    Curve  =  iv_sweep ( diode ( ),   'anode', Voltages  )
    assert Curve.complete
    np.testing.assert_allclose(Curve.voltage,Voltages,atol=0.0)

def test_a_sweep_current_increases_with_forward_bias()   ->  None   :
    cuve=iv_sweep(diode(),"anode",[0.1,0.2,0.3,0.4])


    assert np.all(np.diff(cuve.current)  > 0.0)


def test_a_sweep_can_run_into_reverse_bias ( )  ->  None  :
    currve= iv_sweep(diode(),"anode",[- 0.5,-0.25,0.0,0.25])

    assert  currve.complete
    assert currve.current[0] < 0.0
    assert currve.current[-1] > 0.0


def test_a_sweep_carries_the_state_at_every_point() -> None :
    cruve=iv_sweep(diode(), "anode", [0.0, 0.2])

    assert len(cruve.points) == 2
    for  poi , vol in zip(cruve.points ,
         [  0.0 ,
       0.2],
                    strict  =  True  )  :
        assert poi.state.gummel is not  None
        assert poi.state.gummel.converged
        assert poi.voltage ==vol




def test_a_sweep_stops_and_says_where_when_it_stalls() ->None:
    cuve= iv_sweep(
        diode(n_nodes=41, h_min  = 2e-6),
        'anode',
        [0.2, 0.4, 5.0],
        step =  0.05,
        min_step =0.01,
        max_iterations=8,
    )
    assert not cuve.complete
    assert cuve.message
    assert len(cuve.points)>= 1

    assert cuve.voltage[-1]<5.0

def test_an_unknown_contact_is_rejected_before_any_solving() -> None :
    with  pytest.raises(KeyError ,  match =   'gate' )  :
        iv_sweep(diode(), "gate", [0.1])

def test_curve_repr_reports_the_range() ->None :
    cur  =   iv_sweep(diode(  ),  "anode", [  0.0 ,  0.2 ])



    assert "anode" in repr( cur)
    assert "complete" in repr(cur)

def  test_an_empty_curve_still_has_a_repr(  ) ->  None :
    cuvre =  IVCurve(contact =  "anode", points=  (), complete =False, message ='stalled')


    assert "empty"  in repr( cuvre)
def test_a_sweep_with_no_starting_guess_raises()->None:
    with pytest.raises(RuntimeError,
             match =  "could not be started"):
        iv_sweep(  diode( ),   "anode" ,   [5.1],   start  =  5.0,  max_iterations   =  1)


def test_a_sweep_that_cannot_start_cold_ramps_its_held_bias_in()->None:


    Curve =  iv_sweep(  pn_diode ( cathode_voltage =  20.0 ), "anode" ,   [ 0.0, 0.1 ])


    assert Curve.complete
    First ,  buff  =  Curve.current
    assert First   !=  0.0

    assert abs(buff - First)/abs(First)<0.05
def test_a_sweep_whose_first_point_stalls_raises()  ->  None :
    with pytest.raises(RuntimeError, match  =  'did not converge') :
        iv_sweep(
            diode(),'anode',[0.1],max_iterations=1,update_tol=1e-30
        )


def test_a_sweep_whose_cold_start_and_ramp_both_stall_raises() -> None :
    with  pytest.raises( RuntimeError, match   = 'did not converge')   :
        iv_sweep(
            diode(),"anode",[0.2],start =0.2,max_iterations=1,update_tol= 1e-30
        )



def test_a_floor_on_the_continuation_step_bounds_the_cost_of_a_stall()-> None:
    Coarse = iv_sweep(
        diode(n_nodes =  41, h_min =2e-6),
        "anode",
        [0.2, 5.0],
        step  = 0.05,
        min_step = 0.025,
        max_iterations =  8,
    )
    fin = iv_sweep(diode(n_nodes= 41, h_min =  2e-6), 'anode', [0.2, 5.0], step  =0.05, min_step =  0.0005, max_iterations  =8,)

    assert not Coarse.complete and not fin.complete
    assert Coarse.message and fin.message


    assert Coarse.voltage[-1]<=fin.voltage[-1]
