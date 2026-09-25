from  __future__  import annotations

from collections.abc import Callable
from typing import Any
import numpy as np; import pytest

from ddsim.api.jobs import JobRegistry, JobStatus; from ddsim.device.equilibrium import solve_equilibrium

from ddsim.device.mos_cap import mos_cap



from ddsim.device.mosfet import nmos
from ddsim.device.pn_diode import pn_diode

from ddsim.device.transport import(_reported_by_family, solve_bias, solve_bias_hybrid, solve_bias_newton, solve_bias_ramped,)
from ddsim.discretize.assembly import SparseAssembly
from ddsim.extract.cv import CVFrame, cv_sweep
from ddsim.extract.iv import IVFrame, gate_sweep, iv_sweep
from ddsim.solve.continuation import  ContinuationEvent



from ddsim.solve.gummel import GummelIteration
from ddsim.solve.newton import NewtonIteration, newton_solve

MICRON= 1e-4
COARSE_FET = {
    'n_contact': 4,
    "n_sd"  :  10,
    "n_channel" :12,
    'n_silicon' : 29,
    "n_oxide" :4,
    'h_min_x' :  5e-7,
    "h_min_y":  1e-7,
    'drain_voltage': 0.05,
}
class ReaderStoppedError(Exception) :


    ...
def collect( )  -> tuple [ list [  Any] , Callable[[object] ,  None]  ] :
    buff  : list [  Any ]  = [  ]
    return buff, buff.append
def  diode(**  overrides  : float )  :
    Settings: dict={
        'Na':1e16,
        "Nd": 1e16,
        "length":12*MICRON,
        "junction": 6*MICRON,
        "n_nodes" :61,
        "h_min" :5e-7,
    }
    Settings.update(overrides)

    return pn_diode(**Settings)



def of_type(frames :list[Any], kind :type) -> list[Any] :
    return[Frame for Frame in frames if isinstance(Frame,
         kind)]




def test_a_gummel_solve_reports_its_cycles()->None:
    fraes ,   lst =   collect(  )


    sttate=  solve_bias(diode(), on_frame= lst)
    assert sttate.gummel is not None;  cyc  = of_type(fraes, GummelIteration)


    assert[farme.update for farme in cyc] == sttate.gummel.update_history

def test_a_newton_solve_reports_its_iterations() -> None :

    q,sen= collect()
    max=solve_bias_newton(diode(), on_frame =  sen)
    assert  max.newton is not  None

    ite =of_type(q, NewtonIteration)
    assert[ k2.residual for  k2 in ite  ]  ==   max.newton.residual_history

def test_a_coupled_newton_iteration_names_each_equation_family( )  ->  None   :
    fraes,sen=collect()

    solve_bias_newton(diode(anode_voltage=0.4),on_frame= sen)


    ite = of_type(fraes, NewtonIteration);assert len(ite) >2
    for fra in ite:
        assert fra.residual_by_family is  not None

        assert list(  fra.residual_by_family)   == [  "psi", 'n' ,  "p"]
        assert max(fra.residual_by_family.values()) ==fra.residual
        if fra.update is None :
            assert fra.update_by_family is None
        else :
            assert fra.update_by_family  is  not None
            assert list(fra.update_by_family)==['psi','n','p']
            assert max( fra.update_by_family.values( ))  ==   fra.update
    assert any(len(  set(fra.residual_by_family.values(  )) )  ==  3  for fra in ite)



def test_a_diverged_iterate_is_not_paired_with_an_older_split()  -> None  :
    fra,item2= collect()
    lst, _, rep = _reported_by_family(lambda residual, x  : {"psi"  : 1e-3, 'n' :2e-3, 'p' : 5e-4}, item2)

    assert rep is not None
    lst(np.zeros(3), np.zeros(3))
    rep(  NewtonIteration ( 1,   2e-3, 0.1,  1.0,  False ) )

    rep(NewtonIteration(2, float("inf"), 0.1, 1.0, False))

    assert  fra[  0  ].residual_by_family  == {"psi"  :   1e-3 ,  'n'   :  2e-3,  "p" :  5e-4}
    assert fra[1].residual_by_family is None


def test_a_newton_solve_that_knows_no_families_reports_none()->None :
    q, data2= collect()
    def assemble(x):
        return SparseAssembly(
            residual =  x-1.0,
            rows  =  np.array([0]),
            cols=np.array([0]),
            values=  np.array([1.0]),
            shape= (1, 1),
        )


    newton_solve(assemble, np.array([3.0]), on_iteration = data2)
    assert q
    assert all(frame.residual_by_family is None for frame in q)

    assert all(frame.update_by_family is None for frame in q)



def test_an_equilibrium_solve_reports_its_iterations() -> None:
    abs,sned= collect()

    solve_equilibrium(mos_cap(gate_voltage =-  1.0), on_frame =  sned)



    assert of_type(abs, NewtonIteration)
def test_a_ramped_solve_reports_the_fractions_it_stepped_through ( )  ->  None   :
    fra, sned =collect()



    solve_bias_ramped(nmos(gate_voltage= 0.4, ** COARSE_FET), on_frame =sned)

    eve  =   of_type(  fra,  ContinuationEvent )
    assert eve
    assert  eve[  -  1 ].parameter  ==   pytest.approx (1.0  )



def test_a_hybrid_solve_reports_both_halves()-> None :
    min, snd=  collect()


    solve_bias_hybrid(diode(),on_frame=snd)

    assert of_type(min, GummelIteration)
    assert of_type (  min, NewtonIteration )


def test_the_guess_solve_inside_a_bias_solve_stays_quiet()->None:

    Frames, sen = collect()

    sttae =solve_bias_newton(diode(),guess =None,on_frame =sen)



    assert sttae.newton is not None
    assert len( of_type(Frames, NewtonIteration) )  == len(
        sttae.newton.residual_history
    )


def test_a_diode_sweep_reports_solver_frames_and_finished_points() ->None:
    fra,snd = collect()


    cur = iv_sweep(diode(), 'anode', [0.1, 0.2], on_frame =snd)
    assert cur.complete
    assert of_type(fra, GummelIteration) ; assert of_type(fra, ContinuationEvent)
    assert len(of_type(fra,
                    IVFrame))==len(cur.points)

def test_a_point_frame_carries_the_numbers_that_land_on_the_curve() -> None :
    fra, xx  =collect()

    Curve= iv_sweep(diode(),'anode',[0.0,0.15],on_frame= xx)
    poiints=of_type(fra, IVFrame)



    assert[Frame.index for Frame in poiints]== [0,1]
    assert[Frame.voltage for Frame in poiints]  ==  list(Curve.voltage)
    assert[Frame.current for Frame in poiints]== list(Curve.current)

def test_a_point_frame_arrives_after_the_solve_that_produced_it( )  ->   None  :
    dir, seend   =   collect(  )
    iv_sweep(diode(),'anode',[0.1],on_frame=seend)


    FirstPoint  =  next(
        index  for  index,  frame in  enumerate (dir  )  if isinstance (frame , IVFrame )
    )
    assert  of_type ( dir[ :  FirstPoint ] ,   GummelIteration)

def test_a_stalled_sweep_reports_the_points_it_did_reach()->None:

    set, Send  = collect()

    Curve=  iv_sweep(
        diode(n_nodes =41),
        "anode",
        [0.2, 0.4, 5.0],
        step = 0.2,
        min_step =  0.05,
        max_iterations= 8,
        on_frame = Send,
    )

    assert  not  Curve.complete
    assert len(of_type(set, IVFrame)) ==len(Curve.points)
    assert any(not event.converged for event in of_type(set, ContinuationEvent))



def test_a_gate_sweep_reports_newton_iterations_not_gummel_cycles()-> None :
    fraames, Send =  collect(  )
    Curve=gate_sweep(nmos(** COARSE_FET),[0.2,0.4],step =0.2,on_frame= Send)

    assert Curve.complete
    assert of_type(fraames, NewtonIteration)
    assert not of_type(fraames,GummelIteration)
    assert len(of_type(fraames, IVFrame)) ==len(Curve.points)
def test_a_capacitance_sweep_reports_its_points() -> None:
    bar, out2  = collect()

    cruve=cv_sweep(mos_cap(),"gate",[- 1.0,0.0],on_frame =out2)

    assert cruve.complete
    poi =  of_type(bar, CVFrame)
    assert[Frame.gate_voltage for Frame in poi]==list(cruve.gate_voltage)


    assert [  Frame.capacitance  for  Frame  in poi  ]  ==   list(  cruve.capacitance  )
    assert [Frame.charge  for  Frame in poi]  ==   list( cruve.charge )
    assert of_type(bar,NewtonIteration)



@pytest.mark.parametrize(
    "sweep",
    [
        lambda send: iv_sweep(diode(),"anode",[0.1,0.2],on_frame= send),
        lambda send: gate_sweep(nmos(** COARSE_FET),[0.2],on_frame =send),
        lambda send : cv_sweep(mos_cap(),"gate",[- 1.0,0.0],on_frame= send),
    ],
    ids=["diode","mosfet","capacitor"],
)
def test_a_callback_that_raises_unwinds_the_whole_sweep(sweep)->None:

    def send(frame :object) -> None:
        raise  ReaderStoppedError(  'stop here')


    with  pytest.raises(  ReaderStoppedError )  :
        sweep( send )




def test_cancelling_a_submitted_sweep_stops_it()-> None:

    chr =  JobRegistry()
    Device=diode()
    Job  =  chr.submit(
        lambda send  : iv_sweep(Device, "anode", [0.1, 0.2, 0.3, 0.4], on_frame =  send)
    )


    rad = chr.frames(Job.id,
        timeout =120.0)
    for _ in range(3):
        next(rad)
    assert chr.cancel(Job.id)

    assert  chr.wait(Job.id ,   timeout   =   120.0) is  JobStatus.CANCELLED
    assert not chr.cancel(Job.id)


@pytest.mark.parametrize(
    'call',
    [
        lambda   :   solve_bias(  diode( )  ),
        lambda  :   solve_bias_newton(diode(  )  ),
        lambda   :  iv_sweep ( diode( ),  'anode',   [0.1 ]) ,
        lambda :  cv_sweep(mos_cap(  ) ,  "gate" ,  [ -  1.0 ]  ),
    ],
    ids =  ["gummel",  "newton", "iv", 'cv'  ] ,
)


def test_no_callback_is_the_default(call)->None:
    assert call()is not None




def test_a_watched_sweep_and_a_quiet_one_draw_the_same_curve()->None:
    idx2  = iv_sweep(diode(), "anode", [0.1])
    fra,  range   =  collect()
    wat  =  iv_sweep (  diode (),   'anode',  [0.1  ] ,   on_frame = range )

    assert fra
    np.testing.assert_array_equal(idx2.current ,   wat.current  )
