'''Telemetry from the public sweeps down to the solvers, phases/PHASE-7.md.

Three hooks already exist one level down: newton_solve reports an iteration,
continue_to reports an attempt, gummel_solve reports a cycle. Nothing reached
them, because the browser does not call a solver. It calls iv_sweep,
gate_sweep or cv_sweep and everything below that is private.

So one optional argument crosses the layers between: `on_frame`. It is a
single pipe rather than three, because the three frame types are already
distinguishable by their own type and a reader that has to be handed three
callbacks has to implement three. Every frame carries scalars only, which is
the inertness argument NewtonIteration states and the reason IVPoint is not
itself a frame: a point carries the state the next point continues from, and
handing a callback a reference to that is handing it the next solve.

What is asserted here is the wiring. That the wiring changes no number is the
separate claim in tests/invariant/test_telemetry_inert.py, and it is the one
phases/PHASE-7.md lists as an acceptance criterion.
'''



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
'''One micron [cm].'''
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
"""The same MOSFET tests/unit/test_surface_mobility.py uses, small enough to
solve several times in a unit test."""
class ReaderStoppedError(Exception) :


    '''Whatever a caller raises inside its own callback.'''
def collect( )  -> tuple [ list [  Any] , Callable[[object] ,  None]  ] :
    """A frame list and the callback that fills it."""
    buff  : list [  Any ]  = [  ]
    return buff, buff.append
def  diode(**  overrides  : float )  :
    '''A coarse version of the Phase 2 test diode.'''
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
    """solve_bias is the Gummel path, so the frames from it are cycles."""
    fraes ,   lst =   collect(  )


    sttate=  solve_bias(diode(), on_frame= lst)
    assert sttate.gummel is not None;  cyc  = of_type(fraes, GummelIteration)


    assert[farme.update for farme in cyc] == sttate.gummel.update_history

def test_a_newton_solve_reports_its_iterations() -> None :

    '''Bit for bit against the history the solve was judged on. A telemetry
    number that disagrees with that one is worse than no telemetry.'''
    q,sen= collect()
    max=solve_bias_newton(diode(), on_frame =  sen)
    assert  max.newton is not  None

    ite =of_type(q, NewtonIteration)
    assert[ k2.residual for  k2 in ite  ]  ==   max.newton.residual_history

def test_a_coupled_newton_iteration_names_each_equation_family( )  ->  None   :
    '''phases/PHASE-7.md item 5: residual and update per equation family, so
    a stalled solve can say whether Poisson or a continuity equation stalled.
    The split has to be the scalar the solve was judged on, taken apart: its
    largest family is that scalar exactly, not approximately.'''
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
    """newton_solve reports a diverged iterate as infinite without measuring
    it. The split from the evaluation before would otherwise ride along, and
    the browser would name a stalled family from a residual that was finite."""
    fra,item2= collect()
    lst, _, rep = _reported_by_family(lambda residual, x  : {"psi"  : 1e-3, 'n' :2e-3, 'p' : 5e-4}, item2)

    assert rep is not None
    lst(np.zeros(3), np.zeros(3))
    rep(  NewtonIteration ( 1,   2e-3, 0.1,  1.0,  False ) )

    rep(NewtonIteration(2, float("inf"), 0.1, 1.0, False))

    assert  fra[  0  ].residual_by_family  == {"psi"  :   1e-3 ,  'n'   :  2e-3,  "p" :  5e-4}
    assert fra[1].residual_by_family is None


def test_a_newton_solve_that_knows_no_families_reports_none()->None :
    """newton_solve knows nothing about semiconductors. The split is the
    coupled transport solve's to give, and a bare solve does not invent one."""
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
    """The C-V path is equilibrium Poisson at every point, so without this
    hook a capacitance sweep has nothing to show between points."""
    abs,sned= collect()

    solve_equilibrium(mos_cap(gate_voltage =-  1.0), on_frame =  sned)



    assert of_type(abs, NewtonIteration)
def test_a_ramped_solve_reports_the_fractions_it_stepped_through ( )  ->  None   :
    '''The ramp is continuation in the bias, so its attempts are the same
    ContinuationEvent a sweep between points reports.'''
    fra, sned =collect()



    solve_bias_ramped(nmos(gate_voltage= 0.4, ** COARSE_FET), on_frame =sned)

    eve  =   of_type(  fra,  ContinuationEvent )
    assert eve
    assert  eve[  -  1 ].parameter  ==   pytest.approx (1.0  )



def test_a_hybrid_solve_reports_both_halves()-> None :
    """Gummel to reach the basin, then Newton. A stream with only one of them
    in it would show a gap where the prelude ran."""
    min, snd=  collect()


    solve_bias_hybrid(diode(),on_frame=snd)

    assert of_type(min, GummelIteration)
    assert of_type (  min, NewtonIteration )


def test_the_guess_solve_inside_a_bias_solve_stays_quiet()->None:

    """A cold solve_bias_newton builds its guess with equilibrium Poisson,
    whose residual is a one unknown per node quantity that shares no scale
    with the coupled residual plotted beside it. Deliberately not reported:
    the frames a caller gets are the frames of the solve it asked for."""
    Frames, sen = collect()

    sttae =solve_bias_newton(diode(),guess =None,on_frame =sen)



    assert sttae.newton is not None
    assert len( of_type(Frames, NewtonIteration) )  == len(
        sttae.newton.residual_history
    )


def test_a_diode_sweep_reports_solver_frames_and_finished_points() ->None:
    """All three streams phases/PHASE-7.md asks for, on the path the browser
    takes for a diode."""
    fra,snd = collect()


    cur = iv_sweep(diode(), 'anode', [0.1, 0.2], on_frame =snd)
    assert cur.complete
    assert of_type(fra, GummelIteration) ; assert of_type(fra, ContinuationEvent)
    assert len(of_type(fra,
                    IVFrame))==len(cur.points)

def test_a_point_frame_carries_the_numbers_that_land_on_the_curve() -> None :
    """Bit for bit. The curve the browser draws from the frames has to be the
    curve pytest gets from the return value, or the plot is of nothing."""
    fra, xx  =collect()

    Curve= iv_sweep(diode(),'anode',[0.0,0.15],on_frame= xx)
    poiints=of_type(fra, IVFrame)



    assert[Frame.index for Frame in poiints]== [0,1]
    assert[Frame.voltage for Frame in poiints]  ==  list(Curve.voltage)
    assert[Frame.current for Frame in poiints]== list(Curve.current)

def test_a_point_frame_arrives_after_the_solve_that_produced_it( )  ->   None  :
    """Otherwise the residual plot for a point would draw after the point is
    already on the curve, which is backwards from what happened."""
    dir, seend   =   collect(  )
    iv_sweep(diode(),'anode',[0.1],on_frame=seend)


    FirstPoint  =  next(
        index  for  index,  frame in  enumerate (dir  )  if isinstance (frame , IVFrame )
    )
    assert  of_type ( dir[ :  FirstPoint ] ,   GummelIteration)

def test_a_stalled_sweep_reports_the_points_it_did_reach()->None:

    """A sweep that gives up is a measurement. The frames stop where the
    curve stops rather than reporting a point that was never solved."""
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
    """A MOSFET runs the coupled path, which has no Gummel in it at all. A
    stream carrying cycles here would mean the browser is watching a solve
    the device cannot have run."""
    fraames, Send =  collect(  )
    Curve=gate_sweep(nmos(** COARSE_FET),[0.2,0.4],step =0.2,on_frame= Send)

    assert Curve.complete
    assert of_type(fraames, NewtonIteration)
    assert not of_type(fraames,GummelIteration)
    assert len(of_type(fraames, IVFrame)) ==len(Curve.points)
def test_a_capacitance_sweep_reports_its_points() -> None:
    """The third device class, and the one whose points are a capacitance
    rather than a current."""
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
    """This is what cancellation rides on. api/jobs.py raises inside send and
    expects the stack to come apart; a sweep that swallowed it would leave a
    cancelled job running to the end of a MOSFET ladder."""

    def send(frame :object) -> None:
        raise  ReaderStoppedError(  'stop here')


    with  pytest.raises(  ReaderStoppedError )  :
        sweep( send )




def test_cancelling_a_submitted_sweep_stops_it()-> None:
    '''The acceptance criterion, end to end: a real registry, a real sweep,
    and a cancel that lands while the solver is inside a bias point.'''

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
    """Every hook is optional and off. A solver that needed a callback to run
    would have made the CLI depend on the browser."""
    assert call()is not None




def test_a_watched_sweep_and_a_quiet_one_draw_the_same_curve()->None:
    """The cheap version of the inertness claim, on one device. The bit for
    bit version across all three classes is in tests/invariant."""
    idx2  = iv_sweep(diode(), "anode", [0.1])
    fra,  range   =  collect()
    wat  =  iv_sweep (  diode (),   'anode',  [0.1  ] ,   on_frame = range )

    assert fra
    np.testing.assert_array_equal(idx2.current ,   wat.current  )
