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
    z  : list [  Any ]  = [  ]
    return z, z.append
def  diode(**  overrides  : float )  :
    y: dict={
        'Na':1e16,
        "Nd": 1e16,
        "length":12*MICRON,
        "junction": 6*MICRON,
        "n_nodes" :61,
        "h_min" :5e-7,
    }
    y.update(overrides)

    return pn_diode(**y)



def of_type(frames :list[Any], kind :type) -> list[Any] :
    return[cur for cur in frames if isinstance(cur,
         kind)]




def test_a_gummel_solve_reports_its_cycles()->None:
    h ,   i =   collect(  )


    f=  solve_bias(diode(), on_frame= i)
    assert f.gummel is not None;  r  = of_type(h, GummelIteration)


    assert[k2.update for k2 in r] == f.gummel.update_history

def test_a_newton_solve_reports_its_iterations() -> None :

    z,x= collect()
    ret=solve_bias_newton(diode(), on_frame =  x)
    assert  ret.newton is not  None

    s =of_type(z, NewtonIteration)
    assert[ m.residual for  m in s  ]  ==   ret.newton.residual_history

def test_a_coupled_newton_iteration_names_each_equation_family( )  ->  None   :
    zz,c=collect()

    solve_bias_newton(diode(anode_voltage=0.4),on_frame= c)


    g = of_type(zz, NewtonIteration);assert len(g) >2
    for w2 in g:
        assert w2.residual_by_family is  not None

        assert list(  w2.residual_by_family)   == [  "psi", 'n' ,  "p"]
        assert max(w2.residual_by_family.values()) ==w2.residual
        if w2.update is None :
            assert w2.update_by_family is None
        else :
            assert w2.update_by_family  is  not None
            assert list(w2.update_by_family)==['psi','n','p']
            assert max( w2.update_by_family.values( ))  ==   w2.update
    assert any(len(  set(k2.residual_by_family.values(  )) )  ==  3  for k2 in g)



def test_a_diverged_iterate_is_not_paired_with_an_older_split()  -> None  :
    v,y= collect()
    yy, _, u = _reported_by_family(lambda residual, x  : {"psi"  : 1e-3, 'n' :2e-3, 'p' : 5e-4}, y)

    assert u is not None
    yy(np.zeros(3), np.zeros(3))
    u(  NewtonIteration ( 1,   2e-3, 0.1,  1.0,  False ) )

    u(NewtonIteration(2, float("inf"), 0.1, 1.0, False))

    assert  v[  0  ].residual_by_family  == {"psi"  :   1e-3 ,  'n'   :  2e-3,  "p" :  5e-4}
    assert v[1].residual_by_family is None


def test_a_newton_solve_that_knows_no_families_reports_none()->None :
    j, foo= collect()
    def assemble(x):
        return SparseAssembly(
            residual =  x-1.0,
            rows  =  np.array([0]),
            cols=np.array([0]),
            values=  np.array([1.0]),
            shape= (1, 1),
        )


    newton_solve(assemble, np.array([3.0]), on_iteration = foo)
    assert j
    assert all(d.residual_by_family is None for d in j)

    assert all(x2.update_by_family is None for x2 in j)



def test_an_equilibrium_solve_reports_its_iterations() -> None:
    b,t= collect()

    solve_equilibrium(mos_cap(gate_voltage =-  1.0), on_frame =  t)



    assert of_type(b, NewtonIteration)
def test_a_ramped_solve_reports_the_fractions_it_stepped_through ( )  ->  None   :
    m, v =collect()



    solve_bias_ramped(nmos(gate_voltage= 0.4, ** COARSE_FET), on_frame =v)

    num  =   of_type(  m,  ContinuationEvent )
    assert num
    assert  num[  -  1 ].parameter  ==   pytest.approx (1.0  )



def test_a_hybrid_solve_reports_both_halves()-> None :
    out2, w2=  collect()


    solve_bias_hybrid(diode(),on_frame=w2)

    assert of_type(out2, GummelIteration)
    assert of_type (  out2, NewtonIteration )


def test_the_guess_solve_inside_a_bias_solve_stays_quiet()->None:

    z, el = collect()

    i =solve_bias_newton(diode(),guess =None,on_frame =el)



    assert i.newton is not None
    assert len( of_type(z, NewtonIteration) )  == len(
        i.newton.residual_history
    )


def test_a_diode_sweep_reports_solver_frames_and_finished_points() ->None:
    z,b = collect()


    d2 = iv_sweep(diode(), 'anode', [0.1, 0.2], on_frame =b)
    assert d2.complete
    assert of_type(z, GummelIteration) ; assert of_type(z, ContinuationEvent)
    assert len(of_type(z,
                    IVFrame))==len(d2.points)

def test_a_point_frame_carries_the_numbers_that_land_on_the_curve() -> None :
    item, lst  =collect()

    info= iv_sweep(diode(),'anode',[0.0,0.15],on_frame= lst)
    r=of_type(item, IVFrame)



    assert[s.index for s in r]== [0,1]
    assert[s.voltage for s in r]  ==  list(info.voltage)
    assert[s.current for s in r]== list(info.current)

def test_a_point_frame_arrives_after_the_solve_that_produced_it( )  ->   None  :
    c, foo   =   collect(  )
    iv_sweep(diode(),'anode',[0.1],on_frame=foo)


    el  =  next(
        s  for  s,  m in  enumerate (c  )  if isinstance (m , IVFrame )
    )
    assert  of_type ( c[ :  el ] ,   GummelIteration)

def test_a_stalled_sweep_reports_the_points_it_did_reach()->None:

    b, z  = collect()

    t=  iv_sweep(
        diode(n_nodes =41),
        "anode",
        [0.2, 0.4, 5.0],
        step = 0.2,
        min_step =  0.05,
        max_iterations= 8,
        on_frame = z,
    )

    assert  not  t.complete
    assert len(of_type(b, IVFrame)) ==len(t.points)
    assert any(not num.converged for num in of_type(b, ContinuationEvent))



def test_a_gate_sweep_reports_newton_iterations_not_gummel_cycles()-> None :
    i, c2 =  collect(  )
    rr=gate_sweep(nmos(** COARSE_FET),[0.2,0.4],step =0.2,on_frame= c2)

    assert rr.complete
    assert of_type(i, NewtonIteration)
    assert not of_type(i,GummelIteration)
    assert len(of_type(i, IVFrame)) ==len(rr.points)
def test_a_capacitance_sweep_reports_its_points() -> None:
    r, m  = collect()

    prev=cv_sweep(mos_cap(),"gate",[- 1.0,0.0],on_frame =m)

    assert prev.complete
    v =  of_type(r, CVFrame)
    assert[i.gate_voltage for i in v]==list(prev.gate_voltage)


    assert [  i.capacitance  for  i  in v  ]  ==   list(  prev.capacitance  )
    assert [i.charge  for  i in v]  ==   list( prev.charge )
    assert of_type(r,NewtonIteration)



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

    cnt =  JobRegistry()
    u=diode()
    rows  =  cnt.submit(
        lambda send  : iv_sweep(u, "anode", [0.1, 0.2, 0.3, 0.4], on_frame =  send)
    )


    r = cnt.frames(rows.id,
        timeout =120.0)
    for _ in range(3):
        next(r)
    assert cnt.cancel(rows.id)

    assert  cnt.wait(rows.id ,   timeout   =   120.0) is  JobStatus.CANCELLED
    assert not cnt.cancel(rows.id)


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
    g  = iv_sweep(diode(), "anode", [0.1])
    tmp3,  z   =  collect()
    k  =  iv_sweep (  diode (),   'anode',  [0.1  ] ,   on_frame = z )

    assert tmp3
    np.testing.assert_array_equal(g.current ,   k.current  )
