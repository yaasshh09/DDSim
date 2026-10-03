from __future__ import annotations
from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass
import numpy as np, numpy.typing as npt
import pytest
from ddsim.solve.linear import SparseLU
from ddsim.solve.newton import NewtonIteration, NewtonResult, newton_solve


@dataclass(frozen =  True)

class  System   :

    residual : npt.NDArray[np.float64]
    rows : npt.NDArray[np.int64] ; cols : npt.NDArray[np.int64]

    values:npt.NDArray[np.float64] ; shape : tuple[int, int]
def  diagonal_system( residual  :  np.ndarray , derivative  : np.ndarray )  ->  System :


    n  = residual.size

    t =np.arange(n,dtype=np.int64)
    return System(
        residual = residual, rows=t, cols=t, values  = derivative, shape = (n, n)
    )



def square_root_problem(target  :np.ndarray) :

    def assemble(x :np.ndarray) ->  System :
        return diagonal_system(x*x- target,2.0*x)
    return assemble

def exponential_problem(target : float):


    def assemble( x :  np.ndarray)   ->   System   :
        return diagonal_system(np.exp(x) -target, np.exp(x))
    return assemble

def test_one_newton_step_solves_a_linear_system_exactly() ->None  :

    def assemble( x  :  np.ndarray )   ->  System  :
        return diagonal_system(3.0* x -6.0, np.full(x.size, 3.0))
    b  = newton_solve(assemble,  np.zeros( 4  )) ; assert b.converged
    assert b.residual_history[1] ==  0.0, 'one step must land on the root'
    assert b.iterations ==2
    np.testing.assert_allclose(b.x, 2.0, rtol  =1e-14)
def test_solves_a_nonlinear_system()->None :
    a=np.array([2.0,9.0,16.0])

    nxt = newton_solve(square_root_problem(a),np.full(3,1.0));  assert  nxt.converged
    np.testing.assert_allclose( nxt.x,  np.sqrt (  a  ) ,   rtol  =  1e-12  )



def  test_converges_quadratically(  ) -> None :
    v2  = newton_solve (square_root_problem( np.array([  2.0] ) ),   np.array([1.0 ] )  )
    y = np.array(v2.residual_history)
    cc  = y[y> 1e-14]



    assert len(cc) >= 3, "need a few iterations to see the tail"
    for z, g in zip(cc[:-1], cc[1 :], strict  =True) :
        assert g   <=   10.0  * z   **  2 + 1e-15


def  test_reports_the_residual_history (  )   ->  None  :
    k =  newton_solve ( square_root_problem(  np.array( [2.0]) ), np.array ([  1.0 ]  ))
    assert len(k.residual_history)==k.iterations +1
    assert k.residual_history[0]  > k.residual_history[- 1]

def test_reports_the_update_history()->None:
    f  =newton_solve(square_root_problem(np.array([2.0])), np.array([1.0]))
    assert  len( f.update_history )  == f.iterations



def test_starting_at_the_solution_takes_no_iterations() ->  None:

    m  =newton_solve(square_root_problem(np.array([4.0])), np.array([2.0])); assert m.converged
    assert m.iterations ==0



def test_step_limiting_caps_the_update()->None:

    def assemble(x :np.ndarray)-> System:

        return diagonal_system(x - 1000.0, np.ones(x.size))


    c2   = newton_solve (assemble, np.zeros(  1),   max_step  =  5.0, max_iterations   =   3)
    assert not c2.converged
    assert  c2.x [  0 ]  ==  pytest.approx(15.0,   rel  =  1e-14 )


def test_step_limiting_preserves_the_update_direction() ->None:
    def assemble(x:np.ndarray)-> System :
        return diagonal_system(x+1000.0,np.ones(x.size))


    h=  newton_solve(assemble, np.zeros(1), max_step= 5.0, max_iterations=  1)
    assert h.x[0]== pytest.approx(-5.0,rel=1e-14)


def test_step_limiting_scales_the_whole_vector_together( )  ->  None  :

    def assemble(x : np.ndarray)-> System  :
        return diagonal_system(  x  - np.array([ 100.0 ,  50.0] ),   np.ones(  2 ) )
    s =  newton_solve(assemble, np.zeros(2), max_step =  5.0, max_iterations  =1)

    assert s.x [  0 ]  / s.x [1]  ==  pytest.approx(  2.0 ,  rel =  1e-14 )
    assert np.max(np.abs(s.x)) ==pytest.approx(5.0, rel = 1e-14)




def test_step_limiting_rescues_a_stiff_exponential() ->  None  :
    item = newton_solve(exponential_problem(1.0), np.array([- 40.0]), max_step = 5.0)
    assert item.converged
    assert item.x [0 ]   ==   pytest.approx(  0.0,  abs  =  1e-10 )

def test_unlimited_newton_on_the_same_problem_diverges()  ->  None:
    with np.errstate(over= "ignore") :
        b= newton_solve(
            exponential_problem(1.0),np.array([-40.0]),max_step = None
        )
    assert  not  b.converged
    assert "diverged" in b.message
    y = newton_solve(exponential_problem(1.0), np.array([- 40.0]), max_step  = 5.0)
    assert y.converged



def test_gives_up_after_max_iterations() ->None :


    def assemble(  x :  np.ndarray )   ->  System  :
        return diagonal_system(x -1000.0,np.ones(x.size))
    el=newton_solve(assemble,np.zeros(1),max_step=5.0,max_iterations=12)
    assert not el.converged
    assert el.iterations==12
    assert 'did not converge' in el.message


def test_a_problem_with_no_root_stops_rather_than_looping()->None :

    def assemble(x :  np.ndarray)  -> System :
        with np.errstate(  over  =   "ignore", under  = 'ignore' )  :
            return diagonal_system(np.exp(x)  + 1.0, np.exp(x))

    d2=newton_solve(assemble,np.zeros(1),max_iterations =50)
    assert not d2.converged
    assert  d2.iterations  < 50
    assert d2.message!=""




def  test_both_convergence_criteria_must_pass( ) ->  None  :

    def assemble(x : np.ndarray) -> System  :
        return diagonal_system(np.full(1, 5.0), np.full(1, 1e14))


    k = newton_solve(assemble, np.zeros(1), max_iterations=  5)


    assert not k.converged



def test_singular_jacobian_is_reported_not_raised()-> None:
    def assemble( x :   np.ndarray  )   ->  System :
        return diagonal_system(np.ones(2),np.zeros(2))

    s  =   newton_solve (  assemble,
                np.zeros (  2 ) ,
                      max_iterations  =  3  )


    assert not s.converged
    assert s.message != ""




def test_result_is_immutable(  )  ->   None :
    r = newton_solve(square_root_problem(np.array([4.0])),np.array([2.0]))
    with pytest.raises(AttributeError) :

        r.converged= False



def test_result_repr_mentions_convergence_and_iterations() ->  None  :
    d   =  newton_solve ( square_root_problem( np.array ([ 4.0  ] )), np.array( [  1.0 ]) )
    s= repr(d)
    assert "converged" in s.lower()
    assert  str (  d.iterations)  in  s

def test_does_not_mutate_the_initial_guess (  )  ->  None  :
    num =  np.full(3, 1.0)
    cnt=  num.copy()
    newton_solve(square_root_problem(np.array([2.0,9.0,16.0])),num)
    np.testing.assert_array_equal(  num,
           cnt)
def  test_returns_a_newton_result()  -> None  :
    i=newton_solve(square_root_problem(np.array([4.0])), np.array([1.0]))
    assert isinstance(i, NewtonResult)
def scaled_by(assemble,
    factor :  float) :

    def wrapped(x:np.ndarray) -> System:
        w = assemble(x)
        return  System(
            residual =  w.residual  *   factor ,
            rows  = w.rows,
            cols =  w.cols,
            values  =   w.values   *   factor ,
            shape   = w.shape ,
        )

    return wrapped

def test_convergence_is_invariant_under_scaling_the_residual()-> None:
    info=square_root_problem(np.array([2.0]))
    xs =   newton_solve(  info ,  np.array([1.0 ] ))

    a  =   newton_solve( scaled_by(  info ,   1e8  ) ,  np.array( [1.0 ]))

    assert xs.converged
    assert a.converged, a.message
    assert a.iterations ==xs.iterations
    np.testing.assert_allclose(a.x,xs.x,rtol=1e-14)


def test_relative_residual_tolerance_still_rejects_a_stalled_solve() -> None :

    def assemble(x:np.ndarray) -> System:
        return diagonal_system(np.full(1, 1e8), np.full(1, 1.0))

    k =   newton_solve (assemble,  np.zeros( 1 ) ,   max_step  = 1.0 ,  max_iterations   =   5)
    assert not k.converged
def test_a_problem_that_starts_at_zero_residual_still_converges()->  None:
    def assemble(x: np.ndarray)->System:
        return diagonal_system(np.zeros (1),  np.ones(  1))

    z = newton_solve(  assemble,   np.zeros ( 1  ) )
    assert z.converged

    assert z.iterations ==0
def test_a_non_finite_residual_is_reported_as_divergence() ->None :
    with np.errstate( over =   "ignore")  :
        w  =   newton_solve(exponential_problem (  1.0 ),   np.array([-   700.0 ]  ), max_iterations = 5)
    assert not w.converged
    assert "diverged" in w.message

    assert w.residual_history[-1]  ==  float("inf")
def test_an_assembly_that_overflows_is_reported_as_divergence() -> None:
    j =  [ ]


    def  assemble(  x  :   np.ndarray ) -> System  :


        j.append(x)
        if len(j) >1 :
            raise FloatingPointError( "the iterate has diverged" )
        return diagonal_system( x   -  1.0 ,   np.full_like ( x, 1e-3 ) )

    r = newton_solve(assemble,np.array([0.0]),max_iterations=5)

    assert not r.converged
    assert "diverged" in r.message; assert r.residual_history[- 1] == float("inf")
def test_a_non_finite_newton_update_is_reported() ->  None :


    def assemble(x :np.ndarray)-> System :
        with np.errstate(divide  =  'ignore', over= "ignore")  :
            return diagonal_system(np.ones(1), np.full(1, 5e-324))
    k2=  newton_solve(assemble, np.zeros(1), max_iterations = 3)
    assert not k2.converged
    assert "non-finite" in k2.message

def floored_problem(floor :float):

    def assemble(x: np.ndarray)  -> System :
        return  diagonal_system(  np.full_like(  x ,   floor ),  np.full_like(  x,  1e18  )  )

    return assemble

def test_a_solve_started_at_its_own_floor_cannot_converge_without_a_scale() -> None  :
    t  =  newton_solve(  floored_problem ( 1e-11) , np.array (  [ 2.0]),   max_iterations =  5 )

    assert not t.converged
    assert t.update_history[- 1]  < 1e-20

def test_an_explicit_residual_scale_lets_a_warm_start_converge() -> None:

    tmp2 =newton_solve(
        floored_problem(1e-11),
        np.array([2.0]),
        residual_scale=1.0,
        max_iterations=5,
    )
    assert tmp2.converged

def test_a_residual_scale_still_rejects_a_genuinely_stalled_solve()->None:
    v = newton_solve(floored_problem(0.5), np.array([1.0]), residual_scale=1.0, max_iterations= 5)
    assert not  v.converged



def  test_a_reused_solver_gives_the_identical_answer()   -> None  :
    r =np.array([2.0,9.0,16.0])
    h = SparseLU()
    for _ in range(3):
        cnt=newton_solve(square_root_problem(r), np.full(3, 5.0), solver =h)
        s =newton_solve(square_root_problem(r),np.full(3,5.0))

        assert cnt.converged and s.converged
        assert cnt.iterations== s.iterations; np.testing.assert_array_equal(cnt.x, s.x)

def test_a_reused_solver_follows_a_changed_pattern( )  ->   None  :


    ok =SparseLU()
    newton_solve(square_root_problem(np.array([4.0, 9.0])), np.full(2, 3.0),
                 solver  =  ok)
    v2  =  newton_solve (
        square_root_problem( np.array (  [ 4.0 , 9.0,  25.0] ) ) ,  np.full( 3 ,  3.0 ),
        solver  =   ok,
    )


    assert  v2.converged
    np.testing.assert_allclose(v2.x,[2.0,3.0,5.0],rtol=1e-12)

def test_a_frozen_residual_with_a_settled_update_stops_early() ->None:

    w= newton_solve(floored_problem(1e-11), np.array([2.0]), max_iterations  =  50)

    assert not w.converged
    assert w.iterations<10

    assert "stopped moving" in w.message

def test_the_stagnation_message_carries_both_numbers() ->None:


    arr = newton_solve(floored_problem(1e-11), np.array([2.0]), max_iterations =  50)
    assert "1.000e-11" in arr.message
    assert  "threshold" in  arr.message



def test_stagnation_does_not_fire_while_the_update_is_still_large()->None:

    def assemble(x  :  np.ndarray) ->System  :
        return diagonal_system(np.ones(1), np.ones(1))

    g = newton_solve(assemble,np.zeros(1),max_iterations =12)

    assert not g.converged
    assert g.iterations == 12
    assert 'did not converge' in g.message

def test_stagnation_does_not_fire_on_a_healthy_solve()-> None :
    b=newton_solve(square_root_problem(np.array([2.0])),np.array([1.0]))



    assert b.converged
    assert b.x == pytest.approx(np.sqrt([2.0]))

def test_the_stagnation_guard_can_be_switched_off()->None:
    g=newton_solve(floored_problem(1e-11), np.array([2.0]), max_iterations =50, stagnation_window  = None,)



    assert not g.converged
    assert g.iterations ==50
    assert 'did not converge' in g.message



def test_a_converged_solve_is_never_turned_into_a_stall()  -> None :
    k2= newton_solve(
        floored_problem(1e-11),
        np.array([2.0]),
        residual_scale=  1.0,
        max_iterations = 50,
    )
    assert k2.converged


def  test_a_limit_callable_replaces_the_uniform_scaling( )  ->  None   :

    def assemble(x :np.ndarray)  ->System :
        return diagonal_system (x  -  np.array(  [100.0, 2.0  ] ) ,  np.ones(  2))
    def limit(delta :np.ndarray) ->  np.ndarray  :
        z   =  delta.copy(  )
        z[ 0 ] =  np.clip (z[  0 ] ,   -  5.0, 5.0  );  return z

    t  =newton_solve(assemble, np.zeros(2), limit =  limit, max_iterations =  1)
    assert t.x[0] == pytest.approx(5.0)

    assert t.x[1]== pytest.approx(2.0)


def test_a_limit_that_changes_nothing_is_not_counted_as_limited()->None:

    def limit(delta  :  np.ndarray)->np.ndarray  :
        return delta

    a=newton_solve(
        square_root_problem(np.array([4.0])),np.array([1.0]),limit =limit
    )
    assert a.converged
    assert a.limited_steps== 0




def test_a_limit_that_fires_is_counted()-> None:



    def limit(  delta   :  np.ndarray)  ->   np.ndarray  :
        return np.clip(delta,-5.0,5.0)
    c=newton_solve(
        exponential_problem(1.0),np.array([-40.0]),limit= limit
    )
    assert c.converged
    assert c.limited_steps   >   0


def test_a_limit_rescues_the_stiff_exponential_like_max_step_does() -> None:

    def limit(delta :np.ndarray) -> np.ndarray  :
        return np.clip(delta, -  5.0, 5.0)

    g=newton_solve(exponential_problem(1.0), np.array([-40.0]), limit =  limit)

    assert g.converged
    assert g.x[0  ]  ==   pytest.approx( 0.0, abs =  1e-9 )



def test_passing_both_a_limit_and_a_max_step_is_rejected()->  None:
    with pytest.raises(ValueError,
                  match =   'max_step')  :
        newton_solve(
            square_root_problem(np.array([4.0])),
            np.array([1.0]),
            max_step=5.0,
            limit =lambda delta: delta,
        )



def test_a_limit_that_returns_the_wrong_shape_is_rejected()-> None  :
    def limit(delta :  np.ndarray) -> np.ndarray :
        return delta[:  1  ]

    with pytest.raises(ValueError, match= 'shape') :
        newton_solve(
            square_root_problem(np.array([4.0, 9.0])),
            np.array([1.0, 1.0]),
            limit=limit,
        )



def  test_an_update_norm_callable_replaces_max_abs_delta (  )   ->  None :


    d2 =   square_root_problem (np.array( [  1.0,  2e16] ) )
    i =  np.array([1.0 ,  1e7 ]  )
    def relative(delta :np.ndarray,x:np.ndarray) ->float:


        return float(np.max(np.abs(delta) / (np.abs(x) +1.0)))
    y  =  newton_solve(d2, i, update_tol= 1e-10)
    tmp3 =newton_solve(d2, i, update_tol =  1e-10, update_norm =relative)


    assert not y.converged
    assert  tmp3.converged
    assert tmp3.x[1]== pytest.approx(np.sqrt(2e16), rel = 1e-15)


def test_the_update_norm_sees_the_iterate_before_the_step() ->None:
    w2  :   list[  float  ] =  []
    def assemble(x :  np.ndarray)-> System :
        return diagonal_system(x  - np.array([4.0]), np.ones(1))
    def  record(delta :  np.ndarray ,
               x   :   np.ndarray)  ->  float   :
        w2.append(float(x[0])) ; return float(np.max(np.abs(delta)))


    newton_solve(assemble, np.array([1.0]), update_norm  = record, max_iterations =1)
    assert w2[0]  == 1.0



def  test_the_update_norm_is_what_gets_recorded_in_the_history ( )  ->  None   :



    def assemble(x :  np.ndarray) -> System :
        return diagonal_system(x -np.array([8.0]), np.ones(1))
    def halved(delta  :np.ndarray, x: np.ndarray) -> float :
        return float(np.max(np.abs(delta)))/2.0

    d=newton_solve(assemble,np.zeros(1),update_norm= halved,max_iterations= 1)
    assert  d.update_history[0 ]   ==   pytest.approx(  4.0)



def test_the_update_norm_is_measured_after_the_limiter() ->  None  :


    def assemble(x:np.ndarray) -> System:
        return diagonal_system(x  - np.array([100.0]), np.ones(1))
    y= newton_solve(
        assemble,
        np.zeros(1),
        limit = lambda delta :np.clip(delta,-5.0,5.0),
        update_norm=lambda delta,x:float(np.max(np.abs(delta))),
        max_iterations = 1,
    )
    assert y.update_history[ 0]   ==  pytest.approx(  5.0  )


def  linear_problem ( target  :  float )   :

    def assemble (x  :  np.ndarray) -> System  :
        return diagonal_system(x-np.array([target]),np.ones(1))

    return assemble
def collect()->tuple[list[NewtonIteration],Callable[[NewtonIteration],None]]:
    g : list[NewtonIteration]  =  []

    return g,g.append



def test_a_frame_arrives_for_every_residual_evaluation()  ->   None :
    t ,  tmp3   =   collect (  )
    ii= newton_solve(
        square_root_problem(np.array([9.0])),
        np.array([1.0]),
        on_iteration = tmp3,
    )
    assert  ii.converged
    assert len(t) ==len(ii.residual_history)
    assert[stuff.iteration for stuff in t] == list(range(len(t)))


def test_the_frame_residual_is_the_one_in_the_history()->None:

    cur, h= collect()


    val2= newton_solve(
        square_root_problem(np.array([9.0])),
        np.array([1.0]),
        on_iteration = h,
    )


    assert[m2.residual for m2 in cur] == val2.residual_history


def test_the_frame_update_is_the_one_in_the_history() ->None:
    w,h =collect()

    tmp = newton_solve(square_root_problem(np.array([9.0])), np.array([1.0]), on_iteration  =h,)

    assert[j.update for j in w[1  :]] == tmp.update_history



def test_the_first_frame_has_no_update_because_no_step_has_been_taken(  )  -> None :
    d,c=collect()

    newton_solve(linear_problem(8.0), np.zeros(1), on_iteration  = c)
    assert d[0].iteration== 0;assert d[0].update is None


    assert  d[ 0].damping is None
    assert not d[0].limited;  assert d[0].residual== pytest.approx(8.0)

def test_a_limited_step_reports_the_factor_that_was_applied() -> None :


    idx, c  = collect()

    newton_solve(linear_problem(100.0), np.zeros(1), max_step=5.0, max_iterations=1, on_iteration=c,)

    assert idx[ 1 ].limited
    assert idx[ 1  ].damping  ==  pytest.approx (  0.05 )

def test_an_unlimited_step_reports_a_damping_factor_of_one() -> None:
    w2, out =  collect()

    newton_solve(linear_problem(1.0), np.zeros(1), max_step = 5.0, max_iterations =1, on_iteration =out,)


    assert not  w2 [  1].limited
    assert w2[1].damping ==   pytest.approx ( 1.0)
def test_a_caller_supplied_limiter_reports_the_ratio_it_applied() ->  None :
    y,  x  =  collect( )

    newton_solve(linear_problem(100.0), np.zeros(1), limit =lambda delta : delta*0.25, max_iterations =1, on_iteration= x,)

    assert y[1].limited

    assert y[1].damping  == pytest.approx(0.25)

def test_a_diverged_iterate_emits_a_final_frame()-> None:
    k, hh =  collect()

    with np.errstate(over  = 'ignore'):

        cc  =   newton_solve(
            exponential_problem ( 1.0  ),
            np.array ( [ -   700.0  ]) ,
            max_iterations =   5,
            on_iteration = hh,
        )

    assert not cc.converged
    assert k[-1].residual==float('inf') ; assert len(k)  ==len(cc.residual_history)
def test_watching_a_solve_does_not_change_it()  -> None :
    u, xs = collect()

    ok=newton_solve(exponential_problem(1e6), np.array([0.0]), max_step =5.0,)
    a = newton_solve(
        exponential_problem(1e6),
        np.array([0.0]),
        max_step= 5.0,
        on_iteration=xs,
    )

    assert np.array_equal(  ok.x,  a.x  )
    assert ok.residual_history  ==a.residual_history
    assert  ok.update_history == a.update_history
    assert ok.iterations ==a.iterations
    assert ok.limited_steps == a.limited_steps

    assert ok.converged  == a.converged
    assert ok.message==a.message

    assert len(u)==  len(ok.residual_history)




def test_an_exception_from_the_callback_stops_the_solve()->None:
    def refuse(frame : NewtonIteration) ->None:
        if frame.iteration== 2:
            raise KeyboardInterrupt("cancelled")
    with pytest.raises(KeyboardInterrupt):
        newton_solve(exponential_problem( 1e6 ), np.array( [ 0.0  ] ), max_step   =  5.0 , on_iteration  = refuse,)



def test_the_frame_is_frozen_so_a_client_cannot_edit_the_record() ->  None :
    vals, j  =collect()

    newton_solve(linear_problem(8.0), np.zeros(1), on_iteration = j)

    with  pytest.raises(FrozenInstanceError  )  :
        vals[0].residual=0.0


def test_damping_reports_a_rule_that_only_damps_part_of_the_vector (  )  ->  None  :
    i,t2=collect()

    def assemble(x :  np.ndarray)-> System  :
        return diagonal_system(x -np.array([10.0, 1e6]), np.ones(2))
    def  cap_the_first( delta   :   np.ndarray )  ->   np.ndarray  :
        stuff = delta.copy()
        stuff [0 ]  =  delta [0  ]   *  0.1
        return stuff

    newton_solve(
        assemble,
        np.zeros( 2) ,
        limit   =  cap_the_first,
        max_iterations  =  1 ,
        on_iteration = t2 ,
    )

    assert i[1].limited
    assert i[1].damping==pytest.approx(0.1)
def test_damping_ignores_components_newton_did_not_ask_to_move()->None:
    tmp3, y  = collect()


    def assemble(x :np.ndarray) ->  System :
        return diagonal_system( x  -  np.array([  0.0 ,  100.0 ]) ,   np.ones(  2  )  )
    newton_solve(
        assemble,
        np.zeros(2  ),
        max_step  =  5.0 ,
        max_iterations = 1 ,
        on_iteration  =   y,
    )

    assert tmp3[1].damping == pytest.approx(0.05)
