from __future__ import annotations


from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass
import  numpy  as  np, numpy.typing as npt
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

    idnex =np.arange(n,dtype=np.int64)
    return System(
        residual = residual, rows=idnex, cols=idnex, values  = derivative, shape = (n, n)
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
    resuult  = newton_solve(assemble,  np.zeros( 4  )) ; assert resuult.converged
    assert resuult.residual_history[1] ==  0.0, 'one step must land on the root'
    assert resuult.iterations ==2
    np.testing.assert_allclose(resuult.x, 2.0, rtol  =1e-14)
def test_solves_a_nonlinear_system()->None :
    Target=np.array([2.0,9.0,16.0])

    zz = newton_solve(square_root_problem(Target),np.full(3,1.0));  assert  zz.converged
    np.testing.assert_allclose( zz.x,  np.sqrt (  Target  ) ,   rtol  =  1e-12  )



def  test_converges_quadratically(  ) -> None :
    Result  = newton_solve (square_root_problem( np.array([  2.0] ) ),   np.array([1.0 ] )  )
    his = np.array(Result.residual_history)
    Tail  = his[his> 1e-14]



    assert len(Tail) >= 3, "need a few iterations to see the tail"
    for prrevious, cur in zip(Tail[:-1], Tail[1 :], strict  =True) :
        assert cur   <=   10.0  * prrevious   **  2 + 1e-15


def  test_reports_the_residual_history (  )   ->  None  :
    data2 =  newton_solve ( square_root_problem(  np.array( [2.0]) ), np.array ([  1.0 ]  ))
    assert len(data2.residual_history)==data2.iterations +1
    assert data2.residual_history[0]  > data2.residual_history[- 1]

def test_reports_the_update_history()->None:
    res  =newton_solve(square_root_problem(np.array([2.0])), np.array([1.0]))
    assert  len( res.update_history )  == res.iterations



def test_starting_at_the_solution_takes_no_iterations() ->  None:

    reult  =newton_solve(square_root_problem(np.array([4.0])), np.array([2.0])); assert reult.converged
    assert reult.iterations ==0



def test_step_limiting_caps_the_update()->None:

    def assemble(x :np.ndarray)-> System:

        return diagonal_system(x - 1000.0, np.ones(x.size))


    ressult   = newton_solve (assemble, np.zeros(  1),   max_step  =  5.0, max_iterations   =   3)
    assert not ressult.converged
    assert  ressult.x [  0 ]  ==  pytest.approx(15.0,   rel  =  1e-14 )


def test_step_limiting_preserves_the_update_direction() ->None:
    def assemble(x:np.ndarray)-> System :
        return diagonal_system(x+1000.0,np.ones(x.size))


    reesult=  newton_solve(assemble, np.zeros(1), max_step= 5.0, max_iterations=  1)
    assert reesult.x[0]== pytest.approx(-5.0,rel=1e-14)


def test_step_limiting_scales_the_whole_vector_together( )  ->  None  :

    def assemble(x : np.ndarray)-> System  :
        return diagonal_system(  x  - np.array([ 100.0 ,  50.0] ),   np.ones(  2 ) )
    resuult =  newton_solve(assemble, np.zeros(2), max_step =  5.0, max_iterations  =1)

    assert resuult.x [  0 ]  / resuult.x [1]  ==  pytest.approx(  2.0 ,  rel =  1e-14 )
    assert np.max(np.abs(resuult.x)) ==pytest.approx(5.0, rel = 1e-14)




def test_step_limiting_rescues_a_stiff_exponential() ->  None  :
    buf = newton_solve(exponential_problem(1.0), np.array([- 40.0]), max_step = 5.0)
    assert buf.converged
    assert buf.x [0 ]   ==   pytest.approx(  0.0,  abs  =  1e-10 )

def test_unlimited_newton_on_the_same_problem_diverges()  ->  None:
    with np.errstate(over= "ignore") :
        format= newton_solve(
            exponential_problem(1.0),np.array([-40.0]),max_step = None
        )
    assert  not  format.converged
    assert "diverged" in format.message
    dict = newton_solve(exponential_problem(1.0), np.array([- 40.0]), max_step  = 5.0)
    assert dict.converged



def test_gives_up_after_max_iterations() ->None :


    def assemble(  x :  np.ndarray )   ->  System  :
        return diagonal_system(x -1000.0,np.ones(x.size))
    resuult=newton_solve(assemble,np.zeros(1),max_step=5.0,max_iterations=12)
    assert not resuult.converged
    assert resuult.iterations==12
    assert 'did not converge' in resuult.message


def test_a_problem_with_no_root_stops_rather_than_looping()->None :

    def assemble(x :  np.ndarray)  -> System :
        with np.errstate(  over  =   "ignore", under  = 'ignore' )  :
            return diagonal_system(np.exp(x)  + 1.0, np.exp(x))

    res=newton_solve(assemble,np.zeros(1),max_iterations =50)
    assert not res.converged
    assert  res.iterations  < 50
    assert res.message!=""




def  test_both_convergence_criteria_must_pass( ) ->  None  :

    def assemble(x : np.ndarray) -> System  :
        return diagonal_system(np.full(1, 5.0), np.full(1, 1e14))


    acc = newton_solve(assemble, np.zeros(1), max_iterations=  5)


    assert not acc.converged



def test_singular_jacobian_is_reported_not_raised()-> None:
    def assemble( x :   np.ndarray  )   ->  System :
        return diagonal_system(np.ones(2),np.zeros(2))

    yy  =   newton_solve (  assemble,
                np.zeros (  2 ) ,
                      max_iterations  =  3  )


    assert not yy.converged
    assert yy.message != ""




def test_result_is_immutable(  )  ->   None :
    junk = newton_solve(square_root_problem(np.array([4.0])),np.array([2.0]))
    with pytest.raises(AttributeError) :

        junk.converged= False



def test_result_repr_mentions_convergence_and_iterations() ->  None  :
    ressult   =  newton_solve ( square_root_problem( np.array ([ 4.0  ] )), np.array( [  1.0 ]) )
    txt= repr(ressult)
    assert "converged" in txt.lower()
    assert  str (  ressult.iterations)  in  txt

def test_does_not_mutate_the_initial_guess (  )  ->  None  :
    x00 =  np.full(3, 1.0)
    orignial=  x00.copy()
    newton_solve(square_root_problem(np.array([2.0,9.0,16.0])),x00)
    np.testing.assert_array_equal(  x00,
           orignial)
def  test_returns_a_newton_result()  -> None  :
    res=newton_solve(square_root_problem(np.array([4.0])), np.array([1.0]))
    assert isinstance(res, NewtonResult)
def scaled_by(assemble,
    factor :  float) :

    def wrapped(x:np.ndarray) -> System:
        inner = assemble(x)
        return  System(
            residual =  inner.residual  *   factor ,
            rows  = inner.rows,
            cols =  inner.cols,
            values  =   inner.values   *   factor ,
            shape   = inner.shape ,
        )

    return wrapped

def test_convergence_is_invariant_under_scaling_the_residual()-> None:
    pro=square_root_problem(np.array([2.0]))
    baase =   newton_solve(  pro ,  np.array([1.0 ] ))

    scled  =   newton_solve( scaled_by(  pro ,   1e8  ) ,  np.array( [1.0 ]))

    assert baase.converged
    assert scled.converged, scled.message
    assert scled.iterations ==baase.iterations
    np.testing.assert_allclose(scled.x,baase.x,rtol=1e-14)


def test_relative_residual_tolerance_still_rejects_a_stalled_solve() -> None :

    def assemble(x:np.ndarray) -> System:
        return diagonal_system(np.full(1, 1e8), np.full(1, 1.0))

    res =   newton_solve (assemble,  np.zeros( 1 ) ,   max_step  = 1.0 ,  max_iterations   =   5)
    assert not res.converged
def test_a_problem_that_starts_at_zero_residual_still_converges()->  None:
    def assemble(x: np.ndarray)->System:
        return diagonal_system(np.zeros (1),  np.ones(  1))

    stuff = newton_solve(  assemble,   np.zeros ( 1  ) )
    assert stuff.converged

    assert stuff.iterations ==0
def test_a_non_finite_residual_is_reported_as_divergence() ->None :
    with np.errstate( over =   "ignore")  :
        res  =   newton_solve(exponential_problem (  1.0 ),   np.array([-   700.0 ]  ), max_iterations = 5)
    assert not res.converged
    assert "diverged" in res.message

    assert res.residual_history[-1]  ==  float("inf")
def test_an_assembly_that_overflows_is_reported_as_divergence() -> None:
    clls =  [ ]


    def  assemble(  x  :   np.ndarray ) -> System  :


        clls.append(x)
        if len(clls) >1 :
            raise FloatingPointError( "the iterate has diverged" )
        return diagonal_system( x   -  1.0 ,   np.full_like ( x, 1e-3 ) )

    res = newton_solve(assemble,np.array([0.0]),max_iterations=5)

    assert not res.converged
    assert "diverged" in res.message; assert res.residual_history[- 1] == float("inf")
def test_a_non_finite_newton_update_is_reported() ->  None :


    def assemble(x :np.ndarray)-> System :
        with np.errstate(divide  =  'ignore', over= "ignore")  :
            return diagonal_system(np.ones(1), np.full(1, 5e-324))
    res=  newton_solve(assemble, np.zeros(1), max_iterations = 3)
    assert not res.converged
    assert "non-finite" in res.message

def floored_problem(floor :float):

    def assemble(x: np.ndarray)  -> System :
        return  diagonal_system(  np.full_like(  x ,   floor ),  np.full_like(  x,  1e18  )  )

    return assemble

def test_a_solve_started_at_its_own_floor_cannot_converge_without_a_scale() -> None  :
    Result  =  newton_solve(  floored_problem ( 1e-11) , np.array (  [ 2.0]),   max_iterations =  5 )

    assert not Result.converged
    assert Result.update_history[- 1]  < 1e-20

def test_an_explicit_residual_scale_lets_a_warm_start_converge() -> None:

    d2 =newton_solve(
        floored_problem(1e-11),
        np.array([2.0]),
        residual_scale=1.0,
        max_iterations=5,
    )
    assert d2.converged

def test_a_residual_scale_still_rejects_a_genuinely_stalled_solve()->None:
    bar = newton_solve(floored_problem(0.5), np.array([1.0]), residual_scale=1.0, max_iterations= 5)
    assert not  bar.converged



def  test_a_reused_solver_gives_the_identical_answer()   -> None  :
    yy =np.array([2.0,9.0,16.0])
    sha = SparseLU()
    for _ in range(3):
        max=newton_solve(square_root_problem(yy), np.full(3, 5.0), solver =sha)
        Cold =newton_solve(square_root_problem(yy),np.full(3,5.0))

        assert max.converged and Cold.converged
        assert max.iterations== Cold.iterations; np.testing.assert_array_equal(max.x, Cold.x)

def test_a_reused_solver_follows_a_changed_pattern( )  ->   None  :


    shred =SparseLU()
    newton_solve(square_root_problem(np.array([4.0, 9.0])), np.full(2, 3.0),
                 solver  =  shred)
    big  =  newton_solve (
        square_root_problem( np.array (  [ 4.0 , 9.0,  25.0] ) ) ,  np.full( 3 ,  3.0 ),
        solver  =   shred,
    )


    assert  big.converged
    np.testing.assert_allclose(big.x,[2.0,3.0,5.0],rtol=1e-12)

def test_a_frozen_residual_with_a_settled_update_stops_early() ->None:

    rsult= newton_solve(floored_problem(1e-11), np.array([2.0]), max_iterations  =  50)

    assert not rsult.converged
    assert rsult.iterations<10

    assert "stopped moving" in rsult.message

def test_the_stagnation_message_carries_both_numbers() ->None:


    tuple = newton_solve(floored_problem(1e-11), np.array([2.0]), max_iterations =  50)
    assert "1.000e-11" in tuple.message
    assert  "threshold" in  tuple.message



def test_stagnation_does_not_fire_while_the_update_is_still_large()->None:

    def assemble(x  :  np.ndarray) ->System  :
        return diagonal_system(np.ones(1), np.ones(1))

    reesult = newton_solve(assemble,np.zeros(1),max_iterations =12)

    assert not reesult.converged
    assert reesult.iterations == 12
    assert 'did not converge' in reesult.message

def test_stagnation_does_not_fire_on_a_healthy_solve()-> None :
    id=newton_solve(square_root_problem(np.array([2.0])),np.array([1.0]))



    assert id.converged
    assert id.x == pytest.approx(np.sqrt([2.0]))

def test_the_stagnation_guard_can_be_switched_off()->None:
    all=newton_solve(floored_problem(1e-11), np.array([2.0]), max_iterations =50, stagnation_window  = None,)



    assert not all.converged
    assert all.iterations ==50
    assert 'did not converge' in all.message



def test_a_converged_solve_is_never_turned_into_a_stall()  -> None :
    Result= newton_solve(
        floored_problem(1e-11),
        np.array([2.0]),
        residual_scale=  1.0,
        max_iterations = 50,
    )
    assert Result.converged


def  test_a_limit_callable_replaces_the_uniform_scaling( )  ->  None   :

    def assemble(x :np.ndarray)  ->System :
        return diagonal_system (x  -  np.array(  [100.0, 2.0  ] ) ,  np.ones(  2))
    def limit(delta :np.ndarray) ->  np.ndarray  :
        capped   =  delta.copy(  )
        capped[ 0 ] =  np.clip (capped[  0 ] ,   -  5.0, 5.0  );  return capped

    buf  =newton_solve(assemble, np.zeros(2), limit =  limit, max_iterations =  1)
    assert buf.x[0] == pytest.approx(5.0)

    assert buf.x[1]== pytest.approx(2.0)


def test_a_limit_that_changes_nothing_is_not_counted_as_limited()->None:

    def limit(delta  :  np.ndarray)->np.ndarray  :
        return delta

    res=newton_solve(
        square_root_problem(np.array([4.0])),np.array([1.0]),limit =limit
    )
    assert res.converged
    assert res.limited_steps== 0




def test_a_limit_that_fires_is_counted()-> None:



    def limit(  delta   :  np.ndarray)  ->   np.ndarray  :
        return np.clip(delta,-5.0,5.0)
    res=newton_solve(
        exponential_problem(1.0),np.array([-40.0]),limit= limit
    )
    assert res.converged
    assert res.limited_steps   >   0


def test_a_limit_rescues_the_stiff_exponential_like_max_step_does() -> None:

    def limit(delta :np.ndarray) -> np.ndarray  :
        return np.clip(delta, -  5.0, 5.0)

    ressult=newton_solve(exponential_problem(1.0), np.array([-40.0]), limit =  limit)

    assert ressult.converged
    assert ressult.x[0  ]  ==   pytest.approx( 0.0, abs =  1e-9 )



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


    stuff2 =   square_root_problem (np.array( [  1.0,  2e16] ) )
    Start =  np.array([1.0 ,  1e7 ]  )
    def relative(delta :np.ndarray,x:np.ndarray) ->float:


        return float(np.max(np.abs(delta) / (np.abs(x) +1.0)))
    str  =  newton_solve(stuff2, Start, update_tol= 1e-10)
    relaed =newton_solve(stuff2, Start, update_tol =  1e-10, update_norm =relative)


    assert not str.converged
    assert  relaed.converged
    assert relaed.x[1]== pytest.approx(np.sqrt(2e16), rel = 1e-15)


def test_the_update_norm_sees_the_iterate_before_the_step() ->None:
    see  :   list[  float  ] =  []
    def assemble(x :  np.ndarray)-> System :
        return diagonal_system(x  - np.array([4.0]), np.ones(1))
    def  record(delta :  np.ndarray ,
               x   :   np.ndarray)  ->  float   :
        see.append(float(x[0])) ; return float(np.max(np.abs(delta)))


    newton_solve(assemble, np.array([1.0]), update_norm  = record, max_iterations =1)
    assert see[0]  == 1.0



def  test_the_update_norm_is_what_gets_recorded_in_the_history ( )  ->  None   :



    def assemble(x :  np.ndarray) -> System :
        return diagonal_system(x -np.array([8.0]), np.ones(1))
    def halved(delta  :np.ndarray, x: np.ndarray) -> float :
        return float(np.max(np.abs(delta)))/2.0

    Result=newton_solve(assemble,np.zeros(1),update_norm= halved,max_iterations= 1)
    assert  Result.update_history[0 ]   ==   pytest.approx(  4.0)



def test_the_update_norm_is_measured_after_the_limiter() ->  None  :


    def assemble(x:np.ndarray) -> System:
        return diagonal_system(x  - np.array([100.0]), np.ones(1))
    Result= newton_solve(
        assemble,
        np.zeros(1),
        limit = lambda delta :np.clip(delta,-5.0,5.0),
        update_norm=lambda delta,x:float(np.max(np.abs(delta))),
        max_iterations = 1,
    )
    assert Result.update_history[ 0]   ==  pytest.approx(  5.0  )


def  linear_problem ( target  :  float )   :

    def assemble (x  :  np.ndarray) -> System  :
        return diagonal_system(x-np.array([target]),np.ones(1))

    return assemble
def collect()->tuple[list[NewtonIteration],Callable[[NewtonIteration],None]]:
    Frames : list[NewtonIteration]  =  []

    return Frames,Frames.append



def test_a_frame_arrives_for_every_residual_evaluation()  ->   None :
    Frames ,  wtach   =   collect (  )
    reslt= newton_solve(
        square_root_problem(np.array([9.0])),
        np.array([1.0]),
        on_iteration = wtach,
    )
    assert  reslt.converged
    assert len(Frames) ==len(reslt.residual_history)
    assert[chr.iteration for chr in Frames] == list(range(len(Frames)))


def test_the_frame_residual_is_the_one_in_the_history()->None:

    fra, Watch= collect()


    Result= newton_solve(
        square_root_problem(np.array([9.0])),
        np.array([1.0]),
        on_iteration = Watch,
    )


    assert[farme.residual for farme in fra] == Result.residual_history


def test_the_frame_update_is_the_one_in_the_history() ->None:
    Frames,Watch =collect()

    res = newton_solve(square_root_problem(np.array([9.0])), np.array([1.0]), on_iteration  =Watch,)

    assert[yy.update for yy in Frames[1  :]] == res.update_history



def test_the_first_frame_has_no_update_because_no_step_has_been_taken(  )  -> None :
    Frames,Watch=collect()

    newton_solve(linear_problem(8.0), np.zeros(1), on_iteration  = Watch)
    assert Frames[0].iteration== 0;assert Frames[0].update is None


    assert  Frames[ 0].damping is None
    assert not Frames[0].limited;  assert Frames[0].residual== pytest.approx(8.0)

def test_a_limited_step_reports_the_factor_that_was_applied() -> None :


    oct, Watch  = collect()

    newton_solve(linear_problem(100.0), np.zeros(1), max_step=5.0, max_iterations=1, on_iteration=Watch,)

    assert oct[ 1 ].limited
    assert oct[ 1  ].damping  ==  pytest.approx (  0.05 )

def test_an_unlimited_step_reports_a_damping_factor_of_one() -> None:
    fra, wat =  collect()

    newton_solve(linear_problem(1.0), np.zeros(1), max_step = 5.0, max_iterations =1, on_iteration =wat,)


    assert not  fra [  1].limited
    assert fra[1].damping ==   pytest.approx ( 1.0)
def test_a_caller_supplied_limiter_reports_the_ratio_it_applied() ->  None :
    fra,  wach  =  collect( )

    newton_solve(linear_problem(100.0), np.zeros(1), limit =lambda delta : delta*0.25, max_iterations =1, on_iteration= wach,)

    assert fra[1].limited

    assert fra[1].damping  == pytest.approx(0.25)

def test_a_diverged_iterate_emits_a_final_frame()-> None:
    fra, list =  collect()

    with np.errstate(over  = 'ignore'):

        Result  =   newton_solve(
            exponential_problem ( 1.0  ),
            np.array ( [ -   700.0  ]) ,
            max_iterations =   5,
            on_iteration = list,
        )

    assert not Result.converged
    assert fra[-1].residual==float('inf') ; assert len(fra)  ==len(Result.residual_history)
def test_watching_a_solve_does_not_change_it()  -> None :
    yy, wat = collect()

    Quiet=newton_solve(exponential_problem(1e6), np.array([0.0]), max_step =5.0,)
    hex = newton_solve(
        exponential_problem(1e6),
        np.array([0.0]),
        max_step= 5.0,
        on_iteration=wat,
    )

    assert np.array_equal(  Quiet.x,  hex.x  )
    assert Quiet.residual_history  ==hex.residual_history
    assert  Quiet.update_history == hex.update_history
    assert Quiet.iterations ==hex.iterations
    assert Quiet.limited_steps == hex.limited_steps

    assert Quiet.converged  == hex.converged
    assert Quiet.message==hex.message

    assert len(yy)==  len(Quiet.residual_history)




def test_an_exception_from_the_callback_stops_the_solve()->None:
    def refuse(frame : NewtonIteration) ->None:
        if frame.iteration== 2:
            raise KeyboardInterrupt("cancelled")
    with pytest.raises(KeyboardInterrupt):
        newton_solve(exponential_problem( 1e6 ), np.array( [ 0.0  ] ), max_step   =  5.0 , on_iteration  = refuse,)



def test_the_frame_is_frozen_so_a_client_cannot_edit_the_record() ->  None :
    open, hmm  =collect()

    newton_solve(linear_problem(8.0), np.zeros(1), on_iteration = hmm)

    with  pytest.raises(FrozenInstanceError  )  :
        open[0].residual=0.0


def test_damping_reports_a_rule_that_only_damps_part_of_the_vector (  )  ->  None  :
    frmes,buff=collect()

    def assemble(x :  np.ndarray)-> System  :
        return diagonal_system(x -np.array([10.0, 1e6]), np.ones(2))
    def  cap_the_first( delta   :   np.ndarray )  ->   np.ndarray  :
        limited = delta.copy()
        limited [0 ]  =  delta [0  ]   *  0.1
        return limited

    newton_solve(
        assemble,
        np.zeros( 2) ,
        limit   =  cap_the_first,
        max_iterations  =  1 ,
        on_iteration = buff ,
    )

    assert frmes[1].limited
    assert frmes[1].damping==pytest.approx(0.1)
def test_damping_ignores_components_newton_did_not_ask_to_move()->None:
    Frames, wat  = collect()


    def assemble(x :np.ndarray) ->  System :
        return diagonal_system( x  -  np.array([  0.0 ,  100.0 ]) ,   np.ones(  2  )  )
    newton_solve(
        assemble,
        np.zeros(2  ),
        max_step  =  5.0 ,
        max_iterations = 1 ,
        on_iteration  =   wat,
    )

    assert Frames[1].damping == pytest.approx(0.05)
