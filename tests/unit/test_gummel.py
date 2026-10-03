from __future__ import annotations
import pytest
from ddsim.solve.gummel import GummelIteration, GummelResult, gummel_solve


State =tuple[float,
       float]
def gauss_seidel_steps(a1 :   float, a2  :  float,   c  :   float ,  b1 :  float =   1.0,   b2  : float  = 1.0)   ->  list  :


    def solve_x(state :State) ->tuple[State,float] :
        res2, i =  state
        u = (b1 - c * i) /  a1
        return(u,   i  ) ,   abs(  u  -  res2  )



    def solve_y(state : State)->tuple[State,float]:


        j, d = state

        d2  = (b2 -c * j)/  a2
        return(j,
                      d2),abs(d2 - d)
    return[solve_x,solve_y]


def exact_solution( a1 :  float,  a2   :   float,  c  :  float ) ->   State :
    s  =  a1  *  a2 -  c *  c; return((a2- c)/ s, (a1- c) / s)


def test_converges_on_a_weakly_coupled_system() ->None :
    flag= gummel_solve((0.0, 0.0), gauss_seidel_steps(4.0, 5.0, 1.0))


    assert flag.converged
    m  = exact_solution(4.0, 5.0, 1.0)
    assert abs(flag.state[0] -m[0])  < 1e-10

    assert abs(  flag.state[ 1]  -   m [1  ]  )   <   1e-10

def test_convergence_is_linear()->None :

    u = gummel_solve ( ( 0.0,   0.0 ),  gauss_seidel_steps (4.0, 5.0,  1.0))
    ok = u.update_history
    c=  [t  /  r for r, t in zip(ok[2:-  1], ok[3 :], strict  =  True) if r > 0.0]
    assert c, "no usable ratios in the history"
    for w in c:

        assert abs(w- c[0])<1e-6

def test_stronger_coupling_takes_more_iterations()->None:
    k   =  gummel_solve( (  0.0, 0.0  ) , gauss_seidel_steps(  10.0, 10.0,   1.0));r= gummel_solve((0.0,0.0),gauss_seidel_steps(10.0,10.0,9.0))
    assert k.converged and r.converged ; assert r.iterations  >k.iterations



def test_reports_failure_when_the_iteration_diverges() ->None:
    r   =   gummel_solve ((  0.0 , 0.0  ), gauss_seidel_steps(  1.0, 1.0,   2.0 ),   max_iterations =  30)


    assert not r.converged
    assert 'did not converge' in r.message


def test_stops_early_on_a_non_finite_update()->None :
    def  explode(  state  :   State  )  ->   tuple[State,   float]   :
        return  state,   float('inf'  )


    thing= gummel_solve((0.0, 0.0), [explode], max_iterations=100)
    assert not thing.converged
    assert thing.iterations  ==  1
    assert 'not finite' in thing.message

def  test_an_exact_step_converges_in_one_cycle ( )  ->  None  :

    def  land ( state : State)  -> tuple[State, float ]  :
        e, _  = state
        return(1.0, 0.0), abs(1.0 -  e)
    b= gummel_solve((0.0,0.0),[land])

    assert b.converged
    assert b.iterations==2
    assert b.state ==(1.0, 0.0)


def  test_steps_run_in_the_order_given( )  -> None :

    dd  : list[  str ]   =   []
    def record(label : str):
        def step(state :  State)-> tuple[State, float]:
            dd.append(label)
            return state, 0.0

        return step


    gummel_solve (( 0.0 , 0.0),   [  record (  "psi"  ),   record( "n" ),   record (  "p"  ) ])
    assert dd[: 3]== ['psi', 'n', "p"]




def test_update_history_records_one_entry_per_cycle()->None:
    tt= gummel_solve((0.0,0.0),gauss_seidel_steps(4.0,5.0,1.0))

    assert  len( tt.update_history  ) ==   tt.iterations



def  test_the_cycle_update_is_the_largest_of_its_steps( ) ->  None  :

    def big(state :State)  ->tuple[State, float]:
        return state, 7.0
    def small(  state  : State  )   -> tuple [ State ,   float]  :
        return state,0.1

    a =gummel_solve((0.0,0.0),[small,big],max_iterations=1)

    assert a.update_history[0] == 7.0


def test_the_original_state_is_not_mutated() -> None:
    b  =(0.0, 0.0)
    gummel_solve(b, gauss_seidel_steps(4.0, 5.0, 1.0))
    assert b== (0.0,0.0)



def test_max_iterations_is_respected()->None:
    w2=gummel_solve(
        (0.0,0.0),gauss_seidel_steps(1.0,1.0,2.0),max_iterations=7
    )
    assert w2.iterations  == 7


def test_an_empty_step_list_raises() ->  None  :
    with pytest.raises(ValueError,match='at least one'):
        gummel_solve((0.0,0.0),[])


def  test_a_non_positive_tolerance_raises()   ->   None :
    with pytest.raises(ValueError,
          match  =  "positive") :
        gummel_solve((0.0,0.0),gauss_seidel_steps(4.0,5.0,1.0),update_tol =0.0)




def test_repr_says_whether_it_converged() ->None:
    j = gummel_solve((0.0, 0.0), gauss_seidel_steps(4.0, 5.0, 1.0))
    assert "converged" in repr(j)

    assert str(j.iterations) in repr(j)



def  test_result_is_generic_over_the_state_type( )  ->  None :

    def append(state :  list[int]) -> tuple[list[int], float] :

        return[* state,len(state)],0.0
    y:  GummelResult[list[int]]=  gummel_solve([], [append])
    assert y.state ==  [0]



def test_a_frame_arrives_for_every_cycle(  )   ->  None  :

    z   :  list[  GummelIteration ]  = []
    item  =  gummel_solve((0.0, 0.0), gauss_seidel_steps(4.0, 4.0, 1.0), on_iteration= z.append,)
    assert item.converged
    assert  len (z )  ==  item.iterations

    assert[v.iteration for v in z]== list(range(1, item.iterations +1))


def test_the_frame_update_is_the_one_in_the_history()-> None :
    cur  :  list[GummelIteration] = []

    t   =  gummel_solve (( 0.0,  0.0), gauss_seidel_steps (4.0,  4.0 , 1.0  ), on_iteration  = cur.append,)

    assert[f.update for f in cur] == t.update_history

def test_a_frame_arrives_before_the_next_cycle_starts()-> None:
    u   :   list[str ]  = [  ]


    def counted(state :State)->tuple[State,float]:
        u.append(  'cycle'); t, k2  =state
        return(t + 1.0,k2),1.0
    gummel_solve(
        (0.0, 0.0),
        [counted],
        max_iterations = 3,
        on_iteration = lambda frame :  u.append('frame'),
    )

    assert u  ==  [ "cycle",  'frame', 'cycle' ,  "frame",   'cycle',   'frame'  ]


def test_a_diverged_cycle_is_reported_before_the_solve_gives_up()-> None :
    y  : list[GummelIteration] =  []
    def diverging(state :State) ->tuple[State, float]  :
        return state,  float (  "inf"  )

    u  = gummel_solve((0.0, 0.0), [diverging], on_iteration  = y.append)

    assert not u.converged
    assert y[- 1].update ==float('inf')

def test_watching_a_cycle_does_not_change_it()-> None:
    nxt:list[GummelIteration]=[]
    f =gummel_solve((0.0,0.0),gauss_seidel_steps(4.0,4.0,1.0))
    j = gummel_solve(
        (0.0, 0.0), gauss_seidel_steps(4.0, 4.0, 1.0), on_iteration  = nxt.append
    )
    assert  f.state  ==   j.state
    assert f.update_history==j.update_history
    assert  f.iterations   ==   j.iterations
    assert f.converged ==j.converged ; assert  f.message   ==  j.message




def  test_an_exception_from_the_callback_stops_the_cycle(  )  ->   None   :

    def refuse(frame  :  GummelIteration) -> None:
        if frame.iteration == 2:
            raise KeyboardInterrupt("cancelled")
    with pytest.raises(KeyboardInterrupt):

        gummel_solve(
            (0.0,0.0),gauss_seidel_steps(4.0,4.0,1.0),on_iteration=refuse
        )
