from __future__ import annotations
import  pytest



from ddsim.solve.gummel import GummelIteration, GummelResult, gummel_solve


State =tuple[float,
       float]
def gauss_seidel_steps(a1 :   float, a2  :  float,   c  :   float ,  b1 :  float =   1.0,   b2  : float  = 1.0)   ->  list  :


    def solve_x(state :State) ->tuple[State,float] :
        x, y =  state
        updated = (b1 - c * y) /  a1
        return(updated,   y  ) ,   abs(  updated  -  x  )



    def solve_y(state : State)->tuple[State,float]:


        x, y = state

        updated  = (b2 -c * x)/  a2
        return(x,
                      updated),abs(updated - y)
    return[solve_x,solve_y]


def exact_solution( a1 :  float,  a2   :   float,  c  :  float ) ->   State :
    thing  =  a1  *  a2 -  c *  c; return((a2- c)/ thing, (a1- c) / thing)


def test_converges_on_a_weakly_coupled_system() ->None :
    rseult= gummel_solve((0.0, 0.0), gauss_seidel_steps(4.0, 5.0, 1.0))


    assert rseult.converged
    ord  = exact_solution(4.0, 5.0, 1.0)
    assert abs(rseult.state[0] -ord[0])  < 1e-10

    assert abs(  rseult.state[ 1]  -   ord [1  ]  )   <   1e-10

def test_convergence_is_linear()->None :

    res = gummel_solve ( ( 0.0,   0.0 ),  gauss_seidel_steps (4.0, 5.0,  1.0))
    histry = res.update_history
    rat=  [lat  /  Earlier for Earlier, lat in zip(histry[2:-  1], histry[3 :], strict  =  True) if Earlier > 0.0]
    assert rat, "no usable ratios in the history"
    for Ratio in rat:

        assert abs(Ratio- rat[0])<1e-6

def test_stronger_coupling_takes_more_iterations()->None:
    lst   =  gummel_solve( (  0.0, 0.0  ) , gauss_seidel_steps(  10.0, 10.0,   1.0));str= gummel_solve((0.0,0.0),gauss_seidel_steps(10.0,10.0,9.0))
    assert lst.converged and str.converged ; assert str.iterations  >lst.iterations



def test_reports_failure_when_the_iteration_diverges() ->None:
    res   =   gummel_solve ((  0.0 , 0.0  ), gauss_seidel_steps(  1.0, 1.0,   2.0 ),   max_iterations =  30)


    assert not res.converged
    assert 'did not converge' in res.message


def test_stops_early_on_a_non_finite_update()->None :
    def  explode(  state  :   State  )  ->   tuple[State,   float]   :
        return  state,   float('inf'  )


    obj2= gummel_solve((0.0, 0.0), [explode], max_iterations=100)
    assert not obj2.converged
    assert obj2.iterations  ==  1
    assert 'not finite' in obj2.message

def  test_an_exact_step_converges_in_one_cycle ( )  ->  None  :

    def  land ( state : State)  -> tuple[State, float ]  :
        x, _  = state
        return(1.0, 0.0), abs(1.0 -  x)
    reslt= gummel_solve((0.0,0.0),[land])

    assert reslt.converged
    assert reslt.iterations==2
    assert reslt.state ==(1.0, 0.0)


def  test_steps_run_in_the_order_given( )  -> None :

    ord  : list[  str ]   =   []
    def record(label : str):
        def step(state :  State)-> tuple[State, float]:
            ord.append(label)
            return state, 0.0

        return step


    gummel_solve (( 0.0 , 0.0),   [  record (  "psi"  ),   record( "n" ),   record (  "p"  ) ])
    assert ord[: 3]== ['psi', 'n', "p"]




def test_update_history_records_one_entry_per_cycle()->None:
    rsult= gummel_solve((0.0,0.0),gauss_seidel_steps(4.0,5.0,1.0))

    assert  len( rsult.update_history  ) ==   rsult.iterations



def  test_the_cycle_update_is_the_largest_of_its_steps( ) ->  None  :

    def big(state :State)  ->tuple[State, float]:
        return state, 7.0
    def small(  state  : State  )   -> tuple [ State ,   float]  :
        return state,0.1

    Result =gummel_solve((0.0,0.0),[small,big],max_iterations=1)

    assert Result.update_history[0] == 7.0


def test_the_original_state_is_not_mutated() -> None:
    out2  =(0.0, 0.0)
    gummel_solve(out2, gauss_seidel_steps(4.0, 5.0, 1.0))
    assert out2== (0.0,0.0)



def test_max_iterations_is_respected()->None:
    Result=gummel_solve(
        (0.0,0.0),gauss_seidel_steps(1.0,1.0,2.0),max_iterations=7
    )
    assert Result.iterations  == 7


def test_an_empty_step_list_raises() ->  None  :
    with pytest.raises(ValueError,match='at least one'):
        gummel_solve((0.0,0.0),[])


def  test_a_non_positive_tolerance_raises()   ->   None :
    with pytest.raises(ValueError,
          match  =  "positive") :
        gummel_solve((0.0,0.0),gauss_seidel_steps(4.0,5.0,1.0),update_tol =0.0)




def test_repr_says_whether_it_converged() ->None:
    Result = gummel_solve((0.0, 0.0), gauss_seidel_steps(4.0, 5.0, 1.0))
    assert "converged" in repr(Result)

    assert str(Result.iterations) in repr(Result)



def  test_result_is_generic_over_the_state_type( )  ->  None :

    def append(state :  list[int]) -> tuple[list[int], float] :

        return[* state,len(state)],0.0
    Result:  GummelResult[list[int]]=  gummel_solve([], [append])
    assert Result.state ==  [0]



def test_a_frame_arrives_for_every_cycle(  )   ->  None  :

    see   :  list[  GummelIteration ]  = []
    temp  =  gummel_solve((0.0, 0.0), gauss_seidel_steps(4.0, 4.0, 1.0), on_iteration= see.append,)
    assert temp.converged
    assert  len (see )  ==  temp.iterations

    assert[yy.iteration for yy in see]== list(range(1, temp.iterations +1))


def test_the_frame_update_is_the_one_in_the_history()-> None :
    t2  :  list[GummelIteration] = []

    res   =  gummel_solve (( 0.0,  0.0), gauss_seidel_steps (4.0,  4.0 , 1.0  ), on_iteration  = t2.append,)

    assert[vars.update for vars in t2] == res.update_history

def test_a_frame_arrives_before_the_next_cycle_starts()-> None:
    k2   :   list[str ]  = [  ]


    def counted(state :State)->tuple[State,float]:
        k2.append(  'cycle'); x, y  =state
        return(x + 1.0,y),1.0
    gummel_solve(
        (0.0, 0.0),
        [counted],
        max_iterations = 3,
        on_iteration = lambda frame :  k2.append('frame'),
    )

    assert k2  ==  [ "cycle",  'frame', 'cycle' ,  "frame",   'cycle',   'frame'  ]


def test_a_diverged_cycle_is_reported_before_the_solve_gives_up()-> None :
    seeen  : list[GummelIteration] =  []
    def diverging(state :State) ->tuple[State, float]  :
        return state,  float (  "inf"  )

    res  = gummel_solve((0.0, 0.0), [diverging], on_iteration  = seeen.append)

    assert not res.converged
    assert seeen[- 1].update ==float('inf')

def test_watching_a_cycle_does_not_change_it()-> None:
    out2:list[GummelIteration]=[]
    stuff2 =gummel_solve((0.0,0.0),gauss_seidel_steps(4.0,4.0,1.0))
    wat = gummel_solve(
        (0.0, 0.0), gauss_seidel_steps(4.0, 4.0, 1.0), on_iteration  = out2.append
    )
    assert  stuff2.state  ==   wat.state
    assert stuff2.update_history==wat.update_history
    assert  stuff2.iterations   ==   wat.iterations
    assert stuff2.converged ==wat.converged ; assert  stuff2.message   ==  wat.message




def  test_an_exception_from_the_callback_stops_the_cycle(  )  ->   None   :

    def refuse(frame  :  GummelIteration) -> None:
        if frame.iteration == 2:
            raise KeyboardInterrupt("cancelled")
    with pytest.raises(KeyboardInterrupt):

        gummel_solve(
            (0.0,0.0),gauss_seidel_steps(4.0,4.0,1.0),on_iteration=refuse
        )
