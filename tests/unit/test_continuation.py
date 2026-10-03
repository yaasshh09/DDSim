from __future__ import annotations
import pytest
from ddsim.solve.continuation import ContinuationEvent,continue_to

def always(value  :  float) -> float :
    return  value


def record_calls(calls   :   list[tuple[ float ,   float ] ]  )   :

    def solve(  parameter  :   float,  guess  :  float  )  ->   float  :
        calls.append((parameter,guess))

        return parameter

    return  solve
def fails_beyond(limit:float, minimum_step :float)  :

    t =[0.0]
    def solve(parameter :float,
       guess:float)->float |None:
        if  parameter  > limit  and  abs( parameter  - t[  -  1] )  >  minimum_step   :

            return  None
        t.append(parameter )
        return parameter

    return solve



def test_reaches_the_target_exactly()   ->  None   :
    yy  = continue_to(
        lambda value, guess : always(value),
        start =  0.0,
        target = 1.0,
        initial =0.0,
        step=0.05,
    )
    assert yy.converged
    assert yy.parameter== 1.0
    assert yy.solution   == 1.0


def test_never_overshoots_the_target()->None :
    r :list[tuple[float,float]] =[]
    continue_to(record_calls(r),start =0.0,target =0.3,initial =0.0,step =0.11)


    assert max (  y for  y, _ in  r  ) <=  0.3

def test_walks_downward_too() -> None :
    w  :   list[tuple[  float, float ] ]  =  [  ]
    res=continue_to(
        record_calls(w),start=0.0,target=-1.0,initial =0.0,step=0.25
    )

    assert res.converged

    assert res.parameter == -1.0
    assert all(i <= 0.0 for i,_ in w)

def test_hands_each_solve_the_previous_solution() ->None :
    e:list[tuple[float,float]]=[]
    continue_to(record_calls(e), start = 0.0, target =1.0, initial=0.0, step =0.25)
    for y,obj in zip(e,e[1 :],strict= False):
        assert obj [1 ]  == y[  0]




def test_a_target_equal_to_the_start_does_nothing() ->  None :
    thing : list[tuple[float, float]] = []
    i= continue_to(
        record_calls(thing),start=0.5,target=0.5,initial =7.0,step = 0.1
    )

    assert i.converged ; assert i.solution== 7.0;  assert thing==[]


def test_the_step_grows_by_the_growth_factor() ->  None  :
    m  :   list[tuple[  float, float  ] ]  =   [  ]

    continue_to(record_calls(m), start =  0.0, target=  100.0, initial = 0.0, step = 1.0, growth = 1.5, max_step= 1e9,)

    j =  [0.0, *  [  h  for h ,  _ in  m  ]  ]
    info = [
        y -u
        for u,y in zip(j[:- 1],j[1:],strict=True)
    ]
    assert info[0]==pytest.approx(1.0)
    assert info[1] == pytest.approx(1.5)
    assert info[2]==  pytest.approx(2.25)


def test_the_step_is_capped() ->None:
    i: list[tuple[float,float]]= []
    continue_to(
        record_calls(i),
        start=0.0,
        target=100.0,
        initial =0.0,
        step= 1.0,
        growth=1.5,
        max_step=2.0,
    )
    el  =[0.0,
           *[u for u,
              _ in i]]
    cc  = [d  -  out2 for out2, d in zip(el[:-  1], el[1 :], strict= True)]
    assert max (cc  )   <=  2.0   +  1e-12


def test_a_failed_step_is_halved_and_retried ( )  ->   None  :


    j=continue_to(
        fails_beyond(0.5,0.2),start=0.0,target=1.0,initial= 0.0,step =0.5
    )

    assert j.converged

    assert j.parameter== 1.0

    mm =[y2 for y2 in j.events if not y2.converged]
    assert  mm,  'nothing was rejected, so the halving path never ran'
    for y2 in mm :
        assert 'halved' in y2.message

def test_every_attempt_is_logged()-> None  :
    c=continue_to(fails_beyond(0.5,0.2),start=0.0,target =1.0,initial= 0.0,step= 0.5)
    assert len(c.events)  > len(  c.accepted )
    assert[v.parameter for v in c.events if v.converged] == list(c.accepted)



def test_gives_up_below_the_minimum_step (  ) -> None  :
    k  =continue_to(
        lambda value, guess  :None,
        start = 0.0,
        target=  1.0,
        initial  = 0.0,
        step = 0.1,
        min_step= 0.01,
    )

    assert not k.converged ; assert k.parameter==  0.0
    assert k.solution==0.0
    assert "minimum step" in k.message


def test_the_partial_solution_survives_a_failure()->None  :
    k2 =  continue_to(
        fails_beyond ( 0.4,  1e-9  ),
        start   =  0.0 ,
        target =  1.0 ,
        initial  =   0.0,
        step   = 0.1,
        min_step   = 0.01,
    )


    assert  not  k2.converged

    assert 0.0 < k2.parameter<=0.4
    assert k2.solution == k2.parameter
def test_running_out_of_attempts_is_reported() -> None:
    b= continue_to(lambda value,guess:always(value), start=0.0, target= 1e6, initial=0.0, step=1.0, max_step =1.0, max_attempts=5,)


    assert  not  b.converged
    assert 'attempts'  in b.message




@pytest.mark.parametrize(('kwargs', "match"), [({'step' : 0.0}, "step must be positive"), ({'step': -  0.1}, 'step must be positive'), ({'step' : 0.1, "growth": 1.0}, 'growth'), ({'step':  0.1, "min_step": 0.5}, "min_step"), ({'step' : 0.1, "max_step" :0.05}, "max_step"), ({'step' : 0.1, "max_attempts" :  0}, "max_attempts"),],)



def test_bad_arguments_raise(kwargs: dict,match :str)->None:
    with pytest.raises(ValueError,match = match) :

        continue_to(lambda value, guess:  always(value), start  = 0.0, target  =1.0, initial=  0.0, **  kwargs,)


def test_repr_reports_where_it_got_to ( )   -> None  :
    d =  continue_to(
        lambda value, guess : always(value), start = 0.0, target=  1.0, initial  =0.0, step = 0.5
    )

    assert '1' in repr(d)
    assert "converged" in repr(d)




def test_event_repr_reports_the_attempt()->None:
    a=continue_to(
        fails_beyond(0.5,0.2),start=0.0,target= 1.0,initial=0.0,step=0.5
    )
    j  = repr( a.events [0 ]  )
    assert 'step' in j

    assert 'ok' in j or "failed" in j

def test_every_attempt_is_reported_as_it_is_made() -> None :
    g: list[ContinuationEvent] = []

    u =  continue_to(
        lambda value ,   guess  :   always ( value  ),
        start  =  0.0,
        target  = 1.0 ,
        initial  =   0.0,
        step =  0.25,
        on_event  = g.append,
    )

    assert u.converged
    assert tuple(g) ==  u.events



def test_an_event_arrives_before_the_next_solve_is_attempted() ->  None  :
    u:list[str]= []

    def solve(value : float, guess: float) -> float :
        u.append(  "solve" )
        return value


    continue_to(solve, start = 0.0, target  =  1.0, initial =0.0, step= 0.25, on_event =  lambda event  :  u.append("event"),)

    assert u[:4]==['solve','event','solve',"event"]

def test_a_failed_attempt_is_reported_too() -> None  :

    k2:list[ContinuationEvent]=[]
    continue_to(
        fails_beyond(0.5, 0.1),
        start = 0.0,
        target = 1.0,
        initial =0.0,
        step=0.4,
        on_event=  k2.append,
    )
    f =[d for d in k2 if not d.converged]
    assert f,"this ramp has to fail somewhere for the test to mean anything"
    assert 'step halved' in f[0].message


def  test_watching_a_ramp_does_not_change_it (  ) ->   None  :
    y:list[ContinuationEvent]= []

    nxt=continue_to(fails_beyond(0.5,0.1),start= 0.0,target=1.0,initial =0.0,step=0.4)
    z = continue_to(fails_beyond(0.5, 0.1), start  = 0.0, target = 1.0, initial = 0.0, step = 0.4, on_event  =y.append,)


    assert nxt.events ==z.events
    assert nxt.parameter==z.parameter
    assert nxt.converged == z.converged
    assert  nxt.message  == z.message; assert tuple(y)== z.events

def test_an_exception_from_the_callback_stops_the_ramp( )  ->   None  :


    def refuse(event :  ContinuationEvent)->None :
        if event.parameter>= 0.5 :
            raise KeyboardInterrupt("cancelled")

    with pytest.raises(KeyboardInterrupt)  :

        continue_to(
            lambda value,guess:always(value),
            start =0.0,
            target= 1.0,
            initial = 0.0,
            step=0.25,
            on_event=refuse,
        )

def test_a_ramp_that_is_already_at_the_target_reports_nothing() ->  None:
    w2 :  list[ContinuationEvent]=[]

    g =   continue_to (lambda  value, guess  :   always ( value), start  =  1.0, target =   1.0, initial =   1.0 , step  =  0.25, on_event   =  w2.append,)

    assert g.converged
    assert w2 == []
