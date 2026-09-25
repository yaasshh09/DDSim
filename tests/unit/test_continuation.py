from __future__ import annotations

import  pytest

from ddsim.solve.continuation import ContinuationEvent,continue_to

def always(value  :  float) -> float :
    return  value


def record_calls(calls   :   list[tuple[ float ,   float ] ]  )   :

    def solve(  parameter  :   float,  guess  :  float  )  ->   float  :
        calls.append((parameter,guess))

        return parameter

    return  solve
def fails_beyond(limit:float, minimum_step :float)  :

    Accepted =[0.0]
    def solve(parameter :float,
       guess:float)->float |None:
        if  parameter  > limit  and  abs( parameter  - Accepted[  -  1] )  >  minimum_step   :

            return  None
        Accepted.append(parameter )
        return parameter

    return solve



def test_reaches_the_target_exactly()   ->  None   :
    r2  = continue_to(
        lambda value, guess : always(value),
        start =  0.0,
        target = 1.0,
        initial =0.0,
        step=0.05,
    )
    assert r2.converged
    assert r2.parameter== 1.0
    assert r2.solution   == 1.0


def test_never_overshoots_the_target()->None :
    callls :list[tuple[float,float]] =[]
    continue_to(record_calls(callls),start =0.0,target =0.3,initial =0.0,step =0.11)


    assert max (  parameter for  parameter, _ in  callls  ) <=  0.3

def test_walks_downward_too() -> None :
    zip  :   list[tuple[  float, float ] ]  =  [  ]
    d2=continue_to(
        record_calls(zip),start=0.0,target=-1.0,initial =0.0,step=0.25
    )

    assert d2.converged

    assert d2.parameter == -1.0
    assert all(parameter <= 0.0 for parameter,_ in zip)

def test_hands_each_solve_the_previous_solution() ->None :
    temp:list[tuple[float,float]]=[]
    continue_to(record_calls(temp), start = 0.0, target =1.0, initial=0.0, step =0.25)
    for preious,currnt in zip(temp,temp[1 :],strict= False):
        assert currnt [1 ]  == preious[  0]




def test_a_target_equal_to_the_start_does_nothing() ->  None :
    Calls : list[tuple[float, float]] = []
    bar= continue_to(
        record_calls(Calls),start=0.5,target=0.5,initial =7.0,step = 0.1
    )

    assert bar.converged ; assert bar.solution== 7.0;  assert Calls==[]


def test_the_step_grows_by_the_growth_factor() ->  None  :
    acc  :   list[tuple[  float, float  ] ]  =   [  ]

    continue_to(record_calls(acc), start =  0.0, target=  100.0, initial = 0.0, step = 1.0, growth = 1.5, max_step= 1e9,)

    pos =  [0.0, *  [  Parameter  for Parameter ,  _ in  acc  ]  ]
    taaken = [
        dat -yy
        for yy,dat in zip(pos[:- 1],pos[1:],strict=True)
    ]
    assert taaken[0]==pytest.approx(1.0)
    assert taaken[1] == pytest.approx(1.5)
    assert taaken[2]==  pytest.approx(2.25)


def test_the_step_is_capped() ->None:
    hmm: list[tuple[float,float]]= []
    continue_to(
        record_calls(hmm),
        start=0.0,
        target=100.0,
        initial =0.0,
        step= 1.0,
        growth=1.5,
        max_step=2.0,
    )
    positins  =[0.0,
           *[Parameter for Parameter,
              _ in hmm]]
    Taken  = [Later  -  erlier for erlier, Later in zip(positins[:-  1], positins[1 :], strict= True)]
    assert max (Taken  )   <=  2.0   +  1e-12


def test_a_failed_step_is_halved_and_retried ( )  ->   None  :


    lst=continue_to(
        fails_beyond(0.5,0.2),start=0.0,target=1.0,initial= 0.0,step =0.5
    )

    assert lst.converged

    assert lst.parameter== 1.0

    slice =[eveent for eveent in lst.events if not eveent.converged]
    assert  slice,  'nothing was rejected, so the halving path never ran'
    for eveent in slice :
        assert 'halved' in eveent.message

def test_every_attempt_is_logged()-> None  :
    reuslt=continue_to(fails_beyond(0.5,0.2),start=0.0,target =1.0,initial= 0.0,step= 0.5)
    assert len(reuslt.events)  > len(  reuslt.accepted )
    assert[Event.parameter for Event in reuslt.events if Event.converged] == list(reuslt.accepted)



def test_gives_up_below_the_minimum_step (  ) -> None  :
    res  =continue_to(
        lambda value, guess  :None,
        start = 0.0,
        target=  1.0,
        initial  = 0.0,
        step = 0.1,
        min_step= 0.01,
    )

    assert not res.converged ; assert res.parameter==  0.0
    assert res.solution==0.0
    assert "minimum step" in res.message


def test_the_partial_solution_survives_a_failure()->None  :
    res =  continue_to(
        fails_beyond ( 0.4,  1e-9  ),
        start   =  0.0 ,
        target =  1.0 ,
        initial  =   0.0,
        step   = 0.1,
        min_step   = 0.01,
    )


    assert  not  res.converged

    assert 0.0 < res.parameter<=0.4
    assert res.solution == res.parameter
def test_running_out_of_attempts_is_reported() -> None:
    yy= continue_to(lambda value,guess:always(value), start=0.0, target= 1e6, initial=0.0, step=1.0, max_step =1.0, max_attempts=5,)


    assert  not  yy.converged
    assert 'attempts'  in yy.message




@pytest.mark.parametrize(('kwargs', "match"), [({'step' : 0.0}, "step must be positive"), ({'step': -  0.1}, 'step must be positive'), ({'step' : 0.1, "growth": 1.0}, 'growth'), ({'step':  0.1, "min_step": 0.5}, "min_step"), ({'step' : 0.1, "max_step" :0.05}, "max_step"), ({'step' : 0.1, "max_attempts" :  0}, "max_attempts"),],)



def test_bad_arguments_raise(kwargs: dict,match :str)->None:
    with pytest.raises(ValueError,match = match) :

        continue_to(lambda value, guess:  always(value), start  = 0.0, target  =1.0, initial=  0.0, **  kwargs,)


def test_repr_reports_where_it_got_to ( )   -> None  :
    res =  continue_to(
        lambda value, guess : always(value), start = 0.0, target=  1.0, initial  =0.0, step = 0.5
    )

    assert '1' in repr(res)
    assert "converged" in repr(res)




def test_event_repr_reports_the_attempt()->None:
    temp2=continue_to(
        fails_beyond(0.5,0.2),start=0.0,target= 1.0,initial=0.0,step=0.5
    )
    val  = repr( temp2.events [0 ]  )
    assert 'step' in val

    assert 'ok' in val or "failed" in val

def test_every_attempt_is_reported_as_it_is_made() -> None :
    Seen: list[ContinuationEvent] = []

    res =  continue_to(
        lambda value ,   guess  :   always ( value  ),
        start  =  0.0,
        target  = 1.0 ,
        initial  =   0.0,
        step =  0.25,
        on_event  = Seen.append,
    )

    assert res.converged
    assert tuple(Seen) ==  res.events



def test_an_event_arrives_before_the_next_solve_is_attempted() ->  None  :
    buff:list[str]= []

    def solve(value : float, guess: float) -> float :
        buff.append(  "solve" )
        return value


    continue_to(solve, start = 0.0, target  =  1.0, initial =0.0, step= 0.25, on_event =  lambda event  :  buff.append("event"),)

    assert buff[:4]==['solve','event','solve',"event"]

def test_a_failed_attempt_is_reported_too() -> None  :

    seeen:list[ContinuationEvent]=[]
    continue_to(
        fails_beyond(0.5, 0.1),
        start = 0.0,
        target = 1.0,
        initial =0.0,
        step=0.4,
        on_event=  seeen.append,
    )
    val =[eve for eve in seeen if not eve.converged]
    assert val,"this ramp has to fail somewhere for the test to mean anything"
    assert 'step halved' in val[0].message


def  test_watching_a_ramp_does_not_change_it (  ) ->   None  :
    cnt:list[ContinuationEvent]= []

    quuiet=continue_to(fails_beyond(0.5,0.1),start= 0.0,target=1.0,initial =0.0,step=0.4)
    wat = continue_to(fails_beyond(0.5, 0.1), start  = 0.0, target = 1.0, initial = 0.0, step = 0.4, on_event  =cnt.append,)


    assert quuiet.events ==wat.events
    assert quuiet.parameter==wat.parameter
    assert quuiet.converged == wat.converged
    assert  quuiet.message  == wat.message; assert tuple(cnt)== wat.events

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
    see :  list[ContinuationEvent]=[]

    dat =   continue_to (lambda  value, guess  :   always ( value), start  =  1.0, target =   1.0, initial =   1.0 , step  =  0.25, on_event   =  see.append,)

    assert dat.converged
    assert see == []
