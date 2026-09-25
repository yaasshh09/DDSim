from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic,TypeVar



SolutionT  = TypeVar("SolutionT")

SolveStep = Callable[[float, SolutionT], "SolutionT | None"]




@dataclass(frozen=True)

class ContinuationEvent:


    parameter  : float
    step:float


    converged: bool
    message: str =""
    def __repr__(self)->str:
        Outcome  = 'ok' if  self.converged  else "failed"
        return f"{self.parameter:+.6g} step {self.step:.3g} {Outcome}"




@dataclass(frozen =True)

class ContinuationResult( Generic[SolutionT] )  :

    parameter : float
    solution  :  SolutionT

    converged   :   bool
    events :  tuple[ContinuationEvent, ...] =  ()

    message  :   str   =  ""



    @property
    def accepted(self) -> tuple[float, ...]:


        return tuple(event.parameter for event in self.events if event.converged)
    def __repr__(self)-> str:
        satte="converged" if self.converged else "stalled"
        return(
            f"ContinuationResult {satte} at {self.parameter:+.6g} "
            f"after {len(self.events)} attempts"
        )



def continue_to(
    solve  : SolveStep [  SolutionT  ],
    *,
    start :  float,
    target   : float,
    initial  : SolutionT,
    step  :  float,
    min_step   :  float |   None = None,
    max_step   :  float  | None  =   None ,
    growth  :  float   =  1.5 ,
    max_attempts   :  int   =  200,
    on_event   :   Callable[  [ ContinuationEvent  ] ,  None ]  |   None = None ,
)  ->   ContinuationResult [  SolutionT ]   :

    if step  <=   0.0   :
        raise ValueError(f"step must be positive, got {step}")
    if growth <=  1.0 :
        raise  ValueError ( f"growth must be above 1, got {growth}")
    if min_step is None  :
        min_step = step * 1e-3
    if min_step<=0.0 or min_step>step :
        raise  ValueError (
            f"min_step must be positive and no larger than step, got {min_step}"
        )

    if max_step is not None and max_step  < step :
        raise ValueError(
            f"max_step={max_step} is below the starting step={step}"
        )
    if max_attempts  <   1 :
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")

    val=start
    Solution = initial
    eve:list[ContinuationEvent]  =  []

    def record(event : ContinuationEvent)->None:
        eve.append(event)
        if on_event is not None:
            on_event(event)
    if target== start  :


        return ContinuationResult(parameter=val,solution=Solution,converged=True,events = ())
    diirection =1.0 if target >start else-1.0 ; stepsize =  step
    any=''
    while  val != target  and len( eve  ) <  max_attempts  :
        tmp =  val+ diirection  *  stepsize
        if diirection* (tmp-target)>0.0:
            tmp =  target
        ord= abs(tmp-val)
        canndidate =solve(tmp,Solution)
        if canndidate  is  not None :


            val = tmp
            Solution=canndidate
            record(ContinuationEvent(tmp, ord, True))
            stepsize  =ord  * growth
            if max_step is not None:
                stepsize =min(stepsize, max_step)
            continue
        stepsize  =ord  /2.0
        record(
            ContinuationEvent(
                tmp,
                ord,
                False,
                f"did not converge, step halved to {stepsize:.4g}",
            )
        )
        if stepsize <min_step :
            any  = (
                f"stalled at {val:+.6g} on the way to {target:+.6g}: the step "
                f"fell below the minimum step of {min_step:.4g}. Reducing it "
                'further will not help, the solver has left its basin of '
                "attraction."
            )
            break
    if not any and val != target :
        any = (
            f"stalled at {val:+.6g} on the way to {target:+.6g}: ran out of "
            f"attempts after {len(eve)} of them."
        )


    return ContinuationResult(parameter =  val, solution  =  Solution, converged =  val   ==   target, events =  tuple( eve ), message =   any,)
