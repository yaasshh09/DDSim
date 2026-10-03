from __future__ import annotations
from ddsim.core.config import CONFIG
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
        hh  = 'ok' if  self.converged  else "failed"
        return f"{self.parameter:+.6g} step {self.step:.3g} {hh}"




@dataclass(frozen =True)

class ContinuationResult( Generic[SolutionT] )  :

    parameter : float
    solution  :  SolutionT

    converged   :   bool
    events :  tuple[ContinuationEvent, ...] =  ()

    message  :   str   =  ""



    @property
    def accepted(self) -> tuple[float, ...]:


        return tuple(m.parameter for m in self.events if m.converged)
    def __repr__(self)-> str:
        info="converged" if self.converged else "stalled"
        return(
            f"ContinuationResult {info} at {self.parameter:+.6g} "
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
    growth  :  float   =  CONFIG.continuation.growth ,
    max_attempts   :  int   =  CONFIG.continuation.max_attempts,
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

    d2=start
    v2 = initial
    r:list[ContinuationEvent]  =  []

    def record(event : ContinuationEvent)->None:
        r.append(event)
        if on_event is not None:
            on_event(event)
    if target== start  :


        return ContinuationResult(parameter=d2,solution=v2,converged=True,events = ())
    rows =1.0 if target >start else-1.0 ; c2 =  step
    a=''
    while  d2 != target  and len( r  ) <  max_attempts  :
        info =  d2+ rows  *  c2
        if rows* (info-target)>0.0:
            info =  target
        a2= abs(info-d2)
        x =solve(info,v2)
        if x  is  not None :


            d2 = info
            v2=x
            record(ContinuationEvent(info, a2, True))
            c2  =a2  * growth
            if max_step is not None:
                c2 =min(c2, max_step)
            continue
        c2  =a2  /2.0
        record(
            ContinuationEvent(
                info,
                a2,
                False,
                f"did not converge, step halved to {c2:.4g}",
            )
        )
        if c2 <min_step :
            a  = (
                f"stalled at {d2:+.6g} on the way to {target:+.6g}: the step "
                f"fell below the minimum step of {min_step:.4g}. Reducing it "
                'further will not help, the solver has left its basin of '
                "attraction."
            )
            break
    if not a and d2 != target :
        a = (
            f"stalled at {d2:+.6g} on the way to {target:+.6g}: ran out of "
            f"attempts after {len(r)} of them."
        )


    return ContinuationResult(parameter =  d2, solution  =  v2, converged =  d2   ==   target, events =  tuple( r ), message =   a,)
