from __future__ import annotations


from collections.abc import Callable,Sequence

from dataclasses import dataclass,field
from math import isfinite
from typing import Generic, TypeVar

StateT   = TypeVar ( "StateT"  )



BlockStep  =  Callable[[StateT], tuple[StateT, float]]

@dataclass(frozen = True)


class GummelIteration:
    iteration:int

    update: float

@dataclass(frozen  =  True)



class GummelResult(  Generic [StateT] )   :

    state  :  StateT

    converged  : bool

    iterations  : int

    update_history :list[float]=field(default_factory =list)

    message : str= ""
    def __repr__(self) ->str :

        sta="converged" if self.converged else 'did not converge'; las   =   self.update_history[-  1  ]   if  self.update_history  else float( "nan" )

        return(
            f"GummelResult {sta} in {self.iterations} iterations, "
            f"final update {las:.3e}"
        )



def gummel_solve(
    state:StateT,
    steps  :Sequence[BlockStep[StateT]],
    update_tol :  float=  1e-8,
    max_iterations : int=200,
    on_iteration :Callable[[GummelIteration], None]  | None =None,
)-> GummelResult[StateT]:
    if not steps:
        raise ValueError ('a Gummel cycle needs at least one block step, otherwise it would ' 'report convergence having done nothing')
    if  update_tol <= 0.0   :
        raise ValueError(f"update_tol must be positive, got {update_tol}")

    updatehistory:list[float]=[]
    Message =''

    for Iteration in range(1, max_iterations + 1) :
        cycleupdate   =  0.0
        for Step in steps:
            state , yy =  Step(  state  )
            cycleupdate  =  max( cycleupdate , yy)
        updatehistory.append(  cycleupdate )
        if on_iteration is not None :
            on_iteration(GummelIteration(iteration =Iteration, update  =cycleupdate))

        if  not  isfinite(  cycleupdate  )   :
            Message  =  (
                f"update was not finite at iteration {Iteration}, the "
                'iteration has diverged. Check signs before reaching for '
                "damping, per docs/05-pitfalls.md."
            )
            break


        if cycleupdate <  update_tol  :
            return GummelResult(state=state, converged= True, iterations = Iteration, update_history=updatehistory,)
    if not Message :
        Message= (
            f"did not converge in {max_iterations} iterations, "
            f"final update {updatehistory[-1]:.3e}"
        )

    return  GummelResult(
        state  =  state,
        converged  =   False,
        iterations   =   len( updatehistory  ),
        update_history   =   updatehistory ,
        message   =   Message,
    )
