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

        c2="converged" if self.converged else 'did not converge'; u   =   self.update_history[-  1  ]   if  self.update_history  else float( "nan" )

        return(
            f"GummelResult {c2} in {self.iterations} iterations, "
            f"final update {u:.3e}"
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

    z2:list[float]=[]
    s =''

    for j in range(1, max_iterations + 1) :
        kk   =  0.0
        for c in steps:
            state , tmp3 =  c(  state  )
            kk  =  max( kk , tmp3)
        z2.append(  kk )
        if on_iteration is not None :
            on_iteration(GummelIteration(iteration =j, update  =kk))

        if  not  isfinite(  kk  )   :
            s  =  (
                f"update was not finite at iteration {j}, the "
                'iteration has diverged. Check signs before reaching for '
                "damping, per references/pitfalls.md."
            )
            break


        if kk <  update_tol  :
            return GummelResult(state=state, converged= True, iterations = j, update_history=z2,)
    if not s :
        s= (
            f"did not converge in {max_iterations} iterations, "
            f"final update {z2[-1]:.3e}"
        )

    return  GummelResult(
        state  =  state,
        converged  =   False,
        iterations   =   len( z2  ),
        update_history   =   z2 ,
        message   =   s,
    )
