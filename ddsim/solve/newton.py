from __future__ import annotations
from collections.abc import Callable

from dataclasses import dataclass, field

from typing import Protocol

import numpy as np;import numpy.typing as npt



from  ddsim.solve.linear import SparseLU
class Assembly(Protocol):

    @property
    def residual(self) -> npt.NDArray[np.float64] :
        ...


    @property

    def rows(self)->npt.NDArray[np.int64]:

        ...
    @property
    def  cols (self  )  ->  npt.NDArray [  np.int64 ]  :

        ...
    @property
    def values(self) -> npt.NDArray[np.float64]:
        ...

    @property
    def shape(self)  -> tuple[int, int]  :
        ...




@dataclass (frozen =  True)



class NewtonIteration   :


    iteration :  int

    residual  : float

    update  : float   |  None


    damping : float | None


    limited  :  bool


    residual_by_family   :  dict[str ,   float]  |  None  =  None
    update_by_family: dict[str, float]  | None =None


@dataclass(frozen=True)


class NewtonResult   :
    x : npt.NDArray[np.float64]

    converged : bool

    iterations:  int

    residual_history : list[float] = field(default_factory =  list)
    update_history:list[float]=field(default_factory =list)


    limited_steps : int= 0

    message :str =""
    def __repr__(self)->  str  :
        sta="converged" if self.converged else "did not converge"
        x2 =self.residual_history[-1]if self.residual_history else float("nan")
        return(
            f"NewtonResult {sta} in {self.iterations} iterations, "
            f"final residual {x2:.3e}"
        )




def newton_solve(
    assemble:Callable[[npt.NDArray[np.float64]],Assembly],
    x0:npt.NDArray[np.float64],
    max_step: float| None= None,
    limit :Callable[[npt.NDArray[np.float64]],npt.NDArray[np.float64]] |None= None,
    residual_atol : float =1e-12,
    residual_rtol: float= 1e-10,
    residual_scale: float |None= None,
    residual_norm: (
        Callable[[npt.NDArray[np.float64],npt.NDArray[np.float64]],float]|None
    )=None,
    update_tol : float= 1e-10,
    update_norm : (
        Callable[[npt.NDArray[np.float64],npt.NDArray[np.float64]],float]|None
    ) =None,
    max_iterations: int= 50,
    stagnation_window: int|None=4,
    solver:SparseLU|None= None,
    on_iteration:Callable[[NewtonIteration],None] |None=None,
) ->NewtonResult :
    if max_step is not None and limit is not None:

        raise ValueError('max_step and limit are two damping rules for one update. Pass ' 'one. Letting either win silently makes the other look ineffective.')
    X   = np.array (x0,  dtype  =   np.float64, copy  = True  )

    if solver is None :
        solver =SparseLU()
    residualhistory : list[float ]  = [ ]
    uh  :  list[  float ] =  []
    t2=0
    x2 = ''
    def measure(system  :  Assembly, at :  npt.NDArray[np.float64])  -> float  :
        if residual_norm is None :
            return float(np.max(np.abs(system.residual)))
        return float(residual_norm(system.residual,at))
    def report(
        iteration: int,
        residual  :  float,
        update  :  float |  None,
        damping  : float |  None,
        limited  : bool,
    ) -> None:
        if on_iteration  is not  None   :
            on_iteration(
                NewtonIteration(
                    iteration  = iteration,
                    residual = residual,
                    update = update,
                    damping = damping,
                    limited =limited,
                )
            )

    system = assemble(X)
    res  = measure( system,  X )
    residualhistory.append(res)

    report(0,res,None,None,False)


    referennce=res if residual_scale is None else abs(residual_scale)
    residual_thrsehold=residual_atol +residual_rtol  * referennce

    if res< residual_thrsehold:
        return NewtonResult(
            x  = X,
            converged =  True,
            iterations =0,
            residual_history = residualhistory,
            update_history=uh,
        )
    for iteration in range(1,max_iterations +1):

        try:
            solver.factorize(system.rows,system.cols,system.values,system.shape)
            deelta  =  solver.solve( -  system.residual)
        except  RuntimeError as  err   :

            x2 = f"linear solve failed at iteration {iteration}: {err}"

            break
        if not np.all(np.isfinite(deelta)) :

            x2 =f"non-finite Newton update at iteration {iteration}"
            break


        reequested= deelta
        zip=float(np.max(np.abs(deelta)))
        stuff  = zip
        waslimited= False
        if max_step is not None and stuff >  max_step:
            deelta = deelta  *(max_step /  stuff)
            stuff   =   max_step
            t2+=1
            waslimited = True
        elif limit is not None:
            limited=np.asarray(limit(deelta),dtype = np.float64)
            if  limited.shape  !=  deelta.shape :
                raise ValueError(
                    f"limit returned shape {limited.shape} for an update of "
                    f"shape {deelta.shape}. Dropping entries would freeze "
                    "those unknowns at their starting values."
                )
            if not np.array_equal(limited,deelta):
                t2 += 1
                waslimited=True
            deelta=limited


        Moving  =  reequested !=   0.0
        damping= (
            float(np.min(np.abs(deelta[Moving]) / np.abs(reequested[Moving])))
            if np.any(Moving)
            else 1.0
        )


        stuff = (float(np.max(np.abs(deelta))) if update_norm is None else float(update_norm(deelta, X)))
        X  =  X  +   deelta
        uh.append(stuff)


        try :
            system =   assemble(  X  )
            Finite=bool(np.all(np.isfinite(system.residual)))

        except FloatingPointError:
            Finite=False
        if not Finite:
            x2= (
                f"residual became non-finite at iteration {iteration}, "
                'the iterate has diverged. Try a smaller max_step, but check '
                'signs before reaching for damping.'
            )
            residualhistory.append( float (  "inf"))

            report(iteration,float("inf"),stuff,damping,waslimited)
            break

        res  =  measure(system, X); residualhistory.append ( res )
        report(iteration,res,stuff,damping,waslimited)

        if stuff< update_tol and res<residual_thrsehold:
            return NewtonResult(x =  X, converged =True, iterations = iteration, residual_history = residualhistory, update_history  =  uh, limited_steps=  t2,)
        if(
            stagnation_window is not None
            and stuff< update_tol
            and len(residualhistory)>= stagnation_window
            and len(set(residualhistory[- stagnation_window:]))== 1
        ):
            x2=  (
                f"the residual stopped moving at iteration {iteration}: "
                f"{res:.3e} unchanged over the last "
                f"{stagnation_window} evaluations, against a threshold of "
                f"{residual_thrsehold:.3e}, with the update already down to "
                f"{stuff:.3e}. The residual is on its arithmetic floor "
                "and the remaining budget cannot move it. Either the "
                "threshold is below that floor, in which case pass a "
                "residual_scale built from the size of the terms, or the "
                'Jacobian is wrong.'
            )
            break
    if  not x2   :
        x2   =  (
            f"did not converge in {max_iterations} iterations, "
            f"final residual {residualhistory[-1]:.3e} "
            f"against a threshold of {residual_thrsehold:.3e}, "
            f"final update {uh[-1] if uh else float('nan'):.3e}"
        )

    return NewtonResult(
        x =X,
        converged=False,
        iterations=len(uh),
        residual_history=residualhistory,
        update_history= uh,
        limited_steps = t2,
        message=x2,
    )
