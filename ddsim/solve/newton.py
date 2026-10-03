from __future__ import annotations
from ddsim.core.config import CONFIG
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
        u="converged" if self.converged else "did not converge"
        y =self.residual_history[-1]if self.residual_history else float("nan")
        return(
            f"NewtonResult {u} in {self.iterations} iterations, "
            f"final residual {y:.3e}"
        )




def newton_solve(
    assemble:Callable[[npt.NDArray[np.float64]],Assembly],
    x0:npt.NDArray[np.float64],
    max_step: float| None= None,
    limit :Callable[[npt.NDArray[np.float64]],npt.NDArray[np.float64]] |None= None,
    residual_atol : float =CONFIG.newton.residual_atol,
    residual_rtol: float= CONFIG.newton.residual_rtol,
    residual_scale: float |None= None,
    residual_norm: (
        Callable[[npt.NDArray[np.float64],npt.NDArray[np.float64]],float]|None
    )=None,
    update_tol : float= CONFIG.newton.update_tol,
    update_norm : (
        Callable[[npt.NDArray[np.float64],npt.NDArray[np.float64]],float]|None
    ) =None,
    max_iterations: int= CONFIG.newton.max_iterations,
    stagnation_window: int|None=CONFIG.newton.stagnation_window,
    solver:SparseLU|None= None,
    on_iteration:Callable[[NewtonIteration],None] |None=None,
) ->NewtonResult :
    if max_step is not None and limit is not None:

        raise ValueError('max_step and limit are two damping rules for one update. Pass ' 'one. Letting either win silently makes the other look ineffective.')
    i   = np.array (x0,  dtype  =   np.float64, copy  = True  )

    if solver is None :
        solver =SparseLU()
    c : list[float ]  = [ ]
    r  :  list[  float ] =  []
    val2=0
    w = ''
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

    nxt = assemble(i)
    z  = measure( nxt,  i )
    c.append(z)

    report(0,z,None,None,False)


    zz=z if residual_scale is None else abs(residual_scale)
    tmp2=residual_atol +residual_rtol  * zz

    if z< tmp2:
        return NewtonResult(
            x  = i,
            converged =  True,
            iterations =0,
            residual_history = c,
            update_history=r,
        )
    for u in range(1,max_iterations +1):

        try:
            solver.factorize(nxt.rows,nxt.cols,nxt.values,nxt.shape)
            xs  =  solver.solve( -  nxt.residual)
        except  RuntimeError as  rr   :

            w = f"linear solve failed at iteration {u}: {rr}"

            break
        if not np.all(np.isfinite(xs)) :

            w =f"non-finite Newton update at iteration {u}"
            break


        m2= xs
        m=float(np.max(np.abs(xs)))
        d  = m
        row= False
        if max_step is not None and d >  max_step:
            xs = xs  *(max_step /  d)
            d   =   max_step
            val2+=1
            row = True
        elif limit is not None:
            t=np.asarray(limit(xs),dtype = np.float64)
            if  t.shape  !=  xs.shape :
                raise ValueError(
                    f"limit returned shape {t.shape} for an update of "
                    f"shape {xs.shape}. Dropping entries would freeze "
                    "those unknowns at their starting values."
                )
            if not np.array_equal(t,xs):
                val2 += 1
                row=True
            xs=t


        h  =  m2 !=   0.0
        tt= (
            float(np.min(np.abs(xs[h]) / np.abs(m2[h])))
            if np.any(h)
            else 1.0
        )


        d = (float(np.max(np.abs(xs))) if update_norm is None else float(update_norm(xs, i)))
        i  =  i  +   xs
        r.append(d)


        try :
            nxt =   assemble(  i  )
            flag=bool(np.all(np.isfinite(nxt.residual)))

        except FloatingPointError:
            flag=False
        if not flag:
            w= (
                f"residual became non-finite at iteration {u}, "
                'the iterate has diverged. Try a smaller max_step, but check '
                'signs before reaching for damping.'
            )
            c.append( float (  "inf"))

            report(u,float("inf"),d,tt,row)
            break

        z  =  measure(nxt, i); c.append ( z )
        report(u,z,d,tt,row)

        if d< update_tol and z<tmp2:
            return NewtonResult(x =  i, converged =True, iterations = u, residual_history = c, update_history  =  r, limited_steps=  val2,)
        if(
            stagnation_window is not None
            and d< update_tol
            and len(c)>= stagnation_window
            and len(set(c[- stagnation_window:]))== 1
        ):
            w=  (
                f"the residual stopped moving at iteration {u}: "
                f"{z:.3e} unchanged over the last "
                f"{stagnation_window} evaluations, against a threshold of "
                f"{tmp2:.3e}, with the update already down to "
                f"{d:.3e}. The residual is on its arithmetic floor "
                "and the remaining budget cannot move it. Either the "
                "threshold is below that floor, in which case pass a "
                "residual_scale built from the size of the terms, or the "
                'Jacobian is wrong.'
            )
            break
    if  not w   :
        w   =  (
            f"did not converge in {max_iterations} iterations, "
            f"final residual {c[-1]:.3e} "
            f"against a threshold of {tmp2:.3e}, "
            f"final update {r[-1] if r else float('nan'):.3e}"
        )

    return NewtonResult(
        x =i,
        converged=False,
        iterations=len(r),
        residual_history=c,
        update_history= r,
        limited_steps = val2,
        message=w,
    )
