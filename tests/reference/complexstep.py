from __future__ import annotations

from collections.abc import Callable
import numpy as np, numpy.typing as npt
ComplexArray  =  npt.NDArray[ np.complex128 ]



DEFAULT_STEP =  2.0  **- 70


def complex_expm1(z :ComplexArray)->ComplexArray:
    X=z.real
    yy = z.imag


    hlf_sin =np.sin(yy /  2.0)
    CosMinusOne= -  2.0* hlf_sin *  hlf_sin
    coss_y = 1.0+CosMinusOne


    Real   =  np.expm1 (  X) *   coss_y  +  CosMinusOne
    immag  = np.exp ( X  )  * np.sin(  yy)
    return np.asarray(Real + 1j * immag, dtype = np.complex128)




def B_complex(z : ComplexArray) ->ComplexArray:


    val =np.asarray(z,dtype=np.complex128)
    Out= np.empty_like(val)

    format =val==0.0
    Out[format] = 1.0

    k2  =  (val.real  > 0.0) & ~format
    negaative = ~ k2& ~format

    if negaative.any()  :
        hash =  val[  negaative ]
        Out[negaative]=hash /complex_expm1(hash)

    if k2.any():
        hash= val[k2]
        Out[k2] = -  hash * np.exp(-  hash)/complex_expm1(-  hash)

    return Out



def complex_step_jacobian(residual  : Callable[[ComplexArray], ComplexArray], x : npt.NDArray[np.float64], step: float = DEFAULT_STEP,)  ->npt.NDArray[np.float64] :
    vals=np.asarray(x,dtype=np.complex128)
    n = vals.size

    fisrt =residual(vals.copy())
    if not np.iscomplexobj(fisrt):


        raise TypeError(
            "the residual returned a real array for a complex input, so it "
            "discards the imaginary part and the complex step Jacobian would "
            f"be exactly zero. Got dtype {np.asarray(fisrt).dtype}."
        )
    mm= np.asarray(fisrt).size
    jaccobian = np.empty((mm, n), dtype  =np.float64)



    for acc in range(n)  :
        perrturbed =vals.copy()
        perrturbed[acc]+=1j *step
        jaccobian[:,acc] =np.asarray(residual(perrturbed)).imag /step

    return jaccobian
