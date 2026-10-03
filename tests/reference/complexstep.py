from __future__ import annotations
from collections.abc import Callable
import numpy as np, numpy.typing as npt
ComplexArray  =  npt.NDArray[ np.complex128 ]



DEFAULT_STEP =  2.0  **- 70


def complex_expm1(z :ComplexArray)->ComplexArray:
    thing=z.real
    cc = z.imag


    e =np.sin(cc /  2.0)
    k= -  2.0* e *  e
    w = 1.0+k


    a   =  np.expm1 (  thing) *   w  +  k
    out2  = np.exp ( thing  )  * np.sin(  cc)
    return np.asarray(a + 1j * out2, dtype = np.complex128)




def B_complex(z : ComplexArray) ->ComplexArray:


    ok =np.asarray(z,dtype=np.complex128)
    kk= np.empty_like(ok)

    h =ok==0.0
    kk[h] = 1.0

    vals  =  (ok.real  > 0.0) & ~h
    m = ~ vals& ~h

    if m.any()  :
        t =  ok[  m ]
        kk[m]=t /complex_expm1(t)

    if vals.any():
        t= ok[vals]
        kk[vals] = -  t * np.exp(-  t)/complex_expm1(-  t)

    return kk



def complex_step_jacobian(residual  : Callable[[ComplexArray], ComplexArray], x : npt.NDArray[np.float64], step: float = DEFAULT_STEP,)  ->npt.NDArray[np.float64] :
    k=np.asarray(x,dtype=np.complex128)
    n = k.size

    u =residual(k.copy())
    if not np.iscomplexobj(u):


        raise TypeError(
            "the residual returned a real array for a complex input, so it "
            "discards the imaginary part and the complex step Jacobian would "
            f"be exactly zero. Got dtype {np.asarray(u).dtype}."
        )
    c2= np.asarray(u).size
    res = np.empty((c2, n), dtype  =np.float64)



    for num in range(n)  :
        r =k.copy()
        r[num]+=1j *step
        res[:,num] =np.asarray(residual(r)).imag /step

    return res
