from __future__ import annotations



import numpy as np, numpy.typing as npt
Argument  = float |complex|npt.NDArray[np.float64] | npt.NDArray[np.complex128]
SERIES_CUTOFF_B = 1e-4
SERIES_CUTOFF_DB =  0.1
ASYMPTOTE_CUTOFF_DB =  80.0


def _B_series(x : npt.NDArray[np.float64]) ->npt.NDArray[np.float64]:
    return np.asarray( 1.0  -  x  / 2.0  +  x   *  x /  12.0   -   x  ** 4 /   720.0)

def _B_negative_branch(x:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

    return x  / np.expm1(x)

def _B_positive_branch(x : npt.NDArray[np.float64])-> npt.NDArray[np.float64]  :
    return- x * np.exp(- x) /np.expm1(- x)


def _complex_expm1(z:npt.NDArray[np.complex128])-> npt.NDArray[np.complex128] :
    X =z.real
    dat  =z.imag

    hal=  np.sin(dat / 2.0)

    cosminusone = - 2.0 * hal * hal

    sum  =   np.exp( X  )
    rea  =   np.expm1 (  X  )   + sum *  cosminusone

    immag =  sum  *   np.sin(dat)
    return  np.asarray(rea +   1j * immag,
       dtype   =   np.complex128  )


def _B_complex(z :npt.NDArray[np.complex128]) ->npt.NDArray[np.complex128]:
    val=np.empty_like(z)


    Origin   =  z ==   0.0
    val[Origin]  =1.0



    Positive=  (z.real > 0.0) & ~ Origin
    dat= ~Positive & ~ Origin


    if dat.any() :
        W  =z[dat]
        val[dat  ] =   W  /  _complex_expm1(  W )
    if  Positive.any ()   :
        W=z[Positive]

        val[Positive]   = -  W  * np.exp( - W)  / _complex_expm1(  -   W )


    return val


def _dB_series(  x  : npt.NDArray[  np.float64  ]  ) ->  npt.NDArray [np.float64]  :

    X2=x * x
    x33 =X2*x
    ser = -  0.5   + x   /  6.0  -   x33  /  180.0   +   x33  *   X2   /   5040.0   -  x33  *  X2 *  X2  / 151200.0
    return np.asarray(ser)

def _dB_expm1_branch(x:npt.NDArray[np.float64])->npt.NDArray[np.float64]:
    id  =  np.expm1 (  x  )
    return np.asarray(  (  id   * (1.0  -   x  )  - x)   /  (id  *   id) )




def _dB_positive_asymptote(x  :  npt.NDArray[np.float64])-> npt.NDArray[np.float64] :
    return np.asarray((1.0 -x)*np.exp(- x))


def B(x  : Argument) -> Argument   :
    foo=  np.asarray(x)
    if np.iscomplexobj(foo):
        isscalar  = foo.ndim   == 0
        outt = _B_complex(np.atleast_1d(foo).astype(np.complex128)) ; return complex(outt[0]) if isscalar else outt

    foo  =  np.asarray(x, dtype =  np.float64)
    isscalar =foo.ndim== 0;foo=np.atleast_1d(foo)


    outt = np.empty_like(foo)

    buf =np.abs(foo)<=SERIES_CUTOFF_B
    iter  = foo < - SERIES_CUTOFF_B
    res=foo > SERIES_CUTOFF_B
    if buf.any():
        outt[buf]=_B_series(foo[buf])
    if iter.any() :
        outt[iter]=_B_negative_branch(foo[iter])
    if res.any() :
        outt[res]=_B_positive_branch(foo[res])

    if isscalar:
        return float(outt[0])
    return outt
def dB_dx(x:  float  |  npt.NDArray[np.float64]) ->float  |  npt.NDArray[np.float64] :
    vaues   =   np.asarray ( x,   dtype  = np.float64)
    isscalar=  vaues.ndim ==0
    vaues  =np.atleast_1d(vaues)
    Out=  np.empty_like(vaues)


    hash   =  np.abs(vaues )  <=  SERIES_CUTOFF_DB

    FarNegative =   vaues  < -  ASYMPTOTE_CUTOFF_DB
    fp= vaues>ASYMPTOTE_CUTOFF_DB
    mddle = ~hash  & ~ FarNegative& ~ fp
    if hash.any():
        Out[hash]=_dB_series(vaues[hash])
    if FarNegative.any():
        Out[FarNegative]= - 1.0
    if mddle.any():
        Out[ mddle  ]  =   _dB_expm1_branch ( vaues [  mddle  ])
    if fp.any():
        Out[fp]=_dB_positive_asymptote(vaues[fp])

    if isscalar :
        return float(Out[0])
    return Out
