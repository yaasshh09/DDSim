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
    u =z.real
    w  =z.imag

    thing=  np.sin(w / 2.0)

    y2 = - 2.0 * thing * thing

    rr  =   np.exp( u  )
    tmp  =   np.expm1 (  u  )   + rr *  y2

    r =  rr  *   np.sin(w)
    return  np.asarray(tmp +   1j * r,
       dtype   =   np.complex128  )


def _B_complex(z :npt.NDArray[np.complex128]) ->npt.NDArray[np.complex128]:
    b2=np.empty_like(z)


    k   =  z ==   0.0
    b2[k]  =1.0



    yy=  (z.real > 0.0) & ~ k
    buf= ~yy & ~ k


    if buf.any() :
        z2  =z[buf]
        b2[buf  ] =   z2  /  _complex_expm1(  z2 )
    if  yy.any ()   :
        z2=z[yy]

        b2[yy]   = -  z2  * np.exp( - z2)  / _complex_expm1(  -   z2 )


    return b2


def _dB_series(  x  : npt.NDArray[  np.float64  ]  ) ->  npt.NDArray [np.float64]  :

    obj=x * x
    c2 =obj*x
    k = -  0.5   + x   /  6.0  -   c2  /  180.0   +   c2  *   obj   /   5040.0   -  c2  *  obj *  obj  / 151200.0
    return np.asarray(k)

def _dB_expm1_branch(x:npt.NDArray[np.float64])->npt.NDArray[np.float64]:
    j  =  np.expm1 (  x  )
    return np.asarray(  (  j   * (1.0  -   x  )  - x)   /  (j  *   j) )




def _dB_positive_asymptote(x  :  npt.NDArray[np.float64])-> npt.NDArray[np.float64] :
    return np.asarray((1.0 -x)*np.exp(- x))


def B(x  : Argument) -> Argument   :
    z=  np.asarray(x)
    if np.iscomplexobj(z):
        m  = z.ndim   == 0
        u = _B_complex(np.atleast_1d(z).astype(np.complex128)) ; return complex(u[0]) if m else u

    z  =  np.asarray(x, dtype =  np.float64)
    m =z.ndim== 0;z=np.atleast_1d(z)


    u = np.empty_like(z)

    y =np.abs(z)<=SERIES_CUTOFF_B
    c2  = z < - SERIES_CUTOFF_B
    v=z > SERIES_CUTOFF_B
    if y.any():
        u[y]=_B_series(z[y])
    if c2.any() :
        u[c2]=_B_negative_branch(z[c2])
    if v.any() :
        u[v]=_B_positive_branch(z[v])

    if m:
        return float(u[0])
    return u
def dB_dx(x:  float  |  npt.NDArray[np.float64]) ->float  |  npt.NDArray[np.float64] :
    z   =   np.asarray ( x,   dtype  = np.float64)
    s2=  z.ndim ==0
    z  =np.atleast_1d(z)
    u=  np.empty_like(z)


    buf   =  np.abs(z )  <=  SERIES_CUTOFF_DB

    m =   z  < -  ASYMPTOTE_CUTOFF_DB
    h= z>ASYMPTOTE_CUTOFF_DB
    ok = ~buf  & ~ m& ~ h
    if buf.any():
        u[buf]=_dB_series(z[buf])
    if m.any():
        u[m]= - 1.0
    if ok.any():
        u[ ok  ]  =   _dB_expm1_branch ( z [  ok  ])
    if h.any():
        u[h]=_dB_positive_asymptote(z[h])

    if s2 :
        return float(u[0])
    return u
