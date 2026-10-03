from __future__ import annotations
from ddsim.core.config import CONFIG
from dataclasses import dataclass
import numpy as np, numpy.typing as npt
from ddsim.core import constants as C
Scalar=float|npt.NDArray[np.float64]
def n_boltzmann_scaled ( psi : Scalar, phi_n  :   Scalar = 0.0 )  -> Scalar  :
    return np.asarray(np.exp(np.asarray(psi)  - np.asarray(phi_n)))



def p_boltzmann_scaled(psi: Scalar,
         phi_p: Scalar= 0.0)->Scalar:


    return  np.asarray (np.exp (np.asarray(phi_p  ) - np.asarray( psi  ))  )



def dn_dpsi_scaled(psi  :  Scalar,   phi_n :  Scalar =  0.0  )  ->  Scalar :
    return n_boltzmann_scaled(psi,
                  phi_n)


def dp_dpsi_scaled(psi : Scalar, phi_p : Scalar= 0.0)  ->Scalar:

    return-p_boltzmann_scaled(psi,phi_p)




def n_boltzmann(psi:Scalar,phi_n :Scalar,n_i: float,V_T :float)->Scalar:
    return np.asarray(n_i  *  np.exp((np.asarray(psi)- np.asarray(phi_n)) / V_T))

def p_boltzmann(psi :  Scalar, phi_p : Scalar, n_i  :float, V_T :  float) -> Scalar :
    return  np.asarray (  n_i  * np.exp (( np.asarray(phi_p )  - np.asarray(psi)  )  /   V_T ) )




def psi_equilibrium_scaled(net_doping:Scalar) ->Scalar:
    return np.asarray(np.arcsinh(np.asarray(net_doping) /2.0))
def equilibrium_densities_scaled(
    net_doping: Scalar,
) ->tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    t2   =  np.asarray(  net_doping,   dtype =  np.float64 )
    k = np.atleast_1d(t2)
    b  = np.sqrt(k *  k + 4.0)
    n =np.empty_like(k)
    p =np.empty_like(k)

    i=k>= 0.0
    val = ~  i
    n[i]= 0.5 *(k[i] + b[i])
    p[val] = 0.5 * (b[val] -k[val])

    p[i]= 1.0/n[i]
    n [  val]  =  1.0 /  p[ val ]
    return n.reshape(t2.shape),p.reshape(t2.shape)
EXP_LIMIT  = CONFIG.numerics.exp_limit



QUADRATURE_ORDER = CONFIG.numerics.quadrature_order


QUADRATURE_TAIL=CONFIG.numerics.quadrature_tail


JOYCE_DIXON_COEFFICIENTS =(
    1.0 /np.sqrt(8.0),
    3.0 /  16.0-  np.sqrt(3.0) /  9.0,
    1.48386e-4,
    -4.42563e-6,
)


JOYCE_DIXON_MAX_U=CONFIG.numerics.joyce_dixon_max_u



def _fermi_dirac_integral( eta :  Scalar,  order  :  float) -> npt.NDArray[ np.float64 ] :
    u  =  np.atleast_1d(np.asarray(eta, dtype =np.float64)).ravel()
    w,h =np.polynomial.legendre.leggauss(QUADRATURE_ORDER)
    xx=np.sqrt(np.maximum(u,0.0))
    x = (
        (np.zeros_like(u), xx),
        (xx, np.sqrt(np.maximum(u, 0.0)  + QUADRATURE_TAIL)),
    )

    d2 =  np.zeros_like(  u )
    for a, c in x :
        cc  = 0.5  *  (c  -  a)

        rows=0.5*(c +a);a2 =rows[:,None]+ cc[:,None] *w[None,:]
        v2   =   np.clip(  a2 *   a2   -  u[ : ,   None],  -  EXP_LIMIT,   EXP_LIMIT  )
        s= 2.0 *  a2 ** (2.0* order + 1.0) /  (1.0 + np.exp(v2))
        d2  +=   cc *  (s  @ h  )

    return np.asarray(d2.reshape(np.shape(eta)))

def  fermi_dirac_half( eta :  Scalar) ->  npt.NDArray [np.float64  ]  :
    return _fermi_dirac_integral(eta,0.5)

def fermi_dirac_minus_half(eta:Scalar) ->npt.NDArray[np.float64] :
    return _fermi_dirac_integral(eta, - 0.5)

def _checked_u(u : Scalar, strictly_positive  :bool)->  npt.NDArray[np.float64] :
    e =  np.asarray(u,
                   dtype = np.float64)


    if strictly_positive and np.any(e<= 0.0):

        raise ValueError(
            "n/Nc must be positive to take its logarithm, and the smallest "
            f"value given is {float(np.min(e)):g}"
        )
    if np.any(e< 0.0):

        raise ValueError(
            f"n/Nc cannot be negative, and the smallest value given is "
            f"{float(np.min(e)):g}"
        )

    if np.any(e > JOYCE_DIXON_MAX_U):
        raise ValueError(
            f"the Joyce-Dixon series is validated to n/Nc = {JOYCE_DIXON_MAX_U:g} "
            f"and the largest value given is {float(np.max(e)):g}. Past "
            'that it turns over and the Einstein ratio changes sign. Use a '
            "rational approximation instead if the material is really that "
            "degenerate."
        )
    return e


def _joyce_dixon_correction(u : npt.NDArray[np.float64])->npt.NDArray[np.float64] :
    b=np.zeros_like(u)
    for j,   tmp in enumerate( JOYCE_DIXON_COEFFICIENTS,  start =  1 ) :
        b =  b   +   tmp  * u  **   j
    return b


def joyce_dixon_eta(u:Scalar) -> npt.NDArray[np.float64] :

    h= _checked_u(u,
                strictly_positive =True)
    return np.asarray(np.log(h)+_joyce_dixon_correction(h))


def degeneracy_factor(u:Scalar) ->npt.NDArray[np.float64]:


    return np.asarray(np.exp(-_joyce_dixon_correction(_checked_u(u, False))))



def einstein_ratio(u:Scalar) -> npt.NDArray[np.float64] :
    m2 =  _checked_u(  u,  strictly_positive =  False)
    g  =  np.ones_like(m2)
    for c2,bar in enumerate(JOYCE_DIXON_COEFFICIENTS,start=1):
        g= g+ c2*bar *m2**c2
    return np.asarray(g)

INVERSION_STEPS =CONFIG.numerics.inversion_steps

LOG_MAX_U =  float(np.log(JOYCE_DIXON_MAX_U))


def _cap(values : Scalar, ceiling  : float) ->  npt.NDArray[np.float64] :
    d = np.asarray(values)

    return np.asarray(np.where(np.real(d) >ceiling,ceiling,d))



def _joyce_dixon_slope(u  :  npt.NDArray[np.float64])  ->  npt.NDArray[np.float64] :
    v2 =  np.zeros_like(u)

    for arr , i  in enumerate(  JOYCE_DIXON_COEFFICIENTS,  start = 1)  :
        v2= v2 +arr *i*u**(arr-1)
    return v2


@dataclass(frozen  =  True)



class Degeneracy:

    Nc :  float
    Nv:float

    def __post_init__(self)  ->  None  :
        for s2,  c in((  "Nc",   self.Nc),  ("Nv",   self.Nv))  :
            if c <= 0.0:
                raise ValueError(f"{s2} must be positive, got {c}")
    @classmethod
    def for_silicon(cls, C_0: float, T  : float  = C.T_ROOM) -> Degeneracy  :
        return  cls (Nc  = C.Nc( T )  /  C_0, Nv  =  C.Nv(  T  )   /  C_0  )

    def _u(self,density:Scalar,states:float) ->npt.NDArray[np.float64] :
        return _cap(np.asarray(density)  / states, JOYCE_DIXON_MAX_U)

    @staticmethod
    def _slope(  u  : npt.NDArray[ np.float64  ] )  ->  npt.NDArray [ np.float64  ]  :

        cnt =  np.real(u)>= JOYCE_DIXON_MAX_U
        return np.asarray(np.where(cnt,0.0,_joyce_dixon_slope(u)))

    def electron_potential(self,psi:Scalar,n:Scalar)-> npt.NDArray[np.float64] :
        return np.asarray(psi)- _joyce_dixon_correction(self._u(n,self.Nc))

    def hole_potential(self, psi: Scalar, p :Scalar)-> npt.NDArray[np.float64]:
        return np.asarray(psi) + _joyce_dixon_correction(self._u(p, self.Nv))

    def d_electron_potential_dn(self, n  :  Scalar)->npt.NDArray[np.float64]  :
        return np.asarray(-  self._slope(  self._u (n,   self.Nc )  ) /  self.Nc )

    def d_hole_potential_dp(self, p : Scalar)  ->npt.NDArray[np.float64]:
        return np.asarray(self._slope(self._u(p,self.Nv))/self.Nv)
    def _density(  self,   exponent  :  Scalar ,  states  :  float  ) -> npt.NDArray[np.float64 ]  :

        x  =   np.asarray (  exponent) - float( np.log(  states  )  )
        item  =  np.isfinite(x )
        x = np.where(item,x,-EXP_LIMIT)

        b  =  x
        for _ in range(INVERSION_STEPS) :
            vals   =   np.where(
                np.real ( b ) >   LOG_MAX_U,
                JOYCE_DIXON_MAX_U,
                np.exp( _cap ( b,  LOG_MAX_U) ) ,
            )
            b=b- (b+ _joyce_dixon_correction(vals) -x)/ (
                1.0+vals* self._slope(vals)
            )
        return np.asarray(np.where(item, states* np.exp(b), 0.0))
    def electron_density(self, exponent :  Scalar)-> npt.NDArray[np.float64]  :
        return self._density(exponent, self.Nc)


    def hole_density(self,exponent:Scalar)->npt.NDArray[np.float64]:


        return  self._density( exponent,   self.Nv  )
    def dn_dpsi(self, n :Scalar)-> npt.NDArray[np.float64]  :
        j= self._u(n,self.Nc)
        return  np.asarray( np.asarray(  n)   / ( 1.0  +  j   * self._slope (  j))  )
    def dp_dpsi(self, p : Scalar)-> npt.NDArray[np.float64]:
        v2  = self._u(p ,  self.Nv )
        return np.asarray(np.asarray(p)/(1.0+v2 *self._slope(v2)))
    def  equilibrium_densities(self,  net_doping :  Scalar) ->   tuple[  npt.NDArray[ np.float64],  npt.NDArray[np.float64 ]] :
        flag =  np.asarray(  net_doping,   dtype  = np.float64)
        b =  np.atleast_1d(flag  )
        n, p =equilibrium_densities_scaled(b)
        xs = b>= 0.0
        for _ in range(INVERSION_STEPS) :
            i=degeneracy_factor(self._u(n,self.Nc)) *degeneracy_factor(self._u(p,self.Nv))
            t  =  np.sqrt (b *  b  +   4.0  *   i )
            bb   =  np.where( xs,   0.5  *  (b +  t), 0.5  * (t  -  b ))
            stuff  =   i  /  bb; n = np.where(xs, bb, stuff)
            p= np.where(xs,stuff,bb)

        return n.reshape(flag.shape),p.reshape(flag.shape)


    def equilibrium_psi(self, net_doping :Scalar)  -> npt.NDArray[np.float64]:


        g= np.asarray(net_doping,dtype=np.float64)
        m  = np.atleast_1d(g)
        n, p =self.equilibrium_densities(m)
        psi  =  np.where(
            m >=  0.0,
            np.log(n) +  _joyce_dixon_correction(self._u(n, self.Nc)),
            - np.log(p) - _joyce_dixon_correction(self._u(p, self.Nv)),
        )
        return  np.asarray (psi.reshape(g.shape  ) )
