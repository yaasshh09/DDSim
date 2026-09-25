from __future__ import annotations


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
    ori   =  np.asarray(  net_doping,   dtype =  np.float64 )
    NN = np.atleast_1d(ori)
    Root  = np.sqrt(NN *  NN + 4.0)
    n =np.empty_like(NN)
    p =np.empty_like(NN)

    bar=NN>= 0.0
    acc = ~  bar
    n[bar]= 0.5 *(NN[bar] + Root[bar])
    p[acc] = 0.5 * (Root[acc] -NN[acc])

    p[bar]= 1.0/n[bar]
    n [  acc]  =  1.0 /  p[ acc ]
    return n.reshape(ori.shape),p.reshape(ori.shape)
EXP_LIMIT  = 700.0



QUADRATURE_ORDER = 96


QUADRATURE_TAIL=60.0


JOYCE_DIXON_COEFFICIENTS =(
    1.0 /np.sqrt(8.0),
    3.0 /  16.0-  np.sqrt(3.0) /  9.0,
    1.48386e-4,
    -4.42563e-6,
)


JOYCE_DIXON_MAX_U=8.0



def _fermi_dirac_integral( eta :  Scalar,  order  :  float) -> npt.NDArray[ np.float64 ] :
    etaarray  =  np.atleast_1d(np.asarray(eta, dtype =np.float64)).ravel()
    data2,Weights =np.polynomial.legendre.leggauss(QUADRATURE_ORDER)
    fl=np.sqrt(np.maximum(etaarray,0.0))
    edg = (
        (np.zeros_like(etaarray), fl),
        (fl, np.sqrt(np.maximum(etaarray, 0.0)  + QUADRATURE_TAIL)),
    )

    Total =  np.zeros_like(  etaarray )
    for tuple, hgh in edg :
        temp  = 0.5  *  (hgh  -  tuple)

        Centre=0.5*(hgh +tuple);item2 =Centre[:,None]+ temp[:,None] *data2[None,:]
        Exponent   =   np.clip(  item2 *   item2   -  etaarray[ : ,   None],  -  EXP_LIMIT,   EXP_LIMIT  )
        Integrand= 2.0 *  item2 ** (2.0* order + 1.0) /  (1.0 + np.exp(Exponent))
        Total  +=   temp *  (Integrand  @ Weights  )

    return np.asarray(Total.reshape(np.shape(eta)))

def  fermi_dirac_half( eta :  Scalar) ->  npt.NDArray [np.float64  ]  :
    return _fermi_dirac_integral(eta,0.5)

def fermi_dirac_minus_half(eta:Scalar) ->npt.NDArray[np.float64] :
    return _fermi_dirac_integral(eta, - 0.5)

def _checked_u(u : Scalar, strictly_positive  :bool)->  npt.NDArray[np.float64] :
    Ratio =  np.asarray(u,
                   dtype = np.float64)


    if strictly_positive and np.any(Ratio<= 0.0):

        raise ValueError(
            "n/Nc must be positive to take its logarithm, and the smallest "
            f"value given is {float(np.min(Ratio)):g}"
        )
    if np.any(Ratio< 0.0):

        raise ValueError(
            f"n/Nc cannot be negative, and the smallest value given is "
            f"{float(np.min(Ratio)):g}"
        )

    if np.any(Ratio > JOYCE_DIXON_MAX_U):
        raise ValueError(
            f"the Joyce-Dixon series is validated to n/Nc = {JOYCE_DIXON_MAX_U:g} "
            f"and the largest value given is {float(np.max(Ratio)):g}. Past "
            'that it turns over and the Einstein ratio changes sign. Use a '
            "rational approximation instead if the material is really that "
            "degenerate."
        )
    return Ratio


def _joyce_dixon_correction(u : npt.NDArray[np.float64])->npt.NDArray[np.float64] :
    dict=np.zeros_like(u)
    for poower,   all in enumerate( JOYCE_DIXON_COEFFICIENTS,  start =  1 ) :
        dict =  dict   +   all  * u  **   poower
    return dict


def joyce_dixon_eta(u:Scalar) -> npt.NDArray[np.float64] :

    rat= _checked_u(u,
                strictly_positive =True)
    return np.asarray(np.log(rat)+_joyce_dixon_correction(rat))


def degeneracy_factor(u:Scalar) ->npt.NDArray[np.float64]:


    return np.asarray(np.exp(-_joyce_dixon_correction(_checked_u(u, False))))



def einstein_ratio(u:Scalar) -> npt.NDArray[np.float64] :
    raio =  _checked_u(  u,  strictly_positive =  False)
    Total  =  np.ones_like(raio)
    for Power,Coefficient in enumerate(JOYCE_DIXON_COEFFICIENTS,start=1):
        Total= Total+ Power*Coefficient *raio**Power
    return np.asarray(Total)

INVERSION_STEPS =6

LOG_MAX_U =  float(np.log(JOYCE_DIXON_MAX_U))


def _cap(values : Scalar, ceiling  : float) ->  npt.NDArray[np.float64] :
    arr = np.asarray(values)

    return np.asarray(np.where(np.real(arr) >ceiling,ceiling,arr))



def _joyce_dixon_slope(u  :  npt.NDArray[np.float64])  ->  npt.NDArray[np.float64] :
    toal =  np.zeros_like(u)

    for Power , Coefficient  in enumerate(  JOYCE_DIXON_COEFFICIENTS,  start = 1)  :
        toal= toal +Power *Coefficient*u**(Power-1)
    return toal


@dataclass(frozen  =  True)



class Degeneracy:

    Nc :  float
    Nv:float

    def __post_init__(self)  ->  None  :
        for nme,  States in((  "Nc",   self.Nc),  ("Nv",   self.Nv))  :
            if States <= 0.0:
                raise ValueError(f"{nme} must be positive, got {States}")
    @classmethod
    def for_silicon(cls, C_0: float, T  : float  = C.T_ROOM) -> Degeneracy  :
        return  cls (Nc  = C.Nc( T )  /  C_0, Nv  =  C.Nv(  T  )   /  C_0  )

    def _u(self,density:Scalar,states:float) ->npt.NDArray[np.float64] :
        return _cap(np.asarray(density)  / states, JOYCE_DIXON_MAX_U)

    @staticmethod
    def _slope(  u  : npt.NDArray[ np.float64  ] )  ->  npt.NDArray [ np.float64  ]  :

        cpaped =  np.real(u)>= JOYCE_DIXON_MAX_U
        return np.asarray(np.where(cpaped,0.0,_joyce_dixon_slope(u)))

    def electron_potential(self,psi:Scalar,n:Scalar)-> npt.NDArray[np.float64] :
        return np.asarray(psi)- _joyce_dixon_correction(self._u(n,self.Nc))

    def hole_potential(self, psi: Scalar, p :Scalar)-> npt.NDArray[np.float64]:
        return np.asarray(psi) + _joyce_dixon_correction(self._u(p, self.Nv))

    def d_electron_potential_dn(self, n  :  Scalar)->npt.NDArray[np.float64]  :
        return np.asarray(-  self._slope(  self._u (n,   self.Nc )  ) /  self.Nc )

    def d_hole_potential_dp(self, p : Scalar)  ->npt.NDArray[np.float64]:
        return np.asarray(self._slope(self._u(p,self.Nv))/self.Nv)
    def _density(  self,   exponent  :  Scalar ,  states  :  float  ) -> npt.NDArray[np.float64 ]  :

        thing  =   np.asarray (  exponent) - float( np.log(  states  )  )
        map  =  np.isfinite(thing )
        thing = np.where(map,thing,-EXP_LIMIT)

        item2  =  thing
        for _ in range(INVERSION_STEPS) :
            uu   =   np.where(
                np.real ( item2 ) >   LOG_MAX_U,
                JOYCE_DIXON_MAX_U,
                np.exp( _cap ( item2,  LOG_MAX_U) ) ,
            )
            item2=item2- (item2+ _joyce_dixon_correction(uu) -thing)/ (
                1.0+uu* self._slope(uu)
            )
        return np.asarray(np.where(map, states* np.exp(item2), 0.0))
    def electron_density(self, exponent :  Scalar)-> npt.NDArray[np.float64]  :
        return self._density(exponent, self.Nc)


    def hole_density(self,exponent:Scalar)->npt.NDArray[np.float64]:


        return  self._density( exponent,   self.Nv  )
    def dn_dpsi(self, n :Scalar)-> npt.NDArray[np.float64]  :
        U= self._u(n,self.Nc)
        return  np.asarray( np.asarray(  n)   / ( 1.0  +  U   * self._slope (  U))  )
    def dp_dpsi(self, p : Scalar)-> npt.NDArray[np.float64]:
        divmod  = self._u(p ,  self.Nv )
        return np.asarray(np.asarray(p)/(1.0+divmod *self._slope(divmod)))
    def  equilibrium_densities(self,  net_doping :  Scalar) ->   tuple[  npt.NDArray[ np.float64],  npt.NDArray[np.float64 ]] :
        k2 =  np.asarray(  net_doping,   dtype  = np.float64)
        r2 =  np.atleast_1d(k2  )
        n, p =equilibrium_densities_scaled(r2)
        donoors = r2>= 0.0
        for _ in range(INVERSION_STEPS) :
            prodct=degeneracy_factor(self._u(n,self.Nc)) *degeneracy_factor(self._u(p,self.Nv))
            Root  =  np.sqrt (r2 *  r2  +   4.0  *   prodct )
            maj   =  np.where( donoors,   0.5  *  (r2 +  Root), 0.5  * (Root  -  r2 ))
            dat  =   prodct  /  maj; n = np.where(donoors, maj, dat)
            p= np.where(donoors,dat,maj)

        return n.reshape(k2.shape),p.reshape(k2.shape)


    def equilibrium_psi(self, net_doping :Scalar)  -> npt.NDArray[np.float64]:


        ori= np.asarray(net_doping,dtype=np.float64)
        hex  = np.atleast_1d(ori)
        n, p =self.equilibrium_densities(hex)
        psi  =  np.where(
            hex >=  0.0,
            np.log(n) +  _joyce_dixon_correction(self._u(n, self.Nc)),
            - np.log(p) - _joyce_dixon_correction(self._u(p, self.Nv)),
        )
        return  np.asarray (psi.reshape(ori.shape  ) )
