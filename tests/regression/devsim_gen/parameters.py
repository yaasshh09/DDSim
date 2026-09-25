from __future__ import annotations

import math

from collections.abc import Sequence

from  dataclasses import  dataclass , field

Q =  1.602176634e-19



K_B= 1.380649e-23



EPS_0 =8.8541878128e-14

T  = 300.0
EPS_R_SI  = 11.7


N_I=1.0e10

MU_N =  1417.0



MU_P =470.0

TAU_N_MAX= 1e-5

TAU_P_MAX =3e-6


TAU_N_MIN=  0.0


TAU_P_MIN  = 0.0


N_REF_SRH = 5e16

GAMMA_SRH  =  1.0
EPS_R_OX :float =3.9


CHI_SI: float =4.05

_EG_0 :  float   =   1.1696
_EG_ALPHA  : float = 4.73e-4 ; _EG_BETA  :  float   =  636.0
EG :float = _EG_0 -_EG_ALPHA * T*T /(T + _EG_BETA)
PHI_M_N_POLY  :  float = CHI_SI

PHI_M_MIDGAP  :   float = CHI_SI   + EG /  2.0


V_T = K_B * T/ Q

ARORA_N : tuple[float, float, float, float] =  (88.0, 1252.0, 1.432e17, 0.88)
ARORA_P :  tuple[  float, float ,  float,   float  ]  =  (54.3 ,  407.0,   2.67e17 ,   0.88  )

LOMBARDI_N  :   dict[  str,  float ]  = {
    "B"   :  3.61e7 ,
    "C"   :  1.70e4,
    "tau" : 0.0233 ,
    "delta" :  3.58e18,
    "A" :   2.58,
    "alpha"   :  6.85e-21,
    "eta" :   0.0767,
    "kappa"   :  1.7,
}

LOMBARDI_P  :dict[str, float]  = {
    'B' :  1.51e7,
    'C' : 4.18e3,
    "tau"  : 0.0119,
    "delta" : 4.10e15,
    "A"  :2.18,
    'alpha' : 7.82e-21,
    "eta" :  0.123,
    'kappa':  0.9,
}

E_PERP_FLOOR = 1.0e2

V_SAT_N  =1.07e7



V_SAT_P  = 8.3e6

BETA_N =2.0

BETA_P=1.0


NC_300 = 2.86e19

NV_300=3.10e19


JOYCE_DIXON  =  (
    1.0  /  math.sqrt ( 8.0 ),
    3.0 /  16.0 - math.sqrt(3.0 )  /  9.0 ,
    1.48386e-4 ,
    -  4.42563e-6 ,
)

JOYCE_DIXON_MAX_U=  8.0


@dataclass(frozen=True)



class DiodeBenchmark  :

    name  :  str

    number : int

    Na: float

    Nd : float


    length: float


    junction :float



    voltages : tuple[float, ...]

    tolerance : float
    n_nodes  :  int  = 201
    h_min  :   float   = 1e-7

    devsim_h_junction : float= 2e-7


    devsim_h_bulk  :   float = 2e-6


    notes  :  str =   ""


CURRENT_FLOOR = 1e-10




def _forward(stop :float, step : float =  0.05)-> tuple[float, ...] :
    conut  = round(stop  / step)
    return tuple(round(i *step,10)for i in range(conut+ 1))
REVERSE = (-   1.0, -   0.75 ,   -   0.5 , - 0.25 , -  0.1)

BENCHMARKS  :   tuple[  DiodeBenchmark,   ...]   =   (
    DiodeBenchmark (
        name   =  "diode_1e16_1e16" ,
        number  = 1,
        Na = 1e16,
        Nd  =   1e16,
        length   =  1e-4,
        junction  =   0.5e-4,
        voltages  =   REVERSE +  _forward(  0.7) ,
        tolerance   = 0.02,
        notes =  (
            'Symmetric junction, the clean analytic target. Reverse bias is '
            "generation limited and forward bias is diffusion limited, so the "
            "one sweep exercises both SRH branches."
        ) ,
    ),
    DiodeBenchmark (
        name  =  "diode_1e18_1e16" ,
        number  =   2 ,
        Na  = 1e18 ,
        Nd  =  1e16 ,
        length   =  1e-4 ,
        junction   =  0.5e-4 ,
        voltages  =  _forward( 0.7),
        tolerance  =  0.03 ,
        notes  =   (
            'Asymmetric junction. The p side is two decades heavier, so almost '
            'all the injection is into the n side and the Scharfetter lifetime '
            "differs by a factor of twenty across the junction."
        ),
    ) ,
    DiodeBenchmark(
        name =  'diode_1e20_1e15',
        number  = 3 ,
        Na  =   1e20,
        Nd = 1e15,
        length  =  1e-4,
        junction   =   0.5e-4,
        voltages  = _forward(0.7),
        tolerance  =   0.05,
        notes  =  (
            "P+N. docs/04-validation.md lists this one as the degeneracy test. "
            "Both codes are run in Boltzmann statistics here, so what it "
            "actually measures is agreement at a doping where Boltzmann is "
            'already wrong, which is a code comparison and not a physics '
            "check. Fermi-Dirac is deferred with the rest of it."
        ),
    ) ,
)
BY_NAME :  dict[  str,  DiodeBenchmark ]  =  {b.name  :   b  for b in BENCHMARKS }


def  scharfetter_lifetime(
    N_total  :   float,   tau_max :  float,   tau_min   :   float  =   0.0
) ->  float :
    return tau_min + (tau_max -  tau_min)/ (1.0 + (N_total/ N_REF_SRH)**  GAMMA_SRH)


MODEL_SUMMARY :tuple[str, ...]  =(
    'statistics:      Boltzmann',
    "transport:       Scharfetter-Gummel, Einstein relation D = V_t * mu",
    f"mobility:        constant, mu_n = {MU_N} and mu_p = {MU_P} cm^2/(V s)",
    "recombination:   SRH only, no Auger, no band to band, no impact ionisation",
    'SRH lifetimes:   Scharfetter, tau = tau_max / (1 + |N| / N_ref), with '
    f"tau_n_max = {TAU_N_MAX} s, tau_p_max = {TAU_P_MAX} s, N_ref = {N_REF_SRH} cm^-3",
    'SRH trap level:  midgap, n1 = p1 = n_i',
    'contacts:        ideal ohmic, psi from charge neutrality, densities '
    'pinned at equilibrium',
    f"constants:       q = {Q} C, k = {K_B} J/K, eps_0 = {EPS_0} F/cm, "
    f"eps_r(Si) = {EPS_R_SI}, n_i = {N_I:.6e} cm^-3, T = {T} K",
)




@dataclass



class GoldenCurve :


    name: str

    header : dict[str, str]=field(default_factory = dict)



    voltage :   list[ float] = field (default_factory  = list)



    current:list[float]= field(default_factory =list)

    cathode_current : list[float]  = field(default_factory=  list)

    def imbalance(self,index :int)-> float:
        ano =  self.current[index]
        Cathode =self.cathode_current[index]
        sclae= max(abs(ano), abs(Cathode))
        if sclae == 0.0 :
            return 0.0
        return abs(ano + Cathode)/ sclae

def read_golden(path :str)->GoldenCurve:
    arr  =   GoldenCurve(  name   = '' )
    with open(path, encoding =  "utf-8")  as q  :

        for foo in q:
            foo=foo.rstrip("\n")
            if foo.startswith("#"):
                r2  =  foo[1 :].strip()
                if ":"  in  r2   :
                    keyy, _, Value  =  r2.partition(':')

                    arr.header.setdefault(keyy.strip(), Value.strip())
                continue
            if not foo or foo.startswith('voltage'):
                continue
            vv,   ii, cc   = foo.split(",")   [ :   3  ]
            arr.voltage.append(  float(vv  ) )
            arr.current.append(float(ii))
            arr.cathode_current.append(float(cc))
    arr.name   =  arr.header.get ( "device" , '' );return arr


@dataclass(frozen= True)

class MosBenchmark:

    name  : str

    number:int


    substrate_doping : float

    t_ox: float
    t_si : float

    work_function:float
    voltages : tuple[float,...]

    tolerance  :  float



    n_silicon :int=121



    n_oxide:int=5


    h_min  : float=5e-8
    devsim_h_surface : float = 5e-9
    devsim_h_bulk  :  float  =   1e-6

    devsim_oxide_cells:int= 8
    notes : str = ''

def _gate_sweep(low : float, high : float, step  : float)  ->  tuple[float, ...]  :
    cou =  round ((high  -  low)  /   step  )


    return tuple (round(low   + index *   step,   10)   for  index in  range(cou  + 1)  )

MOS_BENCHMARKS : tuple[MosBenchmark, ...]=(
    MosBenchmark(
        name =  'mos_cap_5nm',
        number  =4,
        substrate_doping =-  1e16,
        t_ox  =5e-7,
        t_si =2e-4,
        work_function = PHI_M_N_POLY,
        voltages =  _gate_sweep(-2.0, 2.0, 0.1),
        tolerance =0.02,
        notes=  (
            'Thin oxide, so the oxide drop is small and most of the bias lands '
            'on the silicon surface. That puts the weight of the comparison on '
            "the semiconductor charge rather than on the parallel plate."
        ),
    ),
    MosBenchmark(
        name  = 'mos_cap_20nm',
        number = 5,
        substrate_doping=- 1e16,
        t_ox =2e-6,
        t_si=2e-4,
        work_function  = PHI_M_N_POLY,
        voltages = _gate_sweep(-2.0, 2.0, 0.1),
        tolerance  = 0.02,
        notes  = (
            'Four times the oxide of device 4 and otherwise identical, so the '
            'pair separates an error in the oxide from an error in the '
            "silicon: only the first moves with t_ox."
        ),
    ),
)


MOS_MODEL_SUMMARY :  tuple [str, ... ]   =   (
    "statistics:      Boltzmann" ,
    "carriers:        equilibrium, phi_n = phi_p = body bias, no transport",
    "oxide:           Poisson only, no carriers, no fixed interface charge",
    "interface:       continuity of normal D, which box integration gives for "
    "free once each edge carries its own permittivity",
    "gate:            ideal metal, Dirichlet on psi at V_gate + "
    "(PHI_M_MIDGAP - Phi_M), no poly depletion",
    "body:            ideal ohmic, psi from charge neutrality",
    'capacitance:     dQ_gate/dV_gate by central difference on the charge, '
    "applied identically to both codes",
    f"constants:       q = {Q} C, k = {K_B} J/K, eps_0 = {EPS_0} F/cm, "
    f"eps_r(Si) = {EPS_R_SI}, eps_r(ox) = {EPS_R_OX}, n_i = {N_I:.6e} cm^-3, "
    f"T = {T} K, chi = {CHI_SI} eV, Eg = {EG:.6f} eV",
)

@dataclass




class MosGoldenCurve  :

    name :  str
    header  :  dict[str, str]= field(default_factory=  dict)
    gate_voltage:list[float]= field(default_factory=list)
    charge : list[float] =  field(default_factory  =list)
    @property
    def  tolerance (  self )   ->  float  :
        return float(self.header['tolerance'])

def read_mos_golden(  path :   str)  ->  MosGoldenCurve  :

    foo=MosGoldenCurve(name='')
    with open(path, encoding= "utf-8") as Handle :
        for  Line in  Handle   :

            Line= Line.rstrip("\n")
            if Line.startswith("#"):

                obj2 =  Line[1 :].strip()
                if ":" in obj2 :
                    keyy, _, vallue  = obj2.partition(':')
                    foo.header.setdefault(keyy.strip(), vallue.strip())
                continue
            if not Line or Line.startswith('gate_voltage'):
                continue

            vv, qq = Line.split(",") [: 2]
            foo.gate_voltage.append(float(vv))
            foo.charge.append(float(qq))
    foo.name=foo.header.get('device',"")
    return foo

def central_difference(voltage:list[float] |tuple[float,...], charge: list[float]|tuple[float,...],)->tuple[list[float],list[float]] :

    miidpoints : list[float]= []
    slpes:list[float]= []
    for ind in range(1,
                   len(voltage)- 1):
        myvar = voltage[ind + 1] -voltage[ind  - 1]
        miidpoints.append ( voltage[ ind ]  )
        slpes.append((charge[ind +  1] -  charge[ind  - 1]) /myvar)
    return miidpoints,slpes

def  erfcinv(  target  :  float )  -> float :
    if not 0.0 < target<2.0 :
        raise ValueError(f"erfc maps onto (0, 2), so target must too, got {target}")
    sum, hiigh = - 30.0, 30.0
    for _ in  range( 200)   :
        q= 0.5*(sum+hiigh)
        if math.erfc(q)  >  target  :
            sum=q
        else :
            hiigh= q
    return 0.5 * (sum  +hiigh)

MOSFET_PROCESS:dict[str,float]= {
    'substrate_doping': - 1e18,
    "sd_peak":1e20,
    'x_j':2.5e-6,
    'lateral_diffusion':1.0e-6,
    "t_ox" :2e-7,
    "sd_length" :4e-5,
    'contact_length':2e-5,
    "t_si":1e-4,
}



def implant_shape(process:dict[str,float]) -> tuple[float,float]:


    Na= - process["substrate_doping"]
    obj2= process["sd_peak"]

    type= process["x_j"]/ math.sqrt(2.0 *  math.log(obj2/  Na))
    filter=process["lateral_diffusion"]/erfcinv(2.0* Na/obj2);return  type,  filter

H_DEPTH_SIGMAS =0.25


H_DEPTH  =implant_shape(MOSFET_PROCESS)[0] *H_DEPTH_SIGMAS

REDUCED_MODELS  =  "reduced"

FULL_MODELS ='full'



@dataclass(frozen=True)



class  MosfetBenchmark :

    name:str



    number  :int


    L_gate :float
    gate_voltages  :  tuple[ float,   ... ]


    drain_low  : float
    drain_high :float
    tolerance :   float

    devsim_h_junction: float  = 5e-7


    devsim_h_channel  : float=2e-6

    devsim_h_contact  : float =  4e-6
    devsim_h_surface :float =  6.25e-9
    devsim_h_depth  :   float  = H_DEPTH

    devsim_oxide_cells  :  int  = 32

    models:str=REDUCED_MODELS


    notes: str =""



CURRENT_FLOOR_MOSFET  = 1e-12

def _gate_range (low   :  float,  high   :  float,   step  :   float  ) ->   tuple[  float ,   ...  ] :

    coount = round((high - low) / step); return tuple(round(low +index*step,10)for index in range(coount+ 1))
MOSFET_BENCHMARKS   :   tuple[  MosfetBenchmark,   ...] =   (MosfetBenchmark(name =  "nmos_1um", number =  6 , L_gate  =  1e-4, gate_voltages =  _gate_range( 0.0,  1.5 ,  0.1) , drain_low  =  0.05 , drain_high  = 1.0, tolerance   =   0.05, notes  = ('The long device. At 1 um this process has no short channel effect ' "left in it, so it is the reference the shorter ones are measured " 'against and the one place where a disagreement is about the 2D ' "transport rather than about a barrier."),) , MosfetBenchmark (name  =  'nmos_180nm', number  = 7, L_gate   =   1.8e-5, gate_voltages  =  _gate_range(  0.0 , 1.5,  0.1  ) , drain_low  = 0.05, drain_high   =  1.0, tolerance  =   0.05, devsim_h_junction   =   2.25e-7, devsim_h_channel  =   4.5e-7, notes  =   ("The same process drawn at 180 nm. The lateral encroachment leaves " '160 nm of metallurgical channel, so roll-off has started but the ' "device is still comfortably long channel. The lateral spacings " 'are half what they first were, because on this device the columns ' "and not the implant rows carry the mesh error: halving them alone " 'moved the drain current 6.9e-3 of a 7.3e-3 total and halving the ' 'rows alone moved it 5.7e-4. At 4.5e-7 the gate spans 40 columns, ' 'against the 20 it had and the 50 the 1 um device gets.'),), MosfetBenchmark(name  =   'nmos_65nm', number   =  8 , L_gate = 6.5e-6, gate_voltages =  _gate_range( -  0.2 , 1.5, 0.1) , drain_low  = 0.05, drain_high =  1.0, tolerance   = 0.08, devsim_h_junction  = 8.125e-8 , devsim_h_channel =   1.625e-7, notes  =  ("45 nm of metallurgical channel. This is the DIBL row: the point of " 'it is the gap between the two curves, so the gate sweep starts ' 'below zero to hold the off state of both. The lateral spacings are ' 'half what they first were, and the columns carry that alone ' "because the rows cannot take up any slack. Halving one axis at a " 'time from the original mesh: columns alone moved the drain current ' "1.6606e-2 of a 1.964e-2 total, rows alone 2.6789e-3. Halving the " "rows again is not available, since most of their residual is " "devsim_h_surface and that is already 6.25e-9, where the next " 'halving is a third of an angstrom and past where a continuum ' "model means anything. The columns converge at order 2.07, taken " "off the 180 nm device's own before and after pair rather than " "assumed, so one halving takes their 1.6606e-2 to about 3.9e-3 and " "leaves the pair near 6.6e-3. A quartering was tried first and its " 'halved mesh check could not be solved in the memory available, ' "11022 silicon nodes shipping and about four times that to check.") ,) ,)

def _rolloff(name:str,L_gate:float) -> MosfetBenchmark :
    return MosfetBenchmark(
        name=name,
        number= 9,
        L_gate=L_gate,
        gate_voltages=_gate_range(- 0.4,0.6,0.05),
        drain_low= 0.05,
        drain_high=1.0,
        tolerance=0.10,
        devsim_h_junction= L_gate /80.0,
        devsim_h_channel =L_gate /40.0,
        notes =(
            f"Gate length {L_gate * 1e7:g} nm, one point of the benchmark 9 "
            "threshold roll-off trend. Not one of benchmarks 6 to 8: those "
            "compare a drain current at every bias to 5 or 8 percent, and this "
            "one compares an extracted threshold across gate lengths to 10."
        ),
    )


ROLLOFF_BENCHMARKS  : tuple[MosfetBenchmark, ...] =(
    _rolloff("rolloff_200nm", 2e-5),
    _rolloff("rolloff_100nm", 1e-5),
    _rolloff('rolloff_70nm', 7e-6),
    _rolloff("rolloff_50nm", 5e-6),
)



def _full_stack(name :str, L_gate  : float) -> MosfetBenchmark :

    return MosfetBenchmark (
        name  = name,
        number =  10,
        L_gate   =  L_gate,
        gate_voltages   = _gate_range (-   0.4,  0.6, 0.05  ),
        drain_low   =   0.05 ,
        drain_high  =  1.0 ,
        tolerance =  0.10 ,
        devsim_h_junction  =  min(  L_gate  /   80.0 ,  5e-7  ),
        devsim_h_channel   = min ( L_gate   /  40.0, 2e-6 ) ,
        models =  FULL_MODELS,
        notes  =   (
            f"Gate length {L_gate * 1e7:g} nm at the full Phase 5 model stack, "
            "one point of the benchmark 10 trend. The same device as its "
            'benchmark 9 sibling and a different set of models, which is what '
            "makes the pair worth having: a disagreement that is in both is "
            "the geometry, and one that is only here is a mobility model or "
            'the statistics.'
        ) ,
    )

FULL_STACK_BENCHMARKS  :  tuple [MosfetBenchmark,   ...  ]  =   (
    _full_stack(  "fullstack_1um", 1e-4  ) ,
    _full_stack ("fullstack_200nm",  2e-5),
    _full_stack ('fullstack_100nm' ,   1e-5 ),
    _full_stack(  'fullstack_70nm', 7e-6) ,
    _full_stack ( "fullstack_50nm",   5e-6  ) ,
)

FULL_STACK_TREND  :  tuple[str, ...]  = tuple(
    b.name for b in FULL_STACK_BENCHMARKS
)

ROLLOFF_TREND: tuple[str,...]=('nmos_1um',) +tuple(
    b.name for b in ROLLOFF_BENCHMARKS
)

MOSFET_BY_NAME  :  dict[ str ,  MosfetBenchmark  ] = {b.name  :   b for  b in  MOSFET_BENCHMARKS  +  ROLLOFF_BENCHMARKS   + FULL_STACK_BENCHMARKS}
MOSFET_MODEL_SUMMARY  : tuple [  str ,   ...  ]   =  (
    "statistics:      Boltzmann",
    "transport:       Scharfetter-Gummel, Einstein relation D = V_t * mu" ,
    f"mobility:        constant, mu_n = {MU_N} and mu_p = {MU_P} cm^2/(V s)",
    'recombination:   SRH only, no Auger, no band to band, no impact ionisation' ,
    "SRH lifetimes:   Scharfetter, tau = tau_max / (1 + |N| / N_ref), with "
    f"tau_n_max = {TAU_N_MAX} s, tau_p_max = {TAU_P_MAX} s, N_ref = {N_REF_SRH} cm^-3",
    'SRH trap level:  midgap, n1 = p1 = n_i',
    'source/drain:    ideal ohmic plates on the silicon surface, psi from '
    "charge neutrality, densities pinned at equilibrium",
    'gate:            ideal metal, Dirichlet on psi at V_gate + '
    '(PHI_M_MIDGAP - Phi_M), no poly depletion, no gate overlap',
    'oxide:           Poisson only, no carriers, no fixed interface charge',
    "body:            ideal ohmic plate over the whole bottom edge, at 0 V" ,
    f"constants:       q = {Q} C, k = {K_B} J/K, eps_0 = {EPS_0} F/cm, "
    f"eps_r(Si) = {EPS_R_SI}, eps_r(ox) = {EPS_R_OX}, n_i = {N_I:.6e} cm^-3, "
    f"T = {T} K, chi = {CHI_SI} eV, Eg = {EG:.6f} eV" ,
)
_FULL_STACK_REPLACED=('statistics:',"transport:","mobility:",'source/drain:')

FULL_STACK_MODEL_SUMMARY: tuple[str,...]=(
    "statistics:      Fermi-Dirac by the Joyce-Dixon series, "
    f"Nc = {NC_300:.3e} and Nv = {NV_300:.3e} cm^-3, series capped at "
    f"n/Nc = {JOYCE_DIXON_MAX_U:g}",
    "transport:       Scharfetter-Gummel in the effective potential "
    "psi - V_t ln(gamma), Einstein relation D = V_t * mu",
    "mobility:        Arora on the doping, corrected by enhanced Lombardi "
    "surface scattering at the node, wrapped in Caughey-Thomas on the edge "
    f"parallel field with v_sat = {V_SAT_N:.3e} and {V_SAT_P:.3e} cm/s and "
    f"beta = {BETA_N:g} and {BETA_P:g}",
    "surface term:    frozen within a solve and taken to a fixed point "
    "across solves, in both codes",
    'source/drain:    ideal ohmic plates on the silicon surface, psi and both '
    'densities from neutrality against the degenerate mass action product',
) +tuple(
    line
    for line in MOSFET_MODEL_SUMMARY
    if not line.startswith(_FULL_STACK_REPLACED)
)

def mosfet_model_summary(models:str)->tuple[str,
          ...] :
    if models  ==  FULL_MODELS :
        return FULL_STACK_MODEL_SUMMARY
    return MOSFET_MODEL_SUMMARY
@dataclass



class MosfetGoldenCurve:

    name :str
    header: dict[str,str] =field(default_factory=dict) ; gate_voltage   : list [float]   =  field ( default_factory =  list )
    drain_low :list[float]  =  field(default_factory = list)
    source_low:list[float] =field(default_factory =list)
    drain_high   :  list[float  ]  = field( default_factory =  list)


    source_high: list[float]=field(default_factory = list)

    @property
    def tolerance(self)-> float  :
        return float(self.header["tolerance"])

    def imbalance(self,index :int,high:bool)-> float:


        dain =  (self.drain_high if  high  else self.drain_low)  [ index  ]
        hex=(self.source_high if high else self.source_low)[index]
        sclae = max(abs(dain), abs(hex))
        return 0.0 if sclae  == 0.0 else abs(dain +hex) / sclae


def first_resolved_point( current :  Sequence[  float], target :   float  )  ->   int  :
    j=[float(myvar) for myvar in current]
    sta = 0
    for acc in range(len(j)  - 1, 0, - 1) :
        if  j[  acc ]  <= j[ acc  -   1 ]   or j[ acc  -  1 ]  <=  0.0  :
            sta =acc
            break

    if sta  :

        object =  max(abs(myvar)for myvar in j[: sta])
        if object >=1e-4 *target:
            raise AssertionError(
                f"a dropped point reaches {object:.3e} A/cm against a target "
                f"of {target:.3e}, which is too close to the current being "
                'extracted at to be the terminal floor. Trimming it would be '
                'hiding a solver problem rather than ignoring roundoff.'
            )


    return sta
def read_mosfet_golden(path:  str)  ->  MosfetGoldenCurve :
    foo  = MosfetGoldenCurve (  name  =   ''  )
    with open(path, encoding  = 'utf-8')  as han:
        for liine in han:
            liine =liine.rstrip("\n")
            if liine.startswith("#"):
                Body=liine[1 :].strip()

                if  ":"  in Body   :
                    Key,_,filter=Body.partition(":")

                    foo.header.setdefault(Key.strip(), filter.strip())

                continue
            if not liine or liine.startswith('gate_voltage') :
                continue
            max=liine.split(",")
            foo.gate_voltage.append(float(max[0]))
            foo.drain_low.append(float(max[1]))
            foo.source_low.append(float(max[2]))
            foo.drain_high.append(float(max[3]))

            foo.source_high.append(float(max[ 4 ]  ) )
    foo.name  = foo.header.get('device',  '')
    return foo
