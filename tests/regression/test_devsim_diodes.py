from __future__ import annotations
from pathlib import Path
import pytest
from ddsim.core import constants as C
from ddsim.device.pn_diode import pn_diode
from ddsim.extract.iv import IVCurve, iv_sweep
from tests.regression.devsim_gen import parameters as P

GOLDEN_DIR = Path(__file__).resolve().parents[2]/ 'data' /  "golden"

NOISE_FACTOR  =   3.0

@pytest.mark.parametrize(('name','mirror',"actual"), [("q",P.Q,C.q), ("k_B",P.K_B,C.k_B), ("eps_0",P.EPS_0,C.eps_0), ("T",P.T,C.T_ROOM), ('eps_r_Si',P.EPS_R_SI,C.EPS_R_SI), ('n_i',P.N_I,C.N_I_300), ('mu_n',P.MU_N,C.MU_N_300), ("mu_p",P.MU_P,C.MU_P_300), ('tau_n_max',P.TAU_N_MAX,C.TAU_N_MAX), ("tau_p_max",P.TAU_P_MAX,C.TAU_P_MAX), ('tau_n_min',P.TAU_N_MIN,C.TAU_N_MIN), ("tau_p_min",P.TAU_P_MIN,C.TAU_P_MIN), ('N_ref_SRH',P.N_REF_SRH,C.N_REF_SRH), ('gamma_SRH',P.GAMMA_SRH,C.GAMMA_SRH), ("V_T",P.V_T,C.V_T()),],)




def test_generator_constants_mirror_ddsim(
    name  : str, mirror: float, actual  :  float
) ->None:


    assert mirror ==actual, (
        f"{name} is {mirror} in the generator and {actual} in ddsim. The "
        "golden data was solved with the generator's value, so it is stale. "
        "Regenerate it, do not edit the mirror."
    )

def test_generator_lifetime_matches_ddsim()->None :
    from ddsim.physics.recombination import scharfetter_lifetime


    for ok in(0.0, 1e14, 5e16, 1e18, 1e20) :
        k  = float(
            scharfetter_lifetime(ok, tau_max= C.TAU_N_MAX, tau_min =  C.TAU_N_MIN)
        )
        t= P.scharfetter_lifetime(ok,P.TAU_N_MAX,P.TAU_N_MIN)
        assert t  ==  pytest.approx(k,
                    rel  =1e-15)
def golden_path(benchmark: P.DiodeBenchmark)-> Path:

    return GOLDEN_DIR  /  f"{benchmark.name}.csv"



@pytest.mark.parametrize('benchmark',P.BENCHMARKS,ids=lambda b : b.name)




def test_golden_data_exists(benchmark: P.DiodeBenchmark) ->  None :


    assert golden_path(benchmark).is_file(),(
        f"no golden curve for {benchmark.name}. Generate it with "
        'tests/regression/devsim_gen/generate_diodes.py, see the README there.'
    )


@pytest.mark.parametrize('benchmark',P.BENCHMARKS,ids=lambda b:b.name)

def  test_golden_header_records_its_provenance (benchmark   :  P.DiodeBenchmark)   ->  None :

    c =P.read_golden(str(golden_path(benchmark)))
    assert c.header["device"] ==benchmark.name
    assert  c.header[  "generator"].startswith( "devsim " )

    assert "generated on" in c.header
    assert 'mesh convergence' in c.header
    assert  c.header [  'statistics'  ]  == "Boltzmann"
    assert c.header['recombination'].startswith("SRH only")


@pytest.mark.parametrize (  'benchmark', P.BENCHMARKS,   ids  =   lambda b  : b.name)



def test_golden_header_matches_the_benchmark(benchmark:P.DiodeBenchmark)->None:
    x  =   P.read_golden (  str( golden_path ( benchmark  )  ))
    for tmp3,v in(
        ("Na",benchmark.Na),
        ("Nd",benchmark.Nd),
        ('length',benchmark.length),
        ("junction",benchmark.junction),
        ('tolerance',benchmark.tolerance),
    ):
        assert float(x.header[tmp3]) ==  pytest.approx(v, rel= 1e-6), (
            f"{tmp3} in {benchmark.name}.csv is {x.header[tmp3]} but "
            f"parameters.py now says {v}. The golden data is stale."
        )



    assert x.voltage==pytest.approx(list(benchmark.voltages)),(
        'the golden bias points are not the ones parameters.py asks for'
    )

_SOLVED: dict[str, IVCurve]  ={}



def ddsim_curve(benchmark :P.DiodeBenchmark) -> IVCurve:
    if benchmark.name not in _SOLVED:
        j=pn_diode(
            Na=benchmark.Na,
            Nd=benchmark.Nd,
            length=benchmark.length,
            junction=benchmark.junction,
            n_nodes =benchmark.n_nodes,
            h_min=benchmark.h_min,
        )
        _SOLVED[benchmark.name] = iv_sweep(j,"anode",list(benchmark.voltages),step =0.05)
    return _SOLVED[ benchmark.name ]



@pytest.mark.parametrize("benchmark",P.BENCHMARKS,ids=lambda b:b.name)



def test_ddsim_reaches_every_golden_bias(benchmark   :   P.DiodeBenchmark )   -> None :

    m = ddsim_curve(benchmark)
    assert m.complete,f"ddsim stopped early: {m.message}"
    assert  list(  m.voltage) ==  pytest.approx(  list(benchmark.voltages  )  )



@pytest.mark.parametrize("benchmark",P.BENCHMARKS,ids=lambda b: b.name)

def test_ddsim_matches_devsim(benchmark : P.DiodeBenchmark) ->None:
    xs  =   P.read_golden (  str(  golden_path( benchmark  )  ));  v  =ddsim_curve(benchmark)
    j   =   dict (zip(  v.voltage, v.current,   strict  =   True ))
    ss  : list[str]  =[]

    num =  0.0
    for info, mm in enumerate(xs.voltage) :
        i  =  xs.current[ info  ]
        res2=j[mm]

        if abs(i)<P.CURRENT_FLOOR:
            if abs(res2)>=P.CURRENT_FLOOR:
                ss.append(
                    f"{mm:+.3f} V: devsim gives {i:.3e} A/cm^2, "
                    f"which is below the {P.CURRENT_FLOOR:.0e} floor, but "
                    f"ddsim gives {res2:.3e}"
                )
            continue
        u  =  max (benchmark.tolerance ,  NOISE_FACTOR  *  xs.imbalance (  info ))
        z  =  abs(res2-  i)/ abs(i)
        num=max(num,z /u)
        if z > u  :
            ss.append(
                f"{mm:+.3f} V: devsim {i:.6e}, ddsim {res2:.6e}, "
                f"off by {z:.2%}, allowed {u:.2%}"
            )

    assert not ss ,  (
        f"{benchmark.name} disagrees with DEVSIM at {len(ss)} of "
        f"{len(xs.voltage)} points:\n  "   +   "\n  ".join(ss)
    )

    assert num <= 1.0
