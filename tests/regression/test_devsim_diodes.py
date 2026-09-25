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


    for dop in(0.0, 1e14, 5e16, 1e18, 1e20) :
        Expected  = float(
            scharfetter_lifetime(dop, tau_max= C.TAU_N_MAX, tau_min =  C.TAU_N_MIN)
        )
        type= P.scharfetter_lifetime(dop,P.TAU_N_MAX,P.TAU_N_MIN)
        assert type  ==  pytest.approx(Expected,
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

    Curve =P.read_golden(str(golden_path(benchmark)))
    assert Curve.header["device"] ==benchmark.name
    assert  Curve.header[  "generator"].startswith( "devsim " )

    assert "generated on" in Curve.header
    assert 'mesh convergence' in Curve.header
    assert  Curve.header [  'statistics'  ]  == "Boltzmann"
    assert Curve.header['recombination'].startswith("SRH only")


@pytest.mark.parametrize (  'benchmark', P.BENCHMARKS,   ids  =   lambda b  : b.name)



def test_golden_header_matches_the_benchmark(benchmark:P.DiodeBenchmark)->None:
    res  =   P.read_golden (  str( golden_path ( benchmark  )  ))
    for Key,Expected in(
        ("Na",benchmark.Na),
        ("Nd",benchmark.Nd),
        ('length',benchmark.length),
        ("junction",benchmark.junction),
        ('tolerance',benchmark.tolerance),
    ):
        assert float(res.header[Key]) ==  pytest.approx(Expected, rel= 1e-6), (
            f"{Key} in {benchmark.name}.csv is {res.header[Key]} but "
            f"parameters.py now says {Expected}. The golden data is stale."
        )



    assert res.voltage==pytest.approx(list(benchmark.voltages)),(
        'the golden bias points are not the ones parameters.py asks for'
    )

_SOLVED: dict[str, IVCurve]  ={}



def ddsim_curve(benchmark :P.DiodeBenchmark) -> IVCurve:
    if benchmark.name not in _SOLVED:
        lst=pn_diode(
            Na=benchmark.Na,
            Nd=benchmark.Nd,
            length=benchmark.length,
            junction=benchmark.junction,
            n_nodes =benchmark.n_nodes,
            h_min=benchmark.h_min,
        )
        _SOLVED[benchmark.name] = iv_sweep(lst,"anode",list(benchmark.voltages),step =0.05)
    return _SOLVED[ benchmark.name ]



@pytest.mark.parametrize("benchmark",P.BENCHMARKS,ids=lambda b:b.name)



def test_ddsim_reaches_every_golden_bias(benchmark   :   P.DiodeBenchmark )   -> None :

    round = ddsim_curve(benchmark)
    assert round.complete,f"ddsim stopped early: {round.message}"
    assert  list(  round.voltage) ==  pytest.approx(  list(benchmark.voltages  )  )



@pytest.mark.parametrize("benchmark",P.BENCHMARKS,ids=lambda b: b.name)

def test_ddsim_matches_devsim(benchmark : P.DiodeBenchmark) ->None:
    gol  =   P.read_golden (  str(  golden_path( benchmark  )  ));  cur  =ddsim_curve(benchmark)
    meaured   =   dict (zip(  cur.voltage, cur.current,   strict  =   True ))
    faliures  : list[str]  =[]

    yy =  0.0
    for ind, Voltage in enumerate(gol.voltage) :
        k2  =  gol.current[ ind  ]
        Actual=meaured[Voltage]

        if abs(k2)<P.CURRENT_FLOOR:
            if abs(Actual)>=P.CURRENT_FLOOR:
                faliures.append(
                    f"{Voltage:+.3f} V: devsim gives {k2:.3e} A/cm^2, "
                    f"which is below the {P.CURRENT_FLOOR:.0e} floor, but "
                    f"ddsim gives {Actual:.3e}"
                )
            continue
        alowed  =  max (benchmark.tolerance ,  NOISE_FACTOR  *  gol.imbalance (  ind ))
        relaive  =  abs(Actual-  k2)/ abs(k2)
        yy=max(yy,relaive /alowed)
        if relaive > alowed  :
            faliures.append(
                f"{Voltage:+.3f} V: devsim {k2:.6e}, ddsim {Actual:.6e}, "
                f"off by {relaive:.2%}, allowed {alowed:.2%}"
            )

    assert not faliures ,  (
        f"{benchmark.name} disagrees with DEVSIM at {len(faliures)} of "
        f"{len(gol.voltage)} points:\n  "   +   "\n  ".join(faliures)
    )

    assert yy <= 1.0
