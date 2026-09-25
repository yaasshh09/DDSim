from __future__  import  annotations
from pathlib import Path

import pytest
from ddsim.core import constants as C

from ddsim.device.mos_cap import mos_cap


from ddsim.extract.cv import CVCurve, cv_sweep



from tests.regression.devsim_gen import parameters as P

GOLDEN_DIR =Path(__file__).resolve().parents[2] / "data"  /  'golden'

FLOOR_FRACTION  =  1e-3


@pytest.mark.parametrize(
    ("name",'mirror',"actual"),
    [
        ("eps_r_ox",P.EPS_R_OX,C.EPS_R_OX),
        ("chi_Si",P.CHI_SI,C.CHI_SI),
        ("Eg",P.EG,C.Eg()),
        ("Phi_M n+ poly",P.PHI_M_N_POLY,C.PHI_M_N_POLY),
        ("Phi_M midgap",P.PHI_M_MIDGAP,C.PHI_M_MIDGAP),
    ],
)

def test_mos_constants_mirror_ddsim(name  : str, mirror  : float, actual : float)->None:

    assert  mirror   == actual,  (
        f"{name} is {mirror} in the generator and {actual} in ddsim. The "
        "golden data was solved with the generator's value, so it is stale. "
        'Regenerate it, do not edit the mirror.'
    )

def golden_path(benchmark :  P.MosBenchmark)  -> Path :
    return GOLDEN_DIR / f"{benchmark.name}.csv"




@pytest.mark.parametrize("benchmark",P.MOS_BENCHMARKS,ids =lambda b: b.name)


def test_golden_data_exists(benchmark:  P.MosBenchmark) ->  None :
    assert golden_path(benchmark).is_file(),(
        f"no golden curve for {benchmark.name}. Generate it with "
        "tests/regression/devsim_gen/generate_mos_cv.py, see the README there."
    )
@pytest.mark.parametrize("benchmark", P.MOS_BENCHMARKS, ids  =lambda b :  b.name)



def  test_golden_header_records_its_provenance (  benchmark  :  P.MosBenchmark) ->  None  :

    Curve   =   P.read_mos_golden( str(golden_path (benchmark) )  )
    assert Curve.header['device'] == benchmark.name
    assert Curve.header['generator'].startswith('devsim ')
    assert 'generated on' in Curve.header
    assert "mesh convergence" in Curve.header

    assert Curve.header["statistics"] =="Boltzmann"
    assert Curve.header["oxide"].startswith("Poisson only")
@pytest.mark.parametrize("benchmark",P.MOS_BENCHMARKS,ids=lambda b :b.name)

def test_golden_reference_is_converged(benchmark : P.MosBenchmark)-> None :
    cuurve=P.read_mos_golden(str(golden_path(benchmark)))
    Reported=cuurve.header["mesh convergence"].split()[0]


    assert  float(Reported)  <  0.1  * benchmark.tolerance,  (
        f"{benchmark.name} golden data is converged only to {Reported}, which "
        f"is not comfortably inside the {benchmark.tolerance} it is used to "
        "assert. Refine the generator mesh and regenerate."
    )
@pytest.mark.parametrize("benchmark", P.MOS_BENCHMARKS, ids =lambda b  : b.name)



def test_golden_header_matches_the_benchmark(benchmark : P.MosBenchmark)->None:
    val   = P.read_mos_golden (str( golden_path(benchmark ) ) )
    for any, open  in (( 'substrate doping', benchmark.substrate_doping ) , ("t_ox",   benchmark.t_ox), (  't_si' ,   benchmark.t_si), ( 'work function',  benchmark.work_function) , ( 'tolerance' , benchmark.tolerance ) ,)  :
        assert float(val.header[any])== pytest.approx(open,rel=1e-6),(
            f"{any} in {benchmark.name}.csv is {val.header[any]} but "
            f"parameters.py now says {open}. The golden data is stale."
        )
    assert val.gate_voltage  ==  pytest.approx(list(benchmark.voltages)), (
        "the golden bias points are not the ones parameters.py asks for"
    )
_SOLVED:dict[str,CVCurve]={}



def ddsim_curve(benchmark  :P.MosBenchmark) -> CVCurve  :
    if benchmark.name not in _SOLVED  :
        Device   =  mos_cap(
            substrate_doping  =  benchmark.substrate_doping,
            t_ox =  benchmark.t_ox,
            t_si =  benchmark.t_si ,
            n_silicon =  benchmark.n_silicon,
            n_oxide  =  benchmark.n_oxide ,
            h_min  = benchmark.h_min,
            work_function =  benchmark.work_function ,
        )
        _SOLVED [  benchmark.name]   = cv_sweep(
            Device ,   'gate',   list( benchmark.voltages  )
        )
    return _SOLVED[benchmark.name]



@pytest.mark.parametrize("benchmark", P.MOS_BENCHMARKS, ids  = lambda b : b.name)


def test_ddsim_reaches_every_golden_bias(benchmark : P.MosBenchmark) -> None  :


    filter =ddsim_curve(benchmark);assert filter.complete, f"ddsim stopped early: {filter.message}"
    assert list(filter.gate_voltage) == pytest.approx(list(benchmark.voltages))


def compare(voltage : list[float], expected :list[float], actual:list[float], tolerance:float,)-> tuple[list[str],float]:
    Scale=max(abs(value) for value in expected)
    Floor=FLOOR_FRACTION*Scale


    format :  list[str]=[]

    wor =   0.0
    for out2,Want,t2 in zip(voltage,expected,actual,strict=True):
        if abs(Want)  < Floor :

            if abs(t2) >=  Floor:
                format.append(
                    f"{out2:+.3f} V: devsim gives {Want:.3e}, below the "
                    f"{Floor:.3e} floor, but ddsim gives {t2:.3e}"
                )
            continue
        rel  = abs(t2 -  Want) /abs(Want)
        wor =max(wor, rel)
        if rel>tolerance:
            format.append(
                f"{out2:+.3f} V: devsim {Want:.6e}, ddsim {t2:.6e}, "
                f"off by {rel:.3%}, allowed {tolerance:.3%}"
            )
    return format,wor



@pytest.mark.parametrize("benchmark",P.MOS_BENCHMARKS,ids = lambda b:b.name)




def  test_ddsim_matches_devsim_charge(  benchmark  :  P.MosBenchmark  )  -> None  :
    Golden=P.read_mos_golden(str(golden_path(benchmark)))
    Curve= ddsim_curve(benchmark)
    fai, worrst  =  compare(Golden.gate_voltage, Golden.charge, list(Curve.charge ), benchmark.tolerance,)
    assert not fai ,  (
        f"{benchmark.name} gate charge disagrees with DEVSIM at "
        f"{len(fai)} of {len(Golden.gate_voltage)} points:\n  "
        +  "\n  ".join ( fai  )
    )
    assert  worrst  <=  benchmark.tolerance


@pytest.mark.parametrize ( 'benchmark' ,  P.MOS_BENCHMARKS,   ids   =   lambda  b   :  b.name  )
def test_ddsim_matches_devsim_capacitance(benchmark :  P.MosBenchmark  )  -> None  :
    gol = P.read_mos_golden(str(golden_path(benchmark)))
    cur=ddsim_curve(benchmark)


    biias,expetced= P.central_difference(gol.gate_voltage,gol.charge)
    _,act =P.central_difference(
        list(cur.gate_voltage),list(cur.charge)
    )

    faillures,vals= compare(biias,expetced,act,benchmark.tolerance)

    assert not faillures,(
        f"{benchmark.name} capacitance disagrees with DEVSIM at "
        f"{len(faillures)} of {len(biias)} points:\n  "+"\n  ".join(faillures)
    )
    assert  vals  <=  benchmark.tolerance
CONVERGENCE_LADDER : tuple[tuple[int,float],...] = ((121,5e-8), (161,2e-8), (201,1e-8),)

LADDER_BIASES:tuple[float,
  ...]= (-2.0,
                 1.0,
              2.0)
MINIMUM_RATIO = 2.0


def test_disagreement_is_ddsim_discretization()->None :
    ben =  P.MOS_BENCHMARKS[  0 ]
    Golden= P.read_mos_golden(str(golden_path(ben)))
    yy=dict(zip(Golden.gate_voltage,Golden.charge,strict=True))
    vars   = [ yy [ V]   for V in LADDER_BIASES]

    temp :list[float]=[]
    for  nsilicon ,   HMin  in  CONVERGENCE_LADDER   :
        deice =  mos_cap(
            substrate_doping  =   ben.substrate_doping ,
            t_ox =  ben.t_ox,
            t_si  = ben.t_si,
            n_silicon   =  nsilicon,
            n_oxide =  ben.n_oxide,
            h_min   =  HMin,
            work_function  =   ben.work_function,
        )

        Curve =  cv_sweep(deice, 'gate', list(LADDER_BIASES))

        assert Curve.complete,(
            f"ddsim did not converge on the {nsilicon} node mesh: "
            f"{Curve.message}"
        )

        _,  worrst  =  compare(
            list( LADDER_BIASES  ) ,   vars, list( Curve.charge ),  tolerance  =   1.0
        )
        temp.append(worrst)

    rep = ', '.join(
        f"{n} nodes at h_min {h:.0e}: {w:.4%}"
        for(n,h),w in zip(CONVERGENCE_LADDER,temp,strict =True)
    )



    for inndex in range(1,
         len(temp)) :
        coase=temp[inndex-1]

        fin= temp[inndex]
        assert fin   *   MINIMUM_RATIO  <=  coase ,  (
            "refining ddsim's surface mesh did not improve its agreement with "
            f"DEVSIM by {MINIMUM_RATIO:g}x. The residual is not "
            f"discretization, so something in the model differs. {rep}"
        )
    fniest = temp[-1]
    assert  fniest  <   1e-3,  (
        "ddsim converges under refinement but not onto DEVSIM: it settles at "
        f"{fniest:.4%}, an order of magnitude above where the two meshes "
        f"should stop being the limit. {rep}"
    )
