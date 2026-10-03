from __future__ import annotations
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

    f   =   P.read_mos_golden( str(golden_path (benchmark) )  )
    assert f.header['device'] == benchmark.name
    assert f.header['generator'].startswith('devsim ')
    assert 'generated on' in f.header
    assert "mesh convergence" in f.header

    assert f.header["statistics"] =="Boltzmann"
    assert f.header["oxide"].startswith("Poisson only")
@pytest.mark.parametrize("benchmark",P.MOS_BENCHMARKS,ids=lambda b :b.name)

def test_golden_reference_is_converged(benchmark : P.MosBenchmark)-> None :
    foo=P.read_mos_golden(str(golden_path(benchmark)))
    u=foo.header["mesh convergence"].split()[0]


    assert  float(u)  <  0.1  * benchmark.tolerance,  (
        f"{benchmark.name} golden data is converged only to {u}, which "
        f"is not comfortably inside the {benchmark.tolerance} it is used to "
        "assert. Refine the generator mesh and regenerate."
    )
@pytest.mark.parametrize("benchmark", P.MOS_BENCHMARKS, ids =lambda b  : b.name)



def test_golden_header_matches_the_benchmark(benchmark : P.MosBenchmark)->None:
    el   = P.read_mos_golden (str( golden_path(benchmark ) ) )
    for out, ret  in (( 'substrate doping', benchmark.substrate_doping ) , ("t_ox",   benchmark.t_ox), (  't_si' ,   benchmark.t_si), ( 'work function',  benchmark.work_function) , ( 'tolerance' , benchmark.tolerance ) ,)  :
        assert float(el.header[out])== pytest.approx(ret,rel=1e-6),(
            f"{out} in {benchmark.name}.csv is {el.header[out]} but "
            f"parameters.py now says {ret}. The golden data is stale."
        )
    assert el.gate_voltage  ==  pytest.approx(list(benchmark.voltages)), (
        "the golden bias points are not the ones parameters.py asks for"
    )
_SOLVED:dict[str,CVCurve]={}



def ddsim_curve(benchmark  :P.MosBenchmark) -> CVCurve  :
    if benchmark.name not in _SOLVED  :
        j   =  mos_cap(
            substrate_doping  =  benchmark.substrate_doping,
            t_ox =  benchmark.t_ox,
            t_si =  benchmark.t_si ,
            n_silicon =  benchmark.n_silicon,
            n_oxide  =  benchmark.n_oxide ,
            h_min  = benchmark.h_min,
            work_function =  benchmark.work_function ,
        )
        _SOLVED [  benchmark.name]   = cv_sweep(
            j ,   'gate',   list( benchmark.voltages  )
        )
    return _SOLVED[benchmark.name]



@pytest.mark.parametrize("benchmark", P.MOS_BENCHMARKS, ids  = lambda b : b.name)


def test_ddsim_reaches_every_golden_bias(benchmark : P.MosBenchmark) -> None  :


    vals =ddsim_curve(benchmark);assert vals.complete, f"ddsim stopped early: {vals.message}"
    assert list(vals.gate_voltage) == pytest.approx(list(benchmark.voltages))


def compare(voltage : list[float], expected :list[float], actual:list[float], tolerance:float,)-> tuple[list[str],float]:
    tmp2=max(abs(u) for u in expected)
    vv=FLOOR_FRACTION*tmp2


    k :  list[str]=[]

    a =   0.0
    for rows,vals,c in zip(voltage,expected,actual,strict=True):
        if abs(vals)  < vv :

            if abs(c) >=  vv:
                k.append(
                    f"{rows:+.3f} V: devsim gives {vals:.3e}, below the "
                    f"{vv:.3e} floor, but ddsim gives {c:.3e}"
                )
            continue
        v  = abs(c -  vals) /abs(vals)
        a =max(a, v)
        if v>tolerance:
            k.append(
                f"{rows:+.3f} V: devsim {vals:.6e}, ddsim {c:.6e}, "
                f"off by {v:.3%}, allowed {tolerance:.3%}"
            )
    return k,a



@pytest.mark.parametrize("benchmark",P.MOS_BENCHMARKS,ids = lambda b:b.name)




def  test_ddsim_matches_devsim_charge(  benchmark  :  P.MosBenchmark  )  -> None  :
    w=P.read_mos_golden(str(golden_path(benchmark)))
    d2= ddsim_curve(benchmark)
    i, v  =  compare(w.gate_voltage, w.charge, list(d2.charge ), benchmark.tolerance,)
    assert not i ,  (
        f"{benchmark.name} gate charge disagrees with DEVSIM at "
        f"{len(i)} of {len(w.gate_voltage)} points:\n  "
        +  "\n  ".join ( i  )
    )
    assert  v  <=  benchmark.tolerance


@pytest.mark.parametrize ( 'benchmark' ,  P.MOS_BENCHMARKS,   ids   =   lambda  b   :  b.name  )
def test_ddsim_matches_devsim_capacitance(benchmark :  P.MosBenchmark  )  -> None  :
    i = P.read_mos_golden(str(golden_path(benchmark)))
    h=ddsim_curve(benchmark)


    r,e= P.central_difference(i.gate_voltage,i.charge)
    _,c =P.central_difference(
        list(h.gate_voltage),list(h.charge)
    )

    x,t= compare(r,e,c,benchmark.tolerance)

    assert not x,(
        f"{benchmark.name} capacitance disagrees with DEVSIM at "
        f"{len(x)} of {len(r)} points:\n  "+"\n  ".join(x)
    )
    assert  t  <=  benchmark.tolerance
CONVERGENCE_LADDER : tuple[tuple[int,float],...] = ((121,5e-8), (161,2e-8), (201,1e-8),)

LADDER_BIASES:tuple[float,
  ...]= (-2.0,
                 1.0,
              2.0)
MINIMUM_RATIO = 2.0


def test_disagreement_is_ddsim_discretization()->None :
    c =  P.MOS_BENCHMARKS[  0 ]
    k= P.read_mos_golden(str(golden_path(c)))
    i=dict(zip(k.gate_voltage,k.charge,strict=True))
    z2   = [ i [ el]   for el in LADDER_BIASES]

    z :list[float]=[]
    for  d ,   rows  in  CONVERGENCE_LADDER   :
        b2 =  mos_cap(
            substrate_doping  =   c.substrate_doping ,
            t_ox =  c.t_ox,
            t_si  = c.t_si,
            n_silicon   =  d,
            n_oxide =  c.n_oxide,
            h_min   =  rows,
            work_function  =   c.work_function,
        )

        t =  cv_sweep(b2, 'gate', list(LADDER_BIASES))

        assert t.complete,(
            f"ddsim did not converge on the {d} node mesh: "
            f"{t.message}"
        )

        _,  b  =  compare(
            list( LADDER_BIASES  ) ,   z2, list( t.charge ),  tolerance  =   1.0
        )
        z.append(b)

    j = ', '.join(
        f"{n} nodes at h_min {h:.0e}: {u:.4%}"
        for(n,h),u in zip(CONVERGENCE_LADDER,z,strict =True)
    )



    for ss in range(1,
         len(z)) :
        r=z[ss-1]

        res= z[ss]
        assert res   *   MINIMUM_RATIO  <=  r ,  (
            "refining ddsim's surface mesh did not improve its agreement with "
            f"DEVSIM by {MINIMUM_RATIO:g}x. The residual is not "
            f"discretization, so something in the model differs. {j}"
        )
    v = z[-1]
    assert  v  <   1e-3,  (
        "ddsim converges under refinement but not onto DEVSIM: it settles at "
        f"{v:.4%}, an order of magnitude above where the two meshes "
        f"should stop being the limit. {j}"
    )
