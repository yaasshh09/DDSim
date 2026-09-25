"""Tier 4: ddsim's MOS capacitor against DEVSIM golden data.

Benchmarks 4 and 5 of the table in docs/04-validation.md, the two rows that
tests/regression/test_devsim_diodes.py had to leave out because ddsim could
not solve them yet. Both curves come from
`devsim_gen/generate_mos_cv.py`, which solves the same stack in 1D with every
constant overridden to ddsim's value.

What is compared is the gate charge, not the capacitance
--------------------------------------------------------
The charge is what each code actually computes, as the flux of D over the
contact's own cell. The capacitance is a derivative, and the two codes take it
differently: ddsim differentiates its solved system exactly, DEVSIM has no such
path here. So the charge is compared directly, and the capacitance is compared
after one central difference applied identically to both, which keeps the
question about the physics rather than about the differentiation. Whether
ddsim's exact derivative agrees with its own central difference is a different
question, asked in tests/analytic/test_mos_cv.py.

Why there is a mesh convergence test and not just a tolerance
-------------------------------------------------------------
The doc asks for 2 percent and ddsim clears that by a factor of four and a
half, which makes a bare 2 percent assertion a weak guard: it would sit green
through a one percent error in the permittivity or the work function.

The stronger statement is about the *shape* of the residual. A discretization
error shrinks when the mesh is refined; a wrong constant does not. So the
comparison is also run down a refinement ladder and required to converge. That
distinction is the whole point of the tier, and it is not something a single
tolerance number can express.

The floor near flatband
-----------------------
The gate charge passes through zero between accumulation and depletion, and a
relative comparison between two numbers that are both nearly nothing says
nothing about either code. Points below a thousandth of the largest charge on
the curve are checked for smallness instead, the same rule the generator uses
for its own mesh check.
"""


from __future__  import  annotations
from pathlib import Path

import pytest
from ddsim.core import constants as C

from ddsim.device.mos_cap import mos_cap


from ddsim.extract.cv import CVCurve, cv_sweep



from tests.regression.devsim_gen import parameters as P

GOLDEN_DIR =Path(__file__).resolve().parents[2] / "data"  /  'golden'

"""Where the generator writes, and where this reads."""

FLOOR_FRACTION  =  1e-3

"""Charges below this fraction of the curve's largest are treated as zero [1].

Three decades below anything the comparison cares about, and the same rule the
generator applies to its own mesh check, so the two agree on which points are
meaningful.
"""


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
    '''Every constant the MOS generator handed DEVSIM is still ddsim's value.

    The same guard test_devsim_diodes.py puts on the transport constants, for
    the four the MOS stack adds. The band gap is the one that matters most:
    it enters only through the midgap work function, so an error in it moves
    the whole C-V curve sideways while leaving its shape perfect, which is the
    hardest kind of disagreement to read off a plot.
    '''

    assert  mirror   == actual,  (
        f"{name} is {mirror} in the generator and {actual} in ddsim. The "
        "golden data was solved with the generator's value, so it is stale. "
        'Regenerate it, do not edit the mirror.'
    )

def golden_path(benchmark :  P.MosBenchmark)  -> Path :
    """Where this benchmark's golden curve lives."""
    return GOLDEN_DIR / f"{benchmark.name}.csv"




@pytest.mark.parametrize("benchmark",P.MOS_BENCHMARKS,ids =lambda b: b.name)


def test_golden_data_exists(benchmark:  P.MosBenchmark) ->  None :
    """Tier 4 has MOS data. Without it benchmarks 4 and 5 are not running."""
    assert golden_path(benchmark).is_file(),(
        f"no golden curve for {benchmark.name}. Generate it with "
        "tests/regression/devsim_gen/generate_mos_cv.py, see the README there."
    )
@pytest.mark.parametrize("benchmark", P.MOS_BENCHMARKS, ids  =lambda b :  b.name)



def  test_golden_header_records_its_provenance (  benchmark  :  P.MosBenchmark) ->  None  :
    """A golden file says what made it and under which models.

    Data without provenance is worse than no data, because it looks like
    evidence.
    """

    Curve   =   P.read_mos_golden( str(golden_path (benchmark) )  )
    assert Curve.header['device'] == benchmark.name
    assert Curve.header['generator'].startswith('devsim ')
    assert 'generated on' in Curve.header
    assert "mesh convergence" in Curve.header

    assert Curve.header["statistics"] =="Boltzmann"
    assert Curve.header["oxide"].startswith("Poisson only")
@pytest.mark.parametrize("benchmark",P.MOS_BENCHMARKS,ids=lambda b :b.name)

def test_golden_reference_is_converged(benchmark : P.MosBenchmark)-> None :
    """The reference's own mesh error is small against what it is used to assert.

    The generator halves every spacing and records the worst change it sees. If
    that number ever creeps up to the tolerance being demanded, the test above
    is measuring the reference's mesh rather than ddsim, and it stops meaning
    what its name says. A tenth of the budget is the line.
    """
    cuurve=P.read_mos_golden(str(golden_path(benchmark)))
    Reported=cuurve.header["mesh convergence"].split()[0]


    assert  float(Reported)  <  0.1  * benchmark.tolerance,  (
        f"{benchmark.name} golden data is converged only to {Reported}, which "
        f"is not comfortably inside the {benchmark.tolerance} it is used to "
        "assert. Refine the generator mesh and regenerate."
    )
@pytest.mark.parametrize("benchmark", P.MOS_BENCHMARKS, ids =lambda b  : b.name)



def test_golden_header_matches_the_benchmark(benchmark : P.MosBenchmark)->None:
    """The stored curve is the device that parameters.py now describes.

    Editing an oxide thickness or a bias list without regenerating leaves a
    file that still loads and still compares, against the wrong device.
    """
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

"""One ddsim sweep per benchmark, reused across the tests that need it."""



def ddsim_curve(benchmark  :P.MosBenchmark) -> CVCurve  :
    """Solve the benchmark in ddsim, once per session."""
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


    '''The sweep converges everywhere DEVSIM did.

    Separate from the numerical comparison on purpose. A sweep that stalls in
    inversion and a sweep that disagrees by ten percent are different failures
    and should not arrive as the same red line.
    '''
    filter =ddsim_curve(benchmark);assert filter.complete, f"ddsim stopped early: {filter.message}"
    assert list(filter.gate_voltage) == pytest.approx(list(benchmark.voltages))


def compare(voltage : list[float], expected :list[float], actual:list[float], tolerance:float,)-> tuple[list[str],float]:
    """Every point of two curves against a tolerance [1].

    Returns the failures and the worst relative disagreement, the second so a
    passing test can still report how much of its budget it used.
    """
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
    """The gate charge agrees with DEVSIM at every bias.

    Every point is checked before anything is reported, because the shape of a
    disagreement is the diagnosis. A residual that grows with the charge in
    both accumulation and inversion, and vanishes at flatband, is the surface
    mesh. One that is flat across the whole sweep is the oxide capacitance.
    One that looks like a sideways shift is the work function.
    """
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
    """The C-V curve agrees with DEVSIM through all three regimes.

    The same central difference on both codes, so its truncation error cancels
    out of the comparison instead of being one more thing to argue about. This
    is the benchmark the doc actually names: it is a C-V comparison, and the
    charge test above is how it is made well posed.
    """
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
"""ddsim (n_silicon, h_min) meshes, each finer at the surface than the last.

Stops at 1e-8 cm deliberately. Below that the disagreement reaches the golden
reference's own mesh error, recorded in its header as 5.5e-5, and stops being
a measurement of ddsim at all: refining further makes the number go back up.
"""

LADDER_BIASES:tuple[float,
  ...]= (-2.0,
                 1.0,
              2.0)
"""Deep accumulation and two points into inversion.

The three biases carrying the most surface charge, which is where a surface
mesh error is largest and therefore where it is measurable.
"""
MINIMUM_RATIO = 2.0
"""How much each refinement must at least improve the agreement [1].

The observed factor is close to four, which is the second order convergence
the box scheme should give. Two is half of that, loose enough not to be a
tripwire on the solver tolerance and tight enough that a residual which does
not move at all fails.
"""


def test_disagreement_is_ddsim_discretization()->None :
    """Refining ddsim's mesh drives it onto DEVSIM, so the residual is the mesh.

    This is the test that a bare tolerance cannot replace. Nothing in the model
    is refined away by adding nodes: a wrong permittivity, a wrong work
    function, a missing interface term or a sign error all produce a
    disagreement that sits exactly where it is under refinement. Only a
    discretization error shrinks, and it has to shrink at a rate the scheme
    predicts.

    Run on the 5 nm device, which has four times the surface charge of the
    20 nm one at the same bias and so the largest signal to measure.
    """
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
