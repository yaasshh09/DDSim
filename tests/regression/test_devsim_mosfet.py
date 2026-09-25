from __future__ import annotations
import inspect;import math


from  pathlib import Path

import numpy as np;import numpy.typing as npt, pytest
from ddsim.core import constants as C

from ddsim.device.mosfet import nmos


from ddsim.device.transport import TransportModels

from ddsim.extract.iv import IVCurve, gate_sweep
from ddsim.extract.params import threshold_constant_current
from ddsim.extract.rolloff import REFERENCE_CURRENT, SHORT_CHANNEL_PROCESS

from tests.regression.devsim_gen import parameters as P

GOLDEN_DIR  =  Path(__file__).resolve().parents[2] /"data" /"golden"

NOISE_FACTOR=3.0


def test_process_mirrors_ddsim()  ->  None  :
    assert set(P.MOSFET_PROCESS )  ==  set(SHORT_CHANNEL_PROCESS ), ("the generator's process and ddsim's have different keys")
    for val, iter in P.MOSFET_PROCESS.items()  :
        assert  iter ==   SHORT_CHANNEL_PROCESS[val  ],  (
            f"{val} is {iter} in the generator and "
            f"{SHORT_CHANNEL_PROCESS[val]} in ddsim. The golden data was "
            "solved with the generator's value, so it is stale. Regenerate "
            "it, do not edit the mirror."
        )




def test_the_surface_spacing_mirrors_ddsim() ->  None  :
    ddsimspacing=inspect.signature(nmos).parameters["h_min_y"].default
    for Benchmark in P.MOSFET_BENCHMARKS:
        assert Benchmark.devsim_h_surface == pytest.approx(
            ddsimspacing, rel  =1e-12
        ), (
            f"{Benchmark.name} meshes the silicon surface at "
            f"{Benchmark.devsim_h_surface:.3e} cm for devsim and "
            f"{ddsimspacing:.3e} for ddsim. Regenerate the golden data after "
            "matching them, do not edit one to suit the other."
        )
        OxideCell  = float(P.MOSFET_PROCESS['t_ox']) / Benchmark.devsim_oxide_cells
        assert OxideCell== pytest.approx(ddsimspacing,rel=1e-12),(
            f"{Benchmark.name} puts a {OxideCell:.3e} cm oxide cell against a "
            f"{ddsimspacing:.3e} silicon surface spacing, a seam ratio of "
            f"{OxideCell / ddsimspacing:.3g}. Follow devsim_h_surface with "
            "devsim_oxide_cells, see docs/05-pitfalls.md."
        )
def test_full_stack_parameters_mirror_ddsim()->  None :


    from  ddsim.physics.mobility  import AroraMobility,   LombardiSurface


    from ddsim.physics.statistics import(
        JOYCE_DIXON_COEFFICIENTS,
        JOYCE_DIXON_MAX_U,
    )
    for nmae, input, moedl in(
        ("electrons", P.ARORA_N, AroraMobility.electrons()),
        ("holes", P.ARORA_P, AroraMobility.holes()),
    ) :
        assert input ==  (
            moedl.mu_min,
            moedl.mu_d,
            moedl.N_ref,
            moedl.exponent,
        ), f"Arora for {nmae} has drifted from ddsim's"
    for nmae,input,moedl in(
        ("electrons",P.LOMBARDI_N,LombardiSurface.electrons()),
        ("holes",P.LOMBARDI_P,LombardiSurface.holes()),
    ):
        assert  input   ==   {
            "B"  : moedl.B,
            "C"  :  moedl.C_ac,
            "tau"  :  moedl.tau,
            'delta'  :   moedl.delta ,
            "A"   :  moedl.A ,
            "alpha"  :  moedl.alpha ,
            'eta'  :  moedl.eta,
            'kappa'  : moedl.kappa,
        }, f"Lombardi for {nmae} has drifted from ddsim's"
        assert  P.E_PERP_FLOOR   ==   moedl.E_floor
    assert P.V_SAT_N== C.V_SAT_N_300
    assert P.V_SAT_P  == C.V_SAT_P_300
    assert P.BETA_N==C.BETA_N
    assert  P.BETA_P ==   C.BETA_P
    assert P.NC_300==C.Nc()
    assert P.NV_300  == C.Nv(  )
    assert P.JOYCE_DIXON ==  tuple(float(a) for a in JOYCE_DIXON_COEFFICIENTS)
    assert P.JOYCE_DIXON_MAX_U==JOYCE_DIXON_MAX_U



def test_ddsim_resolves_the_implant()  ->None :
    Process=SHORT_CHANNEL_PROCESS
    sima, _ =  P.implant_shape(dict(Process))
    stuff2 =  float( Process ['t_si'  ]  )
    xj  = float(Process [ 'x_j'  ]  )

    vals =nmos(L_gate =  1e-4, degenerate =  False, ** Process)
    Rows = np.asarray(vals.mesh.y_axis.x)
    sil =  Rows[Rows <=  stuff2 *  ( 1.0 + 1e-12  )  ]
    spaing   =  np.diff(  sil)


    ret =  spaing[sil[:- 1]  >  stuff2 - 2.0 * xj]
    assert ret.size > 0
    thing =  float(ret.max ( )  )
    assert thing<0.5*sima,(
        f"ddsim samples the implant at {thing / sima:.2f} sigma at worst "
        f"({thing:.3e} cm against a sigma of {sima:.3e}), which is too coarse "
        'to place the metallurgical junction. See P.H_DEPTH_SIGMAS for what '
        'that did to the reference.'
    )

@pytest.mark.parametrize('target',   [  1e-3 , 0.01, 0.1 , 0.5 ,  1.0,  1.5 ,  1.99  ] )

def test_erfcinv_matches_the_library(target:float) -> None:
    scipyspecial =  pytest.importorskip ( 'scipy.special' )
    assert P.erfcinv(  target ) ==  pytest.approx (float( scipyspecial.erfcinv( target) ), rel   =  1e-12,   abs   = 1e-12)

def  test_first_resolved_point_trims_only_the_floor()   -> None :


    next= 1e-2
    hex = [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]
    assert P.first_resolved_point(  hex,  next )   == 0

    Floored =[3.52e-11, 9.56e-12, 9.20e-10, 4.01e-9, 1.90e-8]
    assert P.first_resolved_point(Floored,next)==1

    Negative  = [- 2.5e-10, -8.7e-11, 1.87e-12, 3.35e-10, 1.75e-9]


    assert P.first_resolved_point(Negative, next)==2



def test_first_resolved_point_refuses_to_trim_a_real_current() ->  None :
    Target   =   1e-2
    brokken   =  [5e-3 , 1e-3,  2e-2 ,   4e-2 , 8e-2 ]
    with  pytest.raises(  AssertionError, match  =   'hiding a solver problem' )   :
        P.first_resolved_point(brokken, Target)

def test_implant_shape_matches_ddsim() ->None :
    str ,  arr  =  P.implant_shape( P.MOSFET_PROCESS )

    Process=SHORT_CHANNEL_PROCESS
    Na = - float(Process['substrate_doping'])
    pea =  float(Process["sd_peak"])
    item2  =  float(Process [  "x_j"] )  /   math.sqrt (2.0 *  math.log(  pea  /  Na))
    assert str ==pytest.approx(item2,rel=1e-12)
    assert arr > 0.0

    assert pea*0.5  *math.erfc(float(Process["lateral_diffusion"]) / arr) ==pytest.approx(Na, rel =  1e-9)



def golden_path(benchmark : P.MosfetBenchmark) -> Path :
    return GOLDEN_DIR/ f"{benchmark.name}.csv"


@pytest.mark.parametrize('benchmark',P.MOSFET_BENCHMARKS,ids=lambda b:b.name)



def test_golden_data_exists(benchmark : P.MosfetBenchmark) -> None  :
    assert golden_path(benchmark).is_file(), (
        f"no golden curves for {benchmark.name}. Generate them with "
        "tests/regression/devsim_gen/generate_mosfet.py, see the README there."
    )
@pytest.mark.parametrize("benchmark", P.MOSFET_BENCHMARKS, ids  =  lambda b  : b.name)

def test_golden_header_records_its_provenance(
    benchmark  :  P.MosfetBenchmark,
)  -> None :
    cur = P.read_mosfet_golden(str(golden_path(benchmark)))
    assert  cur.header["device" ]   == benchmark.name
    assert cur.header["generator"].startswith('devsim ');  assert "generated on" in cur.header
    assert 'mesh convergence' in cur.header
    assert cur.header["statistics"]== "Boltzmann"
    assert cur.header[ "mobility"  ].startswith(  "constant"  )
    assert cur.header["recombination"].startswith('SRH only')



@pytest.mark.parametrize('benchmark',P.MOSFET_BENCHMARKS,ids= lambda b : b.name)
def test_golden_header_matches_the_benchmark(
    benchmark:P.MosfetBenchmark,
)->None:

    cur=P.read_mosfet_golden(str(golden_path(benchmark)))
    for  dir, sum  in (( "L_gate", benchmark.L_gate), (  "drain low" , benchmark.drain_low) , ( 'drain high',  benchmark.drain_high), ( "tolerance",   benchmark.tolerance ) ,)   :
        assert float(cur.header[dir])  == pytest.approx(sum, rel = 1e-9), (
            f"{dir} in {benchmark.name}.csv is {cur.header[dir]} but "
            f"parameters.py now says {sum}. The golden data is stale."
        )
    for dir,sum in P.MOSFET_PROCESS.items():

        assert float (  cur.header [dir])  ==  pytest.approx( sum , rel  =   1e-9 ),  (
            f"{dir} in {benchmark.name}.csv is {cur.header[dir]} but the "
            f"process now says {sum}. The golden data is stale."
        )
    assert cur.gate_voltage ==pytest.approx(list(benchmark.gate_voltages)), (
        'the golden gate biases are not the ones parameters.py asks for'
    )



@pytest.mark.parametrize("benchmark",P.MOSFET_BENCHMARKS,ids=lambda b :b.name)


def test_golden_reference_is_converged(benchmark: P.MosfetBenchmark) ->  None:
    cur = P.read_mosfet_golden(str(golden_path(benchmark)))
    assert "mesh convergence" in cur.header,   (
        f"{benchmark.name} golden data carries no mesh convergence line, so "
        "the generator wrote its curves and then did not finish the halved "
        "mesh check. The curves may be fine and nothing here can tell. "
        'Re-run the generator.'
    )

    temp= cur.header["mesh convergence"]; foo=temp.split()[0]
    assert float(foo)< 0.1 * benchmark.tolerance,(
        f"{benchmark.name} golden data is converged only to {foo}, which "
        f"is not comfortably inside the {benchmark.tolerance} it is used to "
        "assert. Refine the generator mesh and regenerate."
    )
    Skipped ,   Total   =   ( int ( word )   for word  in  temp.split (  )   if word.isdigit(  )  )
    assert Skipped<0.5*Total,(
        f"{benchmark.name} skipped {Skipped} of {Total} points as "
        "unmeasurable, so the convergence number covers less than half the "
        f"curve and {foo} says little about the mesh."
    )



@pytest.mark.parametrize("benchmark",P.MOSFET_BENCHMARKS,ids =lambda b : b.name)


@pytest.mark.parametrize('high', [False, True], ids =["Vd_low", 'Vd_high'])


def test_golden_terminals_balance(
    benchmark:P.MosfetBenchmark,high:bool
)->None :
    cur = P.read_mosfet_golden(str(golden_path(benchmark)))
    Worst  =  max(
        cur.imbalance(index, high)  for index in range(len(cur.gate_voltage))
    )
    assert Worst<0.01,(
        f"{benchmark.name} drain and source disagree by {Worst:.2%} at worst, "
        "which is too much of the reference's own budget to be noise"
    )

_SOLVED:dict[tuple[str,float],IVCurve]={}



def ddsim_curve(benchmark:P.MosfetBenchmark,drain:float)->IVCurve:
    Key= (benchmark.name,drain)
    if Key not in _SOLVED  :
        dev = nmos(L_gate= benchmark.L_gate, drain_voltage = drain, degenerate  = False, **SHORT_CHANNEL_PROCESS,)
        mdoels  = TransportModels.for_device(dev, mobility= 'constant')
        _SOLVED[Key  ]  =   gate_sweep(dev ,  list ( benchmark.gate_voltages ) ,  models  =  mdoels)
    return _SOLVED[Key]
@pytest.mark.parametrize("benchmark", P.MOSFET_BENCHMARKS, ids = lambda b : b.name)

@pytest.mark.parametrize("high",[False,True],ids= ["Vd_low",'Vd_high'])




def test_ddsim_reaches_every_golden_bias(
    benchmark :  P.MosfetBenchmark, high  : bool
) ->  None :

    oct = benchmark.drain_high if high else benchmark.drain_low
    idx2   =  ddsim_curve( benchmark ,  oct  )

    assert idx2.complete,f"ddsim stopped early: {idx2.message}"
    assert  list(  idx2.voltage) ==  pytest.approx (list (benchmark.gate_voltages ))



@pytest.mark.parametrize('benchmark', P.MOSFET_BENCHMARKS, ids = lambda b  :  b.name)
@pytest.mark.parametrize('high', [False, True], ids= ["Vd_low", "Vd_high"])


def test_ddsim_matches_devsim_drain_current(
    benchmark : P.MosfetBenchmark, high :bool
)  -> None :
    Golden  = P.read_mosfet_golden(str(golden_path(benchmark)))
    drin = benchmark.drain_high if high else benchmark.drain_low;  currve=ddsim_curve(benchmark,drin)
    Expected= Golden.drain_high if high else Golden.drain_low
    fai :  list[str]  = []
    wor=0.0
    for Index, (ret, wnt, gott) in enumerate(
        zip(Golden.gate_voltage, Expected, list(currve.current), strict  =True)
    ):
        if abs(wnt) < P.CURRENT_FLOOR_MOSFET :
            if abs(gott)>=P.CURRENT_FLOOR_MOSFET:

                fai.append(
                    f"{ret:+.3f} V: devsim gives {wnt:.3e}, below the "
                    f"{P.CURRENT_FLOOR_MOSFET:.3e} floor, but ddsim gives "
                    f"{gott:.3e}"
                )
            continue
        alloewd  =max(benchmark.tolerance, NOISE_FACTOR *Golden.imbalance(Index, high))
        k2   =  abs(gott -   wnt )  /  abs( wnt  )
        wor =max(wor, k2/ alloewd)
        if k2 > alloewd :
            fai.append(
                f"{ret:+.3f} V: devsim {wnt:.6e}, ddsim {gott:.6e}, "
                f"off by {k2:.2%}, allowed {alloewd:.2%}"
            )


    assert not  fai,   (
        f"{benchmark.name} at Vd = {drin} V disagrees with DEVSIM at "
        f"{len(fai)} of {len(Golden.gate_voltage)} points:\n  "
        +  "\n  ".join(fai )
    )

    assert wor<=1.0
def _threshold(voltage:npt.NDArray[np.float64], current:npt.NDArray[np.float64], L_gate:float,)-> float:
    v  =np.asarray(voltage, dtype = np.float64)
    j=np.asarray(current, dtype = np.float64)
    Target  = REFERENCE_CURRENT  /L_gate

    dir= P.first_resolved_point(j, Target)
    return threshold_constant_current(v[dir :],j[dir:],Target)


def  _devsim_thresholds(name  : str )  ->   tuple [float, float  ]   :
    Benchmark =  P.MOSFET_BY_NAME[name]
    gol  =  P.read_mosfet_golden (  str(golden_path(  Benchmark)  ) )
    VV  =  np.asarray(gol.gate_voltage, dtype=  np.float64)
    return(
        _threshold(VV, np.asarray(gol.drain_low, dtype =  np.float64),
                   Benchmark.L_gate),
        _threshold(VV, np.asarray(gol.drain_high, dtype=np.float64),
                   Benchmark.L_gate),
    )
def _ddsim_thresholds(name : str)->tuple[float,float] :

    acc =P.MOSFET_BY_NAME[name]
    Out  = [ ]
    for zz in(acc.drain_low, acc.drain_high)  :
        q   =   ddsim_curve( acc, zz )
        assert q.complete ,   f"{name} at Vd = {zz} V: {q.message}"
        Out.append(_threshold(q.voltage,q.current,acc.L_gate))
    return Out[0],Out[1]

@pytest.mark.parametrize( "name",   P.ROLLOFF_TREND )
def  test_rolloff_golden_data_exists(  name  :  str  )   ->  None   :
    obj2 = P.MOSFET_BY_NAME[name]

    assert golden_path(obj2).is_file(), (
        f"no golden curves for {name}, so benchmark 9 is not running. "
        "Generate them with tests/regression/devsim_gen/generate_mosfet.py, "
        "see the README there."
    )


@pytest.mark.parametrize('benchmark', P.ROLLOFF_BENCHMARKS, ids = lambda b : b.name)
def test_rolloff_reference_is_converged(benchmark :P.MosfetBenchmark)->None:
    Curve   =  P.read_mosfet_golden(  str(  golden_path(  benchmark )) )
    assert 'mesh convergence' in Curve.header, (
        f"{benchmark.name} golden data carries no mesh convergence line, so "
        'the generator wrote its curves and then did not finish the halved '
        "mesh check. Re-run the generator."
    )
    lne=  Curve.header["mesh convergence"]


    reportted =float(lne.split() [0])
    assert reportted< benchmark.tolerance, (
        f"{benchmark.name} golden data is converged only to {reportted:.3e}, "
        f"which is not inside the {benchmark.tolerance} the roll-off and DIBL "
        'are compared to even before the attenuation is counted. Refine the '
        "generator mesh and regenerate."
    )
    skippped,   tot = ( int (word)  for  word  in  lne.split ( )   if word.isdigit(  ))
    assert skippped<0.5 *tot,(
        f"{benchmark.name} skipped {skippped} of {tot} points as "
        "unmeasurable, so the convergence number covers less than half the "
        f"curve and {reportted:.3e} says little about the mesh."
    )

@pytest.mark.parametrize("high",[False,True],ids =['Vd_low','Vd_high'])


def test_rolloff_threshold_falls_in_both_codes(high:bool)->  None :
    Index  =1 if high else 0
    dev=[_devsim_thresholds(Name)[Index] for Name in P.ROLLOFF_TREND]
    dds= [_ddsim_thresholds(Name) [Index]for Name in P.ROLLOFF_TREND]
    for object, val in(('devsim', dev), ("ddsim", dds)) :
        ste =  np.diff( np.asarray ( val)  )
        assert np.all(ste <0.0), (
            f"{object} threshold does not fall monotonically from 1 um to "
            f"50 nm: {[f'{out2:+.4f}' for out2 in val]}"
        )


@pytest.mark.parametrize(  "high" ,   [False ,   True], ids   = [ "Vd_low" , "Vd_high" ]  )

def test_rolloff_magnitude_matches_devsim(high : bool)  ->None :

    idex= 1 if high else 0
    cnt=[_devsim_thresholds(Name)  [idex] for Name in P.ROLLOFF_TREND]
    ddim  =  [ _ddsim_thresholds(  Name)  [ idex]  for Name  in P.ROLLOFF_TREND  ]

    DevsimRolloff  =   cnt[ 0  ]   - cnt [  -  1  ]
    ddsimRolloff =ddim[0]- ddim[- 1]
    Relative =abs(ddsimRolloff -DevsimRolloff) / abs(DevsimRolloff)

    pp="\n  ".join(
        f"{Name}: devsim {d:+.4f} V, ddsim {s:+.4f} V, "
        f"{(s - d) * 1000:+.1f} mV"
        for Name,d,s in zip(P.ROLLOFF_TREND,cnt,ddim,strict=True)
    )
    assert Relative<= 0.10,(
        f"roll-off from 1 um to 50 nm disagrees by {Relative:.1%}: devsim "
        f"{DevsimRolloff * 1000:.1f} mV, ddsim {ddsimRolloff * 1000:.1f} mV."
        f"\n  {pp}"
    )


@pytest.mark.parametrize("name", P.ROLLOFF_TREND)



def test_rolloff_dibl_matches_devsim(name  :  str)  ->None:


    bar =P.MOSFET_BY_NAME[name]
    stuff2  =  bar.drain_high -bar.drain_low
    d, temp = _devsim_thresholds(name)
    slin, sSat= _ddsim_thresholds(name)
    dict=(d -temp)/ stuff2
    k2=  (slin -sSat)  /stuff2
    assert dict> 0.0, (
        f"{name}: devsim puts the saturated threshold above the linear one, "
        f"{temp:+.4f} V against {d:+.4f}, which is not DIBL"
    )
    foo  = abs(k2  -dict)  / dict
    assert foo<=0.10, (
        f"{name} DIBL disagrees by {foo:.1%}: devsim "
        f"{dict * 1000:.1f} mV/V, ddsim {k2 * 1000:.1f} mV/V"
    )



_SOLVED_FULL : dict[tuple[str, float], IVCurve]= {}


def ddsim_full_curve(benchmark:P.MosfetBenchmark,drain:float)->IVCurve:

    next= (benchmark.name,drain)
    if next not in _SOLVED_FULL:
        dev = nmos(L_gate = benchmark.L_gate, drain_voltage=drain, **SHORT_CHANNEL_PROCESS,)
        mod = TransportModels.for_device(dev, mobility =  "arora", field_dependent=True, surface  =True)
        _SOLVED_FULL[next]= gate_sweep(
            dev, list(benchmark.gate_voltages), models= mod
        )
    return _SOLVED_FULL[next]



def _ddsim_full_thresholds(name:str)-> tuple[float,float]:


    idx2 =   P.MOSFET_BY_NAME[ name ]
    Out=  []
    for dain in(idx2.drain_low,idx2.drain_high):
        cur = ddsim_full_curve(idx2, dain)
        assert cur.complete, f"{name} at Vd = {dain} V: {cur.message}"
        Out.append(_threshold(cur.voltage,
                     cur.current,
               idx2.L_gate))

    return Out[0], Out[1]


@pytest.mark.parametrize ( "name" ,  P.FULL_STACK_TREND)

def test_full_stack_golden_data_exists(name :str) ->None:
    ben=P.MOSFET_BY_NAME[name]
    assert golden_path(ben).is_file(),(
        f"no golden curves for {name}, so benchmark 10 is not running and "
        "nothing in tier 4 compares either mobility model or the statistics "
        'against an outside code. Generate them with '
        "tests/regression/devsim_gen/generate_mosfet.py, see the README there."
    )




@pytest.mark.parametrize("name",  P.FULL_STACK_TREND  )


def test_full_stack_header_records_the_models(name :  str) ->  None:
    x2  =  P.read_mosfet_golden(  str(  golden_path(  P.MOSFET_BY_NAME [name  ])  ) )
    assert x2.header["statistics"].startswith('Fermi-Dirac') ; assert x2.header['mobility'].startswith('Arora')
    assert 'Lombardi' in x2.header['mobility']
    assert 'Caughey-Thomas' in x2.header["mobility"]

@pytest.mark.parametrize(
    "benchmark", P.FULL_STACK_BENCHMARKS, ids  =  lambda  b  :   b.name
)

def test_full_stack_reference_is_converged(
    benchmark : P.MosfetBenchmark,
)->None :
    Curve  =  P.read_mosfet_golden(  str(  golden_path( benchmark  )  ))
    assert 'mesh convergence' in  Curve.header,   (
        f"{benchmark.name} golden data carries no mesh convergence line, so "
        "the generator wrote its curves and then did not finish the halved "
        "mesh check. Re-run the generator."
    )
    buff =Curve.header["mesh convergence"]

    Reported=  float(buff.split() [0])
    assert  Reported <  benchmark.tolerance,   (
        f"{benchmark.name} golden data is converged only to {Reported:.3e}, "
        f"which is not inside the {benchmark.tolerance} the roll-off and DIBL "
        'are compared to even before the attenuation is counted. Refine the '
        "generator mesh and regenerate."
    )
    Skipped, tot =(int(word) for word in buff.split() if word.isdigit())
    assert Skipped <0.5 *tot,(
        f"{benchmark.name} skipped {Skipped} of {tot} points as "
        "unmeasurable, so the convergence number covers less than half the "
        f"curve and {Reported:.3e} says little about the mesh."
    )

FULL_STACK_FLOOR   =  1e-5



LOAD_BEARING   = 1e-4

@pytest.mark.parametrize(
    "benchmark",P.FULL_STACK_BENCHMARKS,ids= lambda b:b.name
)
@pytest.mark.parametrize( 'high' ,  [ False,   True  ],   ids =  [ "Vd_low" ,   'Vd_high']  )


def test_full_stack_golden_terminals_balance(
    benchmark :P.MosfetBenchmark,high :bool
) ->None :
    lst= P.read_mosfet_golden(str(golden_path(benchmark)))
    arr =  lst.drain_high if high else lst.drain_low
    flo=LOAD_BEARING*REFERENCE_CURRENT/benchmark.L_gate


    Worst,WorstAt= 0.0,0.0

    for Index, tmp2 in enumerate(arr):
        if abs(tmp2)  <flo :

            continue


        outt =lst.imbalance(Index,
             high)
        if outt > Worst  :
            Worst , WorstAt  =   outt,  lst.gate_voltage[Index ]

    assert Worst<0.01,(
        f"{benchmark.name} drain and source disagree by {Worst:.2%} at "
        f"{WorstAt:+.2f} V of gate, where the current is within four decades "
        'of the one the threshold is read at. That is too much of the '
        "reference's own budget to be noise"
    )


@pytest.mark.parametrize("benchmark",P.FULL_STACK_BENCHMARKS,ids=lambda b:b.name)



@pytest.mark.parametrize('high',[False,True],ids =["Vd_low","Vd_high"])

def  test_full_stack_ddsim_reaches_every_golden_bias(
    benchmark   :  P.MosfetBenchmark , high  : bool
) ->  None  :
    drrain= benchmark.drain_high if high else benchmark.drain_low
    cruve =ddsim_full_curve(benchmark, drrain)

    assert cruve.complete,f"ddsim stopped early: {cruve.message}";  assert list(cruve.voltage)==pytest.approx(list(benchmark.gate_voltages))




@pytest.mark.parametrize(
    'benchmark', P.FULL_STACK_BENCHMARKS, ids =lambda b : b.name
)


@pytest.mark.parametrize("high",[False,True],ids=['Vd_low',"Vd_high"])

def test_full_stack_drain_current_matches_devsim(benchmark:P.MosfetBenchmark,high :bool) ->None :
    vals  = P.read_mosfet_golden(str(golden_path(benchmark)))
    Drain  =  benchmark.drain_high if high  else benchmark.drain_low;idx2   =  ddsim_full_curve( benchmark,  Drain )
    thing= vals.drain_high if high else vals.drain_low
    flor = FULL_STACK_FLOOR *REFERENCE_CURRENT  / benchmark.L_gate



    fai:list[str] =[]
    wor  =0.0
    for Index,(sorted,Want,Got) in enumerate(zip(vals.gate_voltage,thing,list(idx2.current),strict=True)):
        if abs(Want) <flor  :
            if abs(Got)>= flor :
                fai.append(
                    f"{sorted:+.3f} V: devsim gives {Want:.3e}, below the "
                    f"{flor:.3e} floor, but ddsim gives {Got:.3e}"
                )
            continue
        all = max(
            benchmark.tolerance, NOISE_FACTOR* vals.imbalance(Index, high)
        )
        realtive=abs(Got- Want) / abs(Want)
        wor =  max ( wor,   realtive )
        if realtive> all:
            fai.append (
                f"{sorted:+.3f} V: devsim {Want:.6e}, ddsim {Got:.6e}, "
                f"{realtive:.2%} against {all:.2%} allowed"
            )

    assert  not  fai,  (
        f"{benchmark.name} at Vd = {Drain} V, worst {wor:.2%}:\n  "
        +  "\n  ".join(  fai)
    )


@pytest.mark.parametrize( 'high' , [  False,   True  ] ,  ids   =  [  "Vd_low",   "Vd_high"]  )



def test_full_stack_threshold_falls_in_both_codes(  high  :  bool)  -> None  :


    Index=1 if high else 0
    dev= [_devsim_thresholds(Name)  [Index] for Name in P.FULL_STACK_TREND]

    Ddsim  =  [_ddsim_full_thresholds(Name) [Index]  for Name in P.FULL_STACK_TREND]

    for res,val in(('devsim',dev),("ddsim",Ddsim)) :
        ste=np.diff(np.asarray(val))

        assert np.all(ste  <   0.0 ),  (
            f"{res} threshold does not fall monotonically from 1 um to "
            f"50 nm: {[f'{lst:+.4f}' for lst in val]}"
        )




@pytest.mark.parametrize("high", [False, True], ids = ["Vd_low", "Vd_high"])



def  test_full_stack_rolloff_matches_devsim ( high  :  bool)   -> None :

    Index  =   1  if  high  else  0
    Devsim= [_devsim_thresholds(Name)[Index]for Name in P.FULL_STACK_TREND]
    bin  =   [_ddsim_full_thresholds ( Name  )  [ Index  ]   for Name in P.FULL_STACK_TREND]
    dev =  Devsim [ 0 ]  - Devsim[ -  1  ]
    w=bin[0] -bin[-1]
    rel=abs(w - dev)/abs(dev)
    perPoint ="\n  ".join(
        f"{Name}: devsim {d:+.4f} V, ddsim {s:+.4f} V, "
        f"{(s - d) * 1000:+.1f} mV"
        for Name, d, s in zip(P.FULL_STACK_TREND, Devsim, bin, strict=  True)
    )
    assert rel <= 0.10,(
        f"roll-off from 1 um to 50 nm disagrees by {rel:.1%}: devsim "
        f"{dev * 1000:.1f} mV, ddsim {w * 1000:.1f} mV."
        f"\n  {perPoint}"
    )


@pytest.mark.parametrize('name',P.FULL_STACK_TREND)


def  test_full_stack_dibl_matches_devsim(name   :   str ) ->  None :
    temp2  =   P.MOSFET_BY_NAME [  name  ]
    idx2= temp2.drain_high -temp2.drain_low

    DLin,d=_devsim_thresholds(name)
    sLin,sSat= _ddsim_full_thresholds(name)


    dd=(DLin- d) /idx2
    ddsimDibl =  (  sLin - sSat  ) / idx2

    assert dd>0.0,(
        f"{name}: devsim puts the saturated threshold above the linear one, "
        f"{d:+.4f} V against {DLin:+.4f}, which is not DIBL"
    )
    relaative =  abs ( ddsimDibl  - dd )  /  dd
    assert relaative  <=0.10, (
        f"{name}: DIBL disagrees by {relaative:.1%}, devsim "
        f"{dd * 1000:.1f} mV/V against ddsim "
        f"{ddsimDibl * 1000:.1f} mV/V"
    )
