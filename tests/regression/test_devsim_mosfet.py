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
    for r, k in P.MOSFET_PROCESS.items()  :
        assert  k ==   SHORT_CHANNEL_PROCESS[r  ],  (
            f"{r} is {k} in the generator and "
            f"{SHORT_CHANNEL_PROCESS[r]} in ddsim. The golden data was "
            "solved with the generator's value, so it is stale. Regenerate "
            "it, do not edit the mirror."
        )




def test_the_surface_spacing_mirrors_ddsim() ->  None  :
    s2=inspect.signature(nmos).parameters["h_min_y"].default
    for b in P.MOSFET_BENCHMARKS:
        assert b.devsim_h_surface == pytest.approx(
            s2, rel  =1e-12
        ), (
            f"{b.name} meshes the silicon surface at "
            f"{b.devsim_h_surface:.3e} cm for devsim and "
            f"{s2:.3e} for ddsim. Regenerate the golden data after "
            "matching them, do not edit one to suit the other."
        )
        info  = float(P.MOSFET_PROCESS['t_ox']) / b.devsim_oxide_cells
        assert info== pytest.approx(s2,rel=1e-12),(
            f"{b.name} puts a {info:.3e} cm oxide cell against a "
            f"{s2:.3e} silicon surface spacing, a seam ratio of "
            f"{info / s2:.3g}. Follow devsim_h_surface with "
            "devsim_oxide_cells, see references/pitfalls.md."
        )
def test_full_stack_parameters_mirror_ddsim()->  None :


    from  ddsim.physics.mobility  import AroraMobility,   LombardiSurface
    from ddsim.physics.statistics import(
        JOYCE_DIXON_COEFFICIENTS,
        JOYCE_DIXON_MAX_U,
    )
    for dd, k, i in(
        ("electrons", P.ARORA_N, AroraMobility.electrons()),
        ("holes", P.ARORA_P, AroraMobility.holes()),
    ) :
        assert k ==  (
            i.mu_min,
            i.mu_d,
            i.N_ref,
            i.exponent,
        ), f"Arora for {dd} has drifted from ddsim's"
    for dd,k,i in(
        ("electrons",P.LOMBARDI_N,LombardiSurface.electrons()),
        ("holes",P.LOMBARDI_P,LombardiSurface.holes()),
    ):
        assert  k   ==   {
            "B"  : i.B,
            "C"  :  i.C_ac,
            "tau"  :  i.tau,
            'delta'  :   i.delta ,
            "A"   :  i.A ,
            "alpha"  :  i.alpha ,
            'eta'  :  i.eta,
            'kappa'  : i.kappa,
        }, f"Lombardi for {dd} has drifted from ddsim's"
        assert  P.E_PERP_FLOOR   ==   i.E_floor
    assert P.V_SAT_N== C.V_SAT_N_300
    assert P.V_SAT_P  == C.V_SAT_P_300
    assert P.BETA_N==C.BETA_N
    assert  P.BETA_P ==   C.BETA_P
    assert P.NC_300==C.Nc()
    assert P.NV_300  == C.Nv(  )
    assert P.JOYCE_DIXON ==  tuple(float(w) for w in JOYCE_DIXON_COEFFICIENTS)
    assert P.JOYCE_DIXON_MAX_U==JOYCE_DIXON_MAX_U



def test_ddsim_resolves_the_implant()  ->None :
    k=SHORT_CHANNEL_PROCESS
    dd, _ =  P.implant_shape(dict(k))
    d =  float( k ['t_si'  ]  )
    t  = float(k [ 'x_j'  ]  )

    u =nmos(L_gate =  1e-4, degenerate =  False, ** k)
    g = np.asarray(u.mesh.y_axis.x)
    item =  g[g <=  d *  ( 1.0 + 1e-12  )  ]
    flag   =  np.diff(  item)


    tmp3 =  flag[item[:- 1]  >  d - 2.0 * t]
    assert tmp3.size > 0
    arr =  float(tmp3.max ( )  )
    assert arr<0.5*dd,(
        f"ddsim samples the implant at {arr / dd:.2f} sigma at worst "
        f"({arr:.3e} cm against a sigma of {dd:.3e}), which is too coarse "
        'to place the metallurgical junction. See P.H_DEPTH_SIGMAS for what '
        'that did to the reference.'
    )

@pytest.mark.parametrize('target',   [  1e-3 , 0.01, 0.1 , 0.5 ,  1.0,  1.5 ,  1.99  ] )

def test_erfcinv_matches_the_library(target:float) -> None:
    z =  pytest.importorskip ( 'scipy.special' )
    assert P.erfcinv(  target ) ==  pytest.approx (float( z.erfcinv( target) ), rel   =  1e-12,   abs   = 1e-12)

def  test_first_resolved_point_trims_only_the_floor()   -> None :


    kk= 1e-2
    ys = [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]
    assert P.first_resolved_point(  ys,  kk )   == 0

    k =[3.52e-11, 9.56e-12, 9.20e-10, 4.01e-9, 1.90e-8]
    assert P.first_resolved_point(k,kk)==1

    item  = [- 2.5e-10, -8.7e-11, 1.87e-12, 3.35e-10, 1.75e-9]


    assert P.first_resolved_point(item, kk)==2



def test_first_resolved_point_refuses_to_trim_a_real_current() ->  None :
    u   =   1e-2
    idx   =  [5e-3 , 1e-3,  2e-2 ,   4e-2 , 8e-2 ]
    with  pytest.raises(  AssertionError, match  =   'hiding a solver problem' )   :
        P.first_resolved_point(idx, u)

def test_implant_shape_matches_ddsim() ->None :
    r ,  g  =  P.implant_shape( P.MOSFET_PROCESS )

    h=SHORT_CHANNEL_PROCESS
    Na = - float(h['substrate_doping'])
    ys =  float(h["sd_peak"])
    y  =  float(h [  "x_j"] )  /   math.sqrt (2.0 *  math.log(  ys  /  Na))
    assert r ==pytest.approx(y,rel=1e-12)
    assert g > 0.0

    assert ys*0.5  *math.erfc(float(h["lateral_diffusion"]) / g) ==pytest.approx(Na, rel =  1e-9)



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
    vv = P.read_mosfet_golden(str(golden_path(benchmark)))
    assert  vv.header["device" ]   == benchmark.name
    assert vv.header["generator"].startswith('devsim ');  assert "generated on" in vv.header
    assert 'mesh convergence' in vv.header
    assert vv.header["statistics"]== "Boltzmann"
    assert vv.header[ "mobility"  ].startswith(  "constant"  )
    assert vv.header["recombination"].startswith('SRH only')



@pytest.mark.parametrize('benchmark',P.MOSFET_BENCHMARKS,ids= lambda b : b.name)
def test_golden_header_matches_the_benchmark(
    benchmark:P.MosfetBenchmark,
)->None:

    k=P.read_mosfet_golden(str(golden_path(benchmark)))
    for  m, d  in (( "L_gate", benchmark.L_gate), (  "drain low" , benchmark.drain_low) , ( 'drain high',  benchmark.drain_high), ( "tolerance",   benchmark.tolerance ) ,)   :
        assert float(k.header[m])  == pytest.approx(d, rel = 1e-9), (
            f"{m} in {benchmark.name}.csv is {k.header[m]} but "
            f"parameters.py now says {d}. The golden data is stale."
        )
    for m,d in P.MOSFET_PROCESS.items():

        assert float (  k.header [m])  ==  pytest.approx( d , rel  =   1e-9 ),  (
            f"{m} in {benchmark.name}.csv is {k.header[m]} but the "
            f"process now says {d}. The golden data is stale."
        )
    assert k.gate_voltage ==pytest.approx(list(benchmark.gate_voltages)), (
        'the golden gate biases are not the ones parameters.py asks for'
    )



@pytest.mark.parametrize("benchmark",P.MOSFET_BENCHMARKS,ids=lambda b :b.name)


def test_golden_reference_is_converged(benchmark: P.MosfetBenchmark) ->  None:
    ret = P.read_mosfet_golden(str(golden_path(benchmark)))
    assert "mesh convergence" in ret.header,   (
        f"{benchmark.name} golden data carries no mesh convergence line, so "
        "the generator wrote its curves and then did not finish the halved "
        "mesh check. The curves may be fine and nothing here can tell. "
        'Re-run the generator.'
    )

    t= ret.header["mesh convergence"]; arr=t.split()[0]
    assert float(arr)< 0.1 * benchmark.tolerance,(
        f"{benchmark.name} golden data is converged only to {arr}, which "
        f"is not comfortably inside the {benchmark.tolerance} it is used to "
        "assert. Refine the generator mesh and regenerate."
    )
    ii ,   num   =   ( int ( el )   for el  in  t.split (  )   if el.isdigit(  )  )
    assert ii<0.5*num,(
        f"{benchmark.name} skipped {ii} of {num} points as "
        "unmeasurable, so the convergence number covers less than half the "
        f"curve and {arr} says little about the mesh."
    )



@pytest.mark.parametrize("benchmark",P.MOSFET_BENCHMARKS,ids =lambda b : b.name)


@pytest.mark.parametrize('high', [False, True], ids =["Vd_low", 'Vd_high'])


def test_golden_terminals_balance(
    benchmark:P.MosfetBenchmark,high:bool
)->None :
    v = P.read_mosfet_golden(str(golden_path(benchmark)))
    e  =  max(
        v.imbalance(z2, high)  for z2 in range(len(v.gate_voltage))
    )
    assert e<0.01,(
        f"{benchmark.name} drain and source disagree by {e:.2%} at worst, "
        "which is too much of the reference's own budget to be noise"
    )

_SOLVED:dict[tuple[str,float],IVCurve]={}



def ddsim_curve(benchmark:P.MosfetBenchmark,drain:float)->IVCurve:
    mm= (benchmark.name,drain)
    if mm not in _SOLVED  :
        nxt = nmos(L_gate= benchmark.L_gate, drain_voltage = drain, degenerate  = False, **SHORT_CHANNEL_PROCESS,)
        b  = TransportModels.for_device(nxt, mobility= 'constant')
        _SOLVED[mm  ]  =   gate_sweep(nxt ,  list ( benchmark.gate_voltages ) ,  models  =  b)
    return _SOLVED[mm]
@pytest.mark.parametrize("benchmark", P.MOSFET_BENCHMARKS, ids = lambda b : b.name)

@pytest.mark.parametrize("high",[False,True],ids= ["Vd_low",'Vd_high'])




def test_ddsim_reaches_every_golden_bias(
    benchmark :  P.MosfetBenchmark, high  : bool
) ->  None :

    res = benchmark.drain_high if high else benchmark.drain_low
    r   =  ddsim_curve( benchmark ,  res  )

    assert r.complete,f"ddsim stopped early: {r.message}"
    assert  list(  r.voltage) ==  pytest.approx (list (benchmark.gate_voltages ))



@pytest.mark.parametrize('benchmark', P.MOSFET_BENCHMARKS, ids = lambda b  :  b.name)
@pytest.mark.parametrize('high', [False, True], ids= ["Vd_low", "Vd_high"])


def test_ddsim_matches_devsim_drain_current(
    benchmark : P.MosfetBenchmark, high :bool
)  -> None :
    xs  = P.read_mosfet_golden(str(golden_path(benchmark)))
    r2 = benchmark.drain_high if high else benchmark.drain_low;  h=ddsim_curve(benchmark,r2)
    k= xs.drain_high if high else xs.drain_low
    bb :  list[str]  = []
    d=0.0
    for y, (s2, t, m2) in enumerate(
        zip(xs.gate_voltage, k, list(h.current), strict  =True)
    ):
        if abs(t) < P.CURRENT_FLOOR_MOSFET :
            if abs(m2)>=P.CURRENT_FLOOR_MOSFET:

                bb.append(
                    f"{s2:+.3f} V: devsim gives {t:.3e}, below the "
                    f"{P.CURRENT_FLOOR_MOSFET:.3e} floor, but ddsim gives "
                    f"{m2:.3e}"
                )
            continue
        w  =max(benchmark.tolerance, NOISE_FACTOR *xs.imbalance(y, high))
        c2   =  abs(m2 -   t )  /  abs( t  )
        d =max(d, c2/ w)
        if c2 > w :
            bb.append(
                f"{s2:+.3f} V: devsim {t:.6e}, ddsim {m2:.6e}, "
                f"off by {c2:.2%}, allowed {w:.2%}"
            )


    assert not  bb,   (
        f"{benchmark.name} at Vd = {r2} V disagrees with DEVSIM at "
        f"{len(bb)} of {len(xs.gate_voltage)} points:\n  "
        +  "\n  ".join(bb )
    )

    assert d<=1.0
def _threshold(voltage:npt.NDArray[np.float64], current:npt.NDArray[np.float64], L_gate:float,)-> float:
    hh  =np.asarray(voltage, dtype = np.float64)
    h=np.asarray(current, dtype = np.float64)
    idx  = REFERENCE_CURRENT  /L_gate

    prev= P.first_resolved_point(h, idx)
    return threshold_constant_current(hh[prev :],h[prev:],idx)


def  _devsim_thresholds(name  : str )  ->   tuple [float, float  ]   :
    vals =  P.MOSFET_BY_NAME[name]
    k  =  P.read_mosfet_golden (  str(golden_path(  vals)  ) )
    h  =  np.asarray(k.gate_voltage, dtype=  np.float64)
    return(
        _threshold(h, np.asarray(k.drain_low, dtype =  np.float64),
                   vals.L_gate),
        _threshold(h, np.asarray(k.drain_high, dtype=np.float64),
                   vals.L_gate),
    )
def _ddsim_thresholds(name : str)->tuple[float,float] :

    d =P.MOSFET_BY_NAME[name]
    v  = [ ]
    for z in(d.drain_low, d.drain_high)  :
        r   =   ddsim_curve( d, z )
        assert r.complete ,   f"{name} at Vd = {z} V: {r.message}"
        v.append(_threshold(r.voltage,r.current,d.L_gate))
    return v[0],v[1]

@pytest.mark.parametrize( "name",   P.ROLLOFF_TREND )
def  test_rolloff_golden_data_exists(  name  :  str  )   ->  None   :
    x = P.MOSFET_BY_NAME[name]

    assert golden_path(x).is_file(), (
        f"no golden curves for {name}, so benchmark 9 is not running. "
        "Generate them with tests/regression/devsim_gen/generate_mosfet.py, "
        "see the README there."
    )


@pytest.mark.parametrize('benchmark', P.ROLLOFF_BENCHMARKS, ids = lambda b : b.name)
def test_rolloff_reference_is_converged(benchmark :P.MosfetBenchmark)->None:
    g   =  P.read_mosfet_golden(  str(  golden_path(  benchmark )) )
    assert 'mesh convergence' in g.header, (
        f"{benchmark.name} golden data carries no mesh convergence line, so "
        'the generator wrote its curves and then did not finish the halved '
        "mesh check. Re-run the generator."
    )
    x2=  g.header["mesh convergence"]


    s =float(x2.split() [0])
    assert s< benchmark.tolerance, (
        f"{benchmark.name} golden data is converged only to {s:.3e}, "
        f"which is not inside the {benchmark.tolerance} the roll-off and DIBL "
        'are compared to even before the attenuation is counted. Refine the '
        "generator mesh and regenerate."
    )
    val,   t = ( int (b2)  for  b2  in  x2.split ( )   if b2.isdigit(  ))
    assert val<0.5 *t,(
        f"{benchmark.name} skipped {val} of {t} points as "
        "unmeasurable, so the convergence number covers less than half the "
        f"curve and {s:.3e} says little about the mesh."
    )

@pytest.mark.parametrize("high",[False,True],ids =['Vd_low','Vd_high'])


def test_rolloff_threshold_falls_in_both_codes(high:bool)->  None :
    s  =1 if high else 0
    z2=[_devsim_thresholds(i)[s] for i in P.ROLLOFF_TREND]
    v2= [_ddsim_thresholds(i) [s]for i in P.ROLLOFF_TREND]
    for k, hh in(('devsim', z2), ("ddsim", v2)) :
        w =  np.diff( np.asarray ( hh)  )
        assert np.all(w <0.0), (
            f"{k} threshold does not fall monotonically from 1 um to "
            f"50 nm: {[f'{c:+.4f}' for c in hh]}"
        )


@pytest.mark.parametrize(  "high" ,   [False ,   True], ids   = [ "Vd_low" , "Vd_high" ]  )

def test_rolloff_magnitude_matches_devsim(high : bool)  ->None :

    t2= 1 if high else 0
    w=[_devsim_thresholds(t)  [t2] for t in P.ROLLOFF_TREND]
    stuff  =  [ _ddsim_thresholds(  t)  [ t2]  for t  in P.ROLLOFF_TREND  ]

    e  =   w[ 0  ]   - w [  -  1  ]
    g =stuff[0]- stuff[- 1]
    x =abs(g -e) / abs(e)

    c="\n  ".join(
        f"{z2}: devsim {b:+.4f} V, ddsim {num:+.4f} V, "
        f"{(num - b) * 1000:+.1f} mV"
        for z2,b,num in zip(P.ROLLOFF_TREND,w,stuff,strict=True)
    )
    assert x<= 0.10,(
        f"roll-off from 1 um to 50 nm disagrees by {x:.1%}: devsim "
        f"{e * 1000:.1f} mV, ddsim {g * 1000:.1f} mV."
        f"\n  {c}"
    )


@pytest.mark.parametrize("name", P.ROLLOFF_TREND)



def test_rolloff_dibl_matches_devsim(name  :  str)  ->None:


    ok =P.MOSFET_BY_NAME[name]
    num  =  ok.drain_high -ok.drain_low
    t2, out2 = _devsim_thresholds(name)
    dd, v= _ddsim_thresholds(name)
    z=(t2 -out2)/ num
    tmp3=  (dd -v)  /num
    assert z> 0.0, (
        f"{name}: devsim puts the saturated threshold above the linear one, "
        f"{out2:+.4f} V against {t2:+.4f}, which is not DIBL"
    )
    y  = abs(tmp3  -z)  / z
    assert y<=0.10, (
        f"{name} DIBL disagrees by {y:.1%}: devsim "
        f"{z * 1000:.1f} mV/V, ddsim {tmp3 * 1000:.1f} mV/V"
    )



_SOLVED_FULL : dict[tuple[str, float], IVCurve]= {}


def ddsim_full_curve(benchmark:P.MosfetBenchmark,drain:float)->IVCurve:

    k= (benchmark.name,drain)
    if k not in _SOLVED_FULL:
        m = nmos(L_gate = benchmark.L_gate, drain_voltage=drain, **SHORT_CHANNEL_PROCESS,)
        f = TransportModels.for_device(m, mobility =  "arora", field_dependent=True, surface  =True)
        _SOLVED_FULL[k]= gate_sweep(
            m, list(benchmark.gate_voltages), models= f
        )
    return _SOLVED_FULL[k]



def _ddsim_full_thresholds(name:str)-> tuple[float,float]:


    nxt =   P.MOSFET_BY_NAME[ name ]
    a=  []
    for bb in(nxt.drain_low,nxt.drain_high):
        s = ddsim_full_curve(nxt, bb)
        assert s.complete, f"{name} at Vd = {bb} V: {s.message}"
        a.append(_threshold(s.voltage,
                     s.current,
               nxt.L_gate))

    return a[0], a[1]


@pytest.mark.parametrize ( "name" ,  P.FULL_STACK_TREND)

def test_full_stack_golden_data_exists(name :str) ->None:
    ret=P.MOSFET_BY_NAME[name]
    assert golden_path(ret).is_file(),(
        f"no golden curves for {name}, so benchmark 10 is not running and "
        "nothing in tier 4 compares either mobility model or the statistics "
        'against an outside code. Generate them with '
        "tests/regression/devsim_gen/generate_mosfet.py, see the README there."
    )




@pytest.mark.parametrize("name",  P.FULL_STACK_TREND  )


def test_full_stack_header_records_the_models(name :  str) ->  None:
    e  =  P.read_mosfet_golden(  str(  golden_path(  P.MOSFET_BY_NAME [name  ])  ) )
    assert e.header["statistics"].startswith('Fermi-Dirac') ; assert e.header['mobility'].startswith('Arora')
    assert 'Lombardi' in e.header['mobility']
    assert 'Caughey-Thomas' in e.header["mobility"]

@pytest.mark.parametrize(
    "benchmark", P.FULL_STACK_BENCHMARKS, ids  =  lambda  b  :   b.name
)

def test_full_stack_reference_is_converged(
    benchmark : P.MosfetBenchmark,
)->None :
    rr  =  P.read_mosfet_golden(  str(  golden_path( benchmark  )  ))
    assert 'mesh convergence' in  rr.header,   (
        f"{benchmark.name} golden data carries no mesh convergence line, so "
        "the generator wrote its curves and then did not finish the halved "
        "mesh check. Re-run the generator."
    )
    x =rr.header["mesh convergence"]

    d=  float(x.split() [0])
    assert  d <  benchmark.tolerance,   (
        f"{benchmark.name} golden data is converged only to {d:.3e}, "
        f"which is not inside the {benchmark.tolerance} the roll-off and DIBL "
        'are compared to even before the attenuation is counted. Refine the '
        "generator mesh and regenerate."
    )
    t, i =(int(v) for v in x.split() if v.isdigit())
    assert t <0.5 *i,(
        f"{benchmark.name} skipped {t} of {i} points as "
        "unmeasurable, so the convergence number covers less than half the "
        f"curve and {d:.3e} says little about the mesh."
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
    w2= P.read_mosfet_golden(str(golden_path(benchmark)))
    c =  w2.drain_high if high else w2.drain_low
    e=LOAD_BEARING*REFERENCE_CURRENT/benchmark.L_gate


    m,d= 0.0,0.0

    for ys, v in enumerate(c):
        if abs(v)  <e :

            continue


        el =w2.imbalance(ys,
             high)
        if el > m  :
            m , d  =   el,  w2.gate_voltage[ys ]

    assert m<0.01,(
        f"{benchmark.name} drain and source disagree by {m:.2%} at "
        f"{d:+.2f} V of gate, where the current is within four decades "
        'of the one the threshold is read at. That is too much of the '
        "reference's own budget to be noise"
    )


@pytest.mark.parametrize("benchmark",P.FULL_STACK_BENCHMARKS,ids=lambda b:b.name)



@pytest.mark.parametrize('high',[False,True],ids =["Vd_low","Vd_high"])

def  test_full_stack_ddsim_reaches_every_golden_bias(
    benchmark   :  P.MosfetBenchmark , high  : bool
) ->  None  :
    k2= benchmark.drain_high if high else benchmark.drain_low
    yy =ddsim_full_curve(benchmark, k2)

    assert yy.complete,f"ddsim stopped early: {yy.message}";  assert list(yy.voltage)==pytest.approx(list(benchmark.gate_voltages))




@pytest.mark.parametrize(
    'benchmark', P.FULL_STACK_BENCHMARKS, ids =lambda b : b.name
)


@pytest.mark.parametrize("high",[False,True],ids=['Vd_low',"Vd_high"])

def test_full_stack_drain_current_matches_devsim(benchmark:P.MosfetBenchmark,high :bool) ->None :
    ii  = P.read_mosfet_golden(str(golden_path(benchmark)))
    w  =  benchmark.drain_high if high  else benchmark.drain_low;r   =  ddsim_full_curve( benchmark,  w )
    flag= ii.drain_high if high else ii.drain_low
    d = FULL_STACK_FLOOR *REFERENCE_CURRENT  / benchmark.L_gate



    c:list[str] =[]
    a2  =0.0
    for x,(w2,buf,g) in enumerate(zip(ii.gate_voltage,flag,list(r.current),strict=True)):
        if abs(buf) <d  :
            if abs(g)>= d :
                c.append(
                    f"{w2:+.3f} V: devsim gives {buf:.3e}, below the "
                    f"{d:.3e} floor, but ddsim gives {g:.3e}"
                )
            continue
        t = max(
            benchmark.tolerance, NOISE_FACTOR* ii.imbalance(x, high)
        )
        m=abs(g- buf) / abs(buf)
        a2 =  max ( a2,   m )
        if m> t:
            c.append (
                f"{w2:+.3f} V: devsim {buf:.6e}, ddsim {g:.6e}, "
                f"{m:.2%} against {t:.2%} allowed"
            )

    assert  not  c,  (
        f"{benchmark.name} at Vd = {w} V, worst {a2:.2%}:\n  "
        +  "\n  ".join(  c)
    )


@pytest.mark.parametrize( 'high' , [  False,   True  ] ,  ids   =  [  "Vd_low",   "Vd_high"]  )



def test_full_stack_threshold_falls_in_both_codes(  high  :  bool)  -> None  :


    k2=1 if high else 0
    g= [_devsim_thresholds(t)  [k2] for t in P.FULL_STACK_TREND]

    k  =  [_ddsim_full_thresholds(t) [k2]  for t in P.FULL_STACK_TREND]

    for ys,a in(('devsim',g),("ddsim",k)) :
        c=np.diff(np.asarray(a))

        assert np.all(c  <   0.0 ),  (
            f"{ys} threshold does not fall monotonically from 1 um to "
            f"50 nm: {[f'{tmp:+.4f}' for tmp in a]}"
        )




@pytest.mark.parametrize("high", [False, True], ids = ["Vd_low", "Vd_high"])



def  test_full_stack_rolloff_matches_devsim ( high  :  bool)   -> None :

    z2  =   1  if  high  else  0
    ii= [_devsim_thresholds(j)[z2]for j in P.FULL_STACK_TREND]
    out  =   [_ddsim_full_thresholds ( j  )  [ z2  ]   for j in P.FULL_STACK_TREND]
    jj =  ii [ 0 ]  - ii[ -  1  ]
    idx=out[0] -out[-1]
    row=abs(idx - jj)/abs(jj)
    t ="\n  ".join(
        f"{w}: devsim {cur:+.4f} V, ddsim {val2:+.4f} V, "
        f"{(val2 - cur) * 1000:+.1f} mV"
        for w, cur, val2 in zip(P.FULL_STACK_TREND, ii, out, strict=  True)
    )
    assert row <= 0.10,(
        f"roll-off from 1 um to 50 nm disagrees by {row:.1%}: devsim "
        f"{jj * 1000:.1f} mV, ddsim {idx * 1000:.1f} mV."
        f"\n  {t}"
    )


@pytest.mark.parametrize('name',P.FULL_STACK_TREND)


def  test_full_stack_dibl_matches_devsim(name   :   str ) ->  None :
    out2  =   P.MOSFET_BY_NAME [  name  ]
    j= out2.drain_high -out2.drain_low

    t,val2=_devsim_thresholds(name)
    x,m= _ddsim_full_thresholds(name)


    a=(t- val2) /j
    a2 =  (  x - m  ) / j

    assert a>0.0,(
        f"{name}: devsim puts the saturated threshold above the linear one, "
        f"{val2:+.4f} V against {t:+.4f}, which is not DIBL"
    )
    jj =  abs ( a2  - a )  /  a
    assert jj  <=0.10, (
        f"{name}: DIBL disagrees by {jj:.1%}, devsim "
        f"{a * 1000:.1f} mV/V against ddsim "
        f"{a2 * 1000:.1f} mV/V"
    )
