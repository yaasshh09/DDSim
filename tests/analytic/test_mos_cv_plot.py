from __future__ import annotations
import pathlib
import matplotlib
import numpy as np, pytest
matplotlib.use("Agg"); import  matplotlib.pyplot as plt
from ddsim.core import constants as C
from ddsim.device.mos_cap import GATE,mos_cap
from ddsim.extract.cv import Response, cv_sweep
from tests.analytic.test_mos_cap import(flatband_voltage, max_depletion_width , oxide_capacitance, threshold_voltage ,)
from tests.analytic.test_mos_cv import debye_length,in_series
from tests.regression.devsim_gen import  parameters as  P

OUTPUT  =  pathlib.Path (  __file__ ).parents [  2] / 'images'

GOLDEN_DIR=pathlib.Path(__file__).resolve().parents[2]/ 'data' /"golden"
BENCHMARK =P.MOS_BENCHMARKS[0]

NA = - BENCHMARK.substrate_doping

T_OX=BENCHMARK.t_ox


T_SI =  BENCHMARK.t_si

METAL = BENCHMARK.work_function

V_FB = flatband_voltage(-  NA, METAL)

V_TH   =   threshold_voltage(  -  NA,   T_OX,   METAL  )
C_OX = oxide_capacitance(T_OX)

C_FB  =  in_series(C_OX, C.eps_Si() /  debye_length(- NA))


C_MIN = in_series(C_OX, C.eps_Si()/ max_depletion_width(- NA))
NANO =1e9

LOWEST = -3.5

STEP  = 0.1

def bias_points()-> list[float]:
    m2= max(BENCHMARK.voltages)
    m=int(round((m2- LOWEST) /STEP)) + 1
    return [  round ( LOWEST  +   STEP   *  cur,  4 )   for cur in  range(  m  )  ]


@pytest.fixture(scope='module')

def device() :
    return  mos_cap(substrate_doping  = BENCHMARK.substrate_doping, t_ox =   BENCHMARK.t_ox, t_si =  BENCHMARK.t_si, n_silicon  =  BENCHMARK.n_silicon , n_oxide =  BENCHMARK.n_oxide, h_min   =  BENCHMARK.h_min , work_function =  BENCHMARK.work_function ,)




@pytest.fixture ( scope =   "module")



def curves( device )  :
    j  =bias_points()

    return{cur :cv_sweep(device,GATE,j,response=cur) for cur in Response}

@pytest.fixture(scope="module")

def landmarks(device) :
    return{
        bb:  cv_sweep(device, GATE, [V_FB, V_TH], response = bb)
        for bb in Response
    }
@pytest.fixture(scope= "module")



def golden():
    s= P.read_mos_golden(str(GOLDEN_DIR/ f"{BENCHMARK.name}.csv"))
    b, aa =  P.central_difference(s.gate_voltage, s.charge)
    return np.asarray (b ),   np.asarray ( aa )




def ddsim_differenced(curves) :

    item=curves[Response.LOW_FREQUENCY]
    y  =  np.isin(np.round(item.gate_voltage, 4), BENCHMARK.voltages)

    e,  w2  =   P.central_difference(
        list (item.gate_voltage[ y ]  ), list(item.charge [  y  ] )
    )
    return np.asarray(e), np.asarray(w2)

def test_both_sweeps_finish(curves):
    for vals, w in curves.items():
        assert w.complete,   f"{vals.value}: {w.message}"

def test_the_sweep_covers_every_golden_bias(curves) :
    g =   np.round(  curves [  Response.LOW_FREQUENCY  ].gate_voltage,  4  )

    ok=sorted(set(BENCHMARK.voltages)- set(g.tolist()))
    assert not ok,f"ddsim never reached {ok}"


def test_the_annotated_numbers_are_the_ones_the_plot_will_show(curves, landmarks) :
    m  = curves[  Response.LOW_FREQUENCY ]
    r =curves[Response.HIGH_FREQUENCY]
    tt= float(m.capacitance[0])
    assert tt == pytest.approx(  C_OX, rel   =   0.02)


    m2 = float(landmarks[Response.LOW_FREQUENCY].capacitance[0])
    assert m2== pytest.approx(C_FB,
                      rel= 1e-3)
    z  = float(landmarks[  Response.HIGH_FREQUENCY].capacitance [ 1  ])
    assert z == pytest.approx(C_MIN, rel= 0.05)


    j   = float (m.capacitance[  -  1 ] )
    assert j == pytest.approx(C_OX, rel = 0.05)
    assert float(r.capacitance[-1]) < 0.2*C_OX



def test_the_overlaid_curves_agree(curves,golden) :
    g, t = golden
    r2 ,   j =   ddsim_differenced(  curves)

    np.testing.assert_allclose(r2, g, atol=1e-9)


    s2  =  float (
        np.max(
            np.abs(j -  t )
            /   np.abs(  t)
        )
    )

    assert s2<BENCHMARK.tolerance, (
        f"worst disagreement {s2:.3%}, allowed {BENCHMARK.tolerance:.3%}"
    )



def test_mos_cv_plot_is_generated(curves, golden)  :
    h  = curves[Response.LOW_FREQUENCY]
    bar =curves[Response.HIGH_FREQUENCY]
    a,s= golden
    _, c = ddsim_differenced(curves)

    item=float(
        np.max(
            np.abs(c-s)
            /np.abs(s)
        )
    )

    d, t2=plt.subplots(figsize =(8.6, 5.6))

    t2.plot(a, s * NANO, 'o', markersize  =5.5, markerfacecolor ='none', markeredgewidth =1.1, color ="tab:orange", label = 'DEVSIM 2.11, central difference on the golden charge', zorder =3,)
    t2.plot(
        h.gate_voltage,
        h.capacitance  *NANO,
        label  = "ddsim, low frequency, every carrier follows",
        linewidth =1.8,
        color = 'tab:blue',
        zorder = 4,
    )


    t2.plot(
        bar.gate_voltage,
        bar.capacitance * NANO,
        '--',
        label =  'ddsim, high frequency, minority carrier held',
        linewidth = 1.8,
        color =  "tab:blue",
        alpha =  0.65,
        zorder =  4,
    )
    m= C_OX *  NANO
    out2 = float(h.gate_voltage[0])
    for z2,b2,obj in(
        (C_OX,r"$C_{ox} = \varepsilon_{ox}/t_{ox}$",0.055),
        (C_FB,r"$C_{FB} = C_{ox} \parallel \varepsilon_{Si}/L_D$",0.055),
        (C_MIN,r"$C_{min} = C_{ox} \parallel \varepsilon_{Si}/W_{max}$",-0.022),
    ) :

        t2.axhline(z2 * NANO,color= 'grey',linestyle=':',linewidth =0.9)
        t2.annotate(
            f"{b2} = {z2 * NANO:.1f}",
            xy  = (out2, z2 *  NANO),
            xytext=(out2+  0.08, z2* NANO- obj *  m),
            fontsize  =  8.5,
            color = 'dimgrey',
        )

    for z, b2 in(
        (V_FB, f"$V_{{FB}}$ = {V_FB:.3f} V"),
        (V_TH, f"$V_{{TH}}$ = {V_TH:.3f} V"),
    ) :
        t2.axvline(z, color= 'grey', linestyle="-.", linewidth= 0.8)
        t2.annotate (
            b2 ,
            xy   = ( z , 0.62  * m  ),
            xytext  =  (z  -  0.05,  0.62 *   m  ),
            fontsize   =  8.5 ,
            rotation  = 90 ,
            va  =  "bottom",
            ha =  'right',
        )


    for w,i in(
        (0.5* (out2+ V_FB),"accumulation"),
        (0.5*(V_FB+V_TH),"depletion"),
        (0.5 * (V_TH +float(h.gate_voltage[-1])),'inversion'),
    ):
        t2.annotate(
            i,
            xy= (w, m  * 1.10),
            ha =  "center",
            fontsize  = 11,
            color  = 'tab:blue',
        )

    t2.annotate(
        f"worst disagreement with DEVSIM: {item:.3%}\n"
        "(same central difference applied to both)",
        xy   =  ( 0.985,   0.28),
        xycoords  =  'axes fraction',
        ha  =  'right',
        fontsize   = 8.5 ,
        color =   'dimgrey' ,
    )

    t2.set_ylim(0.0,m * 1.20)
    t2.set_xlim(  h.gate_voltage[0],  h.gate_voltage[ -  1]  )
    t2.set_xlabel("gate bias [V]")

    t2.set_ylabel(  "capacitance [nF/cm$^2$]")
    t2.set_title(
        f"MOS capacitor C-V, {NA:.0e} cm$^{{-3}}$ p-type, "
        f"{T_OX * 1e7:.0f} nm oxide, n+ poly gate"
    )
    t2.legend(loc ='upper center', bbox_to_anchor=(0.5,- 0.13), ncol=1, fontsize =9, frameon=False,)


    OUTPUT.mkdir (  parents  =  True, exist_ok =  True  )
    t=OUTPUT/ "mos_cap_cv.png"
    d.savefig(  t ,  dpi =  140,   bbox_inches  = 'tight' )

    plt.close( d )



    assert t.exists()
    assert t.stat().st_size>10_000
